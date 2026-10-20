"""Fetch official Taiwan bars for a forward package with the ingestion provider code.

The sequence is the CPU preparation's Taiwan loop: corporate actions for the interval,
benchmark months (which also define the official sessions), then one adjusted quote
request per session. Responses are cached per end date, so a re-run resumes without
network calls and a later end date never reuses a partial month.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_forecasting.data.ingestion import (
    _canonical_raw_batch,
    _fetch_taiwan_actions,
    _fetch_taiwan_adjusted_date,
    _fetch_taiwan_benchmark_month,
    _ordered_thread_map,
    _taiwan_materialization_dates,
    _taiwan_materialization_months,
)
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.data.providers.http import CachedJsonClient, NetworkRequestBudget
from stock_forecasting.data.providers.taiwan import TPExProvider, TWSEProvider
from stock_forecasting.data.sample_universe import market_sessions
from stock_forecasting.data_policy import load_data_policy

FETCH_SCHEMA_VERSION = "1.0"
FETCH_KIND = "forward-taiwan-fetch"
DATASET_PROFILE = "tw_only"
DEFAULT_START = "2025-03-01"
# Same pacing as the CPU preparation defaults (per provider).
DEFAULT_REQUESTS_PER_SECOND = 0.5
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
PROVIDERS = {"twse_official": TWSEProvider, "tpex_official": TPExProvider}
RETRYABLE_STATUS_CODES = frozenset({403, 408, 425, 429})
BENCHMARK_SYMBOLS = {"twse_official": "TAIEX.TW", "tpex_official": "TPEX.TWO"}


def _headers(provider: str) -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Referer": (
            "https://www.twse.com.tw/"
            if provider == "twse_official"
            else "https://www.tpex.org.tw/"
        ),
    }


def _validated_interval(start: str, end_exclusive: str) -> None:
    for name, value in (("start", start), ("end_exclusive", end_exclusive)):
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a YYYY-MM-DD date") from None
    if start >= end_exclusive:
        raise ValueError("start must precede end_exclusive")
    if end_exclusive > datetime.now(UTC).date().isoformat():
        raise ValueError("end_exclusive cannot be in the future; sessions must be published")


def _fetch_provider(
    name: str,
    *,
    start: str,
    end_exclusive: str,
    cache_root: Path,
    requests_per_second: float,
    max_backoff_seconds: float,
    workers: int,
    budget: NetworkRequestBudget,
    progress: Callable[[str], None],
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    client = CachedJsonClient(
        provider=name,
        raw_cache_root=cache_root,
        max_requests_per_second=requests_per_second,
        max_backoff_seconds=max_backoff_seconds,
        headers=_headers(name),
        retryable_status_codes=RETRYABLE_STATUS_CODES,
        request_budget=budget,
    )
    provider = PROVIDERS[name](client)
    actions = _fetch_taiwan_actions(provider, start=start, exclusive_end=end_exclusive)
    frames: list[pd.DataFrame] = []
    sessions: set[str] = set()
    months = _taiwan_materialization_months(start, end_exclusive)
    with closing(
        _ordered_thread_map(
            lambda month: _fetch_taiwan_benchmark_month(
                provider, month=month, dataset_profile=DATASET_PROFILE
            ),
            months,
            max_workers=min(workers, len(months)),
            thread_name_prefix=f"{name}-month",
        )
    ) as fetched_months:
        for fetched in fetched_months:
            frames.append(fetched.frame)
            sessions.update(
                _taiwan_materialization_dates(
                    fetched.frame, start=start, exclusive_end=end_exclusive
                )
            )
    if not sessions:
        raise ValueError(f"{name} benchmark history has no sessions in the interval")
    ordered = sorted(sessions)
    dropped = 0
    with closing(
        _ordered_thread_map(
            lambda day: _fetch_taiwan_adjusted_date(
                provider,
                trading_date=day,
                dataset_profile=DATASET_PROFILE,
                action_frame=actions.frame,
            ),
            ordered,
            max_workers=min(workers, len(ordered)),
            thread_name_prefix=f"{name}-date",
        )
    ) as fetched_dates:
        for count, (fetched, adjusted) in enumerate(fetched_dates, start=1):
            frames.append(adjusted)
            dropped += int(fetched.metadata.get("dropped_rows", 0))
            if count % 50 == 0 or count == len(ordered):
                progress(f"{name}: {count}/{len(ordered)} sessions")
    return frames, {
        "sessions": ordered,
        "benchmark_months": months,
        "corporate_action_rows": len(actions.frame),
        "dropped_source_rows": dropped,
    }


def calendar_differences(
    official_sessions: list[str], *, start: str, end_exclusive: str
) -> dict[str, list[str]]:
    """Official TWSE sessions against the pinned cleaning calendar over the interval."""
    last = (pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)).date().isoformat()
    calendar = {
        value.date().isoformat()
        for value in market_sessions(load_data_policy()["calendars"]["TWSE"], start, last)
    }
    official = set(official_sessions)
    return {
        "calendar_sessions_without_trading": sorted(calendar - official),
        "trading_days_missing_from_calendar": sorted(official - calendar),
    }


def fetch_taiwan(
    *,
    output_root: str | Path,
    end_exclusive: str,
    start: str = DEFAULT_START,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    workers: int = 2,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Write ``raw/market.parquet`` and ``fetch.json`` under ``output_root``."""
    _validated_interval(start, end_exclusive)
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(output_root)
    raw_path = root / "raw" / "market.parquet"
    if raw_path.exists():
        raise FileExistsError(f"Forward raw data already exists: {raw_path}")
    cache_root = root / "api-cache"
    budget = NetworkRequestBudget(10**9, limited_providers=frozenset())
    # I/O bound and paced per provider: one thread per exchange, bounded per-request workers.
    with ThreadPoolExecutor(max_workers=len(PROVIDERS), thread_name_prefix="forward") as pool:
        futures = {
            name: pool.submit(
                _fetch_provider,
                name,
                start=start,
                end_exclusive=end_exclusive,
                cache_root=cache_root,
                requests_per_second=requests_per_second,
                max_backoff_seconds=max_backoff_seconds,
                workers=workers,
                budget=budget,
                progress=progress,
            )
            for name in PROVIDERS
        }
        results = {name: future.result() for name, future in futures.items()}
    frames = [frame for name in PROVIDERS for frame in results[name][0] if not frame.empty]
    raw = _canonical_raw_batch(pd.concat(frames, ignore_index=True))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    official = results["twse_official"][1]["sessions"]
    manifest = {
        "schema_version": FETCH_SCHEMA_VERSION,
        "kind": FETCH_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_profile": DATASET_PROFILE,
        "start_inclusive": start,
        "end_exclusive": end_exclusive,
        "requests_per_second_per_provider": requests_per_second,
        "providers": {
            name: {
                key: value for key, value in results[name][1].items() if key != "sessions"
            }
            | {
                "sessions": len(results[name][1]["sessions"]),
                "first_session": results[name][1]["sessions"][0],
                "last_session": results[name][1]["sessions"][-1],
            }
            for name in PROVIDERS
        },
        "network_requests": budget.provider_counts,
        "raw": {
            "relative_path": raw_path.relative_to(root).as_posix(),
            "sha256": sha256_file(raw_path),
            "rows": len(raw),
            "symbols": int(raw["symbol"].nunique()),
            "benchmarks_present": sorted(
                set(BENCHMARK_SYMBOLS.values()) & set(raw["symbol"].unique())
            ),
        },
        "calendar": calendar_differences(official, start=start, end_exclusive=end_exclusive),
    }
    atomic_write_json(root / "fetch.json", manifest)
    return manifest


def load_fetch(fetch_root: str | Path) -> dict[str, Any]:
    root = Path(fetch_root)
    manifest = json.loads((root / "fetch.json").read_text())
    if manifest.get("kind") != FETCH_KIND or manifest.get("schema_version") != FETCH_SCHEMA_VERSION:
        raise ValueError(f"Not a forward Taiwan fetch: {root}")
    raw_path = root / manifest["raw"]["relative_path"]
    if sha256_file(raw_path) != manifest["raw"]["sha256"]:
        raise ValueError("Forward raw data changed after the fetch manifest was written")
    return manifest
