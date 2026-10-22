"""Bounded process-pool adjustment of Taiwan sessions must match the inline path."""

from __future__ import annotations

import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import stock_forecasting.data.ingestion as ingestion
from stock_forecasting.data.ingestion import (
    MATERIALIZATION_INLINE_BACKEND,
    MATERIALIZATION_PROCESS_BACKEND,
    MATERIALIZATION_WORKER_MEMORY_BYTES,
    MATERIALIZATION_WORKERS_ENV,
    IngestionOptions,
    _adjust_taiwan_session,
    _adjust_taiwan_session_in_worker,
    _initialize_taiwan_adjustment_worker,
    _materialization_worker_count,
    _ordered_process_map,
    ingest_daily_ohlcv,
)
from stock_forecasting.data.providers.base import ProviderFetch, RequestRecord
from stock_forecasting.data.providers.taiwan import TPExProvider, TWSEProvider
from stock_forecasting.data.schema import normalize_ohlcv_frame

_DATES = pd.bdate_range("2024-01-02", "2024-03-29", tz="UTC")
_START, _END = "2024-01-02", "2024-04-01"


def _require_spawn_pool() -> None:
    """Skip in sandboxes that forbid the semaphores spawn pools need."""

    try:
        with get_context("spawn").Pool(1) as pool:
            assert pool.apply(abs, (-1,)) == 1
    except Exception as error:
        # Any failure (semaphores, sysconf, fork limits) means pools are unavailable here.
        pytest.skip(f"spawn process pools are unavailable: {type(error).__name__}")


def _request(provider: str, operation: str) -> RequestRecord:
    digest = hashlib.sha256(f"{provider}:{operation}".encode()).hexdigest()
    return RequestRecord(
        provider=provider,
        request_sha256=digest,
        cache_relative_path=f"{provider}/{digest[:2]}/{digest}.json",
        response_sha256=hashlib.sha256(b"{}").hexdigest(),
        response_size_bytes=2,
        cache_hit=True,
        requested_at="2026-10-22T00:00:00+00:00",
    )


def _price(symbol: str, day: str) -> float:
    seed = int(hashlib.sha256(f"{symbol}:{day}".encode()).hexdigest()[:8], 16)
    return 20.0 + (seed % 10_000) / 100.0


def _session_frame(
    *,
    symbols: list[str],
    suffix: str,
    provider: str,
    market: str,
    day: str,
    dataset_profile: str,
) -> pd.DataFrame:
    # Mirror the columns _parse_quotes produces: no adjustment columns yet.
    rows = []
    for code in symbols:
        close = _price(code, day)
        rows.append(
            {
                "timestamp": day,
                "symbol": f"{code}.{suffix}",
                "asset_type": "stock",
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": float(1_000 + (int(code) * 7) % 5_000),
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": code,
                "is_active": pd.NA,
                "dataset_profile": dataset_profile,
            }
        )
    return normalize_ohlcv_frame(pd.DataFrame(rows))


def _benchmark_frame(
    *, symbol: str, provider: str, market: str, dates: pd.DatetimeIndex, dataset_profile: str
) -> pd.DataFrame:
    index = np.arange(len(dates), dtype=np.float64)
    close = 10_000.0 * np.exp(index * 0.0004)
    return normalize_ohlcv_frame(
        pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "asset_type": "index",
                "open": close * 0.999,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": np.full(len(dates), 5_000_000.0),
                "adjusted_close": close * 1.01,
                "split_adjusted_volume": np.full(len(dates), 5_000_000.0),
                "adjustment_source": "official_total_return_index",
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": symbol.split(".", maxsplit=1)[0],
                "is_active": True,
                "dataset_profile": dataset_profile,
            }
        )
    )


class _FixtureTaiwanMixin:
    calls: ClassVar[list[str]]
    symbols: ClassVar[list[str]]
    benchmark_symbol: str

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        del start, end
        self.calls.append("actions")
        events = []
        for offset, code in enumerate(self.symbols):
            # Every symbol has a mid-range ex-date so the adjustment changes rows.
            events.append(
                {
                    "timestamp": _DATES[20 + offset],
                    "symbol": f"{code}.{self.suffix}",
                    "price_factor": 0.95 - offset * 0.01,
                    "share_multiplier": 1.0 + offset * 0.1,
                    "source": "fixture",
                }
            )
        return ProviderFetch(
            frame=pd.DataFrame(events),
            requests=(_request(self.name, "actions"),),
            metadata={"dropped_rows": 0},
        )

    def fetch_benchmark_month(self, *, month: str, dataset_profile: str) -> ProviderFetch:
        self.calls.append(f"benchmark:{month}")
        dates = pd.DatetimeIndex([value for value in _DATES if value.strftime("%Y-%m") == month])
        return ProviderFetch(
            frame=_benchmark_frame(
                symbol=self.benchmark_symbol,
                provider=self.name,
                market=self.market,
                dates=dates,
                dataset_profile=dataset_profile,
            ),
            requests=(_request(self.name, f"benchmark:{month}"),),
            metadata={"dropped_rows": 0},
        )

    def fetch_date(self, *, date: str, dataset_profile: str) -> ProviderFetch:
        self.calls.append(f"daily:{date}")
        return ProviderFetch(
            frame=_session_frame(
                symbols=self.symbols,
                suffix=self.suffix,
                provider=self.name,
                market=self.market,
                day=date,
                dataset_profile=dataset_profile,
            ),
            requests=(_request(self.name, f"daily:{date}"),),
            metadata={"dropped_rows": 1, "date": date},
        )


class _FixtureTWSEProvider(_FixtureTaiwanMixin, TWSEProvider):
    calls: ClassVar[list[str]] = []
    symbols: ClassVar[list[str]] = ["1101", "2330", "2454"]
    benchmark_symbol = "TAIEX.TW"


class _FixtureTPExProvider(_FixtureTaiwanMixin, TPExProvider):
    calls: ClassVar[list[str]] = []
    symbols: ClassVar[list[str]] = ["6488", "8069"]
    benchmark_symbol = "TPEX.TWO"


def _options(tmp_path: Path, attempt: str) -> IngestionOptions:
    root = tmp_path / attempt
    return IngestionOptions(
        profile="tw_only",
        start=_START,
        end=_END,
        output=root / "raw" / "market.parquet",
        manifest_root=root,
        raw_cache_root=root / "api-cache",
        provider_checkpoint_root=root / "provider-checkpoints",
        dataset_request_sha256="a" * 64,
        workers=6,
    )


def _twse_payload(symbols: list[str]) -> dict[str, Any]:
    fields = [
        "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
        "開盤價", "最高價", "最低價", "收盤價",
    ]
    data = []
    for code in symbols:
        close = _price(code, "payload")
        data.append(
            [
                code,
                "fixture",
                f"{int(1_000 + int(code) % 900):,}",
                "10",
                "1,000",
                f"{close * 0.99:,.2f}",
                f"{close * 1.02:,.2f}",
                f"{close * 0.98:,.2f}",
                f"{close:,.2f}",
            ]
        )
    return {"stat": "OK", "tables": [{"fields": fields, "data": data}]}


class _StubClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def get_json(self, endpoint: str, *, params: dict[str, Any]) -> tuple[Any, RequestRecord]:
        del endpoint, params
        return self.payload, _request("twse_official", "daily")


def _action_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-06-20", tz="UTC"),
                "symbol": "2330.TW",
                "price_factor": 0.97,
                "share_multiplier": 1.0,
                "source": "fixture",
            },
            {
                "timestamp": pd.Timestamp("2024-07-01", tz="UTC"),
                "symbol": "1101.TW",
                "price_factor": 0.9,
                "share_multiplier": 1.25,
                "source": "fixture",
            },
        ]
    )


def test_split_adjustment_matches_fetch_adjusted_date() -> None:
    provider = TWSEProvider(_StubClient(_twse_payload(["1101", "2330", "2454"])))
    action_frame = _action_frame()

    fetched, reference = provider.fetch_adjusted_date(
        date="2024-03-01", dataset_profile="tw_only", action_frame=action_frame
    )
    split = _adjust_taiwan_session(
        provider.fetch_date(date="2024-03-01", dataset_profile="tw_only").frame,
        action_frame=action_frame,
        adjustment_source=provider.action_adjustment_source,
    )

    pd.testing.assert_frame_equal(split, reference)
    assert fetched.metadata["date"] == "2024-03-01"
    assert reference.loc[reference["symbol"] == "2330.TW", "adjusted_close"].item() == (
        pytest.approx(reference.loc[reference["symbol"] == "2330.TW", "close"].item() * 0.97)
    )
    assert _adjust_taiwan_session(
        pd.DataFrame(), action_frame=action_frame, adjustment_source="x"
    ).empty


def test_worker_count_is_bounded_by_override_cpu_memory_and_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    monkeypatch.delenv(MATERIALIZATION_WORKERS_ENV, raising=False)
    monkeypatch.setattr(ingestion, "detect_visible_cpu_count", lambda: 16)
    monkeypatch.setattr(
        ingestion, "detect_available_memory", lambda: SimpleNamespace(available_bytes=8 * gib)
    )

    # 8 GiB * 0.4 shared by two pools leaves 1.6 GiB, i.e. four 384 MiB workers each.
    count, plan = _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)
    assert (count, plan["backend"], plan["cpu_limit"]) == (
        4,
        MATERIALIZATION_PROCESS_BACKEND,
        8,
    )
    assert plan["memory_budget_bytes"] // MATERIALIZATION_WORKER_MEMORY_BYTES == 4

    monkeypatch.setattr(
        ingestion, "detect_available_memory", lambda: SimpleNamespace(available_bytes=64 * gib)
    )
    assert _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)[0] == 8
    assert _materialization_worker_count(requested=8, tasks=3, concurrent_pools=2)[0] == 3
    assert _materialization_worker_count(requested=8, tasks=1, concurrent_pools=2) == (
        1,
        {
            "requested": 8,
            "override": None,
            "tasks": 1,
            "concurrent_pools": 2,
            "per_worker_bytes": MATERIALIZATION_WORKER_MEMORY_BYTES,
            "count": 1,
            "backend": MATERIALIZATION_INLINE_BACKEND,
        },
    )
    monkeypatch.setattr(ingestion, "detect_visible_cpu_count", lambda: 3)
    assert _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)[0] == 1

    # The override caps the request, and "1" must not probe the host at all.
    monkeypatch.setattr(ingestion, "detect_visible_cpu_count", lambda: 16)
    monkeypatch.setenv(MATERIALIZATION_WORKERS_ENV, "2")
    assert _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)[0] == 2

    def forbidden() -> Any:
        raise AssertionError("inline planning must not probe host resources")

    monkeypatch.setattr(ingestion, "detect_visible_cpu_count", forbidden)
    monkeypatch.setattr(ingestion, "detect_available_memory", forbidden)
    monkeypatch.setenv(MATERIALIZATION_WORKERS_ENV, "1")
    count, plan = _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)
    assert (count, plan["override"], plan["backend"]) == (1, 1, MATERIALIZATION_INLINE_BACKEND)

    monkeypatch.setenv(MATERIALIZATION_WORKERS_ENV, "0")
    with pytest.raises(ValueError, match=MATERIALIZATION_WORKERS_ENV):
        _materialization_worker_count(requested=8, tasks=2_600, concurrent_pools=2)
    monkeypatch.delenv(MATERIALIZATION_WORKERS_ENV)
    with pytest.raises(ValueError, match="positive"):
        _materialization_worker_count(requested=0, tasks=2_600, concurrent_pools=2)


def test_ordered_process_map_keeps_order_and_contextualizes_worker_errors() -> None:
    _require_spawn_pool()
    action_frame = _action_frame()
    sessions = [
        (
            day,
            _session_frame(
                symbols=["1101", "2330", "2454"],
                suffix="TW",
                provider="twse_official",
                market="TWSE",
                day=day,
                dataset_profile="tw_only",
            ),
        )
        for day in ("2024-06-03", "2024-06-21", "2024-07-02", "2024-07-03", "2024-07-04")
    ]
    sessions.append(("2024-07-05", pd.DataFrame()))

    results = list(
        _ordered_process_map(
            _adjust_taiwan_session_in_worker,
            sessions,
            payload=lambda session: session[1],
            max_workers=2,
            initializer=_initialize_taiwan_adjustment_worker,
            initargs=(action_frame, "twse_twt49u"),
        )
    )

    assert [session[0] for session, _result in results] == [day for day, _ in sessions]
    for (day, frame), adjusted in results:
        expected = _adjust_taiwan_session(
            frame, action_frame=action_frame, adjustment_source="twse_twt49u"
        )
        pd.testing.assert_frame_equal(adjusted, expected)
        if day == "2024-07-05":
            assert adjusted.empty

    def contextualize(session: tuple[str, pd.DataFrame], error: Exception) -> Exception:
        return RuntimeError(f"{session[0]}:{type(error).__name__}")

    broken = [("2024-07-08", pd.DataFrame({"close": [1.0]}))]
    with pytest.raises(RuntimeError, match="2024-07-08:"):
        list(
            _ordered_process_map(
                _adjust_taiwan_session_in_worker,
                broken,
                payload=lambda session: session[1],
                max_workers=2,
                initializer=_initialize_taiwan_adjustment_worker,
                initargs=(action_frame, "twse_twt49u"),
                contextualize=contextualize,
            )
        )
    with pytest.raises(ValueError, match="positive"):
        list(_ordered_process_map(abs, [1], payload=lambda value: value, max_workers=0))


def test_process_pool_acquisition_matches_inline_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_spawn_pool()
    monkeypatch.setattr(ingestion, "TWSEProvider", _FixtureTWSEProvider)
    monkeypatch.setattr(ingestion, "TPExProvider", _FixtureTPExProvider)
    _FixtureTWSEProvider.calls.clear()
    _FixtureTPExProvider.calls.clear()

    monkeypatch.setenv(MATERIALIZATION_WORKERS_ENV, "1")
    inline_options = _options(tmp_path, "inline")
    inline = ingest_daily_ohlcv(inline_options)
    inline_calls = (list(_FixtureTWSEProvider.calls), list(_FixtureTPExProvider.calls))
    _FixtureTWSEProvider.calls.clear()
    _FixtureTPExProvider.calls.clear()

    monkeypatch.setenv(MATERIALIZATION_WORKERS_ENV, "3")
    pooled_options = _options(tmp_path, "pooled")
    pooled = ingest_daily_ohlcv(pooled_options)

    assert (list(_FixtureTWSEProvider.calls), list(_FixtureTPExProvider.calls)) == inline_calls
    inline_frame = pd.read_parquet(inline_options.output)
    pooled_frame = pd.read_parquet(pooled_options.output)
    pd.testing.assert_frame_equal(pooled_frame, inline_frame)
    assert "adjustment_source" in pooled_frame.columns
    assert (pooled_frame["adjusted_close"] != pooled_frame["close"]).any()
    assert (
        inline_options.output.read_bytes() == pooled_options.output.read_bytes()
    ), "process-pool materialization must be byte-identical to the inline path"

    def request_log(options: IngestionOptions) -> list[dict[str, Any]]:
        path = options.manifest_root / "manifests" / "api-request-log.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert request_log(pooled_options) == request_log(inline_options)

    inline_plans = inline["api_policy"]["taiwan_adjustment_workers_by_provider"]
    pooled_plans = pooled["api_policy"]["taiwan_adjustment_workers_by_provider"]
    assert set(inline_plans) == set(pooled_plans) == {"tpex_official", "twse_official"}
    assert all(plan["backend"] == MATERIALIZATION_INLINE_BACKEND for plan in inline_plans.values())
    expected = min(3, max(1, ingestion.detect_visible_cpu_count() // 2))
    for plan in pooled_plans.values():
        assert plan["override"] == 3
        assert plan["count"] <= expected
        if expected > 1:
            assert plan["backend"] == MATERIALIZATION_PROCESS_BACKEND
            assert plan["count"] > 1
    quality = {key: value for key, value in pooled["quality"].items() if key != "execution"}
    assert quality == {key: value for key, value in inline["quality"].items() if key != "execution"}
