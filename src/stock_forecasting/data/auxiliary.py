"""Prepare auxiliary Taiwan research tables: flows, margin, TAIFEX and US series.

The tables never enter the training universe or the dataset identity. Each source
is fetched independently through the cached, throttled provider clients; one
failing source is recorded in the manifest without discarding the others, and
the research job only uses sources whose status is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_forecasting.data.manifest import atomic_write_json
from stock_forecasting.data.providers.http import (
    CachedJsonClient,
    ProviderRequestError,
    TpexRelayTransport,
)
from stock_forecasting.data.providers.taifex import CachedFormCsvClient, TAIFEXSource
from stock_forecasting.data.providers.taiwan_flows import (
    TPExInstitutionalSource,
    TPExMarginSource,
    TWSEInstitutionalSource,
    TWSEMarginSource,
)
from stock_forecasting.data.sample_universe import market_sessions

AUXILIARY_SCHEMA_VERSION = "1.0"
TWSE_SOURCES = ("twse_t86", "twse_margin")
TPEX_SOURCES = ("tpex_institutional", "tpex_margin")
TAIFEX_SOURCES = ("taifex_futures", "taifex_put_call")
SOURCES = (*TWSE_SOURCES, *TPEX_SOURCES, *TAIFEX_SOURCES, "eodhd_series")
DEFAULT_EODHD_SERIES = ("SOX.INDX", "USDTWD.FOREX", "SOXX.US")
# Every other column of an auxiliary table is numeric and is stored as float64.
TEXT_COLUMNS = frozenset({"timestamp", "symbol", "market", "commodity", "investor"})
EODHD_ENDPOINT = "https://eodhd.com/api/eod/{symbol}"
# Asia/Taipei clock time after which session t's value is published; a day offset
# of 1 means the value only exists on the next calendar morning (US closes).
PUBLICATION = {
    "twse_t86": {"local_time": "16:30", "day_offset": 0},
    "tpex_institutional": {"local_time": "16:30", "day_offset": 0},
    "twse_margin": {"local_time": "21:30", "day_offset": 0},
    "tpex_margin": {"local_time": "21:30", "day_offset": 0},
    "taifex_futures": {"local_time": "15:30", "day_offset": 0},
    "taifex_put_call": {"local_time": "15:30", "day_offset": 0},
    "eodhd_series": {"local_time": "06:00", "day_offset": 1},
}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}
RETRYABLE_STATUS_CODES = frozenset({403, 408, 425, 429})
MAX_LISTED_DATES = 20


@dataclass(frozen=True)
class AuxiliaryOptions:
    start: str
    end: str
    output_root: Path
    raw_cache_root: Path
    sources: tuple[str, ...] = SOURCES
    eodhd_series: tuple[str, ...] = DEFAULT_EODHD_SERIES
    taiwan_requests_per_second: float = 0.5
    taifex_requests_per_second: float = 0.5
    eodhd_requests_per_second: float = 2.0
    max_backoff_seconds: float = 60.0
    tpex_proxy_url: str | None = None
    tpex_proxy_token: str | None = field(default=None, repr=False)
    eodhd_api_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.sources) - set(SOURCES))
        if unknown or not self.sources:
            raise ValueError(f"Unknown or empty auxiliary sources: {unknown}")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("Auxiliary start must precede the exclusive end")
        if (self.tpex_proxy_url is None) != (self.tpex_proxy_token is None):
            raise ValueError("TPEX_PROXY_URL and TPEX_PROXY_TOKEN must be configured together")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _TableWriter:
    """Append per-session frames to one Parquet file without holding the table."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.temporary = destination.with_name(f".{destination.name}.partial")
        self.writer = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Fix the schema: a value column that is all None on the first session must not
        # become a null-typed column that later sessions cannot be cast to.
        frame = frame.copy()
        for column in frame.columns:
            if column not in TEXT_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.temporary.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.temporary, table.schema)
        self.writer.write_table(table.cast(self.writer.schema))
        self.rows += len(frame)

    def close(self) -> dict[str, Any] | None:
        if self.writer is None:
            return None
        self.writer.close()
        os.replace(self.temporary, self.destination)
        return {
            "path": self.destination.name,
            "rows": self.rows,
            "sha256": _sha256(self.destination),
        }


def _error_record(error: BaseException) -> dict[str, Any]:
    if isinstance(error, ProviderRequestError):
        return {"type": "provider", **error.metadata()}
    return {"type": type(error).__name__, "message": str(error)[:500]}


def _status(result: dict[str, Any]) -> str:
    # Sessions without rows (for example typhoon closures) are listed, not failures.
    if result.get("error") is not None:
        return "partial" if result.get("rows") else "failed"
    return "complete" if result.get("rows") else "empty"


def _date_loop(
    sources: Iterable[Any],
    sessions: list[str],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    """Fetch every session of the given same-host sources in order (one host, one QPS)."""
    sources = list(sources)
    writers = {
        source.name: _TableWriter(output_root / f"{source.name}.parquet") for source in sources
    }
    state = {
        source.name: {"covered": 0, "empty_sessions": [], "error": None, "requests": 0, "hits": 0}
        for source in sources
    }
    try:
        for date in sessions:
            for source in sources:
                record = state[source.name]
                if record["error"] is not None:
                    continue
                try:
                    fetched = source.fetch_date(date)
                except (ProviderRequestError, ValueError) as error:
                    # A layout change or a stopped provider is systematic; stop this source.
                    record["error"] = {"date": date, **_error_record(error)}
                    continue
                record["requests"] += len(fetched.requests)
                record["hits"] += sum(request.cache_hit for request in fetched.requests)
                if fetched.frame.empty:
                    record["empty_sessions"].append(date)
                else:
                    record["covered"] += 1
                    writers[source.name].write(fetched.frame)
    finally:
        for name, writer in writers.items():
            state[name]["table"] = writer.close()
    for record in state.values():
        record["rows"] = (record.get("table") or {}).get("rows", 0)
        record["empty_session_count"] = len(record["empty_sessions"])
        record["empty_sessions"] = record["empty_sessions"][:MAX_LISTED_DATES]
    return state


def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    first, stop = pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1)
    ranges, cursor = [], first
    while cursor <= stop:
        month_end = min(cursor + pd.offsets.MonthEnd(0), stop)
        ranges.append((cursor.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        cursor = month_end + pd.Timedelta(days=1)
    return ranges


def _taifex_tables(
    client: CachedFormCsvClient,
    names: tuple[str, ...],
    options: AuxiliaryOptions,
) -> dict[str, dict[str, Any]]:
    source = TAIFEXSource(client)
    fetchers: dict[str, Callable[..., Any]] = {
        "taifex_futures": source.fetch_futures_institutional,
        "taifex_put_call": source.fetch_put_call_ratio,
    }
    results = {}
    for name in names:
        writer = _TableWriter(options.output_root / f"{name}.parquet")
        record: dict[str, Any] = {"error": None, "requests": 0, "hits": 0, "months": 0}
        try:
            for start, end in _month_ranges(options.start, options.end):
                fetched = fetchers[name](start=start, end=end)
                record["requests"] += len(fetched.requests)
                record["hits"] += sum(request.cache_hit for request in fetched.requests)
                record["months"] += 1
                writer.write(fetched.frame)
        except (ProviderRequestError, ValueError) as error:
            record["error"] = _error_record(error)
        finally:
            record["table"] = writer.close()
        record["rows"] = (record["table"] or {}).get("rows", 0)
        results[name] = record
    return results


def _eodhd_series(client: CachedJsonClient, options: AuxiliaryOptions) -> dict[str, Any]:
    writer = _TableWriter(options.output_root / "eodhd_series.parquet")
    symbols: dict[str, Any] = {}
    last_session = (pd.Timestamp(options.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for symbol in options.eodhd_series:
        try:
            payload, request = client.get_json(
                EODHD_ENDPOINT.format(symbol=symbol),
                params={
                    "api_token": options.eodhd_api_token,
                    "fmt": "json",
                    "period": "d",
                    "order": "a",
                    "from": options.start,
                    "to": last_session,
                },
            )
        except ProviderRequestError as error:
            # Plan entitlements differ by series; an unavailable symbol is not fatal.
            symbols[symbol] = {"status": "unavailable", "error": _error_record(error)}
            continue
        if not isinstance(payload, list) or not payload:
            symbols[symbol] = {"status": "empty", "cache_hit": request.cache_hit}
            continue
        frame = pd.DataFrame(payload)
        missing = {"date", "open", "high", "low", "close"} - set(frame.columns)
        if missing:
            symbols[symbol] = {"status": "failed", "error": {"missing_fields": sorted(missing)}}
            continue
        if "adjusted_close" not in frame:
            frame["adjusted_close"] = frame["close"]
        prices = ("open", "high", "low", "close", "adjusted_close")
        frame = frame.rename(columns={"date": "timestamp"})[["timestamp", *prices]].astype(
            dict.fromkeys(prices, "float64")
        )
        frame.insert(1, "symbol", symbol)
        writer.write(frame)
        symbols[symbol] = {"status": "complete", "rows": len(frame), "cache_hit": request.cache_hit}
    table = writer.close()
    complete = [name for name, value in symbols.items() if value["status"] == "complete"]
    return {
        "error": None if complete else {"type": "no_series_available"},
        "rows": (table or {}).get("rows", 0),
        "table": table,
        "symbols": symbols,
    }


def prepare_auxiliary_data(
    options: AuxiliaryOptions,
    *,
    json_client_factory: Callable[..., CachedJsonClient] = CachedJsonClient,
    form_client_factory: Callable[..., CachedFormCsvClient] = CachedFormCsvClient,
    session_dates: Callable[[str, str], list[str]] | None = None,
) -> dict[str, Any]:
    """Fetch the requested sources and write tables plus one manifest."""
    options.output_root.mkdir(parents=True, exist_ok=True)
    last_day = (pd.Timestamp(options.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if session_dates is None:
        sessions = [
            stamp.strftime("%Y-%m-%d")
            for stamp in market_sessions("XTAI", options.start, last_day)
        ]
    else:
        sessions = session_dates(options.start, last_day)

    def taiwan_client(provider: str, referer: str, transport=None) -> CachedJsonClient:
        return json_client_factory(
            provider=provider,
            raw_cache_root=options.raw_cache_root,
            max_requests_per_second=options.taiwan_requests_per_second,
            max_backoff_seconds=options.max_backoff_seconds,
            headers={**BROWSER_HEADERS, "Referer": referer},
            retryable_status_codes=set(RETRYABLE_STATUS_CODES),
            transport=transport,
        )

    loops: list[Callable[[], dict[str, dict[str, Any]]]] = []
    twse_names = tuple(name for name in TWSE_SOURCES if name in options.sources)
    if twse_names:
        client = taiwan_client("twse_auxiliary", "https://www.twse.com.tw/")
        sources = [TWSEInstitutionalSource(client), TWSEMarginSource(client)]
        loops.append(lambda: _date_loop(
            [source for source in sources if source.name in twse_names],
            sessions,
            options.output_root,
        ))
    tpex_names = tuple(name for name in TPEX_SOURCES if name in options.sources)
    if tpex_names:
        transport = (
            TpexRelayTransport(origin=options.tpex_proxy_url, token=options.tpex_proxy_token)
            if options.tpex_proxy_url is not None
            else None
        )
        client = taiwan_client("tpex_auxiliary", "https://www.tpex.org.tw/", transport)
        tpex_sources = [TPExInstitutionalSource(client), TPExMarginSource(client)]
        loops.append(lambda: _date_loop(
            [source for source in tpex_sources if source.name in tpex_names],
            sessions,
            options.output_root,
        ))
    results: dict[str, dict[str, Any]] = {}
    # One worker per exchange host; each loop is sequential under its own throttle.
    with ThreadPoolExecutor(max_workers=max(1, len(loops)), thread_name_prefix="aux") as pool:
        futures = [pool.submit(loop) for loop in loops]
        taifex_names = tuple(name for name in TAIFEX_SOURCES if name in options.sources)
        if taifex_names:
            form_client = form_client_factory(
                provider="taifex_auxiliary",
                raw_cache_root=options.raw_cache_root,
                max_requests_per_second=options.taifex_requests_per_second,
                max_backoff_seconds=options.max_backoff_seconds,
                headers={**BROWSER_HEADERS, "Referer": "https://www.taifex.com.tw/"},
                retryable_status_codes=set(RETRYABLE_STATUS_CODES),
            )
            results.update(_taifex_tables(form_client, taifex_names, options))
        if "eodhd_series" in options.sources:
            if not options.eodhd_api_token:
                results["eodhd_series"] = {"error": {"type": "missing_eodhd_api_token"}, "rows": 0}
            else:
                results["eodhd_series"] = _eodhd_series(
                    json_client_factory(
                        provider="eodhd_auxiliary",
                        raw_cache_root=options.raw_cache_root,
                        max_requests_per_second=options.eodhd_requests_per_second,
                        max_backoff_seconds=options.max_backoff_seconds,
                    ),
                    options,
                )
        for future in futures:
            results.update(future.result())

    manifest = {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "kind": "auxiliary-research-data",
        "created_at": datetime.now(UTC).isoformat(),
        "start": options.start,
        "end_exclusive": options.end,
        "sessions": len(sessions),
        "calendar": "XTAI",
        "tpex_transport": "cloud_run_relay" if options.tpex_proxy_url else "direct",
        "sources": {},
    }
    for name in options.sources:
        result = results.get(name, {"error": {"type": "not_run"}, "rows": 0})
        manifest["sources"][name] = {
            "status": _status(result),
            "publication": PUBLICATION[name],
            **{key: value for key, value in result.items() if key != "covered"},
            **({"covered_sessions": result["covered"]} if "covered" in result else {}),
        }
    atomic_write_json(options.output_root / "auxiliary-manifest.json", manifest)
    return manifest


def load_manifest(output_root: Path) -> dict[str, Any]:
    return json.loads((Path(output_root) / "auxiliary-manifest.json").read_text())
