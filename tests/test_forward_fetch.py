"""Forward Taiwan fetch orchestration with fake providers (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_forecasting.data.providers.base import ProviderFetch
from stock_forecasting.data.providers.http import TpexRelayTransport
from stock_forecasting.data.sample_universe import market_sessions
from stock_forecasting.forward import fetch as forward_fetch

START, END = "2026-05-01", "2026-07-01"


def _frame(day, symbol, asset_type, market, provider):
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(day, tz="UTC")],
            "symbol": [symbol],
            "asset_type": [asset_type],
            "market": [market],
            "currency": ["TWD"],
            "provider": [provider],
            "dataset_profile": ["tw_only"],
            "source_symbol": [symbol.split(".")[0]],
            "is_active": [True],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "adjusted_close": [100.5],
            "volume": [1000.0],
            "split_adjusted_volume": [1000.0],
        }
    )


def _fake(name, market, stock, index, sessions, calls):
    class Fake:
        def __init__(self, client):
            self.client = client
            calls.append((name, "client", client))

        def fetch_actions(self, *, start, end):
            calls.append((name, "actions", start, end))
            return ProviderFetch(frame=pd.DataFrame(), requests=(), metadata={})

        def fetch_benchmark_month(self, *, month, dataset_profile):
            assert dataset_profile == "tw_only"
            days = [day for day in sessions if day.startswith(month)]
            frame = pd.concat([_frame(day, index, "index", market, name) for day in days])
            return ProviderFetch(frame=frame, requests=(), metadata={})

        def fetch_adjusted_date(self, *, date, dataset_profile, action_frame):
            calls.append((name, "date", date))
            fetched = ProviderFetch(frame=pd.DataFrame(), requests=(), metadata={"dropped_rows": 1})
            return fetched, _frame(date, stock, "stock", market, name)

    return Fake


@pytest.fixture
def official(monkeypatch):
    calendar = [
        day.date().isoformat()
        for day in market_sessions("XTAI", START, "2026-06-30")
    ]
    # One unscheduled closure that the pinned calendar does not know.
    traded = [day for day in calendar if day != "2026-06-10"]
    calls = []
    monkeypatch.setattr(
        forward_fetch,
        "PROVIDERS",
        {
            "twse_official": _fake("twse_official", "TWSE", "2330.TW", "TAIEX.TW", traded, calls),
            "tpex_official": _fake("tpex_official", "TPEX", "6488.TWO", "TPEX.TWO", traded, calls),
        },
    )
    return traded, calls


def test_fetch_writes_raw_bars_and_reports_calendar_differences(tmp_path, official):
    traded, calls = official
    manifest = forward_fetch.fetch_taiwan(
        output_root=tmp_path, start=START, end_exclusive=END, progress=lambda _message: None
    )
    raw = pd.read_parquet(tmp_path / "raw" / "market.parquet")
    assert set(raw["symbol"]) == {"2330.TW", "6488.TWO", "TAIEX.TW", "TPEX.TWO"}
    assert len(raw) == 4 * len(traded)
    assert manifest["providers"]["twse_official"]["sessions"] == len(traded)
    assert manifest["providers"]["tpex_official"]["dropped_source_rows"] == len(traded)
    assert manifest["raw"]["benchmarks_present"] == ["TAIEX.TW", "TPEX.TWO"]
    assert manifest["calendar"] == {
        "calendar_sessions_without_trading": ["2026-06-10"],
        "trading_days_missing_from_calendar": [],
    }
    assert ("twse_official", "actions", START, "2026-06-30") in calls
    assert forward_fetch.load_fetch(tmp_path)["raw"]["rows"] == len(raw)
    with pytest.raises(FileExistsError):
        forward_fetch.fetch_taiwan(output_root=tmp_path, start=START, end_exclusive=END)


def test_fetch_rejects_tampering_and_invalid_intervals(tmp_path, official):
    forward_fetch.fetch_taiwan(
        output_root=tmp_path, start=START, end_exclusive=END, progress=lambda _message: None
    )
    payload = json.loads((tmp_path / "fetch.json").read_text())
    payload["raw"]["sha256"] = "0" * 64
    (tmp_path / "fetch.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed after"):
        forward_fetch.load_fetch(tmp_path)
    for start, end in ((END, START), ("2026-5-01", END), (START, "2999-01-01")):
        with pytest.raises(ValueError):
            forward_fetch.fetch_taiwan(output_root=tmp_path / "x", start=start, end_exclusive=end)


def test_shared_cache_reads_every_dataset_cache_and_relays_only_tpex(tmp_path, official):
    _traded, calls = official
    datasets = tmp_path / "volume" / "datasets"
    (datasets / "training" / "api-cache").mkdir(parents=True)
    shared = datasets / "forward-taiwan-2026-07-01" / "api-cache"
    relay = TpexRelayTransport(origin="https://relay-test.a.run.app", token="t" * 40)
    manifest = forward_fetch.fetch_taiwan(
        output_root=tmp_path / "fetch",
        start=START,
        end_exclusive=END,
        cache_root=shared,
        shared_cache=True,
        tpex_transport=relay,
        progress=lambda _message: None,
    )
    clients = {name: client for name, kind, *rest in calls if kind == "client" for client in rest}
    for name, client in clients.items():
        assert client.raw_cache_root == shared
        assert client.read_cache_roots == ((datasets / "training" / "api-cache").resolve(),)
        assert client.cache_scope(
            "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST", {"date": "20260601"}
        ) == {"month": "complete"}
        assert client.transport is (relay if name == "tpex_official" else None)
    assert manifest["cache"] == {
        "root": str(shared), "shared_read_roots": 1, "tpex_relay": True
    }
    assert not (tmp_path / "fetch" / "api-cache").exists()
    with pytest.raises(ValueError, match="datasets"):
        forward_fetch.fetch_taiwan(
            output_root=tmp_path / "other",
            start=START,
            end_exclusive=END,
            cache_root=tmp_path / "loose" / "api-cache",
            shared_cache=True,
        )
