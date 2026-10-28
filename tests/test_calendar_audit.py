"""The cleaning-calendar audit: unlisted closures and off-calendar bars are named, not dropped."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_forecasting.data import calendar_audit
from stock_forecasting.data.sample_universe import market_sessions


def _raw(tmp_path, dates_by_market):
    rows = []
    for market, dates in dates_by_market.items():
        for symbol in ("1101", "1102"):
            for day in dates:
                rows.append({
                    "timestamp": pd.Timestamp(day, tz="UTC"), "symbol": f"{symbol}.{market}",
                    "market": market, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                    "volume": 1000.0, "adjusted_close": 10.0,
                })
    path = tmp_path / "market.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _sessions(start, end):
    return [d.strftime("%Y-%m-%d") for d in market_sessions("XTAI", start, end)]


def test_data_that_trades_on_every_calendar_session_passes(tmp_path):
    # June to August 2026 includes the 2026-07-10 typhoon closure, which the policy lists.
    days = _sessions("2026-06-01", "2026-08-31")
    assert "2026-07-10" not in days and "2026-07-09" in days
    report = calendar_audit.require_consistent_calendar(_raw(tmp_path, {"TWSE": days}))
    entry = report["TWSE"]
    assert entry["calendar"] == "XTAI" and entry["trading_days"] == len(days)
    assert entry["sessions_without_bars"] == [] and entry["bars_off_calendar"] == []


def test_an_unlisted_closure_is_named_and_fails(tmp_path):
    days = [d for d in _sessions("2026-06-01", "2026-08-31") if d != "2026-07-14"]
    raw = _raw(tmp_path, {"TWSE": days, "TPEX": [*days, "2026-07-14"]})
    report = calendar_audit.audit_calendar(calendar_audit.observed_trading_dates(raw))
    assert report["TWSE"]["sessions_without_bars"] == ["2026-07-14"]
    assert report["TPEX"]["sessions_without_bars"] == []
    with pytest.raises(ValueError, match=r"2026-07-14.*calendar_overrides\.XTAI\.closed"):
        calendar_audit.require_consistent_calendar(raw)


def test_a_bar_on_a_calendar_holiday_is_named_as_an_open_candidate(tmp_path):
    days = [*_sessions("2026-06-01", "2026-08-31"), "2026-07-11"]  # a Saturday
    raw = _raw(tmp_path, {"TWSE": days})
    with pytest.raises(ValueError, match=r"2026-07-11.*calendar_overrides\.XTAI\.open"):
        calendar_audit.require_consistent_calendar(raw)
    assert calendar_audit.main(["--raw", str(raw)]) == 1


def test_markets_without_a_configured_calendar_are_skipped(tmp_path):
    days = _sessions("2026-06-01", "2026-06-30")
    raw = _raw(tmp_path, {"XX": days})
    report = calendar_audit.require_consistent_calendar(raw)
    assert report["XX"]["skipped"]


def test_the_dataset_end_bounds_the_audit(tmp_path):
    # Bars beyond the requested end are a ragged tail; a closure inside the range still counts.
    days = [d for d in _sessions("2026-06-01", "2026-10-31") if d != "2026-10-23"]
    raw = _raw(tmp_path, {"TWSE": days})
    report = calendar_audit.audit_calendar(
        calendar_audit.observed_trading_dates(raw), end_exclusive="2026-10-23")
    assert report["TWSE"]["last"] < "2026-10-23"
    assert report["TWSE"]["sessions_without_bars"] == []
    assert calendar_audit.main(["--raw", str(raw), "--end-exclusive", "2026-10-23"]) == 0
    inside = tmp_path / "inside"
    inside.mkdir()
    gap = _raw(inside, {"TWSE": [d for d in days if d != "2026-10-01"]})
    with pytest.raises(ValueError, match="2026-10-01"):
        calendar_audit.require_consistent_calendar(gap, end_exclusive="2026-10-23")
