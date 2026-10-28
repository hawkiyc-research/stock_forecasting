"""Check the pinned cleaning calendar against the sessions the raw data actually traded.

The sample universe (``data/sample_universe.py``) treats every calendar session as a required
bar: a session with no bar inside a 128-session window rejects the window as an
``asset_market_session_gap``. An exchange closure the pinned ``exchange_calendars`` release does
not know (a typhoon day, for example 2026-07-10) therefore silently empties about six months
of samples unless it is listed under ``calendar_overrides.closed`` in
``configs/data_cleaning.json``. This audit makes that loud: for every market of a raw
``market.parquet`` it compares the calendar sessions over the data's own date range with the
dates on which at least one instrument of that market has a bar, and fails when the two
disagree. The policy stays the single source of truth; the audit never edits it, it names the
dates that need an exchange announcement and an override.

Run before the feature stores are built (``stock-forecasting-build-features``) and on demand::

    python -m stock_forecasting.data.calendar_audit \\
        --raw /runpod-volume/datasets/<id>/raw/market.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from stock_forecasting.data_policy import load_data_policy


def observed_trading_dates(raw_path: Path) -> dict[str, list[str]]:
    """Per market, the sorted dates (exchange-local labels) with at least one bar."""
    parquet = pq.ParquetFile(raw_path)
    seen: dict[str, set] = {}
    for batch in parquet.iter_batches(columns=["timestamp", "market"], batch_size=1 << 18):
        frame = batch.to_pandas()
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_localize(None)
        days = stamps.normalize()
        for market, group in pd.Series(days, index=frame["market"].to_numpy()).groupby(level=0):
            seen.setdefault(str(market), set()).update(group.unique())
    return {
        market: sorted(pd.Timestamp(day).strftime("%Y-%m-%d") for day in days)
        for market, days in seen.items()
    }


def audit_calendar(
    observed: dict[str, list[str]],
    policy: dict[str, Any] | None = None,
    *,
    end_exclusive: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Per market: calendar sessions without any bar, and bars on days the calendar closes.

    ``end_exclusive`` (the dataset request's end date) bounds the audited range: bars the raw
    file may hold beyond the requested range are a ragged tail, not evidence either way.
    """
    from stock_forecasting.data.sample_universe import market_sessions

    policy = policy or load_data_policy()
    report: dict[str, dict[str, Any]] = {}
    for market, days in sorted(observed.items()):
        if end_exclusive is not None:
            days = [day for day in days if day < end_exclusive]
        if not days:
            continue
        calendar = policy["calendars"].get(market)
        if calendar is None:
            report[market] = {"calendar": None, "skipped": "no calendar is configured"}
            continue
        sessions = {
            value.strftime("%Y-%m-%d") for value in market_sessions(calendar, days[0], days[-1])
        }
        traded = set(days)
        report[market] = {
            "calendar": calendar,
            "first": days[0],
            "last": days[-1],
            "trading_days": len(traded),
            "sessions_without_bars": sorted(sessions - traded),
            "bars_off_calendar": sorted(traded - sessions),
        }
    return report


def inconsistencies(report: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for market, entry in report.items():
        if entry.get("skipped"):
            continue
        if entry["sessions_without_bars"]:
            lines.append(
                f"{market} ({entry['calendar']}): calendar sessions without any bar "
                f"{entry['sessions_without_bars']} -> after checking the exchange announcement, "
                f"add them to calendar_overrides.{entry['calendar']}.closed in "
                "configs/data_cleaning.json"
            )
        if entry["bars_off_calendar"]:
            lines.append(
                f"{market} ({entry['calendar']}): bars on days the calendar closes "
                f"{entry['bars_off_calendar']} -> after checking the exchange announcement, add "
                f"them to calendar_overrides.{entry['calendar']}.open in configs/data_cleaning.json"
            )
    return lines


def require_consistent_calendar(
    raw_path: Path,
    policy: dict[str, Any] | None = None,
    *,
    end_exclusive: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Audit ``raw_path`` and raise when the calendar and the data disagree."""
    report = audit_calendar(
        observed_trading_dates(Path(raw_path)), policy, end_exclusive=end_exclusive
    )
    problems = inconsistencies(report)
    if problems:
        raise ValueError(
            "The cleaning calendar disagrees with the raw data; every window across these "
            "dates would be rejected as an asset_market_session_gap:\n  " + "\n  ".join(problems)
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="raw market.parquet")
    parser.add_argument("--end-exclusive", help="dataset end date (YYYY-MM-DD, exclusive)")
    args = parser.parse_args(argv)
    report = audit_calendar(observed_trading_dates(args.raw), end_exclusive=args.end_exclusive)
    print(json.dumps(report, indent=2))
    problems = inconsistencies(report)
    for line in problems:
        print(f"CALENDAR MISMATCH: {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
