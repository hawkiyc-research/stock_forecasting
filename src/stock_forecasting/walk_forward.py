"""Quarterly expanding-window walk-forward schedule (experiment protocol 13.12 item 4).

Fold q predicts the signal dates of one calendar quarter. Its training window starts at a fixed
date and ends ``embargo_sessions`` trading sessions before the quarter, so no training label
overlaps the prediction period; the last ``validation_months`` of the window are the fold's
validation sample for early stopping and selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SCHEDULE_SCHEMA = "walk-forward-schedule-v1"
DEFAULT_SCHEDULE_PATH = Path("configs/walk_forward/tw_quarterly_v1.json")


@dataclass(frozen=True)
class Fold:
    """Exclusive end bounds; every date is a UTC calendar date string."""

    index: int
    name: str
    train_start: str
    validation_start: str
    train_end: str
    predict_start: str
    predict_end: str


def load_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> dict:
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema",
        "name",
        "calendar",
        "train_start",
        "data_end",
        "first_quarter",
        "last_quarter",
        "embargo_sessions",
        "validation_months",
    }
    if not required <= set(schedule) or schedule["schema"] != SCHEDULE_SCHEMA:
        raise ValueError("Walk-forward schedule is incomplete or uses an unsupported schema")
    if type(schedule["embargo_sessions"]) is not int or schedule["embargo_sessions"] < 1:
        raise ValueError("embargo_sessions must be a positive integer")
    if type(schedule["validation_months"]) is not int or schedule["validation_months"] < 1:
        raise ValueError("validation_months must be a positive integer")
    pd.Period(schedule["first_quarter"], freq="Q")
    pd.Period(schedule["last_quarter"], freq="Q")
    if "periods" not in schedule:
        # 2026-11-02 rule: the report periods follow the data end, they are not hand-written.
        schedule["periods"] = derive_periods(schedule)
        schedule["periods_derived"] = True
    elif not isinstance(schedule["periods"], dict) or not schedule["periods"]:
        raise ValueError("periods must be a non-empty object or absent (derived)")
    return schedule


def derive_periods(schedule: dict) -> dict[str, list[str]]:
    """Report periods from ``data_end`` (the user's 2026-11-02 rule).

    P3, the out-of-sample period, is the last complete calendar quarter before ``data_end``
    plus whatever part of the current quarter the data covers: at least one quarter, under two
    (data_end 2026-10-23 → P3 from 2026-07-01; 2027-01-01 → 2026Q4 only; 2027-03-08 → 2026Q4
    plus 2027Q1 to 03-08). P2 is the four quarters before P3; P1 runs from the first fold's
    quarter to P2. Every bound is a half-open ISO date.
    """
    data_end = pd.Timestamp(schedule["data_end"]).normalize()
    last_day = pd.Period(data_end - pd.Timedelta(days=1), freq="Q")
    complete = data_end >= (last_day + 1).start_time.normalize()
    p3_start = (last_day if complete else last_day - 1).start_time.normalize()
    p2_start = (pd.Period(p3_start, freq="Q") - 4).start_time.normalize()
    p1_start = pd.Period(schedule["first_quarter"], freq="Q").start_time.normalize()
    if not p1_start < p2_start:
        raise ValueError("The schedule is too short for P1 before the four P2 quarters")
    fmt = "%Y-%m-%d"
    return {
        "P1": [p1_start.strftime(fmt), p2_start.strftime(fmt)],
        "P2": [p2_start.strftime(fmt), p3_start.strftime(fmt)],
        "P3": [p3_start.strftime(fmt), data_end.strftime(fmt)],
    }


def quarterly_folds(sessions: pd.DatetimeIndex, schedule: dict) -> list[Fold]:
    """Folds over the given trading sessions (tz-naive dates, ascending, unique)."""
    if not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError("Sessions must be ascending and unique")
    sessions = sessions.tz_localize(None) if sessions.tz is not None else sessions
    sessions = sessions.normalize()
    train_start = pd.Timestamp(schedule["train_start"])
    data_end = pd.Timestamp(schedule["data_end"])
    if sessions[0] < train_start or sessions[-1] >= data_end:
        raise ValueError("Sessions must lie inside [train_start, data_end)")
    first = pd.Period(schedule["first_quarter"], freq="Q")
    last = pd.Period(schedule["last_quarter"], freq="Q")
    if last < first:
        raise ValueError("last_quarter precedes first_quarter")
    embargo, months = int(schedule["embargo_sessions"]), int(schedule["validation_months"])
    folds = []
    quarter = first
    while quarter <= last:
        quarter_start = quarter.start_time.normalize()
        next_start = (quarter + 1).start_time.normalize()
        position = int(sessions.searchsorted(quarter_start, side="left"))
        if position >= len(sessions):
            raise ValueError(f"No sessions remain for {quarter}")
        predict_start = sessions[position]
        predict_end = min(next_start, data_end)
        if position - embargo <= 0:
            raise ValueError(f"Fold {quarter} leaves no training sessions after the embargo")
        train_end = sessions[position - embargo]
        validation_start = train_end - pd.DateOffset(months=months)
        if validation_start <= train_start:
            raise ValueError(f"Fold {quarter} has no training sessions before its validation")
        validation_position = int(sessions.searchsorted(validation_start, side="left"))
        folds.append(
            Fold(
                index=len(folds),
                name=str(quarter),
                train_start=train_start.strftime("%Y-%m-%d"),
                validation_start=sessions[validation_position].strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                predict_start=predict_start.strftime("%Y-%m-%d"),
                predict_end=predict_end.strftime("%Y-%m-%d"),
            )
        )
        quarter += 1
    return folds


def taiwan_sessions(schedule: dict) -> pd.DatetimeIndex:
    """Trading sessions of the schedule's calendar over [train_start, data_end)."""
    from stock_forecasting.data.sample_universe import market_sessions

    sessions = market_sessions(schedule["calendar"], schedule["train_start"], schedule["data_end"])
    sessions = sessions.tz_localize(None) if sessions.tz is not None else sessions
    sessions = sessions.normalize()
    # The calendar helper includes its end date; the data contract excludes it.
    return sessions[sessions < pd.Timestamp(schedule["data_end"])]


def fold_frame(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([asdict(fold) for fold in folds])


def assign_folds(dates: pd.DatetimeIndex, folds: list[Fold]) -> pd.Series:
    """Fold index for each signal date (-1 outside every prediction period)."""
    dates = dates.tz_localize(None) if dates.tz is not None else dates
    starts = pd.DatetimeIndex([fold.predict_start for fold in folds])
    ends = pd.DatetimeIndex([fold.predict_end for fold in folds])
    position = starts.searchsorted(dates.normalize(), side="right") - 1
    valid = (position >= 0) & (dates.normalize().values < ends.values[position.clip(0)])
    result = pd.Series(position, index=dates)
    result[~valid] = -1
    return result


def period_of(dates: pd.DatetimeIndex, schedule: dict) -> pd.Series:
    """Report period name (P1, P2, ...) for each date; empty string outside every period."""
    dates = dates.tz_localize(None) if dates.tz is not None else dates
    result = pd.Series("", index=dates, dtype=object)
    for name, (start, end) in schedule["periods"].items():
        days = dates.normalize()
        inside = (days >= pd.Timestamp(start)) & (days < pd.Timestamp(end))
        result[inside] = name
    return result
