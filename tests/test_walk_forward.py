"""Quarterly walk-forward schedule: fold bounds, embargo, coverage and period labels."""

import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schedule():
    from stock_forecasting.walk_forward import DEFAULT_SCHEDULE_PATH, load_schedule

    return load_schedule(ROOT / DEFAULT_SCHEDULE_PATH)


@pytest.fixture(scope="module")
def sessions(schedule):
    from stock_forecasting.walk_forward import taiwan_sessions

    return taiwan_sessions(schedule)


def test_schedule_is_the_registered_design(schedule):
    assert schedule["calendar"] == "XTAI"
    assert (schedule["train_start"], schedule["data_end"]) == ("2016-01-01", "2026-06-01")
    assert (schedule["first_quarter"], schedule["last_quarter"]) == ("2023Q1", "2026Q2")
    assert schedule["embargo_sessions"] == 15 and schedule["validation_months"] == 12
    assert schedule["periods"] == {
        "P1": ["2023-01-01", "2025-06-01"], "P2": ["2025-06-01", "2026-06-01"],
    }


def test_load_schedule_rejects_other_schemas(tmp_path):
    from stock_forecasting.walk_forward import load_schedule

    payload = json.loads((ROOT / "configs/walk_forward/tw_quarterly_v1.json").read_text())
    payload["embargo_sessions"] = 0
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="embargo_sessions"):
        load_schedule(path)


def test_fourteen_contiguous_quarterly_folds_with_embargo(schedule, sessions):
    from stock_forecasting.walk_forward import assign_folds, fold_frame, quarterly_folds

    folds = quarterly_folds(sessions, schedule)
    assert [fold.name for fold in folds] == [
        str(q) for q in pd.period_range("2023Q1", "2026Q2", freq="Q")
    ]
    assert len(folds) == 14
    assert folds[0].predict_start == "2023-01-03"
    assert folds[-1].predict_end == "2026-06-01"
    positions = {str(s.date()): i for i, s in enumerate(sessions)}
    for fold in folds:
        assert fold.train_start == "2016-01-01"
        start, end = positions[fold.predict_start], positions[fold.train_end]
        assert start - end == 15
        # The last training signal date's 14-session label closes before the quarter.
        assert sessions[end - 1 + 14] < pd.Timestamp(fold.predict_start)
        gap = pd.Timestamp(fold.train_end) - pd.Timestamp(fold.validation_start)
        assert pd.Timedelta(days=355) <= gap <= pd.Timedelta(days=372)
        assert fold.validation_start > fold.train_start
    for previous, fold in itertools.pairwise(folds):
        assert previous.predict_end <= fold.predict_start
        gap = pd.Timestamp(fold.predict_start) - pd.Timestamp(previous.predict_end)
        assert gap <= pd.Timedelta(days=10)
    frame = fold_frame(folds)
    assert list(frame.columns) == [
        "index", "name", "train_start", "validation_start", "train_end", "predict_start",
        "predict_end",
    ]
    evaluated = sessions[sessions >= pd.Timestamp("2023-01-01")]
    assignment = assign_folds(evaluated, folds)
    assert (assignment >= 0).all()
    counts = assignment.value_counts().sort_index()
    assert counts.index.tolist() == list(range(14))
    assert counts.iloc[:-1].min() >= 50
    # The last fold ends with the data (2026-04 to 2026-05), two months rather than three.
    assert 35 <= counts.iloc[-1] <= 45
    before = assign_folds(sessions[sessions < pd.Timestamp("2023-01-01")], folds)
    assert (before == -1).all()


def test_period_labels(schedule):
    from stock_forecasting.walk_forward import period_of

    dates = pd.DatetimeIndex(["2022-12-30", "2023-01-03", "2025-05-30", "2025-06-02", "2026-05-29"])
    assert period_of(dates, schedule).tolist() == ["", "P1", "P1", "P2", "P2"]


def test_extended_schedule_adds_two_folds_and_the_p3_period():
    """tw_quarterly_v2 (protocol 13.20) keeps every v1 fold and adds 2026Q3, 2026Q4 and P3."""
    from stock_forecasting.walk_forward import (
        DEFAULT_SCHEDULE_PATH,
        load_schedule,
        period_of,
        quarterly_folds,
        taiwan_sessions,
    )

    v1 = load_schedule(ROOT / DEFAULT_SCHEDULE_PATH)
    v2 = load_schedule(ROOT / "configs/walk_forward/tw_quarterly_v2.json")
    assert (v2["train_start"], v2["data_end"]) == ("2016-01-01", "2026-10-23")
    assert (v2["first_quarter"], v2["last_quarter"]) == ("2023Q1", "2026Q4")
    assert v2["periods"]["P1"] == v1["periods"]["P1"] and v2["periods"]["P2"] == v1["periods"]["P2"]
    assert v2["periods"]["P3"] == ["2026-06-01", "2026-10-23"]
    folds_v1 = quarterly_folds(taiwan_sessions(v1), v1)
    folds_v2 = quarterly_folds(taiwan_sessions(v2), v2)
    assert [f.name for f in folds_v2] == [f.name for f in folds_v1] + ["2026Q3", "2026Q4"]
    # Every shared fold trains on the same window; only the truncated last v1 fold predicts more.
    for a, b in zip(folds_v1[:-1], folds_v2[: len(folds_v1) - 1], strict=True):
        assert (a.train_start, a.train_end, a.predict_start, a.predict_end) == (
            b.train_start, b.train_end, b.predict_start, b.predict_end
        )
    assert str(folds_v2[-1].predict_end) == "2026-10-23"
    dates = pd.DatetimeIndex(["2026-05-29", "2026-06-01", "2026-10-22", "2026-10-23"])
    assert period_of(dates, v2).tolist() == ["P2", "P3", "P3", ""]
