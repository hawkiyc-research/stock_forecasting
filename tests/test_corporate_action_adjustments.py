"""Synthetic splits, capital reductions and same-day dividends through the adjustment path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.data.adjustments import (
    apply_cumulative_adjustments,
    asof_adjusted_window,
    execution_total_return,
)


def _bars(symbol: str, closes: list[float], *, start: str = "2025-06-02") -> pd.DataFrame:
    # Business-day bars whose open is one tick under the close and whose volume is constant.
    timestamps = pd.bdate_range(start, periods=len(closes), tz="UTC")
    close = np.asarray(closes, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(closes), 1_000.0),
        }
    )


def _action(
    symbol: str,
    timestamp: str,
    price_factor: float,
    share_multiplier: float,
    source: str,
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "price_factor": price_factor,
        "share_multiplier": share_multiplier,
        "source": source,
    }


def test_etf_split_scales_earlier_adjusted_close_and_split_volume() -> None:
    # 0050-like 1:4 split: the ETF closed near 190 before a five-session halt and
    # resumed near 47.5 on 2025-06-18.
    frame = _bars("0050.TW", [188.0, 189.0, 190.0, 47.5, 47.6], start="2025-06-04")
    frame.loc[3:, "timestamp"] = pd.DatetimeIndex(["2025-06-18", "2025-06-19"], tz="UTC")
    events = pd.DataFrame([_action("0050.TW", "2025-06-18", 47.16 / 188.65, 4.0, "twse_twtcau")])

    adjusted = apply_cumulative_adjustments(frame, events)

    before = adjusted.iloc[:3]
    after = adjusted.iloc[3:]
    np.testing.assert_allclose(before["adjusted_close"], before["close"] * (47.16 / 188.65))
    np.testing.assert_allclose(after["adjusted_close"], after["close"])
    np.testing.assert_allclose(before["split_adjusted_volume"], 4_000.0)
    np.testing.assert_allclose(after["split_adjusted_volume"], 1_000.0)
    # Raw prices and raw volume are never rewritten.
    np.testing.assert_allclose(adjusted["close"], frame["close"])
    np.testing.assert_allclose(adjusted["open"], frame["open"])
    np.testing.assert_allclose(adjusted["volume"], 1_000.0)
    # The adjusted series is continuous across the halt: 190 * 0.25 ≈ 47.5.
    ratio = adjusted.loc[3, "adjusted_close"] / adjusted.loc[2, "adjusted_close"]
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_loss_covering_capital_reduction_raises_earlier_adjusted_close() -> None:
    # 10.5 before the halt, 17.09 reference after a 1,000 -> 614 share reduction.
    frame = _bars("2025.TW", [10.0, 10.5, 17.09, 17.0], start="2025-01-27")
    frame.loc[2:, "timestamp"] = pd.DatetimeIndex(["2025-02-12", "2025-02-13"], tz="UTC")
    events = pd.DataFrame(
        [_action("2025.TW", "2025-02-12", 17.09 / 10.5, 10.5 / 17.09, "twse_twtauu")]
    )

    adjusted = apply_cumulative_adjustments(frame, events)

    assert adjusted.loc[1, "adjusted_close"] == pytest.approx(17.09)
    assert adjusted.loc[0, "adjusted_close"] == pytest.approx(10.0 * 17.09 / 10.5)
    assert adjusted.loc[2, "adjusted_close"] == pytest.approx(17.09)
    assert adjusted.loc[1, "split_adjusted_volume"] == pytest.approx(1_000.0 * 10.5 / 17.09)
    assert adjusted.loc[2, "split_adjusted_volume"] == pytest.approx(1_000.0)


def test_split_and_cash_dividend_on_the_same_date_multiply_into_one_factor() -> None:
    frame = _bars("0050.TW", [200.0, 200.0, 49.0, 49.0])
    events = pd.DataFrame(
        [
            _action("0050.TW", "2025-06-04", 0.98, 1.0, "twse_twt49u"),
            _action("0050.TW", "2025-06-04", 0.25, 4.0, "twse_twtcau"),
        ]
    )

    adjusted = apply_cumulative_adjustments(frame, events)

    np.testing.assert_allclose(adjusted.loc[:1, "adjusted_close"], 200.0 * 0.98 * 0.25)
    np.testing.assert_allclose(adjusted.loc[2:, "adjusted_close"], 49.0)
    np.testing.assert_allclose(adjusted.loc[:1, "split_adjusted_volume"], 4_000.0)
    np.testing.assert_allclose(adjusted.loc[2:, "split_adjusted_volume"], 1_000.0)


def test_reduction_after_an_earlier_dividend_compounds_both_factors() -> None:
    frame = _bars("3623.TWO", [40.0, 42.0, 43.8, 66.0, 65.0])
    events = pd.DataFrame(
        [
            _action("3623.TWO", "2025-06-03", 0.95, 1.0, "tpex_exdailyq"),
            _action("3623.TWO", "2025-06-05", 66.0 / 43.8, 43.8 / 66.0, "tpex_revivt"),
        ]
    )

    adjusted = apply_cumulative_adjustments(frame, events)

    assert adjusted.loc[0, "adjusted_close"] == pytest.approx(40.0 * 0.95 * 66.0 / 43.8)
    assert adjusted.loc[1, "adjusted_close"] == pytest.approx(42.0 * 66.0 / 43.8)
    assert adjusted.loc[2, "adjusted_close"] == pytest.approx(66.0)
    assert adjusted.loc[3, "adjusted_close"] == pytest.approx(66.0)


def test_cutoff_window_and_execution_return_see_no_split_gap() -> None:
    frame = _bars("0050.TW", [190.0, 190.0, 190.0, 47.5, 47.5, 47.5])
    events = pd.DataFrame([_action("0050.TW", "2025-06-05", 0.25, 4.0, "twse_twtcau")])
    adjusted = apply_cumulative_adjustments(frame, events)

    # A window whose cutoff is after the split rescales the earlier raw bars.
    window = asof_adjusted_window(adjusted)
    np.testing.assert_allclose(window["close"], [47.5] * 6)
    np.testing.assert_allclose(window["open"], [47.5 * 0.99] * 6)
    np.testing.assert_allclose(window["volume"], [4_000.0] * 3 + [1_000.0] * 3)

    # A window whose cutoff is before the split keeps the raw scale (causal factor 1).
    earlier = asof_adjusted_window(adjusted.iloc[:3])
    np.testing.assert_allclose(earlier["close"], [190.0] * 3)
    np.testing.assert_allclose(earlier["volume"], [1_000.0] * 3)

    # Entering before the split and exiting after it is a flat total return.
    gross = execution_total_return(
        adjusted,
        entry_at=adjusted.loc[1, "timestamp"],
        exit_at=adjusted.loc[4, "timestamp"],
    )
    assert gross == pytest.approx(47.5 / (190.0 * 0.99 * 0.25) - 1.0)
