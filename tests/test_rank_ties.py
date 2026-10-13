"""Numerical RankIC regressions for the cloud test environment."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import spearmanr

from stock_forecasting.metrics import _rank, cross_sectional_metrics


def metrics(targets, signals, dates=None):
    return cross_sectional_metrics(
        targets=np.asarray(targets, dtype=np.float64),
        signals=np.asarray(signals, dtype=np.float64),
        dates=dates or ["2026-01-02"] * len(targets),
        symbols=[f"symbol-{i}" for i in range(len(targets))],
        transaction_cost_bps=0.0,
    )


def test_average_ranks_and_empty_input():
    np.testing.assert_array_equal(_rank(np.asarray([3, 1, 1, 2])), [3.0, 0.5, 0.5, 2.0])
    np.testing.assert_array_equal(_rank(np.asarray([5, 5, 5])), [1.0, 1.0, 1.0])
    assert _rank(np.asarray([])).shape == (0,)


def test_spearman_with_ties_is_order_invariant():
    targets = np.asarray([1.0, 1.0, 2.0, 3.0])
    signals = np.asarray([1.0, 2.0, 2.0, 4.0])
    # Centered average ranks are [-1, -1, .5, 1.5] and [-1.5, 0, 0, 1.5].
    # Their dot product is 3.75 and each squared norm is 4.5: rho = 5/6.
    expected = 5.0 / 6.0
    assert math.isclose(spearmanr(targets, signals).statistic, expected, abs_tol=1e-12)
    for permutation in itertools.permutations(range(4)):
        indices = list(permutation)
        result = metrics(targets[indices], signals[indices])
        assert math.isclose(result["rank_ic_mean"], expected, abs_tol=1e-12)
        assert result["rank_ic_status"] == "ok"
        assert result["rank_ic_evaluated_dates"] == 1.0
        assert result["rank_ic_method"] == "spearman_average_ties_v1"
        assert result["rank_ic_ir_status"] == "unavailable"


def test_constant_target_does_not_create_rank_ic_but_keeps_portfolio_date():
    result = metrics([2, 2, 2], [1, 2, 3])
    assert result["rank_ic_mean"] == 0.0
    assert result["rank_ic_status"] == "unavailable"
    assert result["rank_ic_evaluated_dates"] == 0.0
    assert result["rank_ic_ir_status"] == "unavailable"
    assert result["evaluated_dates"] == 1.0
    assert result["net_long_short_mean_log_return"] == 0.0


def test_constant_signal_keeps_existing_portfolio_skip_and_marks_rank_unavailable():
    result = metrics([1, 2, 3], [2, 2, 2])
    assert result["rank_ic_status"] == "unavailable"
    assert result["rank_ic_evaluated_dates"] == 0.0
    assert result["evaluated_dates"] == 0.0
    assert result["rank_ic_ir_status"] == "unavailable"


def test_constant_and_insufficient_dates_are_not_included_in_rank_mean():
    result = metrics(
        [1, 2, 3, 4, 4, 4, 1, 2, 3, 1, float("nan"), 3],
        [3, 1, 2, 1, 2, 3, 4, 4, 4, 1, 2, 3],
        [day for day in ("2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05") for _ in range(3)],
    )
    assert math.isclose(result["rank_ic_mean"], -0.5, abs_tol=1e-12)
    assert result["rank_ic_status"] == "ok"
    assert result["rank_ic_evaluated_dates"] == 1.0
    assert result["evaluated_dates"] == 2.0


def test_rank_ir_uses_only_valid_dates_and_remains_unannualized():
    result = metrics(
        [1, 2, 3, 1, 2, 3],
        [3, 1, 2, 1, 2, 3],
        ["2026-01-02"] * 3 + ["2026-01-03"] * 3,
    )
    assert math.isclose(result["rank_ic_mean"], 0.25, abs_tol=1e-12)
    assert math.isclose(result["rank_ic_ir"], 0.25 / math.sqrt(1.125), abs_tol=1e-12)
    assert result["rank_ic_ir_status"] == "ok"
    assert result["rank_ic_evaluated_dates"] == 2.0
