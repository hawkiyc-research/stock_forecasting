import numpy as np
import pytest

from twbacktest import stats
from twbacktest.costs import Scenario, primary_scenario, scenarios, side_costs

COSTS = {
    "commission_rate": 0.001425,
    "transaction_tax": {"stock": 0.003, "etf": 0.001},
    "day_trade_tax": {"stock": 0.0015, "etf": 0.001},
    "slippage_per_side": {"TWSE": 0.001, "TPEX": 0.002},
    "discounts": [1.0, 0.6],
    "slippage_multipliers": [0, 1],
    "turnover_modes": ["conservative", "optimistic"],
    "primary": {"discount": 0.6, "slippage_multiplier": 1, "turnover": "conservative"},
}


def test_side_costs_apply_discount_tax_day_trade_and_market_slippage():
    markets = np.array([0, 1, 0], dtype=np.int8)
    etf = np.array([False, False, True])
    scenario = Scenario(0.6, 1.0, "conservative")
    buy, sell = side_costs(COSTS, scenario, markets, etf, same_day=False)
    commission = 0.001425 * 0.6
    assert buy == pytest.approx([commission + 0.001, commission + 0.002, commission + 0.001])
    assert sell == pytest.approx([
        commission + 0.003 + 0.001, commission + 0.003 + 0.002, commission + 0.001 + 0.001,
    ])
    _buy, same_day = side_costs(COSTS, scenario, markets, etf, same_day=True)
    assert same_day[0] == pytest.approx(commission + 0.0015 + 0.001)
    assert same_day[2] == pytest.approx(commission + 0.001 + 0.001)
    free, _ = side_costs(COSTS, Scenario(1.0, 0.0, "conservative"), markets, etf, same_day=False)
    assert free == pytest.approx(np.full(3, 0.001425))


def test_scenario_grid_and_primary():
    assert len(scenarios(COSTS)) == 8
    assert primary_scenario(COSTS) == Scenario(0.6, 1.0, "conservative")


def test_bootstrap_ratio_drawdown_and_compounding():
    mean, low, high = stats.bootstrap_mean(np.full(40, 0.01), block=5, replicates=200, seed=1)
    assert (mean, low, high) == pytest.approx((0.01, 0.01, 0.01))
    series = np.array([0.01, -0.02, 0.03, 0.0, np.nan])
    assert stats.ratio(series, 252) == pytest.approx(
        np.mean(series[:4]) / np.std(series[:4], ddof=1) * np.sqrt(252)
    )
    assert np.isnan(stats.ratio(np.array([0.1, 0.2]), 252))
    assert stats.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert stats.compound_annual(np.array([0.1, 0.1]), 2) == pytest.approx(0.21)
    first = stats.block_indices(20, 4, 10, 7)
    assert first.shape == (10, 20) and np.array_equal(first, stats.block_indices(20, 4, 10, 7))
