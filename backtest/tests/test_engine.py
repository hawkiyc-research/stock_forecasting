import numpy as np
import pytest

from twbacktest import engine
from twbacktest.data import SplitIndex


def make_index(date_codes, market_codes, symbol_codes):
    count = len(date_codes)
    return SplitIndex(
        split="validation",
        rows=np.arange(count),
        date_codes=np.array(date_codes, dtype=np.int32),
        dates=tuple(f"2025-06-0{day + 1}" for day in range(max(date_codes) + 1)),
        market_codes=np.array(market_codes, dtype=np.int8),
        is_etf=np.zeros(count, dtype=bool),
        symbol_codes=np.array(symbol_codes, dtype=np.int32),
    )


def test_universe_groups_rows_by_date_and_market():
    index = make_index([1, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 1, 2, 3, 4])
    twse = engine.universe(index, (0,))
    assert [group.tolist() for group in twse.groups] == [[1, 4], [0]]
    both = engine.universe(index, (0, 1))
    assert [sorted(group.tolist()) for group in both.groups] == [[1, 3, 4], [0, 2]]


# Two stocks (A = symbol 0, B = symbol 1) on three dates; row = 2 * date + stock.
SCORES = np.array([2.0, 1.0, 2.0, 1.0, 1.0, 2.0])  # A, A, then B ranked first
SIMPLE = np.array([0.10, 0.00, 0.05, -0.05, 0.02, 0.04])
BENCH = np.full(6, 0.01)
BUY = np.full(6, 0.01)
SELL = np.full(6, 0.02)
INDEX = make_index([0, 0, 1, 1, 2, 2], [0] * 6, [0, 1, 0, 1, 0, 1])


def run(optimistic=False, hedged=False, k=1, h=1):
    group = engine.universe(INDEX, (0,))
    ranked = engine.rankings(SCORES, group.groups)
    return engine.batches(
        ranked, group.groups, k=k, h=h, symbols=INDEX.symbol_codes, simple=SIMPLE, bench=BENCH,
        buy=BUY, sell=SELL, optimistic=optimistic, hedged=hedged,
    )


def test_conservative_batches_pay_full_round_trips():
    result = run()
    assert result.gross == pytest.approx([0.10, 0.05, 0.04])
    assert result.net == pytest.approx([1.10 * 0.98 - 1.01, 1.05 * 0.98 - 1.01, 1.04 * 0.98 - 1.01])
    assert result.universe == pytest.approx([0.05, 0.0, 0.03])
    assert result.index == pytest.approx([0.01, 0.01, 0.01])
    assert np.isnan(result.turnover[0]) and result.turnover[1:] == pytest.approx([0.0, 1.0])


def test_optimistic_turnover_waives_costs_of_kept_stocks():
    result = run(optimistic=True)
    # A is kept from date 0 to date 1: no sell on date 0, no buy on date 1.
    assert result.net == pytest.approx([1.10 - 1.01, 1.05 * 0.98 - 1.0, 1.04 * 0.98 - 1.01])


def test_hedged_batches_subtract_the_index():
    result = run(hedged=True)
    assert result.gross == pytest.approx([0.09, 0.04, 0.03])
    assert result.net == pytest.approx(
        [1.10 * 0.98 - 1.01 - 0.01, 1.05 * 0.98 - 1.01 - 0.01, 1.04 * 0.98 - 1.01 - 0.01]
    )


def test_two_day_sleeves_compare_batches_h_dates_apart():
    result = run(k=1, h=2, optimistic=True)
    # Sleeve 0 holds dates 0 and 2 (A then B); sleeve 1 holds date 1 only.
    assert np.isnan(result.turnover[:2]).all() and result.turnover[2] == pytest.approx(1.0)
    assert result.net == pytest.approx([1.10 * 0.98 - 1.01, 1.05 * 0.98 - 1.01,
                                        1.04 * 0.98 - 1.01])


def test_summary_reports_excess_over_the_equal_weight_population():
    result = run()
    summary = engine.summarize(
        result, h=1, hedged=False, annualization_days=252,
        bootstrap={"replicates": 50, "seed": 3, "min_block": 1, "block_per_holding_day": 1},
    )
    excess = result.net - result.universe
    assert summary["net_excess_mean"] == pytest.approx(excess.mean())
    assert summary["net_excess_annual"] == pytest.approx(excess.mean() * 252)
    assert summary["gross_excess_mean"] == pytest.approx((result.gross - result.universe).mean())
    assert summary["break_even_round_trip"] == pytest.approx(summary["gross_excess_mean"])
    assert summary["batches"] == 3 and summary["hit_rate_net"] == pytest.approx(1.0)
    assert summary["net_excess_ir"] == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    )
