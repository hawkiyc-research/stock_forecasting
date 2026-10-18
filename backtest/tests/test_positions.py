import numpy as np
import pytest

from twbacktest import engine, positions
from twbacktest.costs import Scenario, side_costs
from twbacktest.data import SplitIndex

SESSIONS = 24
DATES = 4  # signal dates are sessions 0..3, so there are three intervals
RNG = np.random.default_rng(5)
# Opens and closes of stocks A (symbol 0) and B (symbol 1) and of the index.
OPEN = 100 * np.exp(np.cumsum(RNG.normal(0, 0.01, size=(3, SESSIONS)), axis=1))
CLOSE = OPEN * np.exp(RNG.normal(0, 0.01, size=(3, SESSIONS)))
A, B, INDEX = 0, 1, 2


def build(opens, closes, scores, *, drop=()):
    """Rows for every (symbol, date) except drop; abs[h-1] = log(close[j+h] / open[j+1])."""
    rows = [(symbol, date) for date in range(DATES) for symbol in (A, B)
            if (symbol, date) not in drop]
    horizons = np.arange(1, 15)
    absolute = np.array([np.log(closes[s, d + horizons] / opens[s, d + 1]) for s, d in rows])
    bench = np.array([np.log(closes[INDEX, d + horizons] / opens[INDEX, d + 1]) for _s, d in rows])
    index = SplitIndex(
        split="validation",
        rows=np.arange(len(rows)),
        date_codes=np.array([d for _s, d in rows], dtype=np.int32),
        dates=tuple(f"2025-06-0{day + 2}" for day in range(DATES)),
        market_codes=np.zeros(len(rows), dtype=np.int8),
        is_etf=np.zeros(len(rows), dtype=bool),
        symbol_codes=np.array([s for s, _d in rows], dtype=np.int32),
    )
    link = positions.links(index)
    move = positions.moves(absolute, absolute - bench, link)
    groups = positions.date_groups(index, (0,))
    score = np.array([scores[d][s] for s, d in rows], dtype=np.float64)
    return index, link, move, groups, engine.rankings(score, groups), absolute


# A leads on date 0, B leads afterwards.
SCORES = [{A: 2.0, B: 1.0}, {A: 1.0, B: 2.0}, {A: 1.0, B: 2.0}, {A: 2.0, B: 1.0}]
BUY, SELL = 0.01, 0.02


def run(opens=OPEN, closes=CLOSE, *, buffer=1, drop=(), buy=BUY, sell=SELL):
    index, link, move, _groups, ranked, _abs = build(opens, closes, SCORES, drop=drop)
    path = positions.simulate(ranked, k=1, buffer=buffer, symbols=index.symbol_codes, link=link,
                              move=move, limit=0.095)
    result = positions.replay(path, np.full(index.count, buy), np.full(index.count, sell))
    return path, result, move


def ratio(symbol, start, end):
    return OPEN[symbol, end] / OPEN[symbol, start]


def test_links_identity_and_gaps_follow_the_prices():
    index, link, move, _groups, _ranked, absolute = build(OPEN, CLOSE, SCORES, drop={(A, 2)})
    a_rows = np.flatnonzero(index.symbol_codes == A)  # dates 0, 1, 3
    assert link.next_row[a_rows].tolist() == [a_rows[1], -1, -1]
    assert link.prev_row[a_rows].tolist() == [-1, a_rows[0], -1]
    pairs, fraction = positions.identity_fraction(absolute, link, 1e-9)
    assert pairs == 4 and fraction == 1.0
    broken = absolute.copy()
    broken[a_rows[0], 5] += 0.01
    assert positions.identity_fraction(broken, link, 1e-9)[1] == pytest.approx(0.75)
    assert move.growth[a_rows[0]] == pytest.approx(np.log(ratio(A, 1, 2)))
    assert move.gap[a_rows[1]] == pytest.approx(np.log(OPEN[A, 2] / CLOSE[A, 1]))
    assert np.isnan(move.gap[a_rows[0]]) and np.isnan(move.growth[a_rows[1]])
    assert move.index_growth[a_rows[0]] == pytest.approx(np.log(ratio(INDEX, 1, 2)))


def test_daily_replacement_pays_each_round_trip_without_day_trade_tax():
    path, result, _move = run()
    # A is bought at the open of session 1, swapped for B at the open of session 2, and B is
    # liquidated at the open of session 4.
    assert result.daily == pytest.approx([
        (1 - BUY) * ratio(A, 1, 2) - 1,
        (1 - SELL) * (1 - BUY) * ratio(B, 2, 3) - 1,
        ratio(B, 3, 4) * (1 - SELL) - 1,
    ])
    assert path.holding_periods.tolist() == [1, 2]
    assert positions.path_statistics(path)["turnover"] == pytest.approx(0.5)


def test_buffer_keeps_a_stock_that_stays_within_b_times_k():
    path, result, _move = run(buffer=2)
    assert result.daily == pytest.approx([
        (1 - BUY) * ratio(A, 1, 2) - 1, ratio(A, 2, 3) - 1, ratio(A, 3, 4) * (1 - SELL) - 1,
    ])
    assert path.holding_periods.tolist() == [3]
    assert positions.path_statistics(path)["turnover"] == 0.0


def test_limit_up_open_leaves_the_slot_in_cash():
    opens = OPEN.copy()
    opens[B, 2] = CLOSE[B, 1] * 1.12
    path, result, _move = run(opens=opens)
    assert path.blocked_buys == 1 and (path.held[1] < 0).all()
    assert result.daily[1] == pytest.approx(-SELL)
    assert result.daily[2] == pytest.approx((1 - BUY) * ratio(B, 3, 4) * (1 - SELL) - 1)


def test_limit_down_open_keeps_the_stock_until_the_next_execution():
    opens = OPEN.copy()
    opens[A, 2] = CLOSE[A, 1] * 0.88
    path, result, _move = run(opens=opens)
    assert path.blocked_sells == 1
    assert result.daily[1] == pytest.approx(opens[A, 3] / opens[A, 2] - 1)
    assert result.daily[2] == pytest.approx((1 - SELL) * (1 - BUY) * ratio(B, 3, 4)
                                            * (1 - SELL) - 1)


def test_missing_next_row_exits_at_the_close_and_waits_one_execution():
    path, result, _move = run(drop={(A, 1)})
    assert path.missing_exits == 1 and path.buy[1, 0] == -1
    assert result.daily == pytest.approx([
        (1 - BUY) * CLOSE[A, 2] / OPEN[A, 1] * (1 - SELL) - 1,
        0.0,
        (1 - BUY) * ratio(B, 3, 4) * (1 - SELL) - 1,
    ])


def test_hedge_and_equal_weight_use_open_to_open_returns():
    path, result, move = run(buffer=2, buy=0.0, sell=0.0)
    hedge = positions.hedge_returns(path, result, move)
    assert hedge == pytest.approx([ratio(INDEX, j + 1, j + 2) - 1 for j in range(3)])
    _index, _link, move, groups, _ranked, _abs = build(OPEN, CLOSE, SCORES)
    expected = [(ratio(A, j + 1, j + 2) + ratio(B, j + 1, j + 2)) / 2 - 1 for j in range(3)]
    assert positions.equal_weight(groups, move) == pytest.approx(expected)


def test_evaluate_reports_both_versions_with_overnight_costs():
    index, link, move, groups, ranked, _abs = build(OPEN, CLOSE, SCORES)
    costs = {"commission_rate": 0.001425, "transaction_tax": {"stock": 0.003, "etf": 0.001},
             "slippage_per_side": {"TWSE": 0.001, "TPEX": 0.002}}
    scenario = Scenario(0.6, 1.0, "positions")
    buy, sell = side_costs(costs, scenario, index.market_codes, index.is_etf, same_day=False)
    assert sell[0] == pytest.approx(0.001425 * 0.6 + 0.003 + 0.001)
    path = positions.simulate(ranked, k=1, buffer=1, symbols=index.symbol_codes, link=link,
                              move=move, limit=0.095)
    benchmark = positions.equal_weight(groups, move)
    rows = positions.evaluate(
        path, move, benchmark, rates={scenario: (buy, sell)}, versions=["long_only", "hedged"],
        annualization_days=252,
        bootstrap={"replicates": 20, "seed": 1, "block": 2},
    )
    assert [row["version"] for row in rows] == ["long_only", "hedged"]
    daily = positions.replay(path, buy, sell).daily
    long_only = rows[0]
    assert long_only["net_excess_mean"] == pytest.approx((daily - benchmark).mean())
    assert long_only["break_even_cost_per_side"] == pytest.approx(
        long_only["gross_excess_annual"] / (2 * long_only["turnover"] * 252)
    )
    assert long_only["holding_mean"] == pytest.approx(1.5)
