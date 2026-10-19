import numpy as np
import pytest

from twbacktest import holding, positions
from twbacktest.data import SplitIndex

DATES, SESSIONS, STOCKS = 6, 30, 3
RNG = np.random.default_rng(9)
OPEN = 100 * np.exp(np.cumsum(RNG.normal(0, 0.01, size=(STOCKS + 1, SESSIONS)), axis=1))
CLOSE = OPEN * np.exp(RNG.normal(0, 0.01, size=(STOCKS + 1, SESSIONS)))
INDEX = STOCKS  # last price row is the index


def build(drop=()):
    """Rows for every (stock, date) except drop; abs[h-1] = log(close[d+h] / open[d+1])."""
    rows = [(s, d) for d in range(DATES) for s in range(STOCKS) if (s, d) not in drop]
    horizons = np.arange(1, 15)
    absolute = np.array([np.log(CLOSE[s, d + horizons] / OPEN[s, d + 1]) for s, d in rows])
    bench = np.array([np.log(CLOSE[INDEX, d + horizons] / OPEN[INDEX, d + 1]) for _s, d in rows])
    index = SplitIndex(
        split="validation", rows=np.arange(len(rows)),
        date_codes=np.array([d for _s, d in rows], dtype=np.int32),
        dates=tuple(f"2025-06-{day + 2:02d}" for day in range(DATES)),
        market_codes=np.zeros(len(rows), dtype=np.int8), is_etf=np.zeros(len(rows), dtype=bool),
        symbol_codes=np.array([s for s, _d in rows], dtype=np.int32),
    )
    return index, absolute, absolute - bench, rows


def test_open_to_open_labels_follow_the_prices():
    index, absolute, alpha, rows = build(drop={(1, 3)})
    table = positions.symbol_date_table(index)
    labels = holding.open_to_open(absolute, alpha, index, table, 2)
    for position, (stock, date) in enumerate(rows):
        if date + 2 >= DATES or (stock == 1 and date + 2 == 3):
            assert np.isnan(labels.stock[position]) and np.isnan(labels.session[position])
            continue
        expected = np.log(OPEN[stock, date + 3] / OPEN[stock, date + 1])
        session = sum(np.log(CLOSE[stock, date + 1 + j] / OPEN[stock, date + 1 + j])
                      for j in range(2))
        assert labels.stock[position] == pytest.approx(expected)
        if stock == 1 and date + 1 == 3:
            # The window's end row exists, but a session inside it has no row.
            assert np.isnan(labels.session[position])
            continue
        assert labels.session[position] == pytest.approx(session)
        assert labels.overnight[position] == pytest.approx(expected - session)
        assert labels.index[position] == pytest.approx(
            np.log(OPEN[INDEX, date + 3] / OPEN[INDEX, date + 1])
        )
    with pytest.raises(ValueError, match="horizons"):
        holding.open_to_open(absolute, alpha, index, table, 14)


def test_rows_ahead_skips_missing_rows():
    index, _absolute, _alpha, rows = build(drop={(0, 2)})
    table = positions.symbol_date_table(index)
    ahead = positions.rows_ahead(index, table, 2)
    first = rows.index((0, 0))
    assert ahead[first] == -1
    assert ahead[rows.index((0, 1))] == rows.index((0, 3))
    assert ahead[rows.index((2, 4))] == -1  # beyond the last date


def test_rank_ic_uses_average_ties():
    scores = np.array([1.0, 1.0, 2.0, 3.0, np.nan])
    target = np.array([0.1, 0.3, 0.2, 0.4, 0.5])
    groups = (np.arange(5), np.array([0, 1]))
    daily = holding.daily_rank_ic(scores, target, groups)
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 3, 2, 4])[0, 1]
    assert daily[0] == pytest.approx(expected) and np.isnan(daily[1])
    summary = holding.rank_summary(np.array([0.1, 0.3, np.nan]))
    assert summary["dates"] == 2 and summary["mean"] == pytest.approx(0.2)
    assert summary["ir"] == pytest.approx(0.2 / np.std([0.1, 0.3], ddof=1))


def test_deciles_and_session_split_round_half_up():
    scores = np.arange(15, dtype=float)  # 15 stocks: 10% rounds to 2
    simple = np.linspace(-0.07, 0.07, 15)
    groups = (np.arange(15),)
    result = holding.deciles(scores, simple, groups, 0.1)
    assert result.top[0] == pytest.approx(simple[-2:].mean() - simple.mean())
    assert result.bottom[0] == pytest.approx(simple[:2].mean() - simple.mean())
    labels = holding.OpenToOpen(stock=simple, index=np.zeros(15), session=simple / 2)
    session, overnight = holding.session_split(scores, labels, groups, 0.1)
    assert session[0] == pytest.approx((simple[-2:].mean() - simple.mean()) / 2)
    assert overnight[0] == pytest.approx(session[0])
