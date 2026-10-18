"""Hold-buffer portfolios with per-position, open-to-open accounting (protocol section 13.5).

Signal dates D_0 .. D_{N-1} are consecutive sessions. Orders decided after the
close of D_j execute at the open of D_{j+1}, and interval j runs from that open
to the open of D_{j+2}. For one stock's rows r (date j) and n (date j + 1),
log(open(D_{j+2}) / open(D_{j+1})) = abs[r, h=2] - abs[n, h=1]; with zero-based
columns that is abs[r, 1] - abs[n, 0]. Capital sits in K slots that are never
rebalanced, so each slot's value is a product of per-interval factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from twbacktest import engine, stats
from twbacktest.costs import Scenario
from twbacktest.data import SplitIndex


@dataclass(frozen=True)
class Links:
    """Row of the same symbol on the next and previous signal dates (-1 when absent)."""

    next_row: np.ndarray
    prev_row: np.ndarray


def links(index: SplitIndex) -> Links:
    dates = len(index.dates)
    table = np.full((dates, int(index.symbol_codes.max()) + 1), -1, dtype=np.int64)
    table[index.date_codes, index.symbol_codes] = np.arange(index.count)
    if int((table >= 0).sum()) != index.count:
        raise ValueError(f"{index.split} has more than one row for a symbol and date")
    next_row = np.full(index.count, -1, dtype=np.int64)
    prev_row = np.full(index.count, -1, dtype=np.int64)
    later = index.date_codes + 1 < dates
    next_row[later] = table[index.date_codes[later] + 1, index.symbol_codes[later]]
    earlier = index.date_codes > 0
    prev_row[earlier] = table[index.date_codes[earlier] - 1, index.symbol_codes[earlier]]
    return Links(next_row=next_row, prev_row=prev_row)


def identity_fraction(log_returns: np.ndarray, link: Links, tolerance: float) -> tuple[int, float]:
    """Adjacent-date pairs, and the share whose implied open-to-open return is the same at
    every horizon (within the tolerance)."""
    rows = np.flatnonzero(link.next_row >= 0)
    if rows.size == 0:
        return 0, float("nan")
    delta = log_returns[rows, 1:] - log_returns[link.next_row[rows], :-1]
    spread = delta.max(axis=1) - delta.min(axis=1)
    return int(rows.size), float(np.mean(spread <= tolerance))


@dataclass(frozen=True)
class Moves:
    """Per-row log returns around the execution that follows the row's signal date."""

    growth: np.ndarray  # next open to the open after it; NaN without a next row
    fallback: np.ndarray  # next open to the close one session later (exit without a next row)
    gap: np.ndarray  # next open over the row's own close; NaN without a previous row
    index_growth: np.ndarray  # the row's index over the same interval (fallback when no next row)


def moves(absolute: np.ndarray, alpha: np.ndarray, link: Links) -> Moves:
    bench = absolute - alpha
    has_next = link.next_row >= 0
    following = link.next_row[has_next]
    growth = np.full(absolute.shape[0], np.nan)
    growth[has_next] = absolute[has_next, 1] - absolute[following, 0]
    index_growth = bench[:, 1].copy()
    index_growth[has_next] = bench[has_next, 1] - bench[following, 0]
    has_prev = link.prev_row >= 0
    previous = link.prev_row[has_prev]
    gap = np.full(absolute.shape[0], np.nan)
    gap[has_prev] = absolute[previous, 1] - absolute[has_prev, 0] - absolute[previous, 0]
    return Moves(growth=growth, fallback=absolute[:, 1].copy(), gap=gap,
                 index_growth=index_growth)


def date_groups(index: SplitIndex, market_codes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Rows of the market on every signal date, aligned with index.dates."""
    groups = engine.universe(index, market_codes).groups
    if len(groups) != len(index.dates):
        raise ValueError(f"{index.split} has signal dates without rows for {market_codes}")
    return groups


def equal_weight(groups: tuple[np.ndarray, ...], move: Moves) -> np.ndarray:
    """Cost-free, daily-rebalanced equal-weight return of each interval."""
    output = np.full(len(groups) - 1, np.nan)
    for j, rows in enumerate(groups[:-1]):
        growth = move.growth[rows]
        growth = growth[np.isfinite(growth)]
        if growth.size:
            output[j] = np.expm1(growth).mean()
    return output


@dataclass(frozen=True)
class Path:
    """Trades and holdings of one portfolio; arrays are [interval, slot] rows, -1 for none."""

    buy: np.ndarray  # bought at the open that starts the interval
    sell_open: np.ndarray  # sold at that open
    sell_close: np.ndarray  # sold at a close inside the interval (no next row)
    held: np.ndarray  # held during the interval
    growth: np.ndarray  # log growth of the slot over the interval, 0 for cash
    final: np.ndarray  # [slot] rows liquidated at the open that ends the last interval
    holding_periods: np.ndarray  # intervals held by every closed position
    blocked_buys: int
    blocked_sells: int
    missing_exits: int


def simulate(
    ranked: list[np.ndarray],
    *,
    k: int,
    buffer: int,
    symbols: np.ndarray,
    link: Links,
    move: Moves,
    limit: float,
) -> Path:
    """Run the hold-buffer rule; ranked holds each signal date's rows from best to worst."""
    intervals = len(ranked) - 1
    if intervals < 1:
        raise ValueError("At least two signal dates are needed")
    rank = np.full(symbols.size, np.inf)
    for rows in ranked:
        rank[rows] = np.arange(1, rows.size + 1)
    up, down = math.log1p(limit), math.log1p(-limit)
    keep_limit = buffer * k
    shape = (intervals, k)
    buy, sell_open, sell_close, held = (np.full(shape, -1, dtype=np.int64) for _ in range(4))
    growth = np.zeros(shape)
    slot = np.full(k, -1, dtype=np.int64)  # the held stock's row on the current signal date
    entry = np.zeros(k, dtype=np.int64)
    waiting = np.zeros(k, dtype=bool)  # cash that arrives only at a close inside the interval
    periods: list[int] = []
    blocked_buys = blocked_sells = missing_exits = 0
    for j in range(intervals):
        sold = set()
        for s in range(k):
            row = slot[s]
            if row < 0 or rank[row] <= keep_limit:
                continue
            if move.gap[row] <= down:  # NaN gaps compare False and count as fillable
                blocked_sells += 1
                continue
            sell_open[j, s] = row
            sold.add(int(symbols[row]))
            periods.append(j - int(entry[s]))
            slot[s] = -1
        taken = {int(symbols[row]) for row in slot if row >= 0} | sold
        candidates = (row for row in ranked[j] if int(symbols[row]) not in taken)
        for s in range(k):
            if slot[s] >= 0 or waiting[s]:
                continue
            row = next(candidates, None)
            if row is None:
                break
            if move.gap[row] >= up:  # the slot stays in cash until the next execution
                blocked_buys += 1
                continue
            buy[j, s] = row
            slot[s] = row
            entry[s] = j
        waiting[:] = False
        for s in range(k):
            row = slot[s]
            if row < 0:
                continue
            held[j, s] = row
            following = link.next_row[row]
            if following >= 0:
                growth[j, s] = move.growth[row]
                slot[s] = following
            else:
                growth[j, s] = move.fallback[row]
                sell_close[j, s] = row
                missing_exits += 1
                periods.append(j - int(entry[s]) + 1)
                slot[s] = -1
                waiting[s] = True
    periods.extend(intervals - int(entry[s]) for s in range(k) if slot[s] >= 0)
    return Path(buy=buy, sell_open=sell_open, sell_close=sell_close, held=held, growth=growth,
                final=slot.copy(), holding_periods=np.array(periods, dtype=np.int64),
                blocked_buys=blocked_buys, blocked_sells=blocked_sells,
                missing_exits=missing_exits)


@dataclass(frozen=True)
class Replay:
    daily: np.ndarray  # portfolio return of each interval
    previous: np.ndarray  # portfolio value at the start of each interval, before trades
    invested: np.ndarray  # [interval, slot] value after the opening trades


def replay(path: Path, buy_rate: np.ndarray, sell_rate: np.ndarray) -> Replay:
    """Apply per-row cost rates (fractions of the traded value) to a simulated path."""
    buy_cost = np.append(buy_rate, 0.0)  # row -1 (no trade) picks the appended zero
    sell_cost = np.append(sell_rate, 0.0)
    opening = (1.0 - sell_cost[path.sell_open]) * (1.0 - buy_cost[path.buy])
    factor = opening * np.exp(path.growth) * (1.0 - sell_cost[path.sell_close])
    factor[-1] *= 1.0 - sell_cost[path.final]
    k = factor.shape[1]
    values = np.cumprod(factor, axis=0) / k
    before = np.vstack([np.full((1, k), 1.0 / k), values[:-1]])
    previous = before.sum(axis=1)
    return Replay(daily=values.sum(axis=1) / previous - 1.0, previous=previous,
                  invested=before * opening)


def hedge_returns(path: Path, result: Replay, move: Moves) -> np.ndarray:
    """Index return of the held slots, as a fraction of the portfolio value."""
    index = np.where(path.held >= 0, np.expm1(move.index_growth[path.held]), 0.0)
    return (result.invested * index).sum(axis=1) / result.previous


def path_statistics(path: Path) -> dict[str, float]:
    k = path.buy.shape[1]
    buys = (path.buy >= 0).sum(axis=1)
    periods = path.holding_periods
    return {
        # Newly bought slots per day after the first execution, as a fraction of K.
        "turnover": float(buys[1:].mean() / k) if buys.size > 1 else float("nan"),
        "cash_fraction": float(np.mean(path.held < 0)),
        "holding_mean": float(periods.mean()) if periods.size else float("nan"),
        "holding_median": float(np.median(periods)) if periods.size else float("nan"),
        "holding_p90": float(np.quantile(periods, 0.9)) if periods.size else float("nan"),
        "positions": int(periods.size),
        "blocked_buys": path.blocked_buys,
        "blocked_sells": path.blocked_sells,
        "missing_exits": path.missing_exits,
    }


def summarize(
    net: np.ndarray,
    gross: np.ndarray,
    benchmark: np.ndarray,
    *,
    turnover: float,
    annualization_days: int,
    bootstrap: dict[str, Any],
) -> dict[str, float]:
    """Metrics of one daily net series against its benchmark (and the cost-free series)."""
    excess = net - benchmark
    gross_excess = gross - benchmark
    mean, low, high = stats.bootstrap_mean(
        excess, block=int(bootstrap["block"]), replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    days = annualization_days
    gross_excess_annual = float(np.nanmean(gross_excess)) * days
    two_way = 2.0 * turnover * days
    return {
        "days": int(np.isfinite(net).sum()),
        "net_excess_mean": mean,
        "net_excess_low": low,
        "net_excess_high": high,
        "net_excess_annual": mean * days,
        "net_excess_annual_low": low * days,
        "net_excess_annual_high": high * days,
        "net_excess_ir": stats.ratio(excess, days),
        "gross_excess_annual": gross_excess_annual,
        "net_return_annual": float(np.nanmean(net)) * days,
        "net_cagr": stats.compound_annual(net, days),
        "sharpe": stats.ratio(net, days),
        "max_drawdown": stats.max_drawdown(net),
        "hit_rate_excess": float(np.mean(excess[np.isfinite(excess)] > 0)),
        "cost_drag_annual": float(np.nanmean(gross - net)) * days,
        # Cost per side (fraction of traded value) at which the gross excess is used up.
        "break_even_cost_per_side": (
            gross_excess_annual / two_way if two_way > 0 else float("nan")
        ),
    }


def evaluate(
    path: Path,
    move: Moves,
    benchmark: np.ndarray,
    *,
    rates: dict[Scenario, tuple[np.ndarray, np.ndarray]],
    versions: list[str],
    annualization_days: int,
    bootstrap: dict[str, Any],
) -> list[dict[str, Any]]:
    """Every cost scenario and version of one simulated path.

    Long-only is measured against the equal-weight benchmark; the hedged version
    subtracts the held slots' index return and is measured against zero.
    """
    shared = path_statistics(path)
    free = np.zeros_like(next(iter(rates.values()))[0])
    gross = replay(path, free, free)
    gross_hedge = gross.daily - hedge_returns(path, gross, move)
    rows = []
    for scenario, (buy_rate, sell_rate) in rates.items():
        result = replay(path, buy_rate, sell_rate)
        series = {
            "long_only": (result.daily, gross.daily, benchmark),
            "hedged": (result.daily - hedge_returns(path, result, move), gross_hedge,
                       np.zeros_like(benchmark)),
        }
        for version in versions:
            net, gross_series, base = series[version]
            rows.append({
                "version": version,
                "discount": scenario.discount,
                "slippage_multiplier": scenario.slippage_multiplier,
                **summarize(net, gross_series, base, turnover=shared["turnover"],
                            annualization_days=annualization_days, bootstrap=bootstrap),
                **shared,
            })
    return rows
