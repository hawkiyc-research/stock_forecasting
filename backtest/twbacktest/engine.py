"""Top-K selection and next-open, h-session holding returns with h start sleeves.

The batch chosen after the close of signal date i is bought at the next open and
sold at the close h sessions later; the same sleeve buys its next batch at the
following open, so sleeve o holds the batches of dates o, o + h, o + 2h, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from twbacktest import stats
from twbacktest.costs import Scenario, side_costs
from twbacktest.data import SplitIndex


@dataclass(frozen=True)
class Universe:
    """Rows of one market subset, grouped by signal date (positions in split arrays)."""

    groups: tuple[np.ndarray, ...]

    @property
    def n_dates(self) -> int:
        return len(self.groups)


def universe(index: SplitIndex, market_codes: tuple[int, ...]) -> Universe:
    member = np.isin(index.market_codes, market_codes)
    positions = np.flatnonzero(member)
    codes = index.date_codes[positions]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(index.dates) + 1))
    groups = tuple(positions[order[bounds[i] : bounds[i + 1]]] for i in range(len(index.dates)))
    return Universe(groups=tuple(group for group in groups if group.size))


def rankings(scores: np.ndarray, groups: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Per date, the rows with a finite score from best to worst (ties keep row order)."""
    ranked = []
    for rows in groups:
        valid = rows[np.isfinite(scores[rows])]
        ranked.append(valid[np.argsort(-scores[valid], kind="stable")])
    return ranked


@dataclass(frozen=True)
class Batches:
    """One value per signal date, in date order."""

    gross: np.ndarray
    net: np.ndarray
    universe: np.ndarray
    index: np.ndarray
    turnover: np.ndarray


def batches(
    ranked: list[np.ndarray],
    groups: tuple[np.ndarray, ...],
    *,
    k: int,
    h: int,
    symbols: np.ndarray,
    simple: np.ndarray,
    bench: np.ndarray,
    buy: np.ndarray,
    sell: np.ndarray,
    optimistic: bool,
    hedged: bool,
) -> Batches:
    """Equal-weight batch returns before and after costs.

    simple and bench are the h-session simple returns of each row and of its
    index; rows are stock-dates, so holdings are compared by symbol code. A row's
    net return is (1 + r)(1 - sell) - 1 - buy; with optimistic turnover, stocks
    held in the previous (or next) batch of the same sleeve pay no buy (or sell)
    cost. The hedged version subtracts the index return.
    """
    selected = [rows[:k] for rows in ranked]
    n = len(selected)
    gross = np.full(n, np.nan)
    net = np.full(n, np.nan)
    turnover = np.full(n, np.nan)
    universe_mean = np.array([simple[rows].mean() if rows.size else np.nan for rows in groups])
    index_mean = np.array([bench[rows].mean() if rows.size else np.nan for rows in groups])
    for i, rows in enumerate(selected):
        if rows.size == 0:
            continue
        names = symbols[rows]
        previous = symbols[selected[i - h]] if i >= h else None
        following = symbols[selected[i + h]] if i + h < n else None
        buy_cost, sell_cost = buy[rows], sell[rows]
        if optimistic and previous is not None:
            buy_cost = np.where(np.isin(names, previous), 0.0, buy_cost)
        if optimistic and following is not None:
            sell_cost = np.where(np.isin(names, following), 0.0, sell_cost)
        returns = simple[rows]
        row_net = (1.0 + returns) * (1.0 - sell_cost) - 1.0 - buy_cost
        if hedged:
            returns = returns - bench[rows]
            row_net = row_net - bench[rows]
        gross[i] = returns.mean()
        net[i] = row_net.mean()
        if previous is not None and previous.size:
            turnover[i] = 1.0 - np.isin(names, previous).mean()
    return Batches(gross=gross, net=net, universe=universe_mean, index=index_mean,
                   turnover=turnover)


def summarize(
    result: Batches,
    *,
    h: int,
    hedged: bool,
    annualization_days: int,
    bootstrap: dict[str, Any],
) -> dict[str, float]:
    """Metrics of one configuration (per-batch values and annualized ones)."""
    per_year = annualization_days / h
    excess = result.net if hedged else result.net - result.universe
    gross_excess = result.gross if hedged else result.gross - result.universe
    sleeves = [np.arange(offset, result.net.size, h) for offset in range(h)]

    def sleeve_mean(function, values) -> float:
        estimates = [function(values[idx]) for idx in sleeves if idx.size >= 3]
        finite = [value for value in estimates if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    block = max(int(bootstrap["min_block"]), int(bootstrap["block_per_holding_day"]) * h)
    mean, low, high = stats.bootstrap_mean(
        excess, block=block, replicates=int(bootstrap["replicates"]), seed=int(bootstrap["seed"])
    )
    finite_net = result.net[np.isfinite(result.net)]
    finite_excess = excess[np.isfinite(excess)]
    return {
        "batches": int(finite_net.size),
        "net_excess_mean": mean,
        "net_excess_low": low,
        "net_excess_high": high,
        "net_excess_annual": mean * per_year,
        "net_excess_annual_low": low * per_year,
        "net_excess_annual_high": high * per_year,
        "net_excess_ir": sleeve_mean(lambda v: stats.ratio(v, per_year), excess),
        "gross_excess_mean": float(np.nanmean(gross_excess)),
        "index_excess_annual": (
            float(np.nanmean(result.net - result.index)) * per_year if not hedged else float("nan")
        ),
        "net_return_annual": float(np.nanmean(result.net)) * per_year,
        "net_cagr": sleeve_mean(lambda v: stats.compound_annual(v, per_year), result.net),
        "sharpe": sleeve_mean(lambda v: stats.ratio(v, per_year), result.net),
        "max_drawdown": sleeve_mean(stats.max_drawdown, result.net),
        "hit_rate_net": float(np.mean(finite_net > 0)) if finite_net.size else float("nan"),
        "hit_rate_excess": (
            float(np.mean(finite_excess > 0)) if finite_excess.size else float("nan")
        ),
        "turnover": float(np.nanmean(result.turnover)) if np.isfinite(result.turnover).any()
        else float("nan"),
        "cost_drag_mean": float(np.nanmean(result.gross - result.net)),
        # Round-trip cost (fraction per batch) at which the gross excess is used up.
        "break_even_round_trip": float(np.nanmean(gross_excess)),
    }


def evaluate(
    scores_h: np.ndarray,
    log_return_h: np.ndarray,
    log_alpha_h: np.ndarray,
    index: SplitIndex,
    group: Universe,
    *,
    h: int,
    top_k: list[int],
    scenario_list: list[Scenario],
    costs: dict[str, Any],
    annualization_days: int,
    bootstrap: dict[str, Any],
) -> list[dict[str, Any]]:
    """All K, version and cost combinations of one signal, market and holding period."""
    simple = np.expm1(log_return_h)
    bench = np.expm1(log_return_h - log_alpha_h)
    ranked = rankings(scores_h, group.groups)
    side = {
        scenario: side_costs(costs, scenario, index.market_codes, index.is_etf, same_day=h == 1)
        for scenario in scenario_list
    }
    rows = []
    for k in top_k:
        for scenario in scenario_list:
            buy, sell = side[scenario]
            for version in ("long_only", "hedged"):
                result = batches(
                    ranked, group.groups, k=k, h=h, symbols=index.symbol_codes,
                    simple=simple, bench=bench, buy=buy,
                    sell=sell, optimistic=scenario.turnover == "optimistic",
                    hedged=version == "hedged",
                )
                rows.append({
                    "k": k,
                    "version": version,
                    "discount": scenario.discount,
                    "slippage_multiplier": scenario.slippage_multiplier,
                    "turnover_mode": scenario.turnover,
                    **summarize(result, h=h, hedged=version == "hedged",
                                annualization_days=annualization_days, bootstrap=bootstrap),
                })
    return rows
