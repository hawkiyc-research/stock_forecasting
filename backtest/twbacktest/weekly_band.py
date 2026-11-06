"""S-A: the weekly low-turnover long band (protocol 13.20 item 3, first written in 13.18 item 7).

The last signal date of every ISO week is a rebalance date. On it the liquidity universe is
ranked by the orthogonalized (v1) h = 5 score of ``scenario_suite``; held names leave the
book only when their rank falls out of the top m x K, empty slots are refilled best-first,
and at most ``turnover_cap`` x K positions are replaced per rebalance (when more names fall
out of the band, only the worst-ranked ones are sold). Between rebalances the book only
carries its positions: a sale blocked by a limit-down open is retried at the next open, and a
name without a following row (halt, delisting) exits at the close the way every band engine
of this package does. Slots freed by such forced exits, and the initial build, are refilled
outside the cap, because they are not discretionary replacements. The book is long only, has
no futures overlay, and is evaluated under the same cost profiles and capital scenarios as
S1 / S2; the equal-weight liquidity universe, 0050 and the TAIEX total-return index are
comparison columns only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, positions
from twbacktest.tw_strategy import (
    CAP_MODES,
    MIN_FILL,
    BandResult,
    _scaled_growth,
    percentile_ranks,
    scaled_rates,
    smooth_by_symbol,
)

SCENARIO = "wk_long"


def weekly_signal_flags(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on the last signal date of every ISO week of ``dates`` (sorted, unique)."""
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)
    calendar = dates.isocalendar()
    key = calendar["year"].to_numpy().astype(np.int64) * 100 + calendar["week"].to_numpy()
    last = ~pd.Series(key).duplicated(keep="last")
    return last.to_numpy()


@dataclass(frozen=True)
class WeeklyBand:
    band: BandResult
    rebalances: int  # rebalance dates inside the sample
    replacements: int  # discretionary band exits executed (each frees a slot for a buy)
    cap_hits: int  # rebalances on which more names fell out of the band than the cap allowed


def simulate_weekly_band(
    ranked: list[np.ndarray],
    *,
    k: int,
    buffer: int,
    turnover_cap: float,
    rebalance: np.ndarray,
    symbols: np.ndarray,
    link: positions.Links,
    move: positions.Moves,
    limit: float,
    lot_cost: np.ndarray | None = None,
    slot_capital: float | None = None,
    odd_lots: bool = False,
    dollar_volume: np.ndarray | None = None,
    participation_cap: float | None = None,
    cap_mode: str = "skip",
    blocked: np.ndarray | None = None,
    stop_fraction: float | None = None,
    adverse: np.ndarray | None = None,
    favorable: np.ndarray | None = None,
    stop_discount: np.ndarray | None = None,
    stop_overshoot: float = 0.0,
) -> WeeklyBand:
    """The band rule of ``tw_strategy.simulate_band`` applied on rebalance dates only, with a
    cap on the number of replacements per rebalance.

    ``ranked`` holds each signal date's rankable rows from best to worst; ``rebalance`` flags
    the signal dates on which the book is reviewed. The capital limits (``lot_cost`` with
    ``slot_capital``, ``dollar_volume`` with ``participation_cap`` under ``cap_mode``) follow
    the long band exactly; the short-book execution layer does not apply to a long-only book.
    A candidate row with ``blocked`` True (under a disposition period, protocol 13.23
    supplement 3) is skipped and counted. The trailing stop (``stop_fraction`` and the intraday
    arrays, supplement 4) follows ``tw_strategy.simulate_band`` exactly and acts every session,
    not only on rebalance dates.
    """
    if stop_fraction is not None and (adverse is None or favorable is None):
        raise ValueError("stop_fraction needs the adverse and favorable intraday arrays")
    if cap_mode not in CAP_MODES:
        raise ValueError(f"Unknown cap_mode {cap_mode!r}")
    if not 0.0 < turnover_cap <= 1.0:
        raise ValueError("turnover_cap must be in (0, 1]")
    intervals = len(ranked) - 1
    if intervals < 1:
        raise ValueError("At least two signal dates are needed")
    if len(rebalance) != len(ranked):
        raise ValueError("rebalance flags must cover every signal date")
    rank = np.full(symbols.size, np.inf)
    for rows in ranked:
        rank[rows] = np.arange(1, rows.size + 1)
    up, down = math.log1p(limit), math.log1p(-limit)
    keep_limit = buffer * k
    cap = max(1, math.floor(turnover_cap * k))
    shape = (intervals, k)
    buy, sell_open, sell_close, held = (np.full(shape, -1, dtype=np.int64) for _ in range(4))
    growth = np.zeros(shape)
    slot = np.full(k, -1, dtype=np.int64)
    entry = np.zeros(k, dtype=np.int64)
    waiting = np.zeros(k, dtype=bool)
    pending = np.zeros(k, dtype=bool)  # a band exit decided but not yet executed
    value = np.full(k, (slot_capital or 0.0) / k)
    odd_slot = np.zeros(k, dtype=bool)
    rows_total = symbols.size
    buy_value = np.full(rows_total, np.nan)
    sell_value = np.full(rows_total, np.nan)
    odd_buy = np.zeros(rows_total, dtype=bool)
    odd_sell = np.zeros(rows_total, dtype=bool)
    slot_fill = np.ones(k)
    fill_row = np.full(rows_total, np.nan)
    deployed = np.zeros(intervals)
    periods: list[int] = []
    blocked_buys = blocked_sells = missing_exits = lot_blocked = buy_attempts = 0
    liquidity_blocked = odd_lot_buys = capped_buys = blocked_skipped = 0
    stop_exits = stop_gap_exits = 0
    stop_pending = np.zeros(k, dtype=bool)
    log_value = np.zeros(k)
    run_max = np.ones(k)
    stop_rows = np.zeros(rows_total, dtype=bool) if stop_fraction is not None else None
    rebalances = replacements = cap_hits = 0

    def close_position(s: int, row: int) -> None:
        sell_value[row] = value[s]
        odd_sell[row] = odd_slot[s]
        odd_slot[s] = False
        slot_fill[s] = 1.0
        pending[s] = False
        stop_pending[s] = False
        log_value[s] = 0.0
        run_max[s] = 1.0

    for j in range(intervals):
        budget = 0
        if rebalance[j]:
            rebalances += 1
            free_before = int(np.sum((slot < 0) & ~waiting))
            exits = [
                (rank[slot[s]], s) for s in range(k)
                if slot[s] >= 0 and rank[slot[s]] > keep_limit
            ]
            exits.sort(key=lambda item: -item[0])  # worst rank first
            if len(exits) > cap:
                cap_hits += 1
                exits = exits[:cap]
            pending[:] = False
            for _rank, s in exits:
                pending[s] = True
            replacements += len(exits)
            budget = free_before + len(exits)
        sold = set()
        for s in range(k):
            row = slot[s]
            if row < 0 or not (pending[s] or stop_pending[s]):
                continue
            if move.gap[row] <= down:
                blocked_sells += 1
                continue
            if stop_pending[s] and not pending[s]:
                stop_gap_exits += 1
                stop_rows[row] = True
            sell_open[j, s] = row
            sold.add(int(symbols[row]))
            periods.append(j - int(entry[s]))
            close_position(s, row)
            slot[s] = -1
        if rebalance[j] and budget > 0:
            taken = {int(symbols[row]) for row in slot if row >= 0} | sold
            candidates = (row for row in ranked[j] if int(symbols[row]) not in taken)
            buys = 0
            for s in range(k):
                if slot[s] >= 0 or waiting[s] or buys >= budget:
                    continue
                odd = False
                fill = 1.0
                while True:
                    row = next(candidates, None)
                    if row is None:
                        break
                    buy_attempts += 1
                    fill = 1.0
                    if blocked is not None and blocked[row]:
                        blocked_skipped += 1
                        continue
                    if slot_capital is not None and dollar_volume is not None \
                            and participation_cap is not None:
                        volume = dollar_volume[row]
                        allowed = participation_cap * volume if np.isfinite(volume) else np.nan
                        if not np.isfinite(allowed) or value[s] > allowed:
                            if cap_mode == "scale" and np.isfinite(allowed) \
                                    and allowed >= MIN_FILL * value[s]:
                                fill = allowed / value[s]
                            else:
                                liquidity_blocked += 1
                                continue
                    if lot_cost is not None and slot_capital is not None:
                        cost = lot_cost[row]
                        if not np.isfinite(cost) or cost > fill * value[s]:
                            if odd_lots and np.isfinite(cost) and cost > 0:
                                odd = True
                            else:
                                lot_blocked += 1
                                continue
                    break
                if row is None:
                    break
                if move.gap[row] >= up:
                    blocked_buys += 1
                    continue
                buy[j, s] = row
                slot[s] = row
                entry[s] = j
                buys += 1
                buy_value[row] = fill * value[s]
                odd_buy[row] = odd
                odd_slot[s] = odd
                odd_lot_buys += int(odd)
                slot_fill[s] = fill
                fill_row[row] = fill
                capped_buys += int(fill < 1.0)
        waiting[:] = False
        for s in range(k):
            row = slot[s]
            if row < 0:
                continue
            held[j, s] = row
            deployed[j] += slot_fill[s] / k
            if stop_fraction is not None:
                cum = math.exp(log_value[s])
                run_max[s] = max(run_max[s], cum)
                stop_level = stop_fraction * run_max[s]
                worst = cum * math.exp(adverse[row]) if np.isfinite(adverse[row]) else np.inf
                if worst <= stop_level and move.gap[row] > down:
                    discount = (stop_discount[row] if stop_discount is not None
                                and np.isfinite(stop_discount[row]) else 0.0)
                    discount += stop_overshoot * (stop_level - worst) / stop_level
                    fill_value = stop_level * (1.0 - discount)
                    growth[j, s] = _scaled_growth(math.log(fill_value / cum), slot_fill[s])
                    sell_close[j, s] = row
                    stop_rows[row] = True
                    stop_exits += 1
                    periods.append(j - int(entry[s]) + 1)
                    value[s] *= math.exp(growth[j, s])
                    close_position(s, row)
                    slot[s] = -1
                    pending[s] = False
                    continue
                if np.isfinite(favorable[row]):
                    run_max[s] = max(run_max[s], cum * math.exp(favorable[row]))
            following = link.next_row[row]
            if following >= 0 and np.isfinite(move.growth[row]):
                growth[j, s] = _scaled_growth(move.growth[row], slot_fill[s])
                slot[s] = following
                fill_row[following] = slot_fill[s]
                if stop_fraction is not None:
                    log_value[s] += move.growth[row]
                    if math.exp(log_value[s]) <= stop_fraction * run_max[s]:
                        stop_pending[s] = True
            else:
                fallback = move.fallback[row] if np.isfinite(move.fallback[row]) else 0.0
                growth[j, s] = _scaled_growth(fallback, slot_fill[s])
                sell_close[j, s] = row
                missing_exits += 1
                periods.append(j - int(entry[s]) + 1)
                value[s] *= math.exp(growth[j, s])
                close_position(s, row)
                slot[s] = -1
                waiting[s] = True
                continue
            value[s] *= math.exp(growth[j, s])
    for s in range(k):
        if slot[s] >= 0:
            periods.append(intervals - int(entry[s]))
            close_position(s, int(slot[s]))
    path = positions.Path(
        buy=buy, sell_open=sell_open, sell_close=sell_close, held=held, growth=growth,
        final=slot.copy(), holding_periods=np.array(periods, dtype=np.int64),
        blocked_buys=blocked_buys, blocked_sells=blocked_sells, missing_exits=missing_exits,
    )
    band = BandResult(
        path=path, lot_blocked=lot_blocked, liquidity_blocked=liquidity_blocked,
        odd_lot_buys=odd_lot_buys, buy_attempts=buy_attempts, buy_value=buy_value,
        sell_value=sell_value, odd_buy=odd_buy, odd_sell=odd_sell, fill_fraction=fill_row,
        capped_buys=capped_buys, deployed=deployed, blocked_skipped=blocked_skipped,
        stop_exits=stop_exits, stop_gap_exits=stop_gap_exits, stop_rows=stop_rows,
    )
    return WeeklyBand(band=band, rebalances=rebalances, replacements=replacements,
                      cap_hits=cap_hits)


def weekly_score(
    data: Any, config: dict[str, Any], *, h: int, score_variant: str
) -> np.ndarray:
    """The daily score S-A ranks on: the smoothed percentile of horizon ``h`` (v0), or that
    score orthogonalized inside the universe against the trailing returns and beta (v1)."""
    from twbacktest.exposure_features import FEATURES, orthogonalize_by_group

    inputs = data.inputs
    column = inputs.horizons.index(h)
    ranks = percentile_ranks(inputs.scores[:, column], data.groups, data.mask)
    signal = config["signal"]
    score = smooth_by_symbol(
        ranks, inputs.index.date_codes, inputs.index.symbol_codes,
        half_life=float(signal["half_life_sessions"]), max_gap=int(signal["max_gap_sessions"]),
    )
    if score_variant == "v1":
        if data.features is None:
            raise ValueError("score variant v1 needs the trailing features")
        regressors = np.column_stack([data.features[name] for name in FEATURES])
        return orthogonalize_by_group(score, regressors, data.groups)
    if score_variant != "v0":
        raise ValueError(f"Unknown score variant {score_variant!r}")
    return score


def weekly_long_book(
    data: Any,
    config: dict[str, Any],
    profile: Any,
    capital: Any,
    *,
    k: int,
    buffer: int,
    participation_cap: float,
    score_variant: str = "v1",
    cache: dict[tuple, Any] | None = None,
    odd_lots: bool = False,
) -> Any:
    """The S-A book of one market under one cost profile and capital scenario."""
    from twbacktest.scenario_suite import (
        Book,
        disposition_mask,
        execution_tags,
        fill_profile,
        load_profiles,
        stop_settings,
        with_impact,
    )

    weekly = config["weekly"]
    h = int(weekly["score_horizon"])
    turnover_cap = float(weekly["turnover_cap"])
    inputs = data.inputs
    fractional = capital.twd is None
    cap_mode = str(load_profiles(config).get("participation_mode", "skip"))
    limit = float(config["price_limit"]["open_gap"])
    cache = {} if cache is None else cache
    score_key = ("weekly-score", h, score_variant)
    if score_key not in cache:
        cache[score_key] = weekly_score(data, config, h=h, score_variant=score_variant)
    score = cache[score_key]
    blocked = disposition_mask(data, config)
    stop = stop_settings(data, config)
    key = ("weekly", capital.name, k, buffer, turnover_cap, limit, cap_mode,
           odd_lots and not fractional, h, score_variant, blocked is not None,
           stop.get("stop_fraction"), stop.get("stop_overshoot"))
    if key not in cache:
        cache[key] = simulate_weekly_band(
            engine.rankings(score, data.groups), k=k, buffer=buffer,
            turnover_cap=turnover_cap, rebalance=weekly_signal_flags(inputs.dates),
            symbols=inputs.index.symbol_codes, link=inputs.link, move=inputs.move,
            limit=limit, lot_cost=None if fractional else data.lot_cost,
            slot_capital=None if fractional else capital.twd,
            odd_lots=odd_lots and not fractional,
            dollar_volume=None if fractional else inputs.dollar_volume,
            participation_cap=None if fractional else participation_cap, cap_mode=cap_mode,
            blocked=blocked, **stop,
        )
    result: WeeklyBand = cache[key]
    band = result.band
    codes = inputs.index.market_codes
    free = np.zeros(inputs.index.count)
    gross = positions.replay(band.path, free, free).daily
    profile = fill_profile(profile, config, capital)
    net = positions.replay(
        band.path, *with_impact(scaled_rates(band, *profile.long_rates(codes)), band, data,
                                config)).daily
    statistics = positions.path_statistics(band.path)
    attempts = max(band.buy_attempts, 1)
    info = {
        "h": h, "k": k, "buffer": buffer, "score_variant": score_variant,
        "leg_matching": "none", "turnover_cap": turnover_cap,
        "turnover": statistics["turnover"],
        "weekly_turnover": (result.replacements / result.rebalances / k
                            if result.rebalances else float("nan")),
        "rebalances": result.rebalances, "replacements": result.replacements,
        "cap_hits": result.cap_hits,
        "holding_mean": statistics["holding_mean"],
        "cash_fraction": statistics["cash_fraction"],
        "limit_blocked": statistics["blocked_buys"] + statistics["blocked_sells"],
        "liquidity_blocked_share": band.liquidity_blocked / attempts,
        "capped_share": band.capped_buys / attempts,
        "deployed_share": float(band.deployed.mean()),
        "cap_mode": cap_mode,
        "disposition_skipped_share": band.blocked_skipped / attempts,
        "stop_exits": float(band.stop_exits), "stop_gap_exits": float(band.stop_gap_exits),
        **execution_tags(config),
    }
    return Book(net=net, gross=gross, info=info, legs={"long": net},
                gross_legs={"long": gross})


def select_weekly(
    data: Any, config: dict[str, Any], profile: Any, participation_cap: float,
    *, score_variant: str, capital: Any, summarize,
) -> tuple[list[dict[str, Any]], tuple[int, int], dict[str, Any]]:
    """Evaluate the S-A grid (K x m) on one market and pick by the selection metric.

    ``summarize`` is ``scenario_suite.summarize_book`` (passed in to avoid an import cycle);
    the rows carry ``stage = grid`` and ``scenario = wk_long``.
    """
    grid = config["weekly"]["grid"]
    rule = config["selection"]
    rows, candidates = [], []
    cache: dict[tuple, Any] = {}
    for k in grid["k"]:
        for m in grid["buffer_multiples"]:
            book = weekly_long_book(data, config, profile, capital, k=int(k), buffer=int(m),
                                    participation_cap=participation_cap,
                                    score_variant=score_variant, cache=cache)
            shared = {"model": None, "market": data.market, "profile": profile.name,
                      "slippage_multiplier": profile.slippage_multiplier,
                      "primary_slippage": True, "capital": capital.name, "capital_twd": None,
                      "stage": "grid", "score_variant": score_variant, "leg_matching": "none"}
            for row in summarize(book, SCENARIO, config, data.masks, data.index_total_return,
                                 shared, benchmarks=data.benchmarks):
                rows.append(row)
                if row["period"] == rule["period"]:
                    candidates.append(((int(k), int(m)), row[rule["metric"]]))
    finite = [(params, value) for params, value in candidates if np.isfinite(value)]
    if not finite:
        return rows, (int(grid["k"][0]), int(grid["buffer_multiples"][0])), {
            "reason": "no finite selection metric"}
    best = max(finite, key=lambda item: (item[1], -item[0][0], -item[0][1]))
    return rows, best[0], {"k": best[0][0], "buffer": best[0][1], "metric": rule["metric"],
                           "value": float(best[1])}
