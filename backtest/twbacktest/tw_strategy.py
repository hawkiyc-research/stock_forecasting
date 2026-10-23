"""Taiwan strategy layer (protocol 13.14): cost-aware band portfolio with a TAIEX futures overlay.

Walk-forward ranking scores (B0, D0, later D1) are turned into a long-only portfolio the way
the industry mitigates costs: a liquidity universe, an EMA-smoothed horizon score, a buy/hold
band (buy only into the top K, sell only when out of the top m x K) with a daily entry cap,
and per-position open-to-open accounting (the stage 1b engine). A TAIEX futures (TX) overlay
adds a hedged version (short TX against the stock value, with a margin reserve and margin
calls), a cash-equitized version (long TX on idle cash) and a TX-long reference. Parameters
are selected on TWSE P1 and the H-D decisions are read on P1+P2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, positions, runner, taifex
from twbacktest.costs import Scenario, primary_scenario, scenarios, side_costs
from twbacktest.data import MARKETS
from twbacktest.topk_dropout import period_masks
from twbacktest.walkforward_data import (
    WalkForwardInputs,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "tw_strategy.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FRACTIONAL = "fractional"
VERSIONS = ("long_only", "hedged", "cash_equitized")
TX_REFERENCE = "tx_long_reference"


# --------------------------------------------------------------------------- universe, signal


def universe_mask(
    inputs: WalkForwardInputs,
    market_codes: tuple[int, ...],
    *,
    min_dollar_volume: float,
    min_close: float,
    exclude_etf: bool,
) -> np.ndarray:
    """Rows that may be ranked on their signal date: market, liquidity, price and asset type."""
    mask = np.isin(inputs.index.market_codes, market_codes)
    if exclude_etf:
        mask &= ~inputs.index.is_etf
    if inputs.dollar_volume is not None:
        volume = inputs.dollar_volume
        mask &= np.isfinite(volume) & (volume >= min_dollar_volume)
    if inputs.raw_close is not None:
        close = inputs.raw_close
        mask &= np.isfinite(close) & (close >= min_close)
    return mask


def percentile_ranks(
    scores: np.ndarray, groups: tuple[np.ndarray, ...], mask: np.ndarray
) -> np.ndarray:
    """Within each date group, the masked rows' score percentile (1 = best); NaN elsewhere."""
    output = np.full(scores.shape[0], np.nan)
    for rows in groups:
        rows = rows[mask[rows] & np.isfinite(scores[rows])]
        if rows.size == 0:
            continue
        order = np.argsort(-scores[rows], kind="stable")
        place = np.empty(rows.size)
        place[order] = np.arange(rows.size)
        output[rows] = 1.0 - place / max(rows.size - 1, 1)
    return output


def smooth_by_symbol(
    values: np.ndarray,
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
    *,
    half_life: float,
    max_gap: int,
) -> np.ndarray:
    """EMA of each symbol's values along its signal dates; reset after a gap over max_gap.

    Rows with NaN values leave the symbol's state untouched and stay NaN.
    """
    if half_life <= 0:
        return values.copy()
    theta = 1.0 - 0.5 ** (1.0 / half_life)
    symbols = int(symbol_codes.max()) + 1 if symbol_codes.size else 0
    state = np.full(symbols, np.nan)
    last = np.full(symbols, -(10**9), dtype=np.int64)
    output = np.full(values.shape[0], np.nan)
    order = np.argsort(date_codes, kind="stable")
    bounds = np.searchsorted(date_codes[order], np.arange(int(date_codes.max()) + 2))
    for t in range(len(bounds) - 1):
        rows = order[bounds[t] : bounds[t + 1]]
        rows = rows[np.isfinite(values[rows])]
        if rows.size == 0:
            continue
        codes = symbol_codes[rows]
        fresh = np.isnan(state[codes]) | (t - last[codes] > max_gap)
        smoothed = np.where(
            fresh, values[rows], (1.0 - theta) * state[codes] + theta * values[rows]
        )
        state[codes] = smoothed
        last[codes] = t
        output[rows] = smoothed
    return output


# --------------------------------------------------------------------------- band simulation


@dataclass(frozen=True)
class CapitalScenario:
    """How a slot's cash turns into shares: fractional (no limits), or a capital amount."""

    name: str
    twd: float | None  # None = fractional slots
    odd_lots: bool  # a slot below one board lot buys odd lots instead of skipping the stock

    @property
    def fractional(self) -> bool:
        return self.twd is None


@dataclass(frozen=True)
class BandResult:
    path: positions.Path
    lot_blocked: int  # candidate buys skipped because one board lot cost more than the slot
    liquidity_blocked: int  # candidate buys skipped by the volume participation cap
    odd_lot_buys: int  # buys executed as odd lots
    buy_attempts: int
    buy_value: np.ndarray  # slot value at the buy, per row (NaN when the row was not bought)
    sell_value: np.ndarray  # slot value at the sale, per row (NaN when the row was not sold)
    odd_buy: np.ndarray  # row bought as odd lots
    odd_sell: np.ndarray  # row sold from an odd-lot position


def simulate_band(
    ranked: list[np.ndarray],
    *,
    k: int,
    buffer: int,
    max_buys: int,
    symbols: np.ndarray,
    link: positions.Links,
    move: positions.Moves,
    limit: float,
    lot_cost: np.ndarray | None = None,
    slot_capital: float | None = None,
    odd_lots: bool = False,
    dollar_volume: np.ndarray | None = None,
    participation_cap: float | None = None,
) -> BandResult:
    """The hold-buffer rule of positions.simulate with a daily entry cap and capital limits.

    ``ranked`` holds each signal date's rankable rows from best to worst (rows outside the
    universe are unranked and therefore sold). A slot keeps its stock while the rank stays
    within buffer x k; empty slots are filled best-first, at most ``max_buys`` per session.
    With ``lot_cost`` (price of one board lot at the execution open) and ``slot_capital``,
    a candidate whose lot costs more than the slot's current value is skipped and counted,
    unless ``odd_lots`` lets the slot buy odd lots (flagged for their own costs). With
    ``dollar_volume`` and ``participation_cap``, a candidate whose slot value exceeds the cap
    times its recent daily dollar volume is skipped and counted (a large-capital limit).
    """
    intervals = len(ranked) - 1
    if intervals < 1:
        raise ValueError("At least two signal dates are needed")
    if max_buys < 1:
        raise ValueError("max_buys must be at least 1")
    rank = np.full(symbols.size, np.inf)
    for rows in ranked:
        rank[rows] = np.arange(1, rows.size + 1)
    up, down = math.log1p(limit), math.log1p(-limit)
    keep_limit = buffer * k
    shape = (intervals, k)
    buy, sell_open, sell_close, held = (np.full(shape, -1, dtype=np.int64) for _ in range(4))
    growth = np.zeros(shape)
    slot = np.full(k, -1, dtype=np.int64)
    entry = np.zeros(k, dtype=np.int64)
    waiting = np.zeros(k, dtype=bool)
    value = np.full(k, (slot_capital or 0.0) / k)  # gross slot value, for capital limits
    odd_slot = np.zeros(k, dtype=bool)
    rows_total = symbols.size
    buy_value = np.full(rows_total, np.nan)
    sell_value = np.full(rows_total, np.nan)
    odd_buy = np.zeros(rows_total, dtype=bool)
    odd_sell = np.zeros(rows_total, dtype=bool)
    periods: list[int] = []
    blocked_buys = blocked_sells = missing_exits = lot_blocked = buy_attempts = 0
    liquidity_blocked = odd_lot_buys = 0

    def close_position(s: int, row: int) -> None:
        sell_value[row] = value[s]
        odd_sell[row] = odd_slot[s]
        odd_slot[s] = False

    for j in range(intervals):
        sold = set()
        for s in range(k):
            row = slot[s]
            if row < 0 or rank[row] <= keep_limit:
                continue
            if move.gap[row] <= down:
                blocked_sells += 1
                continue
            sell_open[j, s] = row
            sold.add(int(symbols[row]))
            periods.append(j - int(entry[s]))
            close_position(s, row)
            slot[s] = -1
        taken = {int(symbols[row]) for row in slot if row >= 0} | sold
        candidates = (row for row in ranked[j] if int(symbols[row]) not in taken)
        buys = 0
        for s in range(k):
            if slot[s] >= 0 or waiting[s] or buys >= max_buys:
                continue
            odd = False
            while True:
                row = next(candidates, None)
                if row is None:
                    break
                buy_attempts += 1
                if slot_capital is not None and dollar_volume is not None \
                        and participation_cap is not None:
                    volume = dollar_volume[row]
                    if not np.isfinite(volume) or value[s] > participation_cap * volume:
                        liquidity_blocked += 1
                        continue
                if lot_cost is not None and slot_capital is not None:
                    cost = lot_cost[row]
                    if not np.isfinite(cost) or cost > value[s]:
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
            buy_value[row] = value[s]
            odd_buy[row] = odd
            odd_slot[s] = odd
            odd_lot_buys += int(odd)
        waiting[:] = False
        for s in range(k):
            row = slot[s]
            if row < 0:
                continue
            held[j, s] = row
            following = link.next_row[row]
            if following >= 0 and np.isfinite(move.growth[row]):
                growth[j, s] = move.growth[row]
                slot[s] = following
            else:
                growth[j, s] = move.fallback[row] if np.isfinite(move.fallback[row]) else 0.0
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
        buy=buy,
        sell_open=sell_open,
        sell_close=sell_close,
        held=held,
        growth=growth,
        final=slot.copy(),
        holding_periods=np.array(periods, dtype=np.int64),
        blocked_buys=blocked_buys,
        blocked_sells=blocked_sells,
        missing_exits=missing_exits,
    )
    return BandResult(
        path=path, lot_blocked=lot_blocked, liquidity_blocked=liquidity_blocked,
        odd_lot_buys=odd_lot_buys, buy_attempts=buy_attempts, buy_value=buy_value,
        sell_value=sell_value, odd_buy=odd_buy, odd_sell=odd_sell,
    )


def capital_scenarios(config: dict[str, Any]) -> list[CapitalScenario]:
    """The fractional primary plus every configured capital scenario."""
    out = [CapitalScenario(FRACTIONAL, None, False)]
    for item in config["capital"].get("scenarios", []):
        out.append(CapitalScenario(str(item["name"]), float(item["twd"]), bool(item["odd_lots"])))
    return out


def rate_adjustments(
    result: BandResult,
    scenario: Scenario,
    config: dict[str, Any],
    *,
    capital: CapitalScenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row additions to the buy and sell cost rates for one capital scenario.

    A minimum commission per trade (NT$20, or the odd-lot minimum) tops up the discounted
    commission on small trades, and odd-lot trades pay extra slippage on both sides.
    """
    rows = result.buy_value.size
    buy_extra, sell_extra = np.zeros(rows), np.zeros(rows)
    if capital.fractional:
        return buy_extra, sell_extra
    settings = config["capital"]
    commission = float(config["costs"]["commission_rate"]) * scenario.discount
    minimum = float(settings["min_commission_twd"])
    odd = settings.get("odd_lot", {})
    odd_minimum = float(odd.get("min_commission_twd", minimum))
    odd_slippage = float(odd.get("extra_slippage", 0.0)) * scenario.slippage_multiplier
    for value, flags, extra in (
        (result.buy_value, result.odd_buy, buy_extra),
        (result.sell_value, result.odd_sell, sell_extra),
    ):
        traded = np.isfinite(value) & (value > 0)
        floor = np.where(flags, odd_minimum, minimum)
        with np.errstate(divide="ignore", invalid="ignore"):
            top_up = np.where(traded, np.maximum(floor / value - commission, 0.0), 0.0)
        extra += np.nan_to_num(top_up) + np.where(flags & traded, odd_slippage, 0.0)
    return buy_extra, sell_extra


# --------------------------------------------------------------------------- futures overlay


@dataclass(frozen=True)
class FuturesSeries:
    """Per interval j (open of signal date j+1 to open of j+2): the front-month TX position."""

    growth: np.ndarray  # simple return between the daily settlements that bracket the interval
    roll: np.ndarray  # True when the interval's opening execution also rolls the contract
    rate: np.ndarray  # cost per side as a fraction of the notional
    index_growth: np.ndarray  # the market index, close to close, over the same sessions
    missing: int  # intervals without a futures row (growth taken as 0)


def futures_series(
    front: pd.DataFrame,
    dates: pd.DatetimeIndex,
    index_close: pd.Series,
    *,
    multiplier: float,
    tax_rate: float,
    commission: float,
    slippage_points: float,
) -> FuturesSeries:
    """Align the continuous front month with the stock engine's intervals."""
    keyed = front.set_index("date")
    intervals = len(dates) - 1
    growth = np.zeros(intervals)
    roll = np.zeros(intervals, dtype=bool)
    rate = np.zeros(intervals)
    index_growth = np.full(intervals, np.nan)
    missing = 0
    stamps = [day.strftime("%Y-%m-%d") for day in dates]
    for j in range(intervals):
        start = stamps[j + 1]
        end = stamps[j + 2] if j + 2 < len(stamps) else None
        if start in keyed.index:
            settlement = float(keyed.at[start, "settlement"])
            notional = settlement * multiplier if settlement > 0 else float("nan")
            rate[j] = tax_rate + (
                (commission + slippage_points * multiplier) / notional if notional > 0 else 0.0
            )
            roll[j] = bool(keyed.at[start, "roll"])
        else:
            rate[j] = tax_rate
        if end is not None and end in keyed.index:
            value = keyed.at[end, "settlement_growth"]
            if np.isfinite(value):
                growth[j] = math.expm1(float(value))
            else:
                missing += 1
        else:
            missing += 1
        if end is not None and start in index_close.index and end in index_close.index:
            index_growth[j] = float(index_close[end] / index_close[start] - 1.0)
    return FuturesSeries(growth=growth, roll=roll, rate=rate, index_growth=index_growth,
                         missing=missing)


@dataclass(frozen=True)
class Overlay:
    net: np.ndarray
    gross: np.ndarray
    margin_calls: int
    reinvestments: int


def hedged_overlay(
    stock_net: np.ndarray,
    stock_gross: np.ndarray,
    futures: FuturesSeries,
    *,
    beta: float,
    reserve_fraction: float,
    call_fraction: float,
    reinvest_fraction: float,
    stock_buy_rate: float,
    stock_sell_rate: float,
) -> Overlay:
    """Short TX against the stock book, with a margin reserve, calls and reinvestment.

    Capital = stock book V + reserve R. At every opening execution the short notional is
    reset to beta x V and the reserve must hold reserve_fraction of it; the stock book earns
    the engine's return and the short earns -notional x growth. A reserve below
    call_fraction x notional is refilled from the stocks (one margin call, paid at the stock
    sell rate); a reserve above reinvest_fraction x notional returns its excess to the stocks
    (paid at the stock buy rate). Futures fees apply per side on notional changes and rolls.
    """

    def run(stock: np.ndarray, buy_rate: float, sell_rate: float, fee: np.ndarray) -> tuple:
        total = np.zeros(stock.size)
        value, reserve = 1.0 / (1.0 + beta * reserve_fraction), 0.0
        reserve = 1.0 - value
        notional_before = 0.0
        calls = reinvestments = 0
        for j in range(stock.size):
            notional = beta * value
            fees = abs(notional - notional_before) * fee[j]
            if futures.roll[j]:
                fees += 2.0 * notional * fee[j]
            capital = value + reserve
            stock_return = stock[j] if np.isfinite(stock[j]) else 0.0
            value *= 1.0 + stock_return
            reserve += -notional * futures.growth[j] - fees
            if reserve < call_fraction * notional:
                shortfall = reserve_fraction * beta * value - reserve
                shortfall = min(max(shortfall, 0.0), value)
                value -= shortfall
                reserve += shortfall * (1.0 - sell_rate)
                calls += 1
            elif reserve > reinvest_fraction * notional:
                excess = reserve - reserve_fraction * beta * value
                if excess > 0:
                    reserve -= excess
                    value += excess * (1.0 - buy_rate)
                    reinvestments += 1
            notional_before = notional
            total[j] = (value + reserve) / capital - 1.0
        return total, calls, reinvestments

    net, calls, reinvestments = run(stock_net, stock_buy_rate, stock_sell_rate, futures.rate)
    gross, _, _ = run(stock_gross, 0.0, 0.0, np.zeros_like(futures.rate))
    return Overlay(net=net, gross=gross, margin_calls=calls, reinvestments=reinvestments)


def cash_equitized_overlay(
    stock_net: np.ndarray,
    stock_gross: np.ndarray,
    cash_fraction: np.ndarray,
    futures: FuturesSeries,
) -> Overlay:
    """Long TX on the stock book's idle cash (empty slots, blocked or infeasible buys)."""
    previous = np.concatenate(([0.0], cash_fraction[:-1]))
    fees = futures.rate * (np.abs(cash_fraction - previous) + 2.0 * cash_fraction * futures.roll)
    exposure = cash_fraction * futures.growth
    return Overlay(net=stock_net + exposure - fees, gross=stock_gross + exposure,
                   margin_calls=0, reinvestments=0)


def cash_fractions(path: positions.Path, result: positions.Replay) -> np.ndarray:
    """Share of the book sitting in cash after each interval's opening trades."""
    cash = np.where(path.held < 0, result.invested, 0.0).sum(axis=1)
    book = result.invested.sum(axis=1)
    return np.where(book > 0, cash / book, 0.0)


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class MarketInputs:
    inputs: WalkForwardInputs
    market: str
    codes: tuple[int, ...]
    groups: tuple[np.ndarray, ...]  # rankable rows per signal date (liquid universe)
    mask: np.ndarray
    benchmark: np.ndarray  # equal-weight return of the liquid universe per interval
    index_interval: np.ndarray  # market index open-to-open per interval
    futures: FuturesSeries
    rates: dict[Scenario, tuple[np.ndarray, np.ndarray]]
    masks: dict[str, np.ndarray]
    lot_cost: np.ndarray


def universe_groups(inputs: WalkForwardInputs, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Masked rows of every signal date, aligned with inputs.dates (possibly empty)."""
    rows = np.flatnonzero(mask)
    codes = inputs.index.date_codes[rows]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(inputs.dates) + 1))
    return tuple(rows[order[bounds[i] : bounds[i + 1]]] for i in range(len(inputs.dates)))


def prepare_market(
    inputs: WalkForwardInputs,
    market: str,
    config: dict[str, Any],
    front: pd.DataFrame,
    index_close: pd.Series,
) -> MarketInputs:
    spec = config["markets"][market]
    codes = tuple(MARKETS.index(c) for c in spec["codes"])
    universe = config["universe"]
    mask = universe_mask(
        inputs,
        codes,
        min_dollar_volume=float(spec["min_dollar_volume"]),
        min_close=float(universe["min_close"]),
        exclude_etf=bool(universe["exclude_etf"]),
    )
    groups = universe_groups(inputs, mask)
    benchmark = positions.equal_weight(groups, inputs.move)
    all_groups = universe_groups(inputs, np.isin(inputs.index.market_codes, codes))
    index_interval = np.array(
        [
            float(np.expm1(inputs.move.index_growth[rows[0]])) if rows.size else np.nan
            for rows in all_groups[:-1]
        ]
    )
    futures_config = config["futures"]
    futures = futures_series(
        front,
        inputs.dates,
        index_close,
        multiplier=float(futures_config["multiplier"]),
        tax_rate=float(futures_config["tax_rate"]),
        commission=float(futures_config["commission_per_contract_twd"]),
        slippage_points=float(futures_config["slippage_points"]),
    )
    rates = {
        scenario: side_costs(
            config["costs"], scenario, inputs.index.market_codes, inputs.index.is_etf,
            same_day=False,
        )
        for scenario in scenarios(config["costs"])
    }
    lot = float(config["capital"]["board_lot"])
    raw_open = inputs.raw_open_1 if inputs.raw_open_1 is not None else np.full(mask.size, np.nan)
    return MarketInputs(
        inputs=inputs,
        market=market,
        codes=codes,
        groups=groups,
        mask=mask,
        benchmark=benchmark,
        index_interval=index_interval,
        futures=futures,
        rates=rates,
        masks=period_masks(inputs.dates[1:], config["periods"]),
        lot_cost=raw_open * lot,
    )


def _mean_rate(rates: tuple[np.ndarray, np.ndarray], rows: np.ndarray) -> tuple[float, float]:
    buy, sell = rates
    if rows.size == 0:
        return float(buy.mean()), float(sell.mean())
    return float(buy[rows].mean()), float(sell[rows].mean())


def evaluate_task(
    market_inputs: MarketInputs,
    config: dict[str, Any],
    *,
    label: str,
    h: int,
    k: int,
    buffer: int,
    capital: CapitalScenario,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every scenario, version and period of one (model, market, h, k, m, capital) path."""
    inputs = market_inputs.inputs
    column = inputs.horizons.index(h)
    ranks = percentile_ranks(inputs.scores[:, column], market_inputs.groups, market_inputs.mask)
    signal = config["signal"]
    smoothed = smooth_by_symbol(
        ranks,
        inputs.index.date_codes,
        inputs.index.symbol_codes,
        half_life=float(signal["half_life_sessions"]),
        max_gap=int(signal["max_gap_sessions"]),
    )
    ranked = engine.rankings(smoothed, market_inputs.groups)
    fractional = capital.fractional
    result = simulate_band(
        ranked,
        k=k,
        buffer=buffer,
        max_buys=max(1, math.ceil(k * float(config["entry_cap_fraction"]))),
        symbols=inputs.index.symbol_codes,
        link=inputs.link,
        move=inputs.move,
        limit=float(config["price_limit"]["open_gap"]),
        lot_cost=None if fractional else market_inputs.lot_cost,
        slot_capital=None if fractional else capital.twd,
        odd_lots=capital.odd_lots,
        dollar_volume=None if fractional else inputs.dollar_volume,
        participation_cap=None if fractional else config["capital"].get("participation_cap"),
    )
    path = result.path
    free = np.zeros(inputs.index.count)
    gross = positions.replay(path, free, free)
    cash = cash_fractions(path, gross)
    futures = market_inputs.futures
    overlay_config = config["futures"]
    market_rows = np.flatnonzero(np.isin(inputs.index.market_codes, market_inputs.codes))
    attempts = result.buy_attempts or 1
    shared = {
        "model": label, "market": market_inputs.market, "h": h, "k": k, "buffer": buffer,
        "capital": capital.name,
        "capital_twd": capital.twd,
        "odd_lots": capital.odd_lots,
        "lot_blocked": result.lot_blocked,
        "lot_blocked_share": result.lot_blocked / attempts,
        "liquidity_blocked": result.liquidity_blocked,
        "liquidity_blocked_share": result.liquidity_blocked / attempts,
        "odd_lot_buys": result.odd_lot_buys,
        "odd_lot_share": result.odd_lot_buys / attempts,
        "futures_missing_intervals": futures.missing,
        **positions.path_statistics(path),
    }
    rows, series = [], {}
    for scenario, (base_buy_rate, base_sell_rate) in market_inputs.rates.items():
        buy_extra, sell_extra = rate_adjustments(result, scenario, config, capital=capital)
        buy_rate, sell_rate = base_buy_rate + buy_extra, base_sell_rate + sell_extra
        net = positions.replay(path, buy_rate, sell_rate)
        buy_mean, sell_mean = _mean_rate((buy_rate, sell_rate), market_rows)
        hedged = hedged_overlay(
            net.daily, gross.daily, futures,
            beta=float(overlay_config["beta"]),
            reserve_fraction=float(overlay_config["reserve_fraction"]),
            call_fraction=float(overlay_config["call_fraction"]),
            reinvest_fraction=float(overlay_config["reinvest_fraction"]),
            stock_buy_rate=buy_mean,
            stock_sell_rate=sell_mean,
        )
        equitized = cash_equitized_overlay(net.daily, gross.daily, cash, futures)
        versions = {
            "long_only": (net.daily, gross.daily, market_inputs.benchmark, 0, 0),
            "hedged": (hedged.net, hedged.gross, np.zeros_like(net.daily),
                       hedged.margin_calls, hedged.reinvestments),
            "cash_equitized": (equitized.net, equitized.gross, market_inputs.benchmark, 0, 0),
        }
        is_primary = scenario == primary_scenario(config["costs"])
        for version in config["versions"]:
            net_series, gross_series, base, calls, reinvestments = versions[version]
            if is_primary:
                series[version] = (net_series, gross_series, base)
            for period, mask in market_inputs.masks.items():
                if not mask.any():
                    continue
                index_excess = net_series[mask] - market_inputs.index_interval[mask]
                rows.append(
                    {
                        **shared,
                        "version": version,
                        "discount": scenario.discount,
                        "slippage_multiplier": scenario.slippage_multiplier,
                        "period": period,
                        **positions.summarize(
                            net_series[mask], gross_series[mask], base[mask],
                            turnover=_turnover(path, mask),
                            annualization_days=int(config["annualization_days"]),
                            bootstrap=config["bootstrap"],
                        ),
                        "turnover": _turnover(path, mask),
                        "index_excess_annual": (
                            float(np.nanmean(index_excess)) * int(config["annualization_days"])
                            if version != "hedged" else float("nan")
                        ),
                        "margin_calls": calls,
                        "reinvestments": reinvestments,
                        "cash_fraction_mean": float(cash[mask].mean()),
                    }
                )
    return rows, series


def _turnover(path: positions.Path, mask: np.ndarray) -> float:
    buys = (path.buy >= 0).sum(axis=1)[mask]
    k = path.buy.shape[1]
    return float(buys[1:].mean() / k) if buys.size > 1 else float("nan")


def tx_reference_rows(
    market_inputs: MarketInputs, config: dict[str, Any], *, label: str
) -> list[dict[str, Any]]:
    """Holding the front-month TX long against the index, close to close, with roll costs."""
    futures = market_inputs.futures
    net = futures.growth - 2.0 * futures.rate * futures.roll
    rows = []
    for period, mask in market_inputs.masks.items():
        if not mask.any():
            continue
        rows.append(
            {
                "model": label, "market": market_inputs.market, "version": TX_REFERENCE,
                "period": period,
                **positions.summarize(
                    net[mask], futures.growth[mask], futures.index_growth[mask],
                    turnover=float("nan"),
                    annualization_days=int(config["annualization_days"]),
                    bootstrap=config["bootstrap"],
                ),
                "rolls": int(futures.roll[mask].sum()),
            }
        )
    return rows


# --------------------------------------------------------------------------- selection


def _primary_mask(grid: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    primary = primary_scenario(config["costs"])
    return (grid["discount"] == primary.discount) & (
        grid["slippage_multiplier"] == primary.slippage_multiplier
    )


def select_parameters(grid: pd.DataFrame, config: dict[str, Any], label: str) -> dict[str, Any]:
    rule = config["selection"]
    rows = grid[
        (grid["model"] == label)
        & (grid["market"] == rule["market"])
        & (grid["period"] == rule["period"])
        & (grid["version"] == rule["version"])
        & (grid["capital"] == FRACTIONAL)
        & _primary_mask(grid, config)
    ]
    rows = rows[np.isfinite(rows[rule["metric"]])]
    if rows.empty:
        return {"h": None, "k": None, "buffer": None, "reason": "no finite selection metric"}
    best = rows.sort_values([rule["metric"], "h", "k", "buffer"],
                            ascending=[False, True, True, True]).iloc[0]
    return {
        "h": int(best["h"]), "k": int(best["k"]), "buffer": int(best["buffer"]),
        "metric": rule["metric"], "value": float(best[rule["metric"]]),
        "market": rule["market"], "period": rule["period"],
    }


def decide(grid: pd.DataFrame, config: dict[str, Any], label: str, chosen: dict[str, Any]):
    rule = config["decision"]
    if chosen.get("h") is None:
        return {"hypothesis": rule["hypothesis"], "passed": None, "reason": "no parameters"}

    def pick(version: str) -> pd.Series | None:
        rows = grid[
            (grid["model"] == label)
            & (grid["market"] == rule["market"])
            & (grid["period"] == rule["period"])
            & (grid["version"] == version)
            & (grid["capital"] == FRACTIONAL)
            & (grid["h"] == chosen["h"])
            & (grid["k"] == chosen["k"])
            & (grid["buffer"] == chosen["buffer"])
            & _primary_mask(grid, config)
        ]
        return None if rows.empty else rows.iloc[0]

    def verdict(row: pd.Series | None, hypothesis: str) -> dict[str, Any]:
        if row is None:
            return {"hypothesis": hypothesis, "passed": None, "reason": "no decision row"}
        lower = float(row["net_excess_annual_low"])
        return {
            "hypothesis": hypothesis,
            "market": rule["market"],
            "period": rule["period"],
            "h": chosen["h"], "k": chosen["k"], "buffer": chosen["buffer"],
            "net_excess_annual": float(row["net_excess_annual"]),
            "net_excess_annual_low": lower,
            "net_excess_annual_high": float(row["net_excess_annual_high"]),
            "net_excess_ir": float(row["net_excess_ir"]),
            "net_return_annual": float(row["net_return_annual"]),
            "net_cagr": float(row["net_cagr"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
            "index_excess_annual": float(row["index_excess_annual"]),
            "turnover": float(row["turnover"]),
            "margin_calls": int(row["margin_calls"]),
            "passed": bool(np.isfinite(lower) and lower > float(rule["lower_bound_above"])),
        }

    return {
        rule["hypothesis"]: verdict(pick(rule["version"]), rule["hypothesis"]),
        rule["hedged_hypothesis"]: verdict(pick(rule["hedged_version"]),
                                           rule["hedged_hypothesis"]),
    }


def forward_candidate(decisions: dict[str, dict[str, Any]], config: dict[str, Any]) -> str | None:
    key = config["decision"]["hypothesis"]
    passing = [
        (model, verdict[key]["net_excess_ir"])
        for model, verdict in decisions.items()
        if verdict.get(key, {}).get("passed")
    ]
    if not passing:
        return None
    return max(passing, key=lambda item: item[1])[0]


# --------------------------------------------------------------------------- run


_SHARED: dict[str, Any] = {}


def _initialize(markets: dict[str, MarketInputs], config: dict[str, Any]) -> None:
    for name in runner.THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    _SHARED["markets"] = markets
    _SHARED["config"] = config


def _run_task(task: tuple[str, str, int, int, int, str]) -> list[dict[str, Any]]:
    label, market, h, k, buffer, capital_name = task
    capital = next(c for c in capital_scenarios(_SHARED["config"]) if c.name == capital_name)
    rows, _ = evaluate_task(
        _SHARED["markets"][(label, market)], _SHARED["config"],
        label=label, h=h, k=k, buffer=buffer, capital=capital,
    )
    return rows


def run_grid(
    markets: dict[tuple[str, str], MarketInputs], config: dict[str, Any], *, workers: int
) -> pd.DataFrame:
    grid = config["grid"]
    tasks = [
        (label, market, int(h), int(k), int(m), capital.name)
        for (label, market) in markets
        for h in grid["score_horizons"]
        for k in grid["k"]
        for m in grid["buffer_multiples"]
        for capital in capital_scenarios(config)
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        _initialize(markets, config)
        for task in tasks:
            rows.extend(_run_task(task))
    else:
        # Bounded spawn pool: each worker receives the prepared inputs once (initializer);
        # tasks are small tuples and results come back in task order.
        with get_context("spawn").Pool(
            processes=workers, initializer=_initialize, initargs=(markets, config)
        ) as pool:
            for result in pool.imap(_run_task, tasks, chunksize=1):
                rows.extend(result)
    for (label, _market), market_inputs in markets.items():
        rows.extend(tx_reference_rows(market_inputs, config, label=label))
    return pd.DataFrame(rows)


def _index_close(prices: pd.DataFrame, symbol: str) -> pd.Series:
    rows = prices[(prices["asset_type"] == "index") & (prices["symbol"] == symbol)]
    series = rows.set_index(rows["date"].dt.strftime("%Y-%m-%d"))["adj_close"]
    return series[~series.index.duplicated()]


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame) -> str:
    lines = ["# 台股策略層回測摘要 / Taiwan strategy-layer backtest summary", ""]
    rows_zh, rows_en = [], []
    for model, chosen in report["selection"].items():
        if chosen.get("h") is None:
            continue
        base = grid[
            (grid["model"] == model) & (grid["capital"] == FRACTIONAL)
            & (grid["h"] == chosen["h"]) & (grid["k"] == chosen["k"])
            & (grid["buffer"] == chosen["buffer"]) & (grid["discount"] == 0.6)
            & (grid["slippage_multiplier"] == 1)
        ]
        for row in base.sort_values(["market", "version", "period"]).itertuples():
            cells = [
                row.model, row.market, row.version, row.period,
                runner.percent(row.net_return_annual), runner.number(row.sharpe),
                runner.percent(row.max_drawdown), runner.percent(row.net_excess_annual),
                f"[{runner.percent(row.net_excess_annual_low)}, "
                f"{runner.percent(row.net_excess_annual_high)}]",
                runner.number(row.net_excess_ir), runner.percent(row.index_excess_annual),
                runner.percent(row.turnover), str(int(row.margin_calls)),
            ]
            line = "| " + " | ".join(cells) + " |"
            rows_zh.append(line)
            rows_en.append(line)
    header_zh = (
        "| 模型 | 市場 | 版本 | 期間 | 年化淨報酬 | Sharpe | 最大回撤 | 年化淨超額 | 95% 區間 | IR "
        "| 對指數超額 | 換手 | 追繳 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    header_en = (
        "| Model | Market | Version | Period | Net return (annual) | Sharpe | Max drawdown "
        "| Net excess (annual) | 95% CI | IR | Excess vs index | Turnover | Margin calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    capital_rows = []
    for model, chosen in report["selection"].items():
        if chosen.get("h") is None:
            continue
        base = grid[
            (grid["model"] == model) & (grid["market"] == "TWSE") & (grid["version"] == "long_only")
            & (grid["period"] == "P1+P2") & (grid["h"] == chosen["h"]) & (grid["k"] == chosen["k"])
            & (grid["buffer"] == chosen["buffer"]) & (grid["discount"] == 0.6)
            & (grid["slippage_multiplier"] == 1)
        ]
        for row in base.itertuples():
            capital_rows.append(
                f"| {row.model} | {row.capital} | {runner.percent(row.net_return_annual)} | "
                f"{runner.number(row.sharpe)} | {runner.percent(row.net_excess_annual)} | "
                f"{runner.percent(row.index_excess_annual)} | "
                f"{runner.percent(row.lot_blocked_share)} | "
                f"{runner.percent(row.liquidity_blocked_share)} | "
                f"{runner.percent(row.odd_lot_share)} | "
                f"{runner.percent(row.turnover)} |"
            )
    capital_header_zh = (
        "| 模型 | 資本情境 | 年化淨報酬 | Sharpe | 年化淨超額 | 對指數超額 "
        "| 整張不足被擋 | 參與率被擋 | 零股買進比例 | 換手 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    capital_header_en = (
        "| Model | Capital scenario | Net return (annual) | Sharpe | Net excess (annual) "
        "| Excess vs index | Lot-blocked | Participation-blocked | Odd-lot buys | Turnover |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    selection_zh = "；".join(
        f"{model}：h = {c.get('h')}、K = {c.get('k')}、m = {c.get('buffer')}"
        for model, c in report["selection"].items()
    )
    selection_en = "; ".join(
        f"{model}: h = {c.get('h')}, K = {c.get('k')}, m = {c.get('buffer')}"
        for model, c in report["selection"].items()
    )
    decisions_zh = "；".join(
        f"{model} {name}：{'通過' if v.get('passed') else '未通過'}"
        for model, verdicts in report["decisions"].items()
        for name, v in verdicts.items()
    )
    decisions_en = "; ".join(
        f"{model} {name}: {'passed' if v.get('passed') else 'not passed'}"
        for model, verdicts in report["decisions"].items()
        for name, v in verdicts.items()
    )
    lines += [
        "## 中文", "",
        f"選定參數（TWSE P1 多頭版淨超額 IR 最高）：{selection_zh}。", "",
        f"判定：{decisions_zh}。前瞻候選模型：{report['forward_candidate'] or '無'}。", "",
        *header_zh, *rows_zh, "",
        "### 資本情境（TWSE 多頭版，P1+P2，主要成本）", "", *capital_header_zh, *capital_rows, "",
        "## English", "",
        f"Selected parameters (highest TWSE P1 long-only net excess IR): {selection_en}.", "",
        f"Decisions: {decisions_en}. Forward candidate: {report['forward_candidate'] or 'none'}.",
        "", *header_en, *rows_en, "",
        "### Capital scenarios (TWSE long-only, P1+P2, primary costs)", "",
        *capital_header_en, *capital_rows, "",
    ]
    return "\n".join(lines)


def run(
    config_path: Path,
    predictions: list[Path],
    models: list[str],
    prices: Path,
    output_root: Path,
    *,
    artifacts: Path,
    workers: int | None,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    price_frame = load_prices(
        prices, liquidity_lookback=int(config["universe"]["liquidity_lookback_sessions"])
    )
    futures_root = artifacts / config["futures"]["root"]
    contract = config["futures"]["contract"]
    front = taifex.continuous_front_month(taifex.load_contract(futures_root, contract), contract)
    markets: dict[tuple[str, str], MarketInputs] = {}
    for path, label in zip(predictions, models, strict=True):
        inputs = build_inputs(load_predictions(path, label), price_frame)
        for market, spec in config["markets"].items():
            markets[(label, market)] = prepare_market(
                inputs, market, config, front, _index_close(price_frame, spec["index"])
            )
    tasks = len(markets) * len(config["grid"]["score_horizons"]) * len(config["grid"]["k"]) * len(
        config["grid"]["buffer_multiples"]) * len(capital_scenarios(config))
    workers = workers or runner.default_workers(tasks)
    grid = run_grid(markets, config, workers=workers)
    selection = {label: select_parameters(grid, config, label) for label in models}
    decisions = {label: decide(grid, config, label, selection[label]) for label in models}
    candidate = forward_candidate(decisions, config)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    daily_frames = []
    for label in models:
        chosen = selection[label]
        if chosen.get("h") is None:
            continue
        for market in config["markets"]:
            market_inputs = markets[(label, market)]
            _, series = evaluate_task(
                market_inputs, config, label=label, h=chosen["h"], k=chosen["k"],
                buffer=chosen["buffer"], capital=CapitalScenario(FRACTIONAL, None, False),
            )
            dates = market_inputs.inputs.dates[1:]
            for version, (net, gross, base) in series.items():
                daily_frames.append(pd.DataFrame({
                    "model": label, "market": market, "version": version,
                    "interval_start": dates.strftime("%Y-%m-%d"), "net": net, "gross": gross,
                    "benchmark": base, "index_open_to_open": market_inputs.index_interval,
                    "tx_growth": market_inputs.futures.growth,
                }))
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_csv(
            output_root / "daily_series.csv", index=False
        )
    report = {
        "config": str(config_path),
        "predictions": {label: str(path) for path, label in zip(predictions, models, strict=True)},
        "prices": str(prices),
        "futures_root": str(futures_root),
        "kind": "tw-strategy-band-tx-overlay",
        "workers": workers,
        "selection": selection,
        "decisions": decisions,
        "forward_candidate": candidate,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, grid), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, action="append", required=True,
                        help="walk-forward predictions.parquet (repeat per model)")
    parser.add_argument("--model", action="append", required=True,
                        help="model name inside the matching --predictions file")
    parser.add_argument("--prices", type=Path, required=True, help="raw market.parquet")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, help="process pool size (default: sized)")
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root, "tw-strategy")
    report = run(
        args.config, args.predictions, args.model, args.prices, output_root,
        artifacts=args.artifacts_root, workers=args.workers,
    )
    print(json.dumps(
        {"output_root": str(output_root), "selection": report["selection"],
         "decisions": report["decisions"], "forward_candidate": report["forward_candidate"]},
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
