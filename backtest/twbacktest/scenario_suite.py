"""Six trading scenarios under retail and institution costs (protocol 13.19).

The walk-forward ranking scores of one or more models are turned into six market-neutral
constructions on the liquidity universe of each market:

* S1 ``md_ls``: multi-day long-short. The long book is the protocol 13.14 band portfolio (buy
  into the top K, sell out of the top m x K, daily entry cap); the short book runs the same
  band rule on the reversed ranking, each short slot reset to its slot value every session
  (return = minus the stock's open-to-open move), with the profile's borrow fee per held day.
  Both books carry the full capital, so the construction is dollar-neutral with gross 2x.
* S2 ``md_ls_tx``: S1 plus a TAIEX futures (TX) overlay sized by the strategy's own trailing
  rolling beta (strictly before the interval), marked settlement to settlement.
* S3 ``dt_ls``: next-session day trade. The h = 1 scores rank the universe after the close of
  t; the top q are bought and the bottom q sold short at the open of t+1 and everything is
  closed at that session's close (day-trade transaction tax).
* S4 ``dt_ls_tx``: S3 plus a TX intraday overlay (open to settlement) sized by rolling beta.
* S5 ``combo`` / S6 ``combo_tx``: half the capital in the multi-day book and half in the day
  trade book, without / with the TX overlays.

Every series is evaluated under every cost profile of ``cost_profiles.json`` (retail and
institution, with slippage sensitivities), per market and for a 50/50 combination of the two
markets. Parameters are selected once (institution costs, TWSE, P1, net Sharpe) and fixed for
every other table. The whole suite is a few hundred small per-interval simulations, so it runs
serially on purpose (process pools would cost more in start-up than they save).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, positions, runner, taifex
from twbacktest.data import MARKETS
from twbacktest.exposure_features import (
    FEATURES,
    orthogonalize_by_group,
    quantile_labels,
    row_features,
    trailing_features,
)
from twbacktest.topk_dropout import period_masks
from twbacktest.tw_strategy import (
    FuturesSeries,
    futures_series,
    percentile_ranks,
    simulate_band,
    smooth_by_symbol,
    universe_groups,
    universe_mask,
)
from twbacktest.walkforward_data import (
    WalkForwardInputs,
    build_inputs,
    load_predictions,
    load_prices,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "scenario_suite.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FRACTIONAL = "fractional"
SCENARIOS = ("md_ls", "md_ls_tx", "dt_ls", "dt_ls_tx", "combo", "combo_tx")
SCENARIO_LABELS = {
    "md_ls": ("S1 短至中天期台股多空", "S1 multi-day stock long-short"),
    "md_ls_tx": ("S2 S1 + 台指期", "S2 S1 + TX overlay"),
    "dt_ls": ("S3 當沖台股多空", "S3 day-trade stock long-short"),
    "dt_ls_tx": ("S4 S3 + 台指期", "S4 S3 + TX overlay"),
    "combo": ("S5 當沖 + 多日", "S5 day trade + multi-day"),
    "combo_tx": ("S6 S5 + 台指期", "S6 S5 + TX overlays"),
}
MULTI_DAY = {"md_ls", "md_ls_tx", "combo", "combo_tx"}
DAY_TRADE = {"dt_ls", "dt_ls_tx", "combo", "combo_tx"}
SHORT_FLOOR = 1e-3  # a short slot cannot lose more than its value in one interval


# --------------------------------------------------------------------------- cost profiles


@dataclass(frozen=True)
class CostProfile:
    """One cost profile (retail or institution) at one slippage multiplier."""

    name: str
    commission: float
    tax: float
    day_trade_tax: float
    slippage: tuple[float, float]  # per market code (TWSE, TPEX), already multiplied
    slippage_multiplier: float
    short_entry_fee: float  # paid on the short sale of a multi-day short (融券手續費)
    borrow_fee_annual: float  # 借券 fee on the short book, charged per held session
    tx_tax: float
    tx_commission: float  # NT$ per contract per side
    tx_slippage_points: float

    @property
    def key(self) -> str:
        return f"{self.name}/slippage x{self.slippage_multiplier:g}"

    def _slip(self, market_codes: np.ndarray) -> np.ndarray:
        return np.asarray(self.slippage, dtype=np.float64)[market_codes]

    def long_rates(self, market_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(buy, sell) rates of a multi-day long position."""
        slip = self._slip(market_codes)
        return self.commission + slip, self.commission + self.tax + slip

    def short_rates(self, market_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(entry = short sale, exit = buy to cover) rates of a multi-day short position."""
        slip = self._slip(market_codes)
        return (self.commission + self.tax + slip + self.short_entry_fee, self.commission + slip)

    def day_trade_round_trip(self, market_codes: np.ndarray) -> np.ndarray:
        """Round-trip cost of a same-session position (either direction): two commissions,
        two slippages and the day-trade tax on the sale."""
        slip = self._slip(market_codes)
        return 2.0 * (self.commission + slip) + self.day_trade_tax

    @property
    def borrow_fee_daily(self) -> float:
        return self.borrow_fee_annual / 252.0


def load_profiles(config: dict[str, Any]) -> dict[str, Any]:
    path = config["cost_profiles"]
    path = Path(path) if Path(path).is_absolute() else CONFIG_DIR / path
    return json.loads(path.read_text(encoding="utf-8"))


def cost_profiles(profiles: dict[str, Any]) -> list[CostProfile]:
    out = []
    for name, spec in profiles["profiles"].items():
        for multiplier in profiles["slippage_multipliers"]:
            slippage = tuple(
                float(spec["slippage_per_side"][market]) * float(multiplier) for market in MARKETS
            )
            out.append(
                CostProfile(
                    name=name,
                    commission=float(spec["commission_rate"]),
                    tax=float(spec["transaction_tax"]),
                    day_trade_tax=float(spec["day_trade_tax"]),
                    slippage=slippage,  # type: ignore[arg-type]
                    slippage_multiplier=float(multiplier),
                    short_entry_fee=float(spec["short_entry_fee"]),
                    borrow_fee_annual=float(spec["borrow_fee_annual"]),
                    tx_tax=float(spec["tx"]["tax_rate"]),
                    tx_commission=float(spec["tx"]["commission_per_contract_twd"]),
                    tx_slippage_points=float(spec["tx"]["slippage_points"]),
                )
            )
    return out


def is_primary(profile: CostProfile, profiles: dict[str, Any]) -> bool:
    return profile.slippage_multiplier == float(profiles["primary_slippage_multiplier"])


@dataclass(frozen=True)
class Capital:
    name: str
    twd: float | None  # None = fractional slots, no capacity limit
    profile: str | None  # the profile this scenario belongs to (None = every profile)


def capital_scenarios(profiles: dict[str, Any]) -> list[Capital]:
    out = [Capital(FRACTIONAL, None, None)]
    for name, spec in profiles["profiles"].items():
        for item in spec.get("capital_scenarios", []):
            out.append(Capital(str(item["name"]), float(item["twd"]), name))
    return out


# --------------------------------------------------------------------------- short positions


def short_moves(move: positions.Moves) -> positions.Moves:
    """Moves of a short slot: value factor 2 - exp(g) (floored), price-limit gaps reversed.

    ``simulate_band`` blocks a buy when the gap is at the limit-up and a sale at the limit-down;
    reversing the gap makes it block a short sale at the limit-down open and a cover at the
    limit-up open.
    """

    def flip(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan)
        ok = np.isfinite(values)
        out[ok] = np.log(np.maximum(2.0 - np.exp(values[ok]), SHORT_FLOOR))
        return out

    return positions.Moves(
        growth=flip(move.growth), fallback=flip(move.fallback), gap=-move.gap,
        index_growth=move.index_growth,
    )


def with_borrow_fee(path: positions.Path, daily_fee: float) -> positions.Path:
    """Charge the borrow fee on every held short slot, per interval."""
    if daily_fee <= 0:
        return path
    growth = path.growth + np.where(path.held >= 0, math.log1p(-daily_fee), 0.0)
    return replace(path, growth=growth)


# --------------------------------------------------------------------------- market inputs


@dataclass(frozen=True)
class MarketData:
    """Everything one market's scenarios need, aligned with the walk-forward inputs."""

    market: str
    inputs: WalkForwardInputs
    mask: np.ndarray  # rows in the liquidity universe
    groups: tuple[np.ndarray, ...]  # universe rows per signal date
    index_interval: np.ndarray  # market index open-to-open return per interval
    index_intraday: np.ndarray  # market index open-to-close return on each interval's session
    index_total_return: np.ndarray  # index close-to-close (dividends included) per interval
    futures: FuturesSeries  # TX settlement-to-settlement per interval (with per-side rates)
    tx_intraday: np.ndarray  # TX open-to-settlement return on each interval's session
    tx_notional_per_contract: np.ndarray  # settlement x multiplier on each interval's session
    intraday: np.ndarray  # per row: log(close / open) of the session after the signal date
    lot_cost: np.ndarray  # one board lot at the execution open, per row
    masks: dict[str, np.ndarray]
    features: dict[str, np.ndarray] | None = None  # trailing returns, beta, turnover spike


def _index_series(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rows = prices[(prices["asset_type"] == "index") & (prices["symbol"] == symbol)]
    rows = rows.drop_duplicates("date").set_index("date").sort_index()
    return rows


def _interval_from_dates(
    series: pd.Series, dates: pd.DatetimeIndex, offset: int
) -> np.ndarray:
    """Value of ``series`` on dates[j + offset] for every interval j (NaN when absent)."""
    intervals = len(dates) - 1
    out = np.full(intervals, np.nan)
    for j in range(intervals):
        day = dates[j + offset]
        if day in series.index:
            value = series[day]
            out[j] = float(value) if np.isfinite(value) else np.nan
    return out


def prepare_market(
    inputs: WalkForwardInputs,
    intraday: np.ndarray,
    market: str,
    config: dict[str, Any],
    prices: pd.DataFrame,
    front: pd.DataFrame,
    features: dict[str, np.ndarray] | None = None,
) -> MarketData:
    spec = config["markets"][market]
    codes = tuple(MARKETS.index(c) for c in spec["codes"])
    universe = config["universe"]
    mask = universe_mask(
        inputs, codes, min_dollar_volume=float(spec["min_dollar_volume"]),
        min_close=float(universe["min_close"]), exclude_etf=bool(universe["exclude_etf"]),
    )
    groups = universe_groups(inputs, mask)
    all_groups = universe_groups(inputs, np.isin(inputs.index.market_codes, codes))
    index_interval = np.array(
        [
            float(np.expm1(inputs.move.index_growth[rows[0]])) if rows.size else np.nan
            for rows in all_groups[:-1]
        ]
    )
    index_rows = _index_series(prices, spec["index"])
    with np.errstate(divide="ignore", invalid="ignore"):
        intraday_index = np.log(index_rows["close_1"] / index_rows["open_1"])
        total_return = np.log(index_rows["close_1"] / index_rows["adj_close"])
    dates = inputs.dates
    index_intraday = np.expm1(_interval_from_dates(intraday_index, dates, 0))
    index_total = np.expm1(_interval_from_dates(total_return, dates, 1))
    tx_cfg = config["tx"]
    multiplier = float(tx_cfg["multiplier"])
    index_close = index_rows["adj_close"].copy()
    index_close.index = index_close.index.strftime("%Y-%m-%d")
    futures = futures_series(
        front, dates, index_close, multiplier=multiplier, tax_rate=0.0, commission=0.0,
        slippage_points=0.0,
    )
    keyed = front.set_index("date")
    stamps = dates.strftime("%Y-%m-%d")
    intervals = len(dates) - 1
    tx_intraday = np.full(intervals, np.nan)
    notional = np.full(intervals, np.nan)
    for j in range(intervals):
        day = stamps[j + 1]
        if day in keyed.index:
            open_, settle = float(keyed.at[day, "open"]), float(keyed.at[day, "settlement"])
            if open_ > 0 and settle > 0:
                tx_intraday[j] = settle / open_ - 1.0
                notional[j] = settle * multiplier
    lot = float(load_profiles(config)["board_lot"])
    raw_open = inputs.raw_open_1 if inputs.raw_open_1 is not None else np.full(mask.size, np.nan)
    return MarketData(
        market=market, inputs=inputs, mask=mask, groups=groups, index_interval=index_interval,
        index_intraday=index_intraday, index_total_return=index_total, futures=futures,
        tx_intraday=tx_intraday, tx_notional_per_contract=notional, intraday=intraday,
        lot_cost=raw_open * lot, masks=period_masks(dates[1:], config["periods"]),
        features=features,
    )


def intraday_returns(predictions: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    """Per prediction row: log(close / open) of the session after the signal date."""
    stocks = prices[prices["asset_type"] != "index"][["symbol", "date", "open_1", "close_1"]]
    merged = predictions[["symbol", "signal_date"]].merge(
        stocks, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left",
        validate="one_to_one",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(merged["close_1"].to_numpy(dtype=np.float64)
                      / merged["open_1"].to_numpy(dtype=np.float64))


def tx_side_rate(data: MarketData, profile: CostProfile, multiplier: float) -> np.ndarray:
    """TX cost per side as a fraction of notional, per interval (NaN notional -> tax only)."""
    notional = data.tx_notional_per_contract
    extra = np.where(
        np.isfinite(notional) & (notional > 0),
        (profile.tx_commission + profile.tx_slippage_points * multiplier) / notional, 0.0,
    )
    return profile.tx_tax + extra


# --------------------------------------------------------------------------- books


@dataclass(frozen=True)
class Book:
    """Daily gross and net returns of one construction on one market, plus its statistics.

    ``legs`` holds the net daily series of the parts (long book, short book, TX overlay), so the
    summaries can report each part's annualized return per period.
    """

    net: np.ndarray
    gross: np.ndarray
    info: dict[str, Any]
    legs: dict[str, np.ndarray] | None = None


def rolling_beta(returns: np.ndarray, market: np.ndarray, *, window: int, min_obs: int):
    """Ex-ante beta of ``returns`` on ``market`` from the trailing window (strictly before j)."""
    beta = np.zeros(returns.size)
    for j in range(returns.size):
        lo = max(0, j - window)
        r, m = returns[lo:j], market[lo:j]
        ok = np.isfinite(r) & np.isfinite(m)
        if ok.sum() < min_obs:
            continue
        r, m = r[ok], m[ok]
        var = m.var()
        if var > 0:
            beta[j] = float(np.cov(r, m, ddof=0)[0, 1] / var)
    return beta


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def momentum_labels(data: MarketData, quintiles: int) -> np.ndarray:
    """Per row: the quintile (0..quintiles-1) of the trailing 20-session return inside the row's
    date group; -1 where the feature is missing or the row is outside the universe."""
    if data.features is None:
        raise ValueError("momentum-matched legs need the trailing features")
    ret20 = data.features["ret20"]
    labels = np.full(data.inputs.index.count, -1, dtype=np.int64)
    for rows in data.groups:
        finite = rows[np.isfinite(ret20[rows])]
        if finite.size:
            labels[finite] = quantile_labels(ret20[finite], quintiles)
    return labels


def _band_long_short(
    data: MarketData,
    score: np.ndarray,
    groups: tuple[np.ndarray, ...],
    profile: CostProfile,
    capital: Capital,
    *,
    k: int,
    buffer: int,
    max_buys: int,
    limit: float,
    participation_cap: float,
) -> dict[str, Any]:
    """One long band book and one short band book on ``groups`` ranked by ``score``."""
    inputs = data.inputs
    fractional = capital.twd is None
    common = {
        "k": k, "buffer": buffer, "max_buys": max_buys, "symbols": inputs.index.symbol_codes,
        "link": inputs.link, "limit": limit,
        "lot_cost": None if fractional else data.lot_cost,
        "slot_capital": None if fractional else capital.twd,
        "dollar_volume": None if fractional else inputs.dollar_volume,
        "participation_cap": None if fractional else participation_cap,
    }
    long_result = simulate_band(engine.rankings(score, groups), move=inputs.move, **common)
    short_result = simulate_band(
        engine.rankings(-score, groups), move=short_moves(inputs.move), **common
    )
    codes = inputs.index.market_codes
    free = np.zeros(inputs.index.count)
    short_path = with_borrow_fee(short_result.path, profile.borrow_fee_daily)
    long_stats = positions.path_statistics(long_result.path)
    short_stats = positions.path_statistics(short_result.path)
    return {
        "long_gross": positions.replay(long_result.path, free, free).daily,
        "long_net": positions.replay(long_result.path, *profile.long_rates(codes)).daily,
        "short_gross": positions.replay(short_result.path, free, free).daily,
        "short_net": positions.replay(short_path, *profile.short_rates(codes)).daily,
        "turnover": 0.5 * (long_stats["turnover"] + short_stats["turnover"]),
        "holding_mean": 0.5 * (long_stats["holding_mean"] + short_stats["holding_mean"]),
        "cash_fraction": 0.5 * (long_stats["cash_fraction"] + short_stats["cash_fraction"]),
        "limit_blocked": (long_stats["blocked_buys"] + long_stats["blocked_sells"]
                          + short_stats["blocked_buys"] + short_stats["blocked_sells"]),
        "liquidity_blocked": long_result.liquidity_blocked + short_result.liquidity_blocked,
        "buy_attempts": long_result.buy_attempts + short_result.buy_attempts,
    }


def multi_day_book(
    data: MarketData,
    config: dict[str, Any],
    profile: CostProfile,
    capital: Capital,
    *,
    h: int,
    k: int,
    buffer: int,
    participation_cap: float,
    score_variant: str = "v0",
    leg_matching: str = "none",
) -> Book:
    """The multi-day long-short book.

    ``score_variant`` ``v0`` is the smoothed percentile score; ``v1`` orthogonalizes it every day
    inside the universe against the trailing 20/60-session returns and the 60-session beta.
    ``leg_matching`` ``none`` runs one long and one short band on the whole universe;
    ``momentum`` runs one pair per quintile of the trailing 20-session return with k / quintiles
    slots each (equal capital per quintile), so both legs share the momentum distribution.
    """
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
        score = orthogonalize_by_group(score, regressors, data.groups)
    elif score_variant != "v0":
        raise ValueError(f"Unknown score variant {score_variant!r}")
    cap_fraction = float(config["multi_day"]["entry_cap_fraction"])
    limit = float(config["price_limit"]["open_gap"])
    if leg_matching == "none":
        parts = [_band_long_short(
            data, score, data.groups, profile, capital, k=k, buffer=buffer,
            max_buys=max(1, math.ceil(k * cap_fraction)), limit=limit,
            participation_cap=participation_cap,
        )]
    elif leg_matching == "momentum":
        quintiles = int(config.get("construction", {}).get("momentum_quintiles", 5))
        labels = momentum_labels(data, quintiles)
        k_sub = max(1, k // quintiles)
        parts = []
        for q in range(quintiles):
            groups_q = tuple(rows[labels[rows] == q] for rows in data.groups)
            parts.append(_band_long_short(
                data, score, groups_q, profile, capital, k=k_sub, buffer=buffer,
                max_buys=max(1, math.ceil(k_sub * cap_fraction)), limit=limit,
                participation_cap=participation_cap,
            ))
    else:
        raise ValueError(f"Unknown leg matching {leg_matching!r}")
    n = len(parts)

    def mean_series(key: str) -> np.ndarray:
        return sum(part[key] for part in parts) / n

    long_net, long_gross = mean_series("long_net"), mean_series("long_gross")
    short_net, short_gross = mean_series("short_net"), mean_series("short_gross")
    attempts = max(sum(part["buy_attempts"] for part in parts), 1)
    info = {
        "h": h, "k": k, "buffer": buffer, "score_variant": score_variant,
        "leg_matching": leg_matching,
        "turnover": float(np.mean([part["turnover"] for part in parts])),
        "holding_mean": _finite_mean([part["holding_mean"] for part in parts]),
        "cash_fraction": float(np.mean([part["cash_fraction"] for part in parts])),
        "limit_blocked": int(sum(part["limit_blocked"] for part in parts)),
        "liquidity_blocked_share": sum(part["liquidity_blocked"] for part in parts) / attempts,
    }
    return Book(net=long_net + short_net, gross=long_gross + short_gross, info=info,
                legs={"long": long_net, "short": short_net})


def day_trade_book(
    data: MarketData,
    config: dict[str, Any],
    profile: CostProfile,
    capital: Capital,
    *,
    fraction: float,
    participation_cap: float,
) -> Book:
    inputs = data.inputs
    column = inputs.horizons.index(int(config["day_trade"]["horizon"]))
    scores = inputs.scores[:, column]
    limit = float(config["price_limit"]["open_gap"])
    up, down = math.log1p(limit), math.log1p(-limit)
    round_trip = profile.day_trade_round_trip(inputs.index.market_codes)
    intervals = len(inputs.dates) - 1
    net = np.zeros(intervals)
    gross = np.zeros(intervals)
    names = np.zeros(intervals)
    deployed = np.zeros(intervals)
    limit_blocked = liquidity_capped = candidates = 0
    volume = (inputs.dollar_volume if inputs.dollar_volume is not None
              else np.full(inputs.index.count, np.nan))

    def pick(order: np.ndarray, count: int, blocked_at: float, sign: float) -> np.ndarray:
        """The first ``count`` names of ``order`` whose open is not at the blocking limit."""
        nonlocal limit_blocked
        chosen: list[int] = []
        for row in order:
            if len(chosen) >= count:
                break
            gap = inputs.move.gap[row]
            if np.isfinite(gap) and ((sign > 0 and gap >= blocked_at)
                                     or (sign < 0 and gap <= blocked_at)):
                limit_blocked += 1
                continue
            chosen.append(int(row))
        return np.array(chosen, dtype=np.int64)

    for j in range(intervals):
        rows = data.groups[j]
        rows = rows[np.isfinite(scores[rows]) & np.isfinite(data.intraday[rows])]
        if rows.size == 0:
            continue
        order = rows[np.argsort(-scores[rows], kind="stable")]
        count = max(1, math.ceil(rows.size * fraction))
        candidates += 2 * count
        long_rows = pick(order, count, up, 1.0)
        short_rows = pick(order[::-1], count, down, -1.0)
        for sign, chosen in ((1.0, long_rows), (-1.0, short_rows)):
            if chosen.size == 0:
                continue
            # Equal slots of 1 / count per leg; unfilled slots stay in cash. With a capital
            # amount, a slot may not exceed the participation cap of the name's daily dollar
            # volume: the excess stays in cash (no cost, no return).
            weights = np.full(chosen.size, 1.0 / count)
            if capital.twd is not None:
                allowed = participation_cap * volume[chosen] / capital.twd
                allowed = np.where(np.isfinite(allowed), allowed, 0.0)
                liquidity_capped += int((weights > allowed).sum())
                weights = np.minimum(weights, allowed)
            deployed[j] += float(weights.sum())
            ret = sign * np.expm1(data.intraday[chosen])
            gross[j] += float((weights * ret).sum())
            net[j] += float((weights * (ret - round_trip[chosen])).sum())
            names[j] += chosen.size
    info = {
        "fraction": fraction, "names_mean": float(names.mean()),
        "turnover": 1.0, "holding_mean": 1.0,
        "limit_blocked_share": limit_blocked / max(candidates, 1),
        "liquidity_capped_share": liquidity_capped / max(candidates, 1),
        "deployed_share": float(deployed.mean()) / 2.0,
    }
    return Book(net=net, gross=gross, info=info)


def tx_overlay(
    book: Book,
    data: MarketData,
    profile: CostProfile,
    config: dict[str, Any],
    *,
    intraday: bool,
) -> Book:
    """Add a TX position of -beta x capital, beta from the book's trailing gross returns."""
    tx_cfg = config["tx"]
    multiplier = float(tx_cfg["multiplier"])
    rate = tx_side_rate(data, profile, multiplier)
    market = data.index_intraday if intraday else data.index_interval
    beta = rolling_beta(
        book.gross, market, window=int(tx_cfg["beta_window"]),
        min_obs=int(tx_cfg["beta_min_observations"]),
    )
    notional = -beta
    growth = data.tx_intraday if intraday else data.futures.growth
    growth = np.where(np.isfinite(growth), growth, 0.0)
    pnl = notional * growth
    if intraday:
        fees = 2.0 * np.abs(notional) * rate
    else:
        previous = np.concatenate(([0.0], notional[:-1]))
        rolls = 2.0 * np.abs(notional) * data.futures.roll * rate
        fees = np.abs(notional - previous) * rate + rolls
    info = {
        **book.info,
        "beta_mean": float(beta.mean()),
        "beta_abs_mean": float(np.abs(beta).mean()),
        "tx_notional_mean": float(np.abs(notional).mean()),
        "tx_fees_annual": float(fees.mean()) * int(config["annualization_days"]),
    }
    legs = {**(book.legs or {}), "tx": pnl - fees}
    return Book(net=book.net + pnl - fees, gross=book.gross + pnl, info=info, legs=legs)


def combine(md: Book, dt: Book, config: dict[str, Any]) -> Book:
    weights = config["combination"]
    w_md, w_dt = float(weights["multi_day_weight"]), float(weights["day_trade_weight"])
    info = {
        **{f"md_{key}": value for key, value in md.info.items()},
        **{f"dt_{key}": value for key, value in dt.info.items()},
        "turnover": w_md * md.info["turnover"] + w_dt * dt.info["turnover"],
    }
    legs = {
        **{f"md_{name}": w_md * series for name, series in (md.legs or {}).items()},
        **{f"dt_{name}": w_dt * series for name, series in (dt.legs or {}).items()},
    }
    return Book(net=w_md * md.net + w_dt * dt.net, gross=w_md * md.gross + w_dt * dt.gross,
                info=info, legs=legs or None)


# --------------------------------------------------------------------------- evaluation


def block_for(scenario: str, config: dict[str, Any]) -> int:
    if scenario in ("dt_ls", "dt_ls_tx"):
        return int(config["day_trade"]["bootstrap_block"])
    if scenario in ("combo", "combo_tx"):
        return int(config["combination"]["bootstrap_block"])
    return int(config["multi_day"]["bootstrap_block"])


def summarize_book(
    book: Book, scenario: str, config: dict[str, Any], masks: dict[str, np.ndarray],
    index_total: np.ndarray, shared: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    days = int(config["annualization_days"])
    bootstrap = {**config["bootstrap"], "block": block_for(scenario, config)}
    for period, mask in masks.items():
        if not mask.any():
            continue
        zero = np.zeros(int(mask.sum()))
        summary = positions.summarize(
            book.net[mask], book.gross[mask], zero,
            turnover=float(book.info.get("turnover", float("nan"))),
            annualization_days=days, bootstrap=bootstrap,
        )
        index = index_total[mask]
        legs = {
            f"{name}_net_annual": float(np.nanmean(series[mask])) * days
            for name, series in (book.legs or {}).items()
        }
        rows.append({
            **shared, "scenario": scenario, "period": period, **summary,
            "hit_rate": float(np.mean(book.net[mask] > 0)),
            "index_total_return_annual": float(np.nanmean(index)) * days,
            **book.info, **legs,
        })
    return rows


def scenario_books(
    data: MarketData, config: dict[str, Any], profile: CostProfile, capital: Capital,
    *, md_params: tuple[int, int, int], fraction: float, participation_cap: float,
    score_variant: str = "v0", leg_matching: str = "none",
) -> dict[str, Book]:
    wanted = tuple(config.get("scenarios", SCENARIOS))
    h, k, buffer = md_params
    books: dict[str, Book] = {}
    md = multi_day_book(data, config, profile, capital, h=h, k=k, buffer=buffer,
                        participation_cap=participation_cap, score_variant=score_variant,
                        leg_matching=leg_matching)
    md_tx = tx_overlay(md, data, profile, config, intraday=False)
    books["md_ls"], books["md_ls_tx"] = md, md_tx
    if any(name in wanted for name in DAY_TRADE):
        dt = day_trade_book(data, config, profile, capital, fraction=fraction,
                            participation_cap=participation_cap)
        dt_tx = tx_overlay(dt, data, profile, config, intraday=True)
        books["dt_ls"], books["dt_ls_tx"] = dt, dt_tx
        books["combo"], books["combo_tx"] = combine(md, dt, config), combine(md_tx, dt_tx, config)
    return {name: books[name] for name in SCENARIOS if name in wanted and name in books}


def combined_market(
    books: dict[str, dict[str, Book]], config: dict[str, Any]
) -> dict[str, Book] | None:
    spec = config.get("combined_market")
    if not spec or any(market not in books for market in spec["weights"]):
        return None
    out = {}
    scenarios = [name for name in SCENARIOS if name in next(iter(books.values()))]
    for scenario in scenarios:
        net = sum(float(w) * books[m][scenario].net for m, w in spec["weights"].items())
        gross = sum(float(w) * books[m][scenario].gross for m, w in spec["weights"].items())
        info = {"turnover": sum(float(w) * books[m][scenario].info["turnover"]
                                for m, w in spec["weights"].items())}
        for key in ("beta_mean", "beta_abs_mean", "tx_notional_mean", "names_mean",
                    "holding_mean", "liquidity_blocked_share", "liquidity_capped_share",
                    "deployed_share"):
            values = [books[m][scenario].info.get(key) for m in spec["weights"]]
            if all(v is not None for v in values):
                info[key] = sum(float(w) * v for v, w in zip(values, spec["weights"].values(),
                                                              strict=True))
        legs = None
        parts = [books[m][scenario].legs for m in spec["weights"]]
        if all(part is not None for part in parts) and parts:
            names = set.intersection(*(set(part) for part in parts))  # type: ignore[arg-type]
            legs = {
                name: sum(float(w) * books[m][scenario].legs[name]  # type: ignore[index]
                          for m, w in spec["weights"].items())
                for name in sorted(names)
            }
        out[scenario] = Book(net=np.asarray(net), gross=np.asarray(gross), info=info, legs=legs)
    return out


def _profile_rows(
    label: str, market: str, profile: CostProfile, capital: Capital, books: dict[str, Book],
    data: MarketData, config: dict[str, Any], *, primary: bool,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    shared = {
        "model": label, "market": market, "profile": profile.name,
        "slippage_multiplier": profile.slippage_multiplier, "primary_slippage": primary,
        "capital": capital.name, "capital_twd": capital.twd, **(extra or {}),
    }
    rows = []
    for scenario, book in books.items():
        rows.extend(summarize_book(book, scenario, config, data.masks, data.index_total_return,
                                   shared))
    return rows


def select_multi_day(
    data: MarketData, config: dict[str, Any], profile: CostProfile, participation_cap: float,
) -> tuple[list[dict[str, Any]], tuple[int, int, int], dict[str, Any]]:
    """Evaluate the S1 grid on the selection market and pick (h, K, m) by P1 net Sharpe."""
    grid = config["multi_day"]["grid"]
    rule = config["selection"]
    rows, candidates = [], []
    fractional = Capital(FRACTIONAL, None, None)
    for h in grid["score_horizons"]:
        for k in grid["k"]:
            for m in grid["buffer_multiples"]:
                book = multi_day_book(data, config, profile, fractional, h=int(h), k=int(k),
                                      buffer=int(m), participation_cap=participation_cap)
                shared = {"model": None, "market": data.market, "profile": profile.name,
                          "slippage_multiplier": profile.slippage_multiplier,
                          "primary_slippage": True, "capital": FRACTIONAL, "capital_twd": None,
                          "stage": "grid"}
                for row in summarize_book(book, "md_ls", config, data.masks,
                                          data.index_total_return, shared):
                    rows.append(row)
                    if row["period"] == rule["period"]:
                        candidates.append(((int(h), int(k), int(m)), row[rule["metric"]]))
    finite = [(params, value) for params, value in candidates if np.isfinite(value)]
    if not finite:
        return rows, (grid["score_horizons"][0], grid["k"][0], grid["buffer_multiples"][0]), {
            "reason": "no finite selection metric"}
    best = max(finite, key=lambda item: (item[1], -item[0][0], -item[0][1], -item[0][2]))
    return rows, best[0], {"h": best[0][0], "k": best[0][1], "buffer": best[0][2],
                           "metric": rule["metric"], "value": float(best[1])}


def select_day_trade(
    data: MarketData, config: dict[str, Any], profile: CostProfile, participation_cap: float,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    rule = config["selection"]
    rows, candidates = [], []
    fractional = Capital(FRACTIONAL, None, None)
    for fraction in config["day_trade"]["grid"]["fractions"]:
        book = day_trade_book(data, config, profile, fractional, fraction=float(fraction),
                              participation_cap=participation_cap)
        shared = {"model": None, "market": data.market, "profile": profile.name,
                  "slippage_multiplier": profile.slippage_multiplier, "primary_slippage": True,
                  "capital": FRACTIONAL, "capital_twd": None, "stage": "grid"}
        for row in summarize_book(book, "dt_ls", config, data.masks, data.index_total_return,
                                  shared):
            rows.append(row)
            if row["period"] == rule["period"]:
                candidates.append((float(fraction), row[rule["metric"]]))
    finite = [(q, value) for q, value in candidates if np.isfinite(value)]
    if not finite:
        return rows, float(config["day_trade"]["grid"]["fractions"][0]), {
            "reason": "no finite selection metric"}
    best = max(finite, key=lambda item: (item[1], -item[0]))
    return rows, best[0], {"fraction": best[0], "metric": rule["metric"], "value": float(best[1])}


def attractiveness(grid: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered 'attractive to the industry' bar, per (model, scenario, market)."""
    rule = config["attractiveness"]
    out: dict[str, Any] = {}
    base = grid[(grid["stage"] == "final") & (grid["profile"] == rule["profile"])
                & grid["primary_slippage"] & (grid["capital"] == FRACTIONAL)]
    keys = ["model", "scenario", "market", "score_variant", "leg_matching"]
    for (model, scenario, market, variant, matching), rows in base.groupby(keys):
        baseline = (variant, matching) == ("v0", "none")
        label = market if baseline else f"{market}|{variant}/{matching}"
        decision = rows[rows["period"] == rule["period"]]
        confirm = rows[rows["period"] == rule["require_positive_period"]]
        if decision.empty:
            continue
        row = decision.iloc[0]
        sharpe_ok = bool(np.isfinite(row["sharpe"]) and row["sharpe"] >= float(rule["min_sharpe"]))
        lower = float(row["net_excess_annual_low"])
        lower_ok = bool(np.isfinite(lower) and lower > float(rule["lower_bound_above"]))
        confirm_ok = bool(not confirm.empty and float(confirm.iloc[0]["net_return_annual"]) > 0)
        out.setdefault(model, {}).setdefault(scenario, {})[label] = {
            "sharpe": float(row["sharpe"]), "sharpe_ok": sharpe_ok,
            "net_return_annual": float(row["net_return_annual"]),
            "net_return_annual_low": lower, "lower_bound_ok": lower_ok,
            "confirmation_net_return_annual": (
                float(confirm.iloc[0]["net_return_annual"]) if not confirm.empty else None),
            "confirmation_ok": confirm_ok,
            "attractive": sharpe_ok and lower_ok and confirm_ok,
        }
    return out


# --------------------------------------------------------------------------- run


def _primary_profile(profiles: dict[str, Any], name: str) -> CostProfile:
    for profile in cost_profiles(profiles):
        if profile.name == name and is_primary(profile, profiles):
            return profile
    raise ValueError(f"No primary profile named {name!r}")


def run(
    config_path: Path,
    predictions: list[Path],
    models: list[str],
    prices_path: Path,
    output_root: Path,
    *,
    artifacts: Path,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = load_profiles(config)
    participation_cap = float(profiles["participation_cap"])
    prices = load_prices(
        prices_path, liquidity_lookback=int(config["universe"]["liquidity_lookback_sessions"])
    )
    tx_cfg = config["tx"]
    front = taifex.continuous_front_month(
        taifex.load_contract(artifacts / tx_cfg["root"], tx_cfg["contract"]), tx_cfg["contract"]
    )
    selection_profile = _primary_profile(profiles, config["selection"]["profile"])
    construction = config.get("construction", {})
    combos = list(itertools.product(construction.get("score_variants", ["v0"]),
                                    construction.get("leg_matching", ["none"])))
    needs_features = any(v != "v0" or m != "none" for v, m in combos)
    features_table = trailing_features(prices_path) if needs_features else None
    wanted = tuple(config.get("scenarios", SCENARIOS))
    fixed = config["multi_day"].get("fixed", {})
    all_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    daily_frames = []
    for path, label in zip(predictions, models, strict=True):
        frame = load_predictions(path, label)
        inputs = build_inputs(frame, prices)
        intraday = intraday_returns(frame, prices)
        features = row_features(frame, features_table) if features_table is not None else None
        markets = {
            market: prepare_market(inputs, intraday, market, config, prices, front, features)
            for market in config["markets"]
        }
        chooser = markets[config["selection"]["market"]]
        if label in fixed:
            spec = fixed[label]
            md_params = (int(spec["h"]), int(spec["k"]), int(spec["buffer"]))
            md_choice = {"h": md_params[0], "k": md_params[1], "buffer": md_params[2],
                         "fixed": True}
            grid_rows: list[dict[str, Any]] = []
        else:
            grid_rows, md_params, md_choice = select_multi_day(
                chooser, config, selection_profile, participation_cap)
        if any(name in wanted for name in DAY_TRADE):
            dt_rows, fraction, dt_choice = select_day_trade(
                chooser, config, selection_profile, participation_cap)
        else:
            dt_rows, fraction, dt_choice = [], float(
                config["day_trade"]["grid"]["fractions"][0]), {"skipped": True}
        for row in grid_rows + dt_rows:
            row["model"] = label
            row["score_variant"], row["leg_matching"] = "v0", "none"
        all_rows.extend(grid_rows + dt_rows)
        selection[label] = {"multi_day": md_choice, "day_trade": dt_choice}
        for (variant, matching), profile in itertools.product(combos, cost_profiles(profiles)):
            primary = is_primary(profile, profiles)
            extra = {"score_variant": variant, "leg_matching": matching}
            for capital in capital_scenarios(profiles):
                if capital.profile is not None and capital.profile != profile.name:
                    continue
                if capital.twd is not None and not primary:
                    continue
                books_by_market = {}
                for market, data in markets.items():
                    books = scenario_books(
                        data, config, profile, capital, md_params=md_params, fraction=fraction,
                        participation_cap=participation_cap, score_variant=variant,
                        leg_matching=matching,
                    )
                    books_by_market[market] = books
                    rows = _profile_rows(label, market, profile, capital, books, data, config,
                                         primary=primary, extra=extra)
                    for row in rows:
                        row["stage"] = "final"
                    all_rows.extend(rows)
                    if primary and capital.twd is None:
                        daily_frames.append(
                            _daily_frame(label, market, profile, books, data, extra))
                combined = combined_market(books_by_market, config)
                if combined is not None:
                    name = config["combined_market"]["name"]
                    first = next(iter(markets.values()))
                    index_total = np.nanmean(
                        np.vstack([markets[m].index_total_return for m in markets]), axis=0)
                    shared = {"model": label, "market": name, "profile": profile.name,
                              "slippage_multiplier": profile.slippage_multiplier,
                              "primary_slippage": primary, "capital": capital.name,
                              "capital_twd": capital.twd, "stage": "final", **extra}
                    for scenario, book in combined.items():
                        all_rows.extend(summarize_book(book, scenario, config, first.masks,
                                                       index_total, shared))
                    if primary and capital.twd is None:
                        daily_frames.append(
                            _daily_frame(label, name, profile, combined, first, extra))
    grid = pd.DataFrame(all_rows)
    verdicts = attractiveness(grid, config)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "scenario-suite-retail-vs-institution",
        "config": str(config_path),
        "cost_profiles": str(config["cost_profiles"]),
        "predictions": {label: str(path) for path, label in zip(predictions, models, strict=True)},
        "prices": str(prices_path),
        "selection": selection,
        "attractiveness": verdicts,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, grid, profiles),
                                            encoding="utf-8")
    return report


def _daily_frame(label: str, market: str, profile: CostProfile, books: dict[str, Book],
                 data: MarketData, extra: dict[str, Any] | None = None) -> pd.DataFrame:
    dates = data.inputs.dates[1:].strftime("%Y-%m-%d")
    frames = []
    for scenario, book in books.items():
        frames.append(pd.DataFrame({
            "model": label, "market": market, "profile": profile.name, "scenario": scenario,
            "interval_start": dates, "net": book.net, "gross": book.gross,
            "index_total_return": data.index_total_return, **(extra or {}),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- summary


def _cell_ci(row) -> str:
    return (f"[{runner.percent(row.net_excess_annual_low)}, "
            f"{runner.percent(row.net_excess_annual_high)}]")


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame, profiles: dict[str, Any]) -> str:
    final = grid[(grid["stage"] == "final") & grid["primary_slippage"]
                 & (grid["capital"] == FRACTIONAL)]
    header = (
        "| 情境 / Scenario | 模型 | 市場 | 成本檔 | 期間 | 年化毛報酬 | 年化淨報酬 | 95% 區間 "
        "| Sharpe | 最大回撤 | 勝率 | 換手 | 成本拖累 | TAIEX 報酬指數 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    lines = ["# 六情境回測摘要（散戶 vs 機構）/ Six-scenario summary (retail vs institution)", ""]
    for model, choice in report["selection"].items():
        md, dt = choice["multi_day"], choice["day_trade"]
        lines.append(f"- {model}: h = {md.get('h')}, K = {md.get('k')}, m = {md.get('buffer')}; "
                     f"q = {dt.get('fraction')}")
    lines.append("")
    lines.extend(header)
    order = {name: i for i, name in enumerate(SCENARIOS)}
    final = final.assign(_order=final["scenario"].map(order))
    sort_keys = ["_order", "model", "market", "profile", "period"]
    for row in final.sort_values(sort_keys).itertuples():
        pct = runner.percent
        lines.append(
            f"| {row.scenario} {row.score_variant}/{row.leg_matching} | {row.model} | "
            f"{row.market} | {row.profile} | {row.period} | "
            f"{pct(row.gross_excess_annual)} | {pct(row.net_return_annual)} | {_cell_ci(row)} | "
            f"{runner.number(row.sharpe)} | {pct(row.max_drawdown)} | {pct(row.hit_rate)} | "
            f"{pct(row.turnover)} | {pct(row.cost_drag_annual)} | "
            f"{pct(row.index_total_return_annual)} |"
        )
    lines.append("")
    lines.append("## 吸引力門檻 / Attractiveness bar")
    lines.append("")
    for model, scenarios in report["attractiveness"].items():
        for scenario, markets in scenarios.items():
            for market, verdict in markets.items():
                word = "達標 / attractive" if verdict["attractive"] else "未達標 / not attractive"
                lines.append(
                    f"- {model} {scenario} {market}: {word} "
                    f"(Sharpe {verdict['sharpe']:.2f}, lower bound "
                    f"{runner.percent(verdict['net_return_annual_low'])}, P2 "
                    f"{runner.percent(verdict['confirmation_net_return_annual'])})"
                )
    return "\n".join(lines) + "\n"


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
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "scenario-suite")
    report = run(args.config, args.predictions, args.model, args.prices, output_root,
                 artifacts=args.artifacts_root)
    print(json.dumps({"output_root": str(output_root), "selection": report["selection"],
                      "attractiveness": report["attractiveness"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
