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
* S-A ``wk_long`` (protocol 13.20): the weekly low-turnover long band of ``weekly_band``, built
  only when the configuration has a ``weekly`` block; K and m are selected on the same market,
  period and metric as S1, the score is the orthogonalized h = 5 score, and the equal-weight
  universe and 0050 (``benchmarks`` block) are comparison columns.

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
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, positions, runner, stats, taifex
from twbacktest.borrow_execution import (
    BorrowData,
    ExecutionSpec,
    execution_specs,
    load_borrow_data,
    short_arrays,
    summarize_borrow,
)
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
    BandResult,
    FuturesSeries,
    futures_series,
    percentile_ranks,
    scaled_rates,
    simulate_band,
    smooth_by_symbol,
    universe_groups,
    universe_mask,
)
from twbacktest.walkforward_data import (
    WalkForwardInputs,
    build_inputs,
    etf_growth,
    load_predictions,
    load_prices,
)
from twbacktest.weekly_band import SCENARIO as WEEKLY
from twbacktest.weekly_band import select_weekly, weekly_long_book

WEEKLY_TX = "wk_long_tx"  # S-A plus the TX overlay (protocol 13.21, S-A-TX)
WEEKLY_ETF = "wk_long_etf"  # S-A plus a short 0050 hedge (protocol 13.21, S-A-ETF)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "scenario_suite.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FRACTIONAL = "fractional"
SCENARIOS = ("md_ls", "md_ls_tx", "dt_ls", "dt_ls_tx", "combo", "combo_tx")  # the 13.19 six
ALL_SCENARIOS = (*SCENARIOS, WEEKLY, WEEKLY_TX, WEEKLY_ETF)  # S-A and its hedges, on request
SCENARIO_LABELS = {
    "md_ls": ("S1 短至中天期台股多空", "S1 multi-day stock long-short"),
    "md_ls_tx": ("S2 S1 + 台指期", "S2 S1 + TX overlay"),
    "dt_ls": ("S3 當沖台股多空", "S3 day-trade stock long-short"),
    "dt_ls_tx": ("S4 S3 + 台指期", "S4 S3 + TX overlay"),
    "combo": ("S5 當沖 + 多日", "S5 day trade + multi-day"),
    "combo_tx": ("S6 S5 + 台指期", "S6 S5 + TX overlays"),
    WEEKLY: ("S-A 週頻低換手多頭", "S-A weekly low-turnover long"),
    WEEKLY_TX: ("S-A-TX 週頻多頭 + 台指期對沖", "S-A-TX weekly long + TX hedge"),
    WEEKLY_ETF: ("S-A-ETF 週頻多頭 + 放空 0050", "S-A-ETF weekly long + short 0050"),
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


def with_borrow_fee(path: positions.Path, daily_fee: float | np.ndarray) -> positions.Path:
    """Charge the borrow fee on every held short slot, per interval (a scalar, or a per-row
    daily rate for tiered fees)."""
    held = path.held >= 0
    if np.isscalar(daily_fee):
        if float(daily_fee) <= 0:
            return path
        charge = np.where(held, math.log1p(-float(daily_fee)), 0.0)
    else:
        rates = np.asarray(daily_fee, dtype=np.float64)[np.maximum(path.held, 0)]
        charge = np.where(held, np.log1p(-rates), 0.0)
    return replace(path, growth=path.growth + charge)


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
    borrow: BorrowData | None = None  # borrowability per row (supplement 4)
    benchmarks: dict[str, np.ndarray] | None = None  # comparison series per interval (13.20)


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
    benchmarks = None
    bench_cfg = config.get("benchmarks")
    if bench_cfg:
        benchmarks = {"universe_ew": positions.equal_weight(groups, inputs.move)}
        etf = bench_cfg.get("etf")
        if etf:
            benchmarks[str(etf.get("name", "etf_0050"))] = etf_growth(
                prices, dates, str(etf["symbol"]), splits=etf.get("splits"))
    return MarketData(
        market=market, inputs=inputs, mask=mask, groups=groups, index_interval=index_interval,
        index_intraday=index_intraday, index_total_return=index_total, futures=futures,
        tx_intraday=tx_intraday, tx_notional_per_contract=notional, intraday=intraday,
        lot_cost=raw_open * lot,
        masks=period_masks(dates[1:], config["periods"], config.get("decision_periods")),
        features=features, benchmarks=benchmarks,
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
    gross_legs: dict[str, np.ndarray] | None = None  # the same parts before costs and fees


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


def disposition_mask(data: MarketData, config: dict[str, Any]) -> np.ndarray | None:
    """Rows under a disposition period when the configuration switches rule D on (protocol
    13.23 supplement 3), else None."""
    exec_cfg = config.get("execution") or {}
    if not exec_cfg.get("disposition_block"):
        return None
    if data.borrow is None or data.borrow.disposition is None:
        raise ValueError("disposition_block needs the market's borrow data with dispositions")
    return data.borrow.disposition


# Tick sizes of the exchange by (unadjusted) price, NT$ (protocol 13.23 supplement 4 (c)).
TICK_TABLE = ((10.0, 0.01), (50.0, 0.05), (100.0, 0.1), (500.0, 0.5), (1000.0, 1.0),
              (np.inf, 5.0))


def tick_size(price: np.ndarray) -> np.ndarray:
    """Tick size per row from the unadjusted price (NaN price -> NaN)."""
    out = np.full(price.shape, np.nan)
    lower = 0.0
    for upper, tick in TICK_TABLE:
        mask = np.isfinite(price) & (price >= lower) & (price < upper)
        out[mask] = tick
        lower = upper
    return out


def stop_settings(data: MarketData, config: dict[str, Any], *, short: bool = False) -> dict:
    """Keyword arguments of the trailing stop for one leg (supplement 4 rule L), or {}."""
    cfg = config.get("stop_loss")
    if not cfg:
        return {}
    inputs = data.inputs
    if inputs.intraday_high is None or inputs.intraday_low is None:
        raise ValueError("stop_loss needs the intraday high and low of the execution session")
    high, low = inputs.intraday_high, inputs.intraday_low
    if short:
        # Position value of a short slot: 2 - exp(price move), floored like short_moves.
        def flip(values: np.ndarray) -> np.ndarray:
            out = np.full(values.shape, np.nan)
            ok = np.isfinite(values)
            out[ok] = np.log(np.maximum(2.0 - np.exp(values[ok]), SHORT_FLOOR))
            return out
        adverse, favorable = flip(high), flip(low)
    else:
        adverse, favorable = low, high
    discount = None
    if cfg.get("half_tick", True) and inputs.raw_open_1 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            discount = 0.5 * tick_size(inputs.raw_open_1) / inputs.raw_open_1
    return {"stop_fraction": float(cfg["fraction"]), "adverse": adverse, "favorable": favorable,
            "stop_discount": discount, "stop_overshoot": float(cfg.get("overshoot", 0.0))}


def impact_rates(
    result: BandResult, data: MarketData, config: dict[str, Any], *, short: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-row (buy, sell) fill-cost fractions of supplement 4 rule F / supplement 5 policy B,
    or None when the fill model is off.

    Policy A (supplement 4): eps = Y x sigma_daily x sqrt(Q / (f x V)) with Q the fill's value
    (NaN or 0 in the fractional scenario -> 0), V the trailing median traded value and f the
    opening auction's share; a stop fill (``stop_rows``) trades in the continuous session, so
    f = 1 for it. Policy B (supplement 5): orders worked through the session after the open,
    so f = 1, every traded row pays half a tick, and the delay term enters as a signed rate:
    the fill is open x exp(theta x r_oc) with r_oc the session's log(close / open), which is a
    buy "cost" of 1 - exp(-theta r) and a sell "cost" of 1 - exp(theta r) for a long book and
    the mirror image for a short book (entry = short sale, exit = cover). Exits at a close
    (missing exits, stops) carry no delay.
    """
    cfg = (config.get("execution") or {}).get("open_fill")
    if not cfg:
        return None
    inputs = data.inputs
    if inputs.volatility is None or inputs.dollar_volume is None:
        raise ValueError("open_fill needs the trailing volatility and traded value")
    policy = str(cfg.get("policy", "A"))
    if policy not in ("A", "B"):
        raise ValueError(f"Unknown fill policy {policy!r}")
    y = float(cfg["impact_y"])
    share = 1.0 if policy == "B" else float(cfg["auction_share"])
    sigma = np.where(np.isfinite(inputs.volatility), inputs.volatility, 0.0)
    volume = inputs.dollar_volume
    rows = sigma.size

    def cost(value: np.ndarray, auction: np.ndarray) -> np.ndarray:
        out = np.zeros(rows)
        ok = np.isfinite(value) & (value > 0) & np.isfinite(volume) & (volume > 0)
        out[ok] = y * sigma[ok] * np.sqrt(value[ok] / (auction[ok] * volume[ok]))
        return out

    opening = np.full(rows, share)
    stop_rows = result.stop_rows if result.stop_rows is not None else np.zeros(rows, dtype=bool)
    buy = cost(result.buy_value, opening)
    sell = cost(result.sell_value, np.where(stop_rows, 1.0, opening))
    if policy == "B":
        path = result.path
        bought = np.zeros(rows, dtype=bool)
        bought[path.buy[path.buy >= 0]] = True
        at_open = np.zeros(rows, dtype=bool)
        at_open[path.sell_open[path.sell_open >= 0]] = True
        at_open[path.final[path.final >= 0]] = True
        at_close = np.zeros(rows, dtype=bool)
        at_close[path.sell_close[path.sell_close >= 0]] = True
        fill = np.where(np.isfinite(result.fill_fraction), result.fill_fraction, 1.0)
        extra_buy, extra_sell = np.zeros(rows), np.zeros(rows)
        if cfg.get("half_tick", True) and inputs.raw_open_1 is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                half = 0.5 * tick_size(inputs.raw_open_1) / inputs.raw_open_1
            half = np.where(np.isfinite(half), half, 0.0)
            extra_buy += np.where(bought, half, 0.0)
            extra_sell += np.where(at_open | at_close, half, 0.0)
        theta = float(cfg.get("delay_theta", 0.0))
        if theta:
            move = np.where(np.isfinite(data.intraday), data.intraday, 0.0)
            sign = -1.0 if short else 1.0
            # Long: buy dearer by exp(theta r) -> cost 1 - exp(-theta r); sell at exp(theta r)
            # -> cost 1 - exp(theta r). Short: the signs flip.
            extra_buy += np.where(bought, 1.0 - np.exp(-sign * theta * move), 0.0)
            extra_sell += np.where(at_open, 1.0 - np.exp(sign * theta * move), 0.0)
        buy = buy + extra_buy * fill
        sell = sell + extra_sell * fill
    return buy, sell


def execution_tags(config: dict[str, Any]) -> dict[str, Any]:
    """Supplement 4 parameters as grid columns (NaN when the rule is off)."""
    fill = (config.get("execution") or {}).get("open_fill") or {}
    stop = config.get("stop_loss") or {}
    odd = (config.get("execution") or {}).get("odd_lots") or {}
    odd_cost = (config.get("execution") or {}).get("odd_lots_cost") or {}
    return {
        "fill_policy": str(fill.get("policy", "A")) if fill else "",
        "delay_theta": float(fill.get("delay_theta", np.nan)) if fill else np.nan,
        "impact_y": float(fill.get("impact_y", np.nan)),
        "auction_share": float(fill.get("auction_share", np.nan)) if fill else np.nan,
        "stop_fraction": float(stop.get("fraction", np.nan)),
        "stop_overshoot": float(stop.get("overshoot", np.nan)),
        "odd_lots": ";".join(f"{name}:{','.join(rows)}" for name, rows in sorted(odd.items())),
        "odd_gap": float(odd_cost.get("gap", np.nan)) if odd else np.nan,
    }


def capital_basis(config: dict[str, Any]) -> str:
    basis = str(config.get("capital_basis", "per_leg"))
    if basis not in ("per_leg", "unit_exposure"):
        raise ValueError(f"Unknown capital_basis {basis!r}")
    return basis


def capital_share(config: dict[str, Any], data: MarketData, legs: int) -> float:
    """Share of the capital one leg of one market's book trades (protocol 13.23 supplement 6).

    ``per_leg`` (the pre-supplement basis): every leg of every market uses the whole capital.
    ``unit_exposure``: the capital is the total across the combined market, so a leg gets the
    market's combination weight (0.5 each for TWSE / TPEX when combined) times 1 / legs.
    """
    if capital_basis(config) == "per_leg":
        return 1.0
    weights = (config.get("combined_market") or {}).get("weights") or {}
    return float(weights.get(data.market, 1.0)) / float(legs)


def fill_profile(profile: CostProfile, config: dict[str, Any], capital: Capital) -> CostProfile:
    """The cost profile a capital row trades under when the fill model is on (supplement 4
    amendment of 2026-11-06): the flat per-side slippage of the cost file was the stand-in for
    execution cost, and the fill model now prices that cost itself (own impact in the opening
    call; half a tick plus impact in the continuous session), so the flat slippage is dropped
    for capital rows. The fractional row (Q -> 0, fill = the open) keeps it, as do runs without
    a fill model, and ``open_fill.keep_flat_slippage`` restores it."""
    cfg = (config.get("execution") or {}).get("open_fill")
    if not cfg or capital.twd is None or cfg.get("keep_flat_slippage", False):
        return profile
    return replace(profile, slippage=(0.0, 0.0))


def odd_lot_spec(config: dict[str, Any], capital: Capital) -> dict[str, Any] | None:
    """Supplement 7: the odd-lot settings of a capital row, or None when the row trades whole
    lots only. ``execution.odd_lots`` maps a cost profile to the capital rows whose long legs
    may buy odd lots; ``execution.odd_lots_cost`` holds the gap ``g`` to the board-lot price,
    whether half a tick is added (policy A; policy B already charges it) and the minimum
    commission per odd-lot trade in NT$."""
    cfg = (config.get("execution") or {}).get("odd_lots")
    if not cfg or capital.twd is None or capital.profile is None:
        return None
    if capital.name not in (cfg.get(capital.profile) or []):
        return None
    cost = (config.get("execution") or {}).get("odd_lots_cost") or {}
    return {
        "gap": float(cost.get("gap", 0.0)),
        "half_tick": bool(cost.get("half_tick", True)),
        "min_commission_twd": float(cost.get("min_commission_twd", 0.0)),
    }


def odd_lot_rates(
    result: BandResult, data: MarketData, config: dict[str, Any], profile: CostProfile,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (buy, sell) cost additions of the rows a long book traded as odd lots: the gap
    ``g`` to the board-lot price (paid on buys, conceded on sells), half a tick under policy A
    (limit orders at the quote), and the minimum-commission top-up max(min / value - rate, 0).
    Rows traded as whole lots get nothing."""
    inputs = data.inputs
    rows = result.buy_value.size
    policy = str(((config.get("execution") or {}).get("open_fill") or {}).get("policy", "A"))
    per_side = float(spec["gap"])
    half = np.zeros(rows)
    if spec["half_tick"] and policy != "B" and inputs.raw_open_1 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            half = 0.5 * tick_size(inputs.raw_open_1) / inputs.raw_open_1
        half = np.where(np.isfinite(half), half, 0.0)
    minimum = float(spec["min_commission_twd"])
    out = []
    for value, flags in ((result.buy_value, result.odd_buy), (result.sell_value, result.odd_sell)):
        traded = flags & np.isfinite(value) & (value > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            top_up = np.where(traded, np.maximum(minimum / value - profile.commission, 0.0), 0.0)
        out.append(np.where(traded, per_side + half, 0.0) + np.nan_to_num(top_up))
    return out[0], out[1]


def with_impact(
    rates: tuple[np.ndarray, np.ndarray], result: BandResult, data: MarketData,
    config: dict[str, Any], *, short: bool = False, profile: CostProfile | None = None,
    capital: Capital | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    buy, sell = rates
    extra = impact_rates(result, data, config, short=short)
    if extra is not None:
        buy, sell = buy + extra[0], sell + extra[1]
    spec = odd_lot_spec(config, capital) if (capital is not None and profile is not None) else None
    if spec is not None and not short:
        odd = odd_lot_rates(result, data, config, profile, spec)
        buy, sell = buy + odd[0], sell + odd[1]
    return buy, sell


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
    cap_mode: str = "skip",
    execution: ExecutionSpec | None = None,
    cache: dict[tuple, Any] | None = None,
    part: int = 0,
    odd_lots: bool = False,
    blocked: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One long band book and one short band book on ``groups`` ranked by ``score``.

    With ``execution`` the short book runs under the borrowability layer of supplement 4,
    once per random replicate, and the replicates' series are averaged. ``cache`` (keyed by
    everything the simulation depends on except the cost rates) lets the slippage
    sensitivities of one profile reuse the simulations; ``part`` separates the matched
    sub-books of one call.
    """
    inputs = data.inputs
    fractional = capital.twd is None
    config = config or {}
    leg_capital = None if fractional else capital.twd * capital_share(config, data, 2)
    odd_lots = bool(odd_lots or odd_lot_spec(config, capital) is not None) and not fractional
    common = {
        "k": k, "buffer": buffer, "max_buys": max_buys, "symbols": inputs.index.symbol_codes,
        "link": inputs.link, "limit": limit,
        "lot_cost": None if fractional else data.lot_cost,
        "slot_capital": leg_capital,
        "dollar_volume": None if fractional else inputs.dollar_volume,
        "participation_cap": None if fractional else participation_cap,
        "cap_mode": cap_mode,
        "odd_lots": odd_lots,  # the long leg only: no margin or SBL short in odd lots
        "blocked": blocked,
    }
    short_common = {**common, "odd_lots": False}
    long_stop = stop_settings(data, config)
    short_stop = stop_settings(data, config, short=True)
    stop_key = (long_stop.get("stop_fraction"), long_stop.get("stop_overshoot"))
    cache = {} if cache is None else cache
    base_key = (part, capital.name, k, buffer, max_buys, limit, cap_mode, odd_lots,
                blocked is not None, stop_key, capital_basis(config))
    long_key = ("long", *base_key)
    if long_key not in cache:
        cache[long_key] = simulate_band(engine.rankings(score, groups), move=inputs.move,
                                        **common, **long_stop)
    long_result = cache[long_key]
    if execution is None:
        short_key = ("short", *base_key, None)
        if short_key not in cache:
            cache[short_key] = [simulate_band(engine.rankings(-score, groups),
                                              move=short_moves(inputs.move), **short_common,
                                              **short_stop)]
        fee: float | np.ndarray = profile.borrow_fee_daily
    else:
        if data.borrow is None:
            raise ValueError("the execution layer needs the market's borrow data")
        short_key = ("short", *base_key, execution.profile, execution.key, execution.seed,
                     execution.replicates, execution.quota_cap, execution.forced_cover)
        if short_key not in cache:
            arrays = short_arrays(execution, data.borrow, fractional=fractional,
                                  default_fee_annual=profile.borrow_fee_annual)
            arrays.pop("fee_daily")
            ranked = engine.rankings(-score, groups)
            move = short_moves(inputs.move)
            cache[short_key] = [
                simulate_band(ranked, move=move, rng=np.random.default_rng(seed),
                              **short_common, **arrays, **short_stop)
                for seed in execution.replicate_seeds()
            ]
        fee = execution.fee_daily(data.borrow.tercile, profile.borrow_fee_annual)
    short_results = cache[short_key]
    codes = inputs.index.market_codes
    free = np.zeros(inputs.index.count)
    profile = fill_profile(profile, config, capital)
    short_rates = profile.short_rates(codes)
    short_gross = np.mean([positions.replay(r.path, free, free).daily for r in short_results],
                          axis=0)
    short_nets = [
        positions.replay(with_borrow_fee(r.path, fee),
                         *with_impact(scaled_rates(r, *short_rates), r, data, config,
                                      short=True)).daily
        for r in short_results
    ]
    long_stats = positions.path_statistics(long_result.path)
    short_stat_list = [positions.path_statistics(r.path) for r in short_results]

    def short_stat(key: str) -> float:
        return float(np.mean([s[key] for s in short_stat_list]))

    def short_count(attr: str) -> float:
        return float(np.mean([getattr(r, attr) for r in short_results]))

    return {
        "long_gross": positions.replay(long_result.path, free, free).daily,
        "long_net": positions.replay(
            long_result.path,
            *with_impact(scaled_rates(long_result, *profile.long_rates(codes)), long_result,
                         data, config, profile=profile, capital=capital)).daily,
        "short_gross": short_gross,
        "short_net": np.mean(short_nets, axis=0),
        "short_net_replicates": [float(np.mean(net)) for net in short_nets],
        "turnover": 0.5 * (long_stats["turnover"] + short_stat("turnover")),
        "holding_mean": 0.5 * (long_stats["holding_mean"] + short_stat("holding_mean")),
        "cash_fraction": 0.5 * (long_stats["cash_fraction"] + short_stat("cash_fraction")),
        "limit_blocked": (long_stats["blocked_buys"] + long_stats["blocked_sells"]
                          + short_stat("blocked_buys") + short_stat("blocked_sells")),
        "liquidity_blocked": long_result.liquidity_blocked + short_count("liquidity_blocked"),
        "buy_attempts": long_result.buy_attempts + short_count("buy_attempts"),
        "short_attempts": short_count("buy_attempts"),
        "capped_buys": long_result.capped_buys + short_count("capped_buys"),
        "deployed_share": float(0.5 * (long_result.deployed.mean()
                                       + np.mean([r.deployed.mean() for r in short_results]))),
        "short_deployed_share": float(np.mean([r.deployed.mean() for r in short_results])),
        "odd_lot_share": long_result.odd_lot_buys / max(long_result.buy_attempts, 1),
        "ineligible_skipped": short_count("ineligible_skipped"),
        "borrow_failed": short_count("borrow_failed"),
        "quota_capped": short_count("quota_capped"),
        "recalls": short_count("recalls"),
        "blocked_skipped": long_result.blocked_skipped + short_count("blocked_skipped"),
        "forced_exits": short_count("forced_exits"),
        "stop_exits": long_result.stop_exits + short_count("stop_exits"),
        "stop_gap_exits": long_result.stop_gap_exits + short_count("stop_gap_exits"),
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
    execution: ExecutionSpec | None = None,
    cache: dict[tuple, Any] | None = None,
    odd_lots: bool = False,
) -> Book:
    """The multi-day long-short book.

    ``score_variant`` ``v0`` is the smoothed percentile score; ``v1`` orthogonalizes it every day
    inside the universe against the trailing 20/60-session returns and the 60-session beta.
    ``leg_matching`` ``none`` runs one long and one short band on the whole universe;
    ``momentum`` runs one pair per quintile of the trailing 20-session return with k / quintiles
    slots each (equal capital per quintile), so both legs share the momentum distribution.
    ``execution`` applies the borrowability layer of supplement 4 to the short book.
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
    cap_mode = str(load_profiles(config).get("participation_mode", "skip"))
    blocked = disposition_mask(data, config)
    if leg_matching == "none":
        parts = [_band_long_short(
            data, score, data.groups, profile, capital, k=k, buffer=buffer,
            max_buys=max(1, math.ceil(k * cap_fraction)), limit=limit,
            participation_cap=participation_cap, cap_mode=cap_mode, execution=execution,
            cache=cache, odd_lots=odd_lots, blocked=blocked, config=config,
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
                participation_cap=participation_cap, cap_mode=cap_mode, execution=execution,
                cache=cache, part=q + 1, odd_lots=odd_lots, blocked=blocked, config=config,
            ))
    else:
        raise ValueError(f"Unknown leg matching {leg_matching!r}")
    n = len(parts)

    def mean_series(key: str) -> np.ndarray:
        return sum(part[key] for part in parts) / n

    long_net, long_gross = mean_series("long_net"), mean_series("long_gross")
    short_net, short_gross = mean_series("short_net"), mean_series("short_gross")
    attempts = max(sum(part["buy_attempts"] for part in parts), 1)
    short_attempts = max(sum(part["short_attempts"] for part in parts), 1)
    days = int(config["annualization_days"])
    replicate_annual = [
        days * float(np.mean(values))
        for values in zip(*(part["short_net_replicates"] for part in parts), strict=True)
    ]
    info = {
        "h": h, "k": k, "buffer": buffer, "score_variant": score_variant,
        "leg_matching": leg_matching,
        "turnover": float(np.mean([part["turnover"] for part in parts])),
        "holding_mean": _finite_mean([part["holding_mean"] for part in parts]),
        "cash_fraction": float(np.mean([part["cash_fraction"] for part in parts])),
        "limit_blocked": int(sum(part["limit_blocked"] for part in parts)),
        "liquidity_blocked_share": sum(part["liquidity_blocked"] for part in parts) / attempts,
        "capped_share": sum(part["capped_buys"] for part in parts) / attempts,
        "deployed_share": float(np.mean([part["deployed_share"] for part in parts])),
        "short_deployed_share": float(np.mean([part["short_deployed_share"] for part in parts])),
        "odd_lot_share": float(np.mean([part["odd_lot_share"] for part in parts])),
        "cap_mode": cap_mode,
        "ineligible_share": sum(part["ineligible_skipped"] for part in parts) / short_attempts,
        "borrow_failed_share": sum(part["borrow_failed"] for part in parts) / short_attempts,
        "quota_capped_share": sum(part["quota_capped"] for part in parts) / short_attempts,
        "recalls": float(sum(part["recalls"] for part in parts)),
        "disposition_skipped_share": sum(part["blocked_skipped"] for part in parts) / attempts,
        "forced_covers": float(sum(part["forced_exits"] for part in parts)),
        "stop_exits": float(sum(part["stop_exits"] for part in parts)),
        "stop_gap_exits": float(sum(part["stop_gap_exits"] for part in parts)),
        **execution_tags(config),
        "replicates": len(replicate_annual),
        "short_net_annual_rep_min": min(replicate_annual),
        "short_net_annual_rep_max": max(replicate_annual),
    }
    if execution is not None:
        info.update({
            "borrow_list": execution.borrow_list,
            "failure_multiplier": execution.failure_multiplier,
            "recall_prob": execution.recall_prob,
            "forced_cover": execution.forced_cover or "",
        })
    # Supplement 6: under unit exposure the book is half long, half short on NT$1 of capital;
    # the legs keep their standalone 100% series for the per-leg columns.
    scale = 0.5 if capital_basis(config) == "unit_exposure" else 1.0
    info["capital_basis"] = capital_basis(config)
    return Book(net=scale * (long_net + short_net), gross=scale * (long_gross + short_gross),
                info=info, legs={"long": long_net, "short": short_net},
                gross_legs={"long": long_gross, "short": short_gross})


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


def etf_short_overlay(
    book: Book, data: MarketData, profile: CostProfile, config: dict[str, Any]
) -> Book:
    """Add a short position in an ETF of -beta x capital (protocol 13.21, S-A-ETF).

    ``beta`` is the book's trailing rolling beta on the ETF's own open-to-open return (the
    ``benchmarks`` series named by ``hedge.etf.name``), reset daily. Opening more short pays
    commission, slippage, the ETF transaction tax and the profile's margin-short fee;
    covering pays commission and slippage; the profile's borrow fee accrues daily on the
    notional. The ETF is assumed lendable without a list or quota limit (0050).
    """
    hedge = config.get("hedge", {}).get("etf")
    if hedge is None or data.benchmarks is None or hedge["name"] not in data.benchmarks:
        raise ValueError("the ETF hedge needs hedge.etf.name and the matching benchmarks series")
    tx_cfg = config["tx"]
    series = np.asarray(data.benchmarks[hedge["name"]], dtype=np.float64)
    beta = rolling_beta(
        book.gross, series, window=int(tx_cfg["beta_window"]),
        min_obs=int(tx_cfg["beta_min_observations"]),
    )
    notional = -beta
    growth = np.where(np.isfinite(series), series, 0.0)
    pnl = notional * growth
    previous = np.concatenate(([0.0], notional[:-1]))
    delta = notional - previous
    sold = np.maximum(-delta, 0.0)  # more short sold at this open
    covered = np.maximum(delta, 0.0)  # short covered at this open
    slip = float(profile.slippage[0])
    entry_rate = profile.commission + slip + float(hedge["tax"]) + profile.short_entry_fee
    exit_rate = profile.commission + slip
    fees = sold * entry_rate + covered * exit_rate + np.abs(notional) * profile.borrow_fee_daily
    info = {
        **book.info,
        "hedge": hedge["name"],
        "beta_mean": float(beta.mean()),
        "beta_abs_mean": float(np.abs(beta).mean()),
        "hedge_notional_mean": float(np.abs(notional).mean()),
        "hedge_fees_annual": float(fees.mean()) * int(config["annualization_days"]),
    }
    legs = {**(book.legs or {}), "etf_short": pnl - fees}
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
    benchmarks: dict[str, np.ndarray] | None = None,
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
        comparison: dict[str, float] = {}
        for name, series in (benchmarks or {}).items():
            part = series[mask]
            comparison[f"{name}_annual"] = float(np.nanmean(part)) * days
            comparison[f"{name}_sharpe"] = stats.ratio(part, days)
            comparison[f"{name}_max_drawdown"] = stats.max_drawdown(part)
        rows.append({
            **shared, "scenario": scenario, "period": period, **summary,
            "hit_rate": float(np.mean(book.net[mask] > 0)),
            "index_total_return_annual": float(np.nanmean(index)) * days,
            **book.info, **legs, **comparison,
        })
    return rows


def scenario_books(
    data: MarketData, config: dict[str, Any], profile: CostProfile, capital: Capital,
    *, md_params: tuple[int, int, int], fraction: float, participation_cap: float,
    score_variant: str = "v0", leg_matching: str = "none",
    execution: ExecutionSpec | None = None, cache: dict[tuple, Any] | None = None,
    wk_params: tuple[int, int] | None = None, weekly: bool = True,
) -> dict[str, Book]:
    wanted = tuple(config.get("scenarios", SCENARIOS))
    h, k, buffer = md_params
    books: dict[str, Book] = {}
    md = multi_day_book(data, config, profile, capital, h=h, k=k, buffer=buffer,
                        participation_cap=participation_cap, score_variant=score_variant,
                        leg_matching=leg_matching, execution=execution, cache=cache)
    md_tx = tx_overlay(md, data, profile, config, intraday=False)
    books["md_ls"], books["md_ls_tx"] = md, md_tx
    weekly_cfg = config.get("weekly")
    # S-A has one construction of its own (its score variant, no leg matching), so it is built
    # once, under the combination whose labels match it.
    if (WEEKLY in wanted and weekly and weekly_cfg is not None and wk_params is not None
            and score_variant == str(weekly_cfg.get("score_variant", "v1"))
            and leg_matching == "none"):
        books[WEEKLY] = weekly_long_book(
            data, config, profile, capital, k=wk_params[0], buffer=wk_params[1],
            participation_cap=participation_cap, score_variant=score_variant, cache=cache)
        if WEEKLY_TX in wanted:
            books[WEEKLY_TX] = tx_overlay(books[WEEKLY], data, profile, config, intraday=False)
        if WEEKLY_ETF in wanted:
            books[WEEKLY_ETF] = etf_short_overlay(books[WEEKLY], data, profile, config)
    if any(name in wanted for name in DAY_TRADE):
        dt = day_trade_book(data, config, profile, capital, fraction=fraction,
                            participation_cap=participation_cap)
        dt_tx = tx_overlay(dt, data, profile, config, intraday=True)
        books["dt_ls"], books["dt_ls_tx"] = dt, dt_tx
        books["combo"], books["combo_tx"] = combine(md, dt, config), combine(md_tx, dt_tx, config)
    return {name: books[name] for name in ALL_SCENARIOS if name in wanted and name in books}


def combined_market(
    books: dict[str, dict[str, Book]], config: dict[str, Any]
) -> dict[str, Book] | None:
    spec = config.get("combined_market")
    if not spec or any(market not in books for market in spec["weights"]):
        return None
    out = {}
    scenarios = [name for name in ALL_SCENARIOS if name in next(iter(books.values()))]
    for scenario in scenarios:
        net = sum(float(w) * books[m][scenario].net for m, w in spec["weights"].items())
        gross = sum(float(w) * books[m][scenario].gross for m, w in spec["weights"].items())
        info = {"turnover": sum(float(w) * books[m][scenario].info["turnover"]
                                for m, w in spec["weights"].items())}
        for key in ("beta_mean", "beta_abs_mean", "tx_notional_mean", "names_mean",
                    "holding_mean", "liquidity_blocked_share", "liquidity_capped_share",
                    "deployed_share", "odd_lot_share", "weekly_turnover", "hedge_notional_mean",
                    "hedge_fees_annual", "tx_fees_annual"):
            values = [books[m][scenario].info.get(key) for m in spec["weights"]]
            if all(v is not None for v in values):
                info[key] = sum(float(w) * v for v, w in zip(values, spec["weights"].values(),
                                                              strict=True))
        hedges = {books[m][scenario].info.get("hedge") for m in spec["weights"]}
        if len(hedges) == 1 and None not in hedges:
            info["hedge"] = hedges.pop()
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
                                   shared, benchmarks=data.benchmarks))
    return rows


def select_multi_day(
    data: MarketData, config: dict[str, Any], profile: CostProfile, participation_cap: float,
    *, score_variant: str = "v0",
) -> tuple[list[dict[str, Any]], tuple[int, int, int], dict[str, Any]]:
    """Evaluate the S1 grid on one market and pick (h, K, m) by the selection metric."""
    grid = config["multi_day"]["grid"]
    rule = config["selection"]
    rows, candidates = [], []
    fractional = Capital(FRACTIONAL, None, None)
    for h in grid["score_horizons"]:
        for k in grid["k"]:
            for m in grid["buffer_multiples"]:
                book = multi_day_book(data, config, profile, fractional, h=int(h), k=int(k),
                                      buffer=int(m), participation_cap=participation_cap,
                                      score_variant=score_variant)
                shared = {"model": None, "market": data.market, "profile": profile.name,
                          "slippage_multiplier": profile.slippage_multiplier,
                          "primary_slippage": True, "capital": FRACTIONAL, "capital_twd": None,
                          "stage": "grid", "score_variant": score_variant,
                          "leg_matching": "none"}
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
    chip_data_root: Path | None = None,
    labels: list[str] | None = None,
    multi_day_default: str | None = None,
) -> dict[str, Any]:
    """Run the suite; ``labels`` names the runs in the outputs (default: the model names) so
    two files carrying the same model name, such as F0 and F2, stay apart.

    ``multi_day_default`` names a run of ``config["multi_day"]["fixed"]``: every run without
    its own entry there takes that run's (h, K, m) instead of the P1 grid selection, so all
    runs share one score position and one construction (protocol 13.23 supplement 1 (g)).
    The configuration file is not changed; the report records where each run's parameters
    came from.
    """
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    labels = list(models) if labels is None else list(labels)
    if len(labels) != len(models):
        raise ValueError("Give one --label per --predictions file, or none")
    if len(set(labels)) != len(labels):
        raise ValueError("Run labels must be unique")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (multi_day_default is not None
            and multi_day_default not in config["multi_day"].get("fixed", {})):
        raise ValueError(
            f"--multi-day-default {multi_day_default!r} is not a run of multi_day.fixed")
    profiles = load_profiles(config)
    participation_cap = float(profiles["participation_cap"])
    exec_cfg = config.get("execution")
    specs: dict[str, list[ExecutionSpec]] = {}
    if exec_cfg is not None:
        if chip_data_root is None:
            raise ValueError("The execution layer needs --chip-data-root (tw_chip_data/data)")
        specs = {name: execution_specs(config, name) for name in profiles["profiles"]}
    coverage: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
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
    weekly_cfg = config.get("weekly")
    if WEEKLY in wanted and weekly_cfg is None:
        raise ValueError("scenario wk_long needs a weekly block in the configuration")
    all_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    daily_frames = []
    for path, model, label in zip(predictions, models, labels, strict=True):
        frame = load_predictions(path, model)
        inputs = build_inputs(frame, prices)
        intraday = intraday_returns(frame, prices)
        features = row_features(frame, features_table) if features_table is not None else None
        markets = {
            market: prepare_market(inputs, intraday, market, config, prices, front, features)
            for market in config["markets"]
        }
        if specs:
            for market, data in markets.items():
                borrow = load_borrow_data(chip_data_root, frame, inputs, data.groups,
                                          datasets=exec_cfg.get("datasets"),
                                          disposition=bool(exec_cfg.get("disposition_block")))
                markets[market] = replace(data, borrow=borrow)
                coverage.setdefault(label, {})[market] = summarize_borrow(borrow, data.groups)
                digests.update(borrow.digests)
        chooser = markets[config["selection"]["market"]]
        grid_variant = construction.get("grid_score_variant", "v0")
        if label in fixed or multi_day_default is not None:
            fixed_from = label if label in fixed else multi_day_default
            spec = fixed[fixed_from]
            md_params = (int(spec["h"]), int(spec["k"]), int(spec["buffer"]))
            md_choice = {"h": md_params[0], "k": md_params[1], "buffer": md_params[2],
                         "fixed": True, "fixed_from": fixed_from}
            grid_rows: list[dict[str, Any]] = []
        else:
            grid_rows, md_params, md_choice = select_multi_day(
                chooser, config, selection_profile, participation_cap,
                score_variant=grid_variant)
            md_choice["score_variant"] = grid_variant
        # Extra grid markets are reported, never used for the selection.
        for market in config["multi_day"].get("grid_markets", []):
            if market == config["selection"]["market"] or market not in markets:
                continue
            more, _params, _choice = select_multi_day(
                markets[market], config, selection_profile, participation_cap,
                score_variant=grid_variant)
            grid_rows.extend(more)
        if any(name in wanted for name in DAY_TRADE):
            dt_rows, fraction, dt_choice = select_day_trade(
                chooser, config, selection_profile, participation_cap)
        else:
            dt_rows, fraction, dt_choice = [], float(
                config["day_trade"]["grid"]["fractions"][0]), {"skipped": True}
        wk_rows: list[dict[str, Any]] = []
        wk_params: tuple[int, int] | None = None
        wk_choice: dict[str, Any] = {"skipped": True}
        if WEEKLY in wanted:
            wk_variant = str(weekly_cfg.get("score_variant", "v1"))
            wk_fixed = weekly_cfg.get("fixed", {})
            if label in wk_fixed:
                wk_params = (int(wk_fixed[label]["k"]), int(wk_fixed[label]["buffer"]))
                wk_choice = {"k": wk_params[0], "buffer": wk_params[1], "fixed": True}
            else:
                wk_rows, wk_params, wk_choice = select_weekly(
                    chooser, config, selection_profile, participation_cap,
                    score_variant=wk_variant, capital=Capital(FRACTIONAL, None, None),
                    summarize=summarize_book)
            wk_choice["score_variant"] = wk_variant
            wk_choice["turnover_cap"] = float(weekly_cfg["turnover_cap"])
        for row in grid_rows + dt_rows + wk_rows:
            row["model"] = label
            row.setdefault("score_variant", "v0")
            row.setdefault("leg_matching", "none")
        all_rows.extend(grid_rows + dt_rows + wk_rows)
        selection[label] = {"model": model, "multi_day": md_choice, "day_trade": dt_choice,
                            "weekly": wk_choice}
        caches: dict[tuple, dict[tuple, Any]] = {}
        point = {
            "label": label, "markets": markets, "config": config, "md_params": md_params,
            "fraction": fraction, "participation_cap": participation_cap, "caches": caches,
            "daily_frames": daily_frames, "wk_params": wk_params,
        }
        for (variant, matching), profile in itertools.product(combos, cost_profiles(profiles)):
            primary = is_primary(profile, profiles)
            execution = specs[profile.name][0] if specs else None
            for capital in capital_scenarios(profiles):
                if capital.profile is not None and capital.profile != profile.name:
                    continue
                if capital.twd is not None and not primary:
                    continue
                all_rows.extend(_evaluate_point(
                    profile, capital, variant=variant, matching=matching, execution=execution,
                    primary=primary, stage="final", **point))
        if specs:
            # Sensitivities of the execution layer: primary slippage, fractional capital.
            for (variant, matching), profile in itertools.product(combos,
                                                                  cost_profiles(profiles)):
                if not is_primary(profile, profiles):
                    continue
                for spec in specs[profile.name][1:]:
                    all_rows.extend(_evaluate_point(
                        profile, Capital(FRACTIONAL, None, None), variant=variant,
                        matching=matching, execution=spec, primary=True, stage="execution",
                        weekly=False, **{**point, "daily_frames": None}))
    grid = pd.DataFrame(all_rows)
    verdicts = attractiveness(grid, config)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(output_root / "daily.csv", index=False)
    yearly_summary(daily, annualization_days=int(config["annualization_days"])).to_csv(
        output_root / "yearly.csv", index=False)
    report = {
        "kind": "scenario-suite-retail-vs-institution",
        "config": str(config_path),
        "cost_profiles": str(config["cost_profiles"]),
        "predictions": {label: {"path": str(path), "model": model}
                        for path, model, label in zip(predictions, models, labels, strict=True)},
        "prices": str(prices_path),
        "selection": selection,
        "attractiveness": verdicts,
        "rows": len(grid),
    }
    if specs:
        report["execution"] = {
            "chip_data_root": str(chip_data_root), "config": exec_cfg,
            "points": {name: [{"failure_multiplier": s.failure_multiplier,
                               "recall_prob": s.recall_prob, "primary": s.primary,
                               "forced_cover": s.forced_cover}
                              for s in items] for name, items in specs.items()},
            "coverage": coverage, "partition_digests": digests,
        }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, grid, profiles),
                                            encoding="utf-8")
    return report


def _evaluate_point(
    profile: CostProfile,
    capital: Capital,
    *,
    label: str,
    markets: dict[str, MarketData],
    config: dict[str, Any],
    md_params: tuple[int, int, int],
    fraction: float,
    participation_cap: float,
    variant: str,
    matching: str,
    execution: ExecutionSpec | None,
    caches: dict[tuple, dict[tuple, Any]],
    primary: bool,
    stage: str,
    daily_frames: list[pd.DataFrame] | None,
    wk_params: tuple[int, int] | None = None,
    weekly: bool = True,
) -> list[dict[str, Any]]:
    """Every scenario of one (profile, capital, construction, execution point) on every market
    and on the combined market; appends the daily series of the primary fractional cells."""
    extra = {"score_variant": variant, "leg_matching": matching}
    if execution is not None:
        extra.update({"failure_multiplier": execution.failure_multiplier,
                      "recall_prob": execution.recall_prob})
    rows: list[dict[str, Any]] = []
    books_by_market = {}
    for market, data in markets.items():
        books = scenario_books(
            data, config, profile, capital, md_params=md_params, fraction=fraction,
            participation_cap=participation_cap, score_variant=variant,
            leg_matching=matching, execution=execution,
            cache=caches.setdefault((market, variant, matching), {}),
            wk_params=wk_params, weekly=weekly,
        )
        books_by_market[market] = books
        market_rows = _profile_rows(label, market, profile, capital, books, data, config,
                                    primary=primary, extra=extra)
        for row in market_rows:
            row["stage"] = stage
        rows.extend(market_rows)
        if daily_frames is not None and primary and capital.twd is None:
            daily_frames.append(_daily_frame(label, market, profile, books, data, extra))
    combined = combined_market(books_by_market, config)
    if combined is not None:
        name = config["combined_market"]["name"]
        first = next(iter(markets.values()))
        index_total = np.nanmean(
            np.vstack([markets[m].index_total_return for m in markets]), axis=0)
        shared = {"model": label, "market": name, "profile": profile.name,
                  "slippage_multiplier": profile.slippage_multiplier,
                  "primary_slippage": primary, "capital": capital.name,
                  "capital_twd": capital.twd, "stage": stage, **extra}
        benchmarks = combined_benchmarks(markets, config)
        for scenario, book in combined.items():
            rows.extend(summarize_book(book, scenario, config, first.masks, index_total,
                                       shared, benchmarks=benchmarks))
        if daily_frames is not None and primary and capital.twd is None:
            daily_frames.append(_daily_frame(label, name, profile, combined, first, extra,
                                             benchmarks=benchmarks))
    return rows


def combined_benchmarks(
    markets: dict[str, MarketData], config: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    """The comparison series of the combined market: the weighted mean of the markets'."""
    spec = config.get("combined_market")
    if not spec or any(markets[m].benchmarks is None for m in spec["weights"]):
        return None
    names = set.intersection(*(set(markets[m].benchmarks) for m in spec["weights"]))
    return {
        name: sum(float(w) * markets[m].benchmarks[name] for m, w in spec["weights"].items())
        for name in sorted(names)
    }


def yearly_summary(daily: pd.DataFrame, *, annualization_days: int) -> pd.DataFrame:
    """Per calendar year of the interval start: compounded net and gross return, annualized
    Sharpe of the daily net series, maximum drawdown, hit rate and the number of intervals, for
    every daily series (model, market, profile, scenario and construction)."""
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["interval_start"]).dt.year
    keys = [c for c in ("model", "market", "profile", "scenario", "score_variant", "leg_matching")
            if c in frame.columns]
    rows = []
    for key, part in frame.groupby([*keys, "year"], sort=True):
        net = part["net"].to_numpy(dtype=np.float64)
        gross = part["gross"].to_numpy(dtype=np.float64)
        record = dict(zip([*keys, "year"], key, strict=True))
        record.update({
            "days": int(np.isfinite(net).sum()),
            "net_return": float(np.prod(1.0 + np.nan_to_num(net)) - 1.0),
            "gross_return": float(np.prod(1.0 + np.nan_to_num(gross)) - 1.0),
            "net_mean_annual": float(np.nanmean(net)) * annualization_days,
            "sharpe": stats.ratio(net, annualization_days),
            "max_drawdown": stats.max_drawdown(net),
            "hit_rate": float(np.mean(net[np.isfinite(net)] > 0)) if net.size else float("nan"),
            "index_total_return": float(
                np.prod(1.0 + np.nan_to_num(part["index_total_return"].to_numpy())) - 1.0),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def _daily_frame(label: str, market: str, profile: CostProfile, books: dict[str, Book],
                 data: MarketData, extra: dict[str, Any] | None = None,
                 benchmarks: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    dates = data.inputs.dates[1:].strftime("%Y-%m-%d")
    comparison = data.benchmarks if benchmarks is None else benchmarks
    frames = []
    for scenario, book in books.items():
        frames.append(pd.DataFrame({
            "model": label, "market": market, "profile": profile.name, "scenario": scenario,
            "interval_start": dates, "net": book.net, "gross": book.gross,
            "index_total_return": data.index_total_return, **(extra or {}),
            **(comparison or {}),
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
        line = (f"- {model}: h = {md.get('h')}, K = {md.get('k')}, m = {md.get('buffer')}; "
                f"q = {dt.get('fraction')}")
        wk = choice.get("weekly") or {}
        if "k" in wk:
            line += (f"; S-A K = {wk['k']}, m = {wk['buffer']}, 週換手上限 / weekly cap "
                     f"{wk.get('turnover_cap')}")
        lines.append(line)
    lines.append("")
    lines.extend(header)
    order = {name: i for i, name in enumerate(ALL_SCENARIOS)}
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
    if "failure_multiplier" in grid.columns and "execution" in report:
        lines.append("## 執行不確定性（空頭腿）/ Execution layer (short book)")
        lines.append("")
        lines.append("| 模型 | 市場 | 成本檔 | 借不到倍率 | 召回機率 | 期間 | 年化淨報酬 "
                     "| 95% 區間 | Sharpe | 名單外 | 借不到 | 限額封頂 | 召回次數 | 空頭重複極差 "
                     "| 處置跳過 | 強制回補 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
                     "| --- | --- | --- | --- |")
        rows = grid[grid["stage"].isin(["final", "execution"]) & grid["primary_slippage"]
                    & (grid["capital"] == FRACTIONAL) & (grid["scenario"] == "md_ls")
                    & grid["failure_multiplier"].notna()]
        keys = ["model", "market", "profile", "failure_multiplier", "recall_prob", "period"]
        for row in rows.sort_values(keys).itertuples():
            pct = runner.percent
            lines.append(
                f"| {row.model} | {row.market} | {row.profile} | {row.failure_multiplier:g} | "
                f"{row.recall_prob:g} | {row.period} | {pct(row.net_return_annual)} | "
                f"{_cell_ci(row)} | {runner.number(row.sharpe)} | {pct(row.ineligible_share)} | "
                f"{pct(row.borrow_failed_share)} | {pct(row.quota_capped_share)} | "
                f"{row.recalls:.0f} | {pct(row.short_net_annual_rep_min)} .. "
                f"{pct(row.short_net_annual_rep_max)} | "
                f"{pct(getattr(row, 'disposition_skipped_share', float('nan')))} | "
                f"{getattr(row, 'forced_covers', float('nan')):.0f} |"
            )
        lines.append("")
    weekly = final[final["scenario"].isin([WEEKLY, WEEKLY_TX, WEEKLY_ETF])]
    if not weekly.empty and "universe_ew_annual" in weekly.columns:
        lines.append("## S-A 對照 / S-A comparison columns")
        lines.append("")
        lines.append("| 情境 | 模型 | 市場 | 成本檔 | 期間 | 年化淨報酬 | 等權宇宙 | 0050 "
                     "| TAIEX 報酬指數 | 最大回撤 | 0050 回撤 | 週換手 | 再平衡次數 "
                     "| 觸頂次數 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
                     "| --- | --- |")
        etf = [c for c in weekly.columns if c.startswith("etf_") and c.endswith("_annual")]
        etf_name = etf[0][: -len("_annual")] if etf else None
        for row in weekly.sort_values(["scenario", "model", "market", "profile",
                                       "period"]).itertuples():
            record = row._asdict()
            pct = runner.percent
            etf_annual = pct(record[f"{etf_name}_annual"]) if etf_name else "n/a"
            etf_dd = pct(record[f"{etf_name}_max_drawdown"]) if etf_name else "n/a"
            lines.append(
                f"| {row.scenario} | {row.model} | {row.market} | {row.profile} | {row.period} | "
                f"{pct(row.net_return_annual)} | {pct(record['universe_ew_annual'])} | "
                f"{etf_annual} | {pct(row.index_total_return_annual)} | "
                f"{pct(row.max_drawdown)} | {etf_dd} | {pct(record['weekly_turnover'])} | "
                f"{record['rebalances']:.0f} | {record['cap_hits']:.0f} |"
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
    parser.add_argument("--label", action="append",
                        help="run name in the outputs for the matching --predictions file "
                             "(default: the model name); needed when two files share a model")
    parser.add_argument("--prices", type=Path, required=True, help="raw market.parquet")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--chip-data-root", type=Path,
                        default=os.environ.get("TW_CHIP_DATA_ROOT"),
                        help="tw_chip_data/data (curated layer) for the execution layer; "
                             "defaults to $TW_CHIP_DATA_ROOT")
    parser.add_argument("--multi-day-default",
                        help="a run named in the configuration's multi_day.fixed; every run "
                             "without its own entry takes its (h, K, m) instead of the P1 "
                             "grid selection (protocol 13.23 supplement 1 (g))")
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "scenario-suite")
    report = run(args.config, args.predictions, args.model, args.prices, output_root,
                 artifacts=args.artifacts_root,
                 chip_data_root=Path(args.chip_data_root) if args.chip_data_root else None,
                 labels=args.label, multi_day_default=args.multi_day_default)
    print(json.dumps({"output_root": str(output_root), "selection": report["selection"],
                      "attractiveness": report["attractiveness"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
