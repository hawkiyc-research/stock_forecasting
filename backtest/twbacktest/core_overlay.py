"""0050 core plus orthogonalized stock long-short overlay (protocol 13.17 supplement).

The market exposure is held directly, the alpha comes from the multi-day long-short books of
protocol 13.19 supplement 2 (v1 score, unmatched, fixed band parameters) with the short book
under supplement 4's primary execution assumptions (borrowable lists, lending quota, borrow
failure draws, tiered fees, averaged random replicates). Two ways of holding the exposure:

* **Variant A** (primary): ``w`` of the capital in 0050 held throughout, each overlay leg at
  ``L`` times the capital; the long leg is paid from the remaining cash ``1 - w`` and, beyond
  that, from a loan secured on the 0050 holding at ``r_fin`` a year; the short leg's proceeds
  stay as collateral and earn nothing. Daily return
  ``w * r_0050 + L * (r_long + r_short) - max(L - (1 - w), 0) * r_fin / 252``, where the leg
  series already carry their trading costs and borrow fees.
* **Variant B** (secondary): a long TX position of one capital, marked settlement to settlement
  with the roll costs of the profile, on a cash margin of 15% of the capital; the remaining 85%
  runs the long-short (each leg 0.85 of the capital).

The benchmark is 0050 bought at the first open of the sample and sold at the last, dividends
reinvested (adjusted prices), with the profile's commission, slippage and the ETF transaction
tax. Variant A's ``(w, L)`` is selected on P1 by the information ratio of "strategy minus
0050" at the primary loan rate and then fixed; H-G asks whether the P1 + P2 difference's
95% interval clears 0 and the P2 difference is positive, per cost profile, informationally
only until P3. Every table lists retail and institution side by side.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import positions, runner, stats, taifex
from twbacktest.borrow_execution import ExecutionSpec, execution_specs, load_borrow_data
from twbacktest.exposure_features import row_features, trailing_features
from twbacktest.scenario_suite import (
    CONFIG_DIR,
    DEFAULT_ARTIFACTS,
    Book,
    Capital,
    CostProfile,
    MarketData,
    _interval_from_dates,
    cost_profiles,
    intraday_returns,
    is_primary,
    load_profiles,
    multi_day_book,
    prepare_market,
    tx_side_rate,
)
from twbacktest.walkforward_data import build_inputs, load_predictions, load_prices

DEFAULT_CONFIG = CONFIG_DIR / "core_overlay.json"
FRACTIONAL = "fractional"
VARIANT_LABELS = {
    "A": ("變體 A：0050 核心 + 現股多空（質押借款）",
          "Variant A: 0050 core + stock L/S (secured loan)"),
    "B": ("變體 B：台指期提供 β + 現股多空", "Variant B: β from TX + stock L/S"),
    "benchmark": ("0050 買進持有", "0050 buy and hold"),
}


# --------------------------------------------------------------------------- configuration


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The overlay configuration and the scenario-suite configuration it builds on."""
    config = json.loads(path.read_text(encoding="utf-8"))
    suite_path = Path(config["suite_config"])
    suite_path = suite_path if suite_path.is_absolute() else CONFIG_DIR / suite_path
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if "execution" not in suite:
        raise ValueError("core_overlay needs a suite configuration with an execution block")
    return config, suite


# --------------------------------------------------------------------------- 0050 core


def etf_growth(prices: pd.DataFrame, dates: pd.DatetimeIndex, symbol: str) -> np.ndarray:
    """Open-to-open total return of the ETF per interval (adjusted next-session opens)."""
    rows = prices[prices["symbol"] == symbol].drop_duplicates("date").set_index("date")
    series = rows["open_1"].sort_index()
    start = _interval_from_dates(series, dates, 0)
    end = _interval_from_dates(series, dates, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        growth = end / start - 1.0
    missing = int((~np.isfinite(growth)).sum())
    if missing > 0.05 * growth.size:
        raise ValueError(f"{symbol} is missing on {missing} of {growth.size} intervals")
    return np.where(np.isfinite(growth), growth, 0.0)


def etf_rates(profile: CostProfile, etf_tax: float) -> tuple[float, float]:
    """(buy, sell) cost rates of the ETF: commission and TWSE slippage, tax on the sale."""
    slip = float(profile.slippage[0])
    return profile.commission + slip, profile.commission + etf_tax + slip


def buy_and_hold(growth: np.ndarray, buy_rate: float, sell_rate: float) -> np.ndarray:
    """Daily returns of one position bought at the first open and sold at the last."""
    net = growth.astype(np.float64).copy()
    net[0] = (1.0 - buy_rate) * (1.0 + growth[0]) - 1.0
    net[-1] = (1.0 + net[-1]) * (1.0 - sell_rate) - 1.0
    return net


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Series:
    net: np.ndarray
    gross: np.ndarray
    info: dict[str, Any]


def variant_a(
    core: np.ndarray, core_rates: tuple[float, float], legs: dict[str, np.ndarray],
    legs_gross: dict[str, np.ndarray], *, w: float, leverage: float, loan_rate: float,
    ltv_max: float, annualization_days: int,
) -> Series:
    """0050 core at weight ``w`` plus the overlay at ``leverage`` per leg, loan beyond the cash."""
    loan = max(leverage - (1.0 - w), 0.0)
    daily_interest = loan * loan_rate / annualization_days
    core_net = buy_and_hold(core, *core_rates)
    overlay_net = leverage * (legs["long"] + legs["short"])
    overlay_gross = leverage * (legs_gross["long"] + legs_gross["short"])
    net = w * core_net + overlay_net - daily_interest
    gross = w * core + overlay_gross
    # Collateral check: the loan against the marked 0050 holding (no forced action, counted).
    holding = w * np.cumprod(1.0 + core)
    shortfall = (loan > ltv_max * holding) if loan > 0 else np.zeros(core.size, dtype=bool)
    return Series(net=net, gross=gross, info={
        "variant": "A", "w": w, "leverage": leverage, "loan_rate": loan_rate, "loan": loan,
        "loan_interest_annual": daily_interest * annualization_days,
        "collateral_shortfall_days": int(shortfall.sum()),
        "collateral_shortfall_share": float(shortfall.mean()),
        "ltv_peak": float(np.max(loan / holding)) if loan > 0 else 0.0,
    })


def variant_b(
    tx_growth: np.ndarray, tx_roll: np.ndarray, tx_rate: np.ndarray,
    legs: dict[str, np.ndarray], legs_gross: dict[str, np.ndarray], *, margin_fraction: float,
    maintenance_fraction: float, overlay_fraction: float, tx_notional: float,
    annualization_days: int,
) -> Series:
    """Long TX of ``tx_notional`` capitals on a cash margin plus the overlay on the rest."""
    growth = np.where(np.isfinite(tx_growth), tx_growth, 0.0)
    fees = 2.0 * tx_roll * tx_rate
    fees[0] += tx_rate[0]
    fees[-1] += tx_rate[-1]
    tx_net = tx_notional * (growth - fees)
    tx_gross = tx_notional * growth
    net = tx_net + overlay_fraction * (legs["long"] + legs["short"])
    gross = tx_gross + overlay_fraction * (legs_gross["long"] + legs_gross["short"])
    # Margin account: the reserve absorbs the TX P&L; below maintenance it is refilled from
    # the overlay book (counted as a margin call, no further effect on the return series).
    reserve = margin_fraction
    calls = 0
    for value in tx_net:
        reserve += value
        if reserve < maintenance_fraction:
            calls += 1
            reserve = margin_fraction
    return Series(net=net, gross=gross, info={
        "variant": "B", "tx_notional": tx_notional, "margin_fraction": margin_fraction,
        "overlay_fraction": overlay_fraction, "margin_calls": calls,
        "tx_fees_annual": float(np.mean(tx_notional * fees)) * annualization_days,
        "tx_net_annual": float(np.mean(tx_net)) * annualization_days,
    })


# --------------------------------------------------------------------------- books


def combine_books(books: dict[str, Book], weights: dict[str, float]) -> Book:
    """Weighted average of the markets' books (the TW combination of the suite)."""
    legs = {
        name: sum(float(weights[m]) * books[m].legs[name] for m in weights)  # type: ignore[index]
        for name in ("long", "short")
    }
    gross_legs = {
        name: sum(float(weights[m]) * books[m].gross_legs[name] for m in weights)  # type: ignore[index]
        for name in ("long", "short")
    }
    info: dict[str, Any] = {}
    for key in ("turnover", "holding_mean", "deployed_share", "short_deployed_share",
                "ineligible_share", "borrow_failed_share", "quota_capped_share",
                "liquidity_blocked_share", "capped_share",
                "short_net_annual_rep_min", "short_net_annual_rep_max"):
        values = [books[m].info.get(key) for m in weights]
        if all(v is not None and np.isfinite(v) for v in values):
            info[key] = sum(float(weights[m]) * v for m, v in zip(weights, values, strict=True))
    return Book(net=legs["long"] + legs["short"], gross=gross_legs["long"] + gross_legs["short"],
                info=info, legs=legs, gross_legs=gross_legs)


def leg_book(
    data: MarketData, suite: dict[str, Any], profile: CostProfile, capital: Capital,
    md_params: tuple[int, int, int], participation_cap: float, execution: ExecutionSpec,
    cache: dict[tuple, Any], *, odd_lots: bool = False,
) -> Book:
    """The v1 unmatched long-short book of one market (legs net and gross)."""
    h, k, buffer = md_params
    return multi_day_book(
        data, suite, profile, capital, h=h, k=k, buffer=buffer,
        participation_cap=participation_cap, score_variant="v1", leg_matching="none",
        execution=execution, cache=cache, odd_lots=odd_lots,
    )


# --------------------------------------------------------------------------- evaluation


def summarize(
    series: Series, benchmark: np.ndarray, masks: dict[str, np.ndarray], index_total: np.ndarray,
    *, annualization_days: int, bootstrap: dict[str, Any], shared: dict[str, Any],
    book_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    turnover = float((book_info or {}).get("turnover", float("nan")))
    for period, mask in masks.items():
        if not mask.any():
            continue
        summary = positions.summarize(
            series.net[mask], series.gross[mask], benchmark[mask], turnover=turnover,
            annualization_days=annualization_days, bootstrap=bootstrap,
        )
        bench = benchmark[mask]
        rows.append({
            **shared, "period": period, **summary,
            "benchmark_net_annual": float(np.nanmean(bench)) * annualization_days,
            "benchmark_cagr": stats.compound_annual(bench, annualization_days),
            "benchmark_sharpe": stats.ratio(bench, annualization_days),
            "benchmark_max_drawdown": stats.max_drawdown(bench),
            "sharpe_diff": summary["sharpe"] - stats.ratio(bench, annualization_days),
            "drawdown_diff": summary["max_drawdown"] - stats.max_drawdown(bench),
            "index_total_return_annual": float(np.nanmean(index_total[mask])) * annualization_days,
            "excess_over_index_annual": (
                float(np.nanmean(series.net[mask])) - float(np.nanmean(index_total[mask]))
            ) * annualization_days,
            **series.info,
            **{k: v for k, v in (book_info or {}).items() if not isinstance(v, np.ndarray)},
        })
    return rows


def _rows_for(
    label: str, market: str, profile: CostProfile, capital: Capital, book: Book, core: np.ndarray,
    core_rates: tuple[float, float], tx: tuple[np.ndarray, np.ndarray, np.ndarray],
    masks: dict[str, np.ndarray], index_total: np.ndarray, config: dict[str, Any],
    days: int, *, grid_a: list[tuple[float, float, float]], include_b: bool,
) -> tuple[list[dict[str, Any]], dict[str, Series]]:
    """All variant rows of one (model, market, profile, capital) and the series behind them."""
    bootstrap = config["bootstrap"]
    shared = {"model": label, "market": market, "profile": profile.name,
              "capital": capital.name, "capital_twd": capital.twd}
    legs_gross = book.gross_legs
    if legs_gross is None or book.legs is None:
        raise ValueError("the overlay needs a book with net and gross legs")
    benchmark = buy_and_hold(core, *core_rates)
    rows: list[dict[str, Any]] = []
    series_out: dict[str, Series] = {}
    ltv_max = float(config["core"]["ltv_max"])
    for w, leverage, loan_rate in grid_a:
        series = variant_a(core, core_rates, book.legs, legs_gross, w=w, leverage=leverage,
                           loan_rate=loan_rate, ltv_max=ltv_max, annualization_days=days)
        series_out[f"A|{w:g}|{leverage:g}|{loan_rate:g}"] = series
        rows.extend(summarize(series, benchmark, masks, index_total, annualization_days=days,
                              bootstrap=bootstrap, shared=shared, book_info=book.info))
    if include_b:
        spec_b = config["variant_b"]
        series = variant_b(*tx, book.legs, legs_gross,
                           margin_fraction=float(spec_b["tx_margin_fraction"]),
                           maintenance_fraction=float(spec_b["tx_maintenance_fraction"]),
                           overlay_fraction=float(spec_b["overlay_fraction"]),
                           tx_notional=float(spec_b["tx_notional"]), annualization_days=days)
        series_out["B"] = series
        rows.extend(summarize(series, benchmark, masks, index_total, annualization_days=days,
                              bootstrap=bootstrap, shared=shared, book_info=book.info))
    bench_series = Series(net=benchmark, gross=core, info={"variant": "benchmark"})
    series_out["benchmark"] = bench_series
    rows.extend(summarize(bench_series, benchmark, masks, index_total, annualization_days=days,
                          bootstrap=bootstrap, shared=shared))
    return rows, series_out


def select_variant_a(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Per (model, market, profile): the (w, L) with the best P1 IR at the primary loan rate."""
    sel = config["selection"]
    primary_rate = float(config["variant_a"]["primary_loan_rate"])
    base = rows[(rows["variant"] == "A") & (rows["capital"] == FRACTIONAL)
                & (rows["period"] == sel["period"]) & (rows["loan_rate"] == primary_rate)]
    out: dict[str, Any] = {}
    for (model, market, profile), part in base.groupby(["model", "market", "profile"]):
        best = part.sort_values(sel["metric"], ascending=False).iloc[0]
        out.setdefault(model, {}).setdefault(market, {})[profile] = {
            "w": float(best["w"]), "leverage": float(best["leverage"]),
            "loan_rate": primary_rate, sel["metric"]: float(best[sel["metric"]]),
        }
    return out


def decide(rows: pd.DataFrame, selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """H-G per (model, market, profile) for the selected variant A and for variant B."""
    rule = config["decision"]
    out: dict[str, Any] = {}
    for model, markets in selection.items():
        for market, profiles in markets.items():
            for profile, chosen in profiles.items():
                base = rows[(rows["model"] == model) & (rows["market"] == market)
                            & (rows["profile"] == profile) & (rows["capital"] == FRACTIONAL)]
                picks = {
                    "A": base[(base["variant"] == "A") & (base["w"] == chosen["w"])
                              & (base["leverage"] == chosen["leverage"])
                              & (base["loan_rate"] == chosen["loan_rate"])],
                    "B": base[base["variant"] == "B"],
                }
                verdicts = {}
                for name, part in picks.items():
                    decision = part[part["period"] == rule["period"]]
                    confirm = part[part["period"] == rule["require_positive_period"]]
                    if decision.empty:
                        continue
                    row = decision.iloc[0]
                    lower = float(row["net_excess_annual_low"])
                    confirm_diff = (float(confirm.iloc[0]["net_excess_annual"])
                                    if not confirm.empty else float("nan"))
                    verdicts[name] = {
                        "net_excess_annual": float(row["net_excess_annual"]),
                        "net_excess_annual_low": lower,
                        "net_excess_annual_high": float(row["net_excess_annual_high"]),
                        "lower_bound_ok": bool(np.isfinite(lower) and lower > 0),
                        "confirmation_net_excess_annual": confirm_diff,
                        "confirmation_ok": bool(np.isfinite(confirm_diff) and confirm_diff > 0),
                        "net_return_annual": float(row["net_return_annual"]),
                        "benchmark_net_annual": float(row["benchmark_net_annual"]),
                        "sharpe": float(row["sharpe"]),
                        "benchmark_sharpe": float(row["benchmark_sharpe"]),
                    }
                    verdicts[name]["passes"] = (verdicts[name]["lower_bound_ok"]
                                                and verdicts[name]["confirmation_ok"])
                out.setdefault(model, {}).setdefault(market, {})[profile] = verdicts
    return out


# --------------------------------------------------------------------------- run


def _market_books(
    markets: dict[str, MarketData], suite: dict[str, Any], profile: CostProfile,
    capital_by_market: dict[str, Capital], md_params: tuple[int, int, int],
    participation_cap: float, execution: ExecutionSpec, caches: dict[tuple, dict],
    *, odd_lots: bool,
) -> dict[str, Book]:
    books = {}
    for market, data in markets.items():
        cache = caches.setdefault((market,), {})
        books[market] = leg_book(data, suite, profile, capital_by_market[market], md_params,
                                 participation_cap, execution, cache, odd_lots=odd_lots)
    return books


def run(
    config_path: Path,
    predictions: list[Path],
    models: list[str],
    prices_path: Path,
    output_root: Path,
    *,
    artifacts: Path,
    chip_data_root: Path,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    config, suite = load_config(config_path)
    profiles = load_profiles(suite)
    participation_cap = float(profiles["participation_cap"])
    days = int(suite["annualization_days"])
    prices = load_prices(prices_path,
                         liquidity_lookback=int(suite["universe"]["liquidity_lookback_sessions"]))
    tx_cfg = suite["tx"]
    front = taifex.continuous_front_month(
        taifex.load_contract(artifacts / tx_cfg["root"], tx_cfg["contract"]), tx_cfg["contract"])
    features_table = trailing_features(prices_path)
    specs = {name: execution_specs(suite, name)[0] for name in profiles["profiles"]}
    weights = {m: float(w) for m, w in suite["combined_market"]["weights"].items()}
    combined_name = suite["combined_market"]["name"]
    wanted_markets = list(config["markets"])
    fixed = suite["multi_day"]["fixed"]
    core_cfg = config["core"]
    va = config["variant_a"]
    all_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    for path, label in zip(predictions, models, strict=True):
        if label not in fixed:
            raise ValueError(f"{label!r} has no fixed band parameters in the suite configuration")
        spec = fixed[label]
        md_params = (int(spec["h"]), int(spec["k"]), int(spec["buffer"]))
        frame = load_predictions(path, label)
        inputs = build_inputs(frame, prices)
        intraday = intraday_returns(frame, prices)
        features = row_features(frame, features_table)
        markets = {}
        for market in suite["markets"]:
            data = prepare_market(inputs, intraday, market, suite, prices, front, features)
            borrow = load_borrow_data(chip_data_root, frame, inputs, data.groups,
                                      datasets=suite["execution"].get("datasets"))
            markets[market] = replace(data, borrow=borrow)
        first = next(iter(markets.values()))
        core = etf_growth(prices, inputs.dates, str(core_cfg["symbol"]))
        index_total = {m: d.index_total_return for m, d in markets.items()}
        index_total[combined_name] = np.nanmean(np.vstack(list(index_total.values())), axis=0)
        caches: dict[tuple, dict] = {}
        coverage[label] = {m: {"sbl_eligible_share": float(d.borrow.sbl_eligible[d.mask].mean())}
                           for m, d in markets.items()}
        for profile in cost_profiles(profiles):
            if not is_primary(profile, profiles):
                continue
            execution = specs[profile.name]
            core_rates = etf_rates(profile, float(core_cfg["etf_tax"]))
            tx = (first.futures.growth, first.futures.roll,
                  tx_side_rate(first, profile, float(tx_cfg["multiplier"])))
            leverages = [float(x) for x in va["leverage"][profile.name]]
            grid_a = [(float(w), lev, float(r)) for w, lev, r in
                      itertools.product(va["weights"], leverages, va["loan_rates"])]
            fractional = Capital(FRACTIONAL, None, None)
            books = _market_books(markets, suite, profile, dict.fromkeys(markets, fractional),
                                  md_params, participation_cap, execution, caches, odd_lots=False)
            books[combined_name] = combine_books(books, weights)
            for market in wanted_markets:
                rows, series = _rows_for(
                    label, market, profile, fractional, books[market], core, core_rates, tx,
                    first.masks, index_total[market], config, days, grid_a=grid_a, include_b=True)
                all_rows.extend(rows)
                if market == config["primary_market"]:
                    daily_frames.append(_daily_frame(label, market, profile, series, inputs.dates,
                                                     index_total[market]))
        # Capital scenarios on the primary market with the selected (w, L).
        grid = pd.DataFrame(all_rows)
        selection = select_variant_a(grid[grid["model"] == label], config)
        for profile in cost_profiles(profiles):
            if not is_primary(profile, profiles):
                continue
            chosen = selection[label][config["capital_market"]][profile.name]
            execution = specs[profile.name]
            core_rates = etf_rates(profile, float(core_cfg["etf_tax"]))
            tx = (first.futures.growth, first.futures.roll,
                  tx_side_rate(first, profile, float(tx_cfg["multiplier"])))
            for item in config["capital_scenarios"].get(profile.name, []):
                total = float(item["twd"])
                odd = bool(item.get("odd_lots", False))
                leg_twd = {m: chosen["leverage"] * total * weights[m] for m in markets}
                capitals = {m: Capital(str(item["name"]), leg_twd[m], profile.name)
                            for m in markets}
                books = _market_books(markets, suite, profile, capitals, md_params,
                                      participation_cap, execution, caches, odd_lots=odd)
                combined = combine_books(books, weights)
                capital = Capital(str(item["name"]), total, profile.name)
                rows, _series = _rows_for(
                    label, config["capital_market"], profile, capital, combined, core, core_rates,
                    tx, first.masks, index_total[config["capital_market"]], config, days,
                    grid_a=[(chosen["w"], chosen["leverage"], chosen["loan_rate"])],
                    include_b=False)
                for row in rows:
                    row["odd_lots"] = odd
                all_rows.extend(rows)
    grid = pd.DataFrame(all_rows)
    grid["variant"] = grid["variant"].astype(str)
    selection = select_variant_a(grid, config)
    decisions = decide(grid, selection, config)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "core-overlay-0050-plus-orthogonalized-long-short",
        "config": str(config_path), "suite_config": str(config["suite_config"]),
        "predictions": {label: str(path) for path, label in zip(predictions, models, strict=True)},
        "prices": str(prices_path), "chip_data_root": str(chip_data_root),
        "selection": selection, "decisions": decisions, "coverage": coverage, "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(report, grid, config),
                                            encoding="utf-8")
    return report


def _daily_frame(label: str, market: str, profile: CostProfile, series: dict[str, Series],
                 dates: pd.DatetimeIndex, index_total: np.ndarray) -> pd.DataFrame:
    stamps = dates[1:].strftime("%Y-%m-%d")
    frames = []
    for key, item in series.items():
        frames.append(pd.DataFrame({
            "model": label, "market": market, "profile": profile.name, "series": key,
            "interval_start": stamps, "net": item.net, "gross": item.gross,
            "index_total_return": index_total,
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- summary


def _label(row, lang: str) -> str:
    variant = str(row.variant)
    if variant == "A":
        return f"A w={row.w:g} L={row.leverage:g} r={100 * row.loan_rate:g}%"
    if variant == "B":
        return "B"
    return VARIANT_LABELS["benchmark"][1 if lang == "en" else 0]


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame, config: dict[str, Any]) -> str:
    lines = ["# 0050 核心加正交化多空疊加摘要 / 0050 core plus orthogonalized L/S summary", ""]
    lines.append("## 選擇與 H-G / Selection and H-G")
    lines.append("")
    for model, markets in report["decisions"].items():
        for market, profiles in markets.items():
            for profile, verdicts in profiles.items():
                chosen = report["selection"][model][market][profile]
                for name, v in verdicts.items():
                    word = "通過 / passes" if v["passes"] else "未通過 / fails"
                    params = (f"w={chosen['w']:g} L={chosen['leverage']:g}" if name == "A" else "")
                    lines.append(
                        f"- {model} {market} {profile} 變體 {name} {params}: {word} "
                        f"(策略 {runner.percent(v['net_return_annual'])} vs 0050 "
                        f"{runner.percent(v['benchmark_net_annual'])}; 差 "
                        f"{runner.percent(v['net_excess_annual'])} "
                        f"[{runner.percent(v['net_excess_annual_low'])}, "
                        f"{runner.percent(v['net_excess_annual_high'])}]; P2 差 "
                        f"{runner.percent(v['confirmation_net_excess_annual'])})")
    lines.append("")
    lines.append("## 分數位置全表 / Fractional rows")
    lines.append("")
    lines.append("| 模型 | 市場 | 成本檔 | 變體 | 期間 | 年化淨報酬 | 0050 | 差 | 95% 區間 "
                 "| Sharpe | 0050 Sharpe | 最大回撤 | 0050 回撤 | 融資利息 | 擔保不足日 | 追繳 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
                 "| --- | --- | --- |")
    part = grid[grid["capital"] == FRACTIONAL].copy()
    part["po"] = part["period"].map({"P1": 0, "P2": 1, "P1+P2": 2})

    def cell(value, fmt) -> str:
        return fmt(value) if pd.notna(value) else "—"

    def count(value) -> str:
        return str(int(value))

    for row in part.sort_values(["model", "market", "profile", "variant", "w", "leverage",
                                 "loan_rate", "po"]).itertuples():
        pct = runner.percent
        lines.append(
            f"| {row.model} | {row.market} | {row.profile} | {_label(row, 'zh')} | {row.period} | "
            f"{pct(row.net_return_annual)} | {pct(row.benchmark_net_annual)} | "
            f"{pct(row.net_excess_annual)} | [{pct(row.net_excess_annual_low)}, "
            f"{pct(row.net_excess_annual_high)}] | {runner.number(row.sharpe)} | "
            f"{runner.number(row.benchmark_sharpe)} | {pct(row.max_drawdown)} | "
            f"{pct(row.benchmark_max_drawdown)} | "
            f"{cell(getattr(row, 'loan_interest_annual', float('nan')), pct)} | "
            f"{cell(getattr(row, 'collateral_shortfall_days', float('nan')), count)} | "
            f"{cell(getattr(row, 'margin_calls', float('nan')), count)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--prices", type=Path, required=True, help="raw market.parquet")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--chip-data-root", type=Path,
                        default=os.environ.get("TW_CHIP_DATA_ROOT"),
                        help="tw_chip_data/data (curated layer); defaults to $TW_CHIP_DATA_ROOT")
    args = parser.parse_args(argv)
    if args.chip_data_root is None:
        parser.error("--chip-data-root (or TW_CHIP_DATA_ROOT) is required")
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "core-overlay")
    report = run(args.config, args.predictions, args.model, args.prices, output_root,
                 artifacts=args.artifacts_root, chip_data_root=Path(args.chip_data_root))
    print(json.dumps({"output_root": str(output_root), "selection": report["selection"],
                      "decisions": report["decisions"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
