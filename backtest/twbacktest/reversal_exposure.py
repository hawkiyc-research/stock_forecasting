"""Protocol 13.19 supplement: how much of the long-short result is reversal exposure?

The S1 book of ``scenario_suite`` lost in P2 because its ranking is in substance a short-term
reversal signal and P2 was a momentum regime. This diagnostic separates the structural exposure
from whatever ranking skill is left, the way the literature does (residual reversal, style
neutralization, conditioning on turnover):

* score variants: ``v0`` = the §13.14 smoothed percentile score; ``v1`` = that score orthogonalized
  every day inside the universe against the trailing 20- and 60-session returns and a 60-session
  beta (OLS residual);
* constructions (daily cohorts held h = 10 sessions, top-minus-bottom quartile, dollar-neutral,
  each leg at full capital like S1): ``plain``; ``momentum`` = ranked and paired inside quintiles of
  the trailing 20-session return, so both legs share the momentum distribution; ``turnover_*`` =
  the plain spread inside terciles of the turnover spike (5-session mean dollar volume over the
  60-session median), reported per tercile;
* targets: the raw forward return (next open to the close 10 sessions later), the raw alpha against
  the market index and the residual target of the run's ``targets.parquet``.

Per market, period and series it reports the mean spread per cohort with a block bootstrap interval,
the annualized gross (x 252 / h) and the net under the retail and institution cost profiles (both
legs enter and exit once per cohort). A few thousand small per-date groups: serial on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats
from twbacktest.exposure_diagnostics import load_residual_targets, quantile_long_short
from twbacktest.futures_universe import _spearman
from twbacktest.scenario_suite import CONFIG_DIR, CostProfile, cost_profiles, is_primary
from twbacktest.topk_dropout import DECISION_PERIOD
from twbacktest.tw_strategy import (
    percentile_ranks,
    smooth_by_symbol,
    universe_groups,
    universe_mask,
)
from twbacktest.walkforward_data import (
    build_holding_returns,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_PROFILES = CONFIG_DIR / "cost_profiles.json"
H = 10
QUINTILES = 5
TERCILES = 3
MIN_NAMES = 20
MIN_BETA_OBSERVATIONS = 40
BETA_WINDOW = 60
TURNOVER_SHORT, TURNOVER_LONG = 5, 60
BLOCK = 20
BOOTSTRAP = {"replicates": 1000, "seed": 11}
PERIODS = {"P1": ("2023-01-02", "2025-06-01"), "P2": ("2025-06-01", "2026-06-01")}
MARKETS = {
    "TWSE": {"code": 0, "index": "TAIEX.TW", "min_dollar_volume": 30_000_000.0},
    "TPEX": {"code": 1, "index": "TPEX.TWO", "min_dollar_volume": 10_000_000.0},
}
MIN_CLOSE = 10.0
VARIANTS = ("v0", "v1")
TARGETS = ("raw", "alpha", "residual")
CONSTRUCTIONS = ("plain", "momentum", "turnover_low", "turnover_mid", "turnover_high")
FEATURES = ("ret20", "ret60", "beta")


# --------------------------------------------------------------------------- features


def trailing_features(prices_path: Path) -> pd.DataFrame:
    """Per (symbol, date): trailing 20/60-session log returns of the adjusted close, the 60-session
    beta to the market index and the turnover spike, all from data up to that date.

    Reads the raw parquet itself because ``load_prices`` drops the volume column.
    """
    frame = pd.read_parquet(
        prices_path,
        columns=["timestamp", "symbol", "asset_type", "market", "close", "adjusted_close",
                 "volume"],
    )
    frame = frame[frame["market"].isin(MARKETS)].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    index_symbols = {spec["index"]: market for market, spec in MARKETS.items()}
    index_rows = frame[(frame["asset_type"] == "index") & frame["symbol"].isin(index_symbols)]
    index_rows = index_rows[["symbol", "date", "adjusted_close"]].copy()
    index_rows["market"] = index_rows["symbol"].map(index_symbols)
    index_rows = index_rows.drop_duplicates(["market", "date"]).sort_values(["market", "date"])
    index_rows["index_ret"] = np.log(
        index_rows["adjusted_close"] / index_rows.groupby("market")["adjusted_close"].shift(1)
    )
    stocks = frame[frame["asset_type"] != "index"].drop_duplicates(["symbol", "date"])
    stocks = stocks.merge(index_rows[["market", "date", "index_ret"]], on=["market", "date"],
                          how="left")
    stocks = stocks.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = stocks.groupby("symbol", sort=False)
    close = stocks["adjusted_close"]
    stocks["ret1"] = np.log(close / grouped["adjusted_close"].shift(1))
    stocks["ret20"] = np.log(close / grouped["adjusted_close"].shift(20))
    stocks["ret60"] = np.log(close / grouped["adjusted_close"].shift(60))
    # Rolling beta from rolling moments, per symbol.
    stocks["rm"] = stocks["ret1"] * stocks["index_ret"]
    stocks["mm"] = stocks["index_ret"] ** 2

    def rolling(column: str, window: int, min_periods: int, how: str) -> pd.Series:
        by_symbol = stocks.groupby("symbol", sort=False)[column]
        roller = by_symbol.rolling(window, min_periods=min_periods)
        return getattr(roller, how)().reset_index(level=0, drop=True)

    e_r, e_m, e_rm, e_mm = (
        rolling(c, BETA_WINDOW, MIN_BETA_OBSERVATIONS, "mean")
        for c in ("ret1", "index_ret", "rm", "mm")
    )
    var_m = e_mm - e_m**2
    with np.errstate(divide="ignore", invalid="ignore"):
        stocks["beta"] = np.where(var_m > 0, (e_rm - e_r * e_m) / var_m, np.nan)
    stocks["traded"] = stocks["close"] * stocks["volume"]
    short = rolling("traded", TURNOVER_SHORT, TURNOVER_SHORT, "mean")
    long = rolling("traded", TURNOVER_LONG, TURNOVER_LONG // 2, "median")
    with np.errstate(divide="ignore", invalid="ignore"):
        stocks["turnover_spike"] = np.log(short / long)
    return stocks[["symbol", "date", "ret20", "ret60", "beta", "turnover_spike"]]


def row_features(predictions: pd.DataFrame, features: pd.DataFrame) -> dict[str, np.ndarray]:
    merged = predictions[["symbol", "signal_date"]].merge(
        features, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left",
        validate="one_to_one",
    )
    return {name: merged[name].to_numpy(dtype=np.float64)
            for name in ("ret20", "ret60", "beta", "turnover_spike")}


# --------------------------------------------------------------------------- scores


def orthogonalize(score: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """OLS residual of ``score`` on an intercept and ``regressors`` [n, k]; NaN where any input is
    missing. With too few complete rows the residual is NaN everywhere."""
    out = np.full(score.shape[0], np.nan)
    ok = np.isfinite(score) & np.all(np.isfinite(regressors), axis=1)
    if ok.sum() <= regressors.shape[1] + 1:
        return out
    design = np.column_stack([np.ones(int(ok.sum())), regressors[ok]])
    coefficients, *_ = np.linalg.lstsq(design, score[ok], rcond=None)
    out[ok] = score[ok] - design @ coefficients
    return out


def quantile_labels(values: np.ndarray, bins: int) -> np.ndarray:
    """Balanced quantile bins (0..bins-1) of a finite vector; equal values share a bin."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=np.int64)
    labels[order] = (np.arange(values.size) * bins) // max(values.size, 1)
    return labels


# --------------------------------------------------------------------------- daily table


def daily_table(
    *,
    dates: np.ndarray,
    groups: tuple[np.ndarray, ...],
    v0: np.ndarray,
    v1: np.ndarray,
    targets: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    costs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One record per signal date: spreads of every (variant, construction, target), the score /
    feature correlations and the per-cohort cost of both legs under every cost profile."""
    records = []
    for j, rows in enumerate(groups):
        rows = rows[np.isfinite(v0[rows]) & np.isfinite(targets["raw"][rows])]
        if rows.size < MIN_NAMES:
            continue
        record: dict[str, Any] = {"signal_date": pd.Timestamp(dates[j]), "names": int(rows.size)}
        ret20 = features["ret20"][rows]
        complete = rows[np.isfinite(ret20) & np.isfinite(v1[rows])
                        & np.isfinite(features["turnover_spike"][rows])]
        record["names_complete"] = int(complete.size)
        if complete.size < MIN_NAMES:
            continue
        momentum = quantile_labels(features["ret20"][complete], QUINTILES).astype(object)
        terciles = quantile_labels(features["turnover_spike"][complete], TERCILES)
        for variant, score in (("v0", v0), ("v1", v1)):
            s = score[complete]
            for target, values in targets.items():
                t = values[complete]
                record[f"{variant}_plain_{target}"] = quantile_long_short(s, t, None)[0]
                record[f"{variant}_momentum_{target}"] = quantile_long_short(s, t, momentum)[0]
                if variant == "v0":
                    for level, name in enumerate(("turnover_low", "turnover_mid", "turnover_high")):
                        part = terciles == level
                        if part.sum() >= MIN_NAMES:
                            record[f"v0_{name}_{target}"] = quantile_long_short(
                                s[part], t[part], None)[0]
                        else:
                            record[f"v0_{name}_{target}"] = np.nan
            for name in FEATURES:
                record[f"corr_{variant}_{name}"] = _spearman(s, features[name][complete])
        for profile, rate in costs.items():
            record[f"cost_{profile}"] = float(np.nanmean(rate[complete]))
        records.append(record)
    return pd.DataFrame(records)


def round_trip_rates(profile: CostProfile, market_codes: np.ndarray) -> np.ndarray:
    """Cost of one cohort per unit of capital: the long leg's buy and sell plus the short leg's
    entry and exit (each leg at full capital, as in S1)."""
    buy, sell = profile.long_rates(market_codes)
    entry, exit_ = profile.short_rates(market_codes)
    return buy + sell + entry + exit_ + profile.borrow_fee_daily * H


# --------------------------------------------------------------------------- summary


def period_mask(dates: pd.Series, period: str) -> np.ndarray:
    if period == DECISION_PERIOD:
        return np.any([period_mask(dates, name) for name in PERIODS], axis=0)
    start, end = PERIODS[period]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()


def summarize(daily: pd.DataFrame, *, model: str, market: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_year = 252.0 / H
    for period in (*PERIODS, DECISION_PERIOD):
        part = daily[period_mask(daily["signal_date"], period)]
        if part.empty:
            continue
        shared = {"model": model, "market": market, "period": period, "days": len(part),
                  "names_mean": float(part["names_complete"].mean())}
        costs = {name: float(part[f"cost_{name}"].mean())
                 for name in ("retail", "institution") if f"cost_{name}" in part}
        for variant in VARIANTS:
            for construction in CONSTRUCTIONS:
                if variant == "v1" and construction.startswith("turnover"):
                    continue
                for target in TARGETS:
                    key = f"{variant}_{construction}_{target}"
                    if key not in part:
                        continue
                    mean, low, high = stats.bootstrap_mean(
                        part[key].to_numpy(dtype=np.float64), block=BLOCK, **BOOTSTRAP)
                    row = {**shared, "variant": variant, "construction": construction,
                           "target": target, "spread": mean, "spread_low": low,
                           "spread_high": high, "gross_annual": mean * per_year,
                           "gross_annual_low": low * per_year, "gross_annual_high": high * per_year}
                    for name, cost in costs.items():
                        row[f"cost_drag_{name}"] = cost * per_year
                        row[f"net_annual_{name}"] = (mean - cost) * per_year
                    rows.append(row)
        for variant in VARIANTS:
            for name in FEATURES:
                key = f"corr_{variant}_{name}"
                rows.append({**shared, "variant": variant, "construction": "correlation",
                             "target": name, "spread": float(part[key].mean()),
                             "spread_low": np.nan, "spread_high": np.nan,
                             "gross_annual": np.nan, "gross_annual_low": np.nan,
                             "gross_annual_high": np.nan})
    return rows


def reading_rules(grid: pd.DataFrame) -> dict[str, Any]:
    """The pre-registered reading of protocol 13.19 supplement item 6, per model and market."""
    out: dict[str, Any] = {}

    def pick(model, market, period, variant, construction, target="raw"):
        rows = grid[(grid["model"] == model) & (grid["market"] == market)
                    & (grid["period"] == period) & (grid["variant"] == variant)
                    & (grid["construction"] == construction) & (grid["target"] == target)]
        return None if rows.empty else rows.iloc[0]

    for (model, market), _ in grid.groupby(["model", "market"]):
        base_p1 = pick(model, market, "P1", "v0", "plain")
        neutral_p1 = pick(model, market, "P1", "v1", "momentum")
        neutral_p2 = pick(model, market, "P2", "v1", "momentum")
        if base_p1 is None or neutral_p1 is None or neutral_p2 is None:
            continue
        ratio = (float(neutral_p1["spread"]) / float(base_p1["spread"])
                 if base_p1["spread"] not in (0, np.nan) and base_p1["spread"] > 0 else np.nan)
        verdict = {
            "p1_spread_v0_plain": float(base_p1["spread"]),
            "p1_spread_v1_momentum": float(neutral_p1["spread"]),
            "p1_retention": ratio,
            "p2_spread_v1_momentum": float(neutral_p2["spread"]),
            "p2_v1_momentum_high": float(neutral_p2["spread_high"]),
            "exposure_is_main_cause": bool(np.isfinite(ratio) and ratio >= 0.5
                                           and float(neutral_p2["spread_high"]) >= 0),
            "model_is_reversal_only": bool(np.isfinite(ratio) and ratio < 0.25),
        }
        turnover = {}
        for period in ("P1", "P2"):
            low = pick(model, market, period, "v0", "turnover_low")
            high = pick(model, market, period, "v0", "turnover_high")
            if low is not None and high is not None:
                turnover[period] = {"low": float(low["spread"]), "high": float(high["spread"]),
                                    "high_above_low": bool(high["spread"] > low["spread"])}
        verdict["turnover"] = turnover
        verdict["conditioning_helps"] = bool(
            turnover and all(v["high_above_low"] for v in turnover.values()))
        out.setdefault(model, {})[market] = verdict
    return out


# --------------------------------------------------------------------------- run


def run(
    predictions: list[Path],
    models: list[str],
    targets: list[Path] | None,
    prices_path: Path,
    output_root: Path,
    *,
    profiles_path: Path = DEFAULT_PROFILES,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    if targets is not None and len(targets) != len(models):
        raise ValueError("Give one --targets per --predictions file, or none")
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    primary = [p for p in cost_profiles(profiles) if is_primary(p, profiles)]
    prices = load_prices(prices_path, liquidity_lookback=20)
    features_table = trailing_features(prices_path)
    all_rows: list[dict[str, Any]] = []
    dailies = []
    for i, (path, model) in enumerate(zip(predictions, models, strict=True)):
        frame = load_predictions(path, model)
        inputs = build_inputs(frame, prices)
        log_return, log_alpha = build_holding_returns(frame, prices)
        raw = np.expm1(log_return[:, H - 1])
        alpha = log_alpha[:, H - 1]
        residual = np.full(inputs.index.count, np.nan)
        if targets is not None:
            table = load_residual_targets(targets[i], (H,))
            merged = frame[["symbol", "signal_date"]].merge(table, on=["symbol", "signal_date"],
                                                              how="left")
            residual = merged[f"residual_h{H:02d}"].to_numpy(dtype=np.float64)
        features = row_features(frame, features_table)
        regressors = np.column_stack([features[name] for name in FEATURES])
        costs = {p.name: round_trip_rates(p, inputs.index.market_codes) for p in primary}
        column = inputs.horizons.index(H)
        for market, spec in MARKETS.items():
            mask = universe_mask(inputs, (spec["code"],),
                                 min_dollar_volume=spec["min_dollar_volume"],
                                 min_close=MIN_CLOSE, exclude_etf=True)
            groups = universe_groups(inputs, mask)
            v0 = smooth_by_symbol(
                percentile_ranks(inputs.scores[:, column], groups, mask),
                inputs.index.date_codes, inputs.index.symbol_codes, half_life=3.0, max_gap=5)
            v1 = np.full(inputs.index.count, np.nan)
            for rows in groups:
                if rows.size:
                    v1[rows] = orthogonalize(v0[rows], regressors[rows])
            daily = daily_table(
                dates=inputs.dates.to_numpy(), groups=groups, v0=v0, v1=v1,
                targets={"raw": raw, "alpha": alpha, "residual": residual},
                features=features, costs=costs,
            )
            if daily.empty:
                continue
            dailies.append(daily.assign(model=model, market=market))
            all_rows.extend(summarize(daily, model=model, market=market))
    grid = pd.DataFrame(all_rows)
    verdicts = reading_rules(grid)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "reversal-exposure-diagnostic",
        "predictions": {m: str(p) for p, m in zip(predictions, models, strict=True)},
        "targets": [str(t) for t in targets] if targets else None,
        "prices": str(prices_path),
        "h": H,
        "reading": verdicts,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(grid, verdicts), encoding="utf-8")
    return report


def summary_markdown(grid: pd.DataFrame, verdicts: dict[str, Any]) -> str:
    pct = runner.percent
    lines = [
        "# 反轉暴露診斷摘要 / Reversal-exposure diagnostic summary", "",
        "| 模型 | 市場 | 期間 | 分數 | 建構 | 目標 | 每批價差 | 95% 區間 | 年化毛 | 機構淨 "
        "| 散戶淨 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    spreads = grid[grid["construction"] != "correlation"]
    for r in spreads.sort_values(["model", "market", "variant", "construction", "target",
                                  "period"]).itertuples():
        institution = pct(getattr(r, "net_annual_institution", float("nan")))
        retail = pct(getattr(r, "net_annual_retail", float("nan")))
        lines.append(
            f"| {r.model} | {r.market} | {r.period} | {r.variant} | {r.construction} | "
            f"{r.target} | {pct(r.spread)} | [{pct(r.spread_low)}, {pct(r.spread_high)}] | "
            f"{pct(r.gross_annual)} | {institution} | {retail} |")
    lines += ["", "## 分數與特徵的平均橫斷面相關 / Mean cross-sectional correlation", "",
              "| 模型 | 市場 | 期間 | 分數 | 特徵 | Spearman |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in grid[grid["construction"] == "correlation"].sort_values(
            ["model", "market", "variant", "target", "period"]).itertuples():
        lines.append(f"| {r.model} | {r.market} | {r.period} | {r.variant} | {r.target} | "
                     f"{runner.number(r.spread)} |")
    lines += ["", "## 事前判讀 / Pre-registered reading", ""]
    for model, markets in verdicts.items():
        for market, v in markets.items():
            lines.append(
                f"- {model} {market}: P1 retention {v['p1_retention']:.2f}, P2 v1-momentum "
                f"{pct(v['p2_spread_v1_momentum'])} (upper {pct(v['p2_v1_momentum_high'])}); "
                f"exposure_is_main_cause={v['exposure_is_main_cause']}, "
                f"model_is_reversal_only={v['model_is_reversal_only']}, "
                f"conditioning_helps={v['conditioning_helps']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--targets", type=Path, action="append",
                        help="the same run's targets.parquet (residual_h10); one per model")
    parser.add_argument("--prices", type=Path, required=True, help="raw market.parquet")
    parser.add_argument("--cost-profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "reversal-exposure")
    report = run(args.predictions, args.model, args.targets, args.prices, output_root,
                 profiles_path=args.cost_profiles)
    print(json.dumps({"output_root": str(output_root), "reading": report["reading"]}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
