"""Protocol 13.17 step 0: how much residual alpha is there to trade, and does it pay for costs?

Descriptive, run before any strategy is built. For each model (B0 / D0 on the residual labels),
holding horizon h in {5, 10}, universe and period it reports:

* the Spearman Rank IC and the top-minus-bottom quartile spread of the *residual* target (what the
  model predicts) and of the *raw* forward return (next open to the close h sessions later);
* dollar-neutral quartile long-short portfolios built three ways -- plain, industry-neutral
  (within every industry of the reference table, pooling small ones) and size-neutral (within the
  two halves by 20-session median dollar volume) -- as gross return per cohort and annualized for
  an overlay of gross notional L = 1 (half the capital on each side);
* the estimated annual cost drag of that overlay in stock futures (per side: transaction tax,
  commission per contract, one tick of slippage; daily replacement of 1/h of each side plus a
  monthly roll) and the stop rule of protocol 13.17 item 0: inside the futures-liquid universe
  at h = 10 on P1+P2, the industry-neutral gross return must reach twice the cost drag.

Universes: ``liquidity`` = the protocol 13.14 stock universe per market (20-session median dollar
volume >= NT$30M on TWSE / 10M on TPEX, close >= NT$10, no ETFs), ranked within the market;
``futures_liquid`` = stocks whose near-month stock-futures contract is liquid on the signal date
(protocol 13.15), ranked across both markets. Returns are the stocks' own prices (futures track
them closely over multi-day holds; the basis is reported elsewhere). A few thousand small per-date
groups: serial on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats, taifex
from twbacktest.futures_universe import PERIODS, _spearman, eligibility_frame, segment_masks
from twbacktest.walkforward_data import (
    build_holding_returns,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_TAIFEX = "diagnostics/taifex-all"
DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[2] / "configs/reference/tw_company_reference_20261022.csv"
)
HORIZONS = (5, 10)
DECISION_PERIOD = "P1+P2"
QUANTILE = 0.25
MIN_NAMES = 20
MIN_CLASS = 4  # a class (industry, size half) needs this many names to rank inside it
OTHER = "__other__"
UNKNOWN = "UNK"
LIQUIDITY = {"TWSE": 30_000_000.0, "TPEX": 10_000_000.0}
MIN_CLOSE = 10.0
CONSTRUCTIONS = ("plain", "industry", "size")
BOOTSTRAP = {"replicates": 1000, "seed": 11}
# Protocol 13.15 / 13.17 futures costs per side.
TAX_RATE = 2e-5
COMMISSION_TWD = 30.0
SLIPPAGE_TICKS = 1
TICK_BANDS = ((10.0, 0.01), (50.0, 0.05), (100.0, 0.1), (500.0, 0.5), (1000.0, 1.0), (None, 5.0))
STANDARD_SHARES = 2000
ROLLS_PER_YEAR = 12
STOP_RULE = {"universe": "futures_liquid", "h": 10, "construction": "industry", "multiple": 2.0}


def tick_size(price: float) -> float:
    for bound, tick in TICK_BANDS:
        if bound is None or price < bound:
            return tick
    return TICK_BANDS[-1][1]


def per_side_cost_rate(price: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Tax + commission + one tick, as a fraction of notional, per name."""
    ticks = np.array([tick_size(float(p)) if np.isfinite(p) else np.nan for p in price])
    with np.errstate(divide="ignore", invalid="ignore"):
        return TAX_RATE + SLIPPAGE_TICKS * ticks / price + COMMISSION_TWD / (price * shares)


def load_reference(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype={"symbol": str, "industry_code": str})
    codes = frame["industry_code"].fillna("").astype(str).str.strip()
    return {
        symbol: (code if code else UNKNOWN)
        for symbol, code in zip(frame["symbol"], codes, strict=True)
    }


def load_residual_targets(path: Path, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = ["symbol", "signal_date", *[f"residual_h{h:02d}" for h in horizons]]
    frame = pd.read_parquet(path, columns=columns)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.drop_duplicates(["symbol", "signal_date"], keep="first")


def pooled_classes(classes: np.ndarray) -> np.ndarray:
    """Classes with fewer than MIN_CLASS names fall into one pooled class."""
    counts = Counter(classes.tolist())
    return np.array([c if counts[c] >= MIN_CLASS else OTHER for c in classes], dtype=object)


def quantile_long_short(
    score: np.ndarray, ret: np.ndarray, classes: np.ndarray | None
) -> tuple[float, int, dict[str, float]]:
    """Dollar-neutral top-minus-bottom quantile spread, ranked inside every class.

    Each class weighs by its name count, so both sides carry the same class weights (industry or
    size neutrality by construction). Returns (spread per cohort, names used, class weights).
    """
    labels = np.zeros(len(score), dtype=object) if classes is None else pooled_classes(classes)
    total_long = total_short = weight_sum = 0.0
    used, weights = 0, {}
    for label in sorted(set(labels.tolist()), key=str):
        idx = np.flatnonzero(labels == label)
        if idx.size < MIN_CLASS:
            continue
        k = max(1, math.ceil(idx.size * QUANTILE))
        order = idx[np.argsort(-score[idx], kind="stable")]
        long, short = order[:k], order[-k:]
        weight = float(idx.size)
        total_long += weight * float(ret[long].mean())
        total_short += weight * float(ret[short].mean())
        weight_sum += weight
        used += idx.size
        weights[str(label)] = weight
    if weight_sum == 0:
        return float("nan"), 0, {}
    return (
        (total_long - total_short) / weight_sum,
        used,
        {k: v / weight_sum for k, v in weights.items()},
    )


def daily_table(
    frame: pd.DataFrame,
    *,
    score: np.ndarray,
    raw_return: np.ndarray,
    residual: np.ndarray,
    cost_rate: np.ndarray,
    industry: np.ndarray,
    size_half: np.ndarray,
    mask: np.ndarray,
    group_key: np.ndarray,
) -> pd.DataFrame:
    """One row per (signal date, ranking group): IC, quartile spreads, three L/S constructions."""
    rows = np.flatnonzero(mask & np.isfinite(score) & np.isfinite(raw_return))
    if rows.size == 0:
        return pd.DataFrame()
    dates = frame["signal_date"].to_numpy()[rows]
    keys = group_key[rows]
    order = np.lexsort((keys, dates))
    rows, dates, keys = rows[order], dates[order], keys[order]
    change = np.r_[True, (dates[1:] != dates[:-1]) | (keys[1:] != keys[:-1]), True]
    records = []
    for start, stop in pairwise(np.flatnonzero(change)):
        group = rows[start:stop]
        if group.size < MIN_NAMES:
            continue
        s, r, e = score[group], raw_return[group], residual[group]
        k = max(1, math.ceil(group.size * QUANTILE))
        ranked = group[np.argsort(-s, kind="stable")]
        top, bottom = ranked[:k], ranked[-k:]
        finite_e = np.isfinite(e)
        record = {
            "signal_date": pd.Timestamp(dates[start]),
            "group": str(keys[start]),
            "names": int(group.size),
            "rank_ic_raw": _spearman(s, r),
            "rank_ic_residual": _spearman(s[finite_e], e[finite_e])
            if finite_e.sum() >= MIN_NAMES
            else np.nan,
            "quartile_spread_raw": float(raw_return[top].mean() - raw_return[bottom].mean()),
            "quartile_spread_residual": float(
                np.nanmean(residual[top]) - np.nanmean(residual[bottom])
            ),
            "universe_return": float(r.mean()),
            "cost_rate_mean": float(np.nanmean(cost_rate[group])),
        }
        for name, classes in (
            ("plain", None),
            ("industry", industry[group]),
            ("size", size_half[group]),
        ):
            spread, used, _weights = quantile_long_short(s, r, classes)
            record[f"ls_{name}"] = spread
            record[f"ls_{name}_names"] = used
        records.append(record)
    return pd.DataFrame(records)


def period_mask(dates: pd.Series, period: str) -> np.ndarray:
    if period == DECISION_PERIOD:
        return np.any([period_mask(dates, name) for name in PERIODS], axis=0)
    start, end = PERIODS[period]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()


def annual_cost_drag(cost_rate: float, h: int) -> float:
    """Overlay with gross notional 1: each day 1/h of both sides is replaced (exit + entry,
    each a side of traded notional 1/h x 1), plus a monthly roll of the whole book (two sides)."""
    daily_traded = 2.0 / h
    return (252.0 * daily_traded + ROLLS_PER_YEAR * 2.0) * cost_rate


def summarize(daily: pd.DataFrame, *, h: int) -> dict[str, Any]:
    if daily.empty:
        return {"days": 0}
    per_date = daily.groupby("signal_date").mean(numeric_only=True).sort_index()
    block_ic = 20
    block_spread = max(5, 2 * h)
    out: dict[str, Any] = {"days": len(per_date), "names_mean": float(daily["names"].mean())}
    for key in ("rank_ic_raw", "rank_ic_residual"):
        mean, low, high = stats.bootstrap_mean(
            per_date[key].to_numpy(), block=block_ic, **BOOTSTRAP
        )
        out[key], out[f"{key}_low"], out[f"{key}_high"] = mean, low, high
    for key in ("quartile_spread_raw", "quartile_spread_residual"):
        mean, low, high = stats.bootstrap_mean(
            per_date[key].to_numpy(), block=block_spread, **BOOTSTRAP
        )
        out[key], out[f"{key}_low"], out[f"{key}_high"] = mean, low, high
    cost_rate = float(per_date["cost_rate_mean"].mean())
    out["cost_rate_per_side"] = cost_rate
    out["cost_drag_annual"] = annual_cost_drag(cost_rate, h)
    per_year = 252.0 / h
    for name in CONSTRUCTIONS:
        mean, low, high = stats.bootstrap_mean(
            per_date[f"ls_{name}"].to_numpy(), block=block_spread, **BOOTSTRAP
        )
        # Gross notional 1 = half the capital on each side, so the overlay earns half the spread.
        out[f"ls_{name}_spread"] = mean
        out[f"ls_{name}_gross_annual"] = 0.5 * mean * per_year
        out[f"ls_{name}_gross_annual_low"] = 0.5 * low * per_year
        out[f"ls_{name}_gross_annual_high"] = 0.5 * high * per_year
        out[f"ls_{name}_net_annual"] = 0.5 * mean * per_year - out["cost_drag_annual"]
        out[f"ls_{name}_names_mean"] = float(daily[f"ls_{name}_names"].mean())
    return out


def stop_rule(grid: pd.DataFrame, model: str) -> dict[str, Any]:
    rows = grid[
        (grid["model"] == model)
        & (grid["universe"] == STOP_RULE["universe"])
        & (grid["h"] == STOP_RULE["h"])
        & (grid["period"] == DECISION_PERIOD)
    ]
    if rows.empty or int(rows.iloc[0]["days"]) == 0:
        return {"model": model, "evaluated": False}
    row = rows.iloc[0]
    gross = float(row[f"ls_{STOP_RULE['construction']}_gross_annual"])
    drag = float(row["cost_drag_annual"])
    return {
        "model": model,
        "evaluated": True,
        "gross_annual": gross,
        "cost_drag_annual": drag,
        "required_gross_annual": STOP_RULE["multiple"] * drag,
        "proceed": bool(np.isfinite(gross) and gross >= STOP_RULE["multiple"] * drag),
    }


def run(
    predictions: list[Path],
    targets: list[Path],
    models: list[str],
    prices: Path,
    taifex_root: Path,
    reference: Path,
    output_root: Path,
    *,
    min_volume: float,
    lookback: int,
) -> dict[str, Any]:
    if not (len(predictions) == len(targets) == len(models)) or not models:
        raise ValueError("Give one --predictions, one --targets and one --model per model")
    price_frame = load_prices(prices)
    eligibility, audit = eligibility_frame(taifex_root, min_volume=min_volume, lookback=lookback)
    _futures, listing = taifex.load(taifex_root)
    shares_by_symbol = listing.groupby("symbol")["shares"].max().to_dict()
    industry_by_symbol = load_reference(reference)
    rows, dailies = [], []
    for path, target_path, label in zip(predictions, targets, models, strict=True):
        frame = load_predictions(path, label)
        inputs = build_inputs(frame, price_frame)
        log_return, _log_alpha = build_holding_returns(frame, price_frame)
        residual_frame = frame[["symbol", "signal_date"]].merge(
            load_residual_targets(target_path, HORIZONS),
            on=["symbol", "signal_date"],
            how="left",
            validate="one_to_one",
        )
        symbols = frame["symbol"].to_numpy()
        is_etf = inputs.index.is_etf
        industry = np.array([industry_by_symbol.get(s, UNKNOWN) for s in symbols], dtype=object)
        shares = np.array([shares_by_symbol.get(s, STANDARD_SHARES) for s in symbols], dtype=float)
        cost_rate = per_side_cost_rate(inputs.raw_open_1, shares)
        dollar_volume = inputs.dollar_volume
        market = frame["market"].to_numpy()
        liquidity_threshold = np.array([LIQUIDITY[m] for m in market])
        liquidity_mask = (
            ~is_etf
            & np.isfinite(dollar_volume)
            & (dollar_volume >= liquidity_threshold)
            & (inputs.raw_close >= MIN_CLOSE)
        )
        futures_mask = segment_masks(frame, eligibility)["futures_liquid"] & ~is_etf
        # Size halves are formed within the ranking group on each date (median dollar volume).
        size_half = np.empty(len(frame), dtype=object)
        size_half[:] = "small"
        universes = {
            "liquidity": (liquidity_mask, market.astype(object)),
            "futures_liquid": (futures_mask, np.full(len(frame), "TW", dtype=object)),
        }
        for universe, (mask, group_key) in universes.items():
            _assign_size_halves(
                size_half, dollar_volume, mask, frame["signal_date"].to_numpy(), group_key
            )
            for h in HORIZONS:
                column = inputs.horizons.index(h)
                raw_return = np.expm1(log_return[:, h - 1])
                residual = residual_frame[f"residual_h{h:02d}"].to_numpy(dtype=np.float64)
                daily = daily_table(
                    frame,
                    score=inputs.scores[:, column],
                    raw_return=raw_return,
                    residual=residual,
                    cost_rate=cost_rate,
                    industry=industry,
                    size_half=size_half,
                    mask=mask,
                    group_key=group_key,
                )
                if daily.empty:
                    continue
                daily = daily.assign(model=label, universe=universe, h=h)
                dailies.append(daily)
                for period in (*PERIODS, DECISION_PERIOD):
                    part = daily[period_mask(daily["signal_date"], period)]
                    rows.append(
                        {
                            "model": label,
                            "universe": universe,
                            "h": h,
                            "period": period,
                            **summarize(part, h=h),
                        }
                    )
    grid = pd.DataFrame(rows)
    verdicts = [stop_rule(grid, label) for label in models]
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "exposure-step0-diagnostic",
        "protocol": "13.17 item 0",
        "predictions": {label: str(path) for path, label in zip(predictions, models, strict=True)},
        "targets": {label: str(path) for path, label in zip(targets, models, strict=True)},
        "prices": str(prices),
        "taifex_root": str(taifex_root),
        "reference": str(reference),
        "min_volume": min_volume,
        "lookback": lookback,
        "quantile": QUANTILE,
        "costs": {
            "tax_rate": TAX_RATE,
            "commission_twd": COMMISSION_TWD,
            "slippage_ticks": SLIPPAGE_TICKS,
            "rolls_per_year": ROLLS_PER_YEAR,
        },
        "stop_rule": {**STOP_RULE, "verdicts": verdicts},
        "eligibility_audit": audit,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(grid, verdicts), encoding="utf-8")
    return report


def _assign_size_halves(
    size_half: np.ndarray,
    dollar_volume: np.ndarray,
    mask: np.ndarray,
    dates: np.ndarray,
    group_key: np.ndarray,
) -> None:
    """Label every masked row 'big' or 'small' by the median dollar volume of its date group."""
    rows = np.flatnonzero(mask & np.isfinite(dollar_volume))
    if rows.size == 0:
        return
    frame = pd.DataFrame(
        {"row": rows, "date": dates[rows], "key": group_key[rows], "dv": dollar_volume[rows]}
    )
    median = frame.groupby(["date", "key"])["dv"].transform("median").to_numpy()
    size_half[rows] = np.where(frame["dv"].to_numpy() >= median, "big", "small")


def _row(row) -> str:
    pct = runner.percent
    cells = [
        row.model,
        row.universe,
        str(int(row.h)),
        row.period,
        f"{row.names_mean:.0f}",
        runner.number(row.rank_ic_residual),
        runner.number(row.rank_ic_raw),
        pct(row.quartile_spread_residual),
        pct(row.quartile_spread_raw),
        pct(row.ls_plain_gross_annual),
        f"{pct(row.ls_industry_gross_annual)} [{pct(row.ls_industry_gross_annual_low)}, "
        f"{pct(row.ls_industry_gross_annual_high)}]",
        pct(row.ls_size_gross_annual),
        pct(row.cost_rate_per_side),
        pct(row.cost_drag_annual),
        pct(row.ls_industry_net_annual),
    ]
    return "| " + " | ".join(cells) + " |"


ZH_COLUMNS = (
    "模型",
    "宇宙",
    "h",
    "期間",
    "平均檔數",
    "Rank IC 殘差",
    "Rank IC 原始",
    "四分位價差 殘差/批",
    "四分位價差 原始/批",
    "多空毛年化 plain",
    "多空毛年化 產業中性 [95%]",
    "多空毛年化 規模中性",
    "單邊成本率",
    "年化成本拖累",
    "產業中性淨年化",
)
EN_COLUMNS = (
    "Model",
    "Universe",
    "h",
    "Period",
    "Mean names",
    "Rank IC residual",
    "Rank IC raw",
    "Quartile spread residual / cohort",
    "Quartile spread raw / cohort",
    "L/S gross annual plain",
    "L/S gross annual industry-neutral [95%]",
    "L/S gross annual size-neutral",
    "Cost rate per side",
    "Annual cost drag",
    "Industry-neutral net annual",
)
ZH_NOTES = (
    "描述性分析。宇宙：liquidity = §13.14 的市場內流動性宇宙（市場內排名）；",
    "futures_liquid = §13.15 的股票期貨流動宇宙（兩市場合併排名）。",
    "報酬為次一開盤買進、h 日後收盤賣出的原始報酬；殘差為 §13.16 的殘差目標。",
    "多空為前後四分位、金額中性；產業中性 = 每個產業內部排名、產業權重等於檔數比例",
    "（不足 4 檔者併入其他）；規模中性 = 當日成交金額中位數上下兩半內部排名。",
    "多空年化以總名目 = 資本（每邊一半）計。",
    "成本：每邊期交稅 0.002%、手續費每口 30 元、1 tick 滑價；每日換 1/h 的部位加每月換月。",
)
EN_NOTES = (
    "Descriptive. Universes: liquidity = the protocol 13.14 per-market liquidity universe",
    "(ranked within the market); futures_liquid = the protocol 13.15 liquid stock-futures",
    "universe (ranked across both markets). Returns: next open to the close h sessions later,",
    "raw; residual = the protocol 13.16 residual target. Long-short = top minus bottom",
    "quartile, dollar-neutral; industry-neutral ranks inside every industry with weights",
    "proportional to name counts (fewer than 4 names pooled); size-neutral ranks inside the",
    "two halves by the date's median dollar volume. Annualized for gross notional = capital",
    "(half per side). Costs per side: 0.002% tax, NT$30 per contract, one tick; 1/h of the",
    "book replaced daily plus a monthly roll.",
)


def _header(columns: tuple[str, ...]) -> tuple[str, str]:
    return "| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)


def _verdict_lines(verdicts: list[dict[str, Any]]) -> list[str]:
    lines = []
    for v in verdicts:
        if not v.get("evaluated"):
            lines.append(f"- {v['model']}: 未評估 / not evaluated")
            continue
        outcome = "繼續 / proceed" if v["proceed"] else "停止 / stop"
        lines.append(
            f"- {v['model']}: 毛年化 {runner.percent(v['gross_annual'])}，"
            f"成本拖累 {runner.percent(v['cost_drag_annual'])}，"
            f"門檻 {runner.percent(v['required_gross_annual'])} -> {outcome}"
        )
    return lines


def summary_markdown(grid: pd.DataFrame, verdicts: list[dict[str, Any]]) -> str:
    table = grid[grid["days"] > 0].sort_values(["model", "universe", "h", "period"])
    body = [_row(row) for row in table.itertuples()]
    verdict_lines = _verdict_lines(verdicts)
    rule_zh = "停止規則（futures_liquid、h = 10、P1+P2、產業中性毛年化 >= 2 x 成本拖累）："
    rule_en = "Stop rule (futures_liquid, h = 10, P1+P2, industry-neutral gross >= 2 x cost drag):"
    return "\n".join(
        [
            "# §13.17 步驟 0：殘差 alpha 幅度與成本診斷 / Step 0: residual alpha vs costs",
            "",
            "## 中文",
            "",
            " ".join(ZH_NOTES),
            "",
            rule_zh,
            *verdict_lines,
            "",
            *_header(ZH_COLUMNS),
            *body,
            "",
            "## English",
            "",
            " ".join(EN_NOTES),
            "",
            rule_en,
            *verdict_lines,
            "",
            *_header(EN_COLUMNS),
            *body,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        action="append",
        required=True,
        help="targets.parquet of the same run (residual_hNN columns)",
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--taifex-root", type=Path, help=f"default <artifacts>/{DEFAULT_TAIFEX}")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--min-volume", type=float, default=100.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    taifex_root = args.taifex_root or args.artifacts_root / DEFAULT_TAIFEX
    output_root = args.output_root or runner.default_output_root(
        args.artifacts_root, "exposure-step0"
    )
    report = run(
        args.predictions,
        args.targets,
        args.model,
        args.prices,
        taifex_root,
        args.reference,
        output_root,
        min_volume=args.min_volume,
        lookback=args.lookback,
    )
    print(
        json.dumps(
            {"output_root": str(output_root), "stop_rule": report["stop_rule"]},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
