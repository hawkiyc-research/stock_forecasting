"""Descriptive diagnostic: ranking skill and decile gross returns inside the stock-futures universe.

Not a pre-registered decision. For each model, market, horizon (1 and 10) and segment (every
ranked stock; stocks with a TAIFEX stock-futures contract listed in the regular session on the
signal date; those whose near-month contract is liquid), it reports the daily Spearman Rank IC
against the label return (next open to the close h sessions later) and the gross return of the
top and bottom score deciles, per period. Eligibility comes from the official TAIFEX daily
download (``twbacktest.taifex download`` for every contract) and the current underlying list,
so stocks whose futures were delisted before the list date are missed (survivorship bias).
The work is a few thousand small per-date groups, so it runs serially on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats, taifex
from twbacktest.walkforward_data import (
    build_holding_returns,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_TAIFEX = "diagnostics/taifex-all"
HORIZONS = (1, 10)
SEGMENTS = ("all", "futures_listed", "futures_liquid")
MARKET_GROUPS = {"TWSE": ("TWSE",), "TPEX": ("TPEX",), "TW": ("TWSE", "TPEX")}
PERIODS = {"P1": ("2023-01-02", "2025-06-01"), "P2": ("2025-06-01", "2026-06-01")}
DECISION_PERIOD = "P1+P2"
MIN_NAMES = 10
BOOTSTRAP = {"replicates": 1000, "seed": 11}


def eligibility_frame(root: Path, *, min_volume: float, lookback: int) -> tuple[pd.DataFrame, dict]:
    """Per (date, symbol): listed in the regular session that day, and liquid near month."""
    futures, listing = taifex.load(root)
    daily, audit = taifex.eligibility(futures, listing, min_volume=min_volume, lookback=lookback)
    daily = daily[["date", "symbol", "liquid"]].copy()
    daily["listed"] = True
    return daily, audit


def segment_masks(
    frame: pd.DataFrame, eligibility: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Row masks of the three segments for a prediction frame (symbol, signal_date)."""
    keyed = frame[["symbol", "signal_date"]].copy()
    keyed["date"] = keyed["signal_date"].dt.strftime("%Y-%m-%d")
    merged = keyed.merge(eligibility, on=["date", "symbol"], how="left")
    listed = merged["listed"].fillna(False).to_numpy(dtype=bool)
    liquid = merged["liquid"].fillna(False).to_numpy(dtype=bool)
    return {"all": np.ones(len(frame), dtype=bool), "futures_listed": listed,
            "futures_liquid": listed & liquid}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_table(
    frame: pd.DataFrame,
    scores: np.ndarray,
    log_return: np.ndarray,
    log_alpha: np.ndarray,
    mask: np.ndarray,
    market_codes: tuple[str, ...],
) -> pd.DataFrame:
    """One row per signal date: Rank IC, decile gross returns and the universe mean."""
    rows = np.flatnonzero(mask & frame["market"].isin(market_codes).to_numpy()
                          & np.isfinite(scores) & np.isfinite(log_return))
    if rows.size == 0:
        return pd.DataFrame()
    dates = frame["signal_date"].to_numpy()[rows]
    order = np.argsort(dates, kind="stable")
    rows, dates = rows[order], dates[order]
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    records = []
    for start, stop in pairwise(bounds):
        group = rows[start:stop]
        if group.size < MIN_NAMES:
            continue
        score = scores[group]
        simple = np.expm1(log_return[group])
        alpha = np.expm1(log_alpha[group]) if log_alpha is not None else np.full(group.size, np.nan)
        decile = max(1, math.ceil(group.size / 10))
        ranked = np.argsort(-score, kind="stable")
        top, bottom = ranked[:decile], ranked[-decile:]
        records.append({
            "signal_date": pd.Timestamp(dates[start]),
            "names": int(group.size),
            "rank_ic": _spearman(score, log_return[group]),
            "top_return": float(simple[top].mean()),
            "bottom_return": float(simple[bottom].mean()),
            "universe_return": float(simple.mean()),
            "top_alpha": float(np.nanmean(alpha[top])),
            "bottom_alpha": float(np.nanmean(alpha[bottom])),
        })
    return pd.DataFrame(records)


def period_mask(dates: pd.Series, period: str) -> np.ndarray:
    if period == DECISION_PERIOD:
        return np.any([period_mask(dates, name) for name in PERIODS], axis=0)
    start, end = PERIODS[period]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()


def summarize(daily: pd.DataFrame, *, h: int) -> dict[str, Any]:
    """Period statistics of the daily table: IC with a block bootstrap, deciles and spread."""
    if daily.empty:
        return {"days": 0}
    ic = daily["rank_ic"].to_numpy()
    block = 5 if h == 1 else 20
    mean, low, high = stats.bootstrap_mean(ic, block=block, replicates=BOOTSTRAP["replicates"],
                                           seed=BOOTSTRAP["seed"])
    finite = ic[np.isfinite(ic)]
    spread = (daily["top_return"] - daily["bottom_return"]).to_numpy()
    s_mean, s_low, s_high = stats.bootstrap_mean(spread, block=max(5, 2 * h),
                                                 replicates=BOOTSTRAP["replicates"],
                                                 seed=BOOTSTRAP["seed"])
    per_year = 252 / h
    return {
        "days": len(daily),
        "names_mean": float(daily["names"].mean()),
        "rank_ic": mean,
        "rank_ic_low": low,
        "rank_ic_high": high,
        "rank_icir": float(finite.mean() / finite.std(ddof=1)) if finite.size > 2 else float("nan"),
        "positive_share": float(np.mean(finite > 0)) if finite.size else float("nan"),
        "top_return_mean": float(daily["top_return"].mean()),
        "bottom_return_mean": float(daily["bottom_return"].mean()),
        "universe_return_mean": float(daily["universe_return"].mean()),
        "top_excess_mean": float((daily["top_return"] - daily["universe_return"]).mean()),
        "top_alpha_mean": float(daily["top_alpha"].mean()),
        "bottom_alpha_mean": float(daily["bottom_alpha"].mean()),
        "spread_mean": s_mean,
        "spread_low": s_low,
        "spread_high": s_high,
        "spread_annual": s_mean * per_year,
        "spread_annual_low": s_low * per_year,
        "spread_annual_high": s_high * per_year,
    }


def run(
    predictions: list[Path],
    models: list[str],
    prices: Path,
    taifex_root: Path,
    output_root: Path,
    *,
    min_volume: float,
    lookback: int,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    price_frame = load_prices(prices)
    eligibility, audit = eligibility_frame(taifex_root, min_volume=min_volume, lookback=lookback)
    rows, dailies = [], []
    for path, label in zip(predictions, models, strict=True):
        frame = load_predictions(path, label)
        inputs = build_inputs(frame, price_frame)
        log_return, log_alpha = build_holding_returns(frame, price_frame)
        masks = segment_masks(frame, eligibility)
        for market, codes in MARKET_GROUPS.items():
            for segment, mask in masks.items():
                for h in HORIZONS:
                    column = inputs.horizons.index(h)
                    daily = daily_table(frame, inputs.scores[:, column], log_return[:, h - 1],
                                        log_alpha[:, h - 1], mask, codes)
                    if daily.empty:
                        continue
                    daily = daily.assign(model=label, market=market, segment=segment, h=h)
                    dailies.append(daily)
                    for period in (*PERIODS, DECISION_PERIOD):
                        part = daily[period_mask(daily["signal_date"], period)]
                        rows.append({"model": label, "market": market, "segment": segment,
                                     "h": h, "period": period, **summarize(part, h=h)})
    grid = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "futures-universe-diagnostic",
        "predictions": {label: str(path) for path, label in zip(predictions, models, strict=True)},
        "prices": str(prices),
        "taifex_root": str(taifex_root),
        "min_volume": min_volume,
        "lookback": lookback,
        "eligibility_audit": audit,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(grid), encoding="utf-8")
    return report


def _row(row) -> str:
    return (
        f"| {row.model} | {row.market} | {row.segment} | {int(row.h)} | {row.period} | "
        f"{row.names_mean:.0f} | {runner.number(row.rank_ic)} "
        f"[{runner.number(row.rank_ic_low)}, {runner.number(row.rank_ic_high)}] | "
        f"{runner.number(row.rank_icir)} | {runner.percent(row.top_return_mean)} | "
        f"{runner.percent(row.bottom_return_mean)} | {runner.percent(row.universe_return_mean)} | "
        f"{runner.percent(row.spread_mean)} "
        f"[{runner.percent(row.spread_low)}, {runner.percent(row.spread_high)}] | "
        f"{runner.percent(row.spread_annual)} |"
    )


def summary_markdown(grid: pd.DataFrame) -> str:
    table = grid[grid["days"] > 0].sort_values(["model", "market", "segment", "h", "period"])
    body = [_row(row) for row in table.itertuples()]
    header_zh = (
        "| 模型 | 市場 | 區段 | h | 期間 | 平均檔數 | Rank IC [95%] | ICIR | 前十分位毛報酬/批 "
        "| 後十分位毛報酬/批 | 宇宙平均/批 | 前減後價差/批 [95%] | 價差年化 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    header_en = (
        "| Model | Market | Segment | h | Period | Mean names | Rank IC [95%] | ICIR "
        "| Top decile gross / batch | Bottom decile gross / batch | Universe mean / batch "
        "| Top minus bottom / batch [95%] | Spread annualized |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    return "\n".join([
        "# 股票期貨宇宙診斷 / Stock-futures universe diagnostic", "",
        "## 中文", "",
        "描述性分析（非事先登記的判定）。區段：all = 該市場全部有分數的股票；"
        "futures_listed = 訊號日當天有股票期貨契約在一般時段掛牌；"
        "futures_liquid = 近月契約 20 日平均成交量達門檻。"
        "報酬為標籤定義的持有（次一開盤買進、h 日後收盤賣出），未扣成本。", "",
        *header_zh, *body, "",
        "## English", "",
        "Descriptive analysis (not a pre-registered decision). Segments: all = every scored "
        "stock of the market; futures_listed = a stock-futures contract is listed in the regular "
        "session on the signal date; futures_liquid = the near-month contract's 20-session mean "
        "volume reaches the threshold. Returns follow the label (buy at the next open, sell at "
        "the close h sessions later), before costs.", "",
        *header_en, *body, "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--taifex-root", type=Path, help=f"default <artifacts>/{DEFAULT_TAIFEX}")
    parser.add_argument("--min-volume", type=float, default=100.0,
                        help="near-month standard-contract volume (20-session mean) for 'liquid'")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    taifex_root = args.taifex_root or args.artifacts_root / DEFAULT_TAIFEX
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "futures-universe")
    report = run(args.predictions, args.model, args.prices, taifex_root, output_root,
                 min_volume=args.min_volume, lookback=args.lookback)
    print(json.dumps({"output_root": str(output_root), "audit": report["eligibility_audit"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
