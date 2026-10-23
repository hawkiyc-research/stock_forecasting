"""Label-aligned holding backtest of walk-forward ranking scores (secondary analysis).

Not a pre-registered decision of protocol 13.13: it reports whether the ranking edge survives
when the position is held exactly the way the label is defined (buy at the next open, sell at
the close h sessions later, h start sleeves, the stage 1 engine and Taiwan costs, day-trade tax
for h = 1). The top-k dropout rule of 13.13 item 3 remains the registered H-B test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, runner
from twbacktest.costs import primary_scenario, scenarios
from twbacktest.data import MARKETS
from twbacktest.walkforward_data import (
    build_holding_returns,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "holding_walkforward.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DECISION_PERIOD = "P1+P2"


def period_frames(
    predictions: pd.DataFrame, periods: dict[str, list[str]]
) -> dict[str, pd.DataFrame]:
    frames = {}
    union = np.zeros(len(predictions), dtype=bool)
    dates = predictions["signal_date"]
    for name, (start, end) in periods.items():
        mask = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()
        frames[name] = predictions[mask].reset_index(drop=True)
        union |= mask
    frames[DECISION_PERIOD] = predictions[union].reset_index(drop=True)
    return frames


def run_period(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    *,
    label: str,
    period: str,
) -> list[dict[str, Any]]:
    inputs = build_inputs(predictions, prices)
    log_return, log_alpha = build_holding_returns(predictions, prices)
    rows = []
    for market, names in config["markets"].items():
        codes = tuple(MARKETS.index(name) for name in names)
        group = engine.universe(inputs.index, codes)
        for h in config["holding_days"]:
            column = inputs.horizons.index(h)
            for row in engine.evaluate(
                inputs.scores[:, column],
                log_return[:, h - 1],
                log_alpha[:, h - 1],
                inputs.index,
                group,
                h=h,
                top_k=config["top_k"],
                scenario_list=scenarios(config["costs"]),
                costs=config["costs"],
                annualization_days=config["annualization_days"],
                bootstrap=config["bootstrap"],
            ):
                rows.append({"model": label, "period": period, "market": market, "h": h, **row})
    return rows


def _primary_mask(grid: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    primary = primary_scenario(config["costs"])
    return (
        (grid["discount"] == primary.discount)
        & (grid["slippage_multiplier"] == primary.slippage_multiplier)
        & (grid["turnover_mode"] == primary.turnover)
    )


def select(grid: pd.DataFrame, config: dict[str, Any], label: str) -> dict[str, Any]:
    rule = config["selection"]
    rows = grid[
        (grid["model"] == label)
        & (grid["market"] == rule["market"])
        & (grid["period"] == rule["period"])
        & (grid["version"] == rule["version"])
        & _primary_mask(grid, config)
    ]
    rows = rows[np.isfinite(rows[rule["metric"]])]
    if rows.empty:
        return {"h": None, "k": None}
    best = rows.sort_values([rule["metric"], "h", "k"], ascending=[False, True, True]).iloc[0]
    out = {
        "h": int(best["h"]),
        "k": int(best["k"]),
        "metric": rule["metric"],
        "value": float(best[rule["metric"]]),
        "market": rule["market"],
        "period": rule["period"],
    }
    for period in ("P2", DECISION_PERIOD):
        match = grid[
            (grid["model"] == label)
            & (grid["market"] == rule["market"])
            & (grid["period"] == period)
            & (grid["version"] == rule["version"])
            & (grid["h"] == out["h"])
            & (grid["k"] == out["k"])
            & _primary_mask(grid, config)
        ]
        if not match.empty:
            row = match.iloc[0]
            out[period] = {
                "net_excess_annual": float(row["net_excess_annual"]),
                "net_excess_annual_low": float(row["net_excess_annual_low"]),
                "net_excess_annual_high": float(row["net_excess_annual_high"]),
                "net_excess_ir": float(row["net_excess_ir"]),
                "gross_excess_mean": float(row["gross_excess_mean"]),
            }
    return out


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame) -> str:
    sel = report["selection"]
    lines = ["# 持有對齊回測摘要 / Holding-aligned backtest summary", ""]
    base = grid[
        (grid["version"] == "long_only")
        & (grid["turnover_mode"] == "conservative")
        & (grid["discount"] == 0.6)
        & (grid["slippage_multiplier"] == 1)
    ]
    base = base[(base["h"] == sel.get("h")) & (base["k"] == sel.get("k"))]
    rows = []
    for row in base.sort_values(["model", "market", "period"]).itertuples():
        low, high = (
            runner.percent(row.net_excess_annual_low),
            runner.percent(row.net_excess_annual_high),
        )
        gross = runner.percent(row.gross_excess_mean)
        rows.append(
            f"| {row.model} | {row.market} | {row.period} | {gross} | "
            f"{runner.percent(row.net_excess_annual)} | [{low}, {high}] | "
            f"{runner.number(row.net_excess_ir)} | {runner.percent(row.turnover)} |"
        )
    header_zh = (
        "| 模型 | 市場 | 期間 | 每批毛超額 | 年化淨超額 | 95% 區間 | IR | 換手 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    header_en = (
        "| Model | Market | Period | Gross excess per batch | Net excess (annual) | 95% CI "
        "| IR | Turnover |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    lines += [
        "## 中文",
        "",
        f"此為次要分析（非 §13.13 事先登記的判定）。選定 h = {sel.get('h')}、"
        f"K = {sel.get('k')}（{sel.get('market')} {sel.get('period')} 淨超額 IR 最高）。",
        "",
        *header_zh,
        *rows,
        "",
        "## English",
        "",
        "Secondary analysis (not a pre-registered decision of §13.13). "
        f"Selected h = {sel.get('h')}, K = {sel.get('k')} "
        f"(highest net excess IR on {sel.get('market')} {sel.get('period')}).",
        "",
        *header_en,
        *rows,
        "",
    ]
    return "\n".join(lines)


def run(
    config_path: Path,
    predictions: Path,
    model: str,
    prices: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    price_frame = load_prices(prices)
    frames = period_frames(load_predictions(predictions, model), config["periods"])
    grid_rows = []
    for period, frame in frames.items():
        if frame.empty:
            continue
        grid_rows.extend(run_period(frame, price_frame, config, label=model, period=period))
    grid = pd.DataFrame(grid_rows)
    selection = select(grid, config, model)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    report = {
        "config": str(config_path),
        "predictions": str(predictions),
        "model": model,
        "kind": "holding-aligned-secondary",
        "selection": selection,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, grid), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(DEFAULT_ARTIFACTS, "holding-wf")
    report = run(args.config, args.predictions, args.model, args.prices, output_root)
    print(
        json.dumps(
            {"output_root": str(output_root), "selection": report["selection"]},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
