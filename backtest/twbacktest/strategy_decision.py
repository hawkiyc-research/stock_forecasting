"""Final selection and the P3 confirmation of protocol 13.20 (items 5 and 6).

Reads a ``scenario_suite`` output (``grid.csv`` and ``daily.csv``) and applies the
pre-registered rules without re-running any simulation:

* **Selection** (item 5): among the final, primary-slippage, fractional cells of the selection
  profile, market and construction, the (run, strategy) with the highest decision-period
  (``P1+P2``) net Sharpe is the selected cell. The S-A K and m were already selected inside
  the suite (TWSE, P1).
* **H-J** (item 6): the selected cell passes when, in the confirmation period (``P3``) under
  the same profile, (i) the annualized net return is above zero and (ii) the maximum drawdown
  is not deeper than the decision period's. Both hold: the strategy may enter paper trading;
  otherwise it stays in research, with no re-selection.
* **Chip increment at the strategy level** (informational): for every strategy and market,
  the paired daily net difference between the treatment run (F2) and the base run (F0) under
  the selection profile, with the block-bootstrap interval over the decision period and the
  point estimate over the confirmation period. An increment is claimed only when the
  decision-period lower bound is above zero.

Everything is read from the files; ``--base`` / ``--treat`` name the runs as the suite's
``--label`` options did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats
from twbacktest.topk_dropout import DECISION_PERIOD, period_masks

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "strategy_decision.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
STRATEGY_LABELS = {
    "md_ls": ("S1 多日多空", "S1 multi-day long-short"),
    "md_ls_tx": ("S2 S1 + 台指期", "S2 S1 + TX overlay"),
    "wk_long": ("S-A 週頻低換手多頭", "S-A weekly low-turnover long"),
}
# Columns of a selected cell reported in the summary and the report.
CELL_COLUMNS = (
    "net_return_annual", "net_excess_annual_low", "net_excess_annual_high", "sharpe",
    "max_drawdown", "turnover", "cost_drag_annual", "hit_rate", "days",
    "index_total_return_annual", "h", "k", "buffer", "weekly_turnover",
    "liquidity_blocked_share", "ineligible_share", "borrow_failed_share",
)


# --------------------------------------------------------------------------- selection


def _cells(grid: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    """The candidate rows: final stage, primary slippage, one profile, market, capital and
    construction, limited to the registered strategies."""
    frame = grid[(grid["stage"] == "final") & grid["primary_slippage"].astype(bool)
                 & (grid["capital"] == rule.get("capital", "fractional"))
                 & (grid["profile"] == rule["profile"]) & (grid["market"] == rule["market"])
                 & grid["scenario"].isin(rule["scenarios"])]
    if "score_variant" in frame.columns and rule.get("score_variant"):
        frame = frame[frame["score_variant"] == rule["score_variant"]]
    if "leg_matching" in frame.columns:
        frame = frame[frame["leg_matching"].fillna("none") == rule.get("leg_matching", "none")]
    if "failure_multiplier" in frame.columns:
        # The execution-layer sensitivities share the final stage only through their primary
        # point; the grid marks the others as stage "execution", so nothing else is needed.
        pass
    return frame.copy()


def select_cell(grid: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    rule = config["selection"]
    cells = _cells(grid, rule)
    decision = cells[cells["period"] == rule["period"]]
    if decision.empty:
        raise ValueError("No decision-period rows match the selection rule")
    metric = rule["metric"]
    finite = decision[np.isfinite(decision[metric].astype(float))]
    if finite.empty:
        raise ValueError(f"The selection metric {metric!r} is not finite in any candidate")
    order = sorted(
        finite.itertuples(),
        key=lambda row: (-float(getattr(row, metric)), str(row.model), str(row.scenario)),
    )
    best = order[0]
    candidates = [
        {"model": str(row.model), "scenario": str(row.scenario),
         metric: float(getattr(row, metric)),
         "net_return_annual": float(row.net_return_annual),
         "max_drawdown": float(row.max_drawdown)}
        for row in order
    ]
    return {"model": str(best.model), "scenario": str(best.scenario), "metric": metric,
            "value": float(getattr(best, metric)), "candidates": candidates}


def _cell_record(row: pd.Series) -> dict[str, Any]:
    return {column: runner.clean(row[column]) for column in CELL_COLUMNS if column in row.index}


def cell_table(grid: pd.DataFrame, config: dict[str, Any], chosen: dict[str, Any]) -> dict:
    """The selected cell in every period, under every profile, on every market (fractional,
    primary slippage) — the rows the reports print."""
    rule = config["selection"]
    base = grid[(grid["stage"] == "final") & grid["primary_slippage"].astype(bool)
                & (grid["capital"] == rule.get("capital", "fractional"))
                & (grid["model"] == chosen["model"]) & (grid["scenario"] == chosen["scenario"])]
    if "score_variant" in base.columns and rule.get("score_variant"):
        base = base[base["score_variant"] == rule["score_variant"]]
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for row in base.itertuples(index=False):
        record = pd.Series(row._asdict())
        table.setdefault(str(record["profile"]), {}).setdefault(str(record["market"]), {})[
            str(record["period"])] = _cell_record(record)
    return table


def confirm(grid: pd.DataFrame, config: dict[str, Any], chosen: dict[str, Any]) -> dict:
    """H-J on the selected cell."""
    rule = config["selection"]
    check = config["confirmation"]
    cells = _cells(grid, rule)
    mine = cells[(cells["model"] == chosen["model"]) & (cells["scenario"] == chosen["scenario"])]
    decision = mine[mine["period"] == rule["period"]]
    forward = mine[mine["period"] == check["period"]]
    if decision.empty or forward.empty:
        return {"period": check["period"], "evaluated": False,
                "reason": "the selected cell has no row in the confirmation period"}
    decision_row, forward_row = decision.iloc[0], forward.iloc[0]
    net = float(forward_row["net_return_annual"])
    forward_dd = float(forward_row["max_drawdown"])
    decision_dd = float(decision_row["max_drawdown"])
    positive = bool(np.isfinite(net) and net > 0)
    # Drawdowns are negative numbers: "not deeper" means not more negative.
    drawdown_ok = bool(np.isfinite(forward_dd) and np.isfinite(decision_dd)
                       and forward_dd >= decision_dd)
    passed = positive and drawdown_ok
    return {
        "period": check["period"], "evaluated": True,
        "net_return_annual": net,
        "net_excess_annual_low": float(forward_row["net_excess_annual_low"]),
        "net_excess_annual_high": float(forward_row["net_excess_annual_high"]),
        "sharpe": float(forward_row["sharpe"]),
        "max_drawdown": forward_dd, "decision_max_drawdown": decision_dd,
        "days": int(forward_row["days"]),
        "positive_return": positive, "drawdown_not_deeper": drawdown_ok,
        "passed": passed,
        "status": ("paper_trading_candidate" if passed else "research"),
    }


# --------------------------------------------------------------------------- increment


def paired_increment(daily: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    inc = config["increment"]
    rule = config["selection"]
    boot = inc["bootstrap"]
    days = int(config["annualization_days"])
    frame = daily[daily["profile"] == rule["profile"]]
    if "score_variant" in frame.columns and rule.get("score_variant"):
        frame = frame[frame["score_variant"] == rule["score_variant"]]
    if "leg_matching" in frame.columns:
        frame = frame[frame["leg_matching"].fillna("none") == rule.get("leg_matching", "none")]
    out = []
    for scenario in rule["scenarios"]:
        for market in sorted(frame["market"].unique()):
            part = frame[(frame["scenario"] == scenario) & (frame["market"] == market)]
            base = part[part["model"] == inc["base"]][["interval_start", "net"]]
            treat = part[part["model"] == inc["treat"]][["interval_start", "net"]]
            if base.empty or treat.empty:
                continue
            joined = base.merge(treat, on="interval_start", suffixes=("_base", "_treat"),
                                validate="one_to_one").sort_values("interval_start")
            dates = pd.DatetimeIndex(pd.to_datetime(joined["interval_start"]))
            masks = period_masks(dates, config["periods"], config.get("decision_periods"))
            diff = (joined["net_treat"] - joined["net_base"]).to_numpy(dtype=np.float64)
            record: dict[str, Any] = {"scenario": scenario, "market": market,
                                      "base": inc["base"], "treat": inc["treat"]}
            for period in (*config["periods"], DECISION_PERIOD):
                mask = masks[period]
                if not mask.any():
                    continue
                mean, low, high = stats.bootstrap_mean(
                    diff[mask], block=int(boot["block"]), replicates=int(boot["replicates"]),
                    seed=int(boot["seed"]))
                record[period] = {"days": int(np.isfinite(diff[mask]).sum()),
                                  "diff_annual": mean * days, "low": low * days,
                                  "high": high * days}
            decision = record.get(DECISION_PERIOD)
            record["increment"] = bool(decision and np.isfinite(decision["low"])
                                       and decision["low"] > 0)
            out.append(record)
    return out


# --------------------------------------------------------------------------- run


def run(suite_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grid = pd.read_csv(suite_root / "grid.csv")
    daily = pd.read_csv(suite_root / "daily.csv")
    chosen = select_cell(grid, config)
    verdict = confirm(grid, config, chosen)
    table = cell_table(grid, config, chosen)
    increments = paired_increment(daily, config)
    report = {
        "kind": "strategy-decision-p3",
        "suite": str(suite_root), "config": str(config_path),
        "selection": chosen, "confirmation": verdict, "cells": table,
        "increment": increments,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(report, config), encoding="utf-8")
    return report


def _label(scenario: str) -> str:
    zh, en = STRATEGY_LABELS.get(scenario, (scenario, scenario))
    return f"{zh} / {en}"


def summary_markdown(report: dict[str, Any], config: dict[str, Any]) -> str:
    pct, num = runner.percent, runner.number
    chosen, verdict = report["selection"], report["confirmation"]
    rule = config["selection"]
    lines = ["# 最後選定與 P3 確認 / Final selection and P3 confirmation", ""]
    lines.append(f"- 選定規則 / rule: {rule['profile']} 成本、{rule['market']}、"
                 f"{rule.get('capital', 'fractional')}、{rule['period']} 的 "
                 f"{chosen['metric']} 最高")
    lines.append(f"- **選定格 / selected cell: {chosen['model']} x "
                 f"{_label(chosen['scenario'])}** ({chosen['metric']} = {num(chosen['value'])})")
    lines.append("")
    lines.append("| 候選 / candidate | 策略 | " f"{chosen['metric']} | 年化淨報酬 | 最大回撤 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in chosen["candidates"]:
        lines.append(f"| {item['model']} | {_label(item['scenario'])} | "
                     f"{num(item[chosen['metric']])} | {pct(item['net_return_annual'])} | "
                     f"{pct(item['max_drawdown'])} |")
    lines.append("")
    lines.append("## H-J（P3 確認）/ H-J (P3 confirmation)")
    lines.append("")
    if not verdict.get("evaluated"):
        lines.append(f"- 未判定 / not evaluated: {verdict.get('reason')}")
    else:
        word = "通過 / passed" if verdict["passed"] else "未通過 / not passed"
        lines.append(
            f"- **{word}**: {verdict['period']} 年化淨報酬 {pct(verdict['net_return_annual'])} "
            f"[{pct(verdict['net_excess_annual_low'])}, {pct(verdict['net_excess_annual_high'])}] "
            f"（{verdict['days']} 日，Sharpe {num(verdict['sharpe'])}）；最大回撤 "
            f"{pct(verdict['max_drawdown'])} 對 {rule['period']} 的 "
            f"{pct(verdict['decision_max_drawdown'])}")
        lines.append(f"- (i) 淨報酬 > 0: {verdict['positive_return']}; (ii) 回撤不更深 / drawdown "
                     f"not deeper: {verdict['drawdown_not_deeper']}; 狀態 / status: "
                     f"{verdict['status']}")
    lines.append("")
    lines.append("## 選定格各期 / The selected cell by period")
    lines.append("")
    lines.append("| 成本檔 | 市場 | 期間 | 年化淨報酬 | 95% 區間 | Sharpe | 最大回撤 | 換手 "
                 "| 成本拖累 | 日數 | TAIEX 報酬指數 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for profile, markets in report["cells"].items():
        for market, periods in markets.items():
            for period, cell in periods.items():
                lines.append(
                    f"| {profile} | {market} | {period} | {pct(cell.get('net_return_annual'))} | "
                    f"[{pct(cell.get('net_excess_annual_low'))}, "
                    f"{pct(cell.get('net_excess_annual_high'))}] | {num(cell.get('sharpe'))} | "
                    f"{pct(cell.get('max_drawdown'))} | {pct(cell.get('turnover'))} | "
                    f"{pct(cell.get('cost_drag_annual'))} | {cell.get('days')} | "
                    f"{pct(cell.get('index_total_return_annual'))} |")
    lines.append("")
    lines.append("## 策略層的籌碼增量（資訊性）/ Chip increment at the strategy level "
                 "(informational)")
    lines.append("")
    inc = config["increment"]
    lines.append(f"{inc['treat']} - {inc['base']} 的每日淨報酬配對差，年化；區間為 "
                 f"{DECISION_PERIOD} 的區塊 bootstrap。")
    lines.append("")
    lines.append("| 策略 | 市場 | P1+P2 差 | 95% 區間 | P3 差 | P3 區間 | 增量 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    forward = config["confirmation"]["period"]
    for item in report["increment"]:
        decision, p3 = item.get(DECISION_PERIOD, {}), item.get(forward, {})
        lines.append(
            f"| {_label(item['scenario'])} | {item['market']} | "
            f"{pct(decision.get('diff_annual'))} | [{pct(decision.get('low'))}, "
            f"{pct(decision.get('high'))}] | {pct(p3.get('diff_annual'))} | "
            f"[{pct(p3.get('low'))}, {pct(p3.get('high'))}] | "
            f"{'是 / yes' if item['increment'] else '否 / no'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, required=True,
                        help="a scenario_suite output directory (grid.csv, daily.csv)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "strategy-decision")
    report = run(args.suite, args.config, output_root)
    print(json.dumps({"output_root": str(output_root), "selection": report["selection"],
                      "confirmation": report["confirmation"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
