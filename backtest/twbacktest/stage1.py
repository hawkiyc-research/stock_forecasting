"""Run the pre-registered stage 1 backtest (experiment protocol section 13.4).

Writes report.json (selection, gate, sensitivities), grid.csv (every
configuration) and a bilingual summary.md to a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import data, engine
from twbacktest.costs import primary_scenario, scenarios
from twbacktest.runner import (
    MAX_WORKERS,
    clean,
    default_output_root,
    default_workers,
    number,
    percent,
    prepare_inputs,
    run_tasks,
)

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "stage1.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
KEY_COLUMNS = ["split", "market", "signal", "kind", "h", "k", "version", "discount",
               "slippage_multiplier", "turnover_mode"]

_STATE: dict[str, Any] = {}


def _initialize(config: dict[str, Any], inputs: dict[str, Any], indexes: dict, totals: dict):
    _STATE.update(config=config, inputs=inputs, indexes=indexes, totals=totals, returns={},
                  universes={})


def _returns(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in _STATE["returns"]:
        _STATE["returns"][split] = data.load_returns(
            _STATE["inputs"]["alpha_baseline"], _STATE["inputs"]["direction_baseline"],
            _STATE["indexes"][split], _STATE["totals"][split],
        )
    return _STATE["returns"][split]


def _universe(split: str, market: str) -> engine.Universe:
    key = (split, market)
    if key not in _STATE["universes"]:
        codes = tuple(data.MARKETS.index(name) for name in _STATE["config"]["markets"][market])
        _STATE["universes"][key] = engine.universe(_STATE["indexes"][split], codes)
    return _STATE["universes"][key]


def run_task(task: tuple[str, str, str]) -> list[dict[str, Any]]:
    """Every holding period, K, version and cost scenario of one split, signal and market."""
    split, signal, market = task
    config = _STATE["config"]
    index = _STATE["indexes"][split]
    spec = config["signals"][signal]
    scores = data.load_signal(spec, inputs=_STATE["inputs"], index=index,
                              total_rows=_STATE["totals"][split])
    absolute, alpha = _returns(split)
    group = _universe(split, market)
    rows = []
    for h in config["holding_days"]:
        for row in engine.evaluate(
            scores[:, h - 1], absolute[:, h - 1], alpha[:, h - 1], index, group, h=h,
            top_k=config["top_k"], scenario_list=scenarios(config["costs"]),
            costs=config["costs"], annualization_days=config["annualization_days"],
            bootstrap=config["bootstrap"],
        ):
            rows.append({"split": split, "market": market, "signal": signal,
                         "kind": spec["kind"], "h": h, **row})
    return rows


def run_grid(config: dict[str, Any], artifacts: Path, workers: int | None) -> tuple:
    inputs, indexes, totals = prepare_inputs(config, artifacts)
    tasks = [(split, signal, market) for split in config["splits"]
             for signal in config["signals"] for market in config["markets"]]
    workers = workers or default_workers(len(tasks))
    rows = run_tasks(tasks, run_task, initializer=_initialize,
                     initargs=(config, inputs, indexes, totals), workers=workers)
    grid = pd.DataFrame(rows).sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    return grid, indexes, workers


def _scenario_mask(grid: pd.DataFrame, scenario) -> pd.Series:
    return ((grid["discount"] == scenario.discount)
            & (grid["slippage_multiplier"] == scenario.slippage_multiplier)
            & (grid["turnover_mode"] == scenario.turnover))


def _match(grid: pd.DataFrame, row: pd.Series, split: str) -> pd.Series | None:
    mask = grid["split"] == split
    for column in KEY_COLUMNS[1:]:
        mask &= grid[column] == row[column]
    matches = grid[mask]
    return matches.iloc[0] if len(matches) else None


def select_and_gate(grid: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Validation-only selection of one primary configuration per market, then the gate."""
    selection, gate = config["selection"], config["gate"]
    primary = primary_scenario(config["costs"])
    candidates = grid[(grid["split"] == selection["split"])
                      & (grid["kind"] == selection["signal_kind"])
                      & (grid["version"] == selection["version"])
                      & _scenario_mask(grid, primary)]
    metric = selection["metric"]
    result = {}
    for market in config["markets"]:
        pool = candidates[(candidates["market"] == market) & np.isfinite(candidates[metric])]
        if pool.empty:
            result[market] = {"selected": None}
            continue
        best = pool.sort_values([metric, "signal", "h", "k"],
                                ascending=[False, True, True, True]).iloc[0]
        test = _match(grid, best, "test")
        passed = bool(
            best["net_excess_low"] > gate["validation_lower_bound_above"]
            and test is not None and test["net_excess_mean"] > gate["test_mean_above"]
        )
        result[market] = {
            "selected": {key: best[key] for key in ("signal", "h", "k")},
            "candidates": len(pool),
            "validation": best.to_dict(),
            "test": None if test is None else test.to_dict(),
            "passed": passed,
        }
    return result


def sensitivities(grid: pd.DataFrame, selection: dict[str, Any]) -> dict[str, Any]:
    """Every cost scenario and version of each primary configuration, both splits."""
    output = {}
    for market, item in selection.items():
        chosen = item.get("selected")
        if not chosen:
            continue
        rows = grid[(grid["market"] == market) & (grid["signal"] == chosen["signal"])
                    & (grid["h"] == chosen["h"]) & (grid["k"] == chosen["k"])]
        output[market] = rows[[*KEY_COLUMNS, "net_excess_annual", "net_excess_annual_low",
                               "net_excess_annual_high", "net_excess_ir", "sharpe",
                               "max_drawdown"]].to_dict("records")
    return output


def best_per_signal(grid: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive: each signal's validation-best (h, K) per market, with its test values."""
    primary = primary_scenario(config["costs"])
    base = grid[(grid["version"] == "long_only") & _scenario_mask(grid, primary)]
    output = []
    for (market, signal), rows in base[base["split"] == "validation"].groupby(
        ["market", "signal"], sort=True
    ):
        rows = rows[np.isfinite(rows["net_excess_ir"])]
        if rows.empty:
            continue
        best = rows.sort_values(["net_excess_ir", "h", "k"], ascending=[False, True, True]).iloc[0]
        test = _match(grid, best, "test")
        output.append({
            "market": market, "signal": signal, "kind": best["kind"], "h": int(best["h"]),
            "k": int(best["k"]),
            "validation_net_excess_annual": best["net_excess_annual"],
            "validation_net_excess_ir": best["net_excess_ir"],
            "test_net_excess_annual": None if test is None else test["net_excess_annual"],
            "test_net_excess_ir": None if test is None else test["net_excess_ir"],
            "test_max_drawdown": None if test is None else test["max_drawdown"],
        })
    return output


def summary_markdown(report: dict[str, Any]) -> str:
    primary = report["config"]["costs"]["primary"]
    lines = ["# 台股回測第一階段摘要 / Taiwan Backtest Stage 1 Summary", "", "## 中文", "",
             f"- 主要成本情境：手續費折扣 {primary['discount']:g}、滑價 "
             f"{primary['slippage_multiplier']:g} 倍、每批全額換股。",
             "- 淨超額報酬：扣除成本後，相對同日全市場等權（同一持有窗口、不計成本）的年化值。",
             "- 主要設定只用 validation 選出；test 只執行一次。", "",
             "| 市場 | 訊號 | h | K | validation 淨超額 [95% 區間] | test 淨超額 [95% 區間] "
             "| test IR | test 最大回撤 | 通過 |",
             "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |"]
    rows_en = []
    for market, item in report["selection"].items():
        chosen = item.get("selected")
        if not chosen:
            continue
        val, test = item["validation"], item["test"] or {}
        cells = (
            f"| {market} | {chosen['signal']} | {chosen['h']} | {chosen['k']} | "
            f"{percent(val['net_excess_annual'])} [{percent(val['net_excess_annual_low'])}, "
            f"{percent(val['net_excess_annual_high'])}] | "
            f"{percent(test.get('net_excess_annual'))} "
            f"[{percent(test.get('net_excess_annual_low'))}, "
            f"{percent(test.get('net_excess_annual_high'))}] | "
            f"{number(test.get('net_excess_ir'))} | {percent(test.get('max_drawdown'))} | "
        )
        lines.append(cells + ("是 |" if item["passed"] else "否 |"))
        rows_en.append(cells + ("yes |" if item["passed"] else "no |"))
    lines += ["", "## English", "",
              f"- Primary costs: commission discount {primary['discount']:g}, slippage "
              f"x{primary['slippage_multiplier']:g}, full replacement each batch.",
              "- Net excess: annualized return after costs over the same-date, same-window, "
              "cost-free equal-weight population.",
              "- Primary configurations are chosen on validation only; test runs once.", "",
              "| Market | Signal | h | K | Validation net excess [95% CI] | Test net excess "
              "[95% CI] | Test IR | Test max drawdown | Passed |",
              "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |", *rows_en]
    return "\n".join(lines) + "\n"


def run(config_path: Path, artifacts: Path, output_root: Path, workers: int | None) -> dict:
    config = json.loads(config_path.read_text())
    grid, indexes, used_workers = run_grid(config, artifacts, workers)
    selection = select_and_gate(grid, config)
    report = {
        "schema_version": "1.0",
        "kind": "stage1-tw-signal-backtest",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "workers": used_workers,
        "data": {
            split: {
                "dates": len(index.dates), "first_date": index.dates[0],
                "last_date": index.dates[-1],
                "rows": {market: int((index.market_codes == code).sum())
                         for code, market in enumerate(data.MARKETS)},
            }
            for split, index in indexes.items()
        },
        "configurations": len(grid),
        "selection": selection,
        "gate_passed_markets": [market for market, item in selection.items()
                                if item.get("passed")],
        "sensitivity": sensitivities(grid, selection),
        "best_per_signal": best_per_signal(grid, config),
    }
    report = clean(report)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path, help="New directory for this run.")
    parser.add_argument("--workers", type=int, help=f"Process workers, 1-{MAX_WORKERS}.")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    output_root = args.output_root or default_output_root(args.artifacts_root, "stage1")
    report = run(args.config, args.artifacts_root, output_root, args.workers)
    print(json.dumps({"output_root": str(output_root),
                      "gate_passed_markets": report["gate_passed_markets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
