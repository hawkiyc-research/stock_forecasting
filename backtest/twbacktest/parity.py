"""Check local Kronos inference against a RunPod export (experiment protocol section 13.7).

Writes report.json and a bilingual summary.md with the pre-registered criteria: the
share of date x market x horizon groups whose cross-sectional Spearman between local and
reference predictions is at least the threshold, and the differences in the open-to-open
Rank IC and top-decile excess of section 13.6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from twbacktest import data, holding, local_predictions, positions
from twbacktest.runner import clean, default_output_root, prepare_inputs
from twbacktest.stage1b import identity_checks

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "local_parity.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"


def compare(config: dict[str, Any], artifacts: Path, local_directory: Path) -> dict[str, Any]:
    split, reference = config["split"], config["reference"]
    run_config = {**config, "splits": [split]}
    inputs, indexes, totals = prepare_inputs(run_config, artifacts)
    checks = identity_checks(run_config, inputs, indexes, totals)
    index = indexes[split]
    export = data.export_directory(inputs["predictions_root"], reference["run"],
                                   reference["checkpoint"])
    if export.name != reference["export"]:
        raise ValueError(f"The newest complete reference export is {export.name}, "
                         f"not the pre-registered {reference['export']}")
    expected = data.load_signal({"kind": "alpha", "members": [[reference["run"],
                                                                reference["checkpoint"]]]},
                                inputs=inputs, index=index, total_rows=totals[split])
    local, run = local_predictions.load_local_scores(local_directory, index)
    if (run.get("run_id"), run["checkpoint"]["name"]) != (reference["run"],
                                                          reference["checkpoint"]):
        raise ValueError("The local run used another checkpoint than the reference")
    absolute, alpha = data.load_returns(inputs["alpha_baseline"], inputs["direction_baseline"],
                                        index, totals[split])
    table = positions.symbol_date_table(index)
    groups = {market: positions.date_groups(index, (data.MARKETS.index(market),))
              for market in config["rank_markets"]}
    criteria = config["criteria"]
    correlations = np.concatenate([
        holding.daily_rank_ic(local[:, h], expected[:, h], rows)
        for rows in groups.values() for h in range(local.shape[1])
    ])
    finite = correlations[np.isfinite(correlations)]
    group_fraction = float(np.mean(finite >= criteria["group_spearman_min"]))
    metrics = {}
    for h in config["horizons"]:
        labels = holding.open_to_open(absolute, alpha, index, table, h)
        simple = np.expm1(labels.stock)
        for market, rows in groups.items():
            values = {}
            for name, scores in (("local", local), ("reference", expected)):
                score = scores[:, h - 1]
                values[name] = {
                    "rank_ic": holding.rank_summary(
                        holding.daily_rank_ic(score, labels.alpha, rows))["mean"],
                    "top_decile": float(np.nanmean(holding.deciles(
                        score, simple, rows, config["decile_fraction"]).top)),
                }
            values["rank_ic_difference"] = abs(values["local"]["rank_ic"]
                                               - values["reference"]["rank_ic"])
            values["top_decile_difference"] = abs(values["local"]["top_decile"]
                                                  - values["reference"]["top_decile"])
            metrics[f"{market}_h{h}"] = values
    difference = np.abs(local - expected)
    passed = {
        "group_spearman": group_fraction >= criteria["group_fraction_min"],
        "rank_ic": all(item["rank_ic_difference"] <= criteria["rank_ic_tolerance"]
                       for item in metrics.values()),
        "top_decile": all(item["top_decile_difference"] <= criteria["top_decile_tolerance"]
                          for item in metrics.values()),
    }
    return {
        "identity": checks[split],
        "groups": int(finite.size),
        "group_fraction": group_fraction,
        "group_spearman_min": float(finite.min()) if finite.size else float("nan"),
        "metrics": metrics,
        "max_abs_difference": float(np.nanmax(difference)),
        "mean_abs_difference": float(np.nanmean(difference)),
        "passed_criteria": passed,
        "passed": all(passed.values()),
        "local_run": run,
    }


def summary_markdown(report: dict[str, Any]) -> str:
    result, criteria = report["result"], report["config"]["criteria"]
    rows = [
        f"| {key} | {item['local']['rank_ic']:+.4f} | {item['reference']['rank_ic']:+.4f} | "
        f"{item['rank_ic_difference']:.4f} | {1e4 * item['local']['top_decile']:+.1f} | "
        f"{1e4 * item['reference']['top_decile']:+.1f} | "
        f"{1e4 * item['top_decile_difference']:.1f} |"
        for key, item in result["metrics"].items()
    ]
    runtime = result["local_run"]["runtime"]
    speed = result["local_run"].get("samples_per_second", float("nan"))
    verdict_zh, verdict_en = ("通過", "passed") if result["passed"] else ("未通過", "failed")
    header_zh = ("| 市場與天期 | 本機 Rank IC | 參照 Rank IC | 差距 | 本機前十分位 bps | "
                 "參照前十分位 bps | 差距 bps |")
    header_en = ("| Market and horizon | Local Rank IC | Reference Rank IC | Difference | "
                 "Local top decile bps | Reference top decile bps | Difference bps |")
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [
        "# 本機推論一致性比對 / Local Inference Parity", "", "## 中文", "",
        f"- 判定：**{verdict_zh}**。",
        f"- 截面 Spearman ≥ {criteria['group_spearman_min']} 的組合比例："
        f"{result['group_fraction']:.4%}（門檻 {criteria['group_fraction_min']:.0%}，"
        f"共 {result['groups']} 組）。",
        f"- 預測最大絕對差 {result['max_abs_difference']:.2e}，平均 "
        f"{result['mean_abs_difference']:.2e}。",
        f"- 本機：{runtime['device']} {runtime['dtype']}，每秒 {speed:.1f} 筆。", "",
        header_zh, divider, *rows, "", "## English", "",
        f"- Verdict: **{verdict_en}**.",
        f"- Share of groups with cross-sectional Spearman ≥ {criteria['group_spearman_min']}: "
        f"{result['group_fraction']:.4%} (threshold {criteria['group_fraction_min']:.0%}, "
        f"{result['groups']} groups).",
        f"- Prediction differences: maximum {result['max_abs_difference']:.2e}, mean "
        f"{result['mean_abs_difference']:.2e}.",
        f"- Local run: {runtime['device']} {runtime['dtype']}, {speed:.1f} samples per second.",
        "", header_en, divider, *rows,
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path, artifacts: Path, local_directory: Path, output_root: Path) -> dict:
    config = json.loads(Path(config_path).read_text())
    result = compare(config, artifacts, local_directory)
    report = clean({
        "schema_version": "1.0",
        "kind": "local-inference-parity",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "local_directory": str(local_directory),
        "result": result,
    })
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True,
                        help="Completed local prediction directory from inference/.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path, help="New directory for this check.")
    args = parser.parse_args(argv)
    output_root = args.output_root or default_output_root(args.artifacts_root, "parity")
    report = run(args.config, args.artifacts_root, args.local, output_root)
    print(json.dumps({"output_root": str(output_root),
                      "passed": report["result"]["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
