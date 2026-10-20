"""Pre-registered stage 3 forward test (protocol section 13.8, its amendment and supplement).

Checks run first and are all recorded: the sealed package (data check 1), the open-to-open
identity on forward Taiwan rows (2), overlap labels against the shared evaluation snapshot
(3), the model-input comparison written by ``forward_pipeline compare-inputs`` (4), and the
pipeline consistency check of R22-IR overlap predictions against the 2026-10-19 test
export. Forward metrics are computed only when every check passes. The section 13.6
evaluation then scores the registered signals on the forward population, and only the
primary hypothesis (R22-IR, TWSE, h = 10) decides. Once a set of forward exports has been
scored, a different set is refused.
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

from twbacktest import data, holding_eval
from twbacktest.runner import (
    MAX_WORKERS,
    clean,
    default_output_root,
    default_workers,
    percent,
    run_tasks,
)
from twbacktest.stage1b import identity_checks

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "stage3.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
KEYS = ["split", "market", "signal", "kind", "h"]
POPULATIONS = ("overlap", "forward")


def forward_paths(config: dict[str, Any], artifacts: Path) -> dict[str, Path]:
    end = config["data"]["end_exclusive"]
    if not end:
        raise ValueError("stage3.json data.end_exclusive is not set; it must precede the fetch")
    root = artifacts / "forward" / end
    return {
        "package": root / "package",
        "labels": root / "labels",
        "checks": root / "checks",
        "predictions": artifacts / "diagnostics" / "forward-predictions" / end,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_package(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Data check 1: the sealed package exists, matches the end date and every file hash."""
    manifest = paths["package"] / "forward-package.json"
    if not manifest.is_file():
        return {"passed": False, "problem": f"missing {manifest}"}
    payload = json.loads(manifest.read_text())
    changed = [
        relative for relative, digest in payload.get("files", {}).items()
        if not (paths["package"] / relative).is_file()
        or _sha256(paths["package"] / relative) != digest
    ]
    problem = None
    if payload.get("end_exclusive") != config["data"]["end_exclusive"]:
        problem = "package end date differs from stage3.json"
    elif not payload.get("files"):
        problem = "package lists no files"
    elif changed:
        problem = f"{len(changed)} package files changed after sealing"
    return {"passed": problem is None, "problem": problem,
            "package_sha256": payload.get("package_sha256"),
            "populations": {name: values.get("samples")
                            for name, values in payload.get("populations", {}).items()}}


def label_inputs(paths: dict[str, Path]) -> tuple[dict[str, Any], dict, dict, dict]:
    """Inputs in the baseline layout: labels/inputs/<population>/ plus the export root."""
    inputs: dict[str, Any] = {"alpha_baseline": paths["labels"],
                              "direction_baseline": paths["labels"],
                              "predictions_root": paths["predictions"], "digests": {}}
    indexes, totals, summaries = {}, {}, {}
    for population in POPULATIONS:
        directory = paths["labels"] / "inputs" / population
        summary = json.loads((directory / "labels.json").read_text())
        for name, digest in summary["files_sha256"].items():
            if _sha256(directory / name) != digest:
                raise ValueError(f"{population} label file changed after it was written: {name}")
        inputs["digests"][population] = summary["sample_membership"]["ordered_symbol_dates_sha256"]
        indexes[population] = data.build_split_index(paths["labels"], population)
        totals[population] = int(summary["samples"])
        summaries[population] = summary
    return inputs, indexes, totals, summaries


def _keys(index: data.SplitIndex) -> list[tuple[str, str]]:
    return [(index.symbol_names[symbol], index.dates[day])
            for symbol, day in zip(index.symbol_codes, index.date_codes, strict=True)]


def _match(reference: data.SplitIndex, candidate: data.SplitIndex) -> tuple[np.ndarray, ...]:
    """Positions (within each index) of the reference rows found in the candidate."""
    position = {key: row for row, key in enumerate(_keys(candidate))}
    pairs = [(row, position[key]) for row, key in enumerate(_keys(reference)) if key in position]
    array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return array[:, 0], array[:, 1]


def _reference(config: dict[str, Any], artifacts: Path) -> tuple[dict[str, Any], data.SplitIndex,
                                                                    int]:
    inputs = {key: artifacts / value for key, value in config["inputs"].items()}
    index = data.build_split_index(inputs["alpha_baseline"], "test")
    total = int(np.load(inputs["alpha_baseline"] / "inputs" / "test" / "targets.npy",
                        mmap_mode="r").shape[0])
    return inputs, index, total


def check_overlap_labels(config, artifacts, inputs, indexes, totals) -> dict[str, Any]:
    """Data check 3: overlap absolute targets against the snapshot's test Taiwan rows."""
    rule = config["data_checks"]
    reference_inputs, reference, reference_total = _reference(config, artifacts)
    old, _ = data.load_returns(reference_inputs["alpha_baseline"],
                               reference_inputs["direction_baseline"], reference, reference_total)
    new, _ = data.load_returns(inputs["alpha_baseline"], inputs["direction_baseline"],
                               indexes["overlap"], totals["overlap"])
    left, right = _match(reference, indexes["overlap"])
    coverage = left.size / reference.count if reference.count else 0.0
    worst = np.abs(old[left] - new[right]).max(axis=1) if left.size else np.empty(0)
    fraction = float((worst <= rule["overlap_tolerance"]).mean()) if left.size else 0.0
    passed = coverage >= rule["overlap_min_coverage"] and fraction >= rule["overlap_min_fraction"]
    return {"passed": bool(passed), "reference_rows": reference.count, "matched_rows": left.size,
            "coverage": coverage, "within_tolerance_fraction": fraction,
            "max_difference": float(worst.max()) if worst.size else None}


def check_inputs(config: dict[str, Any], paths: dict[str, Path],
                 package_sha256: str | None) -> dict[str, Any]:
    """Data check 4: the recorded comparison passed with the registered thresholds."""
    rule = config["data_checks"]["input_comparison"]
    path = paths["checks"] / "input-comparison.json"
    if not path.is_file():
        return {"passed": False, "problem": f"missing {path}"}
    result = json.loads(path.read_text())
    problem = None
    if (result.get("tolerance"), result.get("min_fraction")) != (
        rule["max_relative_difference"], rule["min_fraction"]
    ):
        problem = "comparison thresholds differ from stage3.json"
    elif result.get("package_sha256") != package_sha256:
        problem = "comparison was run on a different package"
    elif not result.get("passed"):
        problem = "input comparison did not pass"
    return {"passed": problem is None, "problem": problem,
            **{key: result.get(key) for key in ("matched_rows", "coverage",
                                               "within_tolerance_fraction",
                                               "max_relative_difference")}}


def _score(config, splits, signals, inputs, indexes, totals, workers) -> tuple:
    scoring = {**config, "signals": signals}
    tasks = [(split, signal) for split in splits for signal in signals]
    rows = run_tasks(tasks, holding_eval.run_task, initializer=holding_eval._initialize,
                     initargs=(scoring, inputs, indexes, totals), workers=workers)
    frame = pd.DataFrame(rows)
    tables = []
    for name in ("ranking", "portfolio"):
        part = frame[frame["table"] == name].dropna(axis=1, how="all").drop(columns="table")
        tables.append(part.sort_values(KEYS, kind="stable").reset_index(drop=True))
    return tuple(tables)


def check_pipeline(config, artifacts, inputs, indexes, totals) -> dict[str, Any]:
    """R22-IR overlap predictions against the 2026-10-19 test export (pipeline check)."""
    rule = config["pipeline_check"]
    spec = config["signals"][rule["signal"]]
    run, checkpoint = spec["members"][0]
    reference_inputs, reference, reference_total = _reference(config, artifacts)
    export = data.export_directory(reference_inputs["predictions_root"], run, checkpoint)
    data.verify_export(export, "test", data.baseline_digest(reference_inputs["alpha_baseline"],
                                                            "test"))
    old = data._take(export / "test" / "predictions.npy", reference.rows, reference_total)
    new = data.load_signal(spec, inputs=inputs, index=indexes["overlap"],
                           total_rows=totals["overlap"])
    left, right = _match(reference, indexes["overlap"])
    coverage = left.size / reference.count if reference.count else 0.0
    identical = bool(left.size) and np.array_equal(old[left], new[right])
    correlation = (float(np.corrcoef(old[left].ravel(), new[right].ravel())[0, 1])
                   if left.size else float("nan"))
    ranking, _ = _score(config, ["overlap"], {rule["signal"]: spec}, inputs, indexes, totals, 1)
    top = rule["top_decile"]
    row = ranking[(ranking["market"] == top["market"]) & (ranking["h"] == top["h"])]
    top_mean = float(row["top_mean"].iloc[0])
    close = abs(top_mean - top["reference"]) <= top["tolerance"]
    passed = coverage >= rule["min_coverage"] and (
        identical or (correlation >= rule["min_correlation"] and close)
    )
    return {"passed": bool(passed), "reference_export": export.name, "matched_rows": left.size,
            "coverage": coverage, "identical": identical, "correlation": correlation,
            "overlap_top_decile": top_mean, "reference_top_decile": top["reference"]}


def secondary_signals(config: dict[str, Any], artifacts: Path) -> tuple[dict, list[str]]:
    """Registered signals plus the newest checkpoint selection's picks, without duplicates."""
    signals = dict(config["signals"])
    notes = []
    reports = sorted((artifacts / "diagnostics" / "backtest").glob("selection-*/report.json"))
    selections = [json.loads(path.read_text()) for path in reports]
    selections = [report for report in selections
                  if report.get("kind") == config["secondary_from_selection"]]
    if not selections:
        notes.append("no checkpoint selection report found; selected checkpoints not added")
        return signals, notes
    members = {tuple(spec["members"][0]): name for name, spec in signals.items()
               if len(spec.get("members", [])) == 1}
    for item in selections[-1]["selection"].values():
        member = (item["run"], item["checkpoint"])
        if member in members:
            notes.append(f"{item['selected']} is the registered signal {members[member]}")
            continue
        signals[item["selected"]] = {"kind": "alpha", "members": [list(member)]}
    return signals, notes


def decide(config: dict[str, Any], ranking: pd.DataFrame,
           portfolio: pd.DataFrame) -> dict[str, Any]:
    primary = config["primary"]
    if (primary["portfolio"]["k"], primary["portfolio"]["buffer"]) != (
        config["portfolio"]["k"], config["portfolio"]["buffer"]
    ) or primary["portfolio"]["s"] != primary["h"]:
        raise ValueError("The primary portfolio differs from the evaluated portfolio")
    bootstrap = config["bootstrap"]
    block = max(int(bootstrap["min_block"]), int(bootstrap["block_per_holding_day"]) * primary["h"])
    if block != primary["top_decile"]["block"]:
        raise ValueError("The evaluated bootstrap block differs from the registered block")

    def pick(frame: pd.DataFrame) -> pd.Series:
        selected = frame[(frame["split"] == "forward") & (frame["signal"] == primary["signal"])
                         & (frame["market"] == primary["market"]) & (frame["h"] == primary["h"])]
        if len(selected) != 1:
            raise ValueError("The primary hypothesis row is missing")
        return selected.iloc[0]

    rank, port = pick(ranking), pick(portfolio)
    top = rank["top_low"] > primary["top_decile"]["lower_bound_above"]
    net = port["net_excess_annual"] > primary["portfolio"]["net_excess_mean_above"]
    return {"passed": bool(top and net), "top_decile_lower_bound_above_zero": bool(top),
            "portfolio_net_excess_above_zero": bool(net),
            "top_mean": float(rank["top_mean"]), "top_low": float(rank["top_low"]),
            "top_high": float(rank["top_high"]), "bottom_mean": float(rank["bottom_mean"]),
            "spread_mean": float(rank["spread_mean"]),
            "net_excess_annual": float(port["net_excess_annual"]),
            "net_excess_annual_low": float(port["net_excess_annual_low"]),
            "net_excess_annual_high": float(port["net_excess_annual_high"])}


def _used_exports(signals: dict, paths: dict[str, Path]) -> dict[str, str]:
    used = {}
    for spec in signals.values():
        for run, checkpoint in spec.get("members", []):
            directory = data.export_directory(paths["predictions"], run, checkpoint)
            used[f"{run}/{checkpoint}"] = directory.name
    return dict(sorted(used.items()))


def refuse_rescoring(artifacts: Path, exports: dict[str, str]) -> None:
    """After scoring starts there are no re-runs: a different export set is refused."""
    for path in sorted((artifacts / "diagnostics" / "backtest").glob("stage3-*/report.json")):
        report = json.loads(path.read_text())
        if report.get("scored") and report.get("exports") != exports:
            raise ValueError(f"Forward exports were already scored in {path.parent.name} "
                             "with a different export set")


def _bps(value: Any) -> str:
    return "—" if value is None else f"{1e4 * float(value):+.1f}"


def summary_markdown(report: dict[str, Any], ranking: pd.DataFrame | None) -> str:
    checks = report["checks"]
    lines = ["# 第三階段前瞻檢驗 / Stage 3 Forward Test", "", "## 中文", ""]
    names_zh = {"package": "資料檢查 1：輸入包", "identity": "資料檢查 2：記帳恆等式",
                "overlap_labels": "資料檢查 3：overlap 標籤", "inputs": "資料檢查 4：模型輸入",
                "pipeline": "管線一致性檢查"}
    names_en = {"package": "Data check 1: input package", "identity": "Data check 2: identity",
                "overlap_labels": "Data check 3: overlap labels",
                "inputs": "Data check 4: model inputs", "pipeline": "Pipeline consistency check"}
    for key, name in names_zh.items():
        lines.append(f"- {name}：{'通過' if checks[key]['passed'] else '未通過'}")
    decision = report.get("decision")
    if decision is None:
        lines += ["", "檢查未全部通過，未計算任何前瞻指標。"]
    else:
        verdict = "通過" if decision["passed"] else "未通過"
        lines += ["", f"- 主要假說（R22-IR、上市、10 日）：{verdict}",
                  f"- 前十分位超額 {_bps(decision['top_mean'])} bps "
                  f"[{_bps(decision['top_low'])}, {_bps(decision['top_high'])}]；"
                  f"後十分位 {_bps(decision['bottom_mean'])}；"
                  f"前後差 {_bps(decision['spread_mean'])}",
                  f"- 固定組合年化淨超額 {percent(decision['net_excess_annual'])} "
                  f"[{percent(decision['net_excess_annual_low'])}, "
                  f"{percent(decision['net_excess_annual_high'])}]"]
    lines += ["", "## English", ""]
    for key, name in names_en.items():
        lines.append(f"- {name}: {'passed' if checks[key]['passed'] else 'failed'}")
    if decision is None:
        lines += ["", "Not every check passed, so no forward metric was computed."]
    else:
        lines += ["", f"- Primary hypothesis (R22-IR, TWSE, 10 sessions): "
                      f"{'passed' if decision['passed'] else 'failed'}",
                  f"- Top-decile excess {_bps(decision['top_mean'])} bps "
                  f"[{_bps(decision['top_low'])}, {_bps(decision['top_high'])}]; "
                  f"bottom decile {_bps(decision['bottom_mean'])}; spread "
                  f"{_bps(decision['spread_mean'])}",
                  f"- Fixed-portfolio annual net excess {percent(decision['net_excess_annual'])} "
                  f"[{percent(decision['net_excess_annual_low'])}, "
                  f"{percent(decision['net_excess_annual_high'])}]"]
    return "\n".join(lines) + "\n"


def _checked(function, *args) -> dict[str, Any]:
    """Record a check that cannot complete (missing export, mismatch) as failed."""
    try:
        return function(*args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        return {"passed": False, "problem": f"{type(error).__name__}: {error}"}


def run(config_path: Path, artifacts: Path, output_root: Path,
        workers: int | None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text())
    paths = forward_paths(config, artifacts)
    checks: dict[str, Any] = {"package": check_package(config, paths)}
    inputs, indexes, totals, summaries = label_inputs(paths)
    try:
        identity = identity_checks({**config, "identity_check": {
            "tolerance": config["data_checks"]["identity_tolerance"],
            "min_fraction": config["data_checks"]["identity_min_fraction"],
        }}, inputs, {"forward": indexes["forward"]}, totals)
        checks["identity"] = {"passed": True, **identity["forward"]}
    except ValueError as error:
        checks["identity"] = {"passed": False, "problem": str(error)}
    checks["overlap_labels"] = _checked(
        check_overlap_labels, config, artifacts, inputs, indexes, totals
    )
    package_sha256 = checks["package"].get("package_sha256")
    checks["inputs"] = _checked(check_inputs, config, paths, package_sha256)
    checks["pipeline"] = _checked(check_pipeline, config, artifacts, inputs, indexes, totals)
    report: dict[str, Any] = {
        "schema_version": "1.0", "kind": "stage3-forward-test",
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "end_exclusive": config["data"]["end_exclusive"],
        "labels": {name: summary["sample_membership"] for name, summary in summaries.items()},
        "checks": checks, "scored": False, "decision": None,
    }
    ranking = portfolio = None
    if all(check["passed"] for check in checks.values()):
        signals, notes = secondary_signals(config, artifacts)
        exports = _used_exports(signals, paths)
        refuse_rescoring(artifacts, exports)
        used = workers or default_workers(len(signals))
        ranking, portfolio = _score(config, ["forward"], signals, inputs, indexes, totals, used)
        report.update(scored=True, signals=signals, notes=notes, exports=exports, workers=used,
                      decision=decide(config, ranking, portfolio))
    output_root.mkdir(parents=True, exist_ok=False)
    if ranking is not None:
        ranking.to_csv(output_root / "ranking.csv", index=False)
        portfolio.to_csv(output_root / "portfolio.csv", index=False)
    report = clean(report)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report, ranking))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path, help="New directory for this run.")
    parser.add_argument("--workers", type=int, help=f"Process workers, 1-{MAX_WORKERS}.")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    output_root = args.output_root or default_output_root(args.artifacts_root, "stage3")
    report = run(args.config, args.artifacts_root, output_root, args.workers)
    print(json.dumps({"output_root": str(output_root), "scored": report["scored"],
                      "checks": {key: value["passed"] for key, value in report["checks"].items()},
                      "decision": report["decision"]}, indent=1, ensure_ascii=False))
    return 0 if report["scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
