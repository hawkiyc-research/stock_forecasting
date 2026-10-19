"""Select checkpoints by a holding-aligned validation metric (protocol section 13.7, amended).

Every candidate checkpoint's RunPod export is checked first: its row order must match the
baseline digest, and its recomputed validation selection score must match the run's
metrics.jsonl record at the checkpoint's step. Candidates are then scored with the
section 13.6 evaluation code; each run keeps the checkpoint with the highest validation
TWSE 10-session top-decile open-to-open excess (ties go to the later step). Test is read
only for the selected checkpoints, and only to describe them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from twbacktest import data, holding_eval
from twbacktest.runner import (
    MAX_WORKERS,
    clean,
    default_output_root,
    default_workers,
    prepare_inputs,
    run_tasks,
)
from twbacktest.stage1b import identity_checks

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "checkpoint_selection.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
KEYS = ["split", "market", "signal", "kind", "h"]


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per (run label, checkpoint), labelled like R22_018399."""
    output = []
    for name, entry in config["candidates"].items():
        for checkpoint in entry["checkpoints"]:
            step = int(checkpoint.split("-", 1)[1])
            output.append({"label": f"{name}_{step:06d}", "group": name, "run": entry["run"],
                           "checkpoint": checkpoint, "step": step})
    return output


def evaluation_config(config: dict[str, Any], chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """A section 13.6 evaluation config whose signals are the given checkpoints."""
    return {
        **config,
        "signals": {item["label"]: {"kind": "alpha",
                                    "members": [[item["run"], item["checkpoint"]]]}
                    for item in chosen},
        "horizons": {"primary": [5, 10], "auxiliary": [3]},
    }


def _recorded_score(records: Path, step: int) -> float | None:
    score = None
    with records.open() as stream:
        for line in stream:
            record = json.loads(line)
            key = "validation/aggregate/selection_score"
            if record.get("step") == step and key in record:
                score = float(record[key])
    return score


def check_exports(config: dict[str, Any], predictions_root: Path,
                  training_records: Path) -> list[dict[str, Any]]:
    """Locate each candidate's newest complete export and compare it with training records."""
    tolerance = float(config["integrity"]["tolerance"])
    results = []
    for item in candidates(config):
        result = {**item, "export": None, "problem": None}
        try:
            directory = data.export_directory(predictions_root, item["run"], item["checkpoint"])
        except FileNotFoundError:
            result["problem"] = "missing export"
            results.append(result)
            continue
        result["export"] = directory.name
        report = json.loads((directory / "export.json").read_text())
        splits = report.get("splits") or {}
        exported = (splits.get("validation") or {}).get("aggregate", {}).get("selection_score")
        records = Path(training_records) / item["run"] / "metrics.jsonl"
        recorded = _recorded_score(records, item["step"]) if records.is_file() else None
        result.update(splits=sorted(splits), exported_score=exported, recorded_score=recorded)
        if report.get("checkpoint", {}).get("global_step") != item["step"]:
            result["problem"] = "export step differs from the checkpoint name"
        elif "validation" not in splits:
            result["problem"] = "export has no validation split"
        elif recorded is None:
            result["problem"] = f"no validation record at step {item['step']} in {records}"
        elif exported is None or abs(float(exported) - recorded) > tolerance:
            result["problem"] = "exported validation score differs from metrics.jsonl"
        results.append(result)
    return results


def select(ranking: pd.DataFrame, config: dict[str, Any],
           chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """Per run, the candidate with the highest validation metric; ties go to the later step."""
    rule = config["rule"]
    column = {"top_decile_excess": "top_mean"}[rule["metric"]]
    rows = ranking[(ranking["split"] == rule["split"]) & (ranking["market"] == rule["market"])
                   & (ranking["h"] == rule["h"])].set_index("signal")
    output = {}
    for group in config["candidates"]:
        members = [item for item in chosen if item["group"] == group]
        scored = sorted(members, key=lambda item: (float(rows.loc[item["label"], column]),
                                                   item["step"]), reverse=True)
        best = scored[0]
        output[group] = {
            "selected": best["label"], "run": best["run"], "checkpoint": best["checkpoint"],
            "candidates": [{"label": item["label"], "step": item["step"],
                            "value": float(rows.loc[item["label"], column]),
                            "low": float(rows.loc[item["label"], "top_low"]),
                            "high": float(rows.loc[item["label"], "top_high"])}
                           for item in sorted(members, key=lambda item: item["step"])],
        }
    return output


def _score(config, labels, split, inputs, indexes, totals, workers) -> tuple:
    rows = run_tasks([(split, label) for label in labels], holding_eval.run_task,
                     initializer=holding_eval._initialize,
                     initargs=(config, inputs, indexes, totals), workers=workers)
    frame = pd.DataFrame(rows)
    tables = []
    for name in ("ranking", "portfolio"):
        part = frame[frame["table"] == name].dropna(axis=1, how="all").drop(columns="table")
        tables.append(part.sort_values(KEYS, kind="stable").reset_index(drop=True))
    return tuple(tables)


def _bps(value: Any) -> str:
    return "—" if value is None else f"{1e4 * float(value):+.1f}"


def summary_markdown(report: dict[str, Any], ranking: pd.DataFrame) -> str:
    rule = report["config"]["rule"]
    test = ranking[(ranking["split"] == "test") & (ranking["market"] == rule["market"])
                   & (ranking["h"] == rule["h"])].set_index("signal")
    rows = []
    for group, item in report["selection"].items():
        for candidate in item["candidates"]:
            mark = "✓" if candidate["label"] == item["selected"] else ""
            chosen = candidate["label"] == item["selected"] and candidate["label"] in test.index
            test_value = _bps(test.loc[candidate["label"], "top_mean"]) if chosen else "—"
            rows.append(f"| {group} | {candidate['step']:06d} | {_bps(candidate['value'])} "
                        f"[{_bps(candidate['low'])}, {_bps(candidate['high'])}] | "
                        f"{test_value} | {mark} |")
    header_zh = ("| run | checkpoint | validation 前十分位超額 bps [95% 區間] | "
                 "test（只算選出者） | 選出 |")
    header_en = ("| Run | Checkpoint | Validation top-decile excess bps [95% CI] | "
                 "Test (selected only) | Selected |")
    divider = "| --- | ---: | --- | ---: | :---: |"
    lines = [
        "# checkpoint 選點 / Checkpoint Selection", "", "## 中文", "",
        f"- 規則：每個 run 選 validation {rule['market']} {rule['h']} 日前十分位開盤到開盤超額"
        "最高的 checkpoint，同分選步數較晚者。",
        "- 每個匯出都已核對列順序與訓練紀錄；test 只計算選出的 checkpoint，只作描述。", "",
        header_zh, divider, *rows, "", "## English", "",
        f"- Rule: each run keeps the checkpoint with the highest validation {rule['market']} "
        f"{rule['h']}-session top-decile open-to-open excess; ties go to the later step.",
        "- Every export was checked against the baseline row order and the training records; "
        "test is computed for the selected checkpoints only and is descriptive.", "",
        header_en, divider, *rows,
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path, artifacts: Path, training_records: Path, output_root: Path,
        workers: int | None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text())
    run_config = {**config, "splits": ["validation", "test"]}
    inputs, indexes, totals = prepare_inputs(run_config, artifacts)
    checks = check_exports(config, inputs["predictions_root"], training_records)
    problems = [f"{item['label']}: {item['problem']}" for item in checks if item["problem"]]
    if problems:
        raise ValueError("Export checks failed:\n" + "\n".join(problems))
    identity = identity_checks(run_config, inputs, indexes, totals)
    chosen = candidates(config)
    scoring = evaluation_config(run_config, chosen)
    used = workers or default_workers(len(chosen))
    validation_ranking, validation_portfolio = _score(
        scoring, [item["label"] for item in chosen], "validation", inputs, indexes, totals, used
    )
    selection = select(validation_ranking, config, chosen)
    # Test is computed for the selected checkpoints only (protocol 13.7 amendment).
    selected = [item["selected"] for item in selection.values()]
    test_ranking, test_portfolio = _score(scoring, selected, "test", inputs, indexes, totals,
                                          min(used, len(selected)))
    ranking = pd.concat([validation_ranking, test_ranking], ignore_index=True)
    portfolio = pd.concat([validation_portfolio, test_portfolio], ignore_index=True)
    report = clean({
        "schema_version": "1.0",
        "kind": "holding-checkpoint-selection",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "workers": used,
        "identity": identity,
        "export_checks": checks,
        "selection": selection,
    })
    output_root.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(output_root / "ranking.csv", index=False)
    portfolio.to_csv(output_root / "portfolio.csv", index=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report, ranking))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--training-records", type=Path, required=True,
                        help="Directory holding <RUN_ID>/metrics.jsonl of the candidate runs.")
    parser.add_argument("--output-root", type=Path, help="New directory for this selection.")
    parser.add_argument("--workers", type=int, help=f"Process workers, 1-{MAX_WORKERS}.")
    parser.add_argument("--check", action="store_true",
                        help="Only list each candidate's export and integrity status.")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    if args.check:
        config = json.loads(args.config.read_text())
        root = args.artifacts_root / config["inputs"]["predictions_root"]
        for item in check_exports(config, root, args.training_records):
            print(f"{item['label']}: {item['problem'] or 'ok'}"
                  + (f" ({item['export']})" if item["export"] else ""))
        return 0
    output_root = args.output_root or default_output_root(args.artifacts_root, "selection")
    report = run(args.config, args.artifacts_root, args.training_records, output_root,
                 args.workers)
    print(json.dumps({"output_root": str(output_root),
                      "selected": {group: item["selected"]
                                   for group, item in report["selection"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
