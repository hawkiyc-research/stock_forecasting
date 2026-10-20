"""Protocol section 13.11: Kronos a-rank64 against the rank baselines by grouped daily Rank IC.

Every model is scored on the same validation and test rows. A daily Spearman Rank IC is
computed per (market, horizon, signal date) on the raw benchmark-relative returns; cells
average horizons within a group and, for Taiwan, TWSE and TPEX with equal weight. Intervals
are circular moving-block bootstraps of the daily series. Before any decision is read, each
model's per-market, per-horizon mean must reproduce the value recorded by its own training
or export job.

Usage: python -m twbacktest.rank_pilot --kronos-run RUN_ID [--workers N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest.runner import MAX_WORKERS, clean, default_output_root, default_workers, run_tasks
from twbacktest.stats import bootstrap_mean

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "rank_pilot.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
KRONOS = "kronos"
_DIGEST_KEYS = {"validation": "validation_membership", "test": "evaluation_membership"}


def baseline_dir(config: dict[str, Any], artifacts: Path) -> Path:
    inputs = config["inputs"]
    return artifacts / inputs["baseline_root"] / inputs["rank_baseline"]


def baseline_digest(root: Path, split: str) -> str:
    payload = json.loads((root / "complete.json").read_text())
    return str(payload[_DIGEST_KEYS[split]]["ordered_symbol_dates_sha256"])


def kronos_export(config: dict[str, Any], artifacts: Path, run: str) -> Path:
    """Newest complete validation-best export of the given rank run."""
    root = artifacts / config["inputs"]["predictions_root"] / run
    candidates = []
    for directory in sorted(root.glob("*/export-*")):
        status, report = directory / "status.json", directory / "export.json"
        if not status.is_file() or not report.is_file():
            continue
        if json.loads(status.read_text()).get("state") != "complete":
            continue
        payload = json.loads(report.read_text())
        if (
            payload.get("prediction_task") == config["kronos"]["prediction_task"]
            and (payload.get("checkpoint") or {}).get("selection")
            == config["kronos"]["checkpoint"]
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"No complete validation-best rank export under {root}")
    return candidates[-1]


def model_paths(
    config: dict[str, Any], artifacts: Path, export: Path, split: str
) -> dict[str, Path]:
    root = baseline_dir(config, artifacts)
    paths = {KRONOS: export / split / "predictions.npy"}
    for name, job in config["baselines"].items():
        paths[name] = root / job / split / "predictions.npy"
    return paths


def population(root: Path, split: str) -> dict[str, Any]:
    """Row order, markets and day keys of one split; groups rows by (market, day)."""
    metadata = np.load(root / "inputs" / split / "metadata.npy")
    markets = metadata["market"].astype(str)
    days = np.array([value[:10] for value in metadata["date"].astype(str)])
    order = np.lexsort((days, markets))
    keys = np.char.add(np.char.add(markets[order], "|"), days[order])
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    stops = np.r_[starts[1:], len(order)]
    groups = [
        (markets[order[a]], days[order[a]], order[a:b])
        for a, b in zip(starts, stops, strict=True)
    ]
    return {"count": len(metadata), "groups": groups}


def daily_rank_ic(
    predictions: np.ndarray, targets: np.ndarray, rows: np.ndarray, minimum: int
) -> np.ndarray:
    """Spearman (average ties) per horizon for one market-day; NaN when undefined."""
    p, y = predictions[rows], targets[rows]
    result = np.full(p.shape[1], np.nan)
    for h in range(p.shape[1]):
        valid = np.isfinite(p[:, h]) & np.isfinite(y[:, h])
        if valid.sum() < minimum:
            continue
        x, z = p[valid, h], y[valid, h]
        if np.ptp(x) == 0 or np.ptp(z) == 0:
            continue
        rx = pd.Series(x).rank(method="average").to_numpy()
        rz = pd.Series(z).rank(method="average").to_numpy()
        result[h] = float(np.corrcoef(rx, rz)[0, 1])
    return result


_STATE: dict[str, Any] = {}


def _initialize(config, artifacts, export, root):
    _STATE.update(config=config, artifacts=artifacts, export=export, root=root, cache={})


def _split_inputs(split: str):
    cache = _STATE["cache"]
    if split not in cache:
        root = _STATE["root"]
        cache[split] = (
            population(root, split),
            np.load(root / "inputs" / split / "targets.npy").astype(np.float64),
        )
    return cache[split]


def run_task(task: tuple[str, str]) -> list[dict[str, Any]]:
    """Daily Rank IC rows for one model on one split."""
    model, split = task
    config = _STATE["config"]
    groups, targets = _split_inputs(split)
    path = model_paths(config, _STATE["artifacts"], _STATE["export"], split)[model]
    predictions = np.load(path).astype(np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(f"{path} has shape {predictions.shape}, expected {targets.shape}")
    minimum = int(config["rank_ic"]["minimum_valid_stocks"])
    rows = []
    for market, day, members in groups["groups"]:
        values = daily_rank_ic(predictions, targets, members, minimum)
        for h, value in enumerate(values, start=1):
            rows.append(
                {"model": model, "split": split, "market": market, "date": day, "h": h,
                 "ic": value}
            )
    return rows


def recorded_means(
    config: dict[str, Any], artifacts: Path, export: Path, split: str
) -> dict[str, dict]:
    """Each job's own per-market, per-horizon mean Rank IC for this split."""
    root = baseline_dir(config, artifacts)
    complete = json.loads((root / "complete.json").read_text())
    result = {}
    for name, job in config["baselines"].items():
        if split == "test":
            model = complete["models"][name]
            metrics = model["seed_results"]["42"] if "seed_results" in model else model["metrics"]
        else:
            metrics = json.loads((root / job / "validation-metrics.json").read_text())
        result[name] = metrics["cross_sectional"]["by_market"]
    summary = json.loads((export / split / "split.json").read_text())
    result[KRONOS] = summary["cross_sectional"]["by_market"]
    return result


def consistency(daily: pd.DataFrame, recorded: dict, tolerance: float) -> dict[str, Any]:
    means = daily.groupby(["model", "split", "market", "h"])["ic"].mean()
    worst, checked = 0.0, 0
    for (model, split, market, h), value in means.items():
        expected = recorded[split][model][market][f"{h}d"]["rank_ic_mean"]
        if not np.isfinite(value):
            value = 0.0  # the job records 0.0 when no date is defined
        worst = max(worst, abs(float(value) - float(expected)))
        checked += 1
    return {"checked": checked, "max_abs_difference": worst, "passed": worst <= tolerance}


def cell_series(
    daily: pd.DataFrame, model: str, split: str, markets: list[str], horizons: list[int]
) -> pd.Series:
    """Daily cell value: mean over the group's horizons, then over its markets."""
    rows = daily[
        (daily["model"] == model)
        & (daily["split"] == split)
        & daily["market"].isin(markets)
        & daily["h"].isin(horizons)
    ]
    per_market = rows.groupby(["date", "market"])["ic"].mean()
    return per_market.groupby("date").mean().sort_index()


def score_cells(daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    boot = config["bootstrap"]
    rows = []
    for split in config["splits"]:
        for model in [KRONOS, *config["baselines"]]:
            for market, members in config["markets"].items():
                for group, horizons in config["horizon_groups"].items():
                    series = cell_series(daily, model, split, members, horizons)
                    mean, low, high = bootstrap_mean(
                        series.to_numpy(), block=int(boot["block_sessions"][group]),
                        replicates=int(boot["replicates"]), seed=int(boot["seed"]),
                    )
                    rows.append({"split": split, "model": model, "market": market,
                                 "horizons": group, "mean": mean, "low": low, "high": high,
                                 "dates": int(np.isfinite(series.to_numpy()).sum())})
    return pd.DataFrame(rows)


def score_pairs(daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    boot = config["bootstrap"]
    rows = []
    for split in config["splits"]:
        for comparator in config["baselines"]:
            for market, members in config["markets"].items():
                for group, horizons in config["horizon_groups"].items():
                    kronos = cell_series(daily, KRONOS, split, members, horizons)
                    other = cell_series(daily, comparator, split, members, horizons)
                    difference = (kronos - other).dropna().sort_index()
                    mean, low, high = bootstrap_mean(
                        difference.to_numpy(), block=int(boot["block_sessions"][group]),
                        replicates=int(boot["replicates"]), seed=int(boot["seed"]),
                    )
                    rows.append({"split": split, "comparator": comparator, "market": market,
                                 "horizons": group, "mean": mean, "low": low, "high": high,
                                 "dates": len(difference)})
    return pd.DataFrame(rows)


def decide(cells: pd.DataFrame, pairs: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    split = config["decision_split"]
    result = {}
    for name, rule in config["decisions"].items():
        if rule["kind"] == "level":
            row = cells[(cells["split"] == split) & (cells["model"] == KRONOS)
                        & (cells["market"] == rule["market"])
                        & (cells["horizons"] == rule["horizons"])].iloc[0]
        else:
            row = pairs[(pairs["split"] == split) & (pairs["comparator"] == rule["comparator"])
                        & (pairs["market"] == rule["market"])
                        & (pairs["horizons"] == rule["horizons"])].iloc[0]
        low = float(row["low"])
        result[name] = {**rule, "mean": float(row["mean"]), "low": low,
                        "high": float(row["high"]), "passed": bool(np.isfinite(low) and low > 0)}
    return result


def summary_markdown(report: dict[str, Any], cells: pd.DataFrame) -> str:
    split = report["config"]["decision_split"]
    table = cells[cells["split"] == split]

    def value(row):
        return f"{row['mean']:.4f} [{row['low']:.4f}, {row['high']:.4f}]"

    lines_zh = ["## 中文", "", f"- 判定樣本：{split}；一致性檢查："
                f"{'通過' if report['consistency']['passed'] else '未通過'}"
                f"（最大差距 {report['consistency']['max_abs_difference']:.2e}）。", ""]
    lines_en = ["## English", "", f"- Decision split: {split}; consistency check: "
                f"{'passed' if report['consistency']['passed'] else 'failed'}"
                f" (largest difference {report['consistency']['max_abs_difference']:.2e}).", ""]
    for name, item in report["decisions"].items():
        mark_zh = "通過" if item["passed"] else "未通過"
        mark_en = "pass" if item["passed"] else "fail"
        text = f"{item['mean']:.4f} [{item['low']:.4f}, {item['high']:.4f}]"
        lines_zh.append(f"- {name}（{item['market']}、{item['horizons']}）：{text}，{mark_zh}")
        lines_en.append(f"- {name} ({item['market']}, {item['horizons']}): {text}, {mark_en}")
    header = "| model | market | short | medium | long |"
    divider = "| --- | --- | --- | --- | --- |"
    body = []
    for (model, market), group in table.groupby(["model", "market"], sort=False):
        cells_by = {row["horizons"]: value(row) for _, row in group.iterrows()}
        body.append(f"| {model} | {market} | {cells_by.get('short', '—')} | "
                    f"{cells_by.get('medium', '—')} | {cells_by.get('long', '—')} |")
    return "\n".join(
        ["# Rank pilot（§13.11）", "", *lines_zh, "", header, divider, *body, "",
         *lines_en, "", header, divider, *body, ""]
    )


def run(
    config_path: Path, artifacts: Path, output_root: Path, kronos_run: str, workers: int | None
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    config = json.loads(config_path.read_text())
    root = baseline_dir(config, artifacts)
    export = kronos_export(config, artifacts, kronos_run)
    payload = json.loads((export / "export.json").read_text())
    horizons = sorted(h for group in config["horizon_groups"].values() for h in group)
    if payload.get("horizons") != horizons:
        raise ValueError("The Kronos export horizons differ from the configured groups")
    digests = {}
    for split in config["splits"]:
        digest = baseline_digest(root, split)
        recorded = payload["splits"][split]["sample_membership"]["ordered_symbol_dates_sha256"]
        if recorded != digest:
            raise ValueError(f"Kronos {split} row order differs from the baseline arrays")
        digests[split] = digest
    models = [KRONOS, *config["baselines"]]
    tasks = [(model, split) for split in config["splits"] for model in models]
    count = workers or default_workers(len(tasks))
    rows = run_tasks(tasks, run_task, initializer=_initialize,
                     initargs=(config, artifacts, export, root), workers=count)
    daily = pd.DataFrame(rows)
    recorded = {split: recorded_means(config, artifacts, export, split)
                for split in config["splits"]}
    check = consistency(daily, recorded, float(config["consistency_tolerance"]))
    if not check["passed"]:
        raise ValueError(f"Daily Rank IC does not reproduce the recorded means: {check}")
    cells, pairs = score_cells(daily, config), score_pairs(daily, config)
    per_horizon = (daily.groupby(["split", "model", "market", "h"])["ic"].mean()
                   .reset_index().rename(columns={"ic": "rank_ic_mean"}))
    report = {
        "config": config,
        "kronos": {"run": kronos_run, "export": str(export),
                   "checkpoint": payload.get("checkpoint", {}).get("path")},
        "digests": digests,
        "consistency": check,
        "decisions": decide(cells, pairs, config),
        "workers": count,
    }
    output_root.mkdir(parents=True)
    for name, frame in (("daily", daily), ("cells", cells), ("pairs", pairs),
                        ("per_horizon", per_horizon)):
        frame.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / "report.json").write_text(
        json.dumps(clean(report), ensure_ascii=False, indent=1) + "\n"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, cells))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kronos-run", required=True, help="The a-rank64 seed-42 run ID")
    parser.add_argument("--workers", type=int, help=f"Processes, 1-{MAX_WORKERS} (default: auto)")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    output = args.output_root or default_output_root(args.artifacts_root, "rank-pilot")
    report = run(args.config, args.artifacts_root, output, args.kronos_run, args.workers)
    print(json.dumps({"output": str(output), "decisions": clean(report["decisions"])},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
