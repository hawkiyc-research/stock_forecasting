"""Protocol 13.11 scorer: daily Rank IC, consistency gate, grouped cells and decisions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twbacktest import rank_pilot

CONFIG = Path(__file__).resolve().parents[1] / "config" / "rank_pilot.json"
BASELINES = ("patchtst", "gbdt", "gru", "dlinear", "ma_crossover")
META = np.dtype([("symbol", "U16"), ("date", "U32"), ("market", "U8"),
                 ("asset_type", "U8"), ("provider", "U16")])


def test_daily_rank_ic_uses_average_ties_and_skips_undefined_days():
    targets = np.array([[0.1, 1.0], [0.2, 1.0], [0.2, 1.0], [0.4, 1.0]])
    predictions = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]])
    values = rank_pilot.daily_rank_ic(predictions, targets, np.arange(4), 3)
    ranks = pd.Series([0.1, 0.2, 0.2, 0.4]).rank(method="average").to_numpy()
    assert values[0] == pytest.approx(np.corrcoef([1, 2, 3, 4], ranks)[0, 1])
    # A constant target has no defined correlation.
    assert np.isnan(values[1])
    assert np.isnan(rank_pilot.daily_rank_ic(predictions, targets, np.arange(2), 3)).all()


def _metrics(daily, model, split):
    frame = daily[(daily["model"] == model) & (daily["split"] == split)]
    means = frame.groupby(["market", "h"])["ic"].mean()
    by_market = {}
    for (market, h), value in means.items():
        by_market.setdefault(market, {})[f"{h}d"] = {
            "rank_ic_mean": 0.0 if not np.isfinite(value) else float(value)
        }
    return {"cross_sectional": {"by_market": by_market}}


def build_artifacts(root: Path, config: dict, *, recorded_offset: float = 0.0) -> str:
    """Synthetic population: Kronos tracks returns, PatchTST is noise."""
    rng = np.random.default_rng(3)
    baseline = root / config["inputs"]["baseline_root"] / config["inputs"]["rank_baseline"]
    run = "run-rank-test"
    export = root / config["inputs"]["predictions_root"] / run / "checkpoint-000010" / "export-1"
    complete = {"models": {}, "validation_membership": {}, "evaluation_membership": {}}
    splits = {}
    calendar = pd.date_range("2025-06-02", periods=100, freq="D")
    for split, offset in (("validation", 0), ("test", 60)):
        rows = []
        for market, stocks in (("TWSE", 9), ("TPEX", 7), ("US", 11)):
            for day in range(40):
                date = calendar[offset + day].isoformat()
                rows += [(f"{market}{s}", date, market, "stock", "fixture") for s in range(stocks)]
        metadata = np.array(rows, dtype=META)
        targets = rng.normal(size=(len(rows), 14)).astype(np.float32)
        predictions = {
            rank_pilot.KRONOS: targets + rng.normal(scale=0.7, size=targets.shape),
            **{name: rng.normal(size=targets.shape) for name in BASELINES},
        }
        directory = baseline / "inputs" / split
        directory.mkdir(parents=True)
        np.save(directory / "metadata.npy", metadata)
        np.save(directory / "targets.npy", targets)
        digest = f"{split}-digest"
        key = "validation_membership" if split == "validation" else "evaluation_membership"
        complete[key] = {"ordered_symbol_dates_sha256": digest}
        for name, job in config["baselines"].items():
            path = baseline / job / split / "predictions.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, predictions[name].astype(np.float32))
        (export / split).mkdir(parents=True)
        kronos = predictions[rank_pilot.KRONOS].astype(np.float32)
        np.save(export / split / "predictions.npy", kronos)
        splits[split] = {"sample_membership": {"ordered_symbol_dates_sha256": digest}}
    # Record each job's own means the way the training code reports them.
    rank_pilot._initialize(config, root, export, baseline)
    daily = pd.DataFrame([row for model in (rank_pilot.KRONOS, *BASELINES)
                          for split in ("validation", "test")
                          for row in rank_pilot.run_task((model, split))])
    for name, job in config["baselines"].items():
        metrics = _metrics(daily, name, "test")
        for market in metrics["cross_sectional"]["by_market"].values():
            for item in market.values():
                item["rank_ic_mean"] += recorded_offset
        complete["models"][name] = (
            {"seed_results": {"42": metrics}} if name != "ma_crossover" else {"metrics": metrics}
        )
        (baseline / job / "validation-metrics.json").write_text(
            json.dumps(_metrics(daily, name, "validation"))
        )
    (baseline / "complete.json").write_text(json.dumps(complete))
    for split in ("validation", "test"):
        summary = _metrics(daily, rank_pilot.KRONOS, split)
        (export / split / "split.json").write_text(json.dumps(summary))
    (export / "status.json").write_text(json.dumps({"state": "complete"}))
    (export / "export.json").write_text(json.dumps({
        "prediction_task": "cross_sectional_rank",
        "checkpoint": {"selection": "validation_best", "path": "/runpod-volume/x"},
        "horizons": list(range(1, 15)),
        "splits": splits,
    }))
    return run


def test_pilot_reproduces_recorded_means_and_decides_on_test(tmp_path):
    config = json.loads(CONFIG.read_text())
    run = build_artifacts(tmp_path / "artifacts", config)
    output = tmp_path / "out"
    report = rank_pilot.run(CONFIG, tmp_path / "artifacts", output, run, workers=1)
    assert report["consistency"]["passed"] and report["consistency"]["checked"] > 0
    decisions = report["decisions"]
    assert decisions["H1"]["passed"] and decisions["H3"]["passed"]
    assert decisions["H2"]["passed"] and decisions["H2"]["comparator"] == "patchtst"
    cells = pd.read_csv(output / "cells.csv")
    noise = cells[(cells["model"] == "patchtst") & (cells["split"] == "test")]
    assert (noise["low"] < 0).all() and (noise["high"] > 0).all()
    # Taiwan cells average TWSE and TPEX; both markets share the 40 test dates.
    assert set(cells[cells["market"] == "TW"]["dates"]) == {40}
    assert (output / "summary.md").read_text().count("H1") == 2


def test_pilot_refuses_inconsistent_records_and_mismatched_rows(tmp_path):
    config = json.loads(CONFIG.read_text())
    run = build_artifacts(tmp_path / "artifacts", config, recorded_offset=1e-6)
    with pytest.raises(ValueError, match="reproduce the recorded means"):
        rank_pilot.run(CONFIG, tmp_path / "artifacts", tmp_path / "out", run, workers=1)
    export = next((tmp_path / "artifacts").rglob("export.json"))
    payload = json.loads(export.read_text())
    payload["splits"]["test"]["sample_membership"]["ordered_symbol_dates_sha256"] = "other"
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="row order differs"):
        rank_pilot.run(CONFIG, tmp_path / "artifacts", tmp_path / "out2", run, workers=1)
