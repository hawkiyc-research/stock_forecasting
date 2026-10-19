import json

import numpy as np
import pandas as pd
import pytest
from conftest import DIGESTS, DIRECTION_ID, fixture_config

from twbacktest import selection

RUNS = {"RA": "run-a-alpha", "RB": "run-b-alpha"}
SCORE = {10: 0.51, 20: 0.47}


def write_exports(artifacts, *, skip_test_for_noise=True, missing=None):
    """Checkpoint 20 of each run foresees returns, checkpoint 10 is noise."""
    rng = np.random.default_rng(3)
    root = artifacts / "diagnostics/signal-predictions"
    for run in RUNS.values():
        for step in (10, 20):
            if (run, step) == missing:
                continue
            export = root / run / f"checkpoint-{step:06d}" / "export-1"
            splits = {}
            for split in ("validation", "test"):
                if split == "test" and step == 10 and skip_test_for_noise:
                    continue
                absolute = np.load(artifacts / DIRECTION_ID / "inputs" / split
                                   / "absolute_targets.npy")
                values = absolute if step == 20 else rng.normal(size=absolute.shape)
                (export / split).mkdir(parents=True, exist_ok=True)
                np.save(export / split / "predictions.npy", values.astype(np.float32))
                splits[split] = {"sample_membership": {"ordered_symbol_dates_sha256":
                                                       DIGESTS[split]},
                                 "aggregate": {"selection_score": SCORE[step]}}
            (export / "export.json").write_text(json.dumps(
                {"splits": splits, "checkpoint": {"global_step": step}}))
            (export / "status.json").write_text(json.dumps({"state": "complete"}))


def write_records(root, *, shift=0.0):
    for run in RUNS.values():
        path = root / run / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(
            json.dumps({"step": step, "validation/aggregate/selection_score": score + shift})
            for step, score in SCORE.items()
        ) + "\n")
    return root


def config_file(tmp_path):
    candidates = {name: {"run": run, "checkpoints": ["checkpoint-000010", "checkpoint-000020"]}
                  for name, run in RUNS.items()}
    return fixture_config(selection.DEFAULT_CONFIG, tmp_path, "selection.json",
                          candidates=candidates)


def test_selection_picks_the_foreseeing_checkpoint_and_scores_test_only_for_it(
        price_artifacts, tmp_path):
    write_exports(price_artifacts)
    records = write_records(tmp_path / "records")
    output = tmp_path / "out"
    report = selection.run(config_file(tmp_path), price_artifacts, records, output, workers=1)
    assert {group: item["selected"] for group, item in report["selection"].items()} == {
        "RA": "RA_000020", "RB": "RB_000020"}
    assert all(item["problem"] is None for item in report["export_checks"])
    ranking = pd.read_csv(output / "ranking.csv")
    assert set(ranking.loc[ranking["split"] == "validation", "signal"]) == {
        "RA_000010", "RA_000020", "RB_000010", "RB_000020"}
    # Test predictions of the noise checkpoints do not exist and are never read.
    assert set(ranking.loc[ranking["split"] == "test", "signal"]) == {"RA_000020", "RB_000020"}
    assert (output / "summary.md").read_text().startswith("# checkpoint 選點")


def test_training_record_mismatch_or_missing_export_stops_before_scoring(
        price_artifacts, tmp_path, capsys):
    write_exports(price_artifacts, missing=("run-b-alpha", 10))
    with pytest.raises(ValueError, match="missing export"):
        selection.run(config_file(tmp_path), price_artifacts, write_records(tmp_path / "ok"),
                      tmp_path / "out", workers=1)
    assert not (tmp_path / "out").exists()
    write_exports(price_artifacts)
    shifted = write_records(tmp_path / "shifted", shift=1e-3)
    with pytest.raises(ValueError, match=r"differs from metrics\.jsonl"):
        selection.run(config_file(tmp_path), price_artifacts, shifted, tmp_path / "out",
                      workers=1)
    selection.main(["--check", "--config", str(config_file(tmp_path)), "--artifacts-root",
                    str(price_artifacts), "--training-records", str(shifted)])
    assert "differs from metrics.jsonl" in capsys.readouterr().out
