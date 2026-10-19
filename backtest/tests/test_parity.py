import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from conftest import ALPHA_ID, CHECKPOINT, RUN, fixture_config

from twbacktest import data, local_predictions, parity

EXPORT = "diagnostics/signal-predictions"


def write_local_run(artifacts, root, split="validation", *, noise=0.0, drop=0, seed=0):
    """A local prediction run copied from the fixture export, in shuffled row order."""
    metadata = np.load(artifacts / ALPHA_ID / "inputs" / split / "metadata.npy")
    exported = np.load(artifacts / EXPORT / RUN / CHECKPOINT / "export-1" / split
                       / "predictions.npy")
    rows = np.flatnonzero(np.isin(metadata["market"], data.MARKETS))
    rng = np.random.default_rng(seed)
    rows = rng.permutation(rows)[: len(rows) - drop]
    values = (exported[rows] + rng.normal(0, noise, size=(len(rows), 14))).astype(np.float32)
    directory = root / f"{split}-local"
    directory.mkdir(parents=True)
    np.save(directory / "predictions.npy", values)
    pd.DataFrame({"symbol": metadata["symbol"][rows], "cutoff_at": metadata["date"][rows],
                  "market": metadata["market"][rows]}).to_parquet(directory / "keys.parquet")
    digest = hashlib.sha256((directory / "predictions.npy").read_bytes()).hexdigest()
    (directory / "run.json").write_text(json.dumps({
        "kind": "local-signal-predictions", "split": split, "run_id": RUN,
        "checkpoint": {"name": CHECKPOINT}, "predictions_sha256": digest,
        "runtime": {"device": "cpu", "dtype": "float32"}, "samples_per_second": 100.0,
    }))
    (directory / "status.json").write_text(json.dumps({"state": "complete"}))
    return directory, rows


def config_file(tmp_path, export="export-1"):
    return fixture_config(parity.DEFAULT_CONFIG, tmp_path, "local_parity.json",
                          reference={"run": RUN, "checkpoint": CHECKPOINT, "export": export})


def test_local_scores_join_by_stock_and_date(price_artifacts, tmp_path):
    directory, _rows = write_local_run(price_artifacts, tmp_path)
    index = data.build_split_index(price_artifacts / ALPHA_ID, "validation")
    scores, run = local_predictions.load_local_scores(directory, index)
    exported = np.load(price_artifacts / EXPORT / RUN / CHECKPOINT / "export-1" / "validation"
                       / "predictions.npy")
    np.testing.assert_allclose(scores, exported[index.rows], rtol=0, atol=1e-6)
    assert run["run_id"] == RUN
    test_index = data.build_split_index(price_artifacts / ALPHA_ID, "test")
    with pytest.raises(ValueError, match="predicts validation"):
        local_predictions.load_local_scores(directory, test_index)


def test_incomplete_or_tampered_local_runs_are_rejected(price_artifacts, tmp_path):
    index = data.build_split_index(price_artifacts / ALPHA_ID, "validation")
    short, _rows = write_local_run(price_artifacts, tmp_path / "short", drop=3)
    with pytest.raises(ValueError, match="cover"):
        local_predictions.load_local_scores(short, index)
    tampered, _rows = write_local_run(price_artifacts, tmp_path / "tampered")
    np.save(tampered / "predictions.npy", np.zeros((index.count, 14), dtype=np.float32))
    with pytest.raises(ValueError, match="SHA-256"):
        local_predictions.load_local_scores(tampered, index)


def test_parity_passes_for_float_noise_and_fails_for_real_differences(price_artifacts,
                                                                      tmp_path):
    close, _rows = write_local_run(price_artifacts, tmp_path / "close", noise=1e-7)
    report = parity.run(config_file(tmp_path), price_artifacts, close, tmp_path / "ok")
    assert report["result"]["passed"] is True
    assert report["result"]["group_fraction"] == 1.0
    assert (tmp_path / "ok" / "summary.md").read_text().startswith("# 本機推論一致性比對")
    far, _rows = write_local_run(price_artifacts, tmp_path / "far", noise=0.05, seed=1)
    report = parity.run(config_file(tmp_path), price_artifacts, far, tmp_path / "bad")
    assert report["result"]["passed"] is False
    assert report["result"]["passed_criteria"]["group_spearman"] is False


def test_parity_requires_the_registered_reference_export(price_artifacts, tmp_path):
    close, _rows = write_local_run(price_artifacts, tmp_path / "close")
    with pytest.raises(ValueError, match="pre-registered"):
        parity.run(config_file(tmp_path, export="export-other"), price_artifacts, close,
                   tmp_path / "out")
