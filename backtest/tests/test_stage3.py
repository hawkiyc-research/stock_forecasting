import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from conftest import ALPHA_ID, CHECKPOINT, DIRECTION_ID, RUN

from twbacktest import stage3

END = "2026-10-20"
PACKAGE_SHA = "p" * 64


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_labels(directory, population):
    files = {name: _sha(directory / name)
             for name in ("metadata.npy", "targets.npy", "absolute_targets.npy")}
    samples = len(np.load(directory / "metadata.npy"))
    (directory / "labels.json").write_text(json.dumps({
        "population": population, "samples": samples, "files_sha256": files,
        "sample_membership": {"ordered_symbol_dates_sha256": f"digest-{population}"},
    }))


@pytest.fixture()
def forward_artifacts(price_artifacts):
    """Overlap = the snapshot's test rows (so checks pass); forward = other dates."""
    root = price_artifacts
    forward = root / "forward" / END
    old_export = root / "diagnostics/signal-predictions" / RUN / CHECKPOINT / "export-1"
    export = root / "diagnostics/forward-predictions" / END / RUN / CHECKPOINT / "export-1"
    for population, split in (("overlap", "test"), ("forward", "validation")):
        target = forward / "labels" / "inputs" / population
        target.mkdir(parents=True)
        for name, baseline in (("metadata.npy", ALPHA_ID), ("targets.npy", ALPHA_ID),
                               ("absolute_targets.npy", DIRECTION_ID)):
            shutil.copy(root / baseline / "inputs" / split / name, target / name)
        _write_labels(target, population)
        (export / population).mkdir(parents=True)
        shutil.copy(old_export / split / "predictions.npy", export / population)
    (export / "status.json").write_text(json.dumps({"state": "complete"}))
    (export / "export.json").write_text(json.dumps({"splits": {
        population: {"sample_membership": {"ordered_symbol_dates_sha256": f"digest-{population}"}}
        for population in ("overlap", "forward")
    }}))
    package = forward / "package"
    package.mkdir()
    (package / "store.bin").write_bytes(b"bars")
    (package / "forward-package.json").write_text(json.dumps({
        "end_exclusive": END, "package_sha256": PACKAGE_SHA,
        "files": {"store.bin": _sha(package / "store.bin")},
        "populations": {"overlap": {"samples": 1}, "forward": {"samples": 1}},
    }))
    checks = forward / "checks"
    checks.mkdir()
    (checks / "input-comparison.json").write_text(json.dumps({
        "tolerance": 1e-6, "min_fraction": 0.99, "package_sha256": PACKAGE_SHA, "passed": True,
        "matched_rows": 10, "coverage": 1.0, "within_tolerance_fraction": 1.0,
        "max_relative_difference": 0.0,
    }))
    return root


@pytest.fixture()
def config_path(tmp_path):
    config = json.loads(stage3.DEFAULT_CONFIG.read_text())
    config["inputs"] = {"alpha_baseline": ALPHA_ID, "direction_baseline": DIRECTION_ID,
                        "predictions_root": "diagnostics/signal-predictions"}
    config["signals"] = {"R22_IR": {"kind": "alpha", "members": [[RUN, CHECKPOINT]]}}
    config["data"]["end_exclusive"] = END
    config["bootstrap"]["replicates"] = 50
    path = tmp_path / "stage3.json"
    path.write_text(json.dumps(config))
    return path


def _run(config_path, artifacts, name):
    output = artifacts / "diagnostics" / "backtest" / name
    return stage3.run(config_path, artifacts, output, workers=1), output


def test_all_checks_pass_before_the_forward_population_is_scored(forward_artifacts,
                                                                 config_path):
    report, output = _run(config_path, forward_artifacts, "stage3-a")
    assert all(check["passed"] for check in report["checks"].values())
    assert report["checks"]["pipeline"]["identical"] is True
    assert report["checks"]["overlap_labels"]["coverage"] == 1.0
    assert report["scored"] is True
    assert set(report["decision"]) >= {"passed", "top_low", "bottom_mean", "spread_mean",
                                       "net_excess_annual"}
    ranking = pd.read_csv(output / "ranking.csv")
    assert set(ranking["split"]) == {"forward"}
    assert set(ranking["h"]) == {5, 10}
    assert report["exports"] == {f"{RUN}/{CHECKPOINT}": "export-1"}
    assert "主要假說" in (output / "summary.md").read_text()


@pytest.mark.parametrize("breakage", ["comparison", "labels", "package", "export"])
def test_any_failed_check_blocks_scoring(forward_artifacts, config_path, breakage):
    forward = forward_artifacts / "forward" / END
    if breakage == "comparison":
        (forward / "checks" / "input-comparison.json").unlink()
        failed = "inputs"
    elif breakage == "labels":
        directory = forward / "labels" / "inputs" / "overlap"
        values = np.load(directory / "absolute_targets.npy")
        np.save(directory / "absolute_targets.npy", values + 0.01)
        _write_labels(directory, "overlap")
        failed = "overlap_labels"
    elif breakage == "package":
        (forward / "package" / "store.bin").write_bytes(b"changed")
        failed = "package"
    else:
        shutil.rmtree(forward_artifacts / "diagnostics" / "forward-predictions")
        failed = "pipeline"
    report, output = _run(config_path, forward_artifacts, "stage3-b")
    assert report["checks"][failed]["passed"] is False
    assert report["scored"] is False and report["decision"] is None
    assert not (output / "ranking.csv").exists()
    assert "未計算任何前瞻指標" in (output / "summary.md").read_text()


def test_a_second_export_set_is_refused_after_scoring(forward_artifacts, config_path):
    _run(config_path, forward_artifacts, "stage3-a")
    base = forward_artifacts / "diagnostics/forward-predictions" / END / RUN / CHECKPOINT
    shutil.copytree(base / "export-1", base / "export-2")
    with pytest.raises(ValueError, match="already scored"):
        _run(config_path, forward_artifacts, "stage3-c")


def test_decision_requires_both_registered_conditions(config_path):
    config = json.loads(config_path.read_text())
    keys = {"split": "forward", "signal": "R22_IR", "market": "TWSE", "h": 10}
    ranking = pd.DataFrame([{**keys, "top_mean": 0.01, "top_low": 0.002, "top_high": 0.02,
                             "bottom_mean": 0.0, "spread_mean": 0.01}])
    portfolio = pd.DataFrame([{**keys, "net_excess_annual": 0.1, "net_excess_annual_low": -0.1,
                               "net_excess_annual_high": 0.3}])
    assert stage3.decide(config, ranking, portfolio)["passed"] is True
    ranking.loc[0, "top_low"] = -0.001
    assert stage3.decide(config, ranking, portfolio)["passed"] is False
    ranking.loc[0, "top_low"] = 0.002
    portfolio.loc[0, "net_excess_annual"] = -0.01
    assert stage3.decide(config, ranking, portfolio)["passed"] is False
    config["primary"]["top_decile"]["block"] = 10
    with pytest.raises(ValueError, match="bootstrap block"):
        stage3.decide(config, ranking, portfolio)
