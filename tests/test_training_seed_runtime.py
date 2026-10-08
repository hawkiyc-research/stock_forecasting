"""Cloud runtime checks for the real config loader and immutable seed snapshots."""

from __future__ import annotations

import argparse
import runpy
import shutil
from pathlib import Path

import pytest

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.runpod.configuration import load_run_config

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))


@pytest.mark.parametrize("seed", [0, 42, 43, 44, 2**32 - 1])
def test_real_config_loader_seed_override_roundtrip(tmp_path, monkeypatch, seed):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    selected = SELECTION["_build_selection"](
        argparse.Namespace(
            stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
            start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
            stocks=[], etfs=[], symbol_limit=None, feature_mode="combined",
        ), tmp_path,
    )
    selected = SELECTION["_with_experiment"](selected, "a-adaptive64", tmp_path)
    selected = SELECTION["_with_training_seed"](selected, seed, tmp_path)
    source = SELECTION["_store_selection"](tmp_path, selected)
    environment = SELECTION["_selection_exports"](source, selected)
    environment.update(PROJECT_ROOT=str(tmp_path), RUNPOD_REMOTE_SELECTION_PATH=str(source))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    path = tmp_path / selected["stage"]["config_path"]
    original = ExperimentConfig.from_yaml(path)
    loaded = load_run_config(path)
    expected = original.model_copy(deep=True)
    expected.training.seed = seed
    assert loaded.model_dump(mode="json") == expected.model_dump(mode="json")
    assert ExperimentConfig.from_yaml(path).training.seed == 42
    resolved = tmp_path / "savedModel/run-fixture/resolved-config.yaml"
    loaded.save_resolved(resolved)
    monkeypatch.setenv("RUNPOD_TRAINING_SEED", "123")
    assert load_run_config(resolved).training.seed == seed
