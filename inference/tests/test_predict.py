import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from localinfer import fetch_models, predict

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO_ROOT / "configs" / "experiments" / "a_alpha_point64.yaml"


def fake_checkpoint(tmp_path, *, digest=None, task="alpha_point"):
    from stock_forecasting.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(EXPERIMENT)
    checkpoint = tmp_path / "run-fixture-alpha" / "checkpoint-000010"
    config.save_resolved(checkpoint / "resolved-config.yaml")
    state = {"model_architecture_sha256": digest or config.model_architecture_digest(),
             "prediction_task": task, "global_step": 10}
    (checkpoint / "trainer-state.json").write_text(json.dumps(state))
    return checkpoint, config


def test_taiwan_positions_follow_the_sequential_ranges():
    dataset = SimpleNamespace(
        _range_ends=np.array([3, 5, 9, 10]),
        _range_symbols=np.array(["2330", "AAPL.US", "6488", "2330"]),
        _index={"2330": {"market": "TWSE"}, "AAPL.US": {"market": "US"},
                "6488": {"market": "TPEX"}},
    )
    assert predict.taiwan_positions(dataset).tolist() == [0, 1, 2, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError, match="no rows"):
        predict.taiwan_positions(dataset, markets=("JP",))


def test_config_is_checked_before_the_kronos_source_moves(tmp_path):
    checkpoint, config = fake_checkpoint(tmp_path)
    loaded, state = predict.load_config(checkpoint, kronos_source=tmp_path / "kronos")
    assert loaded.model.kronos_source_root == tmp_path / "kronos"
    assert state["global_step"] == 10
    assert loaded.model.prediction_task == config.model.prediction_task
    stale, _ = fake_checkpoint(tmp_path / "other", digest="0" * 64)
    with pytest.raises(ValueError, match="architecture"):
        predict.load_config(stale)
    wrong, _ = fake_checkpoint(tmp_path / "wrong", task="absolute_direction")
    with pytest.raises(ValueError, match="signal checkpoint"):
        predict.load_config(wrong)


def test_vendored_kronos_source_matches_the_pinned_digests():
    from stock_forecasting.factory import (
        PINNED_KRONOS_SOURCE_REVISION,
        verify_kronos_source_revision,
    )

    revision = verify_kronos_source_revision(predict.VENDORED_KRONOS,
                                             PINNED_KRONOS_SOURCE_REVISION)
    assert revision == PINNED_KRONOS_SOURCE_REVISION


def test_devices_workers_and_output_names(tmp_path, monkeypatch):
    assert predict.select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError):
        predict.select_device("cuda")
    monkeypatch.setenv("LOCAL_INFERENCE_WORKERS", "3")
    assert predict.default_workers() == 3
    monkeypatch.setenv("LOCAL_INFERENCE_WORKERS", "99")
    with pytest.raises(ValueError):
        predict.default_workers()
    monkeypatch.delenv("LOCAL_INFERENCE_WORKERS")
    assert 1 <= predict.default_workers() <= predict.MAX_WORKERS
    checkpoint = tmp_path / "run-x-alpha" / "checkpoint-000010"
    output = predict.output_directory(tmp_path / "out", checkpoint, "validation")
    assert output.parent == tmp_path / "out" / "run-x-alpha" / "checkpoint-000010"
    assert output.name.startswith("validation-")


def test_pinned_models_come_from_the_checkpoint(tmp_path):
    checkpoint, config = fake_checkpoint(tmp_path)
    pairs = fetch_models.pinned_models(checkpoint)
    assert pairs == [
        (config.model.time_series_model_id, config.model.time_series_model_revision),
        (config.model.time_series_tokenizer_id, config.model.time_series_tokenizer_revision),
    ]
