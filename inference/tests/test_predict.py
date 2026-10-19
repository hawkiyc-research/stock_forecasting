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


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.robust_scales = torch.tensor([0.0123456789, 0.0234567891])
        self.numeric_branch = None


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.alpha_head = TinyHead()
        self.market_loss_weights = torch.tensor([1.0, 1.0])


def test_bf16_precision_mirrors_the_cuda_cast():
    model = TinyModel()
    scales = model.alpha_head.robust_scales.clone()
    predict.apply_precision(model, "fp32")
    assert model.frozen.weight.dtype == torch.float32
    predict.apply_precision(model, "bf16")
    assert model.frozen.weight.dtype == torch.bfloat16
    assert model.alpha_head.layer.weight.dtype == torch.float32  # trainable stays fp32
    assert torch.equal(model.alpha_head.robust_scales, scales)  # not rounded through bf16
    assert model.market_loss_weights.dtype == torch.float32
    with pytest.raises(ValueError):
        predict.apply_precision(model, "fp16")
    with predict.autocast(torch.device("cpu"), "bf16"):
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_pilot_selects_one_taiwan_date_and_summarizes_agreement():
    from localinfer import pilot

    metadata = np.array(
        [("2330", "2025-06-03T00:00:00+00:00", "TWSE"),
         ("AAPL", "2025-06-02T00:00:00+00:00", "US"),
         ("6488", "2025-06-03T00:00:00+00:00", "TPEX"),
         ("2317", "2025-06-04T00:00:00+00:00", "TWSE")],
        dtype=[("symbol", "<U8"), ("date", "<U32"), ("market", "<U8")],
    )
    date, positions = pilot.date_positions(metadata, None)
    assert date == "2025-06-03" and positions.tolist() == [0, 2]
    with pytest.raises(ValueError):
        pilot.date_positions(metadata, "2025-06-02")
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(6, 3))
    markets = np.array(["TWSE"] * 3 + ["TPEX"] * 3)
    summary = pilot.agreement(reference.copy(), reference, markets)
    assert summary["TWSE"]["min"] == pytest.approx(1.0) and summary["TPEX"]["rows"] == 3
