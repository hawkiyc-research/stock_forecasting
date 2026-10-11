"""Cloud acceptance for deterministic task-gradient checkpoint boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
    save_training_completion_result,
)
from stock_forecasting.config import DirectionConfig, ExperimentConfig
from stock_forecasting.models.direction import MarketResidualDualDirectionHead
from stock_forecasting.multitask_gradients import SharedTaskGradientManager
from stock_forecasting.run_contract import training_resume_contract_fingerprint

ROOT = Path(__file__).resolve().parents[1]
V2_DIRECTION_FIELDS = {
    "architecture",
    "class_balance_power",
    "market_residual_hidden_dim",
    "prior_smoothing",
    "market_weight_max",
    "gradient_management",
    "gradient_diagnostics_interval",
    "probability_bins",
    "max_daily_entries",
    "selection_pinball_guard_tolerance",
    "selection_pinball_guard_weight",
}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA"),
    ),
]


class _CheckpointModel(nn.Module):
    def __init__(self, config: ExperimentConfig, *, manager: bool, device: str):
        super().__init__()
        self.lora_a = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], device=device))
        self.alpha_head = nn.Module()
        horizons = config.data.alpha_horizons
        self.alpha_head.register_buffer("robust_scales", torch.ones(len(horizons), device=device))
        if config.model.direction.enabled:
            self.direction_head = MarketResidualDualDirectionHead(
                32,
                horizons,
                config.model.direction,
                0,
            ).to(device)
            self.direction_head.restore_references(
                {
                    "schema_version": 2,
                    "samples": 8192,
                    "references": [0.3, math.log(3), math.log(3)],
                    "priors": [
                        [[[0.5, 0.25, 0.25] for _ in horizons] for _ in range(2)] for _ in range(4)
                    ],
                    "market_weights": [0.6, 1.2, 2.0, 1.0],
                    "basis": "train_null_quantiles_and_uniform_classes_v2",
                }
            )
            self.direction_head.set_pinball_guard_reference(0.35)
        self._task_gradient_manager = (
            SharedTaskGradientManager(
                self.named_parameters(),
                seed=config.training.seed,
                diagnostics_interval=1,
                mode=config.model.direction.gradient_management,
            )
            if manager
            else None
        )


def _config(*, v2: bool = True) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    if v2:
        config.model.feature_mode = "combined"
        config.model.explicit_output_scale = True
        config.model.direction = DirectionConfig(
            enabled=True,
            architecture="market_residual_v2",
            gradient_management="pcgrad",
        )
    return config


def _run(tmp_path: Path, config: ExperimentConfig) -> Path:
    directory = tmp_path / "direction-v2-checkpoint-run"
    directory.mkdir()
    contract, digest = training_resume_contract_fingerprint(config)
    (directory / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": directory.name,
                "run_key": directory.name,
                "training_resume_contract": contract,
                "training_resume_contract_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return directory


def _optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return optimizer, LambdaLR(optimizer, lambda _step: 1.0)


def _step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    shared = model.lora_a
    losses = {
        "regression": shared[0] + 2 * shared[1],
        "absolute": -shared[0] + shared[2],
        "alpha": shared[0] - shared[1] - shared[2],
    }
    manager = model._task_gradient_manager
    correction = manager.prepare_correction(losses)
    sum(losses.values()).backward()
    manager.apply_correction(correction)
    gradient = shared.grad.detach().clone()
    optimizer.step()
    scheduler.step()
    manager.optimizer_step_completed()
    return gradient


def _save(config, directory, model, optimizer, scheduler):
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_directory=directory,
        global_step=1,
        epoch=0,
        batch_index=1,
        selection_metric_name="direction/selection_loss",
        selection_metric_value=0.75,
        selection_metric_mode="min",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_projection_checkpoint_restores_counters_and_the_next_optimizer_step(tmp_path, device):
    config = _config()
    original = _CheckpointModel(config, manager=True, device=device)
    optimizer, scheduler = _optimizer(original)
    _step(original, optimizer, scheduler)
    checkpoint = _save(config, _run(tmp_path, config), original, optimizer, scheduler)
    saved = json.loads((checkpoint / "trainer-state.json").read_text())
    assert saved["runtime_task_gradients"] == original._task_gradient_manager.state_dict()

    restored = _CheckpointModel(config, manager=True, device=device)
    restored_optimizer, restored_scheduler = _optimizer(restored)
    load_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler, config=config)
    assert restored._task_gradient_manager.state_dict() == saved["runtime_task_gradients"]
    assert restored.direction_head.reference_state() == original.direction_head.reference_state()
    expected_gradient = _step(original, optimizer, scheduler)
    actual_gradient = _step(restored, restored_optimizer, restored_scheduler)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(restored.lora_a, original.lora_a, rtol=0, atol=0)
    assert (
        restored._task_gradient_manager.state_dict() == original._task_gradient_manager.state_dict()
    )


def test_completion_result_also_preserves_the_task_gradient_state(tmp_path):
    config = _config()
    model = _CheckpointModel(config, manager=True, device="cpu")
    optimizer, scheduler = _optimizer(model)
    _step(model, optimizer, scheduler)
    result = save_training_completion_result(
        model=model,
        config=config,
        run_directory=_run(tmp_path, config),
        global_step=1,
        completed_epochs=1,
        processed_train_samples=4,
        planned_train_samples=4,
        validation_evaluations=1,
        stop_reason="epochs_completed",
        early_stopping_state={},
        metrics={"direction/selection_loss": 0.75},
    )
    saved = json.loads((result / "training-result.json").read_text())
    assert saved["runtime_task_gradients"] == model._task_gradient_manager.state_dict()


@pytest.mark.parametrize("damage", ["missing", "null", "list", "seed", "mode", "counter", "tasks"])
def test_invalid_projection_state_is_rejected_before_optimizer_restore(
    tmp_path,
    monkeypatch,
    damage,
):
    config = _config()
    model = _CheckpointModel(config, manager=True, device="cpu")
    optimizer, scheduler = _optimizer(model)
    checkpoint = _save(config, _run(tmp_path, config), model, optimizer, scheduler)
    state_path = checkpoint / "trainer-state.json"
    state = json.loads(state_path.read_text())
    if damage == "missing":
        state.pop("runtime_task_gradients")
    elif damage == "null":
        state["runtime_task_gradients"] = None
    elif damage == "list":
        state["runtime_task_gradients"] = []
    else:
        field, value = {
            "seed": ("seed", -1),
            "mode": ("mode", "monitor"),
            "counter": ("microbatches", True),
            "tasks": ("task_names", ["same", "same"]),
        }[damage]
        state["runtime_task_gradients"][field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    monkeypatch.setattr(optimizer, "load_state_dict", lambda _value: calls.append("optimizer"))
    with pytest.raises(ValueError, match=r"shared.*gradient"):
        load_checkpoint(checkpoint, model, optimizer, scheduler)
    assert calls == []


def test_readonly_loading_does_not_require_or_restore_a_task_gradient_manager(tmp_path):
    config = _config()
    model = _CheckpointModel(config, manager=True, device="cpu")
    optimizer, scheduler = _optimizer(model)
    _step(model, optimizer, scheduler)
    checkpoint = _save(config, _run(tmp_path, config), model, optimizer, scheduler)
    readonly = _CheckpointModel(config, manager=False, device="cpu")
    state = load_checkpoint(checkpoint, readonly, config=config)
    assert readonly._task_gradient_manager is None
    assert state["runtime_task_gradients"]["optimizer_steps"] == 1
    torch.testing.assert_close(readonly.lora_a, model.lora_a)
    monitored = _CheckpointModel(config, manager=True, device="cpu")
    before = copy.deepcopy(monitored._task_gradient_manager.state_dict())
    load_checkpoint(checkpoint, monitored, config=config)
    assert monitored._task_gradient_manager.state_dict() == before


def test_training_resume_cannot_silently_drop_its_gradient_manager(tmp_path):
    config = _config()
    model = _CheckpointModel(config, manager=True, device="cpu")
    optimizer, scheduler = _optimizer(model)
    checkpoint = _save(config, _run(tmp_path, config), model, optimizer, scheduler)
    no_manager = _CheckpointModel(config, manager=False, device="cpu")
    optimizer, scheduler = _optimizer(no_manager)
    with pytest.raises(ValueError, match="requires its shared task-gradient manager"):
        load_checkpoint(checkpoint, no_manager, optimizer, scheduler)


def test_legacy_training_resume_still_works_without_a_gradient_manager(tmp_path):
    config = _config(v2=False)
    model = _CheckpointModel(config, manager=False, device="cpu")
    optimizer, scheduler = _optimizer(model)
    checkpoint = _save(config, _run(tmp_path, config), model, optimizer, scheduler)
    state_path = checkpoint / "trainer-state.json"
    state = json.loads(state_path.read_text())
    assert state["runtime_task_gradients"] is None
    # Earlier artifacts did not contain the new additive runtime field.
    state.pop("runtime_task_gradients")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    restored = _CheckpointModel(config, manager=False, device="cpu")
    optimizer, scheduler = _optimizer(restored)
    load_checkpoint(checkpoint, restored, optimizer, scheduler, config=config)
    torch.testing.assert_close(restored.lora_a, model.lora_a)


def test_legacy_direction_architecture_digest_matches_the_pre_v2_payload():
    config = _config(v2=False)
    config.model.direction = DirectionConfig(enabled=True)
    payload = json.loads(config.model.model_dump_json())
    assert payload["direction"].keys() >= V2_DIRECTION_FIELDS
    for name in V2_DIRECTION_FIELDS:
        payload["direction"].pop(name)
    payload.pop("feature_mode")
    payload.pop("alpha_head_fp32")
    for name in (
        "market_aware",
        "explicit_output_scale",
        "decoupled_output_scale",
        "independent_ranking_head",
        "market_residual_hidden_dim",
        "ranking_numeric_features",
        "unfreeze_last_blocks",
        "ranking_loss_weight",
        "ranking_max_pairs",
    ):
        payload.pop(name)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert config.model.architecture_digest() == hashlib.sha256(encoded).hexdigest()
    original = config.model.architecture_digest()
    config.model.direction.probability_bins = 25
    config.model.direction.class_balance_power = 0.5
    assert config.model.architecture_digest() == original
    config.model.direction.absolute_weight = 1.5
    assert config.model.architecture_digest() != original


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability_bins", 25),
        ("class_balance_power", 0.5),
        ("market_residual_hidden_dim", 32),
        ("gradient_diagnostics_interval", 123),
        ("gradient_management", "monitor"),
        ("selection_pinball_guard_weight", 2.0),
    ],
)
def test_v2_architecture_digest_still_hashes_all_its_active_fields(field, value):
    config = _config()
    original = config.model.architecture_digest()
    setattr(config.model.direction, field, value)
    assert config.model.architecture_digest() != original
