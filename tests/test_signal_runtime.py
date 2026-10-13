"""Cloud acceptance for inference and canonical independent-task checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
    save_training_completion_result,
    validate_checkpoint_trainer_state,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)
from stock_forecasting.run_contract import training_resume_contract_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def configuration(task):
    payload = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml").as_dict()
    payload["data"]["h_start"] = 1
    payload["validation"]["calibrate_intervals"] = False
    payload["model"].update(
        prediction_task=task, feature_mode="combined", explicit_output_scale=True, market_aware=True
    )
    return ExperimentConfig.model_validate(payload)


def inputs():
    time = torch.arange(128, dtype=torch.float32)[None, :, None]
    offset = torch.arange(4, dtype=torch.float32)[:, None, None]
    close = (4 + 0.003 * time + 0.04 * (time * 0.3 + offset).sin()).exp()
    asset = torch.cat((close, close * 1.01, close * 0.99, close, close * 1000), -1)
    benchmark = asset.clone()
    benchmark[..., :4] *= (0.005 * time).exp()
    return dict(
        asset_ohlcv=asset,
        benchmark_ohlcv=benchmark,
        market_ids=torch.arange(4),
        target_alpha=torch.randn(4, 14) * 0.03,
        target_absolute=torch.randn(4, 14) * 0.03,
    )


def create_model(config, data):
    bundle = build_model_bundle(config, torch.device("cpu"), robust_scales=[0.03] * 14)
    features = historical_scale_features(
        data["asset_ohlcv"], data["benchmark_ohlcv"], extended=True
    )
    bundle.model.alpha_head.numeric_branch.set_statistics(
        fit_scale_feature_statistics(features.numpy(), {"split": "train"})
    )
    return bundle.model


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_canonical_checkpoint_resume_completion_and_readonly_load(task, tmp_path):
    config = configuration(task)
    torch.manual_seed(31)
    data = inputs()
    original = create_model(config, data)
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)

    def step(network, opt, schedule):
        opt.zero_grad(set_to_none=True)
        network(**data).loss.backward()
        opt.step()
        schedule.step()

    step(original, optimizer, scheduler)
    run = tmp_path / f"signal-{task}-checkpoint-run"
    run.mkdir()
    contract, digest = training_resume_contract_fingerprint(config)
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run.name,
                "run_key": run.name,
                "training_resume_contract": contract,
                "training_resume_contract_sha256": digest,
            }
        )
    )
    config.save_resolved(run / "resolved-config.yaml")
    checkpoint = save_checkpoint(
        model=original,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_directory=run,
        global_step=1,
        epoch=0,
        batch_index=1,
        selection_metric_name="primary_5d/selection_score",
        selection_metric_value=0.3,
        selection_metric_mode="min",
    )
    state = validate_checkpoint_trainer_state(checkpoint)
    assert state["prediction_task"] == task
    assert state["model_output_schema_version"] == "signal-1.0"
    assert state["runtime_direction_loss"] is None
    restored = create_model(config, data)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = LambdaLR(restored_optimizer, lambda _: 1.0)
    load_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler, config=config)
    step(original, optimizer, scheduler)
    step(restored, restored_optimizer, restored_scheduler)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    readonly = create_model(config, data)
    load_checkpoint(checkpoint, readonly, config=config, allow_historical_inference=True)
    result = save_training_completion_result(
        model=original,
        config=config,
        run_directory=run,
        global_step=2,
        completed_epochs=1,
        processed_train_samples=8,
        planned_train_samples=8,
        validation_evaluations=1,
        stop_reason="epochs_completed",
        early_stopping_state={},
        metrics={"primary_5d/selection_score": 0.3},
    )
    completion = json.loads((result / "training-result.json").read_text())
    assert completion["prediction_task"] == task
    assert completion["model_output_schema_version"] == "signal-1.0"
    incompatible = configuration("absolute_direction" if task == "alpha_point" else "alpha_point")
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, readonly, config=incompatible, allow_historical_inference=True)


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_inference_exports_only_real_signal_task_without_cost_calibration(
    task, tmp_path, monkeypatch
):
    import stock_forecasting.cli.infer as inference
    import stock_forecasting.direction_costs as cost_module

    config = configuration(task)
    data = inputs()
    model = create_model(config, data)
    checkpoint = tmp_path / "signal-infer" / "checkpoint-000001"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        inference, "run_preflight", lambda *_a, **_kw: SimpleNamespace(require_success=lambda: None)
    )
    monkeypatch.setattr(inference, "build_model_bundle", lambda *_a: SimpleNamespace(model=model))
    monkeypatch.setattr(inference, "resolve_checkpoint", lambda *_a, **_kw: checkpoint)
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_a, **_kw: {})
    monkeypatch.setattr(inference, "provenance_summary", lambda *_a: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent signal inference queried transaction costs")

    monkeypatch.setattr(cost_module, "direction_cost_band", forbidden)
    observation = SimpleNamespace(
        asset_series=data["asset_ohlcv"][:1],
        benchmark_series=data["benchmark_ohlcv"][:1],
        asset_attention_mask=torch.ones(1, 128, dtype=torch.bool),
        benchmark_attention_mask=torch.ones(1, 128, dtype=torch.bool),
        asset_timestamp_features=torch.zeros(1, 128, 5, dtype=torch.long),
        benchmark_timestamp_features=torch.zeros(1, 128, 5, dtype=torch.long),
        symbol="AAPL.US",
        benchmark_symbol="VTI.US",
        asset_type="stock",
        metadata={"market": "US"},
        benchmark_metadata={},
        window_start_at="2025-12-01",
        cutoff_at="2026-06-10",
    )
    result = inference.infer_observation(config, checkpoint, observation)
    assert (
        result["prediction_task"] == task and result["model_output_schema_version"] == "signal-1.0"
    )
    for absent in (
        "calibrated_forecast",
        "direction",
        "ranking_scores",
        "interval_calibration_status",
    ):
        assert absent not in result
    forecast = result["forecast"]
    assert len(forecast["by_horizon"]) == 14
    if task == "alpha_point":
        assert forecast["estimand"] == "robust_conditional_location"
        assert set(forecast["by_horizon"]["5d"]) == {"alpha_point"}
    else:
        assert forecast["neutral_simple_return_interval"] == [-0.01, 0.01]
        assert forecast["neutral_boundaries_inclusive"] is True
        assert sum(
            forecast["by_horizon"]["5d"]["absolute_direction_probabilities"]
        ) == pytest.approx(1)
    assert "alpha_quantiles" not in json.dumps(result)
