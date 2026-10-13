"""Independent-task train, checkpoint, resume and full holdout acceptance."""

import json

import pytest
import torch
from safetensors.torch import load_file
from test_direction_v2_training import (
    FIXED_SPLIT,
    _config,
    _write_ready_manifest,
)
from test_direction_v2_training import deterministic_resume_cuda as deterministic_resume_cuda
from torch.utils.data import SequentialSampler

from stock_forecasting import training
from stock_forecasting.cli.evaluate import evaluate_checkpoint
from stock_forecasting.config import ModelConfig
from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.manifest import artifact_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA")
@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_independent_signal_train_resume_full_test(
    tmp_path, market_frame, monkeypatch, deterministic_resume_cuda, task
):
    raw, store = tmp_path / "market.parquet", tmp_path / "bar-store"
    market_frame.to_parquet(raw, index=False)
    prepared = build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        window_size=128,
        bucket_count=2,
        batch_rows=512,
        purge_bars=20,
        embargo_bars=14,
        fixed_split=FIXED_SPLIT,
        workers=2,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(market_frame))}
        },
    )
    config = _config(tmp_path, raw, store)
    config.model = ModelConfig.model_validate(
        {
            **config.model.model_dump(),
            "prediction_task": task,
            "direction": {"enabled": False},
            "decoupled_output_scale": False,
        }
    )
    config.training.checkpoint_monitor = "primary_5d/selection_score"
    _write_ready_manifest(config, market_frame, prepared)
    evaluations = []
    original_evaluate = training.evaluate_loader

    def observe(bundle, loader, experiment, device, **kwargs):
        assert isinstance(loader.sampler, SequentialSampler)
        assert len(loader.sampler) == len(loader.dataset) and not loader.drop_last
        assert loader.num_workers == 2
        result = original_evaluate(bundle, loader, experiment, device, **kwargs)
        assert result["prediction_task"] == task and result["samples"] == len(loader.dataset)
        evaluations.append((loader.dataset.split, result["sample_membership"]))
        return result

    monkeypatch.setattr(training, "evaluate_loader", observe)
    monkeypatch.setattr("stock_forecasting.cli.evaluate.evaluate_loader", observe)
    monkeypatch.setenv("WANDB_RUN_ID", f"qa-{task}-continuous")
    monkeypatch.setenv("RUNPOD_RUN_KEY", f"qa-{task}-continuous")
    continuous = training.train(config)
    expected = load_file(continuous.completion_result / "adapter.safetensors")
    events = [
        json.loads(line)
        for line in (continuous.run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    session = next(row for row in events if row.get("event") == "training_session_started")
    assert "pinball" not in session["loss_definition"]
    metric_name = "train/normalized_smooth_l1" if task == "alpha_point" else "train/cross_entropy"
    assert any(metric_name in row for row in events)
    assert not any("train/pinball_loss" in row for row in events)
    assert continuous.processed_train_samples == 128 and continuous.validation_evaluations == 4
    monkeypatch.setenv("WANDB_RUN_ID", f"qa-{task}-interrupted")
    monkeypatch.setenv("RUNPOD_RUN_KEY", f"qa-{task}-interrupted")
    original_save, checkpoints = training.save_ranked_checkpoint, []

    def interrupt(**kwargs):
        saved, _ = original_save(**kwargs)
        checkpoints.append(saved)
        raise RuntimeError("injected signal checkpoint interruption")

    with monkeypatch.context() as interruption:
        interruption.setattr(training, "save_ranked_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="injected signal"):
            training.train(config)
    config.training.resume_checkpoint = checkpoints[0]
    resumed = training.train(config)
    assert resumed.processed_train_samples == continuous.processed_train_samples
    actual = load_file(resumed.completion_result / "adapter.safetensors")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)
    state = json.loads((resumed.completion_result / "training-result.json").read_text())
    assert state["model_output_schema_version"] == "signal-1.0"
    report = evaluate_checkpoint(config, resumed.final_checkpoint, split="test")
    assert report["metrics"]["prediction_task"] == task and report["split"] == "test"
    assert report["metrics"]["execution"]["full_population"] is True
    memberships = [membership for split, membership in evaluations if split == "validation"]
    assert all(value == memberships[0] for value in memberships)
    (tmp_path / "signal-evidence.json").write_text(
        json.dumps(
            {
                "prediction_task": task,
                "processed_train_samples": resumed.processed_train_samples,
                "resume_weights_match": True,
                "validation": memberships[0],
                "test": report["metrics"]["sample_membership"],
            },
            indent=2,
        )
    )
