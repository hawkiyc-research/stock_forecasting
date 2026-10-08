"""Cloud-only end-to-end training interruption and hardware-reprobe acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from stock_forecasting import training
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.manifest import artifact_metadata

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
@pytest.mark.parametrize(
    "schedule,seed",
    [("cosine", 42), ("sample_plateau", 42), ("sample_plateau", 43), ("sample_plateau", 44)],
)
def test_training_checkpoint_resume_after_reprobe_preserves_final_weights(
    tmp_path,
    market_frame,
    monkeypatch,
    schedule,
    seed,
):
    raw = tmp_path / "market.parquet"
    market_frame.to_parquet(raw, index=False)
    store = tmp_path / "bar-store"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        window_size=32,
        bucket_count=2,
        batch_rows=512,
        purge_bars=20,
        embargo_bars=14,
        download_manifest={
            "artifacts": {
                "raw": artifact_metadata(
                    raw,
                    root=tmp_path,
                    row_count=len(market_frame),
                )
            }
        },
    )
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.raw_path = raw
    config.data.bar_store_path = store
    config.data.input_length = 32
    config.data.train_fraction = 1.0
    config.data.max_samples = 64
    config.data.label_scale_calibration_samples = 64
    config.model.resampler_dropout = 0.1
    config.model.benchmark_conditioner_dropout = 0.1
    config.model.alpha_head_dropout = 0.1
    config.training.stage = "stage2"
    config.training.seed = seed
    config.training.yearly_sampling_decay = 0.8
    config.training.epochs = 2
    config.training.num_workers = 2
    config.training.batch_size = 8
    config.training.evaluation_batch_size = 64
    config.training.gradient_accumulation_steps = 2
    config.training.target_effective_batch_size = 16
    if schedule == "sample_plateau":
        config.training.warmup_samples = 16
        config.training.sample_decay_start = 32
        config.training.sample_decay_end = 96
        config.training.learning_rate_schedule = schedule
        config.training.market_loss_weights = [1, 2, 2, 1]
        config.model.market_aware = True
        config.model.market_residual_hidden_dim = 16
    config.training.evaluations_per_epoch = 2
    config.training.loss_log_points_per_epoch = 4
    config.training.checkpoint_save_top_k = 2
    config.training.output_root = tmp_path / "savedModel"
    config.wandb.directory = tmp_path / "tracking"
    config.validation.require_prebuilt_baselines = False

    def select_run(name):
        monkeypatch.setenv("WANDB_RUN_ID", name)
        monkeypatch.setenv("RUNPOD_RUN_KEY", name)

    select_run("qa-continuous")
    continuous = training.train(config)
    recorded = json.loads((continuous.run_directory / "run-manifest.json").read_text())
    assert recorded["training_resume_contract"]["training"]["seed"] == seed
    assert continuous.global_step == 8 and continuous.processed_train_samples == 128
    expected = load_file(continuous.completion_result / "adapter.safetensors")

    select_run("qa-interrupted")
    original_save = training.save_ranked_checkpoint
    checkpoints = []

    def interrupt_after_commit(**kwargs):
        saved, _ranking = original_save(**kwargs)
        assert saved is not None and kwargs["global_step"] == 2
        checkpoints.append(saved)
        raise RuntimeError("injected interruption after durable checkpoint")

    with monkeypatch.context() as interruption:
        interruption.setattr(training, "save_ranked_checkpoint", interrupt_after_commit)
        with pytest.raises(RuntimeError, match="injected interruption"):
            training.train(config)
    config.training.resume_checkpoint = checkpoints[0]
    measured = []
    original_measure = training._measure_cuda_batch

    def observe_probe(**kwargs):
        measured.append(kwargs["training"])
        return original_measure(**kwargs)

    # Force the public resume path to retune without rewriting checkpoint manifests.
    monkeypatch.setattr(
        training,
        "runtime_resource_plan_reuse_reason",
        lambda *_args, **_kwargs: "pipeline_probe_version_changed",
    )
    monkeypatch.setattr(training, "_measure_cuda_batch", observe_probe)
    resumed = training.train(config)
    assert True in measured and False in measured
    assert resumed.global_step == continuous.global_step
    assert resumed.processed_train_samples == continuous.processed_train_samples
    assert resumed.completed_epochs == continuous.completed_epochs == 2
    assert resumed.validation_evaluations == continuous.validation_evaluations == 4
    actual = load_file(resumed.completion_result / "adapter.safetensors")
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)
    runtime = json.loads((resumed.run_directory / "runtime-execution-plan.json").read_text())
    assert runtime["runtime_execution_plan"]["resume_canonical_global_step"] == 2
