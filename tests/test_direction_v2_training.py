"""Cloud-only fixed-split v2 train, full evaluation and durable resume acceptance."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import SequentialSampler

from stock_forecasting import training
from stock_forecasting.cli.evaluate import evaluate_checkpoint
from stock_forecasting.config import DataConfig, DirectionConfig, ExperimentConfig
from stock_forecasting.data.bar_store import (
    bar_store_preparation_provenance,
    build_symbol_bar_store,
)
from stock_forecasting.data.content_identity import (
    content_identity_digest,
    dataset_content_identity,
)
from stock_forecasting.data.manifest import (
    DATASET_MANIFEST_KIND,
    DATASET_MANIFEST_SCHEMA_VERSION,
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    storage_preparation_spec,
)
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE
from stock_forecasting.dataset_identity import DEFAULT_DATASET_STORAGE_PREPARATION
from stock_forecasting.preflight import run_preflight
from stock_forecasting.run_contract import _dataset_resume_contract
from stock_forecasting.tracking import collect_dataset_provenance

ROOT = Path(__file__).resolve().parents[1]
FIXED_SPLIT = {
    "train_end": "2022-10-01",
    "validation_end": "2023-02-01",
    "test_end": "2023-07-01",
}


@pytest.fixture
def deterministic_resume_cuda(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Control CUDA nondeterminism only for resume-equivalence acceptance.
    # Production AMP and kernel settings remain unchanged.
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _config(tmp_path, raw, store):
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.training.stage = "stage2"
    config.data = DataConfig.model_validate(
        {
            **config.data.model_dump(),
            **FIXED_SPLIT,
            "raw_path": raw,
            "bar_store_path": store,
            "manifest_path": tmp_path / "dataset-manifest.json",
            "dataset_profile": "us_tw_eodhd",
            "require_ready_manifest": True,
            "input_length": 128,
            "train_fraction": 1.0,
            "max_samples": 64,
            "label_scale_calibration_samples": 64,
        }
    )
    config.model.feature_mode = "combined"
    config.model.explicit_output_scale = True
    config.model.decoupled_output_scale = True
    config.model.market_aware = True
    config.model.resampler_dropout = 0.1
    config.model.benchmark_conditioner_dropout = 0.1
    config.model.alpha_head_dropout = 0.1
    config.model.direction = DirectionConfig(
        enabled=True,
        architecture="market_residual_v2",
        loss_reference_samples=32,
        market_residual_hidden_dim=16,
        gradient_management="none",
    )
    config.training.seed = 42
    config.training.yearly_sampling_decay = 0.8
    config.training.epochs = 2
    config.training.num_workers = 2
    config.training.batch_size = 8
    config.training.evaluation_batch_size = 32
    config.training.gradient_accumulation_steps = 2
    config.training.target_effective_batch_size = 16
    config.training.evaluations_per_epoch = 2
    config.training.checkpoint_monitor = "direction/selection_loss"
    config.training.loss_log_points_per_epoch = 4
    config.training.checkpoint_save_top_k = 2
    config.training.output_root = tmp_path / "savedModel"
    config.wandb.directory = tmp_path / "tracking"
    config.validation.require_prebuilt_baselines = False
    config.validation.calibrate_intervals = False
    config.validation.output_root = tmp_path / "evaluations"
    config.runtime.log_root = tmp_path / "logs"
    return config


def _write_ready_manifest(config, frame, prepared):
    # Bind the real fixture artifacts; exercise production provenance and readiness gates.
    root = config.data.resolved_manifest_path.parent
    store = json.loads(prepared.bar_store_manifest_path.read_text())
    provenance = bar_store_preparation_provenance(
        window_size=config.data.input_length,
        max_horizon=config.data.max_horizon,
        benchmark_mapping_sha256=canonical_json_sha256({}),
        max_abs_log_return=config.data.max_abs_log_return,
        train_fraction=DEFAULT_DATASET_STORAGE_PREPARATION["train_fraction"],
        validation_fraction=DEFAULT_DATASET_STORAGE_PREPARATION["validation_fraction"],
        purge_bars=DEFAULT_DATASET_STORAGE_PREPARATION["purge_bars"],
        compatibility_stride=config.data.stride,
        effective_sample_stride=config.data.sample_stride,
        compatibility_embargo_bars=config.data.embargo_trading_days,
        effective_embargo_bars=config.data.effective_embargo_trading_days,
        target_horizon=config.data.forecast_horizon,
        diagnostic_horizons=config.data.diagnostic_horizons,
        flat_volatility_multiplier=config.data.flat_volatility_multiplier,
        fixed_split=config.data.fixed_split,
    )
    storage_spec = storage_preparation_spec(provenance)
    content_identity = dataset_content_identity(config.data.selected_datasets)
    symbol_values = sorted(frame["symbol"].unique().tolist())
    symbols = {
        "count": len(symbol_values),
        "values": symbol_values,
        "asset_type_counts": dict(
            sorted(Counter(str(value) for value in frame["asset_type"]).items())
        ),
    }
    atomic_write_json(
        config.data.resolved_manifest_path,
        {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "kind": DATASET_MANIFEST_KIND,
            "state": "ready",
            "training_security_scope": TRAINING_SECURITY_SCOPE,
            "dataset_profile": config.data.dataset_profile,
            "selected_datasets": config.data.selected_datasets,
            "providers": sorted(frame["provider"].unique().tolist()),
            "markets": sorted(frame["market"].unique().tolist()),
            "symbols": symbols,
            "universe_sha256": canonical_json_sha256(symbols),
            "date_range": {
                "start_inclusive": frame["timestamp"].min().date().isoformat(),
                "end_exclusive": (frame["timestamp"].max().date() + timedelta(days=1)).isoformat(),
            },
            "data_content_identity": content_identity,
            "data_pipeline_digest": content_identity_digest(content_identity),
            "split_counts": prepared.split_counts,
            "split_audit": prepared.split_audit,
            "preparation_provenance": provenance,
            "storage_preparation_spec": storage_spec,
            "storage_preparation_spec_sha256": canonical_json_sha256(storage_spec),
            "label_statistics": {
                "state": "runtime_calibration_required",
                "source_split": "train",
                "horizons_available": list(range(1, config.data.max_horizon + 1)),
                "robust_scale_method": "max(iqr,mad_x_1.4826,1e-4)",
                "prediction_units": "benchmark_relative_log_return",
            },
            "quality": {"bar_store": prepared.quality},
            "artifacts": {
                "raw": artifact_metadata(config.data.raw_path, root=root, row_count=len(frame)),
                "bar_store_manifest": artifact_metadata(
                    prepared.bar_store_manifest_path, root=root, row_count=1
                ),
                "symbol_index": artifact_metadata(
                    prepared.symbol_index_path,
                    root=root,
                    row_count=store["symbol_index"]["row_count"],
                ),
                "cutoff_ranges": artifact_metadata(
                    prepared.cutoff_ranges_path,
                    root=root,
                    row_count=store["cutoff_ranges"]["row_count"],
                ),
            },
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
@pytest.mark.parametrize("architecture", ["market_residual_v2", "market_ordinal_v3"])
def test_fixed_split_direction_v2_train_resume_and_full_holdout(
    tmp_path,
    market_frame,
    monkeypatch,
    deterministic_resume_cuda,
    architecture,
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
    config.model.direction.architecture = architecture
    schema = 3 if architecture == "market_ordinal_v3" else 2
    _write_ready_manifest(config, market_frame, prepared)
    # Validate every manifest consumer before the expensive CUDA training acceptance.
    run_preflight(config, require_data=True).require_success()
    provenance = collect_dataset_provenance(config)
    assert provenance["status"] == "ready"
    assert provenance["symbols"]["count"] == market_frame["symbol"].nunique()
    dataset_contract = _dataset_resume_contract(config)
    assert dataset_contract["status"] == "ready"
    assert dataset_contract["universe_sha256"] == canonical_json_sha256(provenance["symbols"])
    assert dataset_contract["artifacts"] == provenance["artifacts"]
    evaluations = []
    original_evaluate = training.evaluate_loader

    def observe_full_evaluation(bundle, loader, experiment, device, **kwargs):
        assert isinstance(loader.sampler, SequentialSampler)
        assert len(loader.sampler) == len(loader.dataset)
        assert loader.num_workers == 2 and not loader.drop_last
        result = original_evaluate(bundle, loader, experiment, device, **kwargs)
        assert result["direction"]["samples"] == len(loader.dataset)
        assert result["direction"]["schema_version"] == schema
        assert result["direction"]["probability_source"] == "population_logits"
        assert sum(row["samples"] for row in result["direction"]["daily"].values()) == len(
            loader.dataset
        )
        evaluations.append((loader.dataset.split, len(loader.dataset), result["sample_membership"]))
        return result

    monkeypatch.setattr(training, "evaluate_loader", observe_full_evaluation)
    # The evaluate CLI imports its function directly, so observe that path too.
    monkeypatch.setattr("stock_forecasting.cli.evaluate.evaluate_loader", observe_full_evaluation)

    def select_run(name):
        monkeypatch.setenv("WANDB_RUN_ID", name)
        monkeypatch.setenv("RUNPOD_RUN_KEY", name)

    select_run("qa-v2-continuous")
    continuous = training.train(config)
    events = [
        json.loads(line)
        for line in (continuous.run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    session = next(row for row in events if row.get("event") == "training_session_started")
    if schema == 3:
        assert "population_absolute_conditional_NLL" in session["loss_definition"]
        assert "Brier_prior_ratio" in session["loss_definition"]
    else:
        assert session["loss_definition"] == (
            "train=market_weighted_normalized_pinball"
            "+fixed_reference_scaled_population_absolute_CE"
            "+fixed_reference_scaled_population_alpha_CE"
            "+weighted_ranking+reference_scaled_representation_regularization; "
            "shared_gradient_management=none(LoRA_only,not_an_additive_loss); "
            "validation=1-equal_market_task_horizon_balanced_decision_accuracy"
            "+frozen_first_full_validation_pinball_guard"
        )
    expected = load_file(continuous.completion_result / "adapter.safetensors")
    manifest = json.loads((continuous.run_directory / "run-manifest.json").read_text())
    assert manifest["training_resume_contract"]["training"]["seed"] == 42
    continuous_state = json.loads(
        (continuous.completion_result / "training-result.json").read_text()
    )
    reference = continuous_state["runtime_direction_loss"]
    assert reference["schema_version"] == schema and reference["samples"] == 32
    assert reference["sampling"]["calibration_seed"] == config.data.calibration_seed
    assert reference["sampling"]["yearly_sampling_decay"] == 0.8
    assert reference["pinball_guard_basis"] == "first_full_validation"
    assert reference["pinball_guard_reference"] > 0
    assert continuous.processed_train_samples == 128 and continuous.validation_evaluations == 4
    flat = training._flatten_metrics(continuous.validation_metrics)
    assert "direction/selection_loss" in flat
    assert "direction/balanced_decision/market_macro/balanced_accuracy" in flat

    select_run("qa-v2-interrupted")
    original_save = training.save_ranked_checkpoint
    checkpoints = []

    def interrupt_after_commit(**kwargs):
        saved, _ranking = original_save(**kwargs)
        assert saved is not None and kwargs["global_step"] == 2
        checkpoints.append(saved)
        raise RuntimeError("injected interruption after durable v2 checkpoint")

    with monkeypatch.context() as interruption:
        interruption.setattr(training, "save_ranked_checkpoint", interrupt_after_commit)
        with pytest.raises(RuntimeError, match="durable v2 checkpoint"):
            training.train(config)
    saved_state = json.loads((checkpoints[0] / "trainer-state.json").read_text())
    saved_reference = saved_state["runtime_direction_loss"]
    assert saved_reference == reference
    config.training.resume_checkpoint = checkpoints[0]
    resumed = training.train(config)
    assert resumed.global_step == continuous.global_step
    assert resumed.processed_train_samples == continuous.processed_train_samples
    assert resumed.validation_evaluations == continuous.validation_evaluations
    actual = load_file(resumed.completion_result / "adapter.safetensors")
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)
    resumed_state = json.loads((resumed.completion_result / "training-result.json").read_text())
    assert resumed_state["runtime_direction_loss"] == saved_reference

    report = evaluate_checkpoint(config, resumed.final_checkpoint, split="test")
    assert report["split"] == "test"
    assert report["data_provenance"]["dataset_profile"] == "us_tw_eodhd"
    assert report["data_provenance"]["split_counts"] == prepared.split_counts
    assert report["samples"] == report["metrics"]["direction"]["samples"]
    assert (
        report["metrics"]["direction"]["selection_definition"]["pinball_guard_reference"]
        == reference["pinball_guard_reference"]
    )
    assert report["metrics"]["direction"]["loss_reference"] == reference
    assert any(split == "test" for split, _count, _membership in evaluations)
    validation_memberships = [
        membership for split, _count, membership in evaluations if split == "validation"
    ]
    assert all(membership == validation_memberships[0] for membership in validation_memberships)

    # Different model initialization must not alter train-only reference constants.
    select_run("qa-v2-seed43")
    config.training.resume_checkpoint = None
    config.training.seed = 43
    second_seed = training.train(config)
    second_state = json.loads((second_seed.completion_result / "training-result.json").read_text())
    other = copy.deepcopy(second_state["runtime_direction_loss"])
    same = copy.deepcopy(reference)
    for value in (same, other):
        value.pop("pinball_guard_reference")
        value.pop("pinball_guard_basis")
    assert other == same
    # Persist compact acceptance evidence only after every assertion has succeeded.
    atomic_write_json(
        tmp_path / "v2-e2e-evidence.json",
        {
            "processed_train_samples": resumed.processed_train_samples,
            "validation_evaluations": resumed.validation_evaluations,
            "full_split_counts": {
                "validation": validation_memberships[0]["samples"],
                "test": report["samples"],
            },
            "seed_reference_equal": other == same,
            "resume_weight_match": True,
            "test_anchor_unchanged": True,
            "validation_membership_sha256": validation_memberships[0][
                "ordered_symbol_dates_sha256"
            ],
        },
    )
