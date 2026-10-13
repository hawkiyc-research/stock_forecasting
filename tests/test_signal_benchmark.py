"""Cloud integration of persisted signal baselines and final checkpoint evaluation."""

from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from test_direction_v2_training import FIXED_SPLIT, _config, _write_ready_manifest
from torch.utils.data import DataLoader

from stock_forecasting import baseline_contract, training, validation_benchmark
from stock_forecasting.baseline_signal import append_label_audit
from stock_forecasting.config import ModelConfig
from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.manifest import artifact_metadata, sha256_file
from stock_forecasting.data.sample_universe import open_clean_dataset, validated_split_roots
from stock_forecasting.data_policy import evaluation_dataset_id, load_data_policy
from stock_forecasting.direction_data import with_direction_targets
from stock_forecasting.signal_evaluation import SignalEvaluationStore


def _persist_zero_baseline(config, root, identity, scales):
    """Publish a real full-population rule result, without fitting learned comparators."""
    root.mkdir(parents=True)
    roots = validated_split_roots(config)
    counts, sources, scores, class_counts = {}, {}, {}, {}
    task = config.model.prediction_task
    job = root / "jobs/rules/zero_return"
    job.mkdir(parents=True)
    (job / "model.json").write_text(json.dumps({"prediction_task": task, "rule": "zero_return"}))
    workers = training.plan_dataloader_workers(2, source="signal-benchmark-fixture")
    assert workers.effective_workers > 0
    for split in ("train", "validation", "test"):
        source = open_clean_dataset(config, split, validated_root=roots[split])
        counts[split] = len(source)
        sources[split] = {
            "manifest_sha256": sha256_file(source.root / "bar-store.json"),
            "sample_universe": source.sample_universe_identity,
            "samples": len(source),
        }
        dataset = with_direction_targets(source, config)
        loader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=training.ModelBatchCollator.for_model(config, training=False),
            **training._loader_process_options(workers, persistent=False),
        )
        store = None
        if split != "train":
            store = SignalEvaluationStore(
                job / split,
                len(source),
                config.data.alpha_horizons,
                scales,
                task,
                beta=config.model.point_loss_beta,
                neutral_threshold=config.model.neutral_return_threshold,
            )
        try:
            for batch in loader:
                absolute = batch.get("target_absolute")
                if split == "train":
                    if absolute is not None:
                        append_label_audit(
                            class_counts,
                            absolute.numpy(),
                            batch["markets"],
                            config.model.neutral_return_threshold,
                            config.data.alpha_horizons,
                        )
                    continue
                target = batch["target_alpha"].numpy()
                prediction = np.zeros_like(target)
                if task == "absolute_direction":
                    prediction = np.full((*target.shape, 3), -30.0, dtype=np.float32)
                    prediction[..., 1] = 0.0
                store.append(
                    target,
                    prediction,
                    target_absolute=None if absolute is None else absolute.numpy(),
                    symbols=batch["symbols"],
                    dates=batch["cutoff_at"],
                    markets=batch["markets"],
                    asset_types=batch["asset_types"],
                    providers=batch["providers"],
                )
            if store is not None:
                scores[split] = store.finish()
        finally:
            if store is not None:
                store.close()
            source.close()
        if split != "train":
            canonical = root / "inputs" / split
            canonical.mkdir(parents=True)
            (job / split / "membership.npy").rename(canonical / "metadata.npy")
            for name in ("targets.npy", "absolute_targets.npy"):
                path = job / split / name
                if path.is_file():
                    path.rename(canonical / name)
    (job / "validation-metrics.json").write_text(json.dumps(scores["validation"]))

    def metadata(path):
        return path.relative_to(root).as_posix(), {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        artifacts = dict(pool.map(metadata, paths))
    payload = {
        "state": "complete",
        "identity": identity,
        "models": {
            "zero_return": {
                "state": "complete",
                "source": "prebuilt_full_data",
                "metrics": scores["test"],
            }
        },
        "sample_counts": counts,
        "robust_scales": scales,
        "data_identity": {"split_sources": sources},
        "evaluation_membership": scores["test"]["sample_membership"],
        "validation_membership": scores["validation"]["sample_membership"],
        "evaluation_data": baseline_contract.shared_evaluation_paths(),
        "artifacts": artifacts,
    }
    if task == "absolute_direction":
        payload["class_label_audit"] = {
            "prediction_task": task,
            "neutral_return_threshold": config.model.neutral_return_threshold,
            "samples": counts["train"],
            "counts": class_counts,
        }
    baseline_contract.validate_complete(payload, identity, require_shared=True)
    (root / "complete.json").write_text(json.dumps(payload))
    return payload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA")
@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_signal_benchmark_reuses_persisted_baseline_and_scores_true_checkpoint(
    tmp_path, market_frame, monkeypatch, task
):
    frame = market_frame.copy()
    # Intraday variation gives every market all three labels at every horizon.
    position = frame.groupby("symbol", sort=False).cumcount().to_numpy()
    frame["open"] = frame["close"] / np.exp(0.05 * np.sin(position * 0.71))
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.006
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.994
    raw, store = tmp_path / "market.parquet", tmp_path / "bar-store"
    frame.to_parquet(raw, index=False)
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
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(frame))}
        },
    )
    config = _config(tmp_path, raw, store)
    config.data.h_start = 1
    config.model = ModelConfig.model_validate(
        {
            **config.model.model_dump(),
            "prediction_task": task,
            "direction": {"enabled": False},
            "decoupled_output_scale": False,
        }
    )
    config.training.epochs = 1
    config.training.evaluations_per_epoch = 1
    config.training.checkpoint_monitor = "primary_5d/selection_score"
    config.validation.models = ["kronos_full", "zero_return"]
    config.validation.seeds = [42]
    _write_ready_manifest(config, frame, prepared)
    monkeypatch.setenv("WANDB_RUN_ID", f"qa-{task}-benchmark")
    monkeypatch.setenv("RUNPOD_RUN_KEY", f"qa-{task}-benchmark")
    completed = training.train(config)
    checkpoint = completed.final_checkpoint
    scales = json.loads((checkpoint / "trainer-state.json").read_text())["runtime_robust_scales"]
    signal_task = {"prediction_task": task}
    parameter = "point_loss_beta" if task == "alpha_point" else "neutral_return_threshold"
    signal_task[parameter] = getattr(config.model, parameter)
    identity = {
        "baseline_id": f"baseline-fixture-{task}",
        "contract": {
            "parameters": {"models": ["zero_return"], "seeds": [42], "signal_task": signal_task},
            "data": {"preparation": {"h_start": 1}},
        },
    }
    # Production prebuilt evaluation requires an explicit shared-source selection.
    # Copy this tiny immutable fixture byte-for-byte so checkpoint selection and
    # holdout still see exactly the same bars, without weakening the source gate.
    manifest = json.loads(config.data.resolved_manifest_path.read_text())
    selection = {
        "dataset_request": {
            "profile": config.data.dataset_profile,
            "revision": "signal-benchmark-fixture",
            "selected_datasets": config.data.selected_datasets,
            "date_range": manifest["date_range"],
            "universe": manifest["symbols"],
            "preparation": manifest["preparation_provenance"],
        }
    }
    shared = (
        tmp_path
        / "datasets"
        / evaluation_dataset_id(selection, load_data_policy())
        / "prepared/bar-store"
    )
    shutil.copytree(store, shared)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setenv("NETWORK_VOLUME_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNPOD_SELECTION_FILE", str(selection_path))
    baseline_root = tmp_path / "baselines" / identity["baseline_id"]
    cached = _persist_zero_baseline(config, baseline_root, identity, scales)
    monkeypatch.setattr(baseline_contract, "runtime_contract", lambda: (tmp_path, identity))
    # The completed-artifact reader and full evaluator remain real production calls.
    config.validation.require_prebuilt_baselines = True
    assert baseline_contract.require_baselines(config) == cached

    def forbidden(*args, **kwargs):
        raise AssertionError("Final holdout attempted to rebuild or refit a baseline")

    for name in ("rule_baseline_suite", "fit_causal_gru", "fit_dlinear", "fit_patchtst"):
        monkeypatch.setattr(validation_benchmark, name, forbidden)
    monkeypatch.setattr(validation_benchmark.GradientBoostingBaseline, "fit", forbidden)
    monkeypatch.setattr("stock_forecasting.baseline_build.build_baselines", forbidden)
    monkeypatch.setattr("stock_forecasting.direction_costs.direction_cost_band", forbidden)
    before = {
        path: (path.stat().st_mtime_ns, sha256_file(path))
        for path in baseline_root.rglob("*")
        if path.is_file()
    }
    run_id = checkpoint.parent.name
    output = config.validation.output_root / run_id / "validation-benchmark.json"
    lifecycle = tmp_path / "lifecycle/runs" / run_id / "validation.json"

    def runner():
        return validation_benchmark.ValidationBenchmark(
            config,
            run_id=run_id,
            checkpoint=checkpoint,
            output=output,
            lifecycle=lifecycle,
            models=config.validation.models,
            seeds=config.validation.seeds,
            resume=True,
            recompute_full_model=True,
        )

    result = runner().run()
    assert result["state"] == "ready" and result["evaluation_split"] == "test"
    assert result["baseline_reused"] and result["test_unlocked"]
    assert result["models"]["zero_return"] == cached["models"]["zero_return"]
    full = result["models"]["kronos_full"]["metrics"]
    assert full["prediction_task"] == task
    assert full["sample_membership"] == cached["evaluation_membership"]
    assert full["evaluation_robust_scales"] == cached["robust_scales"]
    assert full["samples"] == cached["sample_counts"]["test"]
    assert full["execution"]["full_population"] is True
    assert result["comparison"]["prediction_task"] == task
    assert "zero_return" in result["comparison"]["full_minus_baseline_date_paired"]
    assert "pinball" not in result["protocol"]["model_selection"]
    assert "coverage" not in full["primary_5d"]
    assert json.loads(lifecycle.read_text())["validation_completed"] is True
    assert {path: (path.stat().st_mtime_ns, sha256_file(path)) for path in before} == before

    monkeypatch.setattr(validation_benchmark, "evaluate_checkpoint", forbidden)
    resumed = runner().run()
    assert resumed["models"] == result["models"]
    assert resumed["comparison"] == result["comparison"]

    # Reject stale provenance and missing canonical labels before any evaluation.
    malformed = copy.deepcopy(cached)
    malformed["data_identity"]["split_sources"]["test"]["sample_universe"] = "different"
    (baseline_root / "complete.json").write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="test population differs"):
        baseline_contract.require_baselines(config)
    (baseline_root / "complete.json").write_text(json.dumps(cached))
    canonical = baseline_root / "inputs/test/targets.npy"
    canonical.rename(canonical.with_suffix(".held"))
    with pytest.raises(ValueError, match="Missing or truncated baseline artifact"):
        baseline_contract.require_baselines(config)
    canonical.with_suffix(".held").rename(canonical)
    assert baseline_contract.require_baselines(config) == cached
