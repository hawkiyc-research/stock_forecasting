"""Opt-in cloud acceptance using real Kronos and existing read-only A/B data.

SIGNAL_QA_DATA_ROOT identifies an existing dataset identity directory, not a new
preparation destination. SIGNAL_QA_INDEX and SIGNAL_QA_HF_HOME must already
exist. SIGNAL_QA_GROUP selects the corresponding a/b config (default b). No
sample indexes, feature statistics, or robust scales are rebuilt.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from stock_forecasting import training
from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_trainer_state,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import canonical_json_sha256, sha256_file
from stock_forecasting.direction_data import with_direction_targets
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.scale_features import (
    EXTENDED_FEATURE_VERSION,
    validate_scale_feature_statistics,
)
from stock_forecasting.run_contract import training_resume_contract_fingerprint

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not os.environ.get("SIGNAL_QA_DATA_ROOT") or not torch.cuda.is_available(),
    reason="Requires an authorized isolated CUDA Pod and existing signal QA paths",
)


def _existing_paths(monkeypatch):
    root = Path(os.environ["SIGNAL_QA_DATA_ROOT"]).resolve(strict=True)
    index = Path(os.environ["SIGNAL_QA_INDEX"]).resolve(strict=True)
    cache = Path(os.environ["SIGNAL_QA_HF_HOME"]).resolve(strict=True)
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HOME", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return root, index


def _cached_train_statistics(root, source, config):
    count = min(config.data.label_scale_calibration_samples, len(source))
    identity = training._robust_scale_identity(
        source, sample_count=count, seed=config.data.calibration_seed
    )
    digest = canonical_json_sha256(identity)
    robust_path = root / "training-cache/robust-scales" / f"{digest}.json"
    assert robust_path.is_file(), "QA requires existing matching train robust scales"
    scales = training._load_cached_robust_scales(
        robust_path, identity=identity, identity_sha256=digest
    )
    assert scales is not None
    statistics_identity = {
        "version": EXTENDED_FEATURE_VERSION,
        "bar_store_manifest_sha256": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "window_size": source.window_size,
        "split": "train",
        "sample_count": count,
        "seed": config.data.calibration_seed,
        "sampler": "blockwise-16-v1",
    }
    statistics_path = (
        root
        / "training-cache/scale-features"
        / f"{canonical_json_sha256(statistics_identity)}.json"
    )
    assert statistics_path.is_file(), "QA requires existing matching train feature statistics"
    statistics = validate_scale_feature_statistics(json.loads(statistics_path.read_text()))
    assert statistics["identity"] == statistics_identity
    return scales, statistics, (robust_path, statistics_path)


def _fingerprints(paths):
    return {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in paths}


def _prediction(output, task):
    return output.alpha_point if task == "alpha_point" else output.absolute_direction_logits


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_real_kronos_signal_gradients_full_subset_evaluation_and_checkpoint(
    tmp_path,
    monkeypatch,
    task,
):
    root, index = _existing_paths(monkeypatch)
    group = os.environ.get("SIGNAL_QA_GROUP", "b")
    if group not in ("a", "b"):
        raise ValueError("SIGNAL_QA_GROUP must be a or b")
    config = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{group}_{task}64.yaml")
    config.model.kronos_source_root = Path(
        os.environ.get("SIGNAL_QA_KRONOS_ROOT", str(ROOT / "src/stock_forecasting/_vendor/kronos"))
    )
    config.runtime.log_root = tmp_path / "logs"
    config.model.lora.dropout = 0.0
    assert config.model.prediction_task == task
    assert config.model.lora.enabled and config.model.lora.rank == 64
    assert config.model.feature_mode == "combined" and config.model.local_files_only
    assert not config.model.direction.enabled and not config.model.ranking_loss_weight
    source = with_direction_targets(
        LazyFinancialWindowDataset(
            root / "prepared/bar-store",
            split="train",
            window_size=128,
            h_start=1,
            sample_index_root=index,
            symbol_cache_size=4,
        ),
        config,
    )
    evaluation = with_direction_targets(
        LazyFinancialWindowDataset(
            root / "prepared/bar-store",
            split="validation",
            window_size=128,
            h_start=1,
            sample_index_root=index,
            symbol_cache_size=4,
        ),
        config,
    )
    original_threads = torch.get_num_threads()
    bundle = optimizer = scheduler = batch = output = None
    try:
        torch.set_num_threads(2)
        scales, statistics, cache_paths = _cached_train_statistics(root, source, config)
        readonly_paths = (
            *cache_paths,
            source.root / "bar-store.json",
            index / "universe.json",
            index / "cutoff-ranges.parquet",
        )
        before = _fingerprints(readonly_paths)
        workers = training.plan_dataloader_workers(2)
        assert workers.effective_workers > 0
        device = torch.device("cuda")
        training.set_global_seed(42)
        bundle = build_model_bundle(
            config, device, robust_scales=scales, scale_feature_statistics=statistics
        )
        assert bundle.lora_parameter_names and bundle.model.direction_head is None
        count = min(16, len(source))
        assert count > 1
        indexes = [i * (len(source) - 1) // (count - 1) for i in range(count)]
        loader = DataLoader(
            Subset(source, indexes),
            batch_size=count,
            collate_fn=training.ModelBatchCollator.for_model(config, training=True),
            pin_memory=True,
            **training._loader_process_options(workers, persistent=False),
        )
        iterator = iter(loader)
        try:
            batch = training._move_batch_to_device(next(iterator), device)
        finally:
            iterator._shutdown_workers()
        assert "direction_bands" not in batch
        assert ("target_absolute" in batch) == (task == "absolute_direction")
        groups, parameters = training._optimizer_parameter_groups(bundle, config)
        optimizer = torch.optim.AdamW(groups, fused=True)
        scheduler = LambdaLR(optimizer, lambda _step: 1.0)
        assert training._shared_task_gradient_manager(bundle, config) is None
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        gradient_norms = []
        bundle.model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with training._autocast_context(config, device):
                output = training.forward_batch(bundle, batch, config, device)
            output.loss.backward()
            assert torch.isfinite(output.loss)
            assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
            lora_norm = sum(
                float(p.grad.float().norm())
                for name, p in bundle.model.named_parameters()
                if "lora_" in name and p.grad is not None
            )
            assert lora_norm > 0
            gradient_norms.append(lora_norm)
            assert any(
                p.grad is not None and p.grad.norm() > 0
                for name, p in bundle.model.named_parameters()
                if name.startswith("resampler.")
            )
            assert all(p.grad is None for p in bundle.model.parameters() if not p.requires_grad)
            optimizer.step()
            scheduler.step()
        assert bundle.model.alpha_head.numeric_branch.fusion[-1].weight.grad.norm() > 0
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed, peak = time.monotonic() - started, torch.cuda.max_memory_allocated()
        evaluation_count = min(32, len(evaluation))
        subset = Subset(evaluation, list(range(evaluation_count)))
        subset.split = "validation"
        evaluation_loader = DataLoader(
            subset,
            batch_size=16,
            collate_fn=training.ModelBatchCollator.for_model(config, training=False),
            pin_memory=True,
            **training._loader_process_options(workers, persistent=False),
        )
        metrics = training.evaluate_loader(bundle, evaluation_loader, config, device)
        assert metrics["prediction_task"] == task
        assert metrics["execution"]["completed_samples"] == evaluation_count
        assert metrics["sample_membership"]["samples"] == evaluation_count
        assert metrics["execution"]["workers"] > 0
        assert "direction" not in metrics and "coverage" not in metrics["aggregate"]
        bundle.model.eval()
        with torch.no_grad(), training._autocast_context(config, device):
            expected = _prediction(
                training.forward_batch(bundle, batch, config, device), task
            ).clone()
        run = tmp_path / f"qa-real-{task}"
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
            model=bundle.model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            run_directory=run,
            global_step=3,
            epoch=0,
            batch_index=3,
            selection_metric_name=config.training.checkpoint_monitor,
            selection_metric_value=metrics["aggregate"]["selection_score"],
            selection_metric_mode="min",
        )
        state = validate_checkpoint_trainer_state(checkpoint)
        assert state["prediction_task"] == task
        assert state["model_output_schema_version"] == "signal-1.0"
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(0.1)
        load_checkpoint(checkpoint, bundle.model, optimizer, scheduler, config=config)
        with torch.no_grad(), training._autocast_context(config, device):
            actual = _prediction(training.forward_batch(bundle, batch, config, device), task)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert before == _fingerprints(readonly_paths)
        record = {
            "prediction_task": task,
            "experiment_group": group,
            "source_root": str(source.root),
            "sample_universe": source.sample_universe_identity,
            "lora_rank": 64,
            "optimizer_steps": 3,
            "lora_gradient_norms": gradient_norms,
            "train_seconds": elapsed,
            "peak_cuda_bytes": peak,
            "gpu": torch.cuda.get_device_name(),
            "validation_subset_samples": evaluation_count,
            "validation_selection_score": metrics["aggregate"]["selection_score"],
            "checkpoint_roundtrip_exact": True,
            "source_caches_unchanged": True,
        }
        (tmp_path / f"{task}-kronos-evidence.json").write_text(json.dumps(record, allow_nan=False))
        print(json.dumps(record, allow_nan=False))
    finally:
        source.close()
        evaluation.close()
        torch.set_num_threads(original_threads)
        del bundle, optimizer, scheduler, batch, output
        gc.collect()
        torch.cuda.empty_cache()
