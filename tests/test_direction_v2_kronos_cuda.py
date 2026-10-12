"""Opt-in V2 acceptance against read-only existing bars and real Kronos CUDA."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from stock_forecasting import training
from stock_forecasting.checkpointing import _load_trainable_model_state, trainable_state_dict
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import canonical_json_sha256, sha256_file
from stock_forecasting.direction_data import DirectionDataset
from stock_forecasting.direction_v2_runtime import calibrate_direction_references
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.scale_features import (
    EXTENDED_FEATURE_VERSION,
    validate_scale_feature_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("DIRECTION_QA_TRAIN_ROOT"),
    reason="Requires an authorized isolated CUDA Pod and existing cleaned data",
)


def dataset(root, index, split):
    return DirectionDataset(
        LazyFinancialWindowDataset(
            root / "prepared/bar-store",
            split=split,
            window_size=128,
            h_start=1,
            sample_index_root=index,
            symbol_cache_size=4,
        )
    )


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_real_kronos_v2_seed_reference_pcgrad_probe_and_checkpoint(tmp_path, monkeypatch, variant):
    torch.set_num_threads(2)
    cache = Path(os.environ["DIRECTION_QA_HF_HOME"])
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HOME", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    train_root = Path(os.environ["DIRECTION_QA_TRAIN_ROOT"])
    evaluation_root = Path(os.environ["DIRECTION_QA_EVALUATION_ROOT"])
    source = dataset(train_root, Path(os.environ["DIRECTION_QA_TRAIN_INDEX"]), "train")
    evaluation = dataset(
        evaluation_root, Path(os.environ["DIRECTION_QA_EVALUATION_INDEX"]), "validation"
    )
    config = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/a_direction64_{variant}.yaml")
    calibrate = calibrate_direction_references
    if variant == "v3":
        from stock_forecasting.direction_ordinal_runtime import (
            calibrate_direction_references as calibrate,
        )
    config.model.kronos_source_root = ROOT / "src/stock_forecasting/_vendor/kronos"
    config.runtime.log_root = tmp_path / "logs"
    config.model.direction.gradient_diagnostics_interval = 1
    config.model.direction.loss_reference_samples = 256
    config.model.lora.dropout = 0.0
    workers = training.plan_dataloader_workers(2)
    assert workers.effective_workers > 0
    calibration_count = min(config.data.label_scale_calibration_samples, len(source))
    robust_identity = training._robust_scale_identity(
        source,
        sample_count=calibration_count,
        seed=config.data.calibration_seed,
    )
    robust_digest = canonical_json_sha256(robust_identity)
    robust_path = train_root / "training-cache/robust-scales" / f"{robust_digest}.json"
    assert robust_path.is_file(), "Read-only QA requires the matching prepared robust-scale cache"
    scales = training._load_cached_robust_scales(
        robust_path,
        identity=robust_identity,
        identity_sha256=robust_digest,
    )
    statistics_identity = {
        "version": EXTENDED_FEATURE_VERSION,
        "bar_store_manifest_sha256": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "window_size": source.window_size,
        "split": "train",
        "sample_count": calibration_count,
        "seed": config.data.calibration_seed,
        "sampler": "blockwise-16-v1",
    }
    statistics_path = (
        train_root
        / "training-cache/scale-features"
        / f"{canonical_json_sha256(statistics_identity)}.json"
    )
    assert statistics_path.is_file(), (
        "Read-only QA requires the matching prepared scale-feature cache"
    )
    statistics = validate_scale_feature_statistics(json.loads(statistics_path.read_text()))
    assert statistics["identity"] == statistics_identity
    device = torch.device("cuda")
    training.set_global_seed(42)
    bundle = build_model_bundle(
        config, device, robust_scales=scales, scale_feature_statistics=statistics
    )
    reference_states = []
    for seed in (42, 43, 44):
        config.training.seed = seed
        training.set_global_seed(seed)
        before_rng, before_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        reference_states.append(
            calibrate(
                bundle, config, source, device, workers, SimpleNamespace(training_batch_size=32)
            )
        )
        assert torch.equal(before_rng, torch.get_rng_state())
        assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state())
    assert reference_states[0] == reference_states[1] == reference_states[2]
    config.training.seed = 42
    training.set_global_seed(42)
    subset = Subset(source, [i * (len(source) - 1) // 15 for i in range(16)])
    loader = DataLoader(
        subset,
        batch_size=16,
        collate_fn=training.ModelBatchCollator.for_model(config, training=True),
        pin_memory=True,
        **training._loader_process_options(workers, persistent=False),
    )
    iterator = iter(loader)
    try:
        batch = training._move_batch_to_device(next(iterator), device)
    finally:
        iterator._shutdown_workers()
    groups, parameters = training._optimizer_parameter_groups(bundle, config)
    optimizer = torch.optim.AdamW(groups, fused=True)
    manager = training._shared_task_gradient_manager(bundle, config)
    bundle.model._task_gradient_manager = manager
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    diagnostics = []
    state_gate_gradients = []
    bundle.model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with training._autocast_context(config, device):
            output = training.forward_batch(bundle, batch, config, device)
        training._backward_training_output(output, bundle, 1, manager)
        assert torch.isfinite(output.loss)
        assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
        assert any(
            p.grad is not None and p.grad.norm() > 0
            for name, p in bundle.model.named_parameters()
            if "lora_" in name
        )
        assert any(
            p.grad is not None and p.grad.norm() > 0
            for name, p in bundle.model.named_parameters()
            if "direction_head.numeric_projection" in name
        )
        if variant == "v3":
            gate_gradient = bundle.model.direction_head.state_gate[-1].weight.grad
            assert gate_gradient is not None and gate_gradient.norm() > 0
            state_gate_gradients.append(float(gate_gradient.norm()))
        optimizer.step()
        diagnostics.append(manager.optimizer_step_completed())
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated()
    evaluation_subset = Subset(evaluation, list(range(32)))
    evaluation_subset.split = "validation"
    evaluation_loader = DataLoader(
        evaluation_subset,
        batch_size=16,
        collate_fn=training.ModelBatchCollator.for_model(config, training=False),
        pin_memory=True,
        **training._loader_process_options(workers, persistent=False),
    )
    metrics = training.evaluate_loader(bundle, evaluation_loader, config, device)
    training._freeze_direction_validation_anchor(bundle, metrics)
    anchor = bundle.model.direction_head.pinball_guard_reference
    again = training.evaluate_loader(bundle, evaluation_loader, config, device)
    training._freeze_direction_validation_anchor(bundle, again)
    assert bundle.model.direction_head.pinball_guard_reference == anchor
    assert metrics["direction"]["samples"] == 32
    assert again["direction"]["overall"] == metrics["direction"]["overall"]
    assert again["direction"]["balanced_decision"] == metrics["direction"]["balanced_decision"]
    save_file(trainable_state_dict(bundle.model), tmp_path / "adapter.safetensors")
    references = bundle.model.direction_head.reference_state()
    state = manager.state_dict()
    bundle.model.eval()
    with torch.no_grad(), training._autocast_context(config, device):
        expected = training.forward_batch(bundle, batch, config, device)
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(0.1)
    _load_trainable_model_state(bundle.model, load_file(tmp_path / "adapter.safetensors"))
    bundle.model.direction_head.restore_references(references)
    manager.load_state_dict(state)
    with torch.no_grad(), training._autocast_context(config, device):
        actual = training.forward_batch(bundle, batch, config, device)
    for name in (
        "alpha_quantiles",
        "ranking_scores",
        "absolute_direction_logits",
        "alpha_direction_logits",
        "absolute_direction_decision_logits",
        "alpha_direction_decision_logits",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    if os.environ.get("DIRECTION_QA_RUNTIME_PROBE") == "1":
        # Existing caches are read-only; probes must never build a new source index.
        original_open = np.lib.format.open_memmap

        def read_only(filename, mode="r+", *args, **kwargs):
            assert mode == "r"
            return original_open(filename, mode, *args, **kwargs)

        with (
            patch("numpy.lib.format.open_memmap", side_effect=read_only),
            patch(
                "stock_forecasting.date_market_sampler.atomic_write_json",
                side_effect=AssertionError("QA cannot rewrite prepared caches"),
            ),
        ):
            measurement = training._measure_cuda_batch(
                bundle=bundle,
                sample=source[0],
                batch_size=16,
                config=config,
                device=device,
                training=True,
                optimizer_state_reserve_bytes=0,
                device_memory_limit_bytes=int(
                    torch.cuda.get_device_properties(0).total_memory * 0.72
                ),
                dataset=source,
                worker_plan=workers,
            )
        assert measurement.accepted
        assert bundle.model.direction_head.reference_state() == references
        assert manager.state_dict() == state
    record = {
        "architecture": config.model.direction.architecture,
        "state_gate_gradient_norms": state_gate_gradients,
        "seed_reference_equal": True,
        "samples": reference_states[0]["samples"],
        "pcgrad_updates": 3,
        "gradient_diagnostics": diagnostics,
        "train_elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak,
        "gpu": torch.cuda.get_device_name(),
        "validation": again["direction"],
        "roundtrip_exact": True,
        "existing_caches_read_only": True,
    }
    (tmp_path / "v2-kronos-evidence.json").write_text(json.dumps(record, allow_nan=False))
    print(json.dumps(record, allow_nan=False))
    source.close()
    evaluation.close()
