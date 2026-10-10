"""Opt-in real Kronos acceptance using existing, read-only cloud data/cache."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
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
from stock_forecasting.direction_data import DirectionDataset
from stock_forecasting.direction_runtime import calibrate_direction_references
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.forecast_optimization import SamplePlateauScheduler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("DIRECTION_QA_TRAIN_ROOT"),
    reason="Requires an authorized isolated CUDA Pod and existing cleaned data",
)
ROOT = Path(__file__).resolve().parents[1]


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


def test_real_kronos_direction_gradient_evaluation_and_roundtrip(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    training.set_global_seed(42)
    # The suite intentionally clears Pod environment variables. Restore only the
    # explicitly authorized read-only cache, not the Pod's credentials or runtime.
    cache = Path(os.environ["DIRECTION_QA_HF_HOME"])
    assert (cache / "hub").is_dir()
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HOME", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    train_root = Path(os.environ["DIRECTION_QA_TRAIN_ROOT"])
    train_index = Path(os.environ["DIRECTION_QA_TRAIN_INDEX"])
    evaluation_root = Path(os.environ["DIRECTION_QA_EVALUATION_ROOT"])
    evaluation_index = Path(os.environ["DIRECTION_QA_EVALUATION_INDEX"])
    config = ExperimentConfig.from_yaml(ROOT / "configs/experiments/a_direction64.yaml")
    config.model.kronos_source_root = ROOT / "src/stock_forecasting/_vendor/kronos"
    config.runtime.log_root = tmp_path / "logs"
    source = dataset(train_root, train_index, "train")
    workers = training.plan_dataloader_workers(2)
    assert workers.effective_workers > 0
    statistics = next(
        row
        for path in sorted((train_root / "training-cache/scale-features").glob("*.json"))
        if len((row := json.loads(path.read_text())).get("center", [])) == 20
    )
    scales = next(
        json.loads(path.read_text())["scales"]
        for path in sorted((train_root / "training-cache/robust-scales").glob("*.json"))
    )
    device = torch.device("cuda")
    bundle = build_model_bundle(
        config, device, robust_scales=scales, scale_feature_statistics=statistics
    )
    subset = Subset(source, [i * (len(source) - 1) // 15 for i in range(16)])
    loader = DataLoader(
        subset,
        batch_size=16,
        collate_fn=training.ModelBatchCollator.for_model(config, training=True),
        pin_memory=True,
        **training._loader_process_options(workers, persistent=False),
    )
    batch = training._move_batch_to_device(next(iter(loader)), device)
    backbone = bundle.model.backbone
    before_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state().clone()
    references = calibrate_direction_references(
        bundle,
        config,
        source,
        device,
        workers,
        SimpleNamespace(training_batch_size=32),
    )
    assert references["samples"] == 1024
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state())
    bundle.model.eval()
    adapter = backbone.technical_adapter
    with torch.no_grad(), training._autocast_context(config, device):
        with_channels = backbone(batch["asset_series"], timestamps=batch["asset_timestamps"])
        backbone.technical_adapter = None
        without_channels = backbone(batch["asset_series"], timestamps=batch["asset_timestamps"])
        backbone.technical_adapter = adapter
    torch.testing.assert_close(
        with_channels.last_hidden_state, without_channels.last_hidden_state, rtol=0, atol=0
    )
    runtime_batch_plan = None
    if os.environ.get("DIRECTION_QA_RUNTIME_PROBE") == "1":
        # Exercise the unchanged production autotuner with the new model and labels.
        before_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in bundle.model.named_parameters()
            if parameter.requires_grad
        }
        before_rng = torch.get_rng_state().clone()
        before_cuda_rng = torch.cuda.get_rng_state().clone()
        assert (source.root.parent / "training-cache/date-market").is_dir()
        original_open_memmap = np.lib.format.open_memmap

        def read_only_memmap(filename, mode="r+", *args, **kwargs):
            assert mode == "r", "Read-only QA must reuse the existing ranking index"
            return original_open_memmap(filename, mode, *args, **kwargs)

        with (
            patch("numpy.lib.format.open_memmap", side_effect=read_only_memmap),
            patch("stock_forecasting.date_market_sampler.atomic_write_json",
                  side_effect=AssertionError("QA must not rewrite a prepared cache")),
        ):
            plan = training.resolve_runtime_batch_plan(
                config,
                bundle=bundle,
                sample=source[0],
                device=device,
                dataset=source,
                worker_plan=workers,
                evaluation_dataset=dataset(evaluation_root, evaluation_index, "validation"),
            )
        assert plan.training_batch_size > 0 and plan.evaluation_batch_size > 0
        assert any(row.accepted for row in plan.training_probe)
        assert any(row.accepted for row in plan.evaluation_probe)
        for name, parameter in bundle.model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.detach().cpu(), before_parameters[name], rtol=0, atol=0
                )
        assert torch.equal(before_rng, torch.get_rng_state())
        assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state())
        assert bundle.model.direction_head.reference_state() == references
        runtime_batch_plan = asdict(plan)
        del before_parameters
    gradient_norms = {}
    for task in ("absolute", "alpha"):
        bundle.model.zero_grad(set_to_none=True)
        with training._autocast_context(config, device):
            output = training.forward_batch(bundle, batch, config, device)
        getattr(output, f"{task}_direction_loss").backward()
        gradient_norms[task] = {
            group: sum(
                float(p.grad.detach().float().norm())
                for name, p in bundle.model.named_parameters()
                if marker in name and p.grad is not None
            )
            for group, marker in (
                ("lora", "lora_"),
                ("technical", "technical_adapter"),
                ("queries", "direction_head.query"),
            )
        }
        assert all(value > 0 for value in gradient_norms[task].values()), gradient_norms
    groups, parameters = training._optimizer_parameter_groups(bundle, config)
    optimizer = torch.optim.AdamW(groups)
    scheduler = SamplePlateauScheduler(
        optimizer, 1, warmup_samples=16, decay_start=32, decay_end=64
    )
    optimizer.zero_grad(set_to_none=True)
    bundle.model.train()
    for _ in range(3):
        with training._autocast_context(config, device):
            output = training.forward_batch(bundle, batch, config, device)
        assert output.representation_loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        assert all(p.grad is None for p in backbone.tokenizer.parameters())
        assert all(p.grad is None for p in backbone.model.parameters() if not p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
        optimizer.step()
        scheduler.step(16)
        optimizer.zero_grad(set_to_none=True)
    cost_boundary_checks = {}
    if audit_path := os.environ.get("DIRECTION_QA_COST_AUDIT"):
        audit = json.loads(Path(audit_path).read_text())
        for split in ("train", "validation", "test"):
            boundary_source = dataset(
                train_root if split == "train" else evaluation_root,
                train_index if split == "train" else evaluation_index,
                split,
            )
            ordinals = [row["ordinal"] for row in audit["splits"][split]["examples"]][:16]
            assert ordinals, f"No actual cost-boundary cases in the {split} audit"
            boundary_subset = Subset(boundary_source, ordinals)
            boundary_subset.split = split
            boundary_loader = DataLoader(
                boundary_subset,
                batch_size=len(ordinals),
                collate_fn=training.ModelBatchCollator.for_model(config, training=True),
                pin_memory=True,
                **training._loader_process_options(workers, persistent=False),
            )
            boundary_batch = training._move_batch_to_device(next(iter(boundary_loader)), device)
            assert torch.isneginf(boundary_batch["direction_bands"][:, 0]).all()
            bundle.model.train()
            optimizer.zero_grad(set_to_none=True)
            with training._autocast_context(config, device):
                boundary_output = training.forward_batch(bundle, boundary_batch, config, device)
            assert torch.isfinite(boundary_output.loss)
            boundary_output.loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
            optimizer.zero_grad(set_to_none=True)
            metrics = training.evaluate_loader(bundle, boundary_loader, config, device)
            assert (
                metrics["direction"]["overall"]["absolute"]["aggregate"]["count"]
                == len(ordinals) * 14
            )
            cost_boundary_checks[split] = {
                "samples": len(ordinals),
                "finite_loss": float(boundary_output.loss.detach()),
                "finite_backward": True,
                "evaluation": metrics["direction"],
            }
            boundary_source.close()
    bundle.model.eval()
    with torch.inference_mode(), training._autocast_context(config, device):
        expected = training.forward_batch(bundle, batch, config, device)
    save_file(trainable_state_dict(bundle.model), tmp_path / "adapter.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "direction": references,
        },
        tmp_path / "resume.pt",
    )
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(0.1)
    _load_trainable_model_state(bundle.model, load_file(tmp_path / "adapter.safetensors"))
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    bundle.model.direction_head.restore_references(state["direction"])
    assert scheduler.processed_samples == 48
    with torch.inference_mode(), training._autocast_context(config, device):
        actual = training.forward_batch(bundle, batch, config, device)
    for name in (
        "alpha_quantiles",
        "ranking_scores",
        "absolute_direction_logits",
        "alpha_direction_logits",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    evaluation = {}
    calibration = None
    for split in ("validation", "test"):
        source = dataset(evaluation_root, evaluation_index, split)
        subset = Subset(source, [i * (len(source) - 1) // 255 for i in range(256)])
        subset.split = split
        loader = DataLoader(
            subset,
            batch_size=32,
            pin_memory=True,
            collate_fn=training.ModelBatchCollator.for_model(config, training=False),
            **training._loader_process_options(workers, persistent=False),
        )
        evaluation[split] = training.evaluate_loader(
            bundle,
            loader,
            config,
            device,
            fit_interval_calibration=split == "validation",
            interval_calibration=calibration,
        )
        if split == "validation":
            calibration = evaluation[split]["interval_calibration"]
        assert (
            evaluation[split]["direction"]["overall"]["absolute"]["aggregate"]["count"] == 256 * 14
        )
        assert evaluation[split]["direction"]["loss_reference"] == references
    result = {
        "gpu": torch.cuda.get_device_name(),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "workers": workers.effective_workers,
        "loss_references": references,
        "ce_only_gradient_norms": gradient_norms,
        "exact_zero_adapter_parity": True,
        "exact_prediction_roundtrip": True,
        "optimizer_steps": 3,
        "evaluation_windows_per_split": 256,
        "runtime_batch_plan": runtime_batch_plan,
        "cost_boundary_checks": cost_boundary_checks,
        "evaluation": evaluation,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    destination = Path(os.environ.get("DIRECTION_QA_REPORT", tmp_path / "result.json"))
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
