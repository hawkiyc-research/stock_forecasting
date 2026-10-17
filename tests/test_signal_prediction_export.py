"""Cloud numerical tests for read-only prediction export and historical checkpoint loading."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import SequentialSampler

import stock_forecasting.training as training
from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    TEMP_CHECKPOINT_POINTER,
    save_ranked_checkpoint,
)
from stock_forecasting.cli import export_signal_predictions as export
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.run_contract import (
    _canonical_payload_digest,
    readonly_checkpoint_contract_digest,
    training_resume_contract,
    training_resume_contract_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = 14


def _config(name, tmp_path):
    config = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{name}.yaml")
    runtime = config.runtime.model_copy(update={"log_root": tmp_path / "logs"})
    return config.model_copy(update={"runtime": runtime})


def _historical_manifest(contract):
    return {
        "training_resume_contract": contract,
        "training_resume_contract_sha256": _canonical_payload_digest(contract),
    }


@pytest.mark.parametrize("name", ["a_alpha_point64", "a_absolute_direction64"])
def test_readonly_loading_accepts_runs_saved_before_additive_training_fields(name, tmp_path):
    config = _config(name, tmp_path)
    stored = copy.deepcopy(training_resume_contract(config))
    # Simulate a manifest written before date_mixing_buffer_samples existed.
    del stored["training"]["date_mixing_buffer_samples"]
    manifest = _historical_manifest(stored)
    assert readonly_checkpoint_contract_digest(config, manifest) == _canonical_payload_digest(
        stored
    )
    changed = copy.deepcopy(stored)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="differ"):
        readonly_checkpoint_contract_digest(config, _historical_manifest(changed))


def _batches(task, samples, batch_size, generator):
    batches = []
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        close = torch.exp(torch.randn(size, 128, generator=generator) * 0.01 + 4)
        series = torch.stack([close] * 4 + [torch.ones_like(close)], dim=-1)
        batch = {
            "asset_series": series,
            "asset_attention_mask": None,
            "target_alpha": torch.randn(size, HORIZONS, generator=generator) * 0.05,
            "symbols": [f"S{start + i}" for i in range(size)],
            "cutoff_at": [f"2025-06-0{2 + (start + i) % 3}T00:00:00+00:00" for i in range(size)],
            "markets": ["US" if (start + i) % 2 else "TWSE" for i in range(size)],
            "asset_types": ["stock"] * size,
            "providers": ["fixture"] * size,
        }
        if task == "absolute_direction":
            batch["target_absolute"] = torch.randn(size, HORIZONS, generator=generator) * 0.05
        batches.append(batch)
    return batches


@pytest.mark.parametrize(
    ("name", "task", "tail"),
    [("a_alpha_point64", "alpha_point", (HORIZONS,)),
     ("a_absolute_direction64", "absolute_direction", (HORIZONS, 3))],
)
def test_export_split_streams_every_row_and_recomputes_metrics(
    name, task, tail, tmp_path, monkeypatch
):
    config = _config(name, tmp_path)
    samples, generator = 37, torch.Generator().manual_seed(7)
    batches = _batches(task, samples, 8, generator)
    outputs = [torch.randn(len(b["symbols"]), *tail, generator=generator) for b in batches]
    expected = torch.cat(outputs).numpy()
    produced = iter(outputs)

    def forward_batch(_bundle, _batch, _config, _device):
        value = next(produced)
        return SimpleNamespace(alpha_point=value, absolute_direction_logits=value)

    monkeypatch.setattr(training, "iter_device_batches", lambda loader, device: iter(batches))
    monkeypatch.setattr(training, "forward_batch", forward_batch)
    head = SimpleNamespace(robust_scales=torch.full((HORIZONS,), 0.05))
    bundle = SimpleNamespace(model=SimpleNamespace(alpha_head=head))
    loader = SimpleNamespace(
        sampler=SequentialSampler(range(samples)), dataset=range(samples), drop_last=False
    )
    summary = export.export_split(bundle, loader, config, torch.device("cpu"), tmp_path / "out")
    saved = np.load(tmp_path / "out" / "predictions.npy")
    np.testing.assert_array_equal(saved, expected.astype(np.float32))
    assert summary["shape"] == [samples, *tail]
    assert summary["samples"] == samples
    assert summary["sample_membership"]["samples"] == samples
    assert json.loads((tmp_path / "out" / "split.json").read_text()) == summary
    assert not (tmp_path / "out" / "predictions.partial.npy").exists()
    assert np.isfinite(summary["loss"])


class _TinyModel(nn.Module):
    def __init__(self, horizons):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.alpha_head = nn.Module()
        self.alpha_head.register_buffer("robust_scales", torch.ones(horizons))


def _ranked_run(root, run_id, steps):
    """Save validation-ranked checkpoints; values above the top five stay resume-only."""
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    run = root / run_id
    run.mkdir()
    contract, digest = training_resume_contract_fingerprint(config)
    (run / "run-manifest.json").write_text(json.dumps({
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract": contract,
        "training_resume_contract_sha256": digest,
    }))
    config.save_resolved(run / "resolved-config.yaml")
    model = _TinyModel(len(config.data.alpha_horizons))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    for step, value in steps:
        save_ranked_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, config=config,
            run_directory=run, global_step=step, epoch=0, batch_index=step,
            selection_metric_name="primary_5d/selection_score",
            selection_metric_value=value, selection_metric_mode="min", save_top_k=5,
            metrics={"primary_5d/selection_score": value},
        )
    return run


def test_selectors_resolve_only_the_best_or_leaderboard_checkpoints(tmp_path):
    run_id = "run-fixture-alpha"
    steps = [(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0), (60, 6.0)]
    run = _ranked_run(tmp_path, run_id, steps)
    # The newest checkpoint has weights on disk but is kept only for resume.
    resume_only = json.loads((run / TEMP_CHECKPOINT_POINTER).read_text())["path"]
    assert resume_only == "checkpoint-000060"
    assert (run / resume_only / "adapter.safetensors").is_file()

    assert export.resolve_selector(run_id, tmp_path) == run / "checkpoint-000010"
    named = export.resolve_selector(f"{run_id}/checkpoint-000030", tmp_path)
    assert named == run / "checkpoint-000030"
    entry = export.leaderboard_entry(named)
    assert (entry["rank"], entry["value"], entry["validation_best"]) == (
        3, 3.0, "checkpoint-000010"
    )
    with pytest.raises(ValueError, match="not retained"):
        export.resolve_selector(f"{run_id}/{resume_only}", tmp_path)
    with pytest.raises(FileNotFoundError):
        export.resolve_selector(f"{run_id}/checkpoint-000099", tmp_path)


@pytest.mark.parametrize(
    "value",
    ["", "run-x/", "run-x/checkpoint-12", "run-x/checkpoint-000010/extra", "../x",
     "/runpod-volume/savedModel/run-x", "run--x", "run-x/../checkpoint-000010"],
)
def test_selector_parsing_refuses_paths(value):
    with pytest.raises(ValueError):
        export.parse_selector(value)
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", value, "--splits", "validation"])


def test_export_refuses_partial_coverage_and_existing_output(tmp_path):
    config = _config("a_alpha_point64", tmp_path)
    bundle = SimpleNamespace(model=SimpleNamespace(alpha_head=None))
    shuffled = SimpleNamespace(sampler=range(4), dataset=range(4), drop_last=False)
    with pytest.raises(ValueError, match="sequential full coverage"):
        export.export_split(bundle, shuffled, config, torch.device("cpu"), tmp_path / "a")
    with pytest.raises(ValueError, match="signal tasks"):
        export.prediction_shape("quantiles", 4, HORIZONS)
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", "run-x-alpha", "--splits", "test"])
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", "run-x-alpha", "--checkpoint", "run-x-alpha"])
