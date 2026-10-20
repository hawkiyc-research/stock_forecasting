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
    scratch = tmp_path / "container-scratch"
    summary = export.export_split(
        bundle, loader, config, torch.device("cpu"), tmp_path / "out", scratch=scratch
    )
    saved = np.load(tmp_path / "out" / "predictions.npy")
    np.testing.assert_array_equal(saved, expected.astype(np.float32))
    assert summary["shape"] == [samples, *tail]
    assert summary["samples"] == samples
    assert summary["sample_membership"]["samples"] == samples
    assert summary["predictions_sha256"] == export.sha256_file(tmp_path / "out/predictions.npy")
    assert json.loads((tmp_path / "out" / "split.json").read_text()) == summary
    assert not (tmp_path / "out" / "predictions.partial.npy").exists()
    # Mapped arrays lived only in container scratch, which is removed afterwards.
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "predictions.npy", "split.json"
    ]
    assert list(scratch.iterdir()) == []
    assert not (tmp_path / "logs" / "evaluation-scratch").exists()
    assert np.isfinite(summary["loss"])
    if task == "alpha_point":
        by_market = summary["cross_sectional"]["by_market"]
        assert set(by_market) == {"US", "TWSE"}
        assert set(by_market["US"]) == {f"{h}d" for h in range(1, HORIZONS + 1)}
        assert by_market["US"]["1d"]["rank_ic_method"] == "spearman_average_ties_v1"
    else:
        assert summary["cross_sectional"] is None


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
        export.export_split(
            bundle, shuffled, config, torch.device("cpu"), tmp_path / "a", scratch=tmp_path
        )
    with pytest.raises(ValueError, match="signal tasks"):
        export.prediction_shape("quantiles", 4, HORIZONS)
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", "run-x-alpha", "--splits", "test"])
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", "run-x-alpha", "--checkpoint", "run-x-alpha"])


@pytest.mark.parametrize("value", ["2026-10-19", "20261332", "2026101", "today"])
def test_reuse_date_must_be_a_real_compact_utc_date(value):
    with pytest.raises(SystemExit):
        export.build_parser().parse_args(
            ["--checkpoint", "run-x-alpha", "--reuse-complete-since", value]
        )
    parsed = export.build_parser().parse_args(
        ["--checkpoint", "run-x-alpha", "--reuse-complete-since", "20261019"]
    )
    assert parsed.reuse_complete_since == "20261019"


def test_scratch_root_must_be_absolute_and_outside_the_network_volume(tmp_path, monkeypatch):
    volume = tmp_path / "runpod-volume"
    volume.mkdir()
    monkeypatch.delenv(export.SCRATCH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="absolute"):
        export.local_scratch_root(volume)
    monkeypatch.setenv(export.SCRATCH_ENV, "relative/scratch")
    with pytest.raises(RuntimeError, match="absolute"):
        export.local_scratch_root(volume)
    for inside in (volume, volume / "tmp" / "scratch"):
        monkeypatch.setenv(export.SCRATCH_ENV, str(inside))
        with pytest.raises(ValueError, match="outside the network volume"):
            export.local_scratch_root(volume)
    monkeypatch.setenv(export.SCRATCH_ENV, str(tmp_path / "container" / "scratch"))
    assert export.local_scratch_root(volume) == (tmp_path / "container" / "scratch").resolve()
    assert (tmp_path / "container" / "scratch").is_dir()


RUNTIME = {"python": "3.12.3", "torch": "2.9.1+cu128", "gpu": "RTX 5090", "mixed_precision": "bf16"}


def _checkpoint(root):
    checkpoint = root / "savedModel" / "run-x-alpha" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    for name in export.CHECKPOINT_FILES:
        (checkpoint / name).write_text(f"{name} contents")
    return checkpoint


def _export(directory, name, checkpoint, config, *, state="complete", runtime=RUNTIME,
            splits=("validation", "test"), forward=None):
    """Write a finished export directory the way run_export records it."""
    path = directory / name
    results = {}
    for split in splits:
        (path / split).mkdir(parents=True)
        predictions = path / split / "predictions.npy"
        np.save(predictions, np.arange(6, dtype=np.float32))
        results[split] = {"predictions_sha256": export.sha256_file(predictions)}
    (path / "status.json").write_text(json.dumps({"state": state}))
    (path / "export.json").write_text(json.dumps({
        **({"forward_package": forward} if forward else {}),
        "schema_version": export.EXPORT_SCHEMA_VERSION,
        "prediction_task": config.model.prediction_task,
        "runtime": runtime,
        "checkpoint": {
            "path": str(checkpoint),
            "model_architecture_sha256": config.model_architecture_digest(),
            "files_sha256": {
                name: export.sha256_file(checkpoint / name) for name in export.CHECKPOINT_FILES
            },
        },
        "splits": results,
    }))
    return path


def test_reuse_accepts_only_an_intact_matching_export_since_the_date(tmp_path):
    config = _config("a_alpha_point64", tmp_path)
    checkpoint = _checkpoint(tmp_path)
    directory = tmp_path / "exports"
    splits = ["validation", "test"]

    def reusable(since="20261019", runtime=RUNTIME):
        return export.reusable_export(directory, checkpoint, config, splits, since, runtime)

    directory.mkdir()
    assert reusable() is None
    older = _export(directory, "export-20261018T001629Z-aaaaaaaa", checkpoint, config)
    assert reusable() is None
    assert reusable(since="20261018") == older
    _export(directory, "export-20261019T113449Z-cccccccc", checkpoint, config, state="failed")
    _export(directory, "export-20261019T111000Z-dddddddd", checkpoint, config,
            runtime={**RUNTIME, "gpu": "NVIDIA H100 80GB HBM3"})
    _export(directory, "export-20261019T110000Z-eeeeeeee", checkpoint, config,
            splits=("validation",))
    assert reusable() is None
    good = _export(directory, "export-20261019T102900Z-bbbbbbbb", checkpoint, config)
    assert reusable() == good
    assert reusable(runtime={**RUNTIME, "torch": "2.10.0"}) is None
    # A prediction file changed after it was recorded is never reused.
    np.save(good / "test" / "predictions.npy", np.zeros(6, dtype=np.float32))
    assert reusable() is None
    # Neither is an export of different checkpoint weights.
    _export(directory, "export-20261019T103000Z-ffffffff", checkpoint, config)
    assert reusable() is not None
    (checkpoint / "adapter.safetensors").write_text("retrained")
    assert reusable() is None


def test_forward_exports_are_reused_only_for_the_same_package(tmp_path):
    config = _config("a_alpha_point64", tmp_path)
    checkpoint = _checkpoint(tmp_path)
    directory = tmp_path / "exports"
    splits = ["overlap", "forward"]
    package = {"package_sha256": "a" * 64, "end_exclusive": "2026-10-20", "path": "/p"}
    _export(directory, "export-20261021T000000Z-aaaaaaaa", checkpoint, config,
            splits=splits, forward=package)

    def reusable(forward):
        return export.reusable_export(
            directory, checkpoint, config, splits, "20261021", RUNTIME, forward
        )

    assert reusable(package) is not None
    assert reusable({**package, "path": "/elsewhere"}) is not None
    assert reusable({**package, "package_sha256": "b" * 64}) is None
    assert reusable(None) is None
    regular = tmp_path / "regular"
    _export(regular, "export-20261021T000000Z-bbbbbbbb", checkpoint, config, splits=splits)
    assert export.reusable_export(
        regular, checkpoint, config, splits, "20261021", RUNTIME, package
    ) is None


def test_forward_arguments_are_checked_before_any_gpu_work():
    with pytest.raises(SystemExit, match="require --forward-package"):
        export.main(["--checkpoint", "run-x-alpha", "--splits", "overlap", "--confirm-test"])
    with pytest.raises(SystemExit, match="only overlap/forward"):
        export.main(
            ["--checkpoint", "run-x-alpha", "--forward-package", "/x", "--splits", "validation"]
        )
    # Both populations by default; overlap repeats the test period.
    with pytest.raises(SystemExit, match="confirm-test"):
        export.main(["--checkpoint", "run-x-alpha", "--forward-package", "/x"])


def test_forward_package_must_be_sealed_and_on_the_volume(tmp_path, monkeypatch):
    import stock_forecasting.forward.package as package_module

    volume = tmp_path / "runpod-volume"
    inside = volume / "forward" / "2026-10-20" / "package"
    outside = tmp_path / "elsewhere" / "package"
    for path in (inside, outside):
        path.mkdir(parents=True)
    payload = {
        "package_sha256": "c" * 64,
        "end_exclusive": "2026-10-20",
        "populations": {"overlap": {"samples": 5}, "forward": {"samples": 3}},
    }
    monkeypatch.setattr(package_module, "load_package", lambda root, verify_files: payload)
    with pytest.raises(ValueError, match="network volume"):
        export.forward_record(outside, volume)
    record = export.forward_record(inside, volume)
    assert record["package_sha256"] == "c" * 64
    assert record["populations"] == {
        "overlap": {"slot": "validation", "samples": 5},
        "forward": {"slot": "test", "samples": 3},
    }
