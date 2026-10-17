"""Export per-sample predictions of an existing signal checkpoint (read-only diagnostic).

A selector is a child RUN_ID (its validation-selected best checkpoint) or
RUN_ID/checkpoint-NNNNNN, which must be retained by that run's validation leaderboard.
The exported row order is the complete sequential evaluation order, the same order as
the shared baseline evaluation arrays; each split records its membership digest so the
local analysis can verify that alignment. Metrics are recomputed with the unchanged
signal evaluation store and saved next to the predictions for comparison with the
recorded benchmark. Training, validation lifecycle and checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import SequentialSampler

from stock_forecasting.checkpointing import load_checkpoint
from stock_forecasting.cli.probe_scales import resolve_probe_checkpoint
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.signal import SIGNAL_TASKS
from stock_forecasting.preflight import run_preflight
from stock_forecasting.run_contract import readonly_training_settings
from stock_forecasting.run_paths import (
    CHECKPOINT_NAME_PATTERN,
    canonical_network_volume_root,
    validate_run_id,
    validate_training_output_root,
)
from stock_forecasting.signal_diagnostic_cloud import (
    audit_candidates,
    collect_audit,
    fixed_train_loader,
)
from stock_forecasting.signal_evaluation import SignalEvaluationStore
from stock_forecasting.signal_training import _past_drawdown
from stock_forecasting.training import build_evaluation_loader, set_global_seed

LOGGER = logging.getLogger(__name__)
EXPORT_SCHEMA_VERSION = "1.0"
SPLITS = ("train", "validation", "test")
CHECKPOINT_FILES = ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json")
# One diagnostic Pod ends with its job, so several checkpoints share one sequential job.
MAX_CHECKPOINTS = 16
# Scratch for the unchanged metric store plus a margin for logs and manifests.
DISK_MARGIN_BYTES = 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--checkpoint",
        metavar="RUN_ID[/checkpoint-NNNNNN]",
        action="append",
        required=True,
        help=(
            "Signal child run ID (its validation-selected best checkpoint) or "
            "RUN_ID/checkpoint-NNNNNN retained by its validation leaderboard; repeat for "
            f"up to {MAX_CHECKPOINTS} checkpoints exported sequentially in one job."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["validation"],
        help=(
            "Validation/test are complete; train is a fixed bounded uniform sample "
            "(default: validation)."
        ),
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required when exporting test, to make repeated test inference explicit.",
    )
    parser.add_argument("--train-samples", type=int, default=16384)
    parser.add_argument("--sample-seed", type=int, default=59)
    parser.add_argument(
        "--audit-samples",
        type=int,
        default=24,
        help="Maximum raw-label audit cases per split, 0-32",
    )
    parser.add_argument(
        "--audit-workers", type=int, default=2, help="Resource-budgeted raw-label I/O workers, 1-8"
    )
    parser.add_argument(
        "--max-export-gib",
        type=float,
        default=4,
        help="Bound total persisted artifacts across this job, at most 64 GiB",
    )
    return parser


def parse_selector(value: str) -> tuple[str, str | None]:
    """Split RUN_ID or RUN_ID/checkpoint-NNNNNN without accepting paths."""
    run_id, separator, name = value.partition("/")
    validate_run_id(run_id)
    if not separator:
        return run_id, None
    if CHECKPOINT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("Checkpoint selector must be RUN_ID or RUN_ID/checkpoint-NNNNNN")
    return run_id, name


def resolve_selector(value: str, saved_model_root: Path) -> Path:
    """Resolve the validation-selected best or a named checkpoint on the leaderboard."""
    run_id, name = parse_selector(value)
    if name is None:
        return resolve_probe_checkpoint(run_id, saved_model_root=saved_model_root)
    source = saved_model_root.resolve(strict=False) / run_id / name
    if not (source / "adapter.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint is not retained on the volume: {value}")
    # The exact-checkpoint path refuses names missing from the validation leaderboard.
    return resolve_probe_checkpoint(source, saved_model_root=saved_model_root)


def leaderboard_entry(checkpoint: Path) -> dict[str, Any]:
    """Record how validation ranked the exported checkpoint."""
    payload = json.loads((checkpoint.parent / "checkpoint-leaderboard.json").read_text())
    row = next(item for item in payload["checkpoints"] if item["path"] == checkpoint.name)
    return {
        "monitor": payload["monitor"],
        "mode": payload["mode"],
        "rank": row["rank"],
        "value": row["value"],
        "validation_best": payload["best_checkpoint"],
    }


def require_export_environment() -> None:
    if not os.environ.get("RUNPOD_POD_ID") or not torch.cuda.is_available():
        raise RuntimeError("Prediction export must run inside an existing CUDA RunPod Pod")
    if os.environ.get("RUNPOD_GPU_WORKFLOW_LEASE_HELD") != "1":
        raise RuntimeError(
            "Use bash scripts/runpod_tmux_launch.sh export-signal-predictions to verify the "
            "mount and GPU lease"
        )


def prediction_shape(task: str, samples: int, horizons: int) -> tuple[int, ...]:
    if task == "alpha_point":
        return (samples, horizons)
    if task == "absolute_direction":
        return (samples, horizons, 3)
    raise ValueError(f"Prediction export supports only signal tasks: {task}")


def require_disk(path: Path, shape: tuple[int, ...]) -> int:
    """Refuse an export whose float32 array cannot fit beside the metric scratch."""
    needed = int(np.prod(shape)) * 4
    free = shutil.disk_usage(path).free
    if free < needed + DISK_MARGIN_BYTES:
        raise OSError(f"Prediction export needs {needed} bytes plus 1 GiB free space at {path}")
    return needed


@torch.inference_mode()
def export_split(
    bundle,
    loader,
    config,
    device,
    output: Path,
    *,
    indices=None,
    audit_samples=24,
    audit_workers=2,
    byte_budget=4 * 1024**3,
) -> dict[str, Any]:
    """Stream predictions to a float32 array and recompute the unchanged signal metrics."""
    from stock_forecasting.training import _autocast_context, forward_batch, iter_device_batches

    if (
        not isinstance(loader.sampler, SequentialSampler)
        or len(loader.sampler) != len(loader.dataset)
        or loader.drop_last
    ):
        raise ValueError("Prediction export requires sequential full coverage without drop_last")
    if loader.num_workers < 1:
        raise ValueError("Cloud prediction diagnostics require resource-budgeted loader workers")
    task = config.model.prediction_task
    count = len(loader.dataset)
    shape = prediction_shape(task, count, len(config.data.alpha_horizons))
    persisted = int(np.prod(shape)) * 4
    if indices is not None:
        from stock_forecasting.evaluation_store import META_DTYPE

        persisted += (
            count
            * (
                META_DTYPE.itemsize
                + 4
                + (2 if task == "absolute_direction" else 1) * len(config.data.alpha_horizons) * 4
            )
            + count * 8
        )
    if persisted + audit_samples * 256 * 1024 > byte_budget:
        raise OSError("Requested prediction artifacts exceed --max-export-gib before inference")
    output.mkdir(parents=True, exist_ok=False)
    require_disk(output, shape)
    partial = output / "predictions.partial.npy"
    scratch = config.runtime.log_root / "evaluation-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    scales = bundle.model.alpha_head.robust_scales.float().cpu().tolist()
    started = time.perf_counter()
    arrays = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=shape)
    try:
        with TemporaryDirectory(prefix="signal-export-", dir=scratch) as directory:
            store = SignalEvaluationStore(
                Path(directory),
                count,
                config.data.alpha_horizons,
                scales,
                task,
                beta=config.model.point_loss_beta,
                neutral_threshold=config.model.neutral_return_threshold,
            )
            try:
                completed, last_log = 0, started
                for batch in iter_device_batches(loader, device):
                    with _autocast_context(config, device):
                        result = forward_batch(bundle, batch, config, device)
                    predictions = (
                        result.alpha_point
                        if task == "alpha_point"
                        else result.absolute_direction_logits
                    )
                    values = predictions.detach().float().cpu().numpy()
                    close = batch["asset_series"][..., 3].float()
                    absolute = batch.get("target_absolute")
                    store.append(
                        batch["target_alpha"].detach().float().cpu().numpy(),
                        values,
                        target_absolute=None
                        if absolute is None
                        else absolute.detach().float().cpu().numpy(),
                        past_drawdown=_past_drawdown(close, batch.get("asset_attention_mask"))
                        .cpu()
                        .numpy(),
                        symbols=batch["symbols"],
                        dates=batch["cutoff_at"],
                        markets=batch["markets"],
                        asset_types=batch["asset_types"],
                        providers=batch["providers"],
                    )
                    arrays[completed : completed + len(values)] = values
                    completed += len(values)
                    now = time.perf_counter()
                    if now - last_log >= 30:
                        LOGGER.info(
                            "Export progress %s/%s at %.1f samples/s",
                            completed,
                            count,
                            completed / (now - started),
                        )
                        last_log = now
                if completed != count:
                    raise RuntimeError("Prediction export did not cover the complete split")
                metrics = store.finish()
                cases = audit_candidates(store, audit_samples)
                audit, audit_plan = collect_audit(
                    loader, cases, indices, requested_workers=audit_workers
                )
                atomic_write_json(
                    output / "audit.json",
                    {
                        "selection": (
                            "high-loss drawdown>=40%; same-date/market control drawdown<20%"
                        ),
                        "requested_max_cases": audit_samples,
                        "worker_plan": audit_plan,
                        "cases": audit,
                        "external_vendor_actions_verified": False,
                    },
                )
                if indices is not None:
                    # Only the bounded train sample is self-contained. Full validation
                    # labels/metadata are reused from the existing shared baseline cache.
                    files = [
                        ("targets.npy", "targets.npy"),
                        ("membership.npy", "metadata.npy"),
                        ("past_drawdown.npy", "past_drawdown.npy"),
                    ]
                    if store.absolute_targets is not None:
                        files.append(("absolute_targets.npy", "absolute_targets.npy"))
                    for array in (
                        store.targets,
                        store.metadata,
                        store.drawdown,
                        store.absolute_targets,
                    ):
                        if array is not None:
                            array.flush()

                    def copy_file(pair):
                        shutil.copyfile(Path(directory) / pair[0], output / pair[1])

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        list(executor.map(copy_file, files))
                    np.save(output / "source_ordinals.npy", np.asarray(indices, dtype=np.int64))
            finally:
                store.close()
        arrays.flush()
    finally:
        del arrays
    final = output / "predictions.npy"
    os.replace(partial, final)
    names = ["predictions.npy", "audit.json"]
    if indices is not None:
        names += ["targets.npy", "metadata.npy", "past_drawdown.npy", "source_ordinals.npy"]
        if task == "absolute_direction":
            names.append("absolute_targets.npy")
    artifacts = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in names
    }
    summary = {
        "artifacts": artifacts,
        "artifact_bytes": sum(row["bytes"] for row in artifacts.values()),
        "population": "fixed_uniform_train_sample" if indices is not None else "complete_split",
        "model_mode": "eval",
        "execution": {
            "loader_workers": loader.num_workers,
            "prefetch_factor": loader.prefetch_factor,
            "pin_memory": loader.pin_memory,
            "source_population_samples": len(loader.dataset.dataset)
            if indices is not None
            else count,
        },
        "schema_version": EXPORT_SCHEMA_VERSION,
        "prediction_task": task,
        "samples": count,
        "shape": list(shape),
        "dtype": "float32",
        "values": "alpha_point" if task == "alpha_point" else "absolute_direction_logits",
        "predictions_sha256": sha256_file(final),
        "sample_membership": metrics.get("sample_membership"),
        "loss": metrics.get("loss"),
        "aggregate": metrics.get("aggregate"),
        "market_macro": metrics.get("market_macro"),
        # Alpha only: per-market/horizon Rank IC and IC-IR, comparable with metrics.jsonl.
        "cross_sectional": metrics.get("cross_sectional"),
        "seconds": time.perf_counter() - started,
    }
    atomic_write_json(output / "split.json", summary)
    return summary


def run_export(
    checkpoint: Path,
    selector: str,
    splits: list[str],
    *,
    train_samples=16384,
    sample_seed=59,
    audit_samples=24,
    audit_workers=2,
    byte_budget=4 * 1024**3,
) -> Path:
    """Run on an existing GPU Pod under the diagnostic lease, without lifecycle changes."""
    require_export_environment()
    volume = canonical_network_volume_root()
    raw_config = yaml.safe_load((checkpoint / "resolved-config.yaml").read_text())
    historical_order = raw_config["training"].get("date_mixing_buffer_samples", 0)
    raw_config["training"] = readonly_training_settings(raw_config["training"])
    config = ExperimentConfig.model_validate(raw_config)
    validate_training_output_root(config.training.output_root)
    if config.model.time_series_backend not in (
        "kronos",
        "chronos2",
    ) or config.model.prediction_task not in (SIGNAL_TASKS):
        raise ValueError(
            "Prediction export requires an independent Kronos/Chronos-2 signal checkpoint"
        )
    run_preflight(
        config,
        require_data=True,
        enforce_runtime_limit=False,
        allow_historical_inference=True,
    ).require_success()
    root = volume / "diagnostics" / "signal-predictions"
    output = (
        root
        / checkpoint.parent.name
        / checkpoint.name
        / f"export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    if output.resolve() != output:
        raise ValueError("Prediction export output must not traverse symlinked directories")
    log_handler = logging.FileHandler(output / "export.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(log_handler)
    atomic_write_json(output / "status.json", {"state": "running", "checkpoint": str(checkpoint)})
    try:
        set_global_seed(config.training.seed)
        device = torch.device("cuda")
        bundle = build_model_bundle(config, device)
        state = load_checkpoint(
            checkpoint,
            bundle.model,
            config=config,
            allow_historical_inference=True,
        )
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        results = {}
        used = 0
        for split in splits:
            indices = None
            if split == "train":
                loader, indices = fixed_train_loader(
                    config, bundle, device, train_samples, sample_seed
                )
            else:
                loader = build_evaluation_loader(config, bundle=bundle, device=device, split=split)
            LOGGER.info("Exporting %s predictions for %s", split, checkpoint)
            results[split] = export_split(
                bundle,
                loader,
                config,
                device,
                output / split,
                indices=indices,
                audit_samples=audit_samples,
                audit_workers=audit_workers,
                byte_budget=byte_budget - used,
            )
            used += results[split]["artifact_bytes"]
            del loader
        report = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "kind": "signal-prediction-export",
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": checkpoint.parent.name,
            "selector": selector,
            "checkpoint": {
                "path": str(checkpoint),
                "selection": "validation_best" if "/" not in selector else "leaderboard_named",
                "leaderboard": leaderboard_entry(checkpoint),
                "global_step": state.get("global_step"),
                "model_architecture_sha256": config.model_architecture_digest(),
                "training_resume_contract_sha256": state.get("training_resume_contract_sha256"),
                "files_sha256": {name: sha256_file(checkpoint / name) for name in CHECKPOINT_FILES},
                "run_manifest_sha256": sha256_file(checkpoint.parent / "run-manifest.json"),
            },
            "prediction_task": config.model.prediction_task,
            "horizons": list(config.data.alpha_horizons),
            "artifact_bytes": used,
            "fixed_train_sample_seed": sample_seed,
            "fixed_train_sample_policy": "uniform without replacement; not recency weighted",
            "original_yearly_sampling_decay": config.training.yearly_sampling_decay,
            "historical_date_mixing_buffer_samples": historical_order,
            "training_weights_updated": False,
            "splits": results,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "mixed_precision": config.training.mixed_precision,
            },
            "export_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        atomic_write_json(output / "export.json", report)
        atomic_write_json(
            output / "status.json", {"state": "complete", "checkpoint": str(checkpoint)}
        )
        LOGGER.info("Prediction export complete: %s", output)
        return output
    except BaseException as error:
        atomic_write_json(
            output / "status.json", {"state": "failed", "error_type": type(error).__name__}
        )
        LOGGER.exception("Prediction export failed; partial artifacts are not completed results")
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        # Release this checkpoint's model before the next one in the same job.
        bundle = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not 16 <= args.train_samples <= 50000
        or not 0 <= args.audit_samples <= 32
        or not 1 <= args.audit_workers <= 8
        or not 0.1 <= args.max_export_gib <= 64
    ):
        raise SystemExit("Invalid train/audit sample, audit worker or export disk bounds")
    selectors = list(dict.fromkeys(args.checkpoint))
    if len(selectors) != len(args.checkpoint) or len(selectors) > MAX_CHECKPOINTS:
        raise SystemExit(f"Pass 1-{MAX_CHECKPOINTS} distinct --checkpoint selectors")
    try:
        for selector in selectors:
            parse_selector(selector)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    splits = list(dict.fromkeys(args.splits))
    if "test" in splits and not args.confirm_test:
        raise SystemExit("--splits test requires --confirm-test")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require_export_environment()
    saved_model_root = canonical_network_volume_root() / "savedModel"
    # Resolve every selector before GPU work so a typo or pruned checkpoint fails fast.
    checkpoints = [resolve_selector(selector, saved_model_root) for selector in selectors]
    if len(set(checkpoints)) != len(checkpoints):
        raise SystemExit("Two --checkpoint selectors resolve to the same checkpoint")
    outputs = []
    used = 0
    for selector, checkpoint in zip(selectors, checkpoints, strict=True):
        output = run_export(
            checkpoint,
            selector,
            splits,
            train_samples=args.train_samples,
            sample_seed=args.sample_seed,
            audit_samples=args.audit_samples,
            audit_workers=args.audit_workers,
            byte_budget=int(args.max_export_gib * 1024**3) - used,
        )
        used += json.loads((output / "export.json").read_text())["artifact_bytes"]
        outputs.append(str(output))
        print(json.dumps({"state": "complete", "output": str(output)}, ensure_ascii=False))
    print(json.dumps({"state": "complete", "exports": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
