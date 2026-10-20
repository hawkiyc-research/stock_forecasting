"""Export per-sample predictions of an existing signal checkpoint (read-only diagnostic).

A selector is a child RUN_ID (its validation-selected best checkpoint) or
RUN_ID/checkpoint-NNNNNN, which must be retained by that run's validation leaderboard.
The exported row order is the complete sequential evaluation order, the same order as
the shared baseline evaluation arrays; each split records its membership digest so the
local analysis can verify that alignment. Metrics are recomputed with the unchanged
signal evaluation store and saved next to the predictions for comparison with the
recorded benchmark. Training, validation lifecycle and checkpoints are never modified.

Memory-mapped arrays live on container-disk scratch, never on the network volume: a volume
I/O error inside a mapping kills the process with SIGBUS, while the ordinary writes used to
publish the finished files raise a normal error that marks the export failed.

With --forward-package the same loader and metric store run on the overlap and forward
populations of a sealed forward input package instead of the configured dataset. No
dataset readiness is required; results go to diagnostics/forward-predictions/<end>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from stock_forecasting.checkpointing import load_checkpoint
from stock_forecasting.cli.probe_scales import resolve_probe_checkpoint
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.signal import SIGNAL_TASKS
from stock_forecasting.preflight import run_preflight
from stock_forecasting.run_paths import (
    CHECKPOINT_NAME_PATTERN,
    canonical_network_volume_root,
    validate_run_id,
    validate_training_output_root,
)
from stock_forecasting.signal_evaluation import SignalEvaluationStore
from stock_forecasting.signal_training import _past_drawdown
from stock_forecasting.training import build_evaluation_loader, set_global_seed

LOGGER = logging.getLogger(__name__)
EXPORT_SCHEMA_VERSION = "1.0"
SPLITS = ("validation", "test")
# Forward populations; overlap repeats the test period and therefore needs --confirm-test.
FORWARD_POPULATIONS = ("overlap", "forward")
CHECKPOINT_FILES = ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json")
# One diagnostic Pod ends with its job, so several checkpoints share one sequential job.
MAX_CHECKPOINTS = 16
# Scratch for the unchanged metric store plus a margin for logs and manifests.
DISK_MARGIN_BYTES = 1024**3
# Container-disk directory for memory-mapped scratch; set by the RunPod launcher.
SCRATCH_ENV = "SIGNAL_EXPORT_SCRATCH_ROOT"
COPY_CHUNK_BYTES = 16 * 1024**2


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
        choices=(*SPLITS, *FORWARD_POPULATIONS),
        help=(
            "Complete evaluation splits to export (default: validation test); with "
            "--forward-package the populations overlap and/or forward (default: both)."
        ),
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help=(
            "Required when exporting test or overlap, to make repeated test-period "
            "inference explicit."
        ),
    )
    parser.add_argument(
        "--forward-package",
        type=Path,
        metavar="PATH",
        help="Sealed forward input package on the network volume (forward pipeline).",
    )
    parser.add_argument(
        "--reuse-complete-since",
        metavar="YYYYMMDD",
        type=since_date,
        help=(
            "Skip a checkpoint whose complete export was created on or after this UTC date "
            "with the same checkpoint files, requested splits, GPU, torch, Python and "
            "precision, after re-verifying its prediction SHA-256 (resumes a crashed job)."
        ),
    )
    return parser


def since_date(value: str) -> str:
    """Accept only a real YYYYMMDD date, the prefix format of export directory names."""
    try:
        if re.fullmatch(r"[0-9]{8}", value) is None:
            raise ValueError
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError("expected a UTC date as YYYYMMDD") from None
    return value


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


def local_scratch_root(volume: Path) -> Path:
    """Return the container-disk scratch root; mapped writes must not touch the volume."""
    value = os.environ.get(SCRATCH_ENV, "")
    if not value or not Path(value).is_absolute():
        raise RuntimeError(f"{SCRATCH_ENV} must name an absolute container-disk directory")
    root, volume = Path(value).resolve(strict=False), volume.resolve(strict=False)
    if root == volume or volume in root.parents:
        raise ValueError(f"{SCRATCH_ENV} must be outside the network volume")
    root.mkdir(parents=True, exist_ok=True)
    return root


def publish_file(source: Path, final: Path) -> None:
    """Copy a finished local file with ordinary writes, fsync it, then rename it into place."""
    partial = final.with_name(f"{final.stem}.partial{final.suffix}")
    with source.open("rb") as reader, partial.open("xb") as writer:
        shutil.copyfileobj(reader, writer, COPY_CHUNK_BYTES)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(partial, final)


def runtime_identity(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    """Everything outside the checkpoint that can change bf16 inference results."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "mixed_precision": config.training.mixed_precision,
    }


def _forward_identity(record: dict[str, Any] | None) -> tuple[str, str] | None:
    if not record:
        return None
    return str(record.get("package_sha256")), str(record.get("end_exclusive"))


def forward_record(package_root: Path, volume: Path) -> dict[str, Any]:
    """Verify a sealed package on the volume and describe it for export.json."""
    from stock_forecasting.forward.package import POPULATION_SLOTS, load_package

    root = package_root.resolve(strict=True)
    if volume.resolve(strict=True) not in root.parents:
        raise ValueError("The forward package must be on the network volume")
    payload = load_package(root, verify_files=True)
    return {
        "path": str(root),
        "package_sha256": payload["package_sha256"],
        "end_exclusive": payload["end_exclusive"],
        "populations": {
            name: {
                "slot": slot,
                "samples": payload["populations"][name]["samples"],
            }
            for name, slot in POPULATION_SLOTS.items()
        },
    }


def reusable_export(
    directory: Path,
    checkpoint: Path,
    config: ExperimentConfig,
    splits: list[str],
    since: str,
    runtime: dict[str, Any],
    forward: dict[str, Any] | None = None,
) -> Path | None:
    """Return the newest complete export since a date that this job would reproduce.

    Only the export source may differ (it changes with scratch handling, not inference),
    so checkpoint files, architecture, runtime and every requested split must match, and
    each prediction file must still hash to the value recorded when it was written. A
    forward export is reused only for the same sealed input package.
    """
    files = None
    for candidate in sorted(directory.glob("export-*"), reverse=True):
        if candidate.name[len("export-") :][:8] < since:
            continue
        try:
            status = json.loads((candidate / "status.json").read_text())
            report = json.loads((candidate / "export.json").read_text())
        except (OSError, ValueError):
            continue
        recorded = report.get("checkpoint") or {}
        if (
            status.get("state") != "complete"
            or report.get("schema_version") != EXPORT_SCHEMA_VERSION
            or report.get("prediction_task") != config.model.prediction_task
            or report.get("runtime") != runtime
            or recorded.get("path") != str(checkpoint)
            or recorded.get("model_architecture_sha256") != config.model_architecture_digest()
            or _forward_identity(report.get("forward_package")) != _forward_identity(forward)
        ):
            continue
        if files is None:
            files = {name: sha256_file(checkpoint / name) for name in CHECKPOINT_FILES}
        results = report.get("splits") or {}
        if recorded.get("files_sha256") == files and all(
            split in results
            and (candidate / split / "predictions.npy").is_file()
            and sha256_file(candidate / split / "predictions.npy")
            == results[split].get("predictions_sha256")
            for split in splits
        ):
            return candidate
    return None


@torch.inference_mode()
def export_split(
    bundle, loader, config, device, output: Path, *, scratch: Path
) -> dict[str, Any]:
    """Stream predictions to a float32 array and recompute the unchanged signal metrics.

    Arrays are mapped under ``scratch`` (container disk) and only the finished predictions
    are copied to ``output`` on the network volume.
    """
    from stock_forecasting.training import _autocast_context, forward_batch, iter_device_batches

    if (
        not isinstance(loader.sampler, SequentialSampler)
        or len(loader.sampler) != len(loader.dataset)
        or loader.drop_last
    ):
        raise ValueError("Prediction export requires sequential full coverage without drop_last")
    task = config.model.prediction_task
    count = len(loader.dataset)
    shape = prediction_shape(task, count, len(config.data.alpha_horizons))
    output.mkdir(parents=True, exist_ok=False)
    require_disk(output, shape)
    scratch.mkdir(parents=True, exist_ok=True)
    require_disk(scratch, shape)
    scales = bundle.model.alpha_head.robust_scales.float().cpu().tolist()
    started = time.perf_counter()
    final = output / "predictions.npy"
    with TemporaryDirectory(prefix="signal-export-", dir=scratch) as directory:
        local = Path(directory) / "predictions.npy"
        arrays = np.lib.format.open_memmap(local, mode="w+", dtype="float32", shape=shape)
        try:
            store = SignalEvaluationStore(
                Path(directory) / "metrics",
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
            finally:
                store.close()
            arrays.flush()
        finally:
            del arrays
        digest = sha256_file(local)
        publish_file(local, final)
    if sha256_file(final) != digest:
        raise OSError(f"Published predictions differ from the local scratch copy: {final}")
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "prediction_task": task,
        "samples": count,
        "shape": list(shape),
        "dtype": "float32",
        "values": "alpha_point" if task == "alpha_point" else "absolute_direction_logits",
        "predictions_sha256": digest,
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


def _loader(config, bundle, device, split: str, forward: dict[str, Any] | None):
    if forward is None:
        return build_evaluation_loader(config, bundle=bundle, device=device, split=split)
    from stock_forecasting.forward.package import open_population

    # The package was re-hashed once in forward_record; loader workers reopen paths only.
    return build_evaluation_loader(
        config,
        bundle=bundle,
        device=device,
        split=forward["populations"][split]["slot"],
        dataset_factory=lambda symbol_cache_size: open_population(
            forward["path"],
            split,
            h_start=config.data.h_start,
            symbol_cache_size=symbol_cache_size,
            verify_files=False,
        ),
    )


def run_export(
    checkpoint: Path,
    selector: str,
    splits: list[str],
    reuse_since: str | None = None,
    forward: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    """Run on an existing GPU Pod under the diagnostic lease, without lifecycle changes.

    Returns the export directory and whether an earlier complete export was reused.
    ``forward`` (from ``forward_record``) switches to the forward package populations.
    """
    require_export_environment()
    volume = canonical_network_volume_root()
    config = ExperimentConfig.from_yaml(checkpoint / "resolved-config.yaml")
    validate_training_output_root(config.training.output_root)
    if config.model.time_series_backend != "kronos" or config.model.prediction_task not in (
        SIGNAL_TASKS
    ):
        raise ValueError("Prediction export requires an independent Kronos signal checkpoint")
    scratch = local_scratch_root(volume)
    device = torch.device("cuda")
    runtime = runtime_identity(config, device)
    directory = (
        volume / "diagnostics" / "signal-predictions"
        if forward is None
        else volume / "diagnostics" / "forward-predictions" / forward["end_exclusive"]
    )
    directory = directory / checkpoint.parent.name / checkpoint.name
    if reuse_since is not None:
        existing = reusable_export(
            directory, checkpoint, config, splits, reuse_since, runtime, forward
        )
        if existing is not None:
            LOGGER.info("Reusing complete export %s for %s", existing, checkpoint)
            return existing, True
    # Forward inference reads only the sealed package, never the configured dataset.
    run_preflight(
        config,
        require_data=forward is None,
        enforce_runtime_limit=False,
        allow_historical_inference=True,
    ).require_success()
    output = directory / f"export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    if output.resolve() != output:
        raise ValueError("Prediction export output must not traverse symlinked directories")
    log_handler = logging.FileHandler(output / "export.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(log_handler)
    atomic_write_json(output / "status.json", {"state": "running", "checkpoint": str(checkpoint)})
    try:
        set_global_seed(config.training.seed)
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
        for split in splits:
            loader = _loader(config, bundle, device, split, forward)
            LOGGER.info("Exporting %s predictions for %s", split, checkpoint)
            results[split] = export_split(
                bundle, loader, config, device, output / split, scratch=scratch
            )
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
                "files_sha256": {
                    name: sha256_file(checkpoint / name) for name in CHECKPOINT_FILES
                },
                "run_manifest_sha256": sha256_file(checkpoint.parent / "run-manifest.json"),
            },
            "prediction_task": config.model.prediction_task,
            "horizons": list(config.data.alpha_horizons),
            "splits": results,
            "runtime": runtime,
            "export_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        if forward is not None:
            report["forward_package"] = forward
        atomic_write_json(output / "export.json", report)
        atomic_write_json(
            output / "status.json", {"state": "complete", "checkpoint": str(checkpoint)}
        )
        LOGGER.info("Prediction export complete: %s", output)
        return output, False
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
    selectors = list(dict.fromkeys(args.checkpoint))
    if len(selectors) != len(args.checkpoint) or len(selectors) > MAX_CHECKPOINTS:
        raise SystemExit(f"Pass 1-{MAX_CHECKPOINTS} distinct --checkpoint selectors")
    try:
        for selector in selectors:
            parse_selector(selector)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    allowed = FORWARD_POPULATIONS if args.forward_package else SPLITS
    splits = list(dict.fromkeys(args.splits or allowed))
    if set(splits) - set(allowed):
        raise SystemExit(
            "--forward-package exports only overlap/forward"
            if args.forward_package
            else "overlap/forward require --forward-package"
        )
    if {"test", "overlap"} & set(splits) and not args.confirm_test:
        raise SystemExit("--splits test or overlap requires --confirm-test")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require_export_environment()
    volume = canonical_network_volume_root()
    saved_model_root = volume / "savedModel"
    # Resolve every selector and the package before GPU work so mistakes fail fast.
    checkpoints = [resolve_selector(selector, saved_model_root) for selector in selectors]
    if len(set(checkpoints)) != len(checkpoints):
        raise SystemExit("Two --checkpoint selectors resolve to the same checkpoint")
    forward = forward_record(args.forward_package, volume) if args.forward_package else None
    outputs = []
    for selector, checkpoint in zip(selectors, checkpoints, strict=True):
        output, reused = run_export(
            checkpoint, selector, splits, args.reuse_complete_since, forward
        )
        outputs.append(str(output))
        state = "reused" if reused else "complete"
        print(json.dumps({"state": state, "output": str(output)}, ensure_ascii=False))
    print(json.dumps({"state": "complete", "exports": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
