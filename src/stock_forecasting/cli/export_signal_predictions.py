"""Export per-sample predictions of an existing signal checkpoint (read-only diagnostic).

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
from stock_forecasting.run_paths import canonical_network_volume_root, validate_training_output_root
from stock_forecasting.signal_evaluation import SignalEvaluationStore
from stock_forecasting.signal_training import _past_drawdown
from stock_forecasting.training import build_evaluation_loader, set_global_seed

LOGGER = logging.getLogger(__name__)
EXPORT_SCHEMA_VERSION = "1.0"
SPLITS = ("validation", "test")
CHECKPOINT_FILES = ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json")
# One diagnostic Pod ends with its job, so several checkpoints share one sequential job.
MAX_CHECKPOINTS = 16
# Scratch for the unchanged metric store plus a margin for logs and manifests.
DISK_MARGIN_BYTES = 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--checkpoint",
        metavar="RUN_ID",
        action="append",
        required=True,
        help=(
            "Signal child run ID whose validation-selected best checkpoint is exported; "
            f"repeat for up to {MAX_CHECKPOINTS} runs exported sequentially in one job."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Complete evaluation splits to export (default: validation test).",
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required when exporting test, to make repeated test inference explicit.",
    )
    return parser


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
def export_split(bundle, loader, config, device, output: Path) -> dict[str, Any]:
    """Stream predictions to a float32 array and recompute the unchanged signal metrics."""
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
            finally:
                store.close()
        arrays.flush()
    finally:
        del arrays
    final = output / "predictions.npy"
    os.replace(partial, final)
    summary = {
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
        "seconds": time.perf_counter() - started,
    }
    atomic_write_json(output / "split.json", summary)
    return summary


def run_export(run_id: str, splits: list[str]) -> Path:
    """Run on an existing GPU Pod under the diagnostic lease, without lifecycle changes."""
    if not os.environ.get("RUNPOD_POD_ID") or not torch.cuda.is_available():
        raise RuntimeError("Prediction export must run inside an existing CUDA RunPod Pod")
    if os.environ.get("RUNPOD_GPU_WORKFLOW_LEASE_HELD") != "1":
        raise RuntimeError(
            "Use bash scripts/runpod_tmux_launch.sh export-signal-predictions to verify the "
            "mount and GPU lease"
        )
    volume = canonical_network_volume_root()
    checkpoint = resolve_probe_checkpoint(run_id, saved_model_root=volume / "savedModel")
    config = ExperimentConfig.from_yaml(checkpoint / "resolved-config.yaml")
    validate_training_output_root(config.training.output_root)
    if config.model.time_series_backend != "kronos" or config.model.prediction_task not in (
        SIGNAL_TASKS
    ):
        raise ValueError("Prediction export requires an independent Kronos signal checkpoint")
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
        for split in splits:
            loader = build_evaluation_loader(config, bundle=bundle, device=device, split=split)
            LOGGER.info("Exporting %s predictions for %s", split, checkpoint)
            results[split] = export_split(bundle, loader, config, device, output / split)
            del loader
        report = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "kind": "signal-prediction-export",
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": checkpoint.parent.name,
            "checkpoint": {
                "path": str(checkpoint),
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
    run_ids = list(dict.fromkeys(args.checkpoint))
    if len(run_ids) != len(args.checkpoint) or len(run_ids) > MAX_CHECKPOINTS:
        raise SystemExit(f"Pass 1-{MAX_CHECKPOINTS} distinct --checkpoint run IDs")
    splits = list(dict.fromkeys(args.splits))
    if "test" in splits and not args.confirm_test:
        raise SystemExit("--splits test requires --confirm-test")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outputs = []
    for run_id in run_ids:
        output = run_export(run_id, splits)
        outputs.append(str(output))
        print(json.dumps({"state": "complete", "output": str(output)}, ensure_ascii=False))
    print(json.dumps({"state": "complete", "exports": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
