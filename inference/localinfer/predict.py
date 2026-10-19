"""Run one retained Kronos signal checkpoint on the Taiwan rows of a local evaluation split.

Read-only and local (experiment protocol section 13.7). The model is assembled from the
checkpoint's own resolved config once its architecture digest matches the trainer state;
only the Kronos source path is redirected to the repository's sha256-pinned copy. The
split comes from a local copy of the shared evaluation bar store and its cleaned sample
index, which are verified but never rebuilt. Predictions are written in float32 with each
row's stock, date and market. No checkpoint, RunPod artifact or lifecycle record changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_KRONOS = REPO_ROOT / "src" / "stock_forecasting" / "_vendor" / "kronos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "diagnostics" / "local-predictions"
TAIWAN = ("TWSE", "TPEX")
SPLITS = ("validation", "test")
OUTPUT_SCHEMA = "1.0"
# One data-loading worker holds its own symbol cache and Parquet buffers.
WORKER_MEMORY_BYTES = 2 * 1024**3
MAX_WORKERS = 8
CHECKPOINT_FILES = ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json")
LOGGER = logging.getLogger("localinfer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_workers() -> int:
    """Data-loading workers from the environment, CPU count and physical memory."""
    configured = os.environ.get("LOCAL_INFERENCE_WORKERS")
    if configured:
        workers = int(configured)
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"LOCAL_INFERENCE_WORKERS must be 1-{MAX_WORKERS}")
        return workers
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        by_memory = max(1, memory // 4 // WORKER_MEMORY_BYTES)
    except (ValueError, OSError, AttributeError):
        by_memory = 1
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 2) // 2, by_memory))


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")
    if name not in {"mps", "cpu"}:
        raise ValueError("Local inference supports only mps, cpu or auto")
    return torch.device(name)


def load_config(checkpoint: Path, *, kronos_source: Path = VENDORED_KRONOS):
    """The checkpoint's own config, checked against its trainer state before any path edit."""
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.models.signal import SIGNAL_TASKS

    config = ExperimentConfig.from_yaml(checkpoint / "resolved-config.yaml")
    state = json.loads((checkpoint / "trainer-state.json").read_text())
    if config.model_architecture_digest() != state.get("model_architecture_sha256"):
        raise ValueError("Checkpoint config does not match its recorded model architecture")
    if (
        config.model.time_series_backend != "kronos"
        or config.model.prediction_task not in SIGNAL_TASKS
        or state.get("prediction_task") != config.model.prediction_task
    ):
        raise ValueError("Local inference requires an independent Kronos signal checkpoint")
    # Only the source location changes; the pinned file digests are verified on import.
    config.model.kronos_source_root = Path(kronos_source)
    return config, state


def taiwan_positions(dataset, markets: tuple[str, ...] = TAIWAN) -> np.ndarray:
    """Sequential positions of the rows whose symbol belongs to the requested markets."""
    ends = np.asarray(dataset._range_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    chosen = [
        np.arange(start, end, dtype=np.int64)
        for start, end, symbol in zip(starts, ends, dataset._range_symbols, strict=True)
        if str(dataset._index[str(symbol)]["market"]) in markets
    ]
    if not chosen:
        raise ValueError("The split has no rows in the requested markets")
    return np.concatenate(chosen)


def open_split(config, bar_store: Path, sample_universe: Path, split: str, *, cache_size: int):
    """The cleaned split of a local bar store; returns (model dataset, base dataset)."""
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset
    from stock_forecasting.direction_data import with_direction_targets
    from stock_forecasting.training_paths import resolve_bar_store_path

    if split not in SPLITS:
        raise ValueError(f"Local inference splits are {', '.join(SPLITS)}")
    universe = json.loads((Path(sample_universe) / "universe.json").read_text())
    if universe.get("state") != "ready":
        raise ValueError("The local sample index is not a ready, published copy")
    base = LazyFinancialWindowDataset(
        resolve_bar_store_path(bar_store),
        split=split,
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        sample_index_root=sample_universe,
        symbol_cache_size=cache_size,
    )
    return with_direction_targets(base, config), base


def _peak_memory() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    peak = usage if platform.system() == "Darwin" else usage * 1024
    result: dict[str, Any] = {"process_max_rss_bytes": int(peak)}
    if torch.backends.mps.is_available():
        result["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return result


def output_directory(root: Path, checkpoint: Path, split: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (Path(root) / checkpoint.parent.name / checkpoint.name
            / f"{split}-{stamp}-{uuid.uuid4().hex[:8]}")


@torch.inference_mode()
def predict(bundle, loader, config, device, count: int, output: Path) -> dict[str, Any]:
    """Stream float32 predictions, keys and alpha targets for every selected row."""
    from stock_forecasting.training import forward_batch, iter_device_batches

    task = config.model.prediction_task
    horizons = len(config.data.alpha_horizons)
    shape = (count, horizons) if task == "alpha_point" else (count, horizons, 3)
    partial = output / "predictions.partial.npy"
    values_out = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=shape)
    targets = np.full((count, horizons), np.nan, dtype=np.float32)
    keys: dict[str, list[str]] = {"symbol": [], "cutoff_at": [], "market": []}
    started = last_log = time.perf_counter()
    completed = 0
    try:
        for batch in iter_device_batches(loader, device):
            result = forward_batch(bundle, batch, config, device)
            values = (result.alpha_point if task == "alpha_point"
                      else result.absolute_direction_logits)
            values = values.detach().float().cpu().numpy()
            size = len(values)
            values_out[completed:completed + size] = values
            targets[completed:completed + size] = (
                batch["target_alpha"].detach().float().cpu().numpy()
            )
            keys["symbol"].extend(str(value) for value in batch["symbols"])
            keys["cutoff_at"].extend(str(value) for value in batch["cutoff_at"])
            keys["market"].extend(str(value) for value in batch["markets"])
            completed += size
            now = time.perf_counter()
            if now - last_log >= 30:
                LOGGER.info("Local inference %s/%s at %.1f samples/s", completed, count,
                            completed / (now - started))
                last_log = now
        if completed != count:
            raise RuntimeError("Local inference did not cover every selected row")
        values_out.flush()
    finally:
        del values_out
    os.replace(partial, output / "predictions.npy")
    np.save(output / "targets_alpha.npy", targets)
    pd.DataFrame(keys).to_parquet(output / "keys.parquet", index=False)
    seconds = time.perf_counter() - started
    return {"samples": count, "shape": list(shape), "seconds": seconds,
            "samples_per_second": count / seconds if seconds > 0 else float("nan")}


def run(
    checkpoint: Path,
    *,
    bar_store: Path,
    sample_universe: Path,
    split: str,
    device_name: str,
    batch_size: int,
    workers: int,
    output_root: Path,
) -> Path:
    from stock_forecasting.checkpointing import load_checkpoint
    from stock_forecasting.factory import build_model_bundle
    from stock_forecasting.training import ModelBatchCollator, set_global_seed

    checkpoint = Path(checkpoint).resolve(strict=True)
    device = select_device(device_name)
    config, state = load_config(checkpoint)
    set_global_seed(config.training.seed)
    dataset, base = open_split(config, bar_store, sample_universe, split, cache_size=32)
    positions = taiwan_positions(base)
    output = output_directory(output_root, checkpoint, split)
    output.mkdir(parents=True, exist_ok=False)
    status = output / "status.json"
    status.write_text(json.dumps({"state": "running", "checkpoint": str(checkpoint)}))
    try:
        # Off CUDA the factory keeps every weight in float32.
        bundle = build_model_bundle(config, device)
        loaded = load_checkpoint(checkpoint, bundle.model)
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        loader = DataLoader(
            Subset(dataset, positions.tolist()),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=ModelBatchCollator.for_model(config, training=False),
            num_workers=workers,
            multiprocessing_context="spawn",
            persistent_workers=False,
            prefetch_factor=2,
        )
        summary = predict(bundle, loader, config, device, len(positions), output)
        report = {
            "schema_version": OUTPUT_SCHEMA,
            "kind": "local-signal-predictions",
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": checkpoint.parent.name,
            "checkpoint": {
                "name": checkpoint.name,
                "global_step": state.get("global_step"),
                "model_architecture_sha256": state.get("model_architecture_sha256"),
                "files_sha256": {name: sha256_file(checkpoint / name)
                                 for name in CHECKPOINT_FILES},
                "missing_frozen_keys": len(loaded.get("missing_frozen_keys") or []),
            },
            "prediction_task": config.model.prediction_task,
            "values": ("alpha_point" if config.model.prediction_task == "alpha_point"
                       else "absolute_direction_logits"),
            "horizons": list(config.data.alpha_horizons),
            "split": split,
            "markets": list(TAIWAN),
            "data": {
                "bar_store_manifest_sha256": sha256_file(Path(bar_store) / "bar-store.json"),
                "sample_universe": base.sample_universe_identity,
                "split_samples": len(base),
            },
            "runtime": {
                "device": device.type,
                "dtype": "float32",
                "batch_size": batch_size,
                "workers": workers,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "machine": platform.machine(),
                **_peak_memory(),
            },
            **summary,
            "predictions_sha256": sha256_file(output / "predictions.npy"),
        }
        (output / "run.json").write_text(json.dumps(report, indent=1))
        status.write_text(json.dumps({"state": "complete", "checkpoint": str(checkpoint)}))
        return output
    except BaseException as error:
        status.write_text(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local checkpoint directory, e.g. .../RUN_ID/checkpoint-018399.")
    parser.add_argument("--bar-store", type=Path, required=True,
                        help="Local copy of the shared evaluation prepared/bar-store.")
    parser.add_argument("--sample-universe", type=Path, required=True,
                        help="Local copy of its prepared/sample-universes/<identity> index.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, help=f"Data-loading workers, 1-{MAX_WORKERS}.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers or default_workers()
    if not 1 <= workers <= MAX_WORKERS or args.batch_size < 1:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS} and --batch-size positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = run(
        args.checkpoint, bar_store=args.bar_store, sample_universe=args.sample_universe,
        split=args.split, device_name=args.device, batch_size=args.batch_size, workers=workers,
        output_root=args.output_root,
    )
    print(json.dumps({"state": "complete", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
