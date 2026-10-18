"""Pieces shared by the backtest runs: inputs, worker sizing, a bounded process pool, output."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from twbacktest import data

# Each task holds the Taiwan rows of two return arrays and one signal (~0.1 GB);
# the budget leaves room for memory-mapped pages and pandas overhead.
WORKER_MEMORY_BYTES = 512 * 1024**2
MAX_WORKERS = 16
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")


def default_workers(tasks: int) -> int:
    configured = os.environ.get("BACKTEST_WORKERS")
    if configured:
        workers = int(configured)
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"BACKTEST_WORKERS must be 1-{MAX_WORKERS}")
        return workers
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        by_memory = max(1, memory // 4 // WORKER_MEMORY_BYTES)
    except (ValueError, OSError, AttributeError):
        by_memory = 2
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 2) - 1, by_memory, tasks))


def prepare_inputs(
    config: dict[str, Any], artifacts: Path
) -> tuple[dict[str, Any], dict[str, data.SplitIndex], dict[str, int]]:
    """Input paths with verified row-order digests, Taiwan split indexes and full row counts."""
    paths = config["inputs"]
    inputs: dict[str, Any] = {
        "alpha_baseline": artifacts / paths["alpha_baseline"],
        "direction_baseline": artifacts / paths["direction_baseline"],
        "predictions_root": artifacts / paths["predictions_root"],
        "digests": {},
    }
    indexes, totals = {}, {}
    for split in config["splits"]:
        digest = data.baseline_digest(inputs["alpha_baseline"], split)
        if digest != data.baseline_digest(inputs["direction_baseline"], split):
            raise ValueError(f"Alpha and direction baselines differ in {split} row order")
        inputs["digests"][split] = digest
        indexes[split] = data.build_split_index(inputs["alpha_baseline"], split)
        totals[split] = int(np.load(inputs["alpha_baseline"] / "inputs" / split / "targets.npy",
                                    mmap_mode="r").shape[0])
    return inputs, indexes, totals


def run_tasks(
    tasks: Sequence[Any],
    function: Callable[[Any], list[dict[str, Any]]],
    *,
    initializer: Callable[..., None],
    initargs: tuple,
    workers: int,
) -> list[dict[str, Any]]:
    """Rows of every task: serial with one worker, else a spawn pool with bounded submission."""
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    rows: list[dict[str, Any]] = []
    if workers == 1:
        initializer(*initargs)
        for task in tasks:
            rows.extend(function(task))
        return rows
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=initializer,
        initargs=initargs,
    ) as pool:
        pending: set = set()
        for task in tasks:
            pending.add(pool.submit(function, task))
            # Backpressure: at most two tasks per worker in flight.
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.extend(future.result())
        for future in pending:
            rows.extend(future.result())
    return rows


def clean(value: Any) -> Any:
    """JSON-ready copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{100 * float(value):+.1f}%"


def number(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):+.2f}"


def default_output_root(artifacts: Path, prefix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return artifacts / "diagnostics" / "backtest" / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
