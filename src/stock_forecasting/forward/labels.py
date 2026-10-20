"""Realized labels of a forward population in the baseline evaluation-array format.

Alpha targets come from the lazy dataset and absolute targets from ``DirectionDataset``,
the same code that produced the validation/test arrays, so the backtest reads these
files exactly like ``<baseline>/inputs/<split>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.evaluation_protocol import sample_membership
from stock_forecasting.evaluation_store import META_DTYPE

LABEL_BATCH_ROWS = 2048


def population_labels(dataset, *, batch_rows: int = LABEL_BATCH_ROWS) -> dict[str, np.ndarray]:
    """Alpha targets, absolute targets and metadata in the dataset's row order."""
    from stock_forecasting.direction_data import DirectionDataset

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    source = DirectionDataset(dataset)
    count = len(source)
    horizons = len(dataset.horizons)
    alpha = np.empty((count, horizons), dtype=np.float32)
    absolute = np.empty((count, horizons), dtype=np.float32)
    metadata = np.empty(count, dtype=META_DTYPE)
    for start in range(0, count, batch_rows):
        stop = min(start + batch_rows, count)
        items = source.__getitems__(list(range(start, stop)))
        alpha[start:stop] = np.stack([item["target_alpha"].numpy() for item in items])
        absolute[start:stop] = np.stack([item["target_absolute"].numpy() for item in items])
        for offset, item in enumerate(items, start=start):
            metadata[offset] = tuple(
                str(item[key])
                for key in ("symbol", "cutoff_at", "market", "asset_type", "provider")
            )
    if not (np.isfinite(alpha).all() and np.isfinite(absolute).all()):
        raise ValueError("Forward labels must be finite")
    return {"targets": alpha, "absolute_targets": absolute, "metadata": metadata}


def write_population_labels(dataset, output: str | Path, *, population: str) -> dict[str, Any]:
    """Write ``targets.npy``, ``absolute_targets.npy``, ``metadata.npy`` and ``labels.json``."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = population_labels(dataset)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", values)
    membership = sample_membership(
        arrays["metadata"]["symbol"].tolist(), arrays["metadata"]["date"].tolist()
    )
    summary = {
        "population": population,
        "samples": len(arrays["metadata"]),
        "sample_membership": membership,
        "files_sha256": {
            f"{name}.npy": sha256_file(directory / f"{name}.npy") for name in arrays
        },
    }
    atomic_write_json(directory / "labels.json", summary)
    return summary
