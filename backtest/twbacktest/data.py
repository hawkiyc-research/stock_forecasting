"""Load the P0 evaluation arrays and exported signals for the Taiwan rows.

All arrays share the row order of the evaluation population. The order is
checked through the sample digests recorded by the baselines and every export;
the large arrays are memory-mapped and only the Taiwan rows are copied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MARKETS = ("TWSE", "TPEX")
HORIZONS = 14
DOWN, UP = 0, 2
_DIGEST_KEYS = {"validation": "validation_membership", "test": "evaluation_membership"}


@dataclass(frozen=True)
class SplitIndex:
    """Compact Taiwan-row description of one split, cheap to send to workers."""

    split: str
    rows: np.ndarray  # positions of the Taiwan rows in the full arrays
    date_codes: np.ndarray  # index into dates, ascending signal dates
    dates: tuple[str, ...]
    market_codes: np.ndarray  # index into MARKETS
    is_etf: np.ndarray
    symbol_codes: np.ndarray  # same code for the same stock on every date

    @property
    def count(self) -> int:
        return int(self.rows.size)


def baseline_digest(baseline_dir: Path, split: str) -> str:
    payload = json.loads((baseline_dir / "complete.json").read_text())
    return str(payload[_DIGEST_KEYS[split]]["ordered_symbol_dates_sha256"])


def build_split_index(alpha_baseline: Path, split: str) -> SplitIndex:
    """Read the metadata once and keep integer codes for the Taiwan rows."""
    metadata = np.load(alpha_baseline / "inputs" / split / "metadata.npy")
    markets = metadata["market"]
    taiwan = np.isin(markets, MARKETS)
    rows = np.flatnonzero(taiwan).astype(np.int64)
    day = np.array([value[:10] for value in metadata["date"][rows]])
    dates, date_codes = np.unique(day, return_inverse=True)
    market_codes = np.where(markets[rows] == MARKETS[0], 0, 1).astype(np.int8)
    _symbols, symbol_codes = np.unique(metadata["symbol"][rows], return_inverse=True)
    return SplitIndex(
        split=split,
        rows=rows,
        date_codes=date_codes.astype(np.int32),
        dates=tuple(str(value) for value in dates),
        market_codes=market_codes,
        is_etf=(metadata["asset_type"][rows] == "etf"),
        symbol_codes=symbol_codes.astype(np.int32),
    )


def _take(path: Path, rows: np.ndarray, expected: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.shape[0] != expected:
        raise ValueError(f"{path} has {array.shape[0]} rows, expected {expected}")
    return np.asarray(array[rows], dtype=np.float64)


def load_returns(
    alpha_baseline: Path, direction_baseline: Path, index: SplitIndex, total_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log returns (next open to close of t+h) and log alpha for the Taiwan rows."""
    absolute = _take(
        direction_baseline / "inputs" / index.split / "absolute_targets.npy", index.rows,
        total_rows,
    )
    alpha = _take(alpha_baseline / "inputs" / index.split / "targets.npy", index.rows, total_rows)
    return absolute, alpha


def export_directory(predictions_root: Path, run: str, checkpoint: str) -> Path:
    """The newest complete export of one checkpoint."""
    candidates = []
    for directory in sorted((predictions_root / run / checkpoint).glob("export-*")):
        status = directory / "status.json"
        if status.is_file() and json.loads(status.read_text()).get("state") == "complete":
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"No complete export for {run}/{checkpoint}")
    return candidates[-1]


def verify_export(directory: Path, split: str, digest: str) -> None:
    payload = json.loads((directory / "export.json").read_text())
    recorded = payload["splits"][split]["sample_membership"]["ordered_symbol_dates_sha256"]
    if recorded != digest:
        raise ValueError(f"{directory} {split} row order differs from the baseline arrays")


def _direction_score(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def load_signal(
    spec: dict[str, Any],
    *,
    inputs: dict[str, Any],
    index: SplitIndex,
    total_rows: int,
) -> np.ndarray:
    """Scores [rows, 14]; higher means a better expected return at that horizon.

    Alpha sources average point predictions; direction sources average class
    probabilities and score P(up) - P(down).
    """
    kind = spec["kind"]
    if "baseline" in spec:
        root = inputs["alpha_baseline" if kind == "alpha" else "direction_baseline"]
        paths = [root / "jobs" / spec["baseline"] / index.split / "predictions.npy"]
    else:
        paths = []
        for run, checkpoint in spec["members"]:
            directory = export_directory(inputs["predictions_root"], run, checkpoint)
            verify_export(directory, index.split, inputs["digests"][index.split])
            paths.append(directory / index.split / "predictions.npy")
    total = None
    for path in paths:
        values = _take(path, index.rows, total_rows)
        if kind == "direction":
            values = _direction_score(values)
        total = values if total is None else total + values
    assert total is not None
    mean = total / len(paths)
    if kind == "direction":
        return mean[..., UP] - mean[..., DOWN]
    return mean
