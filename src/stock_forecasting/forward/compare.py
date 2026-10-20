"""Data check 4: forward-package model inputs against the shared evaluation snapshot.

Rows are matched by symbol and signal date. For each matched row the float64 asset and
benchmark windows from ``array_batch`` (the arrays the model tensors are cast from) are
compared by maximum relative difference, and timestamp features must be equal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TAIWAN_MARKETS = ("TWSE", "TPEX")
COMPARE_BATCH_ROWS = 4096


def row_keys(dataset, markets=TAIWAN_MARKETS) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Ordinals and (symbol, signal date) keys of the dataset rows in the given markets."""
    ordinals, keys = [], []
    begin = 0
    for row in dataset.ranges.itertuples(index=False):
        count = int(row.count)
        symbol = str(row.symbol)
        if str(dataset._index[symbol]["market"]) in markets:
            timestamps = dataset._numeric_symbol(symbol)["timestamps"]
            for offset in range(count):
                ordinals.append(begin + offset)
                keys.append((symbol, timestamps[int(row.start_index) + offset].isoformat()))
        begin += count
    return np.asarray(ordinals, dtype=np.int64), keys


def _relative_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
    return np.abs(left - right) / scale


def compare_inputs(
    reference,
    candidate,
    *,
    tolerance: float,
    markets=TAIWAN_MARKETS,
    batch_rows: int = COMPARE_BATCH_ROWS,
) -> dict[str, Any]:
    """Match reference rows in the candidate and measure their input differences."""
    reference_ordinals, reference_keys = row_keys(reference, markets)
    candidate_ordinals, candidate_keys = row_keys(candidate, markets)
    position = {key: index for index, key in enumerate(candidate_keys)}
    matched = [
        (reference_ordinals[index], candidate_ordinals[position[key]])
        for index, key in enumerate(reference_keys)
        if key in position
    ]
    worst = np.empty(len(matched), dtype=np.float64)
    stamps_equal = np.empty(len(matched), dtype=bool)
    targets_difference = np.empty(len(matched), dtype=np.float64)
    for start in range(0, len(matched), batch_rows):
        pairs = np.asarray(matched[start : start + batch_rows], dtype=np.int64)
        left = reference.array_batch(pairs[:, 0], include_metadata=False)
        right = candidate.array_batch(pairs[:, 1], include_metadata=False)
        stop = start + len(pairs)
        worst[start:stop] = np.maximum(
            _relative_difference(left["asset"], right["asset"]).max(axis=(1, 2)),
            _relative_difference(left["benchmark"], right["benchmark"]).max(axis=(1, 2)),
        )
        stamps_equal[start:stop] = (
            left["timestamp_features"] == right["timestamp_features"]
        ).all(axis=(1, 2))
        targets_difference[start:stop] = np.abs(left["targets"] - right["targets"]).max(axis=1)
    within = (worst <= tolerance) & stamps_equal
    return {
        "markets": list(markets),
        "reference_rows": len(reference_keys),
        "candidate_rows": len(candidate_keys),
        "matched_rows": len(matched),
        "coverage": len(matched) / len(reference_keys) if reference_keys else None,
        "tolerance": tolerance,
        "within_tolerance_rows": int(within.sum()),
        "within_tolerance_fraction": float(within.mean()) if len(matched) else None,
        "exact_rows": int(((worst == 0) & stamps_equal).sum()),
        "timestamp_mismatch_rows": int((~stamps_equal).sum()),
        "max_relative_difference": float(worst.max()) if len(matched) else None,
        # Descriptive only: alpha labels are covered by data check 3.
        "max_alpha_target_difference": float(targets_difference.max()) if len(matched) else None,
    }
