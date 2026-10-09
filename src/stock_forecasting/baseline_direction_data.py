"""Lazy, past-price cost bands and shared execution labels for baseline tasks."""

from __future__ import annotations

import numpy as np

from stock_forecasting.direction_costs import direction_cost_band


def direction_array_batch(source, indices, costs):
    """Decode each symbol once; keep absolute labels out of predictor inputs."""
    ordinals = np.asarray(indices, dtype=np.int64)
    if ordinals.ndim != 1 or not len(ordinals):
        raise ValueError("Direction batch must contain a bounded one-dimensional index list")
    ordinals = np.where(ordinals < 0, ordinals + len(source), ordinals)
    if (ordinals < 0).any() or (ordinals >= len(source)).any():
        raise IndexError("Direction batch index is outside the split")
    ranges = np.searchsorted(source._range_ends, ordinals, side="right")
    cutoffs = source._range_starts[ranges] + ordinals - source._range_begins[ranges]
    symbols = source._range_symbols[ranges]
    result = {
        "asset": np.empty((len(indices), source.window_size, 5), dtype=np.float64),
        "benchmark": np.empty((len(indices), source.window_size, 5), dtype=np.float64),
        "targets": np.empty((len(indices), len(source.horizons)), dtype=np.float64),
        "absolute_targets": np.empty((len(indices), len(source.horizons)), dtype=np.float32),
        "direction_bands": np.empty((len(indices), 2), dtype=np.float32),
        "metadata": [None] * len(indices),
        "horizons": source.horizons,
    }
    for symbol in np.unique(symbols):
        selected = np.flatnonzero(symbols == symbol)
        batch = source.array_batch(ordinals[selected], include_timestamps=False)
        bars = source._numeric_symbol(symbol)
        entry_position = cutoffs[selected] + 1
        entry = bars["raw"][entry_position, 0] * bars["price_factor"][entry_position]
        exits = cutoffs[selected, None] + np.asarray(source.horizons)[None]
        gross = bars["adjusted_close"][exits] / entry[:, None]
        if not np.isfinite(gross).all() or (gross <= 0).any():
            raise ValueError("Absolute execution returns must be finite and positive")
        result["absolute_targets"][selected] = np.log(gross).astype(np.float32)
        for key in ("asset", "benchmark", "targets"):
            result[key][selected] = batch[key]
        for j, position in enumerate(selected):
            metadata = batch["metadata"][j]
            row = metadata[3]
            # Match the main collator's float32 cutoff price, never the future entry.
            band = direction_cost_band(
                float(np.float32(batch["asset"][j, -1, 3])),
                str(row["market"]),
                str(row["asset_type"]),
                str(metadata[0]),
                costs,
            )
            result["direction_bands"][position] = (
                band["lower_log_return"],
                band["upper_log_return"],
            )
            result["metadata"][position] = metadata
    return result


def numpy_direction_labels(targets, bands):
    """Use the same inclusive neutral interval and float32 arithmetic as Torch."""
    targets, bands = np.asarray(targets, dtype=np.float32), np.asarray(bands, dtype=np.float32)
    if targets.ndim != 2 or bands.shape != (len(targets), 2):
        raise ValueError("Direction labels require [batch,horizon] returns and [batch,2] bands")
    return np.where(targets < bands[:, :1], 0, np.where(targets > bands[:, 1:], 2, 1)).astype(
        np.uint8
    )
