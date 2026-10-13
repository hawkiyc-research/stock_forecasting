"""Main-model-only absolute labels over the unchanged cleaned lazy sample universe."""

from __future__ import annotations

import numpy as np
import torch


class DirectionDataset:
    """Add bounded vectorized labels without rewriting bars, indexes or baselines."""

    def __init__(self, source) -> None:
        self.source = source

    def __getattr__(self, name):
        source = self.__dict__.get("source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def __len__(self):
        return len(self.source)

    @property
    def symbol_cache_size(self):
        return self.source.symbol_cache_size

    @symbol_cache_size.setter
    def symbol_cache_size(self, value):
        self.source.symbol_cache_size = value

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        ordinals = np.asarray(indices, dtype=np.int64)
        if ordinals.ndim != 1:
            raise ValueError("Direction batch indexes must be one-dimensional")
        if len(ordinals) == 0:
            return []
        ordinals = np.where(ordinals < 0, ordinals + len(self), ordinals)
        if (ordinals < 0).any() or (ordinals >= len(self)).any():
            raise IndexError("Direction batch index is outside the split")
        ranges = np.searchsorted(self.source._range_ends, ordinals, side="right")
        cutoffs = self.source._range_starts[ranges] + ordinals - self.source._range_begins[ranges]
        symbols = self.source._range_symbols[ranges]
        result = [None] * len(ordinals)
        for symbol in np.unique(symbols):
            selected = np.flatnonzero(symbols == symbol)
            # Consume absolute labels while the original reader's symbol cache is hot.
            # A second pass over an entire mixed-symbol batch would decode bars twice.
            items = self.source.__getitems__(ordinals[selected].tolist())
            bars = self.source._numeric_symbol(symbol)
            entry_position = cutoffs[selected] + 1
            entry = bars["raw"][entry_position, 0] * bars["price_factor"][entry_position]
            exits = cutoffs[selected, None] + np.asarray(self.source.horizons)[None]
            gross = bars["adjusted_close"][exits] / entry[:, None]
            if not np.isfinite(gross).all() or (gross <= 0).any():
                raise ValueError("Absolute execution returns must be finite and positive")
            targets = torch.from_numpy(np.log(gross).astype(np.float32))
            for offset, item, target in zip(selected, items, targets, strict=True):
                item["target_absolute"] = target
                result[offset] = item
        return result


def with_direction_targets(dataset, config):
    required = config.model.direction.enabled or (
        getattr(config.model, "prediction_task", "quantiles") == "absolute_direction"
    )
    if not required or isinstance(dataset, DirectionDataset):
        return dataset
    if dataset.series_mode != "raw":
        raise ValueError("Absolute-return labels require raw adjusted execution prices")
    return DirectionDataset(dataset)
