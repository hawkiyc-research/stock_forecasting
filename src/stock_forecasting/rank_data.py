"""Main-model rank labels on the lazy dataset, kept outside the baseline identity sources."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from stock_forecasting.data.rank_labels import attach_rank_labels

RANK_TASK = "cross_sectional_rank"


class RankLabelDataset:
    """Add each window's gauss-rank labels to Kronos items without changing bars or indexes."""

    def __init__(self, source) -> None:
        if getattr(source, "rank_label_root", None) is None:
            raise ValueError("Rank-label items require a dataset with attached rank labels")
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
            raise ValueError("Rank batch indexes must be one-dimensional")
        if len(ordinals) == 0:
            return []
        ordinals = np.where(ordinals < 0, ordinals + len(self), ordinals)
        if (ordinals < 0).any() or (ordinals >= len(self)).any():
            raise IndexError("Rank batch index is outside the split")
        items = self.source.__getitems__(ordinals.tolist())
        ranges = np.searchsorted(self.source._range_ends, ordinals, side="right")
        symbols = self.source._range_symbols[ranges]
        cutoffs = pd.to_datetime([item["cutoff_at"] for item in items], utc=True).asi8
        for symbol in np.unique(symbols):
            selected = np.flatnonzero(symbols == symbol)
            # The reader's per-symbol cache was just filled while building these items.
            labels = self.source._rank_targets(symbol, ordinals[selected], cutoffs[selected])
            for offset, values in zip(selected, torch.from_numpy(labels), strict=True):
                items[offset]["target_rank"] = values
        return items


def with_rank_labels(dataset, config):
    """Attach existing rank labels and expose them as item targets for the rank task."""
    if getattr(config.model, "prediction_task", "quantiles") != RANK_TASK:
        return dataset
    attach_rank_labels(dataset, config.model.rank_min_group_size)
    return RankLabelDataset(dataset)


def batch_rank_targets(batch, task, count, horizons):
    """Rank labels as numpy, or NaN where a population has no rank-label store."""
    if task != RANK_TASK:
        return None
    ranks = batch.get("target_rank")
    if ranks is None:
        # Forward populations are unlabelled; their Rank IC still uses raw returns.
        return np.full((count, horizons), np.nan, dtype=np.float32)
    return ranks.detach().float().cpu().numpy()
