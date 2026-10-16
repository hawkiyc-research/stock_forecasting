"""Sample-level random training order with independent annual weighting."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stock_forecasting.data.dataset import BlockwisePermutationSampler
from stock_forecasting.date_market_sampler import date_market_index
from stock_forecasting.random_sampling import SAMPLING_VERSION, annual_quotas, shuffled_positions


class RandomTrainingSampler(BlockwisePermutationSampler):
    """Shuffle individual windows globally, retaining lazy multiworker decoding."""

    def __init__(
        self,
        source,
        *,
        requested_workers: int,
        scratch_root: Path,
        max_scratch_bytes: int,
        yearly_decay: float | None = None,
        **kwargs,
    ):
        super().__init__(len(source), **kwargs)
        self.scratch_root = Path(scratch_root)
        self.max_scratch_bytes = max_scratch_bytes
        self.index_path = None
        self.sampling_summary = None
        self.strata = [{"start": 0, "count": len(source), "samples": len(self)}]
        if yearly_decay is not None:
            self.index_path = date_market_index(source, requested_workers=requested_workers)
            index = np.load(self.index_path, mmap_mode="r")
            try:
                origin = date(1970, 1, 1)
                first = (origin + timedelta(days=int(index["group"][0]) // 256)).year
                last = (origin + timedelta(days=int(index["group"][-1]) // 256)).year
                strata = []
                for year in range(first, last + 1):
                    lower = (date(year, 1, 1) - origin).days * 256
                    upper = (date(year + 1, 1, 1) - origin).days * 256
                    start, stop = np.searchsorted(index["group"], [lower, upper])
                    if stop > start:
                        strata.append(
                            {"year": year, "start": int(start), "count": int(stop - start)}
                        )
                self.strata = annual_quotas(strata, len(self), yearly_decay)
                self.sampling_summary = self.strata
            finally:
                index._mmap.close()
        self.sampling_contract = {
            "version": SAMPLING_VERSION,
            "ordering": "global_sample_fisher_yates",
            "selection": "without_replacement_per_annual_cycle",
            "stable_subset": self.target_count < self.sample_count,
            "selected_samples": len(self),
            "max_scratch_bytes": max_scratch_bytes,
            "required_scratch_bytes": 8 * (
                len(self) + max(row["count"] for row in self.strata if row["samples"])
            ),
        }

    def __iter__(self):
        index = np.load(self.index_path, mmap_mode="r") if self.index_path is not None else None
        positions = shuffled_positions(
            self.strata,
            seed=self.seed,
            epoch=self.epoch,
            scratch_root=self.scratch_root,
            max_scratch_bytes=self.max_scratch_bytes,
            stable_subset=self.target_count < self.sample_count,
        )
        try:
            for position in positions:
                yield position if index is None else int(index["ordinal"][position])
        finally:
            positions.close()
            if index is not None:
                index._mmap.close()
