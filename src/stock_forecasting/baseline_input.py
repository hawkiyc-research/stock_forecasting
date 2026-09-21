"""Fixed-shape neural baseline batches without per-window tensor reconstruction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stock_forecasting.baseline_storage import loader_options


class NeuralBatchDataset(Dataset):
    """Keep windows lazy and emit only tensors consumed by neural baselines."""

    def __init__(self, source, *, metadata=False):
        self.source = source
        self.metadata = metadata

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        arrays = self.source.array_batch(
            indices, include_metadata=self.metadata, include_timestamps=False
        )
        streams = []
        for key in ("asset", "benchmark"):
            # Preserve the existing float32 preprocessing and reduction order.
            values = arrays[key].astype(np.float32)
            reference = np.maximum(values[:, -1:, 3:4], 1e-12)
            values[:, :, :4] = np.log(np.maximum(values[:, :, :4], 1e-12) / reference)
            volume = np.log1p(np.maximum(values[:, :, 4], 0.0))
            scale = volume.std(axis=1, keepdims=True)
            values[:, :, 4] = (volume - volume.mean(axis=1, keepdims=True)) / np.where(
                scale > 1e-6, scale, 1.0
            )
            streams.append(values)
        batch = {
            "sequences": torch.from_numpy(np.stack(streams, axis=1)),
            "target_alpha": torch.from_numpy(arrays["targets"].astype(np.float32)),
        }
        if self.metadata:
            metadata = arrays["metadata"]
            batch.update(
                symbols=[row[0] for row in metadata],
                cutoff_at=[row[2].isoformat() for row in metadata],
                markets=[str(row[3]["market"]) for row in metadata],
                asset_types=[str(row[3]["asset_type"]) for row in metadata],
                providers=[str(row[3]["provider"]) for row in metadata],
            )
        return batch


def collate_neural_batch(batch):
    """The dataset already assembled a contiguous batch before pinning."""
    return batch


def neural_loader(
    source, *, workers, batch_size=None, batch_sampler=None, prefetch=2, metadata=False
):
    options = (
        {"batch_sampler": batch_sampler}
        if batch_sampler is not None
        else {"batch_size": batch_size, "shuffle": False, "drop_last": False}
    )
    return DataLoader(
        NeuralBatchDataset(source, metadata=metadata),
        **options,
        collate_fn=collate_neural_batch,
        pin_memory=True,
        **loader_options(workers, prefetch, persistent=True),
    )


def close_neural_loader(loader):
    """Release the active pool at a phase boundary, including queued batches."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            if not iterator._shutdown:
                # Stop submission, then drain only the already-prefetched tasks.
                # Closing IPC while a worker exports tensor storage can abort its
                # C++ background thread. Discarded batches never advance the
                # training sample cursor. These internals are pinned to torch 2.9.1.
                pending = iterator._send_idx - iterator._rcvd_idx
                capacity = loader.num_workers * loader.prefetch_factor
                if not 0 <= pending <= capacity:
                    raise RuntimeError("Neural loader exceeded its bounded prefetch contract")
                iterator._sampler_iter = iter(())
                for _ in range(pending):
                    next(iterator)
        finally:
            try:
                iterator._shutdown_workers()
            finally:
                loader._iterator = None
