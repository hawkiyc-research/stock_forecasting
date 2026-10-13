"""Cost-free single-task neural baselines over lazy, unchanged bar windows."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from stock_forecasting.baseline_signal_contract import log_thresholds


def fixed_return_labels(targets, threshold):
    """Keep equality neutral and compare the actual log-return label space."""
    values = np.asarray(targets, dtype=np.float32)
    lower, upper = np.asarray(log_thresholds(threshold), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Direction returns must be finite")
    return np.where(values < lower, 0, np.where(values > upper, 2, 1)).astype(np.uint8)


def signal_array_batch(source, indices, *, include_metadata=True):
    """Compute adjusted next-open to horizon-close labels without any cost inputs."""
    ordinals = np.asarray(indices, dtype=np.int64)
    if ordinals.ndim != 1 or not len(ordinals):
        raise ValueError("Signal batches need a bounded nonempty index vector")
    ordinals = np.where(ordinals < 0, ordinals + len(source), ordinals)
    if (ordinals < 0).any() or (ordinals >= len(source)).any():
        raise IndexError("Signal sample index is outside the split")
    ranges = np.searchsorted(source._range_ends, ordinals, side="right")
    cutoffs = source._range_starts[ranges] + ordinals - source._range_begins[ranges]
    symbols = source._range_symbols[ranges]
    result = {
        "asset": np.empty((len(indices), source.window_size, 5), dtype=np.float64),
        "benchmark": np.empty((len(indices), source.window_size, 5), dtype=np.float64),
        "targets": np.empty((len(indices), len(source.horizons)), dtype=np.float64),
        "absolute_targets": np.empty((len(indices), len(source.horizons)), dtype=np.float32),
        "horizons": source.horizons,
    }
    if include_metadata:
        result["metadata"] = [None] * len(indices)
    for symbol in np.unique(symbols):
        selected = np.flatnonzero(symbols == symbol)
        batch = source.array_batch(
            ordinals[selected], include_timestamps=False, include_metadata=include_metadata
        )
        bars = source._numeric_symbol(symbol)
        entry_position = cutoffs[selected] + 1
        entry = bars["raw"][entry_position, 0] * bars["price_factor"][entry_position]
        exits = cutoffs[selected, None] + np.asarray(source.horizons)[None]
        gross = bars["adjusted_close"][exits] / entry[:, None]
        if not np.isfinite(gross).all() or (gross <= 0).any():
            raise ValueError("Absolute returns require finite positive price ratios")
        result["absolute_targets"][selected] = np.log(gross).astype(np.float32)
        for key in ("asset", "benchmark", "targets"):
            result[key][selected] = batch[key]
        if include_metadata:
            for offset, metadata in zip(selected, batch["metadata"], strict=True):
                result["metadata"][offset] = metadata
    return result


class SignalBatchDataset(Dataset):
    """One bounded vectorized batch, with no materialized window dataset."""

    def __init__(self, source, *, neural=False, metadata=False, task="alpha_point"):
        self.source, self.neural, self.metadata, self.task = source, neural, metadata, task

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        from stock_forecasting.baselines import baseline_arrays_from_windows

        if self.neural and self.task == "alpha_point":
            from stock_forecasting.baseline_input import NeuralBatchDataset

            return NeuralBatchDataset(self.source, metadata=self.metadata).__getitems__(indices)
        arrays = (
            signal_array_batch(
                self.source, indices, include_metadata=not self.neural or self.metadata
            )
            if self.task == "absolute_direction"
            else self.source.array_batch(indices, include_timestamps=False)
        )
        if not self.neural:
            return baseline_arrays_from_windows(arrays)
        # Neural jobs do not calculate unused tree/rule indicator features.
        streams = []
        for key in ("asset", "benchmark"):
            values = arrays[key].astype(np.float32)
            reference = np.maximum(values[:, -1:, 3:4], 1e-12)
            values[:, :, :4] = np.log(np.maximum(values[:, :, :4], 1e-12) / reference)
            volume = np.log1p(np.maximum(values[:, :, 4], 0.0))
            scale = volume.std(axis=1, keepdims=True)
            values[:, :, 4] = (volume - volume.mean(axis=1, keepdims=True)) / np.where(
                scale > 1e-6, scale, 1.0
            )
            streams.append(values)
        result = {
            "sequences": torch.from_numpy(np.stack(streams, axis=1)),
            "target_alpha": torch.from_numpy(arrays["targets"].astype(np.float32)),
        }
        if self.task == "absolute_direction":
            result["target_absolute"] = torch.from_numpy(arrays["absolute_targets"])
        if self.metadata:
            metadata = arrays["metadata"]
            result.update(
                symbols=[row[0] for row in metadata],
                cutoff_at=[row[2].isoformat() for row in metadata],
                markets=[str(row[3]["market"]) for row in metadata],
                asset_types=[str(row[3]["asset_type"]) for row in metadata],
                providers=[str(row[3]["provider"]) for row in metadata],
            )
        return result


def signal_neural_loader(
    source,
    *,
    task,
    workers,
    batch_size=None,
    batch_sampler=None,
    prefetch=2,
    metadata=False,
    **_unused,
):
    from stock_forecasting.baseline_input import collate_neural_batch
    from stock_forecasting.baseline_storage import loader_options

    options = (
        {"batch_sampler": batch_sampler}
        if batch_sampler is not None
        else {"batch_size": batch_size, "shuffle": False, "drop_last": False}
    )
    return DataLoader(
        SignalBatchDataset(source, neural=True, metadata=metadata, task=task),
        **options,
        collate_fn=collate_neural_batch,
        pin_memory=True,
        **loader_options(workers, prefetch, persistent=True),
    )


class SignalBaseline(nn.Module):
    """An independent encoder/head optimized for exactly one supervised task."""

    def __init__(self, base, task):
        super().__init__()
        self.base, self.task = base, dict(task)
        self.horizons = base.horizons
        self.prediction_task = task["prediction_task"]
        # The former quantile decoder is not part of this model's objective.
        for name in ("head", "linear"):
            decoder = getattr(base, name, None)
            if decoder is not None:
                for parameter in decoder.parameters():
                    parameter.requires_grad_(False)
        width = base.representation_dimension + 20
        outputs = len(self.horizons) * (3 if self.prediction_task == "absolute_direction" else 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, outputs))

    def forward(self, sequences):
        hidden = self.base.encode(sequences)
        moments = torch.cat(
            (sequences.mean(2).flatten(1), sequences.std(2, unbiased=False).flatten(1)), 1
        )
        output = self.head(torch.cat((hidden, moments), 1))
        shape = (len(sequences), len(self.horizons))
        return output.reshape(
            (*shape, 3) if self.prediction_task == "absolute_direction" else shape
        )


def signal_forward_loss(model, batch, scales, device):
    prediction = model(batch["sequences"].to(device, non_blocking=True))
    if model.prediction_task == "alpha_point":
        target = batch["target_alpha"].to(device, non_blocking=True)
        loss = F.smooth_l1_loss(
            prediction / scales, target / scales, beta=model.task["point_loss_beta"]
        )
    else:
        target = batch["target_absolute"].to(device, non_blocking=True)
        lower, upper = log_thresholds(model.task["neutral_return_threshold"])
        labels = torch.where(target < lower, 0, torch.where(target > upper, 2, 1))
        loss = F.cross_entropy(prediction.flatten(0, 1), labels.flatten())
    return prediction, None, loss, (loss,)


def append_label_audit(audit, targets, markets, threshold, horizons):
    """Stream exact train class counts without collecting labels or windows."""
    labels = fixed_return_labels(targets, threshold)
    market_array = np.asarray(markets)
    for market in ["overall", *np.unique(market_array).tolist()]:
        selected = labels if market == "overall" else labels[market_array == market]
        rows = audit.setdefault(market, {})
        for i, horizon in enumerate(horizons):
            counts = np.bincount(selected[:, i], minlength=3)
            previous = np.asarray(rows.get(str(horizon), [0, 0, 0]), dtype=np.int64)
            rows[str(horizon)] = (previous + counts).tolist()
    return audit


def validate_label_audit(audit, task, count, horizons):
    """Missing train classes are fatal; extreme imbalance is explicitly reported."""
    from stock_forecasting.baseline_signal_contract import validate_class_audit

    settings = audit.get("balance_warning_thresholds", {})
    return validate_class_audit(audit, task, count, horizons, **settings)
