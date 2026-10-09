"""Bounded train-only loss calibration and streaming direction evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecasting.models.direction import DIRECTION_CLASSES, direction_labels


def calibrate_direction_references(bundle, config, dataset, device, workers, batch_plan) -> dict:
    """Fit once before updates, preserving RNG; resume restores the saved values."""
    from stock_forecasting.training import (
        ModelBatchCollator,
        _autocast_context,
        _loader_process_options,
        forward_batch,
        iter_device_batches,
    )

    if dataset.split != "train":
        raise ValueError("Direction loss references must use training data only")
    count = min(len(dataset), config.model.direction.loss_reference_samples)
    indices = random.Random(config.data.calibration_seed).sample(range(len(dataset)), count)
    loader = DataLoader(
        dataset,
        sampler=indices,
        batch_size=min(batch_plan.training_batch_size, 128),
        collate_fn=ModelBatchCollator.for_model(config, training=False),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.data.calibration_seed),
        **_loader_process_options(workers, persistent=False),
    )
    was_training = bundle.model.training
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    seen = 0
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        bundle.model.eval()
        # Kronos retains rotary tensors across forwards. They must remain usable
        # by the subsequent training pass, rather than becoming inference tensors.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for batch in iter_device_batches(loader, device):
                with _autocast_context(config, device):
                    output = forward_batch(bundle, batch, config, device)
                size = len(batch["symbols"])
                sums += (
                    torch.stack(
                        [
                            output.pinball_loss,
                            output.absolute_direction_loss,
                            output.alpha_direction_loss,
                        ]
                    ).double()
                    * size
                )
                seen += size
    finally:
        bundle.model.train(was_training)
    if seen != count:
        raise RuntimeError("Direction reference calibration did not consume its bounded sample")
    state = {"schema_version": 1, "samples": seen, "references": (sums / seen).cpu().tolist()}
    bundle.model.direction_head.restore_references(state)
    return bundle.model.direction_head.reference_state()


class DirectionMetrics:
    """Retain only confusion/count/loss sums per horizon, market and month."""

    def __init__(self, horizons):
        self.horizons = list(horizons)
        self.slices = {}
        self.pinball_sum = 0.0
        self.samples = 0

    def append(self, output, batch):
        logits = torch.stack([output.absolute_direction_logits, output.alpha_direction_logits], 1)
        labels = torch.stack(
            [
                direction_labels(batch[name], batch["direction_bands"])
                for name in ("target_absolute", "target_alpha")
            ],
            1,
        )
        prediction = logits.argmax(-1)
        ce = -logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        values = torch.stack([labels.float(), prediction.float(), ce], -1).detach().cpu().numpy()
        truth, predicted, losses = (
            values[..., 0].astype(int),
            values[..., 1].astype(int),
            values[..., 2],
        )
        size = len(truth)
        self.samples += size
        self.pinball_sum += float(output.pinball_loss) * size
        groups = {"overall": np.arange(size)}
        for key, entries in (
            ("market", batch["markets"]),
            ("month", [str(v)[:7] for v in batch["cutoff_at"]]),
        ):
            entries = np.asarray(entries)
            for entry in np.unique(entries):
                groups[f"{key}/{entry}"] = np.flatnonzero(entries == entry)
        for name, indices in groups.items():
            state = self.slices.setdefault(
                name,
                {
                    "confusion": np.zeros((2, len(self.horizons), 3, 3), dtype=np.int64),
                    "ce": np.zeros((2, len(self.horizons)), dtype=np.float64),
                    "joint": np.zeros((len(self.horizons), 3), dtype=np.int64),
                },
            )
            offsets = np.arange(2 * len(self.horizons)).reshape(1, 2, -1) * 9
            encoded = offsets + truth[indices] * 3 + predicted[indices]
            state["confusion"] += np.bincount(
                encoded.ravel(), minlength=2 * len(self.horizons) * 9
            ).reshape(state["confusion"].shape)
            state["ce"] += losses[indices].sum(0)
            joint = (predicted[indices, 0] == 2) & (predicted[indices, 1] == 2)
            state["joint"][:, 0] += joint.sum(0)
            state["joint"][:, 1] += (joint & (truth[indices, 0] == 2)).sum(0)
            state["joint"][:, 2] += (
                joint & (truth[indices, 0] == 2) & (truth[indices, 1] == 2)
            ).sum(0)

    @staticmethod
    def summarize(matrix, ce):
        support, predictions = matrix.sum(1), matrix.sum(0)
        correct = np.diag(matrix)
        total = int(support.sum())
        recall = np.divide(correct, support, out=np.zeros(3), where=support > 0)
        precision = np.divide(correct, predictions, out=np.zeros(3), where=predictions > 0)
        f1 = np.divide(
            2 * correct, support + predictions, out=np.zeros(3), where=support + predictions > 0
        )
        return {
            "count": total,
            "accuracy": float(correct.sum() / max(total, 1)),
            "balanced_accuracy": float(recall[support > 0].mean()) if total else 0.0,
            "macro_f1": float(f1.mean()),
            "majority_accuracy": float(support.max() / max(total, 1)),
            "cross_entropy": float(ce / max(total, 1)),
            "confusion": matrix.tolist(),
            "classes": {
                name: {
                    "support": int(support[i]),
                    "predicted": int(predictions[i]),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                }
                for i, name in enumerate(DIRECTION_CLASSES)
            },
            "up_signal_fraction": float(predictions[2] / max(total, 1)),
        }

    def finish(self, head, costs):
        if not self.samples:
            raise ValueError("Direction evaluation cannot be empty")
        result = {
            "schema_version": 1,
            "class_order": list(DIRECTION_CLASSES),
            "cost_scenario": costs.model_dump(),
            "loss_reference": head.reference_state(),
        }
        for name, state in self.slices.items():
            summary = {}
            for i, task in enumerate(("absolute", "alpha")):
                summary[task] = {
                    "aggregate": self.summarize(state["confusion"][i].sum(0), state["ce"][i].sum()),
                    "horizons": {
                        str(h): self.summarize(state["confusion"][i, j], state["ce"][i, j])
                        for j, h in enumerate(self.horizons)
                    },
                }
            summary["joint_up"] = {
                str(h): {
                    "signals": int(row[0]),
                    "absolute_up_precision": float(row[1] / max(row[0], 1)),
                    "both_up_precision": float(row[2] / max(row[0], 1)),
                }
                for h, row in zip(self.horizons, state["joint"], strict=True)
            }
            if "/" in name:
                group, key = name.split("/", 1)
                result.setdefault(group, {})[key] = summary
            else:
                result[name] = summary
        weights = head.weights().detach().cpu().tolist()
        result["pinball_loss"] = self.pinball_sum / self.samples
        result["selection_loss"] = result["pinball_loss"] + sum(
            weights[i] * result["overall"][task]["aggregate"]["cross_entropy"]
            for i, task in enumerate(("absolute", "alpha"))
        )
        return result
