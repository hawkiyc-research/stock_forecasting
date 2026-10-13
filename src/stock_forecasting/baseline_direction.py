"""Dual supervised baseline heads, fixed loss references and persisted evaluation."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from stock_forecasting.baselines import _normalized_pinball_torch
from stock_forecasting.direction_runtime import DirectionMetrics
from stock_forecasting.models.direction import (
    classification_loss,
    direction_cost_features,
    direction_labels,
)


class DirectionalBaseline(nn.Module):
    """Keep the alpha decoder and attach both tasks to the trainable representation."""

    def __init__(self, base, config):
        super().__init__()
        self.base, self.config = base, config
        # Absolute stream moments retain market moves in the relative DLinear model.
        dimension = base.representation_dimension + 22
        self.absolute = nn.Linear(dimension, len(base.horizons) * 3)
        self.alpha = nn.Linear(dimension, len(base.horizons) * 3)
        self.register_buffer("references", torch.tensor([1.0, math.log(3), math.log(3)]))
        self.register_buffer("reference_count", torch.tensor(0, dtype=torch.long))

    def forward(self, sequences):
        return self.base(sequences)

    def forward_tasks(self, sequences, bands):
        hidden = self.base.encode(sequences)
        quantiles = self.base.decode(hidden)
        moments = torch.cat(
            (sequences.mean(2).flatten(1), sequences.std(2, unbiased=False).flatten(1)), 1
        )
        features = torch.cat((hidden, moments, direction_cost_features(bands).to(hidden.dtype)), 1)
        shape = (len(sequences), len(self.base.horizons), 3)
        return (
            quantiles,
            self.absolute(features).reshape(shape),
            self.alpha(features).reshape(shape),
        )

    def weights(self):
        return (
            self.references[0]
            / self.references[1:]
            * self.references.new_tensor([self.config.absolute_weight, self.config.alpha_weight])
        )

    def reference_state(self):
        return {
            "schema_version": 1,
            "samples": int(self.reference_count),
            "references": self.references.detach().cpu().tolist(),
        }

    def restore_references(self, state):
        values = state.get("references", [])
        if (
            state.get("schema_version") != 1
            or len(values) != 3
            or any(not math.isfinite(v) or v <= 0 for v in values)
            or type(state.get("samples")) is not int
            or state["samples"] < 1
        ):
            raise ValueError("Invalid baseline direction loss references")
        self.references.copy_(self.references.new_tensor(values))
        self.reference_count.fill_(state["samples"])


def forward_loss(model, batch, scales, device):
    if getattr(model, "prediction_task", "quantiles") != "quantiles":
        from stock_forecasting.baseline_signal import signal_forward_loss

        return signal_forward_loss(model, batch, scales, device)
    sequences = batch["sequences"].to(device, non_blocking=True)
    target = batch["target_alpha"].to(device, non_blocking=True)
    if not isinstance(model, DirectionalBaseline):
        q = model(sequences)
        pinball = _normalized_pinball_torch(q, target, scales)
        return q, None, pinball, (pinball,)
    bands = batch["direction_bands"].to(device, non_blocking=True)
    q, absolute, alpha = model.forward_tasks(sequences, bands)
    pinball = _normalized_pinball_torch(q, target, scales)
    losses = (
        pinball,
        classification_loss(
            absolute,
            direction_labels(batch["target_absolute"].to(device, non_blocking=True), bands),
        ),
        classification_loss(alpha, direction_labels(target, bands)),
    )
    total = losses[0] + model.weights()[0] * losses[1] + model.weights()[1] * losses[2]
    return q, (absolute, alpha), total, losses


def calibrate_references(model, source, config, runtime, scales):
    """Freeze initial-model losses from training only, with no optimizer updates."""
    import random

    from stock_forecasting.baseline_input import close_neural_loader, neural_loader

    if source.split != "train":
        raise ValueError("Baseline loss references must use training data only")
    count = min(len(source), config.model.direction.loss_reference_samples)
    rows = random.Random(config.data.calibration_seed).sample(range(len(source)), count)
    size = min(128, runtime["training_batch_size"])
    batches = [rows[start : start + size] for start in range(0, count, size)]
    loader = neural_loader(
        source,
        workers=runtime["loader_workers"],
        batch_sampler=batches,
        prefetch=runtime["prefetch_factor"],
        direction_costs=config.model.direction.costs,
    )
    was_training = model.training
    sums = torch.zeros(3, dtype=torch.float64, device="cuda")
    seen = 0
    try:
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            for batch in loader:
                _, _, _, losses = forward_loss(model, batch, scales, "cuda")
                length = len(batch["target_alpha"])
                sums += torch.stack(losses).double() * length
                seen += length
    finally:
        close_neural_loader(loader)
        model.train(was_training)
    if seen != count:
        raise RuntimeError("Baseline direction reference sample is incomplete")
    model.restore_references(
        {
            "schema_version": 1,
            "samples": seen,
            "references": (sums / seen).clamp_min(1e-8).cpu().tolist(),
        }
    )


class BaselineDirectionEvaluation:
    """Stream shared metrics and model-only probability/class arrays in bounded chunks."""

    def __init__(self, horizons, costs, count, *, head=None, output=None, source="supervised"):
        self.metrics = DirectionMetrics(horizons)
        self.head, self.costs, self.count, self.source = head, costs, count, source
        self.position = 0
        self.probabilities = self.classes = None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            if source == "supervised":
                self.probabilities = np.lib.format.open_memmap(
                    output / "direction-probabilities.npy",
                    mode="w+",
                    dtype="float32",
                    shape=(count, 2, len(horizons), 3),
                )
            else:
                self.classes = np.lib.format.open_memmap(
                    output / "direction-classes.npy",
                    mode="w+",
                    dtype="uint8",
                    shape=(count, 2, len(horizons)),
                )

    def append(
        self,
        probabilities,
        targets,
        bands,
        markets,
        dates,
        *,
        pinball=0.0,
        classes=None,
        log_probabilities=None,
    ):
        from stock_forecasting.baseline_direction_data import numpy_direction_labels

        truth = np.stack([numpy_direction_labels(targets[:, i], bands) for i in range(2)], 1)
        size = len(truth)
        if self.position + size > self.count:
            raise ValueError("Direction evaluation exceeds its full population")
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if (
                probabilities.shape != (*truth.shape, 3)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)
            ):
                raise ValueError("Invalid baseline direction probabilities")
            classes = probabilities.argmax(-1)
            if log_probabilities is not None:
                log_probabilities = np.asarray(log_probabilities, dtype=np.float32)
                if (
                    log_probabilities.shape != probabilities.shape
                    or not np.isfinite(log_probabilities).all()
                    or not np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6)
                ):
                    raise ValueError("Invalid baseline direction log probabilities")
            else:
                log_probabilities = np.log(np.maximum(probabilities, 1e-30))
            losses = -np.take_along_axis(log_probabilities, truth[..., None], -1)[..., 0]
        else:
            classes = np.asarray(classes)
            if classes.shape != truth.shape or not np.isin(classes, [0, 1, 2]).all():
                raise ValueError("Invalid deterministic direction classes")
            losses = None
        self.metrics.append_arrays(truth, classes, losses, markets, dates, pinball=pinball)
        stop = self.position + size
        if self.probabilities is not None:
            self.probabilities[self.position : stop] = probabilities
        if self.classes is not None:
            self.classes[self.position : stop] = classes
        self.position = stop

    def finish(self):
        if self.position != self.count:
            raise ValueError("Direction evaluation did not enumerate the full split")
        result = self.metrics.finish(self.head, self.costs)
        result["prediction_source"] = self.source
        result["task_order"] = ["absolute", "alpha"]
        result["samples"] = self.position
        return result

    def close(self):
        for array in (self.probabilities, self.classes):
            if array is not None:
                array.flush()
                array._mmap.close()


def reference_head(state, parameters):
    """Non-neural learners use the same frozen reference-weight formula."""
    weights = np.asarray(state["references"])[0] / np.asarray(state["references"])[1:]
    weights *= [parameters["absolute_weight"], parameters["alpha_weight"]]
    return SimpleNamespace(reference_state=lambda: state, weights=lambda: torch.tensor(weights))
