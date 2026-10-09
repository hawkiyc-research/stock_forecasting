"""Full-population tree classification helpers with fixed train-only references."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from stock_forecasting.baseline_direction_data import numpy_direction_labels
from stock_forecasting.evaluation_store import CHUNK_ROWS, Moments


class ConstantDirectionClassifier:
    """A one-class training population has no decision boundary to fit."""

    def __init__(self, label):
        self.classes_ = np.asarray([label], dtype=np.int64)

    def predict_proba(self, values):
        return np.ones((len(values), 1), dtype=np.float64)


def make_classifiers(horizons, settings, seed):
    return [
        [
            HistGradientBoostingClassifier(
                learning_rate=settings["learning_rate"],
                max_iter=1,
                max_leaf_nodes=settings["max_leaf_nodes"],
                l2_regularization=settings["l2_regularization"],
                early_stopping=False,
                warm_start=True,
                random_state=seed,
            )
            for _ in horizons
        ]
        for _ in range(2)
    ]


def full_labels(arrays, task, horizon):
    """Only one compact label column is live during each native parallel fit."""
    target = arrays["absolute_targets" if task == 0 else "targets"]
    labels = np.empty(len(target), dtype=np.uint8)
    for start in range(0, len(target), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(target))
        labels[start:stop] = numpy_direction_labels(
            target[start:stop, horizon : horizon + 1], arrays["direction_bands"][start:stop]
        )[:, 0]
    return labels


def predict_probabilities(classifiers, features):
    result = np.zeros((len(features), 2, len(classifiers[0]), 3), dtype=np.float32)
    for task, models in enumerate(classifiers):
        for horizon, model in enumerate(models):
            result[:, task, horizon, model.classes_.astype(int)] = model.predict_proba(features)
    return result


def initial_references(arrays, horizons, scales, parameters, seed):
    """Measure a smoothed intercept-only reference on a fixed training sample."""
    import random

    size = min(len(arrays["targets"]), parameters["loss_reference_samples"])
    rows = random.Random(seed).sample(range(len(arrays["targets"])), size)
    y = np.asarray(arrays["targets"][rows])
    q = np.broadcast_to(np.quantile(y, [0.1, 0.5, 0.9], axis=0).T, (*y.shape, 3))
    moments = Moments(horizons, scales)
    moments.add(y, q)
    references = [moments.result()["aggregate"]["selection_score"]]
    for field in ("absolute_targets", "targets"):
        labels = numpy_direction_labels(arrays[field][rows], arrays["direction_bands"][rows])
        total = 0.0
        for horizon in range(len(horizons)):
            counts = np.bincount(labels[:, horizon], minlength=3) + 1
            total += -np.log((counts / counts.sum())[labels[:, horizon]]).mean()
        references.append(total / len(horizons))
    # Perfectly constant labels still need a finite reference for objective scaling.
    return {"schema_version": 1, "samples": size, "references": [max(v, 1e-8) for v in references]}
