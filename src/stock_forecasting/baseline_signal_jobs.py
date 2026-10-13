"""Single-task baseline evaluation and resumable CPU models using full populations."""

from __future__ import annotations

import json
import math
import pickle
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from stock_forecasting.baseline_signal import fixed_return_labels, signal_forward_loss
from stock_forecasting.baseline_storage import SIGNAL_NAMES, open_tabular
from stock_forecasting.baselines import RULE_BASELINE_NAMES
from stock_forecasting.data.manifest import atomic_write_json
from stock_forecasting.evaluation_store import CHUNK_ROWS


def _store(output, count, horizons, scales, task):
    from stock_forecasting.signal_evaluation import SignalEvaluationStore

    return SignalEvaluationStore(
        Path(output),
        count,
        horizons,
        scales,
        task["prediction_task"],
        beta=task.get("point_loss_beta", 1.0),
        neutral_threshold=task.get("neutral_return_threshold", 0.01),
    )


@torch.inference_mode()
def neural_score(model, loader, horizons, scales, output=None):
    """Full ordered traversal, with bounded disk-backed metrics and no sampling."""
    original = model.training
    model.eval()
    scale_tensor = torch.as_tensor(scales, device=next(model.parameters()).device)
    with ExitStack() as stack:
        if output is None:
            output = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="baseline-signal-validation-")
            )
        store = _store(output, len(loader.dataset), horizons, scales, model.task)
        if model.prediction_task == "absolute_direction":
            # This evaluation owns only a scratch copy; inputs/ keeps the canonical labels.
            stack.callback((Path(output) / "absolute_targets.npy").unlink, missing_ok=True)
        stack.callback(store.close)
        try:
            for batch in loader:
                prediction, _, _, _ = signal_forward_loss(
                    model, batch, scale_tensor, scale_tensor.device
                )
                store.append(
                    batch["target_alpha"].numpy(),
                    prediction.float().cpu().numpy(),
                    symbols=batch["symbols"],
                    dates=batch["cutoff_at"],
                    markets=batch["markets"],
                    asset_types=batch["asset_types"],
                    providers=batch["providers"],
                    target_absolute=batch.get("target_absolute", None).numpy()
                    if "target_absolute" in batch
                    else None,
                )
            return store.finish()
        finally:
            model.train(original)


def tabular_score(arrays, predict, horizons, scales, task, output=None):
    """Stream every tabular row through the same evaluator as the main model."""
    with ExitStack() as stack:
        if output is None:
            output = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="baseline-signal-validation-")
            )
        store = _store(output, len(arrays["targets"]), horizons, scales, task)
        if task["prediction_task"] == "absolute_direction":
            stack.callback((Path(output) / "absolute_targets.npy").unlink, missing_ok=True)
        stack.callback(store.close)
        for start in range(0, len(arrays["targets"]), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(arrays["targets"]))
            metadata = arrays["metadata"][start:stop]
            store.append(
                arrays["targets"][start:stop],
                predict(start, stop),
                symbols=metadata["symbol"].tolist(),
                dates=metadata["date"].tolist(),
                markets=metadata["market"].tolist(),
                asset_types=metadata["asset_type"].tolist(),
                providers=metadata["provider"].tolist(),
                target_absolute=arrays["absolute_targets"][start:stop]
                if task["prediction_task"] == "absolute_direction"
                else None,
            )
        return store.finish()


def _full_fixed_labels(arrays, horizon, threshold):
    """Keep only one compact class column in RAM for native parallel tree fitting."""
    result = np.empty(len(arrays["targets"]), dtype=np.uint8)
    for start in range(0, len(result), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(result))
        result[start:stop] = fixed_return_labels(
            arrays["absolute_targets"][start:stop, horizon], threshold
        )
    return result


def train_signal_gbdt(root, directory, seed, parameters, horizons, scales):
    """Warm-start all-data HistGBDT with persisted per-horizon fitting cursors."""
    from stock_forecasting.baseline_build import _save_pickle

    arrays = {
        split: open_tabular(root / "inputs", split) for split in ("train", "validation", "test")
    }
    task, settings = parameters["signal_task"], parameters["gbdt"]
    classification = task["prediction_task"] == "absolute_direction"
    constructor = (
        HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    )
    models = [
        constructor(
            **({"loss": "log_loss"} if classification else {"loss": "absolute_error"}),
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
    best, stale, iteration, active, cursor, finished = math.inf, 0, 0, None, 0, False
    resume = directory / "resume.pkl"
    if resume.is_file():
        with resume.open("rb") as stream:
            state = pickle.load(stream)
        if state.get("signal_task") != task or state.get("schema_version") != 1:
            raise ValueError("GBDT resume does not match the independent task")
        models, best, stale, iteration, active, cursor, finished = (
            state[key]
            for key in ("models", "best", "stale", "iteration", "active", "cursor", "finished")
        )

    def save():
        _save_pickle(
            resume,
            {
                "schema_version": 1,
                "signal_task": task,
                "models": models,
                "best": best,
                "stale": stale,
                "iteration": iteration,
                "active": active,
                "cursor": cursor,
                "finished": finished,
            },
        )

    def prediction(split, start, stop):
        features = arrays[split]["features"][start:stop]
        if not classification:
            return np.column_stack([model.predict(features) for model in models])
        probability = np.zeros((len(features), len(horizons), 3), dtype=np.float32)
        for h, model in enumerate(models):
            probability[:, h, model.classes_.astype(int)] = model.predict_proba(features)
        # The common evaluator consumes logits, not class probabilities.
        return np.log(np.maximum(probability, 1e-30))

    def score(split, output=None):
        return tabular_score(
            arrays[split], lambda a, b: prediction(split, a, b), horizons, scales, task, output
        )

    def training_diagnostics():
        from stock_forecasting.signal_evaluation import SignalMoments

        # This bounded diagnostic never limits the complete population used by fit().
        count = min(len(arrays["train"]["targets"]), 1024)
        rows = np.random.default_rng(parameters["calibration_seed"]).choice(
            len(arrays["train"]["targets"]), size=count, replace=False
        )
        features = arrays["train"]["features"][rows]
        if classification:
            probabilities = np.zeros((count, len(horizons), 3), dtype=np.float64)
            for h, model in enumerate(models):
                probabilities[:, h, model.classes_.astype(int)] = model.predict_proba(features)
            p = np.log(np.maximum(probabilities, 1e-30))
            target = arrays["train"]["absolute_targets"][rows]
        else:
            p = np.column_stack([model.predict(features) for model in models])
            target = arrays["train"]["targets"][rows]
        moments = SignalMoments(
            horizons,
            scales,
            task["prediction_task"],
            task.get("point_loss_beta", 1),
            task.get("neutral_return_threshold", 0.01),
        )
        moments.add(target, p)
        return {
            "scope": "fixed_train_diagnostic_not_training_subsampling",
            "samples": count,
            "metrics": moments.result(),
        }

    checkpoints = list(
        range(settings["validation_every"], settings["max_iter"] + 1, settings["validation_every"])
    )
    if not checkpoints or checkpoints[-1] != settings["max_iter"]:
        checkpoints.append(settings["max_iter"])
    for target_iteration in checkpoints:
        if finished or target_iteration <= iteration:
            continue
        if active != target_iteration:
            active, cursor = target_iteration, 0
        for h in range(cursor, len(horizons)):
            allocation = root / "live-resources.json"
            threads = (
                json.loads(allocation.read_text())["gbdt_threads"] if allocation.is_file() else 1
            )
            model = models[h]
            model.set_params(max_iter=target_iteration)
            labels = (
                _full_fixed_labels(arrays["train"], h, task["neutral_return_threshold"])
                if classification
                else arrays["train"]["targets"][:, h]
            )
            if classification and len(np.unique(labels)) != 3:
                raise ValueError(
                    f"GBDT training horizon {horizons[h]} is missing a direction class"
                )
            with threadpool_limits(limits=threads):
                model.fit(arrays["train"]["features"], labels)
            del labels
            cursor = h + 1
            save()
            atomic_write_json(
                directory / "progress.json",
                {
                    "phase": "fitting",
                    "iteration": target_iteration,
                    "completed_models": cursor,
                    "total_models": len(horizons),
                    "signal_task": task,
                },
            )
        metrics = score("validation")
        value = metrics["aggregate"]["selection_score"]
        improved = value < best - parameters["early_stopping_min_delta"]
        best, stale = (value, 0) if improved else (best, stale + 1)
        if improved:
            _save_pickle(
                directory / "model.pkl",
                {
                    "models": models,
                    "horizons": horizons,
                    "scales": scales,
                    "seed": seed,
                    "signal_task": task,
                    "objective": parameters["objectives"]["gbdt"],
                },
            )
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "phase": "validation",
                        "iteration": target_iteration,
                        "selection_loss": value,
                        "objective": parameters["objectives"]["gbdt"],
                        "training": training_diagnostics(),
                    }
                )
                + "\n"
            )
        finished = stale >= parameters["early_stopping_patience_evaluations"]
        iteration, active, cursor = target_iteration, None, 0
        save()
    with (directory / "model.pkl").open("rb") as stream:
        models = pickle.load(stream)["models"]
    validation = score("validation", directory / "validation")
    atomic_write_json(directory / "validation-metrics.json", validation)
    return score("test", directory / "test")


def train_signal_rules(root, directory, horizons, scales, parameters):
    """Retain every existing rule; point fitting uses complete train residuals only."""
    task = parameters["signal_task"]
    classification = task["prediction_task"] == "absolute_direction"
    arrays = {
        split: open_tabular(root / "inputs", split) for split in ("train", "validation", "test")
    }
    result = {}
    multipliers = np.asarray(horizons, dtype=np.float64) / 5
    for name in RULE_BASELINE_NAMES:
        child = directory / name
        child.mkdir(exist_ok=True)
        completed = child / "result.json"
        if completed.is_file():
            result[name] = json.loads(completed.read_text())
            continue
        offsets = np.zeros(len(horizons), dtype=np.float64)
        if not classification and name != "zero_return":
            for h, multiplier in enumerate(multipliers):
                residual = np.array(arrays["train"]["targets"][:, h], dtype=np.float64)
                if name == "always_buy":
                    offsets[h] = max(
                        float(np.median(residual, overwrite_input=True)), 0.25 * scales[h]
                    )
                    continue
                else:
                    residual -= arrays["train"]["signals"][:, SIGNAL_NAMES.index(name)] * multiplier
                offsets[h] = np.median(residual, overwrite_input=True)
        objective = (
            (
                "constant_zero_alpha"
                if name == "zero_return"
                else "constant_positive_alpha_train_median_floor"
                if name == "always_buy"
                else parameters["objectives"]["rules"]
            )
            if not classification
            else (
                "constant_up_class"
                if name == "always_buy"
                else "constant_neutral_class"
                if name == "zero_return"
                else parameters["objectives"]["rules"]
            )
        )
        atomic_write_json(
            child / "model.json",
            {
                "rule": name,
                "signal_task": task,
                "objective": objective,
                "horizons": horizons,
                "residual_median": offsets.tolist(),
                **(
                    {
                        "probability_source": "deterministic_rule_with_numeric_floor",
                        "logit_margin": 30.0,
                    }
                    if classification
                    else {}
                ),
            },
        )

        def prediction(split, start, stop, *, name=name, offsets=offsets):
            if classification:
                if name in ("always_buy", "zero_return"):
                    labels = np.full(
                        (stop - start, len(horizons)),
                        2 if name == "always_buy" else 1,
                        dtype=np.uint8,
                    )
                else:
                    values = (
                        arrays[split]["absolute_signals"][
                            start:stop, SIGNAL_NAMES.index(name), None
                        ]
                        * multipliers
                    )
                    labels = fixed_return_labels(values, task["neutral_return_threshold"])
                logits = np.full((*labels.shape, 3), -30.0, dtype=np.float32)
                np.put_along_axis(logits, labels[..., None], 0.0, axis=-1)
                return logits
            if name in ("always_buy", "zero_return"):
                return np.broadcast_to(offsets, (stop - start, len(horizons)))
            return (
                arrays[split]["signals"][start:stop, SIGNAL_NAMES.index(name), None] * multipliers
                + offsets
            )

        validation = tabular_score(
            arrays["validation"],
            lambda a, b: prediction("validation", a, b),
            horizons,
            scales,
            task,
            child / "validation",
        )
        if classification:
            validation.update(
                probability_source="deterministic_rule_with_numeric_floor",
                logit_margin=30.0,
                probability_floor_before_normalization=math.exp(-30),
                calibration_comparison="not_a_learned_or_calibrated_probability_model",
            )
        atomic_write_json(child / "validation-metrics.json", validation)
        metrics = tabular_score(
            arrays["test"],
            lambda a, b: prediction("test", a, b),
            horizons,
            scales,
            task,
            child / "test",
        )
        if classification:
            metrics.update(
                probability_source="deterministic_rule_with_numeric_floor",
                logit_margin=30.0,
                probability_floor_before_normalization=math.exp(-30),
                calibration_comparison="not_a_learned_or_calibrated_probability_model",
            )
        result[name] = {
            "state": "complete",
            "source": "prebuilt_full_train_rule",
            "metrics": metrics,
        }
        atomic_write_json(completed, result[name])
    return result
