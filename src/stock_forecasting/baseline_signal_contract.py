"""Dependency-free contracts for independent, cost-free baseline tasks."""

from __future__ import annotations

import copy
import math


def signal_task(selection):
    """Read only label/objective semantics, never the main model's capacity."""
    supplied = selection.get("baseline_task", {})
    if not isinstance(supplied, dict):
        raise ValueError("baseline_task must be a task-parameter object")
    task = supplied.get("prediction_task", "quantiles")
    if task == "quantiles":
        return {"prediction_task": task}
    if task == "alpha_point":
        beta = supplied.get("point_loss_beta", 1.0)
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError("point_loss_beta must be finite and positive")
        return {"prediction_task": task, "point_loss_beta": float(beta)}
    if task == "absolute_direction":
        threshold = supplied.get("neutral_return_threshold", 0.01)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 < threshold < 1
        ):
            raise ValueError("neutral_return_threshold must be between zero and one")
        return {"prediction_task": task, "neutral_return_threshold": float(threshold)}
    raise ValueError(f"Unknown baseline prediction task: {task}")


def parameters_for_task(parameters, selection):
    """Legacy tasks retain their exact contract; new tasks cannot consume cost labels."""
    result = copy.deepcopy(parameters)
    task = signal_task(selection)
    if task["prediction_task"] == "quantiles":
        return result
    result["signal_task"] = task
    result["direction"] = {"enabled": False}
    result["objectives"] = (
        {
            "neural": "normalized_smooth_l1",
            "gbdt": "absolute_error",
            "rules": "train_residual_median",
        }
        if task["prediction_task"] == "alpha_point"
        else {"neural": "cross_entropy", "gbdt": "log_loss", "rules": "fixed_return_threshold"}
    )
    if task["prediction_task"] == "absolute_direction":
        result.setdefault(
            "class_balance_audit", {"minimum_fraction": 0.01, "maximum_fraction": 0.98}
        )
    return result


def log_thresholds(threshold):
    """A symmetric simple-return interval is asymmetric in log-return space."""
    task = signal_task(
        {
            "baseline_task": {
                "prediction_task": "absolute_direction",
                "neutral_return_threshold": threshold,
            }
        }
    )
    value = task["neutral_return_threshold"]
    return math.log1p(-value), math.log1p(value)


def validate_signal_metrics(metrics, task, samples, horizons):
    """Reject legacy quantile/double-direction results as single-task completion."""
    if not isinstance(metrics, dict) or metrics.get("prediction_task") != task["prediction_task"]:
        raise ValueError("Baseline metrics do not match the independent prediction task")
    if metrics.get("samples") != samples:
        raise ValueError("Baseline signal evaluation does not cover the full population")
    if (
        task["prediction_task"] == "alpha_point"
        and metrics.get("point_loss_beta") != task["point_loss_beta"]
    ):
        raise ValueError("Baseline point-loss beta does not match")
    if (
        task["prediction_task"] == "absolute_direction"
        and metrics.get("neutral_return_threshold") != task["neutral_return_threshold"]
    ):
        raise ValueError("Baseline fixed-return thresholds do not match")
    actual = metrics.get("per_horizon", {})
    if set(actual) != {f"{h}d" for h in horizons}:
        raise ValueError("Baseline signal evaluation has incomplete horizons")


def validate_class_audit(
    audit, task, count, horizons, *, minimum_fraction=0.01, maximum_fraction=0.98
):
    """Validate full train class support on the local control host without ML imports."""
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (minimum_fraction, maximum_fraction)
        )
        or not 0 <= minimum_fraction < maximum_fraction <= 1
    ):
        raise ValueError("Invalid training class-balance warning thresholds")
    if (
        not isinstance(audit, dict)
        or not isinstance(audit.get("counts"), dict)
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("Train class audit needs a complete nonempty count object")
    if (
        audit.get("prediction_task") != "absolute_direction"
        or audit.get("neutral_return_threshold") != task["neutral_return_threshold"]
    ):
        raise ValueError("Train class audit has a different label contract")
    expected_horizons = {str(h) for h in horizons}
    if (
        audit.get("samples") != count
        or not isinstance(audit["counts"].get("overall"), dict)
        or set(audit["counts"]["overall"]) != expected_horizons
    ):
        raise ValueError("Train class audit did not cover every training window/horizon")
    markets = set(audit["counts"]) - {"overall"}
    if not markets:
        raise ValueError("Train class audit requires market-specific counts")
    warnings = []
    for market, rows in audit["counts"].items():
        if not isinstance(rows, dict) or set(rows) != expected_horizons:
            raise ValueError(f"Incomplete train market audit: {market}")
        for horizon, values in rows.items():
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(type(v) is not int or v <= 0 for v in values)
            ):
                raise ValueError(
                    f"Missing training class: market={market}, horizon={horizon}, counts={values}"
                )
            total = sum(values)
            if market == "overall" and total != count:
                raise ValueError("Train class audit counts do not match the full population")
            fractions = [v / total for v in values]
            if min(fractions) < minimum_fraction or max(fractions) > maximum_fraction:
                warnings.append(
                    {"market": market, "horizon": horizon, "class_fractions": fractions}
                )
    for horizon in expected_horizons:
        totals = [sum(audit["counts"][market][horizon][c] for market in markets) for c in range(3)]
        if totals != audit["counts"]["overall"][horizon]:
            raise ValueError("Train market class counts do not sum to the full population")
    return warnings
