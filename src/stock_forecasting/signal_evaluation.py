"""Bounded, full-population evaluation for independent predictive signal tasks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from stock_forecasting.evaluation_store import CHUNK_ROWS, META_DTYPE
from stock_forecasting.metrics import cross_sectional_metrics
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count


def direction_labels(values, threshold):
    """Classify simple-return boundaries in the stored log-return coordinates."""
    values = np.asarray(values, dtype=np.float32)
    if not 0 < threshold < 1 or not np.isfinite(values).all():
        raise ValueError(
            "Direction labels need finite returns and a threshold between zero and one"
        )
    return np.where(
        values < np.float32(np.log1p(-threshold)),
        0,
        np.where(values > np.float32(np.log1p(threshold)), 2, 1),
    ).astype(np.int64)


class SignalMoments:
    """Accumulate exact population sums rather than averages of batch averages."""

    def __init__(self, horizons, scales, task, beta=1.0, threshold=0.01):
        self.horizons, self.task = list(horizons), task
        self.scales = np.asarray(scales, dtype=np.float64)
        self.beta, self.threshold, self.count = float(beta), float(threshold), 0
        if (
            task not in ("alpha_point", "absolute_direction")
            or not np.isfinite(beta)
            or beta <= 0
            or not 0 < threshold < 1
            or self.scales.shape != (len(horizons),)
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Invalid independent signal evaluation contract")
        self.sums = np.zeros((10, len(horizons)), dtype=np.float64)
        self.confusion = np.zeros((len(horizons), 3, 3), dtype=np.int64)
        self.calibration = np.zeros((len(horizons), 15, 3), dtype=np.float64)

    def add(self, targets, predictions):
        y, p = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
        expected = y.shape if self.task == "alpha_point" else (*y.shape, 3)
        if (
            y.shape != (len(y), len(self.horizons))
            or p.shape != expected
            or not np.isfinite(y).all()
            or not np.isfinite(p).all()
        ):
            raise ValueError("Nonfinite or incompatible signal predictions/targets")
        if self.task == "alpha_point":
            error = p - y
            residual = np.abs(error / self.scales)
            loss = np.where(
                residual < self.beta, 0.5 * residual**2 / self.beta, residual - 0.5 * self.beta
            )
            values = (
                loss,
                np.abs(error),
                error**2,
                np.sign(p) == np.sign(y),
                y,
                p,
                y * y,
                p * p,
                y * p,
                error,
            )
            self.sums += np.stack([v.sum(0) for v in values])
        else:
            labels = direction_labels(y, self.threshold)
            shifted = p - p.max(axis=-1, keepdims=True)
            logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            probabilities = np.exp(logp)
            loss = -np.take_along_axis(logp, labels[..., None], -1)[..., 0]
            predicted = probabilities.argmax(-1)
            onehot = np.eye(3)[labels]
            self.sums[0] += loss.sum(0)
            self.sums[1] += ((probabilities - onehot) ** 2).sum(-1).sum(0)
            self.sums[2] += (np.abs(predicted - labels) == 2).sum(0)
            confidence = probabilities.max(-1)
            bins = np.minimum((confidence * 15).astype(int), 14)
            for i in range(len(self.horizons)):
                self.confusion[i] += np.bincount(
                    labels[:, i] * 3 + predicted[:, i], minlength=9
                ).reshape(3, 3)
                self.calibration[i, :, 0] += np.bincount(bins[:, i], minlength=15)
                self.calibration[i, :, 1] += np.bincount(
                    bins[:, i], weights=confidence[:, i], minlength=15
                )
                self.calibration[i, :, 2] += np.bincount(
                    bins[:, i], weights=(predicted[:, i] == labels[:, i]), minlength=15
                )
        self.count += len(y)
        return loss.mean(1)

    def result(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty signal population")
        m = self.sums / self.count
        per = {}
        for i, horizon in enumerate(self.horizons):
            if self.task == "alpha_point":
                covariance = m[8, i] - m[4, i] * m[5, i]
                denom = np.sqrt(max(0, m[6, i] - m[4, i] ** 2) * max(0, m[7, i] - m[5, i] ** 2))
                item = {
                    "normalized_smooth_l1": float(m[0, i]),
                    "mae": float(m[1, i]),
                    "rmse": float(np.sqrt(m[2, i])),
                    "normalized_mae": float(m[1, i] / self.scales[i]),
                    "direction_agreement": float(m[3, i]),
                    "correlation": float(np.clip(covariance / denom, -1, 1))
                    if denom > 1e-20
                    else 0.0,
                    "correlation_status": "ok" if denom > 1e-20 else "unavailable",
                    "bias": float(m[9, i]),
                }
            else:
                confusion = self.confusion[i]
                support, predicted_support = confusion.sum(1), confusion.sum(0)
                diagonal = np.diag(confusion)
                recalls = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
                precision = np.divide(
                    diagonal, predicted_support, out=np.zeros(3), where=predicted_support > 0
                )
                f1 = np.divide(
                    2 * precision * recalls,
                    precision + recalls,
                    out=np.zeros(3),
                    where=(precision + recalls) > 0,
                )
                cal = self.calibration[i]
                item = {
                    "cross_entropy": float(m[0, i]),
                    "brier_score": float(m[1, i]),
                    "accuracy": float(diagonal.sum() / self.count),
                    "balanced_accuracy": float(recalls[support > 0].mean()),
                    "balanced_accuracy_status": "ok"
                    if (support > 0).all()
                    else "missing_target_class",
                    "macro_f1": float(f1.mean()),
                    "opposite_error_rate": float(m[2, i]),
                    "expected_calibration_error": float(
                        np.abs(cal[:, 1] - cal[:, 2]).sum() / self.count
                    ),
                    "majority_class_accuracy": float(support.max() / self.count),
                    "confusion_matrix": confusion.tolist(),
                    "class_support": support.tolist(),
                    "predicted_class_counts": predicted_support.tolist(),
                    "class_recall": recalls.tolist(),
                    "class_precision": precision.tolist(),
                    "predicted_class_share": (predicted_support / self.count).tolist(),
                }
            item["selection_score"] = float(m[0, i])
            per[f"{horizon}d"] = item
        aggregate = {
            key: float(np.mean([v[key] for v in per.values()]))
            for key, value in next(iter(per.values())).items()
            if isinstance(value, (float, int))
        }
        primary = per.get("5d", next(iter(per.values())))
        return {
            "aggregate": aggregate,
            "per_horizon": per,
            "primary_5d": {**primary, "selection_score": aggregate["selection_score"]},
        }


class SignalEvaluationStore:
    """Store one task's outputs; never synthesize quantiles for point forecasts."""

    def __init__(self, root, count, horizons, scales, task, beta=1.0, neutral_threshold=0.01):
        self.root, self.count = Path(root), int(count)
        self.horizons, self.scales, self.task = list(horizons), list(map(float, scales)), task
        self.beta, self.threshold, self.offset = float(beta), float(neutral_threshold), 0
        SignalMoments(self.horizons, self.scales, task, beta, neutral_threshold)
        if count < 1:
            raise ValueError("Evaluation population must be nonempty")
        self.root.mkdir(parents=True, exist_ok=True)
        width = 1 if task == "alpha_point" else 3
        required = count * (
            (1 + width + int(task == "absolute_direction")) * len(horizons) * 4
            + META_DTYPE.itemsize
            + 4
        )
        if shutil.disk_usage(self.root).free < required + 1024**3:
            raise OSError(f"Signal evaluation needs {required} bytes plus 1 GiB scratch space")
        shape = (count, len(horizons)) if width == 1 else (count, len(horizons), 3)
        self.targets = np.lib.format.open_memmap(
            self.root / "targets.npy", mode="w+", dtype="float32", shape=(count, len(horizons))
        )
        self.absolute_targets = (
            np.lib.format.open_memmap(
                self.root / "absolute_targets.npy",
                mode="w+",
                dtype="float32",
                shape=(count, len(horizons)),
            )
            if task == "absolute_direction"
            else None
        )
        self.predictions = np.lib.format.open_memmap(
            self.root / "predictions.npy", mode="w+", dtype="float32", shape=shape
        )
        self.metadata = np.lib.format.open_memmap(
            self.root / "membership.npy", mode="w+", dtype=META_DTYPE, shape=(count,)
        )
        self.drawdown = np.lib.format.open_memmap(
            self.root / "past_drawdown.npy", mode="w+", dtype="float32", shape=(count,)
        )
        self.drawdown[:] = np.nan
        self._result = None

    def append(
        self,
        targets,
        predictions,
        *,
        symbols,
        dates,
        markets,
        asset_types,
        providers,
        target_absolute=None,
        past_drawdown=None,
    ):
        y, p = np.asarray(targets), np.asarray(predictions)
        count, stop = len(y), self.offset + len(y)
        expected = (count, len(self.horizons))
        if (
            stop > self.count
            or y.shape != expected
            or p.shape != self.predictions[:count].shape
            or not np.isfinite(y).all()
            or not np.isfinite(p).all()
        ):
            raise ValueError("Signal evaluation batch violates the population contract")
        section = slice(self.offset, stop)
        self.targets[section], self.predictions[section] = y, p
        if self.absolute_targets is not None:
            absolute = np.asarray(target_absolute)
            if absolute.shape != expected or not np.isfinite(absolute).all():
                raise ValueError("Classification requires explicit finite absolute-return targets")
            self.absolute_targets[section] = absolute
        for key, values in (
            ("symbol", symbols),
            ("date", dates),
            ("market", markets),
            ("asset_type", asset_types),
            ("provider", providers),
        ):
            strings = [str(v) for v in values]
            if len(strings) != count or any(
                len(v) > META_DTYPE[key].itemsize // 4 for v in strings
            ):
                raise ValueError(f"Invalid or truncated signal metadata: {key}")
            self.metadata[key][section] = strings
        if past_drawdown is not None:
            values = np.asarray(past_drawdown)
            if (
                values.shape != (count,)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError("Past drawdown must be one finite fraction per sample")
            self.drawdown[section] = values
        self.offset = stop
        self._result = None

    def _moments(self):
        return SignalMoments(self.horizons, self.scales, self.task, self.beta, self.threshold)

    def _cross_sectional(self):
        requested = int(os.environ.get("FORECAST_METRIC_WORKERS", "4"))
        if requested < 1:
            raise ValueError("FORECAST_METRIC_WORKERS must be positive")
        memory = detect_available_memory().available_bytes
        workers = min(
            requested,
            detect_visible_cpu_count(),
            max(1, int(memory * 0.15) // max(self.count * 256, 64 * 1024**2)),
        )
        if workers == 1:
            print("Signal metrics: CPU/memory budget permits one bounded worker", flush=True)
        tasks = [
            (str(market), index)
            for market in np.unique(self.metadata["market"])
            for index in range(len(self.horizons))
        ]

        def metric(task):
            market, index = task
            rows = np.flatnonzero(self.metadata["market"] == market)
            values = cross_sectional_metrics(
                targets=self.targets[rows, index],
                signals=self.predictions[rows, index],
                dates=self.metadata["date"][rows],
                symbols=self.metadata["symbol"][rows],
                annualization_horizon=self.horizons[index],
            )
            return (
                market,
                f"{self.horizons[index]}d",
                {key: value for key, value in values.items() if key.startswith("rank_ic")},
            )

        result = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for market, horizon, values in pool.map(metric, tasks):
                result.setdefault(market, {})[horizon] = values
        return {"signal": "alpha_point", "by_market": result}

    def finish(self, **_kwargs):
        if self.offset != self.count:
            raise ValueError(f"Incomplete signal evaluation: {self.offset}/{self.count}")
        if self._result is not None:
            return self._result
        for array in (self.targets, self.predictions, self.metadata, self.drawdown):
            array.flush()
        if self.absolute_targets is not None:
            self.absolute_targets.flush()
        total = self._moments()
        groups = {
            name: {}
            for name in ("market", "asset_type", "provider", "month", "year", "past_drawdown")
        }
        digest, daily = hashlib.sha256(b"["), {}
        for start in range(0, self.count, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.count)
            target_store = self.targets if self.absolute_targets is None else self.absolute_targets
            y, p, meta = (
                target_store[start:stop],
                self.predictions[start:stop],
                self.metadata[start:stop],
            )
            losses = total.add(y, p)
            for name, subgroups in groups.items():
                if name == "past_drawdown":
                    dd = self.drawdown[start:stop]
                    values = np.where(
                        ~np.isfinite(dd),
                        "unavailable",
                        np.where(
                            dd < 0.2,
                            "below_20pct",
                            np.where(dd < 0.4, "20_to_40pct", "at_least_40pct"),
                        ),
                    )
                elif name in ("month", "year"):
                    values = np.asarray([v[: 7 if name == "month" else 4] for v in meta["date"]])
                else:
                    values = meta[name]
                for value in np.unique(values):
                    subgroup = subgroups.setdefault(str(value), self._moments())
                    selected = values == value
                    subgroup.add(y[selected], p[selected])
            for day in np.unique(meta["date"]):
                selected = meta["date"] == day
                old_sum, old_count = daily.get(str(day), (0.0, 0))
                daily[str(day)] = (
                    old_sum + float(losses[selected].sum()),
                    old_count + int(selected.sum()),
                )
            for index, row in enumerate(meta, start):
                if index:
                    digest.update(b",")
                digest.update(
                    json.dumps(
                        [str(row["symbol"]), str(row["date"])],
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                )
        digest.update(b"]")
        dates = self.metadata["date"]
        order = np.argsort(dates, kind="stable")
        boundaries = np.flatnonzero(dates[order][1:] != dates[order][:-1]) + 1
        for rows in np.split(order, boundaries):
            if len(np.unique(self.metadata["symbol"][rows])) != len(rows):
                raise ValueError("Duplicate stock-date in full signal evaluation")
        del order, boundaries
        result = total.result()
        result.update(
            {
                "prediction_task": self.task,
                "output_schema": "signal-1.0",
                "loss": result["aggregate"]["selection_score"],
                "samples": self.count,
                "sample_membership": {
                    "ordered_symbol_dates_sha256": digest.hexdigest(),
                    "samples": self.count,
                    "unique_dates": len(daily),
                    "cutoff_start": min(daily),
                    "cutoff_end": max(daily),
                },
                "evaluation_robust_scales": self.scales,
                "daily_signal_loss": {day: value / n for day, (value, n) in sorted(daily.items())},
                "subgroups": {
                    name: {
                        key: {"samples": value.count, **value.result()}
                        for key, value in sorted(subgroups.items())
                    }
                    for name, subgroups in groups.items()
                },
                "drawdown_diagnostics": {
                    "source": "observed_input_close",
                    "thresholds": [0.2, 0.4],
                    "filters_samples": False,
                },
            }
        )
        if self.task == "alpha_point":
            result["point_loss_beta"] = self.beta
            result["cross_sectional"] = self._cross_sectional()
        else:
            result["neutral_return_threshold"] = self.threshold
            result["class_order"] = ["down", "neutral", "up"]
        result["market_macro"] = {
            key: float(
                np.mean(
                    [group["aggregate"][key] for group in result["subgroups"]["market"].values()]
                )
            )
            for key in result["aggregate"]
        }
        self._result = result
        return result

    def metrics(self):
        return self.finish()

    def close(self):
        for array in (self.targets, self.predictions, self.metadata, self.drawdown):
            array.flush()
            array._mmap.close()
        if self.absolute_targets is not None:
            self.absolute_targets.flush()
            self.absolute_targets._mmap.close()
