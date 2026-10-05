"""Main-model diagnostics and validation-only interval calibration.

Calibration is model-specific and never fits factors from holdout labels.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from stock_forecasting.evaluation_store import CHUNK_ROWS, EvaluationStore, Moments
from stock_forecasting.metrics import cross_sectional_metrics
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count


class ForecastEvaluationStore(EvaluationStore):
    def __init__(self, *args, ranking: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = None
        if ranking:
            self.scores = np.lib.format.open_memmap(
                self.root / "ranking-scores.npy",
                mode="w+",
                dtype="float32",
                shape=(self.count, len(self.horizons)),
            )

    def append(self, *args, ranking_scores=None, **kwargs):
        start = self.offset
        super().append(*args, **kwargs)
        if self.scores is not None:
            if (
                ranking_scores is None
                or ranking_scores.shape != (self.offset - start, len(self.horizons))
                or not np.isfinite(ranking_scores).all()
            ):
                raise ValueError("Independent ranking scores are missing or invalid")
            self.scores[start : self.offset] = ranking_scores

    def _workers(self):
        # Quantiles and date-group reductions release the GIL. Each worker may
        # copy one population-sized vector, never the complete prediction cube.
        memory = detect_available_memory().available_bytes
        requested = int(os.environ.get("FORECAST_METRIC_WORKERS", "4"))
        if requested < 1:
            raise ValueError("FORECAST_METRIC_WORKERS must be positive")
        workers = min(
            requested,
            detect_visible_cpu_count(),
            max(1, int(memory * 0.15) // max(self.count * 128, 64 * 1024**2)),
        )
        if workers == 1:
            print("Forecast metrics: CPU/memory budget permits one bounded worker", flush=True)
        return workers

    def ranking_metrics(self):
        if self.scores is None:
            return None
        self.scores.flush()

        def metric(task):
            market, index = task
            selected = np.flatnonzero(self.metadata["market"] == market)
            h = self.horizons[index]
            return (
                market,
                f"{h}d",
                cross_sectional_metrics(
                    targets=self.targets[selected, index],
                    signals=self.scores[selected, index],
                    dates=self.metadata["date"][selected],
                    symbols=self.metadata["symbol"][selected],
                    annualization_horizon=h,
                ),
            )

        result = {}
        tasks = [
            (str(market), i)
            for market in np.unique(self.metadata["market"])
            for i in range(len(self.horizons))
        ]
        # At most four markets times fourteen horizons, with bounded workers.
        with ThreadPoolExecutor(max_workers=self._workers()) as pool:
            for market, horizon, metrics in pool.map(metric, tasks):
                result.setdefault(market, {})[horizon] = metrics
        return {"signal": "independent_dimensionless_score", "by_market": result}

    def fit_interval_calibration(self):
        if self.offset != self.count:
            raise ValueError("Calibration requires the full validation population")
        factors = {}
        markets = ["__pooled__", *map(str, np.unique(self.metadata["market"]))]

        def fit(market):
            selected = (
                slice(None)
                if market == "__pooled__"
                else np.flatnonzero(self.metadata["market"] == market)
            )
            count = self.count if market == "__pooled__" else len(selected)
            if count < 128:
                return market, None
            lower, upper = [], []
            for i in range(len(self.horizons)):
                y = self.targets[selected, i]
                median = self.predictions[selected, i, 1]
                floor = max(self.scales[i] * 1e-6, 1e-10)
                left = np.maximum(median - self.predictions[selected, i, 0], floor)
                right = np.maximum(self.predictions[selected, i, 2] - median, floor)
                lower.append(
                    float(max(1e-4, np.quantile((median - y) / left, 0.9, method="higher")))
                )
                upper.append(
                    float(max(1e-4, np.quantile((y - median) / right, 0.9, method="higher")))
                )
            return market, {"samples": count, "lower": lower, "upper": upper}

        with ThreadPoolExecutor(max_workers=self._workers()) as pool:
            for market, value in pool.map(fit, markets):
                if value is not None:
                    factors[market] = value
        if "__pooled__" not in factors:
            raise ValueError("At least 128 validation windows are required for calibration")
        return {
            "schema_version": 1,
            "fit_split": "validation",
            "horizons": self.horizons,
            "method": "market_horizon_asymmetric_multiplicative_tail_quantiles",
            "target_coverage": 0.8,
            "factors": factors,
            "guarantee": "No exchangeability or future 80-percent coverage guarantee.",
        }

    def calibrated_metrics(self, calibration):
        validate_calibration(calibration, self.horizons)
        total = Moments(self.horizons, self.scales)
        markets = {}
        for start in range(0, self.count, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.count)
            labels = self.metadata["market"][start:stop]
            prediction = calibrate_predictions(self.predictions[start:stop], labels, calibration)
            target = self.targets[start:stop]
            total.add(target, prediction)
            for market in np.unique(labels):
                selected = labels == market
                markets.setdefault(str(market), Moments(self.horizons, self.scales)).add(
                    target[selected],
                    prediction[selected],
                )
        return {
            "samples": self.count,
            **total.result(),
            "by_market": {m: v.result() for m, v in markets.items()},
        }

    def close(self):
        if self.scores is not None:
            self.scores.flush()
            self.scores._mmap.close()
        super().close()


def validate_calibration(calibration, horizons):
    if (
        calibration.get("schema_version") != 1
        or calibration.get("fit_split") != "validation"
        or list(calibration.get("horizons", [])) != list(horizons)
        or "__pooled__" not in calibration.get("factors", {})
    ):
        raise ValueError("Interval calibration does not match the validation/horizon contract")
    for value in calibration["factors"].values():
        for side in ("lower", "upper"):
            factors = np.asarray(value[side])
            if (
                factors.shape != (len(horizons),)
                or not np.isfinite(factors).all()
                or (factors <= 0).any()
            ):
                raise ValueError("Invalid interval calibration factors")


def calibrate_predictions(predictions, markets, calibration):
    output = np.array(predictions, dtype=np.float32, copy=True)
    markets = np.asarray(markets)
    for market in np.unique(markets):
        selected = markets == market
        factor = calibration["factors"].get(str(market), calibration["factors"]["__pooled__"])
        q = output[selected]
        q[..., 0] = q[..., 1] - (q[..., 1] - q[..., 0]) * np.asarray(factor["lower"])
        q[..., 2] = q[..., 1] + (q[..., 2] - q[..., 1]) * np.asarray(factor["upper"])
        output[selected] = q
    return output
