"""Main-model diagnostics and validation-only interval calibration.

Frozen calibration is validation-only. Explicit online diagnostics may update
separate width multipliers using strictly earlier, matured holdout labels.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from stock_forecasting.evaluation_store import CHUNK_ROWS, EvaluationStore, Moments
from stock_forecasting.interval_calibration import (
    apply_factors,
    fit_regime_calibration,
    prequential_predictions,
    validate_regimes,
    volatility_ratio,
)
from stock_forecasting.metrics import cross_sectional_metrics
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count


class ForecastEvaluationStore(EvaluationStore):
    def __init__(self, *args, ranking: bool = False, calibration_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = None
        self.calibration_policy = calibration_policy
        self.volatility = None
        if calibration_policy is not None and calibration_policy["mode"] == "regime_adaptive":
            extra_bytes = self.count * (4 + 12 * len(self.horizons))
            if shutil.disk_usage(self.root).free < extra_bytes + 1024**3:
                super().close()
                raise OSError("Insufficient scratch disk for bounded adaptive evaluation")
            self.volatility = np.lib.format.open_memmap(
                self.root / "past-volatility.npy", mode="w+", dtype="float32", shape=(self.count,)
            )
        if ranking:
            self.scores = np.lib.format.open_memmap(
                self.root / "ranking-scores.npy",
                mode="w+",
                dtype="float32",
                shape=(self.count, len(self.horizons)),
            )

    def append(self, *args, ranking_scores=None, scale_features=None, **kwargs):
        start = self.offset
        super().append(*args, **kwargs)
        if self.volatility is not None:
            values = volatility_ratio(scale_features)
            if len(values) != self.offset - start:
                raise ValueError("Historical volatility must align with evaluation samples")
            self.volatility[start:self.offset] = values
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
            max(1, int(memory * 0.15) // max(self.count * 256, 64 * 1024**2)),
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
        result = {
            "schema_version": 1,
            "fit_split": "validation",
            "horizons": self.horizons,
            "method": "market_horizon_asymmetric_multiplicative_tail_quantiles",
            "target_coverage": 0.8,
            "factors": factors,
            "guarantee": "No exchangeability or future 80-percent coverage guarantee.",
        }
        if self.volatility is not None:
            self.volatility.flush()
            return fit_regime_calibration(self, result, self.calibration_policy)
        return result

    def calibrated_metrics(self, calibration):
        validate_calibration(calibration, self.horizons)
        return self._calibrated_summary(
            lambda start, stop: calibrate_predictions(
                self.predictions[start:stop], self.metadata["market"][start:stop], calibration,
                None if self.volatility is None else self.volatility[start:stop],
            )
        )

    def adaptive_metrics(self, calibration, *, calendars=None):
        validate_calibration(calibration, self.horizons)
        output = np.lib.format.open_memmap(
            self.root / "prequential-predictions.npy", mode="w+", dtype="float32",
            shape=self.predictions.shape,
        )
        try:
            evidence = prequential_predictions(self, calibration, output, calendars=calendars)
            return {**self._calibrated_summary(lambda start, stop: output[start:stop]),
                    "online_feedback": evidence, "in_sample_fit": False}
        finally:
            output.flush()
            output._mmap.close()

    def _calibrated_summary(self, predictions):
        total = Moments(self.horizons, self.scales)
        markets = {}
        months = {}
        market_months = {}
        for start in range(0, self.count, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.count)
            labels = self.metadata["market"][start:stop]
            prediction = predictions(start, stop)
            target = self.targets[start:stop]
            total.add(target, prediction)
            dates = np.asarray([v[:7] for v in self.metadata["date"][start:stop]])
            for month in np.unique(dates):
                selected = dates == month
                months.setdefault(str(month), Moments(self.horizons, self.scales)).add(
                    target[selected], prediction[selected]
                )
            for market in np.unique(labels):
                selected = labels == market
                markets.setdefault(str(market), Moments(self.horizons, self.scales)).add(
                    target[selected],
                    prediction[selected],
                )
                for month in np.unique(dates[selected]):
                    rows = selected & (dates == month)
                    market_months.setdefault(str(market), {}).setdefault(
                        str(month), Moments(self.horizons, self.scales)
                    ).add(target[rows], prediction[rows])
        result = {
            "samples": self.count,
            **total.result(),
            "by_market": {m: v.result() for m, v in markets.items()},
            "by_month": {m: v.result() for m, v in months.items()},
            "by_market_month": {m: {d: v.result() for d, v in groups.items()}
                                for m, groups in market_months.items()},
        }
        result["market_macro"] = market_macro_metrics(result)
        return result

    def close(self):
        if self.volatility is not None:
            self.volatility.flush()
            self.volatility._mmap.close()
        if self.scores is not None:
            self.scores.flush()
            self.scores._mmap.close()
        super().close()


def validate_calibration(calibration, horizons):
    if (
        calibration.get("schema_version") not in (1, 2)
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
    validate_regimes(calibration, horizons)


def calibrate_predictions(predictions, markets, calibration, volatility=None):
    return apply_factors(predictions, markets, calibration, volatility)


def market_macro_metrics(metrics):
    """Diagnostic equal-market averages, never a substitute checkpoint monitor."""
    markets = list(metrics.get("by_market", {}).values())
    if not markets:
        return {}
    return {
        "market_count": len(markets),
        "weighting": "equal_market_diagnostic_only",
        "aggregate": {k: float(np.mean([m["aggregate"][k] for m in markets]))
                      for k in markets[0]["aggregate"]},
        "per_horizon": {
            h: {k: float(np.mean([m["per_horizon"][h][k] for m in markets]))
                for k in values}
            for h, values in markets[0]["per_horizon"].items()
        },
    }
