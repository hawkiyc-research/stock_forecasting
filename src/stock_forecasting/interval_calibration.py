"""Past-only regime calibration and explicitly prequential, delayed feedback.

These are empirical calibration procedures, not exchangeability guarantees.
Frozen parameters are fitted only on validation. Online holdout feedback never
changes the frozen artifact, checkpoint selection, median, or ranking score.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def volatility_ratio(features):
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("Regime calibration requires historical scale features")
    result = values[:, 2] / np.maximum(values[:, 5], 1e-8)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Historical volatility ratio must be finite and nonnegative")
    return result.astype(np.float32)


def weighted_quantile(values, weights, quantile=0.9):
    """An empirical weighted quantile; no unsupported conformal guarantee."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    position = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[order[min(int(position), len(order) - 1)]])


def fit_regime_calibration(store, static_reference, policy):
    if store.volatility is None or store.offset != store.count:
        raise ValueError("Regime calibration requires full validation and historical volatility")

    def fit(market):
        index = (np.arange(store.count) if market == "__pooled__"
                 else np.flatnonzero(store.metadata["market"] == market))
        dates = np.asarray([v[:10] for v in store.metadata["date"][index]])
        unique, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
        # Equal mass per date prevents thousands of correlated stocks from
        # masquerading as thousands of independent calibration dates.
        weights = np.exp2((inverse - (len(unique) - 1)) / policy["recency_half_life_sessions"])
        weights /= counts[inverse]
        reference = static_reference["factors"].get(
            market, static_reference["factors"]["__pooled__"]
        )
        thresholds = np.quantile(store.volatility[index], [1 / 3, 2 / 3]).tolist()
        regimes = np.searchsorted(thresholds, store.volatility[index], side="right")

        def estimate(selected, prior):
            days = len(np.unique(dates[selected]))
            n = int(np.count_nonzero(selected))
            if n < policy["minimum_group_samples"] or days < policy["minimum_group_dates"]:
                return {**prior, "samples": n, "dates": days, "shrinkage_weight": 0.0}
            strength = days / (days + policy["shrinkage_dates"])
            rows, w = index[selected], weights[selected]
            result = {"samples": n, "dates": days, "shrinkage_weight": strength}
            for side, column, sign in (("lower", 0, -1), ("upper", 2, 1)):
                factors = []
                for h in range(len(store.horizons)):
                    median = store.predictions[rows, h, 1]
                    width = np.maximum(sign * (store.predictions[rows, h, column] - median),
                                       max(store.scales[h] * 1e-6, 1e-10))
                    score = sign * (store.targets[rows, h] - median) / width
                    factor = max(1e-4, weighted_quantile(score, w))
                    # Geometric shrinkage preserves strictly positive widths.
                    factors.append(math.exp(strength * math.log(factor)
                                            + (1 - strength) * math.log(prior[side][h])))
                result[side] = factors
            return result

        base = estimate(np.ones(len(index), dtype=bool), reference)
        return market, base, {
            "thresholds": thresholds,
            "factors": [estimate(regimes == r, base) for r in range(3)],
        }

    markets = ["__pooled__", *map(str, np.unique(store.metadata["market"]))]
    with ThreadPoolExecutor(max_workers=store._workers()) as pool:
        rows = list(pool.map(fit, markets))
    return {
        "schema_version": 2,
        "fit_split": "validation",
        "horizons": store.horizons,
        "target_coverage": 0.8,
        "method": "market_volatility_regime_recency_shrinkage",
        "regime_feature": "past_relative_return_std20_over_std60",
        "factors": {market: base for market, base, _ in rows},
        "regimes": {market: regimes for market, _, regimes in rows},
        "policy": dict(policy),
        "static_reference": static_reference,
        "fit_date_end": max(str(v)[:10] for v in store.metadata["date"]),
        "guarantee": "Empirical calibration; no guaranteed future or per-regime 80% coverage.",
    }


def apply_factors(predictions, markets, calibration, volatility=None):
    output = np.array(predictions, dtype=np.float32, copy=True)
    markets = np.asarray(markets)
    if output.ndim != 3 or output.shape[2] != 3 or markets.shape != (len(output),):
        raise ValueError("Calibration requires aligned predictions and markets")
    adaptive = calibration["schema_version"] == 2
    if adaptive:
        volatility = np.asarray(volatility)
        if (volatility.shape != (len(output),) or not np.isfinite(volatility).all()
                or (volatility < 0).any()):
            raise ValueError("Regime calibration requires aligned past-only volatility ratios")
    for market in np.unique(markets):
        selected = np.flatnonzero(markets == market)
        name = str(market) if str(market) in calibration["factors"] else "__pooled__"
        if adaptive:
            group = calibration["regimes"][name]
            regimes = np.searchsorted(group["thresholds"], volatility[selected], side="right")
            lower = np.asarray([f["lower"] for f in group["factors"]])[regimes]
            upper = np.asarray([f["upper"] for f in group["factors"]])[regimes]
        else:
            lower = np.asarray(calibration["factors"][name]["lower"])
            upper = np.asarray(calibration["factors"][name]["upper"])
        q = output[selected]
        q[..., 0] = q[..., 1] - (q[..., 1] - q[..., 0]) * lower
        q[..., 2] = q[..., 1] + (q[..., 2] - q[..., 1]) * upper
        output[selected] = q
    return output


def validate_regimes(calibration, horizons):
    if calibration.get("schema_version") != 2:
        return
    from stock_forecasting.config import IntervalCalibrationConfig

    policy = IntervalCalibrationConfig.model_validate(calibration.get("policy", {}))
    if policy.mode != "regime_adaptive" or calibration.get("regime_feature") != (
        "past_relative_return_std20_over_std60"
    ):
        raise ValueError("Invalid adaptive calibration policy or feature")
    regimes = calibration.get("regimes", {})
    if set(regimes) != set(calibration["factors"]):
        raise ValueError("Regime and market calibration populations disagree")
    for group in regimes.values():
        thresholds = np.asarray(group["thresholds"])
        if (thresholds.shape != (2,) or not np.isfinite(thresholds).all()
                or thresholds[0] < 0 or thresholds[1] < thresholds[0]
                or len(group["factors"]) != 3):
            raise ValueError("Invalid historical volatility regime thresholds")
        for factor in group["factors"]:
            for side in ("lower", "upper"):
                values = np.asarray(factor[side])
                if (values.shape != (len(horizons),) or not np.isfinite(values).all()
                        or (values <= 0).any()):
                    raise ValueError("Invalid regime calibration factors")


def prequential_predictions(store, calibration, output, *, calendars=None):
    """Write chronological predictions; access labels only after scheduled exit.

    Each market is independent and processed by a bounded thread pool. The
    chronological loop is necessarily serial within a market: today's feedback
    changes tomorrow's widths. Arrays remain memory mapped, not materialized.
    """
    if calibration["schema_version"] != 2 or store.volatility is None:
        raise ValueError("Delayed-feedback evaluation requires regime calibration")
    policy = calibration["policy"]
    dates = np.asarray([str(v)[:10] for v in store.metadata["date"]], dtype="datetime64[D]")
    if str(dates.min()) <= calibration["fit_date_end"]:
        raise ValueError("Online evaluation must start strictly after frozen calibration dates")
    label_boundary = calibration.get("fit_label_end_exclusive")
    if label_boundary is not None and str(dates.min()) < label_boundary:
        raise ValueError("Online evaluation cannot precede the validation label boundary")
    markets = list(map(str, np.unique(store.metadata["market"])))
    if calendars is None:
        from stock_forecasting.data.sample_universe import market_sessions
        from stock_forecasting.data_policy import load_data_policy

        names = load_data_policy()["calendars"]
        calendars = {
            m: market_sessions(
                names[m], str(dates.min()), str(dates.max() + np.timedelta64(120, "D"))
            )
            .tz_localize(None).to_numpy().astype("datetime64[D]")
            for m in markets
        }

    def evaluate(market):
        rows = np.flatnonzero(store.metadata["market"] == market)
        rows = rows[np.argsort(dates[rows], kind="stable")]
        days, starts = np.unique(dates[rows], return_index=True)
        sessions = np.asarray(calendars[market], dtype="datetime64[D]")
        positions = np.searchsorted(sessions, days)
        if ((positions + max(store.horizons) >= len(sessions)).any()
                or not np.array_equal(sessions[positions], days)):
            raise ValueError("Online label maturity requires the complete market session calendar")
        regime = calibration["regimes"].get(market, calibration["regimes"]["__pooled__"])
        log_offsets = np.zeros((3, len(store.horizons), 2), dtype=np.float64)
        limit = math.log(policy["online_max_multiplier"])
        pending = {}
        feedback_batches = 0
        observed = np.zeros(len(store.horizons), dtype=np.int64)
        for i, day in enumerate(days):
            # Strictly previous sessions are used, even though signals are after
            # close. Same-day and not-yet-mature labels cannot affect this day.
            for maturity in sorted(k for k in pending if k < day):
                for earlier, h in pending.pop(maturity):
                    groups = np.searchsorted(
                        regime["thresholds"], store.volatility[earlier], side="right"
                    )
                    y = store.targets[earlier, h]
                    q = output[earlier, h]
                    errors = np.stack((y < q[:, 0], y > q[:, 2]), axis=-1)
                    for r in range(3):
                        selected = groups == r
                        n = int(selected.sum())
                        if n:
                            strength = n / (n + policy["minimum_group_samples"])
                            log_offsets[r, h] += policy["online_learning_rate"] * strength * (
                                errors[selected].mean(axis=0) - 0.1
                            )
                    np.clip(log_offsets, -limit, limit, out=log_offsets)
                    observed[h] += len(earlier)
                    feedback_batches += 1
            current = rows[starts[i]:starts[i + 1] if i + 1 < len(starts) else len(rows)]
            # A date group is bounded by the universe, not the time-series length.
            q = apply_factors(store.predictions[current], [market] * len(current),
                              calibration, store.volatility[current])
            groups = np.searchsorted(regime["thresholds"], store.volatility[current], side="right")
            multipliers = np.exp(log_offsets[groups])
            q[..., 0] = q[..., 1] - (q[..., 1] - q[..., 0]) * multipliers[..., 0]
            q[..., 2] = q[..., 1] + (q[..., 2] - q[..., 1]) * multipliers[..., 1]
            output[current] = q
            for h, horizon in enumerate(store.horizons):
                maturity = sessions[positions[i] + horizon]
                pending.setdefault(maturity, []).append((current, h))
        return market, {
            "last_forecast_date": str(days[-1]),
            "feedback_batches": feedback_batches,
            "matured_samples_per_horizon": observed.tolist(),
            "final_width_multipliers": np.exp(log_offsets).tolist(),
        }

    with ThreadPoolExecutor(max_workers=min(store._workers(), len(markets))) as pool:
        evidence = dict(pool.map(evaluate, markets))
    output.flush()
    return {
        "protocol": "prequential_previous_session_matured_labels_only",
        "feedback_delay": "scheduled_exit_at_t_plus_h_must_be_strictly_before_forecast_date",
        "checkpoint_selection_uses_online_metrics": False,
        "frozen_calibration_modified": False,
        "by_market": evidence,
    }
