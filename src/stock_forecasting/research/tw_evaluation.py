"""Labels, decomposed classification metrics, date-block bootstrap and criteria."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPSILON = 1e-12
DOWN, NEUTRAL, UP, INVALID = 0, 1, 2, -1


def band_labels(returns: np.ndarray, bands: np.ndarray | float) -> np.ndarray:
    """Down/neutral/up from log returns and simple-return half-widths; equality is neutral."""
    bands = np.broadcast_to(np.asarray(bands, dtype=float), np.shape(returns))
    lower, upper = np.log1p(-np.minimum(bands, 0.999)), np.log1p(bands)
    labels = np.where(returns < lower, DOWN, np.where(returns > upper, UP, NEUTRAL))
    return np.where(np.isfinite(returns) & np.isfinite(bands), labels, INVALID).astype(np.int8)


def hybrid_bands(sigma: np.ndarray, floors: np.ndarray, k: float, horizon: int) -> np.ndarray:
    """max(cost floor, k * daily volatility * sqrt(h)) as a simple-return half-width."""
    return np.maximum(floors, k * sigma * np.sqrt(horizon))


def calibrate_k(
    returns: np.ndarray,
    sigma: np.ndarray,
    floors: np.ndarray,
    horizon: int,
    target: float,
    *,
    high: float = 20.0,
    iterations: int = 60,
) -> dict[str, float]:
    """Bisect k so the train neutral share equals the target (monotone in k)."""
    valid = np.isfinite(returns) & np.isfinite(sigma)
    returns, sigma, floors = returns[valid], sigma[valid], floors[valid]

    def share(k: float) -> float:
        labels = band_labels(returns, hybrid_bands(sigma, floors, k, horizon))
        return float(np.mean(labels == NEUTRAL))

    low_share, high_share = share(0.0), share(high)
    if low_share >= target:
        return {"k": 0.0, "neutral_share": low_share, "floor_binding": True}
    if high_share < target:
        return {"k": high, "neutral_share": high_share, "floor_binding": False}
    low, high_k = 0.0, high
    for _ in range(iterations):
        middle = 0.5 * (low + high_k)
        if share(middle) < target:
            low = middle
        else:
            high_k = middle
    return {"k": high_k, "neutral_share": share(high_k), "floor_binding": False}


def prior_probabilities(labels: np.ndarray, markets: np.ndarray) -> dict[str, np.ndarray]:
    """Jeffreys-smoothed train class shares per market."""
    priors = {}
    for market in np.unique(markets):
        selected = labels[(markets == market) & (labels != INVALID)]
        counts = np.bincount(selected, minlength=3).astype(float) + 0.5
        priors[str(market)] = counts / counts.sum()
    return priors


def ce_parts(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row three-class CE, the move part and the direction part (NaN if neutral)."""
    p = np.clip(probabilities.astype(float), EPSILON, 1.0)
    rows = np.arange(len(labels))
    safe = np.where(labels == INVALID, NEUTRAL, labels)
    ce = -np.log(p[rows, safe])
    neutral = np.clip(p[:, NEUTRAL], EPSILON, 1 - EPSILON)
    move = np.where(labels == NEUTRAL, -np.log(neutral), -np.log(1 - neutral))
    up = np.clip(p[:, UP] / (p[:, UP] + p[:, DOWN]), EPSILON, 1 - EPSILON)
    direction = np.where(
        labels == UP, -np.log(up), np.where(labels == DOWN, -np.log(1 - up), np.nan)
    )
    invalid = labels == INVALID
    return {
        "ce": np.where(invalid, np.nan, ce),
        "move": np.where(invalid, np.nan, move),
        "direction": np.where(invalid, np.nan, direction),
    }


def date_only_probabilities(probabilities: np.ndarray, keys: pd.DataFrame) -> np.ndarray:
    """Replace each row with its date x market mean predicted distribution."""
    frame = pd.DataFrame(probabilities, columns=["p0", "p1", "p2"])
    for column in keys.columns:
        frame[column] = keys[column].to_numpy()
    means = frame.groupby(list(keys.columns))[["p0", "p1", "p2"]].transform("mean")
    return means.to_numpy()


def per_date_sums(values: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """Sum and count of finite values per date, in date order."""
    frame = pd.DataFrame({"date": dates, "value": values})
    frame = frame[np.isfinite(frame["value"])]
    grouped = frame.groupby("date", sort=True)["value"]
    return pd.DataFrame({"sum": grouped.sum(), "count": grouped.count()})


def block_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    replicates: int,
    block: int,
    seed: int,
) -> dict[str, float]:
    """Circular moving-block bootstrap of a ratio of per-date sums (date order kept)."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    days = len(numerator)
    estimate = float(numerator.sum() / denominator.sum()) if days else float("nan")
    if days < 2:
        return {"estimate": estimate, "low": float("nan"), "high": float("nan")}
    block = max(1, min(block, days))
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(days / block))
    starts = rng.integers(0, days, size=(replicates, blocks))
    offsets = np.arange(block)
    indexes = ((starts[:, :, None] + offsets[None, None, :]) % days).reshape(replicates, -1)
    indexes = indexes[:, :days]
    values = numerator[indexes].sum(axis=1) / denominator[indexes].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {"estimate": estimate, "low": float(low), "high": float(high)}


def robust_scale(values: np.ndarray) -> float:
    """max(IQR, 1.4826 * MAD) of finite values, as for the project's alpha scales."""
    finite = values[np.isfinite(values)]
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(max(q75 - q25, 1.4826 * mad, 1e-12))


def smooth_l1(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute < 1.0, 0.5 * absolute * absolute, absolute - 0.5)


def rank_ic(
    prediction: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    markets: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-market date-level Spearman IC (average ties, >=3 finite, non-constant)."""
    frame = pd.DataFrame({"p": prediction, "y": target, "date": dates, "market": markets})
    frame = frame[np.isfinite(frame["p"]) & np.isfinite(frame["y"])]
    daily: dict[str, list[float]] = {}
    for (market, _date), group in frame.groupby(["market", "date"], sort=True):
        if len(group) < 3 or group["p"].nunique() < 2 or group["y"].nunique() < 2:
            continue
        value = np.corrcoef(rankdata(group["y"]), rankdata(group["p"]))[0, 1]
        if np.isfinite(value):
            daily.setdefault(str(market), []).append(float(value))
    result = {}
    for market, values in daily.items():
        array = np.asarray(values)
        std = array.std(ddof=1) if len(array) > 1 else 0.0
        result[market] = {
            "mean": float(array.mean()),
            "ir": float(array.mean() / std) if std > 0 else 0.0,
            "dates": len(array),
        }
    return result


def balanced_accuracy(labels: np.ndarray, predicted: np.ndarray) -> float:
    valid = labels != INVALID
    recalls = [
        float(np.mean(predicted[valid & (labels == label)] == label))
        for label in (DOWN, NEUTRAL, UP)
        if np.any(valid & (labels == label))
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def backtest(
    signal: np.ndarray,
    log_returns: np.ndarray,
    floors: np.ndarray,
    dates: np.ndarray,
    *,
    top_fraction: float,
) -> dict[str, float]:
    """Descriptive: each date, buy the top fraction by signal (signal > 0) at next open.

    Returns are net of the round-trip cost floor. Overlapping holding periods make
    the t statistic optimistic for horizons above one session.
    """
    frame = pd.DataFrame({"signal": signal, "net": np.expm1(log_returns) - floors, "date": dates})
    frame = frame[np.isfinite(frame["signal"]) & np.isfinite(frame["net"])]
    daily = []
    for _date, group in frame.groupby("date", sort=True):
        count = max(1, int(round(len(group) * top_fraction)))
        chosen = group.nlargest(count, "signal")
        chosen = chosen[chosen["signal"] > 0]
        if not chosen.empty:
            daily.append(float(chosen["net"].mean()))
    array = np.asarray(daily)
    if array.size == 0:
        return {"dates": 0}
    std = array.std(ddof=1) if array.size > 1 else 0.0
    return {
        "dates": int(array.size),
        "mean_net_return": float(array.mean()),
        "hit_rate": float(np.mean(array > 0)),
        "t_statistic": float(array.mean() / std * np.sqrt(array.size)) if std > 0 else 0.0,
    }


def evaluate_criteria(
    direction_sums: dict[tuple[str, int], pd.DataFrame],
    prior_sums: dict[int, pd.DataFrame],
    *,
    groups: list[str],
    criteria: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Pre-registered rule: a group passes on >= N decision horizons where its direction
    CE beats the train prior by the relative margin and both the prior and F0 gaps have
    95% date-block intervals above zero."""
    outcome: dict[str, Any] = {}
    for group in groups:
        if group == "F0":
            continue
        horizons = {}
        for horizon in criteria["decision_horizons"]:
            model = direction_sums.get((group, horizon))
            base = direction_sums.get(("F0", horizon))
            prior = prior_sums.get(horizon)
            if model is None or base is None or prior is None:
                horizons[horizon] = {"evaluated": False}
                continue
            count = model["count"].to_numpy()
            prior_gap = block_bootstrap(
                prior["sum"].to_numpy() - model["sum"].to_numpy(), count, **bootstrap
            )
            base_gap = block_bootstrap(
                base["sum"].to_numpy() - model["sum"].to_numpy(), count, **bootstrap
            )
            prior_mean = prior["sum"].sum() / prior["count"].sum()
            relative = prior_gap["estimate"] / prior_mean
            passed = (
                relative >= criteria["min_relative_improvement"]
                and prior_gap["low"] > 0
                and base_gap["low"] > 0
            )
            horizons[horizon] = {
                "evaluated": True,
                "relative_improvement_vs_prior": float(relative),
                "gap_vs_prior": prior_gap,
                "gap_vs_F0": base_gap,
                "passed": bool(passed),
            }
        passed_count = sum(item.get("passed", False) for item in horizons.values())
        outcome[group] = {
            "horizons": horizons,
            "passed_horizons": passed_count,
            "passed": passed_count >= criteria["min_passing_horizons"],
        }
    outcome["information_found"] = any(
        value["passed"] for key, value in outcome.items() if key != "information_found"
    )
    return outcome
