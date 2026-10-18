"""Circular block bootstrap and per-series statistics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def block_indices(length: int, block: int, replicates: int, seed: int) -> np.ndarray:
    """Circular moving-block resampling indices [replicates, length], fixed by the seed."""
    block = max(1, min(block, length))
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(length / block))
    starts = rng.integers(0, length, size=(replicates, blocks))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]) % length
    return indices.reshape(replicates, -1)[:, :length]


def bootstrap_mean(
    values: np.ndarray, *, block: int, replicates: int, seed: int
) -> tuple[float, float, float]:
    """Mean and 95% interval of a date-ordered series (non-finite values dropped)."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        mean = float(values.mean()) if values.size else float("nan")
        return mean, float("nan"), float("nan")
    means = values[block_indices(values.size, block, replicates, seed)].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def ratio(values: np.ndarray, periods_per_year: float) -> float:
    """Annualized mean / standard deviation; NaN with fewer than three values."""
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    deviation = values.std(ddof=1)
    if deviation == 0:
        return float("nan")
    return float(values.mean() / deviation * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return float((wealth / peaks - 1.0).min())


def compound_annual(returns: np.ndarray, periods_per_year: float) -> float:
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + returns))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / returns.size) - 1.0
