"""Holding-aligned, open-to-open evaluation of ranking signals (protocol section 13.6).

For a stock's row on signal date d and its row on date d + h, the identity of
section 13.5 gives log(open(d+1+h) / open(d+1)) = abs[d, h+1] - abs[d+h, 1]; with
zero-based columns that is abs[d, h] - abs[d+h, 0]. The sessions inside that
window are the open-to-close returns abs[d+j, 0] for j = 0 .. h-1; the rest of
the window is overnight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twbacktest import positions, stats
from twbacktest.data import SplitIndex

MAX_HORIZON = 13  # abs has 14 columns, and the identity needs column h


@dataclass(frozen=True)
class OpenToOpen:
    """Log returns over the h sessions after each row's next open (NaN when unavailable)."""

    stock: np.ndarray
    index: np.ndarray
    session: np.ndarray  # sum of the h open-to-close returns inside the window

    @property
    def alpha(self) -> np.ndarray:
        return self.stock - self.index

    @property
    def overnight(self) -> np.ndarray:
        return self.stock - self.session


def open_to_open(
    absolute: np.ndarray, alpha: np.ndarray, index: SplitIndex, table: np.ndarray, h: int
) -> OpenToOpen:
    if not 1 <= h <= MAX_HORIZON:
        raise ValueError(f"Open-to-open horizons must be 1-{MAX_HORIZON}")
    bench = absolute - alpha
    ahead = positions.rows_ahead(index, table, h)
    available = ahead >= 0
    stock = np.full(index.count, np.nan)
    index_return = np.full(index.count, np.nan)
    stock[available] = absolute[available, h] - absolute[ahead[available], 0]
    index_return[available] = bench[available, h] - bench[ahead[available], 0]
    session = np.zeros(index.count)
    for offset in range(h):
        rows = positions.rows_ahead(index, table, offset)
        session += np.where(rows >= 0, absolute[np.maximum(rows, 0), 0], np.nan)
    session[~available] = np.nan
    return OpenToOpen(stock=stock, index=index_return, session=session)


def _top_count(size: int, fraction: float) -> int:
    return max(1, int(np.floor(size * fraction + 0.5)))


def daily_rank_ic(scores: np.ndarray, target: np.ndarray, groups) -> np.ndarray:
    """Per-date Spearman correlation with average ties; NaN when undefined."""
    output = np.full(len(groups), np.nan)
    for j, rows in enumerate(groups):
        valid = rows[np.isfinite(scores[rows]) & np.isfinite(target[rows])]
        if valid.size < 3:
            continue
        x = pd.Series(scores[valid]).rank(method="average").to_numpy()
        y = pd.Series(target[valid]).rank(method="average").to_numpy()
        if x.min() == x.max() or y.min() == y.max():
            continue
        output[j] = np.corrcoef(x, y)[0, 1]
    return output


@dataclass(frozen=True)
class Deciles:
    """Per-date excess over the same-date equal weight of labelled stocks."""

    top: np.ndarray
    bottom: np.ndarray


def deciles(scores: np.ndarray, simple: np.ndarray, groups, fraction: float) -> Deciles:
    top = np.full(len(groups), np.nan)
    bottom = np.full(len(groups), np.nan)
    for j, rows in enumerate(groups):
        valid = rows[np.isfinite(scores[rows]) & np.isfinite(simple[rows])]
        if valid.size == 0:
            continue
        order = valid[np.argsort(-scores[valid], kind="stable")]
        count = _top_count(valid.size, fraction)
        mean = simple[valid].mean()
        top[j] = simple[order[:count]].mean() - mean
        bottom[j] = simple[order[-count:]].mean() - mean
    return Deciles(top=top, bottom=bottom)


def session_split(scores: np.ndarray, labels: OpenToOpen, groups, fraction: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Top-decile excess log return in the sessions and overnight, per date."""
    session = np.full(len(groups), np.nan)
    overnight = np.full(len(groups), np.nan)
    for j, rows in enumerate(groups):
        valid = rows[np.isfinite(scores[rows]) & np.isfinite(labels.stock[rows])
                     & np.isfinite(labels.session[rows])]
        if valid.size == 0:
            continue
        order = valid[np.argsort(-scores[valid], kind="stable")]
        chosen = order[:_top_count(valid.size, fraction)]
        session[j] = labels.session[chosen].mean() - labels.session[valid].mean()
        overnight[j] = labels.overnight[chosen].mean() - labels.overnight[valid].mean()
    return session, overnight


def interval(series: np.ndarray, *, block: int, bootstrap: dict) -> dict[str, float]:
    mean, low, high = stats.bootstrap_mean(
        series, block=block, replicates=int(bootstrap["replicates"]), seed=int(bootstrap["seed"])
    )
    return {"mean": mean, "low": low, "high": high}


def rank_summary(daily: np.ndarray) -> dict[str, float]:
    finite = daily[np.isfinite(daily)]
    deviation = finite.std(ddof=1) if finite.size > 1 else 0.0
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "ir": float(finite.mean() / deviation) if deviation > 0 else float("nan"),
        "dates": int(finite.size),
    }
