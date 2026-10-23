#!/usr/bin/env python3
"""Score a residualized-label B0 run against the standard B0 run (protocol 13.16, H-F).

Reads two downloaded walk-forward outputs (artifacts/lgbm/<launch>/): the candidate trained on
a label variant (results.json has cells_residual and daily_rank_ic_residual.parquet) and the
reference standard B0. Reports, on P1+P2 and per period:

  (i)  the candidate's holding cell (h = 5-10) Rank IC against the RESIDUAL target in P2, with
       its 95% block-bootstrap interval (block 20) -- H-F condition (i): lower bound > 0;
  (ii) the paired daily difference candidate - reference of the short cell (h = 1) against the
       RAW alpha target on P1+P2 (block 5) -- H-F condition (ii): lower bound > -0.01.

Run with the backtest environment (pandas + pyarrow):
  backtest/.venv/bin/python scripts/score_label_variant.py \
      --candidate artifacts/lgbm/<variant-launch> --reference artifacts/lgbm/<b0-launch>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = {"short": (1,), "holding": (5, 6, 7, 8, 9, 10), "long": (11, 12, 13, 14)}
BLOCKS = {"short": 5, "holding": 20, "long": 28}
PERIODS = {"P1": ("2023-01-02", "2025-06-01"), "P2": ("2025-06-01", "2026-06-01")}
REPLICATES, SEED = 1000, 11


def bootstrap(values: np.ndarray, block: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(values.mean()) if values.size else math.nan, math.nan, math.nan
    rng = np.random.default_rng(SEED)
    block = max(1, min(block, values.size))
    starts = rng.integers(0, values.size, size=(REPLICATES, math.ceil(values.size / block)))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(REPLICATES, -1)
    means = values[index[:, : values.size] % values.size].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def cell_series(daily: pd.DataFrame, model: str, horizons: tuple[int, ...]) -> pd.Series:
    rows = daily[(daily["model"] == model) & daily["h"].isin(horizons)]
    per_market = rows.groupby(["date", "market", "h"])["ic"].mean()
    return per_market.groupby(["date", "h"]).mean().groupby("date").mean().sort_index()


def period_mask(dates: pd.DatetimeIndex, period: str) -> np.ndarray:
    if period == "P1+P2":
        return np.any([period_mask(dates, p) for p in PERIODS], axis=0)
    start, end = PERIODS[period]
    return np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)))


def load(root: Path) -> tuple[dict, str, pd.DataFrame, pd.DataFrame | None]:
    results = json.loads((root / "results.json").read_text())
    model = results["model"]
    daily = pd.read_parquet(root / "daily_rank_ic.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    residual_path = root / "daily_rank_ic_residual.parquet"
    residual = None
    if residual_path.is_file():
        residual = pd.read_parquet(residual_path)
        residual["date"] = pd.to_datetime(residual["date"])
    return results, model, daily, residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="JSON report path (default: candidate root)")
    args = parser.parse_args()
    _c_results, c_model, c_daily, c_residual = load(args.candidate)
    _r_results, r_model, r_daily, _ = load(args.reference)
    report: dict = {"candidate": c_model, "reference": r_model, "cells": {}, "paired": {},
                    "residual_cells": {}}
    for cell, horizons in CELLS.items():
        c_series = cell_series(c_daily, c_model, horizons)
        r_series = cell_series(r_daily, r_model, horizons)
        common = c_series.index.intersection(r_series.index)
        diff = (c_series.loc[common] - r_series.loc[common])
        for period in ("P1", "P2", "P1+P2"):
            mask = period_mask(pd.DatetimeIndex(common), period)
            mean, low, high = bootstrap(diff.to_numpy()[mask], BLOCKS[cell])
            report["paired"][f"{cell}/{period}"] = {"difference": mean, "low": low, "high": high,
                                                    "days": int(mask.sum())}
            c_mask = period_mask(pd.DatetimeIndex(c_series.index), period)
            c_mean, c_low, c_high = bootstrap(c_series.to_numpy()[c_mask], BLOCKS[cell])
            report["cells"][f"{cell}/{period}"] = {"rank_ic": c_mean, "low": c_low, "high": c_high}
        if c_residual is not None:
            series = cell_series(c_residual, c_model, horizons)
            for period in ("P1", "P2", "P1+P2"):
                mask = period_mask(pd.DatetimeIndex(series.index), period)
                mean, low, high = bootstrap(series.to_numpy()[mask], BLOCKS[cell])
                report["residual_cells"][f"{cell}/{period}"] = {"rank_ic": mean, "low": low,
                                                                "high": high}
        else:
            # Older downloads lack the residual daily file; results.json carries the same cells.
            for row in _c_results.get("cells_residual", []):
                if row["model"] == c_model and row["cell"] == cell:
                    report["residual_cells"][f"{cell}/{row['period']}"] = {
                        "rank_ic": row["rank_ic"], "low": row["ci95_low"], "high": row["ci95_high"],
                        "source": "results.json",
                    }
    cond_i = report["residual_cells"].get("holding/P2", {}).get("low", math.nan)
    cond_ii = report["paired"]["short/P1+P2"]["low"]
    report["H-F"] = {
        "condition_i_p2_holding_residual_low": cond_i,
        "condition_i_passed": bool(np.isfinite(cond_i) and cond_i > 0),
        "condition_ii_short_paired_low": cond_ii,
        "condition_ii_passed": bool(np.isfinite(cond_ii) and cond_ii > -0.01),
    }
    verdict = report["H-F"]
    verdict["passed"] = verdict["condition_i_passed"] and verdict["condition_ii_passed"]
    output = args.output or (args.candidate / "label-variant-score.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["H-F"], indent=2))
    for key in ("short/P1+P2", "holding/P1+P2", "holding/P2"):
        print(key, "paired", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in report["paired"][key].items()},
              "residual",
              {k: round(v, 4) for k, v in report["residual_cells"].get(key, {}).items()})
    print("written", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
