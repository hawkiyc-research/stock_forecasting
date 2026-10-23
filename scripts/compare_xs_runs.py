#!/usr/bin/env python3
"""Pair two downloaded cross-stock runs on common signal dates (protocol 13.13 D1 vs D0).

The GPU job pairs every experiment against the LightGBM B0 of the same label variant. The D1
question is instead "does the frozen encoder beat D0 on the same labels", so this script reads
two artifacts/xs/<launch>/ directories and reports model - baseline per cell and period, on the
raw alpha target and, when both runs carry it, on the residual target, with the protocol's
block-bootstrap intervals.

Run with the inference environment (torch + pandas):
  PYTHONPATH=src inference/.venv/bin/python scripts/compare_xs_runs.py \
      --model artifacts/xs/<d1-launch> --baseline artifacts/xs/<d0-launch>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_forecasting.cross_stock_training import paired_comparison
from stock_forecasting.walk_forward import DEFAULT_SCHEDULE_PATH, load_schedule


def load(root: Path, name: str) -> tuple[str, pd.DataFrame | None]:
    results = json.loads((root / "results.json").read_text())
    path = root / name
    if not path.is_file():
        return str(results["model"]), None
    daily = pd.read_parquet(path)
    daily["date"] = pd.to_datetime(daily["date"])
    return str(results["model"]), daily


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="artifacts/xs/<launch> of D1")
    parser.add_argument("--baseline", type=Path, required=True, help="artifacts/xs/<launch> of D0")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output", type=Path, help="JSON report (default: model root)")
    args = parser.parse_args()
    schedule = load_schedule(args.schedule)
    model_results = json.loads((args.model / "results.json").read_text())
    base_results = json.loads((args.baseline / "results.json").read_text())
    if model_results.get("label_variant") != base_results.get("label_variant"):
        raise SystemExit("The two runs were trained on different label variants")
    report = {
        "model_root": str(args.model),
        "baseline_root": str(args.baseline),
        "label_variant": model_results.get("label_variant"),
        "paired": {},
    }
    for target, name in (
        ("alpha", "daily_rank_ic.parquet"),
        ("residual", "daily_rank_ic_residual.parquet"),
    ):
        model_name, model_daily = load(args.model, name)
        base_name, base_daily = load(args.baseline, name)
        if model_daily is None or base_daily is None:
            continue
        rows = paired_comparison(
            model_daily, base_daily, schedule, model=model_name, baseline=base_name
        )
        report["paired"][target] = rows
        for row in rows:
            if row["period"] == "P1+P2" or row["cell"] == "holding":
                print(
                    f"{target:8s} {row['cell']:8s} {row['period']:6s} "
                    f"{row['model']} - {row['baseline']} = {row['difference']:+.4f} "
                    f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] days={row['days']}"
                )
    output = args.output or (args.model / "paired-vs-xs.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("written", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
