"""One-date pilot: speed and agreement of local inference with a RunPod export.

A diagnostic, not the pre-registered parity check of protocol section 13.7. It runs the
Taiwan rows of one signal date, chosen through baseline metadata that shares the export's
row order, and prints rows per second plus the cross-sectional Spearman between local
and reference predictions for each market and horizon.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from localinfer import predict


def date_positions(metadata: np.ndarray, date: str | None) -> tuple[str, np.ndarray]:
    """Row positions of the Taiwan rows of one date (default: the first Taiwan date)."""
    days = np.array([str(value)[:10] for value in metadata["date"]])
    taiwan = np.isin(metadata["market"], predict.TAIWAN)
    chosen = date or str(min(days[taiwan]))
    positions = np.flatnonzero(taiwan & (days == chosen))
    if positions.size == 0:
        raise ValueError(f"No Taiwan rows on {chosen}")
    return chosen, positions


def agreement(local: np.ndarray, reference: np.ndarray, markets: np.ndarray) -> dict:
    """Cross-sectional Spearman per market and horizon, summarized."""
    output = {}
    for market in predict.TAIWAN:
        selected = markets == market
        if selected.sum() < 3:
            continue
        values = [
            pd.Series(local[selected, h]).rank().corr(pd.Series(reference[selected, h]).rank())
            for h in range(local.shape[1])
        ]
        output[market] = {"rows": int(selected.sum()), "min": float(np.min(values)),
                          "median": float(np.median(values))}
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bar-store", type=Path, required=True)
    parser.add_argument("--sample-universe", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Baseline inputs/<split>/metadata.npy (the export's row order).")
    parser.add_argument("--reference", type=Path, required=True,
                        help="RunPod export <split>/predictions.npy of the same checkpoint.")
    parser.add_argument("--split", choices=predict.SPLITS, default="validation")
    parser.add_argument("--date", help="Signal date YYYY-MM-DD (default: first Taiwan date).")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=predict.PRECISIONS, default="fp32")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    device = predict.select_device(args.device)
    metadata = np.load(args.metadata)
    date, positions = date_positions(metadata, args.date)
    config, _state = predict.load_config(Path(args.checkpoint).resolve(strict=True))
    from stock_forecasting.checkpointing import load_checkpoint
    from stock_forecasting.factory import build_model_bundle
    from stock_forecasting.training import (
        ModelBatchCollator,
        forward_batch,
        iter_device_batches,
    )

    bundle = build_model_bundle(config, device)
    predict.apply_precision(bundle.model, args.precision)
    load_checkpoint(args.checkpoint, bundle.model)
    bundle.model.requires_grad_(False)
    bundle.model.eval()
    dataset, base = predict.open_split(config, args.bar_store, args.sample_universe,
                                       args.split, cache_size=64)
    if len(base) != len(metadata):
        raise ValueError("The local split and the baseline metadata have different lengths")
    loader = DataLoader(Subset(dataset, positions.tolist()), batch_size=args.batch_size,
                        collate_fn=ModelBatchCollator.for_model(config, training=False))
    values, symbols, batch_seconds = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        batch_started = time.perf_counter()
        for batch in iter_device_batches(loader, device):
            with predict.autocast(device, args.precision):
                output = forward_batch(bundle, batch, config, device).alpha_point
            values.append(output.float().cpu().numpy())
            symbols.extend(str(value) for value in batch["symbols"])
            batch_seconds.append(time.perf_counter() - batch_started)
            batch_started = time.perf_counter()
    seconds = time.perf_counter() - started
    if symbols != [str(value) for value in metadata["symbol"][positions]]:
        raise ValueError("Local rows do not follow the baseline metadata order")
    local = np.concatenate(values)
    reference = np.asarray(np.load(args.reference, mmap_mode="r")[positions])
    summary = agreement(local, reference, np.asarray(metadata["market"][positions]))
    steady = sum(batch_seconds[1:])
    report = {
        "date": date, "rows": len(positions), "device": device.type,
        "precision": args.precision, "seconds": round(seconds, 1),
        "rows_per_second": round(len(positions) / seconds, 2),
        "rows_per_second_after_first_batch": (
            round((len(positions) - len(values[0])) / steady, 2) if steady > 0 else None
        ),
        "mean_abs_difference": float(np.abs(local - reference).mean()),
        "spearman": summary,
        "all_groups_at_least_0_999": all(item["min"] >= 0.999 for item in summary.values()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
