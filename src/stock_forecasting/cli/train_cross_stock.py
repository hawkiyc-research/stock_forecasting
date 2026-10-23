"""Train a cross-stock transformer experiment (protocol 13.13 D0 / A1-A4) on a RunPod GPU Pod.

Builds (or reuses) the cross-stock sample cache of every split from the ready feature and
rank-label stores, trains the experiment over the quarterly walk-forward folds with the given
seeds, and writes the B0-compatible outputs plus the paired comparison against the LightGBM
baseline under ``--output-root``.

With ``--label-variant`` (protocol 13.16) the residualized store's ranks replace the standard
labels, the scores are also evaluated against the residual targets, and the baseline must be
the B0 run trained on the same variant.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.cross_stock_training import (
    BASELINE_MODEL,
    DEFAULT_SEEDS,
    EXPERIMENTS,
    run_cross_stock,
    table_from_caches,
    write_xs_manifest,
)
from stock_forecasting.data.feature_library import DEFAULT_SPEC_PATH, load_spec
from stock_forecasting.data.xs_cache import ensure_xs_cache, open_xs_cache
from stock_forecasting.walk_forward import DEFAULT_SCHEDULE_PATH, load_schedule
from stock_forecasting.walk_forward_lgbm import build_folds

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/a_rank64.yaml")
MAX_WORKERS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(os.environ.get("FEATURE_EXPERIMENT_CONFIG", str(DEFAULT_EXPERIMENT_CONFIG))),
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--experiment", default="t-xs-scratch", choices=sorted(EXPERIMENTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, required=True, help="New directory.")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Directory holding the B0 daily_rank_ic.parquet for the paired comparison.",
    )
    parser.add_argument("--workers", type=int, help=f"Cache-build processes, 1-{MAX_WORKERS}.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, nargs="*", help="Fold indexes to run (smoke).")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-groups-per-epoch", type=int, help="Smoke runs only.")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument(
        "--resume-root", type=Path, help="Earlier output root whose finished folds are reused."
    )
    parser.add_argument(
        "--offline-encoders",
        action="store_true",
        help="Verify encoder weights in the local Hugging Face cache instead of downloading.",
    )
    parser.add_argument(
        "--label-variant",
        type=Path,
        help="Residualized label spec (configs/labels/NAME.json) whose ready store supplies "
        "the ranks and residual targets (protocol 13.16).",
    )
    return parser


def load_baseline(
    root: Path, label_variant: str | None
) -> tuple[str, pd.DataFrame, pd.DataFrame | None]:
    """The baseline model name and daily Rank IC series, checked against the label variant."""
    path = root / "daily_rank_ic.parquet"
    if not path.is_file():
        raise SystemExit(f"Baseline daily Rank IC is missing: {path}")
    model, baseline_variant = BASELINE_MODEL, None
    results_path = root / "results.json"
    if results_path.is_file():
        results = json.loads(results_path.read_text())
        model = str(results.get("model", BASELINE_MODEL))
        baseline_variant = results.get("label_variant")
    if (baseline_variant or None) != (label_variant or None):
        raise SystemExit(
            f"Baseline {root} was trained on label variant {baseline_variant!r}, "
            f"this run uses {label_variant!r}"
        )
    daily = pd.read_parquet(path)
    if model not in set(daily["model"]):
        raise SystemExit(f"Baseline file has no {model} rows")
    residual = None
    residual_path = root / "daily_rank_ic_residual.parquet"
    if label_variant is not None and residual_path.is_file():
        residual = pd.read_parquet(residual_path)
    return model, daily, residual


def main(argv: list[str] | None = None) -> int:
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.sample_universe import open_clean_dataset, validated_split_roots

    args = build_parser().parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS}")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise SystemExit("--seeds must be distinct non-negative integers")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"--output-root must be a new or empty directory: {args.output_root}")
    config = ExperimentConfig.from_yaml(args.experiment_config)
    spec = load_spec(args.spec)
    schedule = load_schedule(args.schedule)
    roots = validated_split_roots(config)
    min_group_size = int(config.model.rank_min_group_size)
    label_spec, variant_name = None, None
    if args.label_variant is not None:
        from stock_forecasting.data.residual_labels import load_label_spec

        label_spec = load_label_spec(args.label_variant)
        variant_name = str(label_spec["name"])
    caches, cache_roots, variants, variant_roots = {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        source = open_clean_dataset(config, split, validated_root=roots[split])
        try:
            root = ensure_xs_cache(source, spec, min_group_size, workers=args.workers)
            caches[split] = open_xs_cache(root)
            if label_spec is not None:
                from stock_forecasting.data.xs_label_variant import (
                    ensure_label_variant,
                    open_label_variant,
                )

                variant_root = ensure_label_variant(
                    caches[split], source, min_group_size, label_spec
                )
                variants[split] = open_label_variant(variant_root)
                variant_roots[split] = str(variant_root)
        finally:
            source.close()
        cache_roots[split] = str(root)
    table = table_from_caches(caches, label_variants=variants if label_spec is not None else None)
    encoder_name = EXPERIMENTS[args.experiment].get("encoder")
    encoder_records = None
    if encoder_name is not None:
        import torch

        from stock_forecasting.cross_stock_training import resolve_device
        from stock_forecasting.encoders import (
            ENCODERS,
            ensure_embeddings,
            load_encoder,
            open_embeddings,
            prefetch,
        )

        device = resolve_device(args.device)
        revisions = prefetch(ENCODERS[encoder_name], local_files_only=args.offline_encoders)
        kronos_kwargs = (
            {
                "kronos_source_root": str(config.model.kronos_source_root),
                "kronos_source_revision": str(config.model.kronos_source_revision),
            }
            if ENCODERS[encoder_name].family == "kronos"
            else {}
        )
        encoder = load_encoder(encoder_name, device=device, **kronos_kwargs)
        stores = []
        for split in ("train", "validation", "test"):
            root = ensure_embeddings(
                caches[split],
                encoder_name,
                encoder=encoder,
                device=device,
                batch_size=args.embedding_batch_size,
                revisions=revisions,
            )
            stores.append(open_embeddings(root))
        del encoder
        table.encoded = torch.from_numpy(
            np.concatenate([np.asarray(store, dtype=np.float32) for store in stores])
        )
        encoder_records = {
            "encoder": encoder_name,
            "revisions": revisions,
            "dim": int(table.encoded.shape[1]),
        }
    folds = build_folds(schedule)
    if args.folds:
        wanted = set(args.folds)
        folds = [fold for fold in folds if fold.index in wanted]
        if not folds:
            raise SystemExit("--folds selected no fold of the schedule")
    baseline_model, baseline_daily, baseline_residual = BASELINE_MODEL, None, None
    if args.baseline_root is not None:
        baseline_model, baseline_daily, baseline_residual = load_baseline(
            args.baseline_root, variant_name
        )
    params = {}
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    if args.patience is not None:
        params["patience"] = args.patience
    results = run_cross_stock(
        table,
        experiment=args.experiment,
        folds=folds,
        schedule=schedule,
        output_root=args.output_root,
        seeds=tuple(args.seeds),
        min_group_size=min_group_size,
        device=args.device,
        params=params,
        baseline_daily=baseline_daily,
        max_groups_per_epoch=args.max_groups_per_epoch,
        resume_root=args.resume_root,
        baseline_model=baseline_model,
        baseline_daily_residual=baseline_residual,
        label_variant=variant_name,
    )
    write_xs_manifest(
        args.output_root,
        results=results,
        experiment_config=str(args.experiment_config),
        schedule=str(args.schedule),
        caches=cache_roots,
        label_variant=variant_name,
        label_variant_arrays=variant_roots or None,
        baseline_root=None if args.baseline_root is None else str(args.baseline_root),
        encoder=encoder_records,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "experiment": args.experiment,
                "label_variant": variant_name,
                "parameters": results["parameters"],
                "device": results["device"],
                "fits": results["fits"],
                "cells": [
                    c
                    for c in results["cells"]
                    if c["model"] == args.experiment and c["period"] == "P1+P2"
                ],
                "cells_residual": [
                    c
                    for c in results["cells_residual"]
                    if c["model"] == args.experiment and c["period"] == "P1+P2"
                ],
                "seed_stability": [s for s in results["seed_stability"] if s["period"] == "P1+P2"],
                "decision": results["decision"],
                "seconds": round(results["seconds"], 1),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
