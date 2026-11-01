"""Train the B0 ``t-lgbm-alpha`` walk-forward baseline (protocol 13.13) on a RunPod CPU Pod.

Reads the selected dataset's ready feature and rank-label stores (built by ``cpu-features``),
trains one LightGBM regressor per fold, horizon and seed with early stopping on the fold
validation's daily Rank IC, and writes per-sample predictions, raw targets, daily Rank IC
series, the cell summary and a manifest under ``--output-root``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from stock_forecasting.data.feature_library import DEFAULT_SPEC_PATH, feature_names, load_spec
from stock_forecasting.walk_forward import DEFAULT_SCHEDULE_PATH, load_schedule
from stock_forecasting.walk_forward_lgbm import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_SEEDS,
    MAX_WORKERS,
    MODEL_NAME,
    assemble_table,
    build_folds,
    locate_stores,
    run_walk_forward,
    write_manifest,
)

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/a_rank64.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(os.environ.get("FEATURE_EXPERIMENT_CONFIG", str(DEFAULT_EXPERIMENT_CONFIG))),
        help="Experiment YAML that names the dataset paths, window, horizons and group size.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output-root", type=Path, required=True, help="New directory.")
    parser.add_argument("--scratch", type=Path, help="Fold matrices (default: output root).")
    parser.add_argument(
        "--resume-root",
        type=Path,
        action="append",
        help="Earlier output root whose partial folds are reused (repeatable: the trainer "
        "takes, per fold, the newest candidate of the same labels, feature spec, schedule and "
        "dataset, and stamps every source it used).",
    )
    parser.add_argument(
        "--extend-from",
        type=Path,
        help="Finished launch of the same model and labels to carry on (2026-11-02 rule): folds "
        "with the same training window are linked (same prediction window) or scored by their "
        "saved models (another window); P3 folds without saved models, and new folds, are "
        "trained; that launch is stamped superseded_by.json and loses its model files.",
    )
    parser.add_argument("--workers", type=int, help=f"Training processes, 1-{MAX_WORKERS}.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--early-stopping", type=int, default=DEFAULT_EARLY_STOPPING)
    parser.add_argument("--folds", type=int, nargs="*", help="Fold indexes to run (smoke).")
    parser.add_argument("--horizons", type=int, nargs="*", help="Horizons to run (smoke).")
    parser.add_argument(
        "--label-variant",
        type=Path,
        help="Residualized label spec (protocol 13.16): train on that store's ranks.",
    )
    parser.add_argument(
        "--chip-data-root",
        type=Path,
        default=os.environ.get("TW_CHIP_DATA_ROOT") or None,
        help="tw_chip_data/data (curated layer) for chip feature specs (protocol 13.18); "
        "defaults to $TW_CHIP_DATA_ROOT. Exported so the store identity can be resolved.",
    )
    return parser


def resume_context(schedule: dict, stores: list) -> str:
    """Schedule name and dataset identity (the three splits' bar-store digests) a partial fold
    belongs to; a resume never reuses folds of another schedule or dataset (protocol 13.20)."""
    digests = [
        str(store.feature_identity.get("source_manifest_sha256", ""))[:16] for store in stores
    ]
    return f"{schedule['name']}|{'|'.join(digests)}"


def main(argv: list[str] | None = None) -> int:
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.sample_universe import validated_split_roots

    args = build_parser().parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS}")
    if args.max_rounds < 1 or args.early_stopping < 1:
        raise SystemExit("--max-rounds and --early-stopping must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise SystemExit("--seeds must be distinct non-negative integers")
    if args.chip_data_root is not None:
        os.environ["TW_CHIP_DATA_ROOT"] = str(args.chip_data_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"--output-root must be a new or empty directory: {args.output_root}")
    config = ExperimentConfig.from_yaml(args.experiment_config)
    spec = load_spec(args.spec)
    label_variant = None
    if args.label_variant is not None:
        from stock_forecasting.data.residual_labels import load_label_spec

        label_variant = load_label_spec(args.label_variant)
    schedule = load_schedule(args.schedule)
    roots = validated_split_roots(config)
    stores = [
        locate_stores(config, spec, split, roots[split], label_variant=label_variant)
        for split in ("train", "validation", "test")
    ]
    folds = build_folds(schedule)
    if args.folds:
        wanted = set(args.folds)
        folds = [fold for fold in folds if fold.index in wanted]
        if not folds:
            raise SystemExit("--folds selected no fold of the schedule")
    horizons = [int(h) for h in config.data.alpha_horizons]
    if args.horizons:
        missing = sorted(set(args.horizons) - set(horizons))
        if missing:
            raise SystemExit(f"--horizons outside the configured horizons: {missing}")
        horizons = [h for h in horizons if h in set(args.horizons)]
    workers = args.workers or 1
    table = assemble_table(
        config, spec, roots, workers=max(1, min(workers, 8)), label_variant=label_variant
    )
    variant_name = label_variant["name"] if label_variant else None
    model_name = f"{MODEL_NAME}-{variant_name}" if variant_name else MODEL_NAME
    results = run_walk_forward(
        table,
        feature_columns=feature_names(spec),
        horizons=horizons,
        folds=folds,
        schedule=schedule,
        output_root=args.output_root,
        seeds=tuple(args.seeds),
        workers=args.workers,
        max_rounds=args.max_rounds,
        early_stopping=args.early_stopping,
        min_group_size=int(config.model.rank_min_group_size),
        scratch=args.scratch,
        resume_root=args.resume_root,
        model_name=model_name,
        label_variant=variant_name,
        resume_context=resume_context(schedule, stores),
        extend_from=args.extend_from,
        feature_spec=str(spec.get("name") or args.spec.stem),
    )
    write_manifest(
        args.output_root,
        config_path=args.experiment_config,
        spec=spec,
        schedule_path=args.schedule,
        stores=stores,
        results=results,
        label_variant=(
            {"spec": str(args.label_variant), "name": variant_name} if variant_name else None
        ),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "backend": results["backend"],
                "folds": len(folds),
                "horizons": horizons,
                "seeds": list(args.seeds),
                "model_folds": results["model_folds"],
                "fits": results["fits"],
                "cells": [
                    cell
                    for cell in results["cells"]
                    if cell["model"] == results["model"] and cell["period"] == "P1+P2"
                ],
                "seconds": round(results["seconds"], 1),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
