"""Build rank labels and the feature library for every split of the selected dataset.

Runs on a RunPod CPU Pod (the ``cpu-features`` tmux workflow). Each split's cleaned sample
universe is opened read-only; the gauss-rank labels (protocol 13.10) and the fixed feature
library (protocol 13.12) are written as Parquet stores beside the split's bar store, and one
``features-manifest.json`` records where they are. Nothing about training, baselines or
checkpoints is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from stock_forecasting.data.feature_library import (
    DEFAULT_SPEC_PATH,
    ensure_features,
    feature_names,
    load_spec,
    spec_identity,
)
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.data.rank_labels import ensure_rank_labels
from stock_forecasting.data.residual_labels import (
    ensure_residual_labels,
    load_label_spec,
)
from stock_forecasting.data.residual_labels import (
    spec_identity as label_spec_identity,
)
from stock_forecasting.data.sample_universe import open_clean_dataset, validated_split_roots

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/a_rank64.yaml")
MANIFEST_NAME = "features-manifest.json"
SPLITS = ("train", "validation", "test")
MAX_WORKERS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(os.environ.get("FEATURE_EXPERIMENT_CONFIG", str(DEFAULT_EXPERIMENT_CONFIG))),
        help="Experiment YAML that names the dataset paths, window, horizons and group size.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-root", type=Path, required=True, help="New directory.")
    parser.add_argument("--workers", type=int, help=f"Process workers, 1-{MAX_WORKERS}.")
    parser.add_argument(
        "--label-variant",
        type=Path,
        help="Residualized label spec (protocol 13.16), built beside the standard store.",
    )
    parser.add_argument(
        "--chip-data-root",
        type=Path,
        default=os.environ.get("TW_CHIP_DATA_ROOT") or None,
        help="tw_chip_data/data (curated layer) for chip feature specs (protocol 13.18); "
             "defaults to $TW_CHIP_DATA_ROOT.",
    )
    return parser


def _volume_relative(path: Path) -> str | None:
    """Path relative to the network volume, for the local download script."""
    volume = os.environ.get("NETWORK_VOLUME_ROOT")
    if not volume:
        return None
    try:
        return str(path.resolve().relative_to(Path(volume).resolve()))
    except ValueError:
        return None


def run(
    config,
    spec: dict,
    output_root: Path,
    *,
    workers: int | None = None,
    label_variant: dict | None = None,
) -> dict:
    """Build both stores for every split and write the manifest; idempotent per identity.

    With ``label_variant`` the residualized label store of that spec is built as well; the
    standard rank-label store stays the reference the models were registered on.
    """
    started = time.monotonic()
    roots = validated_split_roots(config)
    splits = {}
    for split in SPLITS:
        source = open_clean_dataset(config, split, validated_root=roots[split])
        try:
            labels = ensure_rank_labels(source, config.model.rank_min_group_size, workers=workers)
            features = ensure_features(source, spec, workers=workers)
            splits[split] = {
                "source": str(source.root),
                "samples": len(source),
                "sample_universe": source.sample_universe_identity,
                "rank_labels": {
                    "root": str(labels),
                    "volume_relative": _volume_relative(labels),
                    "manifest_sha256": sha256_file(labels / "rank-labels.json"),
                },
                "features": {
                    "root": str(features),
                    "volume_relative": _volume_relative(features),
                    "manifest_sha256": sha256_file(features / "features.json"),
                },
            }
            if label_variant is not None:
                residual = ensure_residual_labels(
                    source, config.model.rank_min_group_size, label_variant, workers=workers
                )
                splits[split]["residual_labels"] = {
                    "root": str(residual),
                    "volume_relative": _volume_relative(residual),
                    "manifest_sha256": sha256_file(residual / "rank-labels.json"),
                    "variant": label_variant["name"],
                }
        finally:
            source.close()
    payload = {
        "schema_version": "1.0",
        "kind": "feature-library-build",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_profile": config.data.dataset_profile,
        "dataset_root": str(Path(config.data.bar_store_path).resolve(strict=False).parent.parent),
        "window_size": int(config.data.input_length),
        "h_start": int(config.data.h_start),
        "rank_min_group_size": int(config.model.rank_min_group_size),
        "spec": {**spec_identity(spec), "columns": len(feature_names(spec))},
        "label_variant": label_spec_identity(label_variant) if label_variant else None,
        "splits": splits,
        "workers": workers,
        "seconds": time.monotonic() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / MANIFEST_NAME, payload)
    return payload


def _dataset_end_exclusive(config) -> str | None:
    """The dataset request's end date from the dataset manifest, when it is readable."""
    try:
        manifest = json.loads(Path(config.data.manifest_path).read_text(encoding="utf-8"))
        value = manifest.get("date_range", {}).get("end_exclusive")
    except (OSError, ValueError, AttributeError):
        return None
    return str(value) if value else None


def main(argv: list[str] | None = None) -> int:
    from stock_forecasting.config import ExperimentConfig

    args = build_parser().parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"--output-root must be a new or empty directory: {args.output_root}")
    if args.chip_data_root is not None:
        os.environ["TW_CHIP_DATA_ROOT"] = str(args.chip_data_root)
    config = ExperimentConfig.from_yaml(args.experiment_config)
    # Fail before any store is built when the pinned calendar lists a session the data never
    # traded (an unlisted exchange closure) or misses a day it did: either one silently empties
    # months of samples through asset_market_session_gap (see data/calendar_audit.py).
    from stock_forecasting.data.calendar_audit import require_consistent_calendar

    calendar_report = require_consistent_calendar(
        Path(config.data.raw_path), end_exclusive=_dataset_end_exclusive(config)
    )
    print(json.dumps({"calendar_audit": calendar_report}, indent=2), flush=True)
    spec = load_spec(args.spec)
    label_variant = load_label_spec(args.label_variant) if args.label_variant else None
    payload = run(
        config, spec, args.output_root, workers=args.workers, label_variant=label_variant
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "columns": payload["spec"]["columns"],
                "samples": {split: entry["samples"] for split, entry in payload["splits"].items()},
                "seconds": round(payload["seconds"], 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
