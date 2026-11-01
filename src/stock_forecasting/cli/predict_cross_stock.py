"""Score signal dates with the saved weights of a cross-stock walk-forward launch.

The 2026-11-02 rule: every launch keeps the best-epoch weights of the folds its P3 needs under
``models/fold-XX/seed-N.pt``. This command builds the same in-memory sample table the trainer
uses (xs caches, label variant, cached frozen-encoder embeddings for a D1 experiment), takes
every (date, market) group whose signal date lies in ``[--start, --end)`` and scores it with
the model fold whose prediction window holds that date (dates after the last window use the
newest model fold; ``--fold`` forces one). One row per sample and model (each seed and the
mean ensemble); ensemble rows carry per-group ranks, 1 = highest score.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from stock_forecasting.cli.train_cross_stock import MAX_WORKERS, build_sample_table
from stock_forecasting.cross_stock_training import score_table_with_launch
from stock_forecasting.data.feature_library import DEFAULT_SPEC_PATH, load_spec

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/a_rank64.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--launch",
        type=Path,
        required=True,
        help="A finished launch directory holding results.json and models/ (the volume's "
        "diagnostics/xs/<launch-id> or a downloaded artifacts/xs/<launch-id>).",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(os.environ.get("FEATURE_EXPERIMENT_CONFIG", str(DEFAULT_EXPERIMENT_CONFIG))),
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument(
        "--label-variant",
        type=Path,
        help="The residualized label spec the launch was trained with (configs/labels/NAME.json).",
    )
    parser.add_argument("--start", required=True, help="First signal date (ISO, inclusive).")
    parser.add_argument("--end", required=True, help="End signal date (ISO, exclusive).")
    parser.add_argument("--fold", type=int, help="Score every date with this model fold.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Parquet file for the scores (default artifacts/predictions/<launch>/"
        "xs-<start>-<end>.parquet); a JSON record is written next to it.",
    )
    parser.add_argument("--workers", type=int, help=f"Cache-build processes, 1-{MAX_WORKERS}.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument(
        "--offline-encoders",
        action="store_true",
        help="Verify encoder weights in the local Hugging Face cache instead of downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from stock_forecasting.config import ExperimentConfig

    args = build_parser().parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS}")
    results_path = args.launch / "results.json"
    if not results_path.is_file():
        raise SystemExit(f"--launch must be a finished launch (results.json): {args.launch}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    experiment = str(results["model"])
    if (results.get("label_variant") or None) != (
        None if args.label_variant is None else Path(args.label_variant).stem
    ):
        raise SystemExit(
            f"The launch was trained on label variant {results.get('label_variant')!r}; "
            "pass the same --label-variant"
        )
    config = ExperimentConfig.from_yaml(args.experiment_config)
    spec = load_spec(args.spec)
    table, _cache_roots, _encoder, _variant, _roots = build_sample_table(
        config,
        spec,
        experiment=experiment,
        label_variant=args.label_variant,
        workers=args.workers,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        offline_encoders=args.offline_encoders,
    )
    scores, record = score_table_with_launch(
        table, args.launch, start=args.start, end=args.end, device=args.device, fold=args.fold
    )
    output = args.output or (
        Path("artifacts/predictions") / args.launch.name / f"xs-{args.start}-{args.end}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output, index=False)
    record.update(
        {
            "output": str(output),
            "experiment_config": str(args.experiment_config),
            "spec": str(args.spec),
            "label_variant": None if args.label_variant is None else str(args.label_variant),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "launch": record["launch"],
                "model": record["model"],
                "rows": record["rows"],
                "folds_by_date": record["folds_by_date"],
                "execution": record["execution"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
