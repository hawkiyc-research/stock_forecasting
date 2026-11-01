"""Score signal dates with the saved models of a LightGBM walk-forward launch.

The 2026-11-02 rule: every launch keeps the models of the folds its P3 needs under
``models/fold-XX/``. This command assembles the feature table of the selected dataset (the
same stores the trainer reads), takes every sample whose signal date lies in ``[--start,
--end)`` and scores it with the model fold whose prediction window holds that date (dates after
the last window use the newest model fold; ``--fold`` forces one). One row per sample and
model (each seed and the mean ensemble); ensemble rows carry per-(date, market) ranks,
1 = highest score. Runs wherever the launch and the feature stores are (a Pod, or a machine
with the stores downloaded).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from stock_forecasting.data.feature_library import DEFAULT_SPEC_PATH, feature_names, load_spec
from stock_forecasting.walk_forward_lgbm import assemble_table, score_table_with_launch

DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/a_rank64.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--launch",
        type=Path,
        required=True,
        help="A finished launch directory holding results.json and models/ (the volume's "
        "diagnostics/lgbm/<launch-id> or a downloaded artifacts/lgbm/<launch-id>).",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(os.environ.get("FEATURE_EXPERIMENT_CONFIG", str(DEFAULT_EXPERIMENT_CONFIG))),
        help="Experiment YAML that names the dataset paths, window, horizons and group size.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument(
        "--label-variant",
        type=Path,
        help="The residualized label spec the launch was trained with (its store supplies the "
        "table; the scores do not depend on labels).",
    )
    parser.add_argument(
        "--chip-data-root",
        type=Path,
        default=os.environ.get("TW_CHIP_DATA_ROOT") or None,
        help="tw_chip_data/data for chip feature specs; defaults to $TW_CHIP_DATA_ROOT.",
    )
    parser.add_argument("--start", required=True, help="First signal date (ISO, inclusive).")
    parser.add_argument("--end", required=True, help="End signal date (ISO, exclusive).")
    parser.add_argument("--fold", type=int, help="Score every date with this model fold.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Parquet file for the scores (default artifacts/predictions/<launch>/"
        "lgbm-<start>-<end>.parquet); a JSON record is written next to it.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Table-assembly processes, 1-8.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.sample_universe import validated_split_roots

    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be 1-8")
    if not (args.launch / "results.json").is_file():
        raise SystemExit(f"--launch must be a finished launch (results.json): {args.launch}")
    if args.chip_data_root is not None:
        os.environ["TW_CHIP_DATA_ROOT"] = str(args.chip_data_root)
    config = ExperimentConfig.from_yaml(args.experiment_config)
    spec = load_spec(args.spec)
    label_variant = None
    if args.label_variant is not None:
        from stock_forecasting.data.residual_labels import load_label_spec

        label_variant = load_label_spec(args.label_variant)
    roots = validated_split_roots(config)
    table = assemble_table(config, spec, roots, workers=args.workers, label_variant=label_variant)
    scores, record = score_table_with_launch(
        table,
        args.launch,
        feature_columns=feature_names(spec),
        start=args.start,
        end=args.end,
        fold=args.fold,
    )
    output = args.output or (
        Path("artifacts/predictions") / args.launch.name / f"lgbm-{args.start}-{args.end}.parquet"
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
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
