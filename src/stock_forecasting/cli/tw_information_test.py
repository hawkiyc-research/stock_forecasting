"""Run the Phase 1 Taiwan information test on a RunPod CPU Pod.

Reads the cleaned samples of the experiment data group and the auxiliary tables,
fits GBDT models per feature set and horizon, and writes report.json plus a
bilingual summary.md. Training, checkpoints and baselines are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from stock_forecasting.research.tw_information import run
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

DEFAULT_CONFIG = Path("configs/research/tw_information_test.json")
# Each spawn worker imports torch and opens the bar-store indexes.
WORKER_MEMORY_BYTES = 1024**3
MAX_WORKERS = 32


def default_workers() -> int:
    """Bound panel workers by visible CPUs (one kept free) and half of available memory."""
    configured = os.environ.get("TW_INFORMATION_WORKERS")
    if configured:
        workers = int(configured)
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"TW_INFORMATION_WORKERS must be 1-{MAX_WORKERS}")
        return workers
    by_memory = detect_available_memory().available_bytes // 2 // WORKER_MEMORY_BYTES
    return max(1, min(MAX_WORKERS, detect_visible_cpu_count() - 1, by_memory))


def default_output_root() -> Path:
    volume = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return volume / "diagnostics" / "tw-information-test" / f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--auxiliary-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/runpod-volume/data")) / "auxiliary",
    )
    parser.add_argument("--output-root", type=Path, help="New directory for this run.")
    parser.add_argument("--workers", type=int, help="Panel workers (default from CPU and memory).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers if args.workers is not None else default_workers()
    if not 1 <= workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1-{MAX_WORKERS}")
    if not (args.auxiliary_root / "auxiliary-manifest.json").is_file():
        raise SystemExit("Auxiliary manifest is missing; run prepare_auxiliary_data first")
    output_root = args.output_root or default_output_root()
    report = run(args.config, args.auxiliary_root, output_root, workers=workers)
    print(json.dumps({
        "output_root": str(output_root),
        "information_found": report["criteria"]["information_found"],
        "skipped_feature_sets": report["skipped_feature_sets"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
