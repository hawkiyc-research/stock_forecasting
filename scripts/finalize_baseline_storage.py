#!/usr/bin/env python3
"""Finalize completed baseline storage on the mounted volume; default to dry run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_forecasting.baseline_result_storage import finalize_baseline_storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--baseline-id", action="append")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--read-mib-per-second", type=int, default=64)
    args = parser.parse_args(argv)
    if not os.environ.get("RUNPOD_POD_ID"):
        parser.error("Run this maintenance command on the Pod with the network volume mounted")
    volume = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    if not volume.is_mount():
        parser.error("The network volume is not mounted")
    roots = volume / "baselines"
    names = args.baseline_id or sorted(
        path.name for path in roots.iterdir() if (path / "complete.json").is_file()
    )
    if not names or len(names) > 64:
        parser.error("Expected between 1 and 64 completed baselines")
    for name in names:
        if Path(name).name != name or not name.startswith("baseline-"):
            parser.error("Invalid baseline ID")
        result = finalize_baseline_storage(
            roots / name, apply=args.apply, workers=args.workers,
            read_mib_per_second=args.read_mib_per_second,
            progress=lambda message: print(message, flush=True),
        )
        summary = {key: value for key, value in result.items() if key != "payload"}
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
