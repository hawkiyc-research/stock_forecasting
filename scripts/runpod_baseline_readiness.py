#!/usr/bin/env python3
"""Compatibility entrypoint for the shared, model-independent dataset gate."""

import runpy
import subprocess
import sys
from pathlib import Path

# Keep existing script callers and dependency-free tests on the same implementation.
globals().update({
    key: value
    for key, value in runpy.run_path(
        str(Path(__file__).with_name("runpod_dataset_readiness.py"))
    ).items()
    if not key.startswith("__")
})

if __name__ == "__main__":
    try:
        raise SystemExit(globals()["main"]())
    except (
        ValueError, RuntimeError, KeyError, TypeError, OSError, subprocess.SubprocessError
    ) as error:
        print(f"Baseline data readiness failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
