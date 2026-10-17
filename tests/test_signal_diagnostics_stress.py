"""Exercise production-sized artifact I/O using only bounded synthetic buffers."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_forecasting.signal_diagnostic_io import NpyReader  # noqa: E402


def write_repeated(path, descr, shape, row):
    header = repr({"descr": descr, "fortran_order": False, "shape": shape})
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
        block = row * 4096
        remaining = shape[0]
        while remaining:
            take = min(remaining, 4096)
            handle.write(block[: take * len(row)])
            remaining -= take


def peak_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=534327)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-rss-mib", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000_000 or not 1 <= args.workers <= 2:
        raise ValueError("Bound the synthetic workload to one million rows and two writers")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="signal-diagnostic-stress-") as directory:
        root = Path(directory)
        required = args.samples * (672 + 168)
        if shutil.disk_usage(root).free < required + 512 * 1024**2:
            raise OSError("Insufficient temporary disk headroom for bounded synthetic arrays")
        fields = [
            ("symbol", "<U48"),
            ("date", "<U40"),
            ("market", "<U24"),
            ("asset_type", "<U24"),
            ("provider", "<U32"),
        ]
        metadata = b"".join(
            value.ljust(width, "\0").encode("utf-32-le")
            for value, width in zip(
                ("FIXTURE", "2025-12-01T00:00:00+00:00", "US", "stock", "fixture"),
                (48, 40, 24, 24, 32),
                strict=True,
            )
        )
        workers = min(args.workers, max(1, os.cpu_count() or 1), 32 * 1024**2 // (4096 * 672 * 2))
        jobs = [
            (root / "metadata.npy", fields, (args.samples,), metadata),
            (
                root / "predictions.npy",
                "<f4",
                (args.samples, 14, 3),
                struct.pack("<42f", *([0.0, 1.0, 2.0] * 14)),
            ),
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [pool.submit(write_repeated, *job) for job in jobs]
            for result in results:
                result.result(timeout=60)
        with (
            NpyReader(root / "metadata.npy") as meta,
            NpyReader(root / "predictions.npy", (14, 3)) as predictions,
        ):
            for i in range(args.samples):
                if meta.text(i, "market") != "US" or predictions.row(i)[-1] != 2.0:
                    raise AssertionError("Windowed reader changed synthetic values")
                if i % 4096 == 0:
                    assert meta._length <= 4 * 1024**2 + 65536
                    assert predictions._length <= 4 * 1024**2 + 65536
            for i in (args.samples - 1, 0, args.samples // 2, 0):
                assert meta.text(i, "symbol") == "FIXTURE"
        report = {
            "samples": args.samples,
            "array_bytes": sum(path.stat().st_size for path, _, _, _ in jobs),
            "writer_threads": workers,
            "mapping_window_bytes": 4 * 1024**2,
            "peak_rss_bytes": peak_bytes(),
            "elapsed_seconds": time.monotonic() - started,
            "model_or_training_executed": False,
            "python_ml_environment_used": False,
            "limitation": (
                "Synthetic artifact reader only; not a CUDA/DataLoader/model memory measurement."
            ),
        }
        if report["peak_rss_bytes"] > args.max_rss_mib * 1024**2:
            raise MemoryError("Synthetic diagnostic reader exceeded its RSS budget")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
