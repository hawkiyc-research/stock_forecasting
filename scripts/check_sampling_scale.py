#!/usr/bin/env python3
"""Check synthetic index coverage, diversity and memory without importing ML."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _available_memory(resources):
    try:
        return resources.detect_available_memory()
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        result = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
        page_size = int(re.search(r"page size of (\d+) bytes", result).group(1))
        pages = sum(
            int(re.search(r"^" + name + r":\s+(\d+)", result, re.M).group(1))
            for name in ("Pages free", "Pages inactive", "Pages speculative")
        )
        return resources.AvailableMemoryEstimate(
            pages * page_size, "macos_vm_stat_free_inactive_speculative", ()
        )


def _check(task):
    seed, size, scratch_bytes, scratch_parent = task
    sampling = _load("sampling_scale", ROOT / "src/stock_forecasting/random_sampling.py")
    seen = bytearray(size)
    started = time.perf_counter()
    first = None
    groups, current = [], set()
    with tempfile.TemporaryDirectory(prefix="sampling-scale-", dir=scratch_parent) as directory:
        iterator = sampling.shuffled_positions(
            [{"start": 0, "count": size, "samples": size}], seed=seed, epoch=0,
            scratch_root=Path(directory), max_scratch_bytes=scratch_bytes,
        )
        for count, index in enumerate(iterator, 1):
            if first is None:
                first = time.perf_counter() - started
            if seen[index]:
                raise AssertionError("Repeated index in an unweighted full epoch")
            seen[index] = 1
            if count <= 128 * 256:
                current.add(index // 128)
                if count % 256 == 0:
                    groups.append(len(current))
                    current.clear()
        if count != size or sum(seen) != size:
            raise AssertionError("The permutation omitted source indices")
        if list(Path(directory).iterdir()):
            raise AssertionError("Sampler scratch files survived generator completion")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return {
        "seed": seed, "samples": count, "unique_samples": sum(seen),
        "seconds": time.perf_counter() - started, "first_index_seconds": first,
        "process_peak_rss_mib": peak_bytes / 1024**2,
        "scratch_bound_bytes": 16 * size,
        "synthetic_128_sample_group_audit": {
            "updates": len(groups), "effective_batch_size": 256,
            "minimum_unique_groups": min(groups), "mean_unique_groups": sum(groups) / len(groups),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4_710_117)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 347])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scratch-max-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 256 or not 1 <= args.workers <= 16 or not 1 <= len(args.seeds) <= 16:
        parser.error("Use at least 256 samples and between 1 and 16 workers/seeds")
    scratch_bytes = args.scratch_max_mib * 1024**2
    if 16 * args.samples > scratch_bytes:
        parser.error("The configured scratch budget cannot hold the permutation")
    resources = _load("sampling_resources", ROOT / "src/stock_forecasting/runtime_resources.py")
    memory = _available_memory(resources)
    per_worker = int(1.25 * (17 * args.samples + 64 * 1024**2))
    memory_workers = int(memory.available_bytes * 0.2) // per_worker
    if memory_workers < 1:
        parser.error("Insufficient available memory for one safely bounded index check")
    workers = min(
        args.workers, len(args.seeds), max(1, resources.detect_visible_cpu_count() - 2),
        memory_workers,
    )
    if workers == 1 and len(args.seeds) > 1:
        print("Only one worker fits the CPU/memory/requested concurrency budget", file=sys.stderr)
    payload = {
        "kind": "stdlib_index_scale_check", "data": "synthetic_ordinals_only_not_market_bars",
        "workers": workers, "available_memory": memory.as_dict(), "runs": [],
    }
    # At most 16 tiny task descriptors are submitted; no dataset is copied to workers.
    with tempfile.TemporaryDirectory(prefix="sampling-check-") as scratch_parent:
        if shutil.disk_usage(scratch_parent).free < workers * 16 * args.samples + 64 * 1024**2:
            parser.error("Insufficient disk for concurrent bounded index checks")
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            payload["runs"] = pool.map_async(
                _check,
                [(seed, args.samples, scratch_bytes, scratch_parent) for seed in args.seeds],
                chunksize=1,
            ).get(timeout=600)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
