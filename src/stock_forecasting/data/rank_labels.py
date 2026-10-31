"""Cross-sectional gauss-rank labels stored as Parquet beside each split's bar store.

Each split keeps its own store under ``<prepared>/rank-labels/<identity>/``. Shards reuse the
bar store's buckets and hold one row group per symbol, in the same canonical cutoff order as
the split's sample ranges, so a dataset reads a label by (symbol, ordinal) and checks the
cutoff timestamp. Ranks are computed within one cutoff date and market of one split.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

RANK_LABEL_SCHEMA = "cross-sectional-gauss-rank-v1"
IMPLEMENTATION_FILES = ("data/rank_labels.py", "data/dataset.py")
DEFAULT_WORKERS_ENV = "FIN_TS_RANK_LABEL_WORKERS"
WORKER_TIMEOUT_SECONDS = 1800


def implementation_sha256() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode() + b"\0" + (package / relative).read_bytes())
    return digest.hexdigest()


def label_identity(source, min_group_size: int) -> dict:
    if type(min_group_size) is not int or min_group_size < 2:
        raise ValueError("rank_min_group_size must be an integer of at least two")
    return {
        "schema": RANK_LABEL_SCHEMA,
        "source_manifest_sha256": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "split": source.split,
        "samples": len(source),
        "horizons": list(source.horizons),
        "target": "benchmark_relative_adjusted_log_return_next_open_to_horizon_close",
        "groups": ["split", "cutoff", "market"],
        "transform": "inverse_normal((average_rank - 0.5) / group_size)",
        "min_group_size": min_group_size,
        "implementation_sha256": implementation_sha256(),
    }


def label_root(source, min_group_size: int) -> tuple[Path, dict]:
    if source.sample_universe_identity is None:
        raise ValueError("Rank labels require a cleaned sample universe")
    identity = label_identity(source, min_group_size)
    return source.root.parent / "rank-labels" / canonical_json_sha256(identity), identity


def _ready(root: Path, identity: dict) -> bool:
    marker = root / "rank-labels.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    if payload.get("identity") != identity or payload.get("state") != "ready":
        raise ValueError("Rank-label store has an inconsistent identity")
    if sha256_file(root / "label-index.parquet") != payload["label_index_sha256"]:
        raise ValueError("Rank-label index is incomplete or corrupted")
    return True


def attach_rank_labels(source, min_group_size: int) -> Path:
    """Attach an existing store; building is an explicit step before workers start."""
    root, identity = label_root(source, min_group_size)
    if not _ready(root, identity):
        raise ValueError(
            f"Rank labels for split={source.split} are missing; build them with "
            "ensure_rank_labels before opening rank-task datasets"
        )
    source.attach_rank_labels(root)
    return root


def ensure_split_rank_labels(config, validated_roots: dict, *, workers: int | None = None) -> dict:
    """Build every split's store before any rank-task dataset or worker pool is opened."""
    from stock_forecasting.data.sample_universe import open_clean_dataset

    result = {}
    for split in ("train", "validation", "test"):
        source = open_clean_dataset(config, split, validated_root=validated_roots[split])
        try:
            root = ensure_rank_labels(source, config.model.rank_min_group_size, workers=workers)
            result[split] = {
                "root": str(root),
                "manifest_sha256": sha256_file(root / "rank-labels.json"),
            }
        finally:
            source.close()
    return result


def group_ids(cutoff_ns: np.ndarray, market_codes: np.ndarray) -> np.ndarray:
    """Dense ids for (cutoff, market); one split is ranked at a time."""
    order = np.lexsort((market_codes, cutoff_ns))
    change = np.ones(len(order), dtype=bool)
    change[1:] = (np.diff(cutoff_ns[order]) != 0) | (np.diff(market_codes[order]) != 0)
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.cumsum(change) - 1
    return result


def gauss_rank(values, groups, min_group_size: int) -> np.ndarray:
    """Inverse-normal average ranks within each group; small groups stay unlabelled."""
    # scipy is installed with the scikit-learn dependency.
    from scipy.special import ndtri

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if values.shape != groups.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank labels need one finite value per grouped sample")
    order = np.lexsort((values, groups))
    sorted_values, sorted_groups = values[order], groups[order]
    group_start = np.ones(len(order), dtype=bool)
    group_start[1:] = sorted_groups[1:] != sorted_groups[:-1]
    starts = np.flatnonzero(group_start)
    sizes = np.diff(np.append(starts, len(order)))
    member = np.repeat(np.arange(len(starts)), sizes)
    position = np.arange(len(order)) - starts[member]
    # Ties share the average of their first and last 1-based positions.
    run_start = group_start.copy()
    run_start[1:] |= sorted_values[1:] != sorted_values[:-1]
    first = np.flatnonzero(run_start)
    last = np.append(first[1:], len(order)) - 1
    average = (position[first] + position[last]) / 2.0 + 1.0
    size = sizes[member].astype(np.float64)
    labels = ndtri((average[np.cumsum(run_start) - 1] - 0.5) / size)
    labels[size < min_group_size] = np.nan
    result = np.empty(len(order), dtype=np.float32)
    result[order] = labels
    return result


def _worker_count(requested: int | None, tasks: int, worker_bytes: int) -> tuple[int, dict]:
    value = requested or int(os.environ.get(DEFAULT_WORKERS_ENV, "8"))
    if value < 1:
        raise ValueError(f"{DEFAULT_WORKERS_ENV} must be positive")
    memory = detect_available_memory().available_bytes
    if memory * 0.4 < worker_bytes:
        raise MemoryError("Insufficient headroom for one rank-label worker")
    count = min(
        value,
        max(1, detect_visible_cpu_count() - 1),
        max(1, int(memory * 0.4) // worker_bytes),
        max(1, tasks),
    )
    plan = {"requested": value, "memory_budget": int(memory * 0.4), "per_worker": worker_bytes}
    return count, plan


def _bounded_map(
    function, tasks, count, initializer=None, initargs=(), timeout=WORKER_TIMEOUT_SECONDS
):
    """Yield results in submission order with at most 2 * count pending tasks.

    ``timeout`` bounds one task's wall time once it is the oldest pending task; callers
    whose tasks legitimately run long (fold training) pass their own budget.
    """
    if count == 1:
        if initializer is not None:
            initializer(*initargs)
        for task in tasks:
            yield function(task)
        return
    with get_context("spawn").Pool(
        processes=count, initializer=initializer, initargs=initargs
    ) as pool:
        iterator, pending, exhausted = iter(tasks), [], False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * count:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    pending.append(pool.apply_async(function, (task,)))
            if pending:
                # Pool termination cancels and joins all children on timeout or error.
                yield pending.pop(0).get(timeout=timeout)


_SOURCE = None


def _open_source(state: dict) -> None:
    global _SOURCE
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset

    _SOURCE = LazyFinancialWindowDataset.__new__(LazyFinancialWindowDataset)
    _SOURCE.__setstate__(state)


def _bucket_returns(task):
    bucket, spans = task
    rows = []
    for _symbol, begin, end in spans:
        batch = _SOURCE.target_batch(np.arange(begin, end))
        rows.append((begin, end, batch["targets"], batch["cutoff_ns"]))
    _SOURCE.close()
    return bucket, rows


def _write_bucket(task):
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, spans, work, output, horizons = task
    ranks = np.load(Path(work) / "ranks.npy", mmap_mode="r")
    cutoff = np.load(Path(work) / "cutoff_ns.npy", mmap_mode="r")
    relative = f"shards/bucket-{bucket:04d}/shard.parquet"
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    schema = pa.schema(
        [("timestamp", pa.timestamp("ns", tz="UTC"))]
        + [(f"rank_h{h:02d}", pa.float32()) for h in horizons]
    )
    index = []
    with pq.ParquetWriter(pending, schema) as writer:
        for group, (symbol, begin, end) in enumerate(spans):
            stamps = pa.array(np.asarray(cutoff[begin:end]), type=pa.int64())
            columns = [stamps.cast(schema[0].type)]
            columns += [pa.array(np.asarray(ranks[begin:end, i])) for i in range(len(horizons))]
            table = pa.Table.from_arrays(columns, schema=schema)
            writer.write_table(table, row_group_size=max(1, end - begin))
            index.append(
                {
                    "symbol": symbol,
                    "shard_relative_path": relative,
                    "row_group": group,
                    "row_count": end - begin,
                    "ordinal_begin": begin,
                }
            )
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    pending.replace(path)
    return relative, {"bytes": path.stat().st_size, "sha256": sha256_file(path)}, index


def _group_statistics(groups, market_codes, markets, min_group_size) -> dict:
    sizes = np.bincount(groups)
    group_market = np.empty(len(sizes), dtype=np.int64)
    group_market[groups] = market_codes
    result = {}
    for code, market in enumerate(markets):
        selected = sizes[group_market == code]
        if not len(selected):
            continue
        masked = selected < min_group_size
        result[market] = {
            "groups": len(selected),
            "unlabelled_groups": int(masked.sum()),
            "unlabelled_samples": int(selected[masked].sum()),
            "samples": int(selected.sum()),
            "group_size_min": int(selected.min()),
            "group_size_median": float(np.median(selected)),
            "group_size_max": int(selected.max()),
        }
    return result


def ensure_rank_labels(source, min_group_size: int, *, workers: int | None = None) -> Path:
    """Build one split's store with bounded process/thread pools; reuse it when ready."""
    from stock_forecasting.data.sample_universe import _exclusive_build

    root, identity = label_root(source, min_group_size)
    if _ready(root, identity):
        return root
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_build(root):
        if _ready(root, identity):
            return root
        started = time.monotonic()
        count, horizons = len(source), list(source.horizons)
        spans = source.symbol_spans()
        markets = sorted({str(source._index[symbol]["market"]) for symbol in spans})
        buckets: dict[int, list] = {}
        for symbol, (begin, end) in sorted(spans.items(), key=lambda item: item[1][0]):
            buckets.setdefault(int(source._index[symbol]["bucket"]), []).append(
                (symbol, begin, end)
            )
        work = root / "work"
        work.mkdir(exist_ok=True)
        targets = np.lib.format.open_memmap(
            work / "targets.npy", mode="w+", dtype="float64", shape=(count, len(horizons))
        )
        cutoff = np.lib.format.open_memmap(
            work / "cutoff_ns.npy", mode="w+", dtype="int64", shape=(count,)
        )
        market_codes = np.empty(count, dtype=np.int64)
        for symbol, (begin, end) in spans.items():
            market_codes[begin:end] = markets.index(str(source._index[symbol]["market"]))
        largest = max(sum(end - begin for _, begin, end in rows) for rows in buckets.values())
        return_workers, return_plan = _worker_count(
            workers, len(buckets), 768 * 1024**2 + largest * (len(horizons) + 1) * 8 * 3
        )
        print(
            f"Rank labels {source.split}: returns for {count} samples in {len(buckets)} "
            f"buckets, workers={return_workers}",
            flush=True,
        )
        covered = 0
        for _bucket, rows in _bounded_map(
            _bucket_returns,
            sorted(buckets.items()),
            return_workers,
            _open_source,
            (source.__getstate__(),),
        ):
            for begin, end, values, stamps in rows:
                targets[begin:end], cutoff[begin:end] = values, stamps
                covered += end - begin
        if covered != count:
            raise RuntimeError("Rank-label returns did not cover every sample exactly once")
        targets.flush()
        cutoff.flush()
        groups = group_ids(np.asarray(cutoff), market_codes)
        ranks = np.lib.format.open_memmap(
            work / "ranks.npy", mode="w+", dtype="float32", shape=(count, len(horizons))
        )
        # numpy releases the GIL while sorting; one column per thread bounds peak memory.
        rank_threads, rank_plan = _worker_count(workers, len(horizons), 256 * 1024**2 + count * 72)

        # The arrays are bound as defaults because the function outlives the `del` below.
        def rank_column(index, targets=targets, ranks=ranks, groups=groups):
            column = np.ascontiguousarray(targets[:, index])
            ranks[:, index] = gauss_rank(column, groups, min_group_size)
            return index

        with ThreadPoolExecutor(max_workers=rank_threads) as pool:
            list(pool.map(rank_column, range(len(horizons))))
        ranks.flush()
        labelled = np.isfinite(ranks[:, 0])
        sizes = np.bincount(groups)[groups]
        if not np.array_equal(labelled, sizes >= min_group_size) or not all(
            np.array_equal(np.isfinite(ranks[:, i]), labelled) for i in range(len(horizons))
        ):
            raise ValueError("Rank labels are missing outside the small-group mask")
        statistics = _group_statistics(groups, market_codes, markets, min_group_size)
        del targets, cutoff, groups, sizes, labelled
        write_workers, write_plan = _worker_count(workers, len(buckets), 512 * 1024**2)
        files, index = {}, []
        for relative, metadata, rows in _bounded_map(
            _write_bucket,
            [
                (bucket, rows, str(work), str(root), horizons)
                for bucket, rows in sorted(buckets.items())
            ],
            write_workers,
        ):
            files[relative] = metadata
            index.extend(rows)
        del ranks
        frame = pd.DataFrame(index).sort_values("ordinal_begin", kind="stable")
        pending = root / "label-index.pending.parquet"
        frame.to_parquet(pending, index=False)
        pending.replace(root / "label-index.parquet")
        atomic_write_json(
            root / "rank-labels.json",
            {
                "state": "ready",
                "identity": identity,
                "label_index_sha256": sha256_file(root / "label-index.parquet"),
                "files": files,
                "groups_by_market": statistics,
                "execution": {
                    "return_workers": return_workers,
                    "return_plan": return_plan,
                    "rank_threads": rank_threads,
                    "rank_plan": rank_plan,
                    "write_workers": write_workers,
                    "write_plan": write_plan,
                    "seconds": time.monotonic() - started,
                    "windows_materialized": False,
                },
            },
        )
        # Scratch arrays are rebuilt from the bar store; only the Parquet store is kept.
        for name in ("targets.npy", "cutoff_ns.npy", "ranks.npy"):
            (work / name).unlink(missing_ok=True)
        print(f"Rank labels {source.split}: ready at {root} ({json.dumps(statistics)})", flush=True)
    return root
