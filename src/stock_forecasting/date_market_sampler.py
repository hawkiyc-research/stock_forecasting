"""Runtime-only date/market index; the prepared bar-store remains immutable."""

from __future__ import annotations

import multiprocessing
import os
import random
import statistics
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.data.dataset import BlockwisePermutationSampler
from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

_SOURCE = None
INDEX_DTYPE = np.dtype([("group", "i8"), ("ordinal", "i8")])
# Index groups encode days * MARKET_CODE_RADIX + market code.
MARKET_CODE_RADIX = 256
DATE_MIXING_RNG_NAMESPACE = "date-market-mixing-v1"


def bounded_local_shuffle(items, capacity, rng):
    """Reorder a stream inside a fixed-capacity buffer without changing its multiset.

    Memory stays O(capacity). Random draws depend only on the stream length, so
    equal-length streams (ordinals or their groups) receive the same permutation.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 2:
        raise ValueError("Shuffle buffer capacity must be an integer of at least two")
    buffer = []
    for item in items:
        if len(buffer) < capacity:
            buffer.append(item)
            continue
        slot = rng.randrange(capacity)
        yield buffer[slot]
        buffer[slot] = item
    rng.shuffle(buffer)
    yield from buffer


def batch_composition_summary(groups, batch_size):
    """Count distinct date/market groups and dates in each complete consecutive batch."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    date_markets, dates = [], []
    for start in range(0, len(groups) - batch_size + 1, batch_size):
        chunk = groups[start : start + batch_size]
        date_markets.append(len(set(chunk)))
        dates.append(len({group // MARKET_CODE_RADIX for group in chunk}))
    if not date_markets:
        return {"batch_size": batch_size, "batches": 0}

    def describe(values):
        return {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }

    return {
        "batch_size": batch_size,
        "batches": len(date_markets),
        "distinct_date_markets": describe(date_markets),
        "distinct_dates": describe(dates),
    }


def _initialize(source):
    global _SOURCE
    import pyarrow
    import torch

    torch.set_num_threads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    source.symbol_cache_size = min(source.symbol_cache_size, 4)
    _SOURCE = source


def _range_rows(task):
    row, offset, size, market_code = task
    frame = _SOURCE._load_symbol(str(row["symbol"]))
    start = int(row["start_index"])
    days = (
        pd.DatetimeIndex(frame["timestamp"].iloc[start : start + size]).as_unit("ns").asi8
        // 86_400_000_000_000
    )
    result = np.empty(size, dtype=INDEX_DTYPE)
    result["group"] = days * 256 + market_code
    result["ordinal"] = np.arange(offset, offset + size)
    return offset, result


def date_market_index(source, *, requested_workers: int) -> Path:
    identity = {
        "version": 1,
        "manifest": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "split": source.split,
        "count": len(source),
        "window": source.window_size,
        "horizons": list(source.horizons),
    }
    root = source.root.parent / "training-cache" / "date-market"
    root.mkdir(parents=True, exist_ok=True)
    key = canonical_json_sha256(identity)
    path, marker = root / f"{key}.npy", root / f"{key}.json"
    if marker.is_file() and path.is_file():
        import json

        if json.loads(marker.read_text()) == identity:
            existing = np.load(path, mmap_mode="r")
            try:
                if existing.shape != (len(source),) or existing.dtype != INDEX_DTYPE:
                    raise ValueError("Runtime ranking index is truncated or incompatible")
            finally:
                existing._mmap.close()
            return path
    memory = detect_available_memory().available_bytes
    workers = min(
        requested_workers,
        max(1, detect_visible_cpu_count() - 2),
        max(1, int(memory * 0.2) // (512 * 1024**2)),
    )
    if workers < 2:
        print("Date/market index: memory/CPU budget permits only one worker", flush=True)
    import shutil

    if shutil.disk_usage(root).free < len(source) * INDEX_DTYPE.itemsize + 1024**3:
        raise OSError("Insufficient disk for the bounded runtime ranking index")
    temporary = path.with_suffix(f".pending-{os.getpid()}.npy")
    index = np.lib.format.open_memmap(temporary, mode="w+", dtype=INDEX_DTYPE, shape=(len(source),))
    market_names = sorted({str(v["market"]) for v in source._index.values()})
    if len(market_names) > 256:
        raise ValueError("Ranking index supports at most 256 market identities")

    def tasks():
        previous = 0
        for position, row in enumerate(source.ranges.to_dict("records")):
            end = int(source._range_ends[position])
            yield (
                row,
                previous,
                end - previous,
                market_names.index(str(source._index[str(row["symbol"])]["market"])),
            )
            previous = end

    pending = []
    iterator = iter(tasks())
    try:
        # Pool context termination cancels and joins workers even after a timeout.
        with multiprocessing.get_context("spawn").Pool(
            workers, initializer=_initialize, initargs=(source,)
        ) as pool:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * workers:
                    task = next(iterator, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.append(pool.apply_async(_range_rows, (task,)))
                if not pending:
                    break
                offset, rows = pending.pop(0).get(timeout=300)
                index[offset : offset + len(rows)] = rows
        # In-place native sort avoids a second full permutation in each worker.
        index.sort(order=["group", "ordinal"], kind="quicksort")
        index.flush()
    finally:
        index._mmap.close()
    temporary.replace(path)
    atomic_write_json(marker, identity)
    return path


class DateMarketSampler(BlockwisePermutationSampler):
    """Emit same-date/market blocks, optionally mixed by a bounded local shuffle.

    Without ``shuffle_buffer`` the order is the historical one. With it, each epoch
    emits exactly the same samples and annual quotas, but optimizer steps draw from
    many dates instead of one or two consecutive date/market blocks.
    """

    def __init__(
        self,
        source,
        *,
        requested_workers: int,
        yearly_decay: float | None = None,
        shuffle_buffer: int | None = None,
        **kwargs,
    ):
        self.index_path = date_market_index(source, requested_workers=requested_workers)
        if yearly_decay is not None and not 0 < yearly_decay <= 1:
            raise ValueError("yearly_decay must be in (0, 1]")
        if shuffle_buffer is not None and (
            isinstance(shuffle_buffer, bool)
            or not isinstance(shuffle_buffer, int)
            or shuffle_buffer < 2
        ):
            raise ValueError("shuffle_buffer must be None or an integer of at least two")
        self.yearly_decay = yearly_decay
        self.shuffle_buffer = shuffle_buffer
        super().__init__(len(source), **kwargs)
        self.sampling_summary = None
        if yearly_decay is not None:
            index = np.load(self.index_path, mmap_mode="r")
            try:
                self.sampling_summary = self.year_quotas(index)
            finally:
                index._mmap.close()
            import json

            print("Annual training sampling: " + json.dumps(self.sampling_summary), flush=True)

    def year_quotas(self, index):
        from datetime import date, timedelta

        epoch = date(1970, 1, 1)
        first = (epoch + timedelta(days=int(index["group"][0]) // 256)).year
        last = (epoch + timedelta(days=int(index["group"][-1]) // 256)).year
        strata = []
        for year in range(first, last + 1):
            lower = (date(year, 1, 1) - epoch).days * 256
            upper = (date(year + 1, 1, 1) - epoch).days * 256
            start, stop = np.searchsorted(index["group"], [lower, upper])
            if stop > start:
                weight = self.yearly_decay ** (last - year)
                strata.append(
                    {
                        "year": year,
                        "start": int(start),
                        "count": int(stop - start),
                        "per_window_weight": weight,
                    }
                )
        mass = np.array([s["count"] * s["per_window_weight"] for s in strata])
        exact = len(self) * mass / mass.sum()
        quotas = np.floor(exact).astype(np.int64)
        # Largest-remainder rounding gives an exact epoch budget without a full
        # per-window probability vector. Ties have deterministic year order.
        remainder = len(self) - int(quotas.sum())
        order = np.argsort(-(exact - quotas), kind="stable")
        quotas[order[:remainder]] += 1
        for row, quota in zip(strata, quotas, strict=True):
            row["samples"] = int(quota)
        return strata

    def _positions(self, index):
        """Yield sorted-index positions in the historical date/market block order."""
        if self.yearly_decay is not None:
            strata = self.year_quotas(index)
            rng = random.Random(self.seed + self.epoch * 1_000_003)
            remaining = [row["samples"] for row in strata]

            def stream(row):
                cycle = 0
                while True:
                    sampler = BlockwisePermutationSampler(
                        row["count"],
                        seed=self.seed + row["year"] * 19 + cycle * 99991,
                        block_size=self.block_size,
                    )
                    sampler.set_epoch(self.epoch)
                    for position in sampler:
                        yield row["start"] + position
                    cycle += 1

            streams = [stream(row) for row in strata]
            while sum(remaining):
                draw = rng.randrange(sum(remaining))
                chosen = 0
                while draw >= remaining[chosen]:
                    draw -= remaining[chosen]
                    chosen += 1
                take = min(self.block_size, remaining[chosen])
                for _ in range(take):
                    yield next(streams[chosen])
                remaining[chosen] -= take
            return
        yield from super().__iter__()

    def _ordered(self, values):
        if self.shuffle_buffer is None:
            return values
        # A dedicated seeded stream keeps the annual block draws unchanged.
        rng = random.Random(f"{DATE_MIXING_RNG_NAMESPACE}:{self.seed}:{self.epoch}")
        return bounded_local_shuffle(values, self.shuffle_buffer, rng)

    def __iter__(self):
        index = np.load(self.index_path, mmap_mode="r")
        try:
            ordinals = index["ordinal"]
            yield from self._ordered(int(ordinals[p]) for p in self._positions(index))
        finally:
            index._mmap.close()

    def batch_composition(self, micro_batch_size, accumulation_steps, max_optimizer_steps=256):
        """Describe date/market diversity of the current epoch's first optimizer steps.

        Groups follow the exact training order. Mixed runs also report the unmixed
        order of the same epoch, so one log shows the change in batch composition.
        """
        sizes = (micro_batch_size, accumulation_steps, max_optimizer_steps)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
            raise ValueError("Batch composition sizes must be positive integers")
        step_size = micro_batch_size * accumulation_steps
        limit = min(len(self), step_size * max_optimizer_steps)
        index = np.load(self.index_path, mmap_mode="r")
        try:
            groups = index["group"]

            def first(mixed):
                values = (int(groups[p]) for p in self._positions(index))
                return list(islice(self._ordered(values) if mixed else values, limit))

            configured = first(True)
            reference = None if self.shuffle_buffer is None else first(False)
        finally:
            index._mmap.close()

        def describe(values):
            return {
                "micro_batch": batch_composition_summary(values, micro_batch_size),
                "optimizer_step": batch_composition_summary(values, step_size),
            }

        result = {
            "epoch": self.epoch,
            "shuffle_buffer_samples": self.shuffle_buffer,
            "evaluated_samples": len(configured),
            "configured_order": describe(configured),
        }
        if reference is not None:
            result["unmixed_reference_order"] = describe(reference)
        return result
