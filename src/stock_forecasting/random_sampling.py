"""Exact sample-level random permutations using bounded disk-backed index storage.

This module intentionally has no ML dependencies. Fisher-Yates swaps have a
sequential dependency; data decoding remains in the existing bounded workers.
"""

from __future__ import annotations

import hashlib
import math
import mmap
import random
import shutil
from array import array
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

SAMPLING_VERSION = "sample-fisher-yates-v1"
AUDIT_VERSION = "optimizer-diversity-audit-v1"
INDEX_CHUNK_SIZE = 65536


def _seed(seed: int, *parts: object) -> int:
    payload = ":".join(map(str, (SAMPLING_VERSION, seed, *parts)))
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=16).digest(), "big")


def annual_quotas(strata: list[dict], target_count: int, decay: float) -> list[dict]:
    """Preserve exact largest-remainder annual budgets independently of ordering."""
    if not strata or target_count < 1 or not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("Annual sampling requires nonempty strata and a positive budget")
    latest = max(row["year"] for row in strata)
    result = [
        {**row, "per_window_weight": decay ** (latest - row["year"])} for row in strata
    ]
    mass = sum(row["count"] * row["per_window_weight"] for row in result)
    if mass <= 0:
        raise ValueError("Annual sampling mass must be positive")
    exact = [target_count * row["count"] * row["per_window_weight"] / mass for row in result]
    quotas = [math.floor(value) for value in exact]
    order = sorted(range(len(result)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[: target_count - sum(quotas)]:
        quotas[i] += 1
    return [{**row, "samples": quota} for row, quota in zip(result, quotas, strict=True)]


@contextmanager
def _mapped_indices(path: Path, count: int):
    if count < 1:
        raise ValueError("A mapped index must contain at least one integer")
    with path.open("w+b") as stream:
        stream.truncate(count * 8)
        with mmap.mmap(stream.fileno(), 0) as mapping:
            values = memoryview(mapping).cast("Q")
            try:
                yield mapping, values
            finally:
                values.release()


def _initialize_range(mapping: mmap.mmap, start: int, count: int) -> None:
    # Only this small chunk is materialized; no Python list covers the dataset.
    for offset in range(0, count, INDEX_CHUNK_SIZE):
        stop = min(offset + INDEX_CHUNK_SIZE, count)
        mapping[offset * 8 : stop * 8] = array("Q", range(start + offset, start + stop)).tobytes()


def shuffled_positions(
    strata: list[dict],
    *,
    seed: int,
    epoch: int,
    scratch_root: Path,
    max_scratch_bytes: int,
    stable_subset: bool = False,
):
    """Yield source positions in a full random order, never contiguous data blocks.

    Selection is without replacement inside each annual cycle. Oversampling
    selects complete cycles plus a partial cycle. Global presentation order
    may place repeated indices together; it does not preserve cycle boundaries.
    Conditional on the selected multiset, the final Fisher-Yates permutation
    uniformly mixes all years and date/market groups. Stable subsets keep Stage
    1 membership fixed across epochs while changing its presentation order.
    """
    if epoch < 0 or not strata:
        raise ValueError("Random sampling requires strata and a nonnegative epoch")
    previous_end = 0
    for row in strata:
        if any(type(row.get(k)) is not int for k in ("start", "count", "samples")):
            raise ValueError("Stratum offsets, counts and quotas must be integers")
        if row["start"] != previous_end or row["count"] < 1 or row["samples"] < 0:
            raise ValueError("Strata must partition the source and have nonnegative quotas")
        previous_end += row["count"]
    total = sum(row["samples"] for row in strata)
    if total < 1:
        raise ValueError("Random sampling requires a positive sample budget")
    largest = max(row["count"] for row in strata if row["samples"])
    required = 8 * (total + largest)
    if required > max_scratch_bytes:
        raise ValueError("Random sampler exceeds its explicit scratch storage budget")
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch_root).free < required + 64 * 1024**2:
        raise OSError("Insufficient disk for sample-level random sampling")
    with TemporaryDirectory(prefix="random-epoch-", dir=scratch_root) as temporary:
        directory = Path(temporary)
        with _mapped_indices(directory / "order.bin", total) as (_, order):
            output = 0
            for row in strata:
                remaining, cycle = row["samples"], 0
                if not remaining:
                    continue
                with _mapped_indices(directory / "cycle.bin", row["count"]) as (mapping, work):
                    while remaining:
                        _initialize_range(mapping, row["start"], row["count"])
                        selection_epoch = 0 if stable_subset else epoch
                        rng = random.Random(
                            _seed(seed, "selection", selection_epoch, row["start"], cycle)
                        )
                        take = min(remaining, row["count"])
                        for i in range(take):
                            j = rng.randrange(i, row["count"])
                            work[i], work[j] = work[j], work[i]
                            order[output] = work[i]
                            output += 1
                        remaining -= take
                        cycle += 1
            random.Random(_seed(seed, "order", epoch)).shuffle(order)
            for position in order:
                yield int(position)


class SamplingDiversityAudit:
    """Bounded diagnostics for samples actually consumed by optimizer updates."""

    def __init__(self, max_updates: int = 128):
        if max_updates < 1:
            raise ValueError("Sampling audit must cover at least one update")
        self.max_updates = max_updates
        self.records: list[dict] = []
        self.pending: Counter = Counter()
        self.market_counts: Counter = Counter()
        self.sample_count = 0

    def observe(self, dates, markets) -> None:
        if len(self.records) >= self.max_updates:
            return
        if len(dates) != len(markets):
            raise ValueError("Sampling audit dates and markets must align")
        for date, market in zip(dates, markets, strict=True):
            day = str(date).split("T", 1)[0].split(" ", 1)[0]
            self.pending[(day, str(market))] += 1

    def optimizer_step(self, step: int) -> None:
        if len(self.records) >= self.max_updates or not self.pending:
            return
        counts = self.pending
        size = sum(counts.values())
        markets = Counter()
        for (_, market), count in counts.items():
            markets[market] += count
        self.records.append(
            {
                "step": int(step),
                "samples": size,
                "unique_dates": len({date for date, _ in counts}),
                "unique_date_markets": len(counts),
                "unique_markets": len(markets),
                "largest_date_market_share": max(counts.values()) / size,
                "effective_date_market_count": size * size / sum(v * v for v in counts.values()),
            }
        )
        self.market_counts.update(markets)
        self.sample_count += size
        self.pending.clear()

    def as_dict(self) -> dict:
        return {
            "version": AUDIT_VERSION,
            "step_coordinate": "runtime_optimizer_step",
            "scope": "consumed_microbatches_grouped_by_actual_optimizer_update",
            "max_updates": self.max_updates,
            "sample_count": self.sample_count,
            "market_counts": dict(self.market_counts),
            "updates": list(self.records),
        }
