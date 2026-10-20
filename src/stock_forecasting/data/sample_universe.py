"""Compact, resumable cleaning indexes over immutable prepared OHLCV shards.

Only ranges and audit counts are written. Bars, windows and targets are never
copied, and the CPU preparation contract is neither changed nor rewritten.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data_policy import evaluation_dataset_id, load_data_policy
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

ALGORITHM = "complete-market-sessions-positive-volume-past-liquidity-v1"
RANGE_COLUMNS = (
    "symbol",
    "split",
    "start_index",
    "stop_index",
    "count",
    "cutoff_start_at",
    "cutoff_end_at",
    "label_end_max_at",
)
_WORKER = None


def market_sessions(name: str, start: str, end: str) -> pd.DatetimeIndex:
    """Return exchange-local date labels, not UTC-converted market-close instants."""
    import exchange_calendars as calendars

    calendar = calendars.get_calendar(name, start=start, end=end)
    # Calendar bounds may fall on a weekend/holiday. Its strict range helper
    # rejects dates outside its first/last session even inside requested bounds.
    sessions = calendar.sessions
    overrides = load_data_policy()["calendar_overrides"].get(name, {"open": [], "closed": []})
    sessions = sessions.union(pd.DatetimeIndex(overrides["open"]))
    sessions = sessions.difference(pd.DatetimeIndex(overrides["closed"]))
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    return pd.DatetimeIndex(sessions).tz_localize("UTC").as_unit("ns")


def _dates(frame):
    return pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).as_unit("ns")


def _row_validity(frame, *, require_volume: bool):
    prices = frame[["open", "high", "low", "close", "adjusted_close"]].to_numpy(float)
    volume = frame["volume"].to_numpy(float)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= np.isfinite(volume) & (volume > 0 if require_volume else volume >= 0)
    valid &= prices[:, 1] + 1e-8 >= np.max(prices[:, [0, 2, 3]], axis=1)
    valid &= prices[:, 2] - 1e-8 <= np.min(prices[:, [0, 1, 3]], axis=1)
    return valid


def eligible_cutoffs(
    frame,
    benchmark,
    sessions,
    *,
    window_size,
    max_horizon,
    policy,
    currency,
    benchmark_is_index=False,
):
    """Validate every market session; liquidity uses only observations through cutoff."""
    count = len(frame)
    candidates = np.arange(window_size - 1, max(window_size - 1, count - max_horizon))
    starts, ends = candidates - window_size + 1, candidates + max_horizon
    reasons = np.full(len(candidates), "", dtype="U48")
    dates = _dates(frame)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Cleaning requires unique, strictly increasing asset dates")
    positions = sessions.get_indexer(dates)

    def any_bad(bad, left=starts, right=ends):
        prefix = np.r_[0, np.cumsum(bad, dtype=np.int64)]
        return prefix[right + 1] != prefix[left]

    def reject(mask, reason):
        reasons[(reasons == "") & mask] = reason

    reject(any_bad(positions < 0), "off_calendar_asset_bar")
    # A difference of two means a missing market session, not a one-day return.
    gaps = np.r_[False, np.diff(positions) != 1]
    reject(any_bad(gaps, starts + 1, ends), "asset_market_session_gap")
    reject(any_bad(~_row_validity(frame, require_volume=True)), "invalid_or_zero_volume_asset")
    aligned = _dates(benchmark).get_indexer(dates)
    reject(any_bad(aligned < 0), "benchmark_market_session_gap")
    valid_benchmark = _row_validity(benchmark, require_volume=not benchmark_is_index)
    benchmark_bad = (aligned < 0) | ~valid_benchmark[np.maximum(aligned, 0)]
    reject(any_bad(benchmark_bad), "invalid_benchmark_bar")

    lookback = min(policy["liquidity_lookback_sessions"], window_size)
    turnover = frame["close"].astype(float) * frame["volume"].astype(float)
    median = turnover.rolling(lookback, min_periods=lookback).median().to_numpy()
    thresholds = policy["minimum_median_daily_turnover"]
    if currency not in thresholds:
        raise ValueError(f"No minimum-turnover policy for currency={currency}")
    reject(
        ~np.isfinite(median[candidates]) | (median[candidates] < thresholds[currency]),
        "insufficient_trailing_liquidity",
    )
    # Large, finite economic returns are diagnostics, NEVER a rejection rule.
    adjusted = frame["adjusted_close"].to_numpy(float)
    transitions = np.r_[False, np.abs(np.diff(np.log(np.maximum(adjusted, 1e-300)))) > 0.5]
    retained_extreme = any_bad(transitions, starts + 1, ends) & (reasons == "")
    return candidates, reasons, retained_extreme


def _initialize(root, index_path, sessions_path, policy, window_size, split_spec):
    global _WORKER
    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    from threadpoolctl import threadpool_limits

    native_limits = threadpool_limits(limits=1)
    index = pd.read_parquet(index_path)
    _WORKER = {
        "root": Path(root),
        "index": {r["symbol"]: r for r in index.to_dict("records")},
        "sessions": {
            name: pd.DatetimeIndex(values)
            for name, values in json.loads(Path(sessions_path).read_text()).items()
        },
        "policy": policy,
        "window_size": window_size,
        "split_spec": split_spec,
        "benchmarks": OrderedDict(),
        "shards": OrderedDict(),
        "native_limits": native_limits,
    }


def _read(row):
    import pyarrow.parquet as parquet

    cache = _WORKER["shards"]
    key = row["shard_relative_path"]
    source = cache.pop(key, None)
    if source is None:
        source = parquet.ParquetFile(_WORKER["root"] / key)
    cache[key] = source
    # Keep parquet metadata/handles bounded instead of reopening a shard for
    # every symbol. Row groups themselves are not retained here.
    while len(cache) > 4 or (
        len(cache) > 1 and sum(v.metadata.serialized_size for v in cache.values()) > 64 * 1024**2
    ):
        cache.popitem(last=False)[1].close()
    return (
        source.read_row_group(int(row["row_group"]))
        .to_pandas()
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _clean_bucket(task):
    bucket, output = task
    rows, audits = [], []
    policy = _WORKER["policy"]
    symbols = sorted(
        r["symbol"]
        for r in _WORKER["index"].values()
        if int(r["bucket"]) == bucket and bool(r["eligible"])
    )
    for symbol in symbols:
        row = _WORKER["index"][symbol]
        benchmark_symbol = str(row["benchmark_symbol"])
        benchmark_row = _WORKER["index"].get(benchmark_symbol)
        if benchmark_row is None:
            raise ValueError(f"Missing benchmark for eligible target {symbol}")
        cache = _WORKER["benchmarks"]
        benchmark = cache.pop(benchmark_symbol, None)
        if benchmark is None:
            benchmark = _read(benchmark_row)
        cache[benchmark_symbol] = benchmark
        while len(cache) > 4:
            cache.popitem(last=False)
        frame = _read(row)
        market = str(row["market"])
        candidates, reasons, extremes = eligible_cutoffs(
            frame,
            benchmark,
            _WORKER["sessions"][market],
            window_size=_WORKER["window_size"],
            max_horizon=14,
            policy=policy,
            currency=str(row["currency"]),
            benchmark_is_index=benchmark_row["asset_type"] == "index",
        )
        dates = _dates(frame)
        for split, lower, upper in _WORKER["split_spec"]:
            region = (dates[candidates] >= pd.Timestamp(lower)) & (
                dates[candidates] < pd.Timestamp(upper)
            )
            crossed = region & (dates[candidates + 14] >= pd.Timestamp(upper))
            selected = region & ~crossed & (reasons == "")
            audit = Counter(str(v) for v in reasons[region & (reasons != "")])
            audit["label_crosses_split_boundary"] = int((crossed & (reasons == "")).sum())
            audits.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "split": split,
                    "candidates": int(region.sum()),
                    "accepted": int(selected.sum()),
                    "retained_extreme_windows": int((extremes & selected).sum()),
                    "excluded": dict(audit),
                }
            )
            # Include false positions in the mask so gaps never merge into a range.
            boundaries = np.diff(np.r_[False, selected, False].astype(np.int8))
            for first, last in zip(
                np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True
            ):
                start, stop = int(candidates[first]), int(candidates[last - 1]) + 1
                rows.append(
                    dict(
                        zip(
                            RANGE_COLUMNS,
                            (
                                symbol,
                                split,
                                start,
                                stop,
                                stop - start,
                                dates[start].isoformat(),
                                dates[stop - 1].isoformat(),
                                dates[stop + 13].isoformat(),
                            ),
                            strict=True,
                        )
                    )
                )
    path = Path(output)
    temporary = path.with_suffix(".pending.parquet")
    pd.DataFrame(rows, columns=RANGE_COLUMNS).to_parquet(temporary, index=False)
    temporary.replace(path)
    atomic_write_json(path.with_suffix(".json"), {"audits": audits})
    return bucket, len(rows), sum(a["accepted"] for a in audits)


@contextmanager
def _exclusive_build(root):
    # RunPod is Linux. flock releases automatically on process death, so a timed
    # out Pod cannot leave an unbreakable cache lock. No training lease is touched.
    import fcntl

    with (root / "build.lock").open("a") as stream:
        deadline = time.monotonic() + 1800
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Another process is still building the cleaned sample index"
                    ) from None
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _split_spec(manifest):
    fixed = manifest["identity"].get("fixed_split")
    if fixed:
        return [
            ("train", "1900-01-01T00:00:00Z", fixed["train_end"] + "T00:00:00Z"),
            (
                "validation",
                fixed["train_end"] + "T00:00:00Z",
                fixed["validation_end"] + "T00:00:00Z",
            ),
            ("test", fixed["validation_end"] + "T00:00:00Z", fixed["test_end"] + "T00:00:00Z"),
        ]
    audit = manifest["split_audit"]
    return [
        ("train", "1900-01-01T00:00:00Z", audit["train_boundary_exclusive"]),
        ("validation", audit["validation_start"], audit["validation_boundary_exclusive"]),
        ("test", audit["test_start"], "2200-01-01T00:00:00Z"),
    ]


def ensure_sample_universe(
    root: Path, *, workers: int | None = None, cache_root: Path | None = None
):
    """Build all split ranges with bounded workers, resuming completed bucket scans."""
    policy = load_data_policy()
    if version("exchange_calendars") != policy["calendar_version"]:
        raise ValueError("The offline calendar version differs from the data-cleaning policy")
    manifest = json.loads((root / "bar-store.json").read_text())
    identity = {
        "algorithm": ALGORITHM,
        "source_manifest_sha256": sha256_file(root / "bar-store.json"),
        "policy": policy,
        "window_size": manifest["identity"]["window_size"],
        "splits": [list(row) for row in _split_spec(manifest)],
    }
    key = canonical_json_sha256(identity)
    parent = cache_root or root.parent / "sample-universes"
    output = parent / key
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "universe.json"

    def ready():
        if not marker.is_file():
            return False
        payload = json.loads(marker.read_text())
        if payload["identity"] != identity or payload["state"] != "ready":
            raise ValueError("Sample-universe cache has an inconsistent policy or source")
        if sha256_file(output / "cutoff-ranges.parquet") != payload["ranges_sha256"]:
            raise ValueError("Sample-universe range index is incomplete or corrupted")
        return True

    if ready():
        return output
    with _exclusive_build(output):
        if ready():
            return output
        index = pd.read_parquet(root / "symbol-index.parquet")
        sessions = {}
        for market in sorted(set(index.loc[index["eligible"], "market"])):
            if market not in policy["calendars"]:
                raise ValueError(f"No exchange calendar is configured for {market}")
            market_rows = index.loc[index["market"] == market]
            start = pd.Timestamp(market_rows["start_at"].min()).date().isoformat()
            end = pd.Timestamp(market_rows["end_at"].max()).date().isoformat()
            sessions[market] = [
                v.isoformat()
                for v in market_sessions(
                    policy["calendars"][market],
                    start,
                    end,
                )
            ]
        atomic_write_json(output / "sessions.json", sessions)
        buckets = sorted(set(int(v) for v in index.loc[index["eligible"], "bucket"]))
        parts = output / "parts"
        parts.mkdir(exist_ok=True)
        pending_tasks = [
            (bucket, str(parts / f"{bucket:04d}.parquet"))
            for bucket in buckets
            if not (parts / f"{bucket:04d}.json").is_file()
            or not (parts / f"{bucket:04d}.parquet").is_file()
        ]
        requested = workers or int(os.environ.get("FIN_TS_CLEANING_WORKERS", "8"))
        if requested < 1:
            raise ValueError("FIN_TS_CLEANING_WORKERS must be positive")
        memory = detect_available_memory().available_bytes
        worker_bytes = 512 * 1024**2 + int(index.memory_usage(deep=True).sum()) * 4
        if pending_tasks and memory * 0.4 < worker_bytes:
            raise MemoryError(
                "Insufficient headroom for one cleaning worker; prepared data is unchanged"
            )
        count = min(
            requested,
            max(1, detect_visible_cpu_count() - 1),
            max(1, int(memory * 0.4) // worker_bytes),
            max(1, len(pending_tasks)),
        )
        if count == 1:
            print(
                f"Cleaning index: one worker; cpu={detect_visible_cpu_count()} "
                f"memory_budget={int(memory * 0.4)} worker_estimate={worker_bytes}",
                flush=True,
            )
        print(
            f"Cleaning index: {len(pending_tasks)}/{len(buckets)} buckets, workers={count}",
            flush=True,
        )
        with get_context("spawn").Pool(
            processes=count,
            initializer=_initialize,
            initargs=(
                str(root),
                str(root / "symbol-index.parquet"),
                str(output / "sessions.json"),
                policy,
                identity["window_size"],
                identity["splits"],
            ),
        ) as pool:
            tasks, pending = iter(pending_tasks), []
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * count:
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.append(pool.apply_async(_clean_bucket, (task,)))
                if pending:
                    # Pool context termination cancels and joins all children on
                    # timeout/exception; completed bucket files remain resumable.
                    bucket, ranges, accepted = pending.pop(0).get(timeout=300)
                    print(
                        f"Cleaning bucket {bucket}: {accepted} windows, {ranges} ranges", flush=True
                    )
        # Compact ranges are bounded by the number of symbol/gap transitions,
        # not by input_length times the number of training windows.
        import pyarrow.parquet as parquet

        estimated = 0
        for bucket in buckets:
            with parquet.ParquetFile(parts / f"{bucket:04d}.parquet") as part:
                estimated += (
                    sum(
                        part.metadata.row_group(i).total_byte_size
                        for i in range(part.metadata.num_row_groups)
                    )
                    * 8
                )
        if estimated > detect_available_memory().available_bytes * 0.4:
            raise MemoryError(
                "Compact cleaning range merge exceeds memory budget; bucket progress is saved"
            )
        ranges = pd.concat(
            [pd.read_parquet(parts / f"{b:04d}.parquet") for b in buckets], ignore_index=True
        ).sort_values(["split", "symbol", "start_index"])
        counts = {
            s: int(ranges.loc[ranges["split"] == s, "count"].sum())
            for s in ("train", "validation", "test")
        }
        temporary = output / "cutoff-ranges.pending.parquet"
        ranges.to_parquet(temporary, index=False)
        temporary.replace(output / "cutoff-ranges.parquet")
        audit, by_market = {}, {}
        for bucket in buckets:
            for row in json.loads((parts / f"{bucket:04d}.json").read_text())["audits"]:
                total = audit.setdefault(row["split"], Counter())
                market = by_market.setdefault(row["market"], {}).setdefault(row["split"], Counter())
                values = {k: row[k] for k in ("candidates", "accepted", "retained_extreme_windows")}
                values.update(row["excluded"])
                total.update(values)
                market.update(values)
        atomic_write_json(
            marker,
            {
                "state": "ready",
                "identity": identity,
                "split_counts": counts,
                "ranges_sha256": sha256_file(output / "cutoff-ranges.parquet"),
                "audit": audit,
                "by_market": by_market,
                "execution": {
                    "workers": count,
                    "worker_memory_estimate": worker_bytes,
                    "max_pending": 2 * count,
                    "windows_materialized": False,
                },
            },
        )
    return output


def split_bar_store(config, split: str, *, validate=True) -> Path:
    from stock_forecasting.training_paths import resolve_bar_store_path

    root = Path(config.data.bar_store_path)
    selection_path = os.environ.get("RUNPOD_SELECTION_FILE") or os.environ.get(
        "RUNPOD_REMOTE_SELECTION_PATH"
    )
    if (
        split != "train"
        and not selection_path
        and getattr(getattr(config, "validation", None), "require_prebuilt_baselines", False)
    ):
        raise ValueError(
            "Production evaluation requires the run selection to resolve its shared source; "
            "use the RunPod workflow instead of falling back to the training snapshot"
        )
    if split != "train" and selection_path:
        selection = json.loads(Path(selection_path).read_text())
        volume = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
        root = (
            volume
            / "datasets"
            / evaluation_dataset_id(selection, load_data_policy())
            / "prepared/bar-store"
        )
        if not root.is_dir():
            raise ValueError(
                "The shared evaluation bar store is missing; no fallback to the train snapshot"
            )
    return resolve_bar_store_path(root) if validate else root


def validated_split_roots(config) -> dict[str, Path]:
    """Verify each immutable source once per operation, not once per split."""
    from stock_forecasting.training_paths import resolve_bar_store_path

    roots = {s: split_bar_store(config, s, validate=False) for s in ("train", "validation", "test")}
    checked = {root: resolve_bar_store_path(root) for root in dict.fromkeys(roots.values())}
    return {s: checked[root] for s, root in roots.items()}


def dataset_provenance(dataset) -> dict:
    """Expose the actual source and cleaned population, not the preparation's old counts."""
    return {
        "split": dataset.split,
        "source": str(dataset.root),
        "samples": len(dataset),
        "sample_universe": dataset.sample_universe_identity,
        "audit": json.loads((dataset.sample_index_root / "universe.json").read_text()),
    }


def open_clean_dataset(config, split: str, *, validated_root=None, **kwargs):
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset

    root = Path(validated_root) if validated_root is not None else split_bar_store(config, split)
    index = ensure_sample_universe(root)
    return LazyFinancialWindowDataset(
        root,
        split=split,
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        sample_index_root=index,
        **kwargs,
    )
