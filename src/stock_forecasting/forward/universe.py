"""Sample-universe builder for forward input packages, with forward-only session changes.

Protocol 13.8 allows exchange-announced session changes from 2026-06-01 (for example
the 2026-07-10 typhoon closure the pinned XTAI calendar does not know) in the forward
package only. ``data/sample_universe.py`` must stay byte-identical: its file hash is part
of every baseline ID (``baseline_contract.BASELINE_SOURCES``), so editing it would orphan
the trained baselines. This module therefore repeats ``ensure_sample_universe`` with the
changes applied to the sessions and recorded in the identity, and reuses every worker
helper unchanged. ``BASE_BUILDER_SHA256`` pins the copied function: when the original
changes, tests/test_forward_package.py fails until this copy is reviewed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from stock_forecasting.data import sample_universe as base
from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data_policy import load_data_policy
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

BASE_BUILDER_SHA256 = "56e6b3c981514748270bab39e6c0ade3683cb0a789d359bceb0c4719c8397876"


def validated_calendar_additions(additions, policy) -> dict:
    """Exchange-announced session changes that apply to one sample universe only."""
    if not additions:
        return {}
    calendars = set(policy["calendars"].values())
    output = {}
    for name, changes in additions.items():
        if name not in calendars or not set(changes) <= {"open", "closed"}:
            raise ValueError("Calendar additions name a configured calendar with open/closed")
        for days in changes.values():
            if days != sorted(set(days)) or any(
                pd.Timestamp(day).date().isoformat() != day for day in days
            ):
                raise ValueError("Calendar addition dates must be sorted unique YYYY-MM-DD")
        output[name] = {kind: list(changes.get(kind, [])) for kind in ("open", "closed")}
        if set(output[name]["open"]) & set(output[name]["closed"]):
            raise ValueError("A calendar addition cannot both open and close a date")
    return output


def apply_calendar_additions(sessions: pd.DatetimeIndex, changes) -> pd.DatetimeIndex:
    if not changes:
        return sessions
    added = pd.DatetimeIndex(changes["open"]).tz_localize("UTC").as_unit("ns")
    removed = pd.DatetimeIndex(changes["closed"]).tz_localize("UTC").as_unit("ns")
    added = added[(added >= sessions.min()) & (added <= sessions.max())]
    return sessions.union(added).difference(removed)


def ensure_forward_sample_universe(
    root: Path,
    *,
    workers: int | None = None,
    calendar_additions=None,
):
    """ensure_sample_universe with forward-only session changes (see the module docstring).

    ``calendar_additions`` ({calendar: {"open": [...], "closed": [...]}}) changes the
    sessions and is recorded in the identity; without it the identity, sessions and
    ranges are those of ensure_sample_universe.
    """
    policy = load_data_policy()
    if version("exchange_calendars") != policy["calendar_version"]:
        raise ValueError("The offline calendar version differs from the data-cleaning policy")
    additions = validated_calendar_additions(calendar_additions, policy)
    manifest = json.loads((root / "bar-store.json").read_text())
    identity = {
        "algorithm": base.ALGORITHM,
        "source_manifest_sha256": sha256_file(root / "bar-store.json"),
        "policy": policy,
        "window_size": manifest["identity"]["window_size"],
        "splits": [list(row) for row in base._split_spec(manifest)],
    }
    if additions:
        identity["calendar_additions"] = additions
    key = canonical_json_sha256(identity)
    parent = root.parent / "sample-universes"
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
    with base._exclusive_build(output):
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
            calendar = policy["calendars"][market]
            sessions[market] = [
                v.isoformat()
                for v in apply_calendar_additions(
                    base.market_sessions(calendar, start, end), additions.get(calendar)
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
            initializer=base._initialize,
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
                        pending.append(pool.apply_async(base._clean_bucket, (task,)))
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
