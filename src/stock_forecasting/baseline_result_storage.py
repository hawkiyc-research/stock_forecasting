"""Publish one shared evaluation population without changing baseline numerics."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stock_forecasting.baseline_contract import (
    duplicate_evaluation_paths,
    shared_evaluation_paths,
    validate_complete,
)

CHUNK_BYTES = 4 * 1024**2
MAX_ARTIFACTS = 4096
DUPLICATE_PATH = re.compile(
    r"jobs/(?:rules/[a-z0-9_]+|[a-z0-9_]+-[0-9]+)/"
    r"(validation|test)/(membership|targets)\.npy"
)


def _regular(root: Path, relative: str):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe baseline artifact: {relative}")
    if path.resolve(strict=True) != path:
        raise ValueError(f"Symlinked baseline artifact: {relative}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Nonregular baseline artifact: {relative}")
    return info


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _atomic_json(path: Path, payload: dict):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class _ReadBudget:
    """Bound aggregate read traffic across a small thread pool."""

    def __init__(self, mib_per_second):
        if not 1 <= mib_per_second <= 1024:
            raise ValueError("Read budget must be between 1 and 1024 MiB/s")
        self.rate = mib_per_second * 1024**2
        self.next_at = time.monotonic()
        self.mutex = threading.Lock()

    def reserve(self, size):
        with self.mutex:
            now = time.monotonic()
            scheduled = max(now, self.next_at)
            self.next_at = scheduled + size / self.rate
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def _workers(requested):
    from stock_forecasting.runtime_resources import (
        detect_available_memory,
        detect_visible_cpu_count,
    )

    requested = int(requested or os.environ.get("BASELINE_STORAGE_WORKERS", "2"))
    if not 1 <= requested <= 8:
        raise ValueError("Baseline storage workers must be between 1 and 8")
    available = detect_available_memory().available_bytes
    count = min(requested, detect_visible_cpu_count(), max(1, available // (128 * 1024**2)))
    if count == 1:
        print(
            f"Baseline storage verification: one worker; available_memory={available}", flush=True
        )
    return count


def _verify_files(root, files, *, workers, budget, progress):
    def verify(item):
        relative, metadata = item
        before = _regular(root, relative)
        if before.st_size != metadata["bytes"]:
            raise ValueError(f"Baseline artifact size mismatch: {relative}")
        digest = hashlib.sha256()
        with (root / relative).open("rb") as stream:
            while True:
                budget.reserve(CHUNK_BYTES)
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
        if digest.hexdigest() != metadata["sha256"]:
            raise ValueError(f"Baseline artifact checksum mismatch: {relative}")
        after = _regular(root, relative)
        if _signature(before) != _signature(after):
            raise ValueError(f"Baseline artifact changed during verification: {relative}")
        return relative, _signature(after)

    verified = {}
    items = list(files.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit bounded batches rather than one future per artifact.
        for start in range(0, len(items), workers):
            for relative, signature in pool.map(verify, items[start : start + workers]):
                verified[relative] = signature
                if progress is not None:
                    progress(f"Verified baseline storage: {len(verified)}/{len(items)} {relative}")
    return verified


def _new_payload(original):
    payload = copy.deepcopy(original)
    shared = shared_evaluation_paths()
    groups = {}
    artifacts = payload["artifacts"]
    duplicates = {}
    for relative, metadata in list(artifacts.items()):
        match = DUPLICATE_PATH.fullmatch(relative)
        if match is None:
            continue
        split, kind = match.groups()
        destination = shared[split][kind]
        if destination in groups and groups[destination] != metadata:
            raise ValueError(f"Baseline models disagree on shared data: {destination}")
        groups[destination] = metadata
        duplicates[relative] = metadata
        del artifacts[relative]
    expected = {relative for paths in shared.values() for relative in paths.values()}
    if set(groups) != expected:
        raise ValueError("Staged baseline outputs do not contain both complete evaluation splits")
    artifacts.update(groups)
    payload["evaluation_data"] = shared
    validate_complete(payload, payload["identity"], require_shared=True)
    return payload, duplicates, groups


def finalize_baseline_storage(
    root: Path,
    *,
    apply: bool = False,
    workers: int | None = None,
    read_mib_per_second: int = 64,
    progress=print,
) -> dict:
    """Verify, publish shared references, then remove only proven redundant arrays.

    The tabular input cache already holds exactly the same canonical population.
    Keep those files and every prediction/weight/metric byte unchanged. An atomic
    completion update precedes removal; interrupted cleanup is safely repeatable.
    """
    root = Path(root).absolute()
    if root.resolve(strict=True) != root or root.parent.name != "baselines":
        raise ValueError("Storage finalization requires a direct, nonsymlink baseline directory")
    if not re.fullmatch(r"baseline-[A-Za-z0-9][A-Za-z0-9_-]*", root.name):
        raise ValueError("Invalid baseline directory name")
    complete = root / "complete.json"
    _regular(root, "complete.json")
    original_bytes = complete.read_bytes()
    current = json.loads(original_bytes)
    if current.get("identity", {}).get("baseline_id") != root.name:
        raise ValueError("Baseline directory and recorded identity differ")
    validate_complete(current, current["identity"])
    if len(current["artifacts"]) > MAX_ARTIFACTS:
        raise ValueError("Baseline artifact list exceeds the bounded suite limit")

    receipt = root / "storage-finalization.json"
    if current.get("evaluation_data") is None:
        updated, duplicates, shared = _new_payload(current)
    else:
        validate_complete(current, current["identity"], require_shared=True)
        updated = current
        shared = {
            path: current["artifacts"][path]
            for paths in shared_evaluation_paths().values() for path in paths.values()
        }
        # Enumerate exact job destinations from saved model/seed parameters, not a glob.
        duplicates = {
            relative: shared[source]
            for relative, source in duplicate_evaluation_paths(current).items()
            if (root / relative).exists() or (root / relative).is_symlink()
        }
    for relative in duplicates:
        if DUPLICATE_PATH.fullmatch(relative) is None:
            raise ValueError(f"Refusing an out-of-scope removal: {relative}")
    protected = {}
    for relative, metadata in current["artifacts"].items():
        if relative not in duplicates:
            info = _regular(root, relative)
            if info.st_size != metadata["bytes"]:
                raise ValueError(f"Protected artifact size mismatch: {relative}")
            protected[relative] = _signature(info)
    count = _workers(workers)
    verified = _verify_files(
        root, {**duplicates, **shared}, workers=count,
        budget=_ReadBudget(read_mib_per_second), progress=progress,
    )
    if complete.read_bytes() != original_bytes:
        raise ValueError("Baseline completion changed during storage verification")
    for relative, signature in protected.items():
        if _signature(_regular(root, relative)) != signature:
            raise ValueError(f"Protected baseline artifact changed: {relative}")
    for relative, signature in verified.items():
        if _signature(_regular(root, relative)) != signature:
            raise ValueError(f"Verified baseline artifact changed: {relative}")
    result = {
        "baseline_id": root.name,
        "apply": apply,
        "duplicate_files": len(duplicates),
        "duplicate_bytes": sum(item["bytes"] for item in duplicates.values()),
        "shared_files": list(shared),
        "workers": count,
        "payload": updated,
    }
    if not apply:
        return result
    # This receipt records progress, not a model/data identity or a reuse gate.
    audit = {key: value for key, value in result.items() if key != "payload"}
    audit.update(
        state="verified", source_complete_sha256=hashlib.sha256(original_bytes).hexdigest()
    )
    if duplicates:
        _atomic_json(receipt, audit)
    if updated != current:
        _atomic_json(complete, updated)
    deleted = 0
    for relative in sorted(duplicates):
        if _signature(_regular(root, relative)) != verified[relative]:
            raise ValueError(f"Duplicate changed before removal: {relative}")
        match = DUPLICATE_PATH.fullmatch(relative)
        assert match is not None
        split, kind = match.groups()
        canonical = shared_evaluation_paths()[split][kind]
        if _signature(_regular(root, canonical)) != verified[canonical]:
            raise ValueError(f"Shared baseline data changed before removal: {canonical}")
        (root / relative).unlink()
        deleted += 1
    for relative, signature in protected.items():
        if _signature(_regular(root, relative)) != signature:
            raise ValueError(f"Protected baseline artifact changed: {relative}")
    for relative in shared:
        if _signature(_regular(root, relative)) != verified[relative]:
            raise ValueError(f"Shared baseline data changed: {relative}")
    if json.loads(complete.read_bytes()) != updated:
        raise ValueError("Published baseline completion changed unexpectedly")
    if duplicates:
        audit.update(state="complete", deleted_files=deleted, finished_at=time.time())
        _atomic_json(receipt, audit)
    elif receipt.is_file():
        # A crash after the final unlink may leave only the progress note stale.
        # This note never determines whether the shared arrays can be consumed.
        pending = json.loads(receipt.read_text())
        if pending.get("state") == "verified":
            pending.update(
                state="complete", deleted_files=pending["duplicate_files"], finished_at=time.time()
            )
            _atomic_json(receipt, pending)
    result["deleted_files"] = deleted
    return result
