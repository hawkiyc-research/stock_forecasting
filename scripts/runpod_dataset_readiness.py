#!/usr/bin/env python3
"""Shared read-only admission from dataset-scoped manifests, independent of model caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
DATA = runpy.run_path(str(ROOT / "src/stock_forecasting/dataset_identity.py"))
RESOURCES = runpy.run_path(str(ROOT / "src/stock_forecasting/runtime_resources.py"))
MAX_METADATA_BYTES = 16 * 1024**2


def digest(payload):
    return READINESS["_payload_sha256"](payload)


def relative_path(value):
    path = PurePosixPath(value) if isinstance(value, str) else None
    if (
        path is None or path.is_absolute() or not path.parts or ".." in path.parts
        or path.as_posix() != value or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)
    ):
        raise ValueError("Unsafe data artifact path")
    return value


class ArtifactReader:
    def __init__(self, project: Path, *, volume: Path | None = None, bucket: str | None = None):
        self.project, self.volume, self.bucket = project, volume, bucket
        if volume is None and not re.fullmatch(r"[A-Za-z0-9_-]+", bucket or ""):
            raise ValueError("Network volume ID must be loaded from the project .env")

    def path(self, key):
        relative_path(key)
        path = self.volume / key
        if not path.resolve().is_relative_to(self.volume.resolve()):
            raise ValueError("Dataset data artifact escapes the mounted volume")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Dataset data artifact is missing or unsafe: {key}")
        return path

    def s3(self, *args):
        result = subprocess.run(
            ["bash", str(self.project / "scripts/runpod_s3_project.sh"), *args],
            capture_output=True, timeout=180,
        )
        if result.returncode:
            # Do not echo credentials, provider responses or environment values.
            raise ValueError(
                "Cannot read a required data artifact; no Pod should be created"
            )
        return result.stdout

    def size(self, key):
        relative_path(key)
        if self.volume is not None:
            return self.path(key).stat().st_size
        return int(self.s3(
            "s3api", "head-object", "--bucket", self.bucket, "--key", key,
            "--query", "ContentLength", "--output", "text",
        ))

    def read(self, key):
        size = self.size(key)
        if not 0 < size <= MAX_METADATA_BYTES:
            raise ValueError(f"Dataset metadata exceeds the bounded read limit: {key}")
        data = (
            self.path(key).read_bytes() if self.volume is not None
            else self.s3("s3", "cp", f"s3://{self.bucket}/{key}", "-", "--only-show-errors")
        )
        if len(data) != size:
            raise ValueError(f"Dataset metadata changed during its read: {key}")
        return data

    def verify(self, item):
        key, metadata = item
        if self.size(key) != metadata["size_bytes"]:
            raise ValueError(f"Dataset data artifact is truncated: {key}")
        if self.volume is not None:
            actual = READINESS["_sha256"](self.path(key))
            if actual != metadata["sha256"]:
                raise ValueError(f"Dataset data artifact checksum mismatch: {key}")


def bounded_map(function, items, workers):
    # AWS CLI subprocesses can use substantially more memory than the metadata itself.
    # Submission is capped at one batch per worker, not one future per shard/window.
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(iterator, workers)):
            yield from pool.map(function, batch)


def worker_count():
    requested = int(os.environ.get(
        "RUNPOD_PREFLIGHT_WORKERS", os.environ.get("RUNPOD_BASELINE_PREFLIGHT_WORKERS", "4")
    ))
    if requested < 1:
        raise ValueError("RUNPOD_BASELINE_PREFLIGHT_WORKERS must be positive")
    try:
        memory = RESOURCES["detect_available_memory"]().available_bytes
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=10)
        page_size = int(re.search(r"page size of ([0-9]+) bytes", stats)[1])
        pages = sum(
            int(re.search(rf"Pages {name}:\s+([0-9]+)", stats)[1])
            for name in ("free", "inactive")
        )
        memory = pages * page_size
    workers = max(1, min(
        requested, 8, RESOURCES["detect_visible_cpu_count"](),
        max(0, memory - 512 * 1024**2) // (256 * 1024**2),
    ))
    if workers == 1:
        print(
            f"Dataset preflight uses one worker: CPU/memory/request cap, available_bytes={memory}",
            file=sys.stderr,
        )
    return workers


def verify_data(
    project: Path, selection: dict, reader: ArtifactReader, *, workers: int,
    training_artifacts: bool = False,
) -> dict:
    request = selection["dataset_request"]
    prefix = f"datasets/{selection['dataset_request_sha256']}/"
    dataset = json.loads(reader.read(prefix + "dataset-manifest.json"))
    if (
        dataset.get("schema_version") != "4.0"
        or dataset.get("kind") != "ohlcv-bar-store-dataset" or dataset.get("state") != "ready"
    ):
        raise ValueError("Selected dataset is not fully prepared")
    for field, expected in (
        ("dataset_profile", request["profile"]),
        ("selected_datasets", request["selected_datasets"]),
        ("date_range", request["date_range"]),
        (
            "storage_preparation_spec",
            DATA["dataset_request_identity_payload"](request)["storage_preparation"],
        ),
    ):
        if dataset.get(field) != expected:
            raise ValueError(f"Selected dataset mismatch: {field}")
    if dataset.get("storage_preparation_spec_sha256") != digest(
        dataset["storage_preparation_spec"]
    ):
        raise ValueError("Dataset dataset storage contract digest mismatch")
    content, _ = READINESS["_project_content_identity"](project)
    expected_content = READINESS["_dataset_content_identity_from_code"](
        {"data_content_identity": content}, request["selected_datasets"]
    )
    if (
        dataset.get("data_content_identity") != expected_content
        or dataset.get("data_pipeline_digest") != digest(expected_content)
    ):
        raise ValueError("Dataset dataset numerical preparation contract mismatch")

    counts = dataset.get("split_counts", {})
    if set(counts) != {"train", "validation", "test"} or any(
        type(v) is not int or v < 1 for v in counts.values()
    ):
        raise ValueError("Dataset requires complete nonempty train/validation/test populations")
    READINESS["_validate_causal_split_audit"](
        dataset.get("split_audit"), counts, "Dataset dataset"
    )
    DATA["validate_fixed_split_audit"](
        dataset.get("split_audit"), request["preparation"]["fixed_split"],
        minimum_evaluation_dates=DATA["MIN_FIXED_EVALUATION_DATES"],
    )

    def record(metadata, base):
        if not isinstance(metadata, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", str(metadata.get("sha256", ""))
        ):
            raise ValueError("Dataset data artifact has no valid checksum")
        if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] < 1:
            raise ValueError("Dataset data artifact has no valid size")
        return base + relative_path(metadata.get("relative_path")), metadata

    artifacts = dataset.get("artifacts", {})
    for name in ("raw", "bar_store_manifest", "symbol_index", "cutoff_ranges"):
        record(artifacts.get(name), prefix)
    if artifacts["bar_store_manifest"]["relative_path"] != "prepared/bar-store/bar-store.json":
        raise ValueError("Dataset bar-store location is not canonical")
    bar_key = prefix + artifacts["bar_store_manifest"]["relative_path"]
    bar_bytes = reader.read(bar_key)
    bar_hash = hashlib.sha256(bar_bytes).hexdigest()
    if (
        bar_hash != artifacts["bar_store_manifest"]["sha256"]
        or len(bar_bytes) != artifacts["bar_store_manifest"]["size_bytes"]
    ):
        raise ValueError("Dataset bar-store manifest checksum/size mismatch")
    bar = json.loads(bar_bytes)
    if (
        bar.get("state") != "ready" or bar.get("kind") != DATA["BAR_STORE_KIND"]
        or bar.get("schema_version") != DATA["BAR_STORE_SCHEMA_VERSION"]
        or bar.get("split_counts") != counts or bar.get("split_audit") != dataset["split_audit"]
        or bar.get("identity_sha256") != digest(bar.get("identity"))
    ):
        raise ValueError("Dataset bar-store identity/population/audit mismatch")
    identity = bar["identity"]
    for field, value in dataset["storage_preparation_spec"].items():
        if field in (
            "bar_store_schema_version", "storage_kind", "window_materialized", "labels_materialized"
        ):
            continue
        if identity.get(field) != value:
            raise ValueError(f"Dataset bar-store preparation mismatch: {field}")
    if (
        identity.get("materialization_digest")
        != expected_content["bar_store_materialization_digest"]
        or identity.get("raw_sha256") != artifacts["raw"]["sha256"]
    ):
        raise ValueError("Dataset bar-store materialization differs from dataset")
    bar_prefix = prefix + "prepared/bar-store/"
    success = json.loads(reader.read(bar_prefix + "_SUCCESS.json"))
    if any(success.get(field) != expected for field, expected in (
        ("schema_version", DATA["BAR_STORE_SCHEMA_VERSION"]), ("kind", "bar-store-success"),
        ("state", "ready"), ("identity_sha256", bar["identity_sha256"]),
        ("bar_store_manifest_sha256", bar_hash), ("split_counts", counts),
        ("symbol_index_sha256", artifacts["symbol_index"]["sha256"]),
        ("cutoff_ranges_sha256", artifacts["cutoff_ranges"]["sha256"]),
    )):
        raise ValueError("Dataset bar-store publication is incomplete or inconsistent")

    download_record = dataset.get("download_manifest", {})
    if download_record.get("relative_path") != "download-manifest.json":
        raise ValueError("Dataset download manifest path is invalid")
    download_bytes = reader.read(prefix + "download-manifest.json")
    if hashlib.sha256(download_bytes).hexdigest() != download_record.get("sha256"):
        raise ValueError("Dataset download manifest checksum mismatch")
    download = json.loads(download_bytes)
    for field in ("dataset_profile", "selected_datasets", "date_range", "training_security_scope"):
        if download.get(field) != dataset.get(field):
            raise ValueError(f"Dataset download provenance mismatch: {field}")
    api = download.get("api_policy", {})
    for field, value in (
        ("symbol_limit", request["universe"]["symbol_limit"]),
        ("include_delisted", request["universe"]["include_delisted_us"]),
        ("cache_revision", request["revision"]),
    ):
        if api.get(field) != value:
            raise ValueError(f"Dataset data acquisition contract mismatch: {field}")
    if request["universe"]["mode"] == "explicit":
        required = set(request["universe"]["us_stocks"] + request["universe"]["us_etfs"])
        if not required <= set(dataset.get("symbols", {}).get("values", [])):
            raise ValueError("Dataset dataset is missing selected symbols")

    items = []
    if training_artifacts:
        # Preserve the main-model gate's raw/audit checks without reading a shared pointer.
        for metadata, expected_path in (
            (artifacts["raw"], "raw/market.parquet"),
            (dataset.get("request_log"), "manifests/api-request-log.jsonl"),
        ):
            key, entry = record(metadata, prefix)
            if entry["relative_path"] != expected_path:
                raise ValueError("Selected dataset acquisition artifact is not canonical")
            items.append((key, entry))
    for name, expected_path in (
        ("symbol_index", "symbol-index.parquet"), ("cutoff_ranges", "cutoff-ranges.parquet")
    ):
        entry = bar.get(name, {})
        expected = {**entry, "relative_path": "prepared/bar-store/" + expected_path}
        if entry.get("relative_path") != expected_path or artifacts[name] != expected:
            raise ValueError(f"Dataset index metadata mismatch: {name}")
        items.append(record(entry, bar_prefix))
    shards = bar.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Dataset bar-store contains no shards")
    seen, rows, groups = set(), 0, 0
    for shard in shards:
        key, entry = record(shard, bar_prefix)
        if key in seen or not re.fullmatch(
            r"shards/bucket-[0-9]{4}/shard.parquet", entry["relative_path"]
        ):
            raise ValueError("Dataset shard inventory is unsafe or duplicated")
        if any(type(entry.get(field)) is not int or entry[field] < 1
               for field in ("row_count", "row_groups")):
            raise ValueError("Dataset shard counts are invalid")
        seen.add(key)
        rows += entry["row_count"]
        groups += entry["row_groups"]
        items.append((key, entry))
    if rows != bar.get("row_count") or groups != bar.get("symbol_count"):
        raise ValueError("Dataset shard totals do not match the complete bar store")
    for _ in bounded_map(reader.verify, items, workers):
        pass
    return {
        "state": "ready", "scope": "prepared_bar_store",
        "dataset_request_sha256": selection["dataset_request_sha256"],
        "date_range": request["date_range"], "prepared_candidate_counts": counts,
        "bar_store_manifest_sha256": bar_hash, "verified_artifacts": len(items),
        "verification": (
            "checksums" if reader.volume is not None else "metadata-checksums-and-object-sizes"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--selection", default=os.environ.get("RUNPOD_SELECTION_FILE"))
    parser.add_argument("--network-volume-root", type=Path)
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    _, selection = SELECTION["_resolve_selection_path"](
        project, args.selection, validate_local_config=False
    )
    reader = ArtifactReader(
        project, volume=args.network_volume_root, bucket=os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )
    result = verify_data(project, selection, reader, workers=worker_count())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError, RuntimeError, KeyError, TypeError, OSError, subprocess.SubprocessError
    ) as error:
        print(f"Selected dataset readiness failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
