#!/usr/bin/env python3
"""Download the rank-label and feature stores built by one cpu-features job.

Every file is verified against the SHA-256 its store manifest records; verified files are kept
on a rerun. Transfers go through the credential-isolated S3 wrapper, a bounded number at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

LIFECYCLE_KEY = "lifecycle/research/features.json"
LIFECYCLE_KIND = "research-features"
REMOTE_ROOT = "diagnostics/features"
SESSION_NAME = "stock-forecasting-cpu-features"
LAUNCH_PATTERN = re.compile(r"launch-[A-Za-z0-9_-]{1,150}")
STORE_FILES = {
    "rank_labels": ("rank-labels.json", "label-index.parquet", "label_index_sha256"),
    "features": ("features.json", "feature-index.parquet", "feature_index_sha256"),
}
MAX_WORKERS = 4
MAX_JSON_BYTES = 16 * 1024 * 1024


class Reader:
    """Read objects of the project volume through the S3 wrapper only."""

    def __init__(self, project_root: Path, volume_id: str, timeout: int) -> None:
        self.wrapper = project_root / "scripts/runpod_s3_project.sh"
        self.volume_id = volume_id
        self.timeout = timeout

    def _copy(self, key: str, destination: str) -> subprocess.CompletedProcess:
        command = [
            "bash", str(self.wrapper), "s3", "cp",
            f"s3://{self.volume_id}/{key}", destination, "--only-show-errors",
        ]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE if destination == "-" else subprocess.DEVNULL,
                stderr=errors,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=self.timeout)
            except BaseException:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        return subprocess.CompletedProcess(command, process.returncode, output, b"")

    def read_json(self, key: str) -> dict | None:
        result = self._copy(key, "-")
        if result.returncode != 0:
            return None
        if len(result.stdout) > MAX_JSON_BYTES:
            raise ValueError(f"Remote JSON is unexpectedly large: {key}")
        payload = json.loads(result.stdout.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Remote JSON is not an object: {key}")
        return payload

    def download(self, key: str, destination: Path, sha256: str | None) -> bool:
        """Fetch one object into place; returns False when it is not on the volume."""
        if destination.is_file() and (sha256 is None or _sha256(destination) == sha256):
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + ".part")
        pending.unlink(missing_ok=True)
        result = self._copy(key, str(pending))
        if result.returncode != 0 or not pending.is_file():
            pending.unlink(missing_ok=True)
            return False
        if sha256 is not None and _sha256(pending) != sha256:
            pending.unlink(missing_ok=True)
            # Do not echo transport output that could contain credential values.
            raise ValueError(f"Downloaded file failed its SHA-256 check: {key}")
        pending.replace(destination)
        return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker_count() -> int:
    requested = int(os.environ.get("RUNPOD_FEATURE_DOWNLOAD_WORKERS", 2))
    if not 1 <= requested <= MAX_WORKERS:
        raise ValueError(f"RUNPOD_FEATURE_DOWNLOAD_WORKERS must be between 1 and {MAX_WORKERS}")
    return min(requested, os.cpu_count() or 1)


def transfer_timeout() -> int:
    value = int(os.environ.get("RUNPOD_FEATURE_TRANSFER_TIMEOUT", 1800))
    if not 60 <= value <= 86400:
        raise ValueError("RUNPOD_FEATURE_TRANSFER_TIMEOUT must be 60-86400 seconds")
    return value


def safe_relative(value: object) -> str:
    """A volume-relative store path: datasets/<sha256>/prepared/<kind>/<identity>."""
    if not isinstance(value, str) or not re.fullmatch(
        r"datasets/[0-9a-f]{64}/prepared/(rank-labels|features)/[0-9a-f]{64}", value
    ):
        raise ValueError("Feature manifest names an unexpected store path")
    return value


def store_plan(reader: Reader, kind: str, entry: dict, local_root: Path) -> list[tuple]:
    """Download the store manifest and index now; return the shard transfers to run."""
    manifest_name, index_name, index_field = STORE_FILES[kind]
    remote = safe_relative(entry.get("volume_relative"))
    # artifacts/features/<dataset sha256>/<rank-labels|features>/<identity>/
    local = local_root / remote.split("/")[1] / Path(*remote.split("/")[3:])
    manifest_path = local / manifest_name
    expected = entry.get("manifest_sha256")
    if not reader.download(f"{remote}/{manifest_name}", manifest_path, expected):
        raise FileNotFoundError(f"Store manifest is missing on the volume: {remote}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("state") != "ready":
        raise ValueError(f"Store is not ready on the volume: {remote}")
    if not reader.download(f"{remote}/{index_name}", local / index_name, payload[index_field]):
        raise FileNotFoundError(f"Store index is missing on the volume: {remote}")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Store manifest lists no shards: {remote}")
    transfers = []
    for relative, metadata in sorted(files.items()):
        if not re.fullmatch(r"shards/bucket-[0-9]{4}/shard\.parquet", relative):
            raise ValueError(f"Store manifest names an unexpected shard: {relative}")
        transfers.append((f"{remote}/{relative}", local / relative, str(metadata["sha256"])))
    return transfers


def run_transfers(reader: Reader, transfers: list[tuple], workers: int) -> int:
    """Bounded submission: at most 2 * workers transfers are pending at once."""
    done = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="feature-download") as pool:
        pending = []
        for key, destination, sha256 in transfers:
            pending.append(pool.submit(reader.download, key, destination, sha256))
            if len(pending) >= 2 * workers:
                if not pending.pop(0).result():
                    raise FileNotFoundError("A store shard is missing on the volume")
                done += 1
        for future in pending:
            if not future.result():
                raise FileNotFoundError("A store shard is missing on the volume")
            done += 1
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("launch_id", nargs="?")
    args = parser.parse_args(argv)
    volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id):
        print("RUNPOD_NETWORK_VOLUME_ID must be loaded from .env", file=sys.stderr)
        return 2
    if args.launch_id is not None and LAUNCH_PATTERN.fullmatch(args.launch_id) is None:
        print("LAUNCH_ID must look like launch-...", file=sys.stderr)
        return 2
    reader = Reader(args.project_root.resolve(), volume_id, transfer_timeout())
    workers = worker_count()
    launch_id, lifecycle = args.launch_id, None
    if launch_id is None:
        lifecycle = reader.read_json(LIFECYCLE_KEY)
        if lifecycle is None:
            print("No feature-build lifecycle marker is on the volume", file=sys.stderr)
            return 4
        launch_id = str(lifecycle.get("launch_id", ""))
        if lifecycle.get("kind") != LIFECYCLE_KIND or LAUNCH_PATTERN.fullmatch(launch_id) is None:
            print("The feature-build lifecycle marker is invalid", file=sys.stderr)
            return 4
        if lifecycle.get("state") == "preparing":
            print(f"Feature build {launch_id} is still running; try again later", file=sys.stderr)
            return 3
        print(f"Latest feature build: {launch_id} (state={lifecycle.get('state')})")
    local_root = args.project_root / "artifacts/features"
    launch_dir = local_root / "launches" / launch_id
    launch_dir.mkdir(parents=True, exist_ok=True)
    if lifecycle is not None:
        (launch_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
    for name in ("combined.log", "status.json"):
        if not reader.download(
            f"logs/tmux/{SESSION_NAME}/{launch_id}/{name}", launch_dir / "logs" / name, None
        ):
            print(f"Not found on the volume: {name}", file=sys.stderr)
    manifest_key = f"{REMOTE_ROOT}/{launch_id}/features-manifest.json"
    if not reader.download(manifest_key, launch_dir / "features-manifest.json", None):
        print("The feature build is incomplete; read logs/combined.log for the failure",
              file=sys.stderr)
        return 4
    manifest = json.loads((launch_dir / "features-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0" or manifest.get("kind") != "feature-library-build":
        print("The feature manifest is unsupported", file=sys.stderr)
        return 4
    transfers = []
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        for kind in STORE_FILES:
            transfers += store_plan(reader, kind, entry[kind], local_root)
    count = run_transfers(reader, transfers, workers)
    print(f"Verified {count} shard files for {launch_id} under {local_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
