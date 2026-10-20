#!/usr/bin/env python3
"""Download a sealed forward input package from the RunPod network volume (stdlib only).

The counterpart of upload-forward-package for a package built by `cpu prepare --forward`.
forward/<END>/package/forward-package.json is read first and every listed file is
downloaded into a private staging directory and checked against its SHA-256; the
directory becomes artifacts/forward/<END>/package only when every file matches. A local
package with the same digest is reported and kept; a different one is never replaced.
"""

import hashlib
import json
import logging
import os
import re
import runpy
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOGGER = logging.getLogger("forward-package-download")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILE = "forward-package.json"
PACKAGE_KIND = "forward-input-package"
# Package manifests embed the fetch manifest and list every file.
MANIFEST_BYTES = 16 * 1024**2
SAFE_RELATIVE = re.compile(r"[A-Za-z0-9_.=-]+(/[A-Za-z0-9_.=-]+)*")
PACKAGE_UPLOAD = runpy.run_path(str(Path(__file__).with_name("upload_runpod_forward_package.py")))
PROBES = runpy.run_path(str(Path(__file__).with_name("download_runpod_probes.py")))
S3Reader = PROBES["S3Reader"]
sha256_file = PROBES["sha256_file"]
worker_count = PROBES["worker_count"]
parallel_results = PROBES["parallel_results"]
package_end = PACKAGE_UPLOAD["package_end"]


def canonical_sha256(value):
    """Same canonical JSON digest as stock_forecasting.data.manifest.canonical_json_sha256."""
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remote_manifest(reader, prefix, end):
    payload, _ = reader.read_json(prefix + PACKAGE_FILE, MANIFEST_BYTES)
    files = payload.get("files")
    if (
        payload.get("kind") != PACKAGE_KIND
        or payload.get("end_exclusive") != end
        or not isinstance(files, dict)
        or not files
        or payload.get("package_sha256") != canonical_sha256(files)
    ):
        raise ValueError("The forward package on the volume is not a sealed package for " + end)
    for relative, digest in files.items():
        if (
            SAFE_RELATIVE.fullmatch(relative) is None
            or ".." in relative.split("/")
            or relative == PACKAGE_FILE
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("The forward package lists an unsafe file: " + str(relative))
    return payload


def main():
    arguments = sys.argv[1:]
    if len(arguments) != 1:
        raise ValueError("Usage: download-forward-package YYYY-MM-DD")
    end = package_end(arguments[0])
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    prefix = f"forward/{end}/package/"
    payload = remote_manifest(reader, prefix, end)
    destination = PROJECT_ROOT / "artifacts" / "forward" / end / "package"
    if (destination / PACKAGE_FILE).exists():
        local, _files = PACKAGE_UPLOAD["verified_files"](destination)
        if local.get("package_sha256") != payload["package_sha256"]:
            raise ValueError("A different forward package exists locally; not replaced")
        print(f"Forward package already downloaded: {payload['package_sha256']}")
        print(f"Local path: {destination}")
        return
    if destination.exists():
        raise ValueError(f"An incomplete local package directory exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    workers = worker_count(PROJECT_ROOT)
    files = sorted(payload["files"].items())
    with tempfile.TemporaryDirectory(prefix=".package-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"

        def transfer(item):
            relative, digest = item
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            reader.download(prefix + relative, target)
            if sha256_file(target) != digest:
                raise ValueError("Downloaded package file SHA-256 mismatch: " + relative)
            return relative

        # I/O bound: bounded AWS CLI workers with at most one batch of futures pending.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="package-download") as pool:
            for count, _relative in enumerate(
                parallel_results(pool, transfer, files, workers), start=1
            ):
                if count % 100 == 0 or count == len(files):
                    print(f"Downloaded {count}/{len(files)} files", flush=True)
        reader.download(prefix + PACKAGE_FILE, staging / PACKAGE_FILE)
        downloaded, _files = PACKAGE_UPLOAD["verified_files"](staging)
        if downloaded.get("package_sha256") != payload["package_sha256"]:
            raise ValueError("The package changed on the volume during the download; retry")
        staging.rename(destination)
    print(f"Forward package downloaded: {payload['package_sha256']}")
    print(f"Local path: {destination}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError, MemoryError, subprocess.SubprocessError) as error:
        LOGGER.error("Forward package download failed: %s", error)
        sys.exit(1)
