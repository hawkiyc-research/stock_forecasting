#!/usr/bin/env python3
"""Add a local forward fetch cache to the network volume's shared provider cache (stdlib only).

Uploads artifacts/forward/<END>/shared-cache (written by forward_pipeline export-cache) to
datasets/forward-taiwan/api-cache/, the cache the CPU preparation's forward mode writes and
every dataset preparation reads. Every local file is re-hashed against shared-cache.json
first. Cache entries are immutable: a key already on the volume with the same size is
kept, one with a different size stops the upload, and nothing is overwritten or deleted.
"""

import json
import logging
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOGGER = logging.getLogger("forward-cache-upload")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = "shared-cache.json"
EXPORT_KIND = "forward-shared-cache-export"
# Must match SHARED_CACHE_ROOT in scripts/runpod_cpu_prepare_forward.sh.
REMOTE_NAMESPACE = "datasets/forward-taiwan/"
REMOTE_CACHE = REMOTE_NAMESPACE + "api-cache/"
CACHE_ENTRY = re.compile(r"[a-z_]+/[0-9a-f]{2}/[0-9a-f]{64}\.json")
MAX_WORKERS = 8
PACKAGE_UPLOAD = runpy.run_path(str(Path(__file__).with_name("upload_runpod_forward_package.py")))
PROBES = runpy.run_path(str(Path(__file__).with_name("download_runpod_probes.py")))
S3Reader = PROBES["S3Reader"]
sha256_file = PROBES["sha256_file"]
upload_file = PACKAGE_UPLOAD["upload_file"]
package_end = PACKAGE_UPLOAD["package_end"]


def verified_export(export_root, end):
    """The export's relative paths and sizes after re-hashing every listed file."""
    manifest = json.loads((export_root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("kind") != EXPORT_KIND or manifest.get("end_exclusive") != end:
        raise ValueError(f"Not a shared-cache export for {end}: {export_root}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The shared-cache export lists no files")
    sizes = {}
    for relative, digest in sorted(files.items()):
        path = export_root / relative
        if (
            CACHE_ENTRY.fullmatch(relative) is None
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Shared-cache export file is missing or changed: {relative}")
        sizes[relative] = path.stat().st_size
    return manifest, sizes


def remote_sizes(reader):
    """Sizes of every cache entry already on the volume, keyed by relative path."""
    sizes, token = {}, None
    while True:
        arguments = [
            "s3api", "list-objects-v2", "--bucket", reader.volume_id, "--prefix", REMOTE_CACHE,
            "--max-keys", "1000", "--no-paginate", "--output", "json",
        ]
        if token:
            arguments.extend(("--continuation-token", token))
        page, _ = reader.json_command(arguments)
        for entry in page.get("Contents", []):
            key = entry.get("Key") if isinstance(entry, dict) else None
            if not isinstance(key, str) or not key.startswith(REMOTE_CACHE):
                raise ValueError("Cache listing escaped the requested prefix")
            sizes[key[len(REMOTE_CACHE):]] = entry.get("Size")
        if page.get("IsTruncated") is not True:
            return sizes
        next_token = page.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise ValueError("Invalid cache listing continuation")
        token = next_token


def main():
    arguments = sys.argv[1:]
    if len(arguments) != 1:
        raise ValueError("Usage: upload-forward-cache YYYY-MM-DD")
    end = package_end(arguments[0])
    export_root = PROJECT_ROOT / "artifacts" / "forward" / end / "shared-cache"
    manifest, local = verified_export(export_root, end)
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    remote = remote_sizes(reader)
    conflicts = sorted(key for key in local if key in remote and remote[key] != local[key])
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} cache entries differ from the volume; nothing is overwritten "
            f"(first: {conflicts[0]})"
        )
    pending = [key for key in sorted(local) if key not in remote]
    print(f"Cache entries: {len(local)}; already on the volume: {len(local) - len(pending)}; "
          f"to upload: {len(pending)}", flush=True)
    workers = int(os.environ.get("RUNPOD_FORWARD_UPLOAD_WORKERS", 4))
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"RUNPOD_FORWARD_UPLOAD_WORKERS must be between 1 and {MAX_WORKERS}")
    # I/O bound: each worker waits on one bounded AWS CLI process; submit in bounded batches.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-upload") as pool:
        reported = 0
        for start in range(0, len(pending), workers * 4):
            batch = pending[start : start + workers * 4]
            for future in [
                pool.submit(upload_file, reader, export_root / key, REMOTE_CACHE + key)
                for key in batch
            ]:
                future.result()
            done = start + len(batch)
            if done - reported >= 100 or done == len(pending):
                reported = done
                print(f"Uploaded {done}/{len(pending)} entries", flush=True)
    remote = remote_sizes(reader)
    missing = sorted(key for key in local if remote.get(key) != local[key])
    if missing:
        raise RuntimeError(f"{len(missing)} cache entries are missing after upload; retry")
    upload_file(
        reader, export_root / MANIFEST_FILE, f"{REMOTE_NAMESPACE}exports/{end}.json"
    )
    print(f"Shared cache now holds this fetch: {len(local)} entries "
          f"({manifest['rekeyed_complete_months']} complete months re-keyed, "
          f"{len(manifest['skipped_unfinished_months'])} unfinished months left out)")
    print(f"Pod path: /runpod-volume/{REMOTE_CACHE.rstrip('/')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        LOGGER.error("Forward cache upload failed: %s", error)
        sys.exit(1)
