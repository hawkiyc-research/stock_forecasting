#!/usr/bin/env python3
"""Upload a sealed forward input package to the RunPod network volume (stdlib only).

Every local file is re-hashed against forward-package.json first. Files go to
forward/<END>/package/ and forward-package.json is uploaded last, so its presence marks a
complete upload. A package already on the volume is never overwritten: the same digest is
reported as uploaded and a different one is refused. The GPU export re-hashes it again.
"""

import json
import logging
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger("forward-package-upload")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILE = "forward-package.json"
MAX_WORKERS = 8
UPLOAD_TIMEOUT_SECONDS = 1800
PROBES = runpy.run_path(str(Path(__file__).with_name("download_runpod_probes.py")))
S3Reader = PROBES["S3Reader"]
sha256_file = PROBES["sha256_file"]
SAFE_RELATIVE = re.compile(r"[A-Za-z0-9_.=-]+(/[A-Za-z0-9_.=-]+)*")


def package_end(value):
    try:
        valid = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Pass the package end date as YYYY-MM-DD")
    return value


def verified_files(package_root):
    """Relative paths of a sealed package after re-hashing every listed file."""
    payload = json.loads((package_root / PACKAGE_FILE).read_text(encoding="utf-8"))
    if payload.get("kind") != "forward-input-package":
        raise ValueError(f"Not a forward input package: {package_root}")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Forward package lists no files")
    for relative, digest in files.items():
        path = package_root / relative
        if (
            SAFE_RELATIVE.fullmatch(relative) is None
            or ".." in relative.split("/")
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Forward package file is missing or changed: {relative}")
    return payload, sorted(files)


def upload_file(reader, source, key):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                "bash", str(reader.wrapper), "s3", "cp", str(source),
                f"s3://{reader.volume_id}/{key}", "--only-show-errors",
            ],
            stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True,
        )
        try:
            code = process.wait(timeout=UPLOAD_TIMEOUT_SECONDS)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        # Do not echo transport output that could contain credential values.
        raise RuntimeError(f"RunPod S3 upload failed (exit {code}); retry the command")


def remote_package(reader, prefix):
    try:
        payload, _ = reader.read_json(prefix + PACKAGE_FILE)
    except RuntimeError:
        return None
    return payload


def main():
    arguments = sys.argv[1:]
    if len(arguments) != 1:
        raise ValueError("Usage: upload-forward-package YYYY-MM-DD")
    end = package_end(arguments[0])
    package_root = PROJECT_ROOT / "artifacts" / "forward" / end / "package"
    payload, files = verified_files(package_root)
    if payload.get("end_exclusive") != end:
        raise ValueError("The local package end date differs from the requested one")
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    prefix = f"forward/{end}/package/"
    existing = remote_package(reader, prefix)
    if existing is not None:
        if existing.get("package_sha256") != payload["package_sha256"]:
            raise ValueError(
                "A different forward package is already on the volume; not overwritten"
            )
        print(f"Forward package already uploaded: {payload['package_sha256']}")
        print(f"Pod path: /runpod-volume/{prefix.rstrip('/')}")
        return
    workers = int(os.environ.get("RUNPOD_FORWARD_UPLOAD_WORKERS", 4))
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"RUNPOD_FORWARD_UPLOAD_WORKERS must be between 1 and {MAX_WORKERS}")
    # I/O bound: each worker waits on one bounded AWS CLI process.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="forward-upload") as pool:
        futures = [
            pool.submit(upload_file, reader, package_root / relative, prefix + relative)
            for relative in files
        ]
        for count, future in enumerate(futures, start=1):
            future.result()
            if count % 25 == 0 or count == len(futures):
                print(f"Uploaded {count}/{len(futures)} files", flush=True)
    upload_file(reader, package_root / PACKAGE_FILE, prefix + PACKAGE_FILE)
    print(f"Forward package uploaded: {payload['package_sha256']}")
    print(f"Pod path: /runpod-volume/{prefix.rstrip('/')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        LOGGER.error("Forward package upload failed: %s", error)
        sys.exit(1)
