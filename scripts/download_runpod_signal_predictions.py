#!/usr/bin/env python3
"""Select, download and verify completed signal-prediction exports without model dependencies."""

# Keep the local control plane compatible with pre-3.11 system Python.
# ruff: noqa: UP017

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
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("signal-prediction-download")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "diagnostics/signal-predictions/"
LOCAL_ROOT = Path("artifacts/diagnostics/signal-predictions")
SPLITS = ("validation", "test")
MAX_RUNS = 16
MAX_WORKERS = 4
WORKER_MEMORY_BYTES = 64 * 1024**2
MAX_STATUS_BYTES = 64 * 1024
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 1800
PROBES = runpy.run_path(str(Path(__file__).with_name("download_runpod_probes.py")))
S3Reader = PROBES["S3Reader"]
parallel_results = PROBES["parallel_results"]
sha256_file = PROBES["sha256_file"]
safe_local_path = PROBES["safe_local_path"]
# The decomposition analysis owns the current baseline identities and inputs.
ANALYSIS = runpy.run_path(str(Path(__file__).with_name("analyze_signal_decomposition.py")))
BASELINE_LOCAL_ROOT = Path("artifacts/diagnostics/signal-decomposition/baselines")


def baseline_files():
    """List the evaluation arrays and small records the decomposition reads."""
    alpha, direction = ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]
    files = [
        f"{direction}/inputs/train/class-label-audit.json",
    ]
    for split in SPLITS:
        files.extend((
            f"{alpha}/inputs/{split}/metadata.npy",
            f"{alpha}/inputs/{split}/targets.npy",
            f"{direction}/inputs/{split}/absolute_targets.npy",
        ))
        for model in ANALYSIS["BASELINE_MODELS"]:
            files.extend((
                f"{alpha}/jobs/{model}/{split}/predictions.npy",
                f"{direction}/jobs/{model}/{split}/predictions.npy",
            ))
    return files


def safe_run_id(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value) is None
        or "--" in value
    ):
        raise ValueError("RUN_ID must be a signal child run ID, not a path or option")
    return value


def worker_count(project_root):
    """Bound transfer threads by request, visible CPUs and reclaimable memory."""
    requested = int(os.environ.get("RUNPOD_SIGNAL_PREDICTION_DOWNLOAD_WORKERS", 2))
    if not 1 <= requested <= MAX_WORKERS:
        raise ValueError("RUNPOD_SIGNAL_PREDICTION_DOWNLOAD_WORKERS must be between 1 and 4")
    resources = runpy.run_path(str(project_root / "src/stock_forecasting/runtime_resources.py"))
    cpus = resources["detect_visible_cpu_count"]()
    try:
        available = resources["detect_available_memory"]().available_bytes
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        )
        page_size = re.search(r"page size of (\d+) bytes", result.stdout)
        counts = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)\.", result.stdout)
        if page_size is None or not counts:
            raise RuntimeError("Unable to determine available macOS memory") from None
        available = int(page_size.group(1)) * sum(map(int, counts))
    memory_limit = available // 4 // WORKER_MEMORY_BYTES
    if memory_limit < 1:
        raise MemoryError("Insufficient memory headroom for one bounded S3 download worker")
    return min(requested, cpus, memory_limit)


def transfer_timeout():
    value = int(
        os.environ.get(
            "RUNPOD_SIGNAL_PREDICTION_TRANSFER_TIMEOUT", DEFAULT_TRANSFER_TIMEOUT_SECONDS
        )
    )
    if not 60 <= value <= 86400:
        raise ValueError("RUNPOD_SIGNAL_PREDICTION_TRANSFER_TIMEOUT must be 60-86400 seconds")
    return value


def download_file(reader, key, destination, timeout):
    """Stream one object to a new file through the credential-isolated wrapper."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                "bash", str(reader.wrapper), "s3", "cp",
                f"s3://{reader.volume_id}/{key}", str(destination), "--only-show-errors",
            ],
            stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code or not destination.is_file():
        # Do not echo transport output that could contain credential values.
        raise RuntimeError(f"RunPod S3 transfer failed (exit {code}); retry the command")


def candidate(reader, key):
    parts = key.split("/")
    if (
        len(parts) != 6 or "/".join(parts[:2]) + "/" != REMOTE_ROOT
        or re.fullmatch(r"checkpoint-[0-9]{6,}", parts[3]) is None
        or re.fullmatch(r"export-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", parts[4]) is None
        or parts[5] != "status.json"
    ):
        return None
    run_id = safe_run_id(parts[2])
    expected_checkpoint = f"/runpod-volume/savedModel/{run_id}/{parts[3]}"
    status, status_hash = reader.read_json(key, MAX_STATUS_BYTES)
    if status.get("state") in {"running", "failed"}:
        return None
    if status.get("state") != "complete" or status.get("checkpoint") != expected_checkpoint:
        raise ValueError("Invalid completed export status: " + key)
    prefix = key[: -len("status.json")]
    report, report_hash = reader.read_json(prefix + "export.json")
    splits = report.get("splits")
    if (
        report.get("schema_version") != "1.0"
        or report.get("kind") != "signal-prediction-export"
        or report.get("run_id") != run_id
        or not isinstance(report.get("checkpoint"), dict)
        or report["checkpoint"].get("path") != expected_checkpoint
        or not isinstance(splits, dict)
        or not splits
        or set(splits) - set(SPLITS)
    ):
        raise ValueError("Export report identity does not match its completed status: " + key)
    for split, summary in splits.items():
        digest = summary.get("predictions_sha256") if isinstance(summary, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Export report has an invalid {split} SHA-256: {key}")
    completed = datetime.fromisoformat(str(report.get("created_at")).replace("Z", "+00:00"))
    if completed.utcoffset() is None:
        raise ValueError("Export report timestamp must include a timezone: " + key)
    return {
        "order": (completed.astimezone(timezone.utc), parts[3], parts[4]),
        "run_id": run_id,
        "relative": Path(*parts[2:5]),
        "prefix": prefix,
        "splits": {split: summary["predictions_sha256"] for split, summary in splits.items()},
        "hashes": {"export.json": report_hash, "status.json": status_hash},
    }


def select_export(reader, run_id):
    newest = None
    for key in reader.status_keys(REMOTE_ROOT + safe_run_id(run_id) + "/"):
        result = candidate(reader, key)
        if result is not None and (newest is None or result["order"] > newest["order"]):
            newest = result
    if newest is None:
        raise FileNotFoundError("No completed signal-prediction export found for " + run_id)
    return newest


def download_export(reader, result, project_root, executor, workers, timeout):
    destination = project_root / LOCAL_ROOT / result["relative"]
    safe_local_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = ["export.json", "export.log"]
    for split in result["splits"]:
        names.extend((f"{split}/split.json", f"{split}/predictions.npy"))
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for split in result["splits"]:
            (staging / split).mkdir()

        def transfer(name):
            target = staging / name
            download_file(reader, result["prefix"] + name, target, timeout)
            if target.stat().st_size == 0:
                raise ValueError("Downloaded export artifact is empty: " + name)
            expected = result["hashes"].get(name)
            split = name.split("/", 1)[0]
            if name.endswith("predictions.npy"):
                expected = result["splits"][split]
            if expected is not None and sha256_file(target) != expected:
                raise ValueError("Downloaded export SHA-256 mismatch: " + name)

        for _ in parallel_results(executor, transfer, names, workers):
            pass
        for split in result["splits"]:
            summary = json.loads((staging / split / "split.json").read_text(encoding="utf-8"))
            if summary.get("predictions_sha256") != result["splits"][split]:
                raise ValueError(f"{split}/split.json disagrees with export.json")
        # The status file is downloaded last so a local copy is complete only when verified.
        transfer("status.json")
        safe_local_path(destination)
        destination.mkdir(exist_ok=True)
        for split in result["splits"]:
            (destination / split).mkdir(exist_ok=True)
        for name in [*names, "status.json"]:
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Unsafe existing export artifact: " + str(target))
            (staging / name).replace(target)
    return destination


def download_baseline_arrays(reader, project_root, executor, workers, timeout, fetch=None):
    """Fetch decomposition inputs, verifying every array against its baseline manifest."""
    fetch = download_file if fetch is None else fetch
    destination = project_root / BASELINE_LOCAL_ROOT
    safe_local_path(destination)
    manifests = {}
    for baseline in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]):
        payload, _digest = reader.read_json(f"baselines/{baseline}/complete.json")
        if payload.get("state") != "complete" or not isinstance(payload.get("artifacts"), dict):
            raise ValueError(f"Baseline {baseline} is not a complete published result")
        manifests[baseline] = payload
        target = destination / baseline / "complete.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def transfer(relative):
        baseline, _, inner = relative.partition("/")
        expected = manifests[baseline]["artifacts"].get(inner, {}).get("sha256")
        target = destination / relative
        if expected is not None and target.is_file() and sha256_file(target) == expected:
            return "verified"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
            staged = Path(temporary) / target.name
            fetch(reader, f"baselines/{relative}", staged, timeout)
            if expected is not None and sha256_file(staged) != expected:
                raise ValueError("Downloaded baseline SHA-256 mismatch: " + relative)
            staged.replace(target)
        return "downloaded"

    states = list(parallel_results(executor, transfer, baseline_files(), workers))
    return destination, states.count("downloaded"), states.count("verified")


def main():
    arguments = sys.argv[1:]
    workers = worker_count(PROJECT_ROOT)
    timeout = transfer_timeout()
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    if arguments == ["--baseline-arrays"]:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="baseline-download")
        try:
            destination, downloaded, verified = download_baseline_arrays(
                reader, PROJECT_ROOT, executor, workers, timeout
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        print(f"Baseline decomposition inputs: downloaded={downloaded} already_verified={verified}")
        print("Result directory: " + str(destination))
        return
    run_ids = [safe_run_id(value) for value in arguments]
    if not 1 <= len(run_ids) <= MAX_RUNS or len(set(run_ids)) != len(run_ids):
        raise ValueError(f"Pass 1-{MAX_RUNS} distinct signal child run IDs or --baseline-arrays")
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prediction-download")
    try:
        for run_id in run_ids:
            result = select_export(reader, run_id)
            LOGGER.info(
                "Selected completed export: %s (%s)", result["relative"], result["order"][0]
            )
            destination = download_export(reader, result, PROJECT_ROOT, executor, workers, timeout)
            splits = sorted(result["splits"])
            print(f"Downloaded signal predictions: run_id={run_id} splits={splits}")
            print("Result directory: " + str(destination))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError, MemoryError, subprocess.SubprocessError) as error:
        LOGGER.error("Signal prediction download failed: %s", error)
        sys.exit(1)
