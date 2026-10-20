#!/usr/bin/env python3
"""Select, download and verify completed signal-prediction exports without model dependencies."""

# Keep the local control plane compatible with pre-3.11 system Python.
# ruff: noqa: UP017

import hashlib
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
# Local Kronos inference reads a copy of the shared evaluation bar store (protocol 13.7).
EVALUATION_LOCAL_ROOT = Path("artifacts/diagnostics/evaluation-data")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# Baseline records (about 4.5 MB) and bar-store manifests exceed the probe JSON limit.
LARGE_JSON_BYTES = 64 * 1024**2
SHARD_PATTERN = re.compile(r"shards/[A-Za-z0-9_-]{1,64}/[A-Za-z0-9_.-]{1,128}\.parquet")
# Forward-pipeline exports live apart so validation/test consumers never select them.
FORWARD_REMOTE_ROOT = "diagnostics/forward-predictions/"
FORWARD_LOCAL_ROOT = Path("artifacts/diagnostics/forward-predictions")
FORWARD_POPULATIONS = ("overlap", "forward")


def export_layout(forward_end=None):
    """Remote prefix, local directory and allowed splits of one export family."""
    if forward_end is None:
        return {"remote": REMOTE_ROOT, "local": LOCAL_ROOT, "splits": SPLITS, "forward_end": None}
    try:
        valid = datetime.strptime(forward_end, "%Y-%m-%d").strftime("%Y-%m-%d") == forward_end
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("--forward needs the package end date as YYYY-MM-DD")
    return {
        "remote": f"{FORWARD_REMOTE_ROOT}{forward_end}/",
        "local": FORWARD_LOCAL_ROOT / forward_end,
        "splits": FORWARD_POPULATIONS,
        "forward_end": forward_end,
    }


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


def safe_selector(value):
    """Split RUN_ID or RUN_ID/checkpoint-NNNNNN, the selectors of the export job."""
    run_id, separator, checkpoint = str(value).partition("/")
    safe_run_id(run_id)
    if separator and re.fullmatch(r"checkpoint-[0-9]{6,}", checkpoint) is None:
        raise ValueError("Selector must be RUN_ID or RUN_ID/checkpoint-NNNNNN")
    return run_id, checkpoint or None


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


def candidate(reader, key, layout=None):
    layout = layout or export_layout()
    root = layout["remote"]
    parts = key[len(root):].split("/") if key.startswith(root) else []
    if (
        len(parts) != 4
        or re.fullmatch(r"checkpoint-[0-9]{6,}", parts[1]) is None
        or re.fullmatch(r"export-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", parts[2]) is None
        or parts[3] != "status.json"
    ):
        return None
    run_id = safe_run_id(parts[0])
    expected_checkpoint = f"/runpod-volume/savedModel/{run_id}/{parts[1]}"
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
        or set(splits) - set(layout["splits"])
        or (report.get("forward_package") or {}).get("end_exclusive") != layout["forward_end"]
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
        "order": (completed.astimezone(timezone.utc), parts[1], parts[2]),
        "run_id": run_id,
        "relative": Path(*parts[:3]),
        "prefix": prefix,
        "splits": {split: summary["predictions_sha256"] for split, summary in splits.items()},
        "hashes": {"export.json": report_hash, "status.json": status_hash},
    }


def select_export(reader, selector, layout=None):
    """Pick the newest completed export of one checkpoint; a bare RUN_ID must be unambiguous."""
    layout = layout or export_layout()
    run_id, checkpoint = safe_selector(selector)
    prefix = layout["remote"] + run_id + "/" + (checkpoint + "/" if checkpoint else "")
    newest, checkpoints = None, set()
    for key in reader.status_keys(prefix):
        result = candidate(reader, key, layout)
        if result is None:
            continue
        checkpoints.add(result["relative"].parts[1])
        if newest is None or result["order"] > newest["order"]:
            newest = result
    if newest is None:
        raise FileNotFoundError("No completed signal-prediction export found for " + selector)
    if len(checkpoints) > 1:
        raise ValueError(
            f"{run_id} has completed exports of {', '.join(sorted(checkpoints))}; "
            "pass RUN_ID/checkpoint-NNNNNN"
        )
    return newest


def download_export(reader, result, project_root, executor, workers, timeout, layout=None):
    destination = project_root / (layout or export_layout())["local"] / result["relative"]
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
        payload, _digest = reader.read_json(
            f"baselines/{baseline}/complete.json", LARGE_JSON_BYTES
        )
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


def canonical_json_sha256(payload):
    """Same canonical form as stock_forecasting.data.manifest.canonical_json_sha256."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_large_json(reader, key, label):
    """Read one bounded JSON record, naming the record when the read fails."""
    try:
        return reader.read_json(key, LARGE_JSON_BYTES)
    except (RuntimeError, ValueError) as error:
        raise type(error)(f"{label} ({key}): {error}") from error


def evaluation_identity(reader):
    """The shared validation/test source and its dataset request, from the alpha baseline."""
    payload, _ = read_large_json(
        reader, f"baselines/{ANALYSIS['ALPHA_ID']}/complete.json", "Baseline record"
    )
    sources = (payload.get("data_identity") or {}).get("split_sources") or {}
    validation, test = sources.get("validation") or {}, sources.get("test") or {}
    identity = (validation.get("manifest_sha256"), validation.get("sample_universe"))
    if (
        identity != (test.get("manifest_sha256"), test.get("sample_universe"))
        or any(not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
               for value in identity)
    ):
        raise ValueError("The baseline does not record one shared validation/test source")
    request = ((payload.get("identity") or {}).get("contract") or {}).get("evaluation_request")
    if not isinstance(request, dict):
        raise ValueError("The baseline does not record its evaluation dataset request")
    return identity[0], identity[1], request


def evaluation_dataset_id(request):
    """The dataset directory of a request, as computed by the data workflow (no listing)."""
    identity = runpy.run_path(
        str(PROJECT_ROOT / "src/stock_forecasting/dataset_identity.py")
    )
    return canonical_json_sha256(identity["dataset_request_identity_payload"](request))


def evaluation_plan(reader):
    """Locate the shared evaluation dataset and list every file to fetch with its digest."""
    manifest_sha256, universe, request = evaluation_identity(reader)
    dataset_id = evaluation_dataset_id(request)
    key = f"datasets/{dataset_id}/prepared/bar-store/bar-store.json"
    manifest, digest = read_large_json(reader, key, "Evaluation bar-store manifest")
    if digest != manifest_sha256:
        raise ValueError(f"{key} differs from the manifest recorded by the baseline")
    universe_key = f"datasets/{dataset_id}/prepared/sample-universes/{universe}/universe.json"
    universe_payload, _ = read_large_json(reader, universe_key, "Sample index record")
    if (
        universe_payload.get("state") != "ready"
        or canonical_json_sha256(universe_payload.get("identity")) != universe
        or (universe_payload.get("identity") or {}).get("source_manifest_sha256")
        != manifest_sha256
        or SHA256_PATTERN.fullmatch(str(universe_payload.get("ranges_sha256"))) is None
    ):
        raise ValueError("The remote sample index does not match the baseline identity")
    store = f"datasets/{dataset_id}/prepared/bar-store/"
    files = [(store + "bar-store.json", None, manifest_sha256),
             (store + "_SUCCESS.json", None, None),
             (universe_key, None, None),
             (universe_key[: -len("universe.json")] + "cutoff-ranges.parquet", None,
              universe_payload["ranges_sha256"])]
    for name in ("symbol_index", "cutoff_ranges"):
        entry = manifest.get(name) or {}
        files.append((store + str(entry.get("relative_path")), entry.get("size_bytes"),
                      entry.get("sha256")))
    for shard in manifest.get("shards") or []:
        relative = shard.get("relative_path") if isinstance(shard, dict) else None
        if not isinstance(relative, str) or SHARD_PATTERN.fullmatch(relative) is None:
            raise ValueError("The evaluation bar store lists an unsafe shard path")
        files.append((store + relative, shard.get("size_bytes"), shard.get("sha256")))
    for key, size, digest in files:
        if (digest is not None and SHA256_PATTERN.fullmatch(str(digest)) is None) or (
            size is not None and (not isinstance(size, int) or size < 1)
        ):
            raise ValueError("Invalid evaluation artifact metadata: " + key)
    if len({key for key, _, _ in files}) != len(files):
        raise ValueError("The evaluation file list repeats a path")
    return {"dataset_id": dataset_id, "manifest_sha256": manifest_sha256,
            "sample_universe": universe, "files": files}


def download_evaluation_data(reader, project_root, executor, workers, timeout, plan,
                             fetch=None):
    """Fetch the planned files, verifying each digest; files already verified are kept."""
    fetch = download_file if fetch is None else fetch
    destination = project_root / EVALUATION_LOCAL_ROOT
    safe_local_path(destination)

    def transfer(item):
        key, size, digest = item
        target = destination / key[len("datasets/"):]
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("Unsafe existing evaluation artifact: " + str(target))
        if digest is not None and target.is_file() and sha256_file(target) == digest:
            return "verified"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
            staged = Path(temporary) / target.name
            fetch(reader, key, staged, timeout)
            if size is not None and staged.stat().st_size != size:
                raise ValueError("Downloaded evaluation artifact size mismatch: " + key)
            if digest is not None and sha256_file(staged) != digest:
                raise ValueError("Downloaded evaluation artifact SHA-256 mismatch: " + key)
            staged.replace(target)
        return "downloaded"

    states = list(parallel_results(executor, transfer, plan["files"], workers))
    record = {key: plan[key] for key in ("dataset_id", "manifest_sha256", "sample_universe")}
    record["files"] = len(plan["files"])
    (destination / plan["dataset_id"] / "evaluation-data.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination / plan["dataset_id"], states.count("downloaded"), states.count("verified")


def main():
    arguments = sys.argv[1:]
    workers = worker_count(PROJECT_ROOT)
    timeout = transfer_timeout()
    reader = S3Reader(PROJECT_ROOT, os.environ.get("RUNPOD_NETWORK_VOLUME_ID"))
    if arguments in (["--evaluation-data"], ["--evaluation-data", "--list"]):
        plan = evaluation_plan(reader)
        known = sum(size for _, size, _ in plan["files"] if size is not None)
        print(f"Evaluation dataset: {plan['dataset_id']}")
        print(f"Sample index: {plan['sample_universe']}")
        print(f"Files: {len(plan['files'])}, listed size: {known / 1024**3:.2f} GiB "
              "(plus small JSON records)")
        if arguments[-1] == "--list":
            return
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evaluation-data")
        try:
            destination, downloaded, verified = download_evaluation_data(
                reader, PROJECT_ROOT, executor, workers, timeout, plan
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        print(f"Evaluation data: downloaded={downloaded} already_verified={verified}")
        print("Result directory: " + str(destination))
        return
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
    forward_end = None
    if arguments[:1] == ["--forward"]:
        if len(arguments) < 3:
            raise ValueError("--forward needs the package end date and 1-16 selectors")
        forward_end, arguments = arguments[1], arguments[2:]
    layout = export_layout(forward_end)
    for value in arguments:
        safe_selector(value)
    if not 1 <= len(arguments) <= MAX_RUNS or len(set(arguments)) != len(arguments):
        raise ValueError(f"Pass 1-{MAX_RUNS} distinct export selectors or --baseline-arrays")
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prediction-download")
    try:
        for selector in arguments:
            result = select_export(reader, selector, layout)
            LOGGER.info(
                "Selected completed export: %s (%s)", result["relative"], result["order"][0]
            )
            destination = download_export(
                reader, result, PROJECT_ROOT, executor, workers, timeout, layout
            )
            splits = sorted(result["splits"])
            print(f"Downloaded signal predictions: selector={selector} splits={splits}")
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
