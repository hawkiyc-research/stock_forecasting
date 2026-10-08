#!/usr/bin/env python3
"""Bounded, selection-pinned multi-Pod launches on the local control host."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LAUNCH_RUNS = 30


@contextlib.contextmanager
def control_lock(root: Path, *, exclusive: bool):
    """Serialize source mutation against all launches from this control checkout."""
    directory = root / ".runpod"
    directory.mkdir(exist_ok=True)
    with (directory / "launch-sync.lock").open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Source sync and Pod launch cannot overlap; retry after it finishes"
            ) from error
        yield


def safe_run_id(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value) is None or "--" in value:
        raise ValueError("A canonical run ID is required")
    return value


def check_admission(pods: list, volume: str, mode: str, run_id: str) -> None:
    """Only distinct, run-scoped train/validation consumers may share a volume."""
    if not isinstance(pods, list):
        raise ValueError("Cannot establish active Pod inventory")
    for pod in pods:
        if not isinstance(pod, dict):
            raise ValueError("Invalid Pod inventory entry")
        pod_volume = pod.get("networkVolumeId") or (pod.get("networkVolume") or {}).get("id")
        if pod_volume != volume:
            continue
        env = pod.get("env", {})
        shared_reader = (
            isinstance(env, dict)
            and env.get("RUNPOD_SCOPED_LIFECYCLE") == "1"
            and env.get("RUNPOD_ROLE") in {"gpu-train", "gpu-validation"}
            and env.get("WANDB_RUN_ID")
        )
        if (
            mode in {"train", "validation"}
            and shared_reader
            and safe_run_id(str(env["WANDB_RUN_ID"])) != safe_run_id(run_id)
        ):
            continue
        # Stopped Pods still own a mount and can be restarted. Do not assume idle
        # or absent telemetry makes a shared writable runtime safe to replace.
        raise ValueError(
            f"Pod {pod.get('id', '?')} still owns this volume: {mode} is incompatible "
            "with that workflow (or the same run is already allocated)"
        )


def admit(root: Path, mode: str, run_id: str = "") -> None:
    volume = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    if re.fullmatch(r"[A-Za-z0-9_-]+", volume) is None:
        raise ValueError("Network volume ID must be loaded from .env")
    result = subprocess.run(
        ["bash", str(root / "scripts/runpodctl_project.sh"), "pod", "list", "--output", "json"],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    pods = json.loads(result.stdout)
    if not isinstance(pods, list):
        raise ValueError("Cannot establish active Pod inventory")

    def details(pod):
        # Never infer no volume/no owner from a list endpoint's omitted fields.
        if not isinstance(pod, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(pod.get("id", ""))):
            raise ValueError("Invalid Pod inventory entry")
        response = subprocess.run(
            [
                "bash",
                str(root / "scripts/runpodctl_project.sh"),
                "pod",
                "get",
                pod["id"],
                "--include-network-volume",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        found = json.loads(response.stdout)
        if not isinstance(found, dict) or found.get("id") != pod["id"]:
            raise ValueError("Pod details do not match the requested identity")
        return found

    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(pods), 16):
            check_admission(list(pool.map(details, pods[start : start + 16])), volume, mode, run_id)


def launch(root: Path, experiments: list[str], workers: int, seeds: list[int] | None = None) -> int:
    selection = runpy.run_path(str(root / "scripts/runpod_selection.py"))
    _, original = selection["_resolve_selection_path"](
        root,
        os.environ.get("RUNPOD_LAUNCH_SELECTION_FILE"),
        validate_local_config=not bool(experiments),
    )
    if len(experiments) != len(set(experiments)):
        raise ValueError("Each experiment may be selected only once per launch")
    if len(experiments) > len(selection["EXPERIMENT_CONFIGS"]):
        raise ValueError("Launch exceeds the number of distinct configured experiments")
    seeds = [] if seeds is None else list(seeds)
    for seed in seeds:
        selection["_validate_training_seed"](seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Each training seed may be selected only once per launch")
    count = max(1, len(experiments)) * max(1, len(seeds))
    if count > MAX_LAUNCH_RUNS:
        raise ValueError(f"At most {MAX_LAUNCH_RUNS} experiment/seed runs can be launched together")
    jobs = []
    for name in experiments or [""]:
        base = selection["_with_experiment"](original, name, root) if name else original
        for seed in seeds or [None]:
            payload = (
                selection["_with_training_seed"](base, seed, root) if seed is not None else base
            )
            selected = selection["_store_selection"](root, payload)
            env = os.environ.copy()
            env["RUNPOD_LAUNCH_SELECTION_FILE"] = str(selected)
            env["RUNPOD_GPU_WORKFLOW"] = "train"
            env["RUNPOD_LAUNCH_CONTROL_LOCK_HELD"] = "1"
            label = name or "configured"
            if seed is not None:
                label += f"-seed{seed}"
                env["RUNPOD_POD_NAME"] = "stock-forecasting-" + label
            elif name:
                env["RUNPOD_POD_NAME"] = "stock-forecasting-" + name
            jobs.append((label, env))
    print(
        f"Training launch plan: {len(jobs)} independent run(s): "
        + ", ".join(name for name, _env in jobs),
        flush=True,
    )

    def command(job, script, *, preflight=False):
        name, env = job
        env = dict(env)
        if preflight:
            env["RUNPOD_PREFLIGHT_ONLY"] = "1"
        try:
            completed = subprocess.run(
                ["bash", str(root / "scripts" / script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return (
                name,
                124,
                output + "\nControl command timed out; inspect Pod inventory before retrying.",
            )
        except OSError as error:
            return name, 127, str(error)
        return name, completed.returncode, completed.stdout

    # A bounded batch of tiny control jobs is submitted. These I/O workers do
    # not load datasets/models. A conservative default of two limits API traffic.
    with (
        control_lock(root, exclusive=False),
        ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool,
    ):
        for script, preflight in (
            ("publish_runpod_selection.sh", False),
            ("create_runpod_pod.sh", True),
        ):
            if script == "publish_runpod_selection.sh" and not experiments and not seeds:
                continue
            results = list(
                pool.map(
                    lambda job, script=script, preflight=preflight: command(
                        job, script, preflight=preflight
                    ),
                    jobs,
                )
            )
            for name, code, output in results:
                print(
                    f"[{name}] {script}\n{output}",
                    file=sys.stderr if code else sys.stdout,
                    flush=True,
                )
            if any(code for _, code, _ in results):
                print("Batch preflight failed; no training Pod was created", file=sys.stderr)
                return 2
        results = list(pool.map(lambda job: command(job, "create_runpod_pod.sh"), jobs))
        for name, code, output in results:
            print(f"[{name}] {'created' if code == 0 else 'launch failed'}\n{output}", flush=True)
        if any(code for _, code, _ in results):
            print(
                "Partial launch: successful Pods keep their own guards. "
                "No automatic retries or deletions.",
                file=sys.stderr,
            )
            return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("launch")
    train.add_argument("--experiment", action="append", default=[])
    train.add_argument("--seed", action="append", type=int, default=[])
    train.add_argument("--launchWorkers", type=int, default=2)
    admission = sub.add_parser("admit")
    admission.add_argument(
        "--mode", choices=("train", "validation", "exclusive", "sync"), required=True
    )
    admission.add_argument("--run-id", default="")
    sub.add_parser("sync")
    create = sub.add_parser("create")
    create.add_argument("--kind", choices=("gpu", "cpu"), required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "admit":
        admit(root, args.mode, args.run_id)
        return 0
    if args.command == "sync":
        with control_lock(root, exclusive=True):
            admit(root, "sync")
            return subprocess.call(
                ["bash", str(root / "scripts/sync_project_to_runpod_volume.sh"), "--apply"],
                env={**os.environ, "RUNPOD_SYNC_CONTROL_LOCK_HELD": "1"},
            )
    if args.command == "create":
        exclusive = args.kind == "cpu" or os.environ.get("RUNPOD_GPU_WORKFLOW") == "baseline"
        script = "create_runpod_cpu_pod.sh" if args.kind == "cpu" else "create_runpod_pod.sh"
        with control_lock(root, exclusive=exclusive):
            return subprocess.call(
                ["bash", str(root / "scripts" / script)],
                env={**os.environ, "RUNPOD_LAUNCH_CONTROL_LOCK_HELD": "1"},
            )
    if not 1 <= args.launchWorkers <= 6:
        raise ValueError("--launchWorkers must be between 1 and 6")
    return launch(root, args.experiment, args.launchWorkers, args.seed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Parallel RunPod control failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
