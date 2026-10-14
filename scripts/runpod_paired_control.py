#!/usr/bin/env python3
"""Local-only, bounded control for two independent models in one paid Pod."""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRED_EXPERIMENTS = ("a-signal64", "b-signal64")


def read_plan(root: Path, path: Path) -> dict:
    contract = runpy.run_path(str(root / "scripts/runpod_paired.py"))
    return contract["load_manifest"](path)


def validate_launch(plan: dict, env: dict) -> None:
    """Bind the paid Pod envelope to its validated plan and primary child."""
    if env.get("WANDB_RUN_ID") != plan["run_id"] or env.get("RUNPOD_GPU_WORKFLOW", "train") != plan["workflow"]:
        raise ValueError("Paired plan does not match the launch identity/workflow")
    mount = env.get("RUNPOD_VOLUME_MOUNT_PATH") or "/runpod-volume"
    expected = f"{mount}/lifecycle/runs/{plan['run_id']}/pair.json"
    if env.get("RUNPOD_PAIRED_PLAN_PATH") != expected:
        raise ValueError("Paired plan path must identify its parent run on the mounted volume")
    matching = [child for child in plan["children"].values()
                if child["selection"]["selection_sha256"] == env.get("RUNPOD_SELECTION_SHA256")]
    if len(matching) != 1:
        raise ValueError("The primary Pod selection is not owned by the paired plan")
    selected = matching[0]["selection"]
    if (env.get("RUNPOD_CONFIG") != selected["stage"]["config_path"]
            or env.get("RUNPOD_STAGE_CONFIG_SHA256") != selected["stage"]["config_sha256"]
            or env.get("RUNPOD_DATASET_REQUEST_SHA256") != selected["dataset_request_sha256"]):
        raise ValueError("The primary Pod configuration/data differ from the paired plan")
    action = env.get("RUNPOD_PAIRED_ACTION")
    if action and (plan["workflow"] != "train" or action not in {"resume", "validate"}):
        raise ValueError("Invalid paired launch action")


def child_environments(root: Path, plan: dict):
    selection = runpy.run_path(str(root / "scripts/runpod_selection.py"))
    jobs = []
    for task, child in plan["children"].items():
        payload = selection["_validate_selection"](child["selection"], project_root=root)
        path = selection["_store_selection"](root, payload)
        env = {**os.environ, **selection["_selection_exports"](path, payload)}
        env["RUNPOD_LAUNCH_SELECTION_FILE"] = str(path)
        jobs.append((task, env))
    return jobs


def _run(root: Path, command: list[str], env: dict, *, quiet=False):
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or f"Control command failed: {command[0]}")
    if not quiet and result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout


def _bucket(root: Path) -> str:
    value = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    if not value:
        value = _run(
            root,
            ["bash", "-c", 'source "$1"; runpod_load_s3_env "$2"; printf "%s" "$RUNPOD_NETWORK_VOLUME_ID"',
             "paired-control", str(root / "scripts/lib/runpod_project_env.sh"), str(root)],
            os.environ.copy(), quiet=True,
        ).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("A valid configured network volume is required")
    return value


def publish(root: Path, path: Path) -> None:
    plan = read_plan(root, path)
    jobs = child_environments(root, plan)
    bucket = _bucket(root)
    # Each job contains metadata only; two requests bound external API pressure.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: _run(root, ["bash", str(root / "scripts/runpod_s3_project.sh"),
            "s3", "cp", job[1]["RUNPOD_SELECTION_FILE"],
            f"s3://{bucket}/{job[1]['RUNPOD_REMOTE_SELECTION_RELATIVE_PATH']}",
            "--only-show-errors"], job[1]), jobs))
    _run(root, ["bash", str(root / "scripts/runpod_s3_project.sh"), "s3", "cp", str(path),
                f"s3://{bucket}/lifecycle/runs/{plan['run_id']}/pair.json", "--only-show-errors"], os.environ.copy())


def baseline_status(root: Path, plan: dict, *, require=False) -> dict:
    jobs = child_environments(root, plan)
    bucket = _bucket(root)

    def check(job):
        task, env = job
        env["RUNPOD_NETWORK_VOLUME_ID"] = bucket
        raw = _run(root, [sys.executable, str(root / "scripts/runpod_baseline_cache.py"),
                          "require" if require else "check", "--project-root", str(root)], env, quiet=True)
        result = json.loads(raw)
        if not isinstance(result, dict) or not isinstance(result.get("complete"), bool):
            raise ValueError(f"Invalid baseline completion result for {task}")
        return task, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(check, jobs))
    complete = all(item["complete"] for item in results.values())
    if require and not complete:
        raise ValueError("Both task baselines must be complete before paired training")
    return {"complete": complete, "tasks": results}


def preflight(root: Path, path: Path) -> dict:
    plan = read_plan(root, path)
    result = baseline_status(root, plan, require=plan["workflow"] == "train")
    mode = "--gpu" if plan["workflow"] == "train" else "--baseline"
    jobs = child_environments(root, plan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: _run(root, ["bash", str(root / "scripts/verify_runpod_stage_readiness.sh"), mode, "--local-selection"], job[1]), jobs))
    print("Paired preflight passed: both task baselines and current data rules checked.", flush=True)
    return result


def prepare_launch(root: Path, original: dict, experiment: str, seed=None, *, workflow="train"):
    contract = runpy.run_path(str(root / "scripts/runpod_paired.py"))
    plan = contract["build_plan"](root, original, experiment, seed=seed, workflow=workflow)
    path = contract["store_plan"](root, plan)
    return plan, path, launch_environment(root, plan, path)


def launch_environment(root: Path, plan: dict, path: Path) -> dict:
    first = child_environments(root, plan)[0][1]
    env = os.environ.copy()
    # A fresh paired launch must not inherit an action from a previous shell.
    env.pop("RUNPOD_PAIRED_ACTION", None)
    env.update(
        RUNPOD_LAUNCH_SELECTION_FILE=first["RUNPOD_LAUNCH_SELECTION_FILE"],
        RUNPOD_PAIRED_MANIFEST=str(path),
        RUNPOD_PAIRED_PLAN_PATH=f"{os.environ.get('RUNPOD_VOLUME_MOUNT_PATH', '/runpod-volume')}/lifecycle/runs/{plan['run_id']}/pair.json",
        RUNPOD_GPU_WORKFLOW=plan["workflow"],
        WANDB_RUN_ID=plan["run_id"],
        RUNPOD_POD_NAME=f"stock-forecasting-{plan['experiment']}-seed{plan['seed']}" if plan["workflow"] == "train" else f"stock-forecasting-{plan['experiment']}-baseline",
    )
    return env


def launch_existing(root: Path, plan: dict, action: str) -> int:
    if action not in {"resume", "validate"} or plan["workflow"] != "train":
        raise ValueError("Only paired training runs can be resumed or validated")
    contract = runpy.run_path(str(root / "scripts/runpod_paired.py"))
    path = contract["store_plan"](root, plan)
    env = launch_environment(root, plan, path)
    env["RUNPOD_PAIRED_ACTION"] = action
    for key in ("RESUME_CHECKPOINT", "VALIDATION_RUN_ID", "VALIDATION_CHECKPOINT"):
        env.pop(key, None)
    control = runpy.run_path(str(root / "scripts/runpod_concurrency.py"))
    with control["control_lock"](root, exclusive=False):
        env["RUNPOD_LAUNCH_CONTROL_LOCK_HELD"] = "1"
        return subprocess.call(["bash", str(root / "scripts/create_runpod_pod.sh")], env=env, cwd=root)


def launch_baseline(root: Path, experiment: str) -> int:
    selection = runpy.run_path(str(root / "scripts/runpod_selection.py"))
    _, original = selection["_resolve_selection_path"](root, None, validate_local_config=False)
    _plan, path, env = prepare_launch(root, original, experiment, workflow="baseline")
    # Keep the source-sync lock across admission and creation, as for training.
    control = runpy.run_path(str(root / "scripts/runpod_concurrency.py"))
    with control["control_lock"](root, exclusive=True):
        env["RUNPOD_LAUNCH_CONTROL_LOCK_HELD"] = "1"
        return subprocess.call(["bash", str(root / "scripts/create_runpod_pod.sh")], env=env, cwd=root)


def status(root: Path, experiment: str) -> int:
    selection = runpy.run_path(str(root / "scripts/runpod_selection.py"))
    _, original = selection["_resolve_selection_path"](root, None, validate_local_config=False)
    contract = runpy.run_path(str(root / "scripts/runpod_paired.py"))
    plan = contract["build_plan"](root, original, experiment, workflow="baseline")
    jobs = child_environments(root, plan)
    # Preserve the current-rule data checks AND result completion, for both tasks.
    def inspect(job):
        task, env = job
        result = subprocess.run(
            [sys.executable, str(root / "scripts/runpod_baseline_readiness.py"),
             "--status", "--project-root", str(root), "--selection", env["RUNPOD_SELECTION_FILE"]],
            cwd=root, env=env, text=True, capture_output=True, timeout=1800,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr or result.stdout or f"Baseline status failed for {task}")
        return task, result.returncode, result.stdout
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(inspect, jobs))
        for task, _code, output in results:
            # The child status describes valid artifacts but its generic next
            # command would select only one task. Emit one paired next action.
            output = "\n".join(line for line in output.splitlines()
                               if not line.startswith("Next: bash scripts/runpod_workflow.sh baseline")) + "\n"
            print(f"[{experiment} / {task}]\n{output}", end="", flush=True)
    incomplete = any(code for _task, code, _output in results)
    if incomplete:
        print(f"Next: bash scripts/runpod_workflow.sh baseline --experiment {experiment}")
    else:
        print(f"Paired baseline {experiment}: BOTH task baselines COMPLETE and reusable.")
    return int(incomplete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "preflight", "baseline-status", "validate"):
        child = sub.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
    for name in ("launch-baseline", "status"):
        launch = sub.add_parser(name)
        launch.add_argument("--experiment", choices=PAIRED_EXPERIMENTS, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_TEST_MODE") != "1":
        raise ValueError("Paired admission must run locally before renting a Pod")
    if args.command == "launch-baseline":
        return launch_baseline(root, args.experiment)
    if args.command == "status":
        return status(root, args.experiment)
    plan = read_plan(root, args.manifest)
    if args.command == "publish":
        publish(root, args.manifest)
    elif args.command == "preflight":
        preflight(root, args.manifest)
    elif args.command == "baseline-status":
        print(json.dumps(baseline_status(root, plan)))
    else:
        validate_launch(plan, os.environ)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Paired RunPod control failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
