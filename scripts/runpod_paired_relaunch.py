#!/usr/bin/env python3
"""Dispatch paired resume/evaluation while preserving legacy single-run commands."""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_children(root, reader, plan, action, *, force=False, execute=subprocess.run):
    status = runpy.run_path(str(root / "scripts/runpod_run_status.py"))
    summary = status["inspect_pair"](reader, plan["run_id"], plan)
    if action == "resume" and summary["state"] == "complete":
        raise ValueError("Both models already have final test results; use validate --force to recompute")
    if action == "validate" and summary["state"] == "complete" and not force:
        print("Both model holdouts are complete; skipped without creating a GPU Pod. Use --force to recompute.")
        return False
    control = runpy.run_path(str(root / "scripts/runpod_paired_control.py"))
    environments = dict(control["child_environments"](root, plan))

    def check(item):
        task, child = item
        result = summary["children"][task]
        trained = result.get("training", {}).get("state") == "complete"
        complete = result.get("state") == "complete" and result.get("evaluation", {}).get("split") == "test"
        if complete and not force:
            return
        if action == "validate" and not trained:
            raise ValueError(f"Paired validation requires completed {task} training")
        manifest = reader.read(f"savedModel/{child['run_id']}/run-manifest.json", optional=True)
        if not manifest:
            if trained or action == "validate":
                raise ValueError(f"Missing persisted model manifest for {task}")
            return
        env = environments[task]
        env["RUNPOD_NETWORK_VOLUME_ID"] = reader.bucket
        # HEAD existing artifacts locally before any paid resume/evaluation Pod.
        execute([
            sys.executable, str(root / "scripts/runpod_remote_checkpoint_preflight.py"),
            "--s3-wrapper", str(root / "scripts/runpod_s3_project.sh"), "--bucket", reader.bucket,
            "--run-id", child["run_id"], "--config", str(root / child["config_path"]),
            "--selection-policy", "best" if trained else "latest",
        ], env=env, cwd=root, check=True, timeout=1800)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, plan["children"].items()))
    return True


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resume", "validate"))
    parser.add_argument("--maxRuntime", "--max-runtime", default="12h")
    parser.add_argument("--gpuId", "--gpu-id", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", dest="evaluation_resume", action="store_const", const="1", default=None)
    parser.add_argument("--no-resume", dest="evaluation_resume", action="store_const", const="0")
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args(arguments)
    if args.action == "resume" and (args.force or args.evaluation_resume is not None):
        parser.error("Evaluation flags are only supported by validate")
    legacy_script = "create_runpod_resume_pod.sh" if args.action == "resume" else "create_runpod_validation_pod.sh"
    if os.environ.get("RUNPOD_TEST_MODE") == "1" and not os.environ.get("RUNPOD_PAIRED_MANIFEST"):
        # Preserve the existing offline payload-only mode, which must do no S3 I/O.
        os.execvp("bash", ["bash", str(ROOT / "scripts" / legacy_script), *arguments[1:]])
    # Reuse the existing CLI bounds; never silently alter duration/GPU semantics.
    seconds = subprocess.check_output([
        "bash", "-c", 'source "$1"; runpod_validate_gpu_id "$2"; runpod_duration_seconds "$3" --maxRuntime',
        "paired-relaunch", str(ROOT / "scripts/lib/runpod_cli.sh"), args.gpuId, args.maxRuntime,
    ], text=True).strip()
    control = runpy.run_path(str(ROOT / "scripts/runpod_paired_control.py"))
    records = runpy.run_path(str(ROOT / "scripts/runpod_runs.py"))
    reader = records["RunRecords"](ROOT, bucket=control["_bucket"](ROOT))
    run_id = args.run_id or reader.choose("resume" if args.action == "resume" else "completed")
    records["validate_run"](run_id)
    plan = reader.pair(run_id)
    if plan is None:
        os.execvp("bash", ["bash", str(ROOT / "scripts" / legacy_script), *arguments[1:]])
    if plan["workflow"] != "train":
        raise ValueError("A baseline pair must be continued through the baseline command")
    if not check_children(ROOT, reader, plan, args.action, force=args.force):
        return 0
    os.environ.update(
        RUNPOD_CLI_MAX_RUNTIME_SECONDS=seconds,
        RUNPOD_CLI_HARD_LIMIT_SECONDS=str(int(seconds) + 3600),
        RUNPOD_CLI_GPU_ID=args.gpuId,
        VALIDATION_RESUME="0" if args.force else (args.evaluation_resume or os.environ.get("VALIDATION_RESUME", "1")),
        VALIDATION_FORCE_RECOMPUTE="1" if args.force else os.environ.get("VALIDATION_FORCE_RECOMPUTE", "0"),
    )
    return control["launch_existing"](ROOT, plan, args.action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Paired relaunch failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
