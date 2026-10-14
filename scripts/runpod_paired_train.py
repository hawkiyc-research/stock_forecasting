#!/usr/bin/env python3
"""Run one paired experiment in one Pod without sharing model gradients or state."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

TASKS = ("alpha_point", "absolute_direction")


def load_json(path: Path, *, optional=False):
    if optional and not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Missing, unsafe, or oversized paired-workflow record: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(path: Path, root: Path):
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve(strict=False)):
        raise ValueError(f"Paired-workflow path must stay inside {root}: {path}")
    return resolved


def child_environment(child, plan, environment, volume: Path, project: Path):
    """Derive every task override from its own immutable selection, never sibling state."""
    selection_tools = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    selection = selection_tools["_validate_selection"](
        child["selection"], project_root=project
    )
    selection_path = inside(volume / child["selection_path"], volume)
    if load_json(selection_path) != selection:
        raise ValueError("Mounted child selection differs from the paired experiment")
    exports = selection_tools["_selection_exports"](selection_path, selection)
    result = dict(environment)
    # Resume and evaluation pointers must never cross from one task to its sibling.
    for key in ("RESUME_CHECKPOINT", "VALIDATION_CHECKPOINT", "RUNPOD_SOURCE_CONFIG_SHA256"):
        result.pop(key, None)
    for key, value in exports.items():
        if key in ("MAX_RUNTIME_SECONDS", "RUNPOD_HARD_LIMIT_SECONDS"):
            continue
        if value.startswith("/runpod-volume/"):
            value = str(volume / value.removeprefix("/runpod-volume/"))
        result[key] = value
    run_id = child["run_id"]
    launch = environment["RUNPOD_LAUNCH_ID"]
    log_directory = volume / "logs" / run_id / "launcher" / launch
    result.update(
        RUNPOD_CONFIG=child["config_path"],
        RUNPOD_REMOTE_SELECTION_PATH=str(selection_path),
        RUNPOD_SELECTION_FILE=str(selection_path),
        RUNPOD_RUN_KEY=run_id,
        WANDB_RUN_ID=run_id,
        VALIDATION_RUN_ID=run_id,
        WANDB_RUN_NAME=f"{child['experiment']}-seed{plan['seed']}",
        RUNPOD_PARENT_RUN_ID=plan["run_id"],
        RUNPOD_SCOPED_LIFECYCLE="1",
        RUNPOD_SHUTDOWN_DIR=str(log_directory),
        RUNPOD_SHUTDOWN_MARKER=str(log_directory / "shutdown.json"),
        RUNPOD_TRAINING_COMPLETED_MARKER=str(log_directory / "training-completed.json"),
        RUNPOD_TRAINING_LIFECYCLE_MARKER=str(volume / "lifecycle/runs" / run_id / "training.json"),
        RUNPOD_VALIDATION_LIFECYCLE_MARKER=str(volume / "lifecycle/runs" / run_id / "validation.json"),
    )
    return result


def completed_child(child, volume: Path, project: Path, *, successful_validation=False):
    """Require durable training and a matching final test report before skipping work."""
    run_id = child["run_id"]
    completion_path = volume / "lifecycle/runs" / run_id / "training-completed.json"
    completion = load_json(completion_path, optional=True)
    if completion is None:
        return False, False
    manifest_path = volume / "savedModel" / run_id / "run-manifest.json"
    manifest = load_json(manifest_path)
    readiness = runpy.run_path(str(project / "scripts/runpod_readiness.py"))
    readiness["_validate_training_completion_payload"](completion, volume, run_id)
    if completion["run_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Child training completion does not match its persisted manifest")
    if (
        manifest.get("run_id") != run_id
        or (manifest.get("training_resume_contract", {}).get("model") or {}).get("prediction_task") != child["task"]
        or (manifest.get("selection_provenance") or {}).get("selection_id")
        != child["selection"]["selection_id"]
    ):
        raise ValueError("Child run manifest does not match its task and selection")
    report = load_json(
        volume / "evaluations" / run_id / "validation-benchmark.json", optional=True
    )
    if report is None:
        return True, False
    if report.get("run_id") != run_id:
        raise ValueError("Child evaluation belongs to a different run")
    if report.get("prediction_task") != child["task"]:
        raise ValueError("Child evaluation belongs to a different prediction task")
    # Validation-only scores are not a final holdout, even if their state is ready.
    complete = (
        report.get("state") == "ready"
        and bool(report.get("completed_at"))
        and report.get("evaluation_split") == "test"
        and report.get("prediction_task") == child["task"]
    )
    evaluation = load_json(volume / "lifecycle/runs" / run_id / "validation.json", optional=True)
    if complete and evaluation and evaluation.get("state") != "ready" and evaluation.get("generated_at"):
        stamp = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp(evaluation["generated_at"]) > stamp(report["completed_at"]):
            # An old ready report cannot hide a newer failed/restarted attempt.
            # The just-returned successful validation CLI intentionally leaves
            # its lifecycle finalizing for this owner to publish ready; only
            # that explicit success boundary may normalize a newer finalizing
            # marker without unnecessarily repeating the numerical evaluation.
            complete = successful_validation and evaluation.get("state") == "finalizing"
    return True, complete


def lifecycle(project, volume, environment, phase, state, *, exit_code=None, completed=False):
    command = [
        environment.get("RUNPOD_PYTHON_BIN", sys.executable),
        str(project / "scripts/runpod_readiness.py"), "write-state",
        "--output", str(volume / "lifecycle/runs" / environment["WANDB_RUN_ID"] / f"{phase}.json"),
        "--network-volume-root", str(volume), "--kind", f"stage1-{phase}",
        "--state", state, "--launch-id", environment["RUNPOD_LAUNCH_ID"],
        "--log-path", environment["RUNPOD_SHUTDOWN_DIR"],
        "--wandb-run-id", environment["WANDB_RUN_ID"],
        "--max-runtime-seconds", environment["MAX_RUNTIME_SECONDS"],
        "--inherit-existing",
    ]
    if phase == "training":
        command += ["--training-completed", str(int(completed))]
    if exit_code is not None:
        command += ["--exit-code", str(exit_code)]
    subprocess.run(command, env=environment, cwd=project, check=True)


def resume_checkpoint(child, volume, project, environment):
    directory = volume / "savedModel" / child["run_id"]
    if not (directory / "checkpoint-leaderboard.json").exists():
        if any(directory.glob("checkpoint-*")):
            raise ValueError("Child has checkpoint artifacts but no committed checkpoint leaderboard")
        if (directory / "run-manifest.json").exists():
            raise ValueError("Child run was initialized but has no committed checkpoint to resume")
        return None
    result = subprocess.run(
        [str(project / ".venv/bin/python"), "-c",
         "import sys; from stock_forecasting.checkpointing import latest_resume_checkpoint; "
         "print(latest_resume_checkpoint(sys.argv[1]))", str(directory)],
        env=environment, cwd=project, check=True, capture_output=True, text=True,
    )
    checkpoint = Path(result.stdout.strip())
    if checkpoint.parent != directory or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError("Child resume discovery returned a noncanonical checkpoint")
    return str(checkpoint)


def run_pair(plan_path: Path, *, environment=None, action="train"):
    environment = dict(os.environ if environment is None else environment)
    volume = Path(environment["NETWORK_VOLUME_ROOT"]).resolve()
    project = inside(Path(environment["PROJECT_ROOT"]), volume)
    pair_tools = runpy.run_path(str(project / "scripts/runpod_paired.py"))
    plan = pair_tools["validate_manifest"](
        load_json(plan_path), run_id=environment["WANDB_RUN_ID"]
    )
    if plan["workflow"] != "train" or action not in ("train", "resume", "validate"):
        raise ValueError("Paired model workflow/action is unsupported")
    expected_plan = volume / "lifecycle/runs" / plan["run_id"] / "pair.json"
    if inside(plan_path, volume) != expected_plan:
        raise ValueError("Paired plan must use its canonical parent run path")
    if environment.get("RUNPOD_RUN_KEY") != plan["run_id"]:
        raise ValueError("Paired parent runtime IDs disagree")
    if environment.get("RUNPOD_SCOPED_LIFECYCLE") != "1":
        raise ValueError("Paired training requires run-scoped lifecycle records")
    children = plan["children"]
    child_environments = {
        task: child_environment(children[task], plan, environment, volume, project)
        for task in TASKS
    }
    if action == "validate":
        for task in TASKS:
            trained, _ = completed_child(children[task], volume, project)
            if not trained:
                raise ValueError(f"Paired validation requires completed {task} training")
    # Check both task selections before starting either expensive model. The
    # ordinary entrypoint verified only the first selection used to launch Pod.
    for child_env in child_environments.values():
        subprocess.run(
            ["bash", str(project / "scripts/verify_runpod_mounted_readiness.sh")],
            env=child_env, cwd=project, check=True,
        )
    progress = {
        "schema_version": 1, "kind": "paired-training-progress", "run_id": plan["run_id"],
        "experiment": plan["experiment"], "seed": plan["seed"], "state": "running",
        "pod_id": environment.get("RUNPOD_POD_ID", ""),
        "children": {task: {"run_id": children[task]["run_id"], "state": "pending"} for task in TASKS},
    }
    progress_path = expected_plan.with_name("pair-status.json")

    def publish():
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(progress_path, progress)

    # Reserve both child identities for the parent before either starts. Holding
    # one GPU model at a time avoids doubling measured activation/worker memory;
    # the unchanged training pipeline still uses its resource-tuned DataLoader.
    with ExitStack() as locks:
        for task in TASKS:
            lock_path = volume / "lifecycle/runs" / children[task]["run_id"] / "gpu-workflow.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = locks.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Paired child already has a workflow owner: {task}") from error
        publish()
        for task in TASKS:
            child = children[task]
            child_env = child_environments[task]
            trained = False
            try:
                stop_path = volume / "lifecycle/runs" / plan["run_id"] / "pods" / environment.get("RUNPOD_POD_ID", "") / "stop-request.json"
                stop_request = load_json(stop_path, optional=True)
                if stop_request == {
                    "schema_version": 1, "kind": "runtime-stop-request",
                    "pod_id": environment.get("RUNPOD_POD_ID", ""), "run_id": plan["run_id"],
                }:
                    raise subprocess.CalledProcessError(124, ["paired-section-boundary"])
                trained, complete = completed_child(child, volume, project)
                force = child_env.get("VALIDATION_FORCE_RECOMPUTE") == "1"
                if complete and not force:
                    progress["children"][task]["state"] = "complete"
                    publish()
                    print(f"Paired {task}: completed training and holdout reused ({child['run_id']})", flush=True)
                    continue
                if action == "validate" and not trained:
                    raise ValueError(f"Paired validation requires completed {task} training")
                progress["children"][task]["state"] = "evaluating" if trained else "training"
                publish()
                print(f"Paired experiment {plan['run_id']}: {task} -> {child['run_id']}", flush=True)
                lifecycle(project, volume, child_env, "training", "finalizing" if trained else "preparing", completed=trained)
                if trained:
                    # The validation wrapper resolves resume/force policy from
                    # its environment; an extra CLI flag would override it.
                    command = ["bash", str(project / "scripts/runpod_validation.sh")]
                else:
                    checkpoint = resume_checkpoint(child, volume, project, child_env)
                    if checkpoint:
                        child_env["RESUME_CHECKPOINT"] = checkpoint
                    command = ["bash", str(project / "scripts/runpod_train_then_validate.sh"), child["config_path"]]
                subprocess.run(command, env=child_env, cwd=project, check=True)
                trained, complete = completed_child(
                    child, volume, project, successful_validation=True
                )
                if not trained or not complete:
                    raise ValueError(f"Paired {task} returned without complete training and matching final test results")
                lifecycle(project, volume, child_env, "validation", "ready", exit_code=0)
                lifecycle(project, volume, child_env, "training", "ready", exit_code=0, completed=True)
                progress["children"][task]["state"] = "complete"
                publish()
            except BaseException as error:
                code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
                try:
                    trained, _ = completed_child(child, volume, project)
                except (ValueError, OSError):
                    # Invalid artifacts are never upgraded to successful work.
                    trained = False
                progress["state"] = "timed_out" if code == 124 else "failed"
                progress["children"][task].update(state=progress["state"], exit_code=code)
                publish()
                lifecycle(project, volume, child_env, "training", "ready" if trained else progress["state"], exit_code=code, completed=trained)
                if trained or Path(child_env["RUNPOD_VALIDATION_LIFECYCLE_MARKER"]).exists():
                    lifecycle(project, volume, child_env, "validation", progress["state"], exit_code=code)
                raise
        progress["state"] = "complete"
        publish()
        # Only the outer tmux/entrypoint may publish terminal parent lifecycle.
        # A child completion can therefore never terminate the sibling's Pod.
        marker = inside(Path(environment["RUNPOD_TRAINING_COMPLETED_MARKER"]), volume / "logs" / plan["run_id"])
        atomic_json(marker, {"training_completed": True, "completed_at": progress["updated_at"], "wandb_run_id": plan["run_id"]})
        lifecycle(project, volume, environment, "training", "finalizing", completed=True)
        lifecycle(project, volume, environment, "validation", "finalizing")
    print(f"Paired experiment complete: {plan['run_id']} (both independent models and holdouts)", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--action", choices=("train", "resume", "validate"), default=os.environ.get("RUNPOD_PAIRED_ACTION", "train"))
    args = parser.parse_args()
    return run_pair(args.manifest, action=args.action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"Paired workflow child failed with exit code {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from error
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Paired workflow failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
