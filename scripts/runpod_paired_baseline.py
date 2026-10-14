#!/usr/bin/env python3
"""Build both task-aligned baseline caches under one existing Pod deadline."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runpod_paired import TASKS, load_manifest, selection_api, write_json


def run(project: Path, volume: Path, manifest: Path, *, environ=None, execute=None) -> int:
    env = dict(os.environ if environ is None else environ)
    execute = subprocess.run if execute is None else execute
    plan = load_manifest(manifest, run_id=env.get("WANDB_RUN_ID"))
    if plan["workflow"] != "baseline" or env.get("RUNPOD_ROLE") != "gpu-baseline":
        raise ValueError("Paired baseline requires its dedicated baseline Pod")
    if env.get("RUNPOD_GPU_WORKFLOW_LEASE_HELD") != "1":
        raise ValueError("Paired baseline requires the outer workflow lease")
    if not env.get("RUNPOD_TMUX_LOG_FILE"):
        raise ValueError("Paired baseline must run inside its outer tmux lifecycle")
    expected = volume / "lifecycle/runs" / plan["run_id"] / "pair.json"
    if manifest.resolve() != expected.resolve():
        raise ValueError("Paired baseline manifest is outside its owning run")
    state = {"schema_version": 1, "kind": "paired-baseline-status", "run_id": plan["run_id"],
             "state": "running", "children": {task: {"state": "pending"} for task in TASKS}}
    status_path = expected.with_name("pair-status.json")

    def publish():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)

    publish()
    for task in TASKS:
        try:
            child = plan["children"][task]
            selected_path = volume / child["selection_path"]
            # Mounted immutable copies must match the plan before any model fitting.
            api = selection_api(project)
            _, selected = api["_resolve_selection_path"](project, str(selected_path), validate_local_config=False)
            if selected["selection_sha256"] != child["selection"]["selection_sha256"]:
                raise ValueError("Paired baseline selection does not match the published plan")
            child_env = dict(env)
            exports = api["_selection_exports"](selected_path, selected)
            for key in ("MAX_RUNTIME_SECONDS", "RUNPOD_HARD_LIMIT_SECONDS"):
                exports.pop(key, None)
            child_env.update(exports)
            child_env.update({"RUNPOD_REMOTE_SELECTION_PATH": str(selected_path),
                              "RUNPOD_SELECTION_FILE": str(selected_path),
                              "DATA_ROOT": str(volume / "datasets" / selected["dataset_request_sha256"]),
                              "RUNPOD_PAIRED_CHILD_TASK": task})
            state["children"][task] = {"state": "running"}
            publish()
            print(f"Paired baseline {plan['run_id']}: {task}", flush=True)
            for command in (
                ["bash", str(project / "scripts/verify_runpod_mounted_readiness.sh"), "--baseline"],
                [str(project / ".venv/bin/python"), "-m", "stock_forecasting.cli.build_baselines",
                 "--selection", str(selected_path)],
            ):
                result = execute(command, env=child_env, cwd=project, check=False)
                if result.returncode:
                    state["state"] = "incomplete"
                    state["children"][task] = {"state": "failed", "exit_code": result.returncode}
                    publish()
                    return result.returncode
            state["children"][task] = {"state": "complete"}
            publish()
        except Exception as error:
            # Keep queryable metadata truthful when a child never reaches its
            # own CLI (missing executable, invalid selection, I/O failure).
            state["state"] = "incomplete"
            state["children"][task] = {"state": "failed", "error_type": type(error).__name__}
            publish()
            raise
    state["state"] = "complete"
    publish()
    print(f"Paired baseline {plan['run_id']}: BOTH task caches complete", flush=True)
    return 0


def main() -> int:
    volume = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    project = Path(os.environ.get("PROJECT_ROOT", str(volume / "stock_forecasting")))
    return run(project, volume, Path(os.environ["RUNPOD_PAIRED_PLAN_PATH"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"Paired baseline failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
