#!/usr/bin/env python3
"""Stop only identity-verified local monitors for one already-deleted Pod."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from recover_runpod_after_wake import host_boot_identity

ROOT = Path(__file__).resolve().parents[1]
MAX_RECORD_BYTES = 256 * 1024
STARTUP_WINDOW_SECONDS = 120
EXIT_TIMEOUT_SECONDS = 45


def assert_pod_absent(pod_id):
    """A stopped Pod still exists; only confirmed absence permits local cleanup."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/runpodctl_project.sh"),
         "pod", "list", "--all", "--output", "json"],
        capture_output=True, text=True, timeout=45, check=False,
    )
    if result.returncode:
        raise RuntimeError("RunPod list failed; no local process was stopped")
    try:
        pods = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("RunPod list returned invalid JSON; cleanup refused") from error
    if not isinstance(pods, list) or any(
        not isinstance(pod, dict) or not isinstance(pod.get("id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", pod["id"]) for pod in pods
    ):
        raise RuntimeError("RunPod list has an invalid structure; cleanup refused")
    if any(pod["id"] == pod_id for pod in pods):
        raise RuntimeError(f"Pod {pod_id} still exists; its local protection must remain running")


def read_record(path):
    if path.is_symlink():
        raise ValueError(f"Refusing symlink control record: {path.name}")
    if not path.exists():
        return None
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError(f"Oversized control record: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid control record: {path.name}")
    return payload


def read_pid(path):
    if path.is_symlink():
        raise ValueError(f"Refusing symlink PID record: {path.name}")
    if not path.exists():
        return None
    if path.stat().st_size > 32:
        raise ValueError(f"Invalid PID record: {path.name}")
    value = int(path.read_text(encoding="utf-8").strip())
    if value <= 1:
        raise ValueError(f"Unsafe PID record: {path.name}")
    return value


def process_identity(pid):
    if type(pid) is not int or pid <= 1:
        raise ValueError("Process PID must be an integer greater than one")
    result = subprocess.run(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "pid=,uid=,lstart=,stat=,command="],
        capture_output=True, text=True, timeout=5, check=False,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
    )
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"Cannot inspect local PID {pid}; cleanup refused")
    fields = result.stdout.strip().split(None, 8)
    if len(fields) != 9 or int(fields[0]) != pid:
        raise RuntimeError(f"Invalid process identity for PID {pid}")
    if fields[7].startswith("Z"):
        return None
    started = datetime.strptime(" ".join(fields[2:7]), "%a %b %d %H:%M:%S %Y")
    return {"pid": pid, "uid": int(fields[1]),
            "started_at": started.replace(tzinfo=UTC).timestamp(),
            "argv": shlex.split(fields[8])}


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Guard creation time is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Guard creation time must include a timezone")
    return parsed.timestamp()


def verify_guard_record(ready, pod_id, boot):
    if not isinstance(ready, dict) or ready.get("pod_id") != pod_id:
        raise ValueError("Guard record does not match the requested Pod")
    if ready.get("state") != "armed" or type(ready.get("pid")) is not int or ready["pid"] <= 1:
        raise ValueError("Guard record is not an armed process identity")
    run_id = ready.get("run_id")
    if not isinstance(run_id, str) or (run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", run_id)):
        raise ValueError("Guard run identity is missing or invalid")
    current_boot, boot_epoch = boot
    armed_at = timestamp(ready.get("armed_at"))
    recorded_boot = ready.get("host_boot_id")
    if not current_boot or not recorded_boot or recorded_boot != current_boot:
        raise ValueError("Guard boot identity is missing or differs from the current host boot")
    if boot_epoch is not None and armed_at < boot_epoch:
        raise ValueError("Guard record predates the current host boot")
    return armed_at


def matching_command(kind, process, ready, log):
    argv = process["argv"]
    if kind == "guard":
        return (len(argv) == 5 and Path(argv[0]).name == "bash"
                and argv[1] == str(ROOT / "scripts/terminate_runpod_after.sh")
                and argv[2] == ready["pod_id"] and argv[3] == str(ready.get("delay_seconds"))
                and argv[4] == str(log))
    if kind == "gpu-monitor":
        return (len(argv) == 11 and re.fullmatch(r"python(?:[0-9.]+)?", Path(argv[0]).name)
                and argv[1:] == [str(ROOT / "scripts/runpod_gpu_monitor.py"), "watch",
                    "--pod-id", ready["pod_id"], "--run-id", ready["run_id"],
                    "--guard-pid", str(ready["pid"]), "--guard-log", str(log)])
    return argv == ["caffeinate", "-is", "-w", str(ready["pid"])] or argv == [
        "/usr/bin/caffeinate", "-is", "-w", str(ready["pid"])]


def inspect_targets(pod_id, directory):
    """At most three local owners are inspected; no system-wide process scan."""
    log = directory / f"{pod_id}.log"
    ready = read_record(directory / f"{pod_id}.ready.json")
    monitor = read_record(directory / f"{pod_id}.gpu-monitor.json")
    awake = read_record(directory / f"{pod_id}.keep-awake.json")
    guard_pid = read_pid(directory / f"{pod_id}.pid")
    awake_pid = read_pid(directory / f"{pod_id}.caffeinate.pid")
    if not any((ready, monitor, awake, guard_pid, awake_pid)):
        return []
    candidates = [
        ("gpu-monitor", (monitor or {}).get("pid"), monitor is not None),
        ("guard", (ready or {}).get("pid", guard_pid), ready is not None or guard_pid is not None),
        ("caffeinate", (awake or {}).get("pid", awake_pid),
         bool(awake and awake.get("state") != "not-required") or awake_pid is not None),
    ]
    boot = host_boot_identity()
    result = []
    for kind, pid, expected in candidates:
        if pid is None:
            if expected:
                result.append({"kind": kind, "pid": None, "state": "blocked",
                               "reason": "Control record is missing its process PID"})
            continue
        item = {"kind": kind, "pid": pid}
        result.append(item)
        try:
            process = process_identity(pid)
            if process is None:
                item.update(state="already_stopped")
                continue
            armed_at = verify_guard_record(ready, pod_id, boot)
            if guard_pid is not None and guard_pid != ready["pid"]:
                raise ValueError("Guard PID file and ready identity differ")
            if kind == "gpu-monitor":
                if (monitor.get("kind") != "runpod-gpu-monitor"
                    or monitor.get("project_root") != str(ROOT)
                    or monitor.get("pod_id") != pod_id or monitor.get("run_id") != ready["run_id"]
                    or monitor.get("guard_pid") != ready["pid"]
                    or monitor.get("guard_identity") != ready or monitor.get("guard_log") != str(log)):
                    raise ValueError("GPU monitor ownership differs from the guard record")
                updated = monitor.get("updated_at")
                if not isinstance(updated, (float, int)) or process["started_at"] > updated + 1:
                    raise ValueError("GPU monitor PID started after its recorded observation")
            if kind == "caffeinate" and (
                not awake or awake.get("state") != "armed" or awake.get("mechanism") != "caffeinate"
                or awake.get("guard_pid") != ready["pid"] or awake_pid != pid
            ):
                raise ValueError("Keep-awake ownership differs from the guard record")
            if process["uid"] != os.getuid():
                raise ValueError("Process belongs to a different local user")
            offset = process["started_at"] - armed_at
            if kind == "guard":
                valid_start = -STARTUP_WINDOW_SECONDS <= offset <= 1
            else:
                valid_start = -1 <= offset <= STARTUP_WINDOW_SECONDS
            if not valid_start:
                raise ValueError("Process creation time does not match this guard launch (possible PID reuse)")
            if not matching_command(kind, process, ready, log):
                raise ValueError("Process argv does not match the exact project/Pod/run owner")
            item.update(state="verified", identity=process)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            item.update(state="blocked", reason=str(error))
    return result


def cleanup(pod_id, directory, *, apply=False):
    assert_pod_absent(pod_id)
    targets = inspect_targets(pod_id, directory)
    if any(item["state"] == "blocked" for item in targets):
        return {"pod_id": pod_id, "state": "blocked", "targets": targets}
    if not apply:
        return {"pod_id": pod_id, "state": "dry_run", "targets": targets}
    # Confirm absence again immediately before signalling; never call a delete API.
    assert_pod_absent(pod_id)
    refreshed = inspect_targets(pod_id, directory)
    previous = {(item["kind"], item["pid"]): item.get("identity") for item in targets}
    if any(item["state"] == "blocked" or (
        item["state"] == "verified" and previous.get((item["kind"], item["pid"])) != item["identity"]
    ) for item in refreshed):
        return {"pod_id": pod_id, "state": "blocked", "targets": refreshed,
                "reason": "Local ownership changed; no signal was sent"}
    for item in refreshed:
        if item["state"] != "verified":
            continue
        current = process_identity(item["pid"])
        if current is None:
            item["state"] = "already_stopped"
            continue
        if current != item["identity"]:
            item.update(state="blocked", reason="PID identity changed before SIGTERM")
            continue
        try:
            os.kill(item["pid"], signal.SIGTERM)
            item["state"] = "stopping"
        except ProcessLookupError:
            item["state"] = "already_stopped"
    deadline = time.monotonic() + EXIT_TIMEOUT_SECONDS
    while any(item["state"] == "stopping" for item in refreshed):
        for item in refreshed:
            if item["state"] != "stopping":
                continue
            current = process_identity(item["pid"])
            if current is None or current != item["identity"]:
                item["state"] = "stopped"
            elif time.monotonic() >= deadline:
                item.update(state="blocked", reason="Process did not exit after SIGTERM; no force-kill sent")
        if any(item["state"] == "stopping" for item in refreshed):
            time.sleep(0.2)
    state = "blocked" if any(item["state"] == "blocked" for item in refreshed) else "complete"
    return {"pod_id": pod_id, "state": state, "targets": refreshed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True, help="One exact, already-deleted Pod ID")
    parser.add_argument("--apply", action="store_true", help="Send SIGTERM to verified local owners")
    parser.add_argument("--guard-dir", type=Path, default=Path(os.environ.get(
        "RUNPOD_GUARD_LOG_DIR", str(Path.home() / ".local/state/runpod-guards"))))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id):
        parser.error("--pod-id must be an exact safe Pod ID")
    try:
        report = cleanup(args.pod_id, args.guard_dir.resolve(), apply=args.apply)
        print(f"Pod {args.pod_id}: confirmed absent; local cleanup {report['state']}")
        for item in report["targets"]:
            print(f"  {item['kind']}: PID {item['pid']} {item['state']}" +
                  (f" — {item['reason']}" if item.get("reason") else ""))
        if not report["targets"]:
            print("  No local monitor records for this Pod.")
        print("No remote Pod changes; logs and shared SSH sockets are preserved.")
        if report["state"] == "dry_run":
            print("Dry run only. Add --apply to stop the verified local processes.")
        return 2 if report["state"] == "blocked" else 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Local monitor cleanup refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
