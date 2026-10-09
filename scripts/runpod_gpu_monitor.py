#!/usr/bin/env python3
"""Local-only GPU alerts, isolated from RunPod lifecycle and termination decisions."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass
from itertools import islice
from logging.handlers import RotatingFileHandler
from pathlib import Path
from xml.etree.ElementTree import ParseError as ETParseError

from runpod_gpu_telemetry import TelemetrySSH, bounded_call

ROOT = Path(__file__).resolve().parents[1]
KIND = "runpod-gpu-monitor"
MAX_STATUS_BYTES = 256 * 1024
MAX_STATUS_FILES = 1000
THERMAL_HARDWARE = (
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "hw_slowdown",
)


@dataclass(frozen=True)
class Settings:
    poll_seconds: float = 30
    confirm_seconds: float = 120
    cooldown_seconds: float = 900
    startup_grace_seconds: float = 300
    high_utilization_pct: float = 80
    clock_drop_fraction: float = 0.20
    query_timeout_seconds: float = 15
    notifications: str = "auto"

    @classmethod
    def from_env(cls):
        values = {}
        ranges = {
            "poll_seconds": (10, 300),
            "confirm_seconds": (30, 3600),
            "cooldown_seconds": (60, 86400),
            "startup_grace_seconds": (0, 1800),
            "high_utilization_pct": (10, 100),
            "clock_drop_fraction": (0.05, 0.8),
            "query_timeout_seconds": (3, 30),
        }
        for name, (low, high) in ranges.items():
            key = "RUNPOD_GPU_MONITOR_" + name.upper()
            value = float(os.environ.get(key, getattr(cls(), name)))
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            values[name] = value
        values["notifications"] = os.environ.get("RUNPOD_GPU_MONITOR_NOTIFICATIONS", "auto")
        if values["notifications"] not in ("auto", "0", "1"):
            raise ValueError("RUNPOD_GPU_MONITOR_NOTIFICATIONS must be auto, 0, or 1")
        return cls(**values)


def read_json(path):
    if path.stat().st_size > MAX_STATUS_BYTES:
        raise ValueError("GPU monitor metadata exceeds its size limit")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("GPU monitor metadata must be an object")
    return value


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".gpu-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def paths(log):
    base = str(log.with_suffix(""))
    return {
        "ready": Path(base + ".ready.json"),
        "status": Path(base + ".gpu-monitor.json"),
        "events": Path(base + ".gpu-events.jsonl"),
        "lock": Path(base + ".gpu-monitor.lock"),
    }


def alive(pid):
    try:
        pid = int(pid)
        if pid <= 1:
            return False
        os.kill(pid, 0)
        return True
    except (ValueError, TypeError, ProcessLookupError):
        return False


def guard_matches(ready_path, identity):
    try:
        ready = read_json(ready_path)
        if (
            any(
                ready.get(key) != identity.get(key)
                for key in ("pid", "pod_id", "run_id", "armed_at", "host_boot_id")
            )
            or ready.get("state") != "armed"
        ):
            return False
        if not alive(identity["pid"]):
            return False
        # PID reuse must not keep an orphan monitor alive after the guard ends.
        command = bounded_call(["/bin/ps", "-p", str(identity["pid"]), "-o", "command="], timeout=3)
        words = shlex.split(command)
        return identity["pod_id"] in words and any(
            Path(word).name == "terminate_runpod_after.sh" for word in words
        )
    except PermissionError as error:
        raise RuntimeError(
            "Local process inspection is denied; GPU monitoring cannot verify guard ownership"
        ) from error
    except (OSError, ValueError, RuntimeError, TimeoutError):
        return False


class Detector:
    """Per-device ordered observations; never classify old counters as current events."""

    def __init__(self, settings):
        self.settings = settings
        self.devices = {}

    def observe(self, sample, now):
        state = self.devices.setdefault(
            sample["uuid"],
            {
                "last_time": None,
                "counters": {},
                "baseline": deque(maxlen=20),
                "low_since": None,
                "clear_since": None,
                "missing_since": None,
                "alerts": set(),
            },
        )
        elapsed = None if state["last_time"] is None else now - state["last_time"]
        contiguous = elapsed is not None and 0 < elapsed <= self.settings.poll_seconds * 3
        deltas = {}
        for key, value in sample["counters_us"].items():
            previous = state["counters"].get(key)
            if value is not None and previous is not None and value >= previous and contiguous:
                deltas[key] = value - previous
        state["last_time"], state["counters"] = now, sample["counters_us"]
        if not contiguous:
            state["low_since"] = state["clear_since"] = None
            state["missing_since"] = None
        reasons = sample["reasons"]
        confirmed = [
            key for key in THERMAL_HARDWARE if reasons.get(key) is True or deltas.get(key, 0) > 0
        ]
        clock = sample.get("sm_mhz") or sample.get("graphics_mhz")
        util = sample.get("utilization_pct")
        core_available = clock is not None and util is not None
        reasons_available = all(reasons.get(key) is not None for key in THERMAL_HARDWARE)
        high_load = core_available and util >= self.settings.high_utilization_pct
        baseline = state["baseline"]
        reference = (
            statistics.median(value for _, value in baseline) if len(baseline) >= 4 else None
        )
        low = bool(
            high_load and reference and clock < reference * (1 - self.settings.clock_drop_fraction)
        )
        if low:
            if state["low_since"] is None:
                state["low_since"] = now
        else:
            state["low_since"] = None
        suspected = low and now - state["low_since"] >= self.settings.confirm_seconds
        alerts = set(confirmed)
        if suspected:
            alerts.add("suspected_low_clock")
        if core_available and reasons_available:
            state["missing_since"] = None
        else:
            if state["missing_since"] is None:
                state["missing_since"] = now
            if now - state["missing_since"] >= self.settings.confirm_seconds:
                alerts.add("telemetry_incomplete")
        power, default_power = sample.get("power_limit_w"), sample.get("default_power_limit_w")
        if power is not None and default_power and power < default_power * 0.9:
            alerts.add("configured_power_limit_below_default")
        # Do not teach the baseline a throttled operating point or slow downward drift.
        if (
            high_load
            and not confirmed
            and not low
            and reasons_available
            and not any(value is True for key, value in reasons.items() if key != "sw_power_cap")
            and (len(baseline) < baseline.maxlen or clock > reference)
        ):
            baseline.append((now, clock))
        recovered = []
        can_recover = core_available and reasons_available
        if "suspected_low_clock" in state["alerts"]:
            can_recover = can_recover and high_load and not low
        if "configured_power_limit_below_default" in state["alerts"]:
            can_recover = can_recover and power is not None and bool(default_power)
        if state["alerts"] - alerts and can_recover:
            if state["clear_since"] is None:
                state["clear_since"] = now
            if now - state["clear_since"] >= self.settings.confirm_seconds:
                recovered = sorted(state["alerts"] - alerts)
                state["alerts"] &= alerts
        else:
            state["clear_since"] = None
        state["alerts"] |= alerts
        if confirmed:
            status = "throttling_detected"
        elif not core_available or not reasons_available:
            status = "limited_telemetry"
        elif alerts:
            status = "warning"
        elif reasons.get("sw_power_cap") is True or deltas.get("sw_power_cap", 0) > 0:
            status = "power_limited"
        elif low:
            status = "observing_low_clock"
        elif not high_load:
            status = "idle_or_low_load"
        elif reference is None:
            status = "learning_baseline"
        else:
            status = "no_throttling_observed"
        return {
            **sample,
            "state": status,
            "alerts": sorted(alerts),
            "recovered": recovered,
            "reference_sm_mhz": reference,
            "counter_deltas_us": deltas,
            "counter_interval_seconds": elapsed if contiguous else None,
        }

    def retain(self, uuids):
        # GPU replacement resets baselines and keeps memory bounded.
        self.devices = {key: value for key, value in self.devices.items() if key in uuids}


def notify(title, message, mode="auto"):
    if mode == "0" or (mode == "auto" and sys.platform != "darwin"):
        return "disabled"
    if sys.platform != "darwin":
        return "unsupported_platform"
    # Arguments are data, never interpolated AppleScript source.
    script = (
        "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run"
    )
    bounded_call(["/usr/bin/osascript", "-e", script, title[:160], message[:900]], timeout=5)
    return "submitted"  # Focus mode/Notification Center can still suppress display.


class Reporter:
    def __init__(self, event_path, settings, notifier=notify):
        self.settings, self.notifier = settings, notifier
        self.last_alerts = {}
        self.logger = logging.getLogger(str(event_path))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = RotatingFileHandler(
            event_path, maxBytes=5 * 1024**2, backupCount=2, encoding="utf-8"
        )
        self.logger.addHandler(self.handler)
        self.notification_status = "not_sent"

    def record(self, kind, **fields):
        self.logger.info(
            json.dumps({"time": time.time(), "kind": kind, **fields}, ensure_ascii=False)
        )

    def alert(self, key, title, message, now):
        if now - self.last_alerts.get(key, float("-inf")) < self.settings.cooldown_seconds:
            return
        self.last_alerts[key] = now
        self.record("alert", key=key, message=message)
        try:
            self.notification_status = self.notifier(title, message, self.settings.notifications)
        except (OSError, RuntimeError, TimeoutError) as error:
            self.notification_status = "failed: " + str(error)[:300]
        self.record("notification", status=self.notification_status)

    def close(self):
        self.handler.close()
        self.logger.removeHandler(self.handler)


def monitor(args):
    files = paths(args.guard_log)
    identity = read_json(files["ready"])
    if (
        identity.get("pid") != args.guard_pid
        or identity.get("pod_id") != args.pod_id
        or identity.get("run_id") != args.run_id
    ):
        raise ValueError("GPU monitor does not own this guard")
    with files["lock"].open("a") as lease:
        settings = Settings.from_env()
        lock_deadline = (
            time.monotonic() + 2 * settings.poll_seconds + settings.query_timeout_seconds
        )
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # A re-armed guard gets a new owner after the old watcher exits.
                if time.monotonic() >= lock_deadline or not guard_matches(files["ready"], identity):
                    raise ValueError("A GPU monitor already owns this guard log") from None
                time.sleep(0.2)
        stopped = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stopped.set())
        reporter = Reporter(files["events"], settings)
        detector = Detector(settings)
        transport = TelemetrySSH(
            ROOT,
            args.pod_id,
            os.environ.get(
                "RUNPOD_GPU_MONITOR_SSH_KEY", str(Path.home() / ".ssh/id_ed25519_runpod")
            ),
            timeout=settings.query_timeout_seconds,
            stop=stopped.is_set,
        )
        started, error_since, was_unavailable = time.monotonic(), None, False
        snapshot = {
            "kind": KIND,
            "project_root": str(ROOT),
            "pod_id": args.pod_id,
            "run_id": args.run_id,
            "pid": os.getpid(),
            "guard_pid": args.guard_pid,
            "guard_identity": identity,
            "guard_log": str(args.guard_log),
            "settings": asdict(settings),
            "gpus": [],
            "state": "starting",
            "notification_status": "not_sent",
        }
        snapshot["updated_at"] = time.time()
        atomic_json(files["status"], snapshot)
        try:
            while not stopped.is_set() and guard_matches(files["ready"], identity):
                now = time.monotonic()
                try:
                    samples = transport.sample()
                    observation_time = time.monotonic()
                    detector.retain({item["uuid"] for item in samples})
                    snapshot["gpus"] = [
                        detector.observe(item, observation_time) for item in samples
                    ]
                    snapshot.update(
                        state="monitoring",
                        pod_name=transport.name,
                        error=None,
                        last_sample_at=time.time(),
                    )
                    if was_unavailable:
                        reporter.alert(
                            "connection_recovered",
                            "GPU 監控已恢復",
                            f"{transport.name}\nPod: {args.pod_id}\nRun: {args.run_id}",
                            now,
                        )
                        reporter.last_alerts.pop("monitor_unavailable", None)
                    error_since, was_unavailable = None, False
                    for gpu in snapshot["gpus"]:
                        for reason in gpu["alerts"]:
                            reporter.alert(
                                gpu["uuid"] + ":" + reason,
                                "RunPod GPU 警示",
                                (
                                    f"{transport.name} | {gpu['name']}\n"
                                    f"Pod: {args.pod_id}\nRun: {args.run_id}\n"
                                    f"{reason}; SM={gpu['sm_mhz']} MHz; "
                                    f"GPU={gpu['utilization_pct']}%; {gpu['temperature_c']} C; "
                                    f"{gpu['power_w']}/{gpu['power_limit_w']} W"
                                ),
                                now,
                            )
                        for reason in gpu["recovered"]:
                            reporter.alert(
                                gpu["uuid"] + ":recovered:" + reason,
                                "GPU 降頻警示已解除",
                                f"{transport.name}\nPod: {args.pod_id}\n{reason}",
                                now,
                            )
                            reporter.last_alerts.pop(gpu["uuid"] + ":" + reason, None)
                    reporter.record(
                        "sample", pod_id=args.pod_id, run_id=args.run_id, gpus=snapshot["gpus"]
                    )
                except (OSError, ValueError, RuntimeError, TimeoutError, ETParseError) as error:
                    if error_since is None:
                        error_since = now
                    waiting = (
                        snapshot.get("last_sample_at") is None
                        and now - started < settings.startup_grace_seconds
                    )
                    snapshot.update(
                        state="waiting_for_ssh" if waiting else "unavailable",
                        error=str(error)[:500],
                    )
                    reporter.record("query_failed", pod_id=args.pod_id, error=snapshot["error"])
                    if not waiting and (
                        now - error_since >= settings.confirm_seconds
                        or snapshot.get("last_sample_at") is None
                    ):
                        was_unavailable = True
                        reporter.alert(
                            "monitor_unavailable",
                            "GPU 監控無法取得資料",
                            f"Pod: {args.pod_id}\nRun: {args.run_id}\n"
                            f"{snapshot['error']}\n原 guard 繼續執行。",
                            now,
                        )
                snapshot.update(
                    updated_at=time.time(), notification_status=reporter.notification_status
                )
                atomic_json(files["status"], snapshot)
                stopped.wait(max(0.1, settings.poll_seconds - (time.monotonic() - now)))
        finally:
            failed = sys.exc_info()[0] is not None and not stopped.is_set()
            if failed:
                snapshot["error"] = str(sys.exc_info()[1])[:500]
            transport.close()
            snapshot.update(state="failed" if failed else "stopped", updated_at=time.time())
            atomic_json(files["status"], snapshot)
            reporter.record("stopped", pod_id=args.pod_id, guard_pid=args.guard_pid)
            reporter.close()
    return 0


def start(args):
    if os.environ.get("RUNPOD_GPU_MONITOR", "1") == "0":
        return 0
    if os.environ.get("RUNPOD_GPU_MONITOR", "1") != "1":
        raise ValueError("RUNPOD_GPU_MONITOR must be 0 or 1")
    Settings.from_env()
    files = paths(args.guard_log)
    ready = read_json(files["ready"])
    if (
        ready.get("pid") != args.guard_pid
        or ready.get("pod_id") != args.pod_id
        or ready.get("run_id") != args.run_id
    ):
        raise ValueError("GPU monitor guard identity mismatch")
    if files["status"].exists():
        previous = read_json(files["status"])
        if (
            previous.get("guard_identity") == ready
            and previous.get("state") not in ("stopped", "failed")
            and alive(previous.get("pid"))
        ):
            return 0
    # Only the control-plane environment is inherited; SSH_AUTH_SOCK is local-only.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("HOME", "PATH", "TMPDIR", "SSH_AUTH_SOCK", "RUNPOD_ENV_FILE")
        or key.startswith("RUNPOD_GPU_MONITOR")
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "watch",
        "--pod-id",
        args.pod_id,
        "--run-id",
        args.run_id,
        "--guard-pid",
        str(args.guard_pid),
        "--guard-log",
        str(args.guard_log),
    ]
    with args.guard_log.open("a") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    # No SSH handshake here: missing keys or slow Pod boot must never block launch.
    time.sleep(0.1)
    if child.poll() is not None and child.returncode:
        raise RuntimeError("GPU monitor failed to start; original guard is unchanged")
    return 0


def status_records(directory, *, pod_id=None, run_id=None):
    result = []
    # These tiny local control records are sequential to bound FDs and avoid a pool per status call.
    for index, path in enumerate(
        sorted(islice(directory.glob("*.gpu-monitor.json"), MAX_STATUS_FILES + 1))
    ):
        if index >= MAX_STATUS_FILES:
            raise ValueError("Too many GPU monitor records; archive old local guard records")
        try:
            item = read_json(path)
            if item.get("kind") != KIND or item.get("project_root") != str(ROOT):
                continue
            if (pod_id and item.get("pod_id") != pod_id) or (
                run_id and item.get("run_id") != run_id
            ):
                continue
            if item.get("state") not in ("stopped", "failed"):
                age = time.time() - float(item.get("updated_at", 0))
                if (
                    not alive(item.get("pid"))
                    or age > float(item["settings"]["poll_seconds"]) * 3 + 30
                ):
                    item = {
                        **item,
                        "state": "stale",
                        "error": "Monitor stopped or telemetry is stale; GPU health is unknown",
                    }
            result.append(item)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.append({"state": "unreadable", "path": str(path), "error": str(error)})
    return result


def print_status(records):
    if not records:
        print("GPU monitor: no local monitoring records (not a GPU health assessment).")
    for item in records:
        print(
            f"GPU monitor: {item.get('pod_name') or item.get('pod_id', '?')} | "
            f"{item['state']} | run={item.get('run_id', '?')}"
        )
        for gpu in item.get("gpus", []):
            label = gpu["state"] if item["state"] == "monitoring" else "last observation only"
            print(
                f"  {gpu['name']} | {label} | SM={gpu['sm_mhz']} MHz | "
                f"GPU={gpu['utilization_pct']}% | "
                f"{gpu['temperature_c']} C | {gpu['power_w']}/{gpu['power_limit_w']} W"
            )
        if item.get("error"):
            print("  " + item["error"])
        print("  Desktop notification: " + item.get("notification_status", "unknown"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "watch"):
        internal = commands.add_parser(name)
        internal.add_argument("--pod-id", required=True)
        internal.add_argument("--run-id", required=True)
        internal.add_argument("--guard-pid", type=int, required=True)
        internal.add_argument("--guard-log", type=Path, required=True)
    status = commands.add_parser(
        "status", help="Read local GPU monitoring records; never query or modify Pods"
    )
    status.add_argument("--pod-id")
    status.add_argument("--run-id")
    status.add_argument("--output", choices=("table", "json"), default="table")
    status.add_argument(
        "--guard-dir",
        type=Path,
        default=Path(
            os.environ.get("RUNPOD_GUARD_LOG_DIR", str(Path.home() / ".local/state/runpod-guards"))
        ),
    )
    commands.add_parser(
        "test-notification", help="Submit one explicitly labelled desktop test notification"
    )
    args = parser.parse_args()
    if args.command in ("start", "watch"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", args.run_id
        ):
            parser.error("Invalid Pod/run identity")
        if not args.guard_log.is_absolute() or args.guard_pid <= 1:
            parser.error("Guard log must be absolute and guard PID must exceed 1")
        return start(args) if args.command == "start" else monitor(args)
    if args.command == "test-notification":
        print(
            "GPU notification test: "
            + notify("RunPod GPU 監控測試", "這是通知測試。不代表 GPU 發生降頻。", "1")
        )
        return 0
    records = status_records(args.guard_dir, pod_id=args.pod_id, run_id=args.run_id)
    if args.output == "json":
        print(json.dumps({"monitors": records}, ensure_ascii=False))
    else:
        print_status(records)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print("GPU monitor: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
