"""Local-only orphan cleanup must never delete Pods or signal reused PIDs."""

from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
API = runpy.run_path(str(ROOT / "scripts/runpod_cleanup_monitors.py"))


class CleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="monitor-cleanup-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.pod = "fixture-pod"
        self.armed = 1_700_000_000
        self.ready = {"state": "armed", "pod_id": self.pod, "pid": 1234,
                      "run_id": "run-fixture", "host_boot_id": "fixture-boot",
                      "armed_at": datetime.fromtimestamp(self.armed, UTC).isoformat(),
                      "delay_seconds": 3600}
        self.monitor = {"kind": "runpod-gpu-monitor", "project_root": str(ROOT),
                        "pod_id": self.pod, "run_id": "run-fixture", "pid": 1235,
                        "guard_pid": 1234, "guard_identity": self.ready,
                        "guard_log": str(self.directory / f"{self.pod}.log"),
                        "updated_at": self.armed + 1000, "state": "monitoring"}
        self.write("ready.json", self.ready)
        self.write("gpu-monitor.json", self.monitor)
        self.write("keep-awake.json", {"state": "armed", "mechanism": "caffeinate",
                                        "pid": 1236, "guard_pid": 1234})
        self.write("pid", "1234\n")
        self.write("caffeinate.pid", "1236\n")
        self.processes = {
            1234: self.identity(1234, ["bash", str(ROOT / "scripts/terminate_runpod_after.sh"),
                self.pod, "3600", str(self.directory / f"{self.pod}.log")], -1),
            1235: self.identity(1235, ["python3", str(ROOT / "scripts/runpod_gpu_monitor.py"),
                "watch", "--pod-id", self.pod, "--run-id", "run-fixture", "--guard-pid",
                "1234", "--guard-log", str(self.directory / f"{self.pod}.log")], 1),
            1236: self.identity(1236, ["caffeinate", "-is", "-w", "1234"], 1),
        }
        self.sent = []
        self.absence_checks = []
        self.globals = API["cleanup"].__globals__
        self.context = patch.dict(self.globals, {
            "assert_pod_absent": lambda pod: self.absence_checks.append(pod),
            "host_boot_identity": lambda: ("fixture-boot", self.armed - 100),
            "process_identity": lambda pid: self.processes.get(pid),
        })
        self.context.start()
        self.addCleanup(self.context.stop)

    def write(self, suffix, value):
        path = self.directory / f"{self.pod}.{suffix}"
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def identity(self, pid, argv, offset=0):
        return {"pid": pid, "uid": os.getuid(), "started_at": self.armed + offset, "argv": argv}

    def terminate(self, pid, sig):
        self.sent.append((pid, sig))
        self.processes.pop(pid, None)

    def run_cleanup(self, apply=False):
        with patch.object(self.globals["os"], "kill", self.terminate):
            return API["cleanup"](self.pod, self.directory, apply=apply)

    def test_dry_run_verifies_all_three_without_any_signal_or_writes(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        report = self.run_cleanup()
        self.assertEqual(report["state"], "dry_run")
        self.assertEqual([item["state"] for item in report["targets"]], ["verified"] * 3)
        self.assertEqual(self.sent, [])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_apply_only_signals_exact_owners_and_is_idempotent(self):
        report = self.run_cleanup(True)
        self.assertEqual(report["state"], "complete")
        self.assertEqual(self.sent, [(1235, signal.SIGTERM), (1234, signal.SIGTERM), (1236, signal.SIGTERM)])
        self.assertEqual(self.absence_checks, [self.pod, self.pod])
        report = self.run_cleanup(True)
        self.assertEqual(report["state"], "complete")
        self.assertEqual(len(self.sent), 3)
        self.assertEqual({item["state"] for item in report["targets"]}, {"already_stopped"})

    def test_existing_or_stopped_pod_blocks_before_local_inspection(self):
        with patch.dict(self.globals, {"assert_pod_absent": lambda _: (_ for _ in ()).throw(RuntimeError("exists"))}):
            with self.assertRaisesRegex(RuntimeError, "exists"):
                self.run_cleanup(True)
        self.assertEqual(self.sent, [])

    def test_other_run_or_project_commands_block_all_signals(self):
        for index, value in ((1, "/other/scripts/runpod_gpu_monitor.py"), (6, "run-other"),
                             (4, "other-pod"), (8, "9999")):
            with self.subTest(index=index):
                original = self.processes[1235]["argv"][index]
                self.processes[1235]["argv"][index] = value
                self.assertEqual(self.run_cleanup(True)["state"], "blocked")
                self.processes[1235]["argv"][index] = original
        self.assertEqual(self.sent, [])

    def test_reused_guard_pid_by_same_script_after_armed_time_is_not_killed(self):
        self.processes[1234]["started_at"] = self.armed + 3600
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_changed_boot_or_missing_boot_blocks_signals(self):
        for boot in ("another-boot", ""):
            with self.subTest(boot=boot), patch.dict(self.globals, {"host_boot_identity": lambda: (boot, self.armed - 100)}):
                self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_reused_monitor_pid_after_snapshot_is_not_killed(self):
        self.processes[1235]["started_at"] = self.armed + 2000
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_keep_awake_for_different_guard_is_not_killed(self):
        self.processes[1236]["argv"][-1] = "9999"
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_pidfile_disagreement_or_different_user_blocks(self):
        self.write("pid", "4321\n")
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.write("pid", "1234\n")
        self.processes[1235]["uid"] = os.getuid() + 1
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_monitor_identity_must_equal_current_ready_record(self):
        self.monitor["guard_identity"] = {**self.ready, "run_id": "run-other"}
        self.write("gpu-monitor.json", self.monitor)
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_missing_records_return_clean_noop(self):
        report = API["cleanup"]("no-record-pod", self.directory, apply=True)
        self.assertEqual(report["state"], "complete")
        self.assertEqual(report["targets"], [])

    def test_missing_pid_is_not_reported_as_completed_cleanup(self):
        self.monitor.pop("pid")
        self.write("gpu-monitor.json", self.monitor)
        self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_permission_denied_blocks_no_bypass(self):
        def denied(pid):
            raise PermissionError("Operation not permitted")
        with patch.dict(self.globals, {"process_identity": denied}):
            self.assertEqual(self.run_cleanup(True)["state"], "blocked")
        self.assertEqual(self.sent, [])

    def test_remote_reappears_at_final_check_blocks_all_signals(self):
        count = 0
        def check(pod):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("Pod exists")
        with patch.dict(self.globals, {"assert_pod_absent": check}):
            with self.assertRaisesRegex(RuntimeError, "exists"):
                self.run_cleanup(True)
        self.assertEqual(self.sent, [])

    def test_identity_change_before_signal_does_not_kill_reused_pid(self):
        calls = 0
        def inspect(pid):
            nonlocal calls
            calls += 1
            if calls > 6 and pid == 1235:
                return {**self.processes[pid], "started_at": self.armed + 3600}
            return self.processes.get(pid)
        with patch.dict(self.globals, {"process_identity": inspect}):
            report = self.run_cleanup(True)
        self.assertEqual(report["state"], "blocked")
        self.assertNotIn((1235, signal.SIGTERM), self.sent)

    def test_no_force_kill_if_process_ignores_term(self):
        with patch.object(self.globals["os"], "kill", lambda pid, sig: self.sent.append((pid, sig))), patch.dict(self.globals, {"EXIT_TIMEOUT_SECONDS": 0}):
            report = API["cleanup"](self.pod, self.directory, apply=True)
        self.assertEqual(report["state"], "blocked")
        self.assertEqual({sig for _, sig in self.sent}, {signal.SIGTERM})

    def test_api_checks_all_pods_and_rejects_failure_or_invalid_response(self):
        original = API["assert_pod_absent"]
        # runpy function globals are patched above, but this function itself is preserved.
        for payload, code in (([], 0), ([{"id": self.pod, "desiredStatus": "EXITED"}], 0),
                              ({"pods": []}, 0), ([{}], 0), ([], 1)):
            with self.subTest(payload=payload, code=code), patch.object(original.__globals__["subprocess"], "run", return_value=subprocess.CompletedProcess([], code, json.dumps(payload), "")) as run:
                if payload == [] and code == 0:
                    original(self.pod)
                else:
                    with self.assertRaises(RuntimeError):
                        original(self.pod)
                command = run.call_args.args[0]
                self.assertEqual(command[-5:], ["pod", "list", "--all", "--output", "json"])

    def test_ps_parser_checks_full_command_and_start_time(self):
        original = API["process_identity"]
        raw = " 1234 501 Tue Nov 14 22:13:20 2023 S bash /project/guard.sh fixture 3600 /log\n"
        with patch.object(original.__globals__["subprocess"], "run", return_value=subprocess.CompletedProcess([], 0, raw, "")):
            result = original(1234)
        self.assertEqual(result["started_at"], self.armed)
        self.assertEqual(result["argv"], ["bash", "/project/guard.sh", "fixture", "3600", "/log"])
        with patch.object(original.__globals__["subprocess"], "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            self.assertIsNone(original(1234))
        with patch.object(original.__globals__["subprocess"], "run", return_value=subprocess.CompletedProcess([], 1, "", "Operation not permitted")):
            with self.assertRaises(RuntimeError):
                original(1234)

    def test_cli_default_is_dry_run_and_retains_existing_dispatch(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(API["main"](["--pod-id", self.pod, "--guard-dir", str(self.directory)]), 0)
        self.assertIn("Dry run only", output.getvalue())
        self.assertEqual(self.sent, [])
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text()
        self.assertIn('runpod_cleanup_monitors.py" "$@"', workflow)
        self.assertIn('runpod_gpu_monitor.py" status "$@"', workflow)
        self.assertIn('recover_runpod_after_wake.py" "$@"', workflow)


@unittest.skipUnless(os.environ.get("RUNPOD_CLEANUP_PROCESS_FIXTURE") == "1",
                     "Opt-in harmless subprocess fixture requires local ps permission")
class RealProcessFixtureTests(unittest.TestCase):
    def test_actual_sigterm_stops_only_the_fixture_guard(self):
        with tempfile.TemporaryDirectory(prefix="cleanup-live-fixture-") as temporary:
            root = Path(temporary).resolve()
            (root / "scripts").mkdir()
            script = root / "scripts/terminate_runpod_after.sh"
            script.write_text("#!/usr/bin/env bash\ntrap 'exit 0' TERM\nwhile :; do sleep 0.1; done\n")
            directory = root / "records"
            directory.mkdir()
            pod = "cleanup-fixture"
            guard = subprocess.Popen(["bash", str(script), pod, "3600", str(directory / f"{pod}.log")],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            protected = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                boot, _ = API["host_boot_identity"]()
                ready = {"state": "armed", "pod_id": pod, "pid": guard.pid,
                         "run_id": "run-fixture", "host_boot_id": boot,
                         "armed_at": datetime.now(UTC).isoformat(), "delay_seconds": 3600}
                (directory / f"{pod}.ready.json").write_text(json.dumps(ready))
                (directory / f"{pod}.pid").write_text(str(guard.pid))
                with patch.dict(API["cleanup"].__globals__, {"ROOT": root, "assert_pod_absent": lambda _: None}):
                    report = API["cleanup"](pod, directory, apply=True)
                self.assertEqual(report["state"], "complete", report)
                guard.wait(timeout=3)
                self.assertIsNone(protected.poll())
                self.assertTrue((directory / f"{pod}.ready.json").exists())
            finally:
                for process in (guard, protected):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
