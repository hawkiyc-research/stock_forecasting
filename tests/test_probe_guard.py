"""Exercise the diagnostic publisher and real local guard with fake cloud transports."""

from __future__ import annotations

import io
import json
import runpy
import shlex
import shutil
import subprocess
import sys
import threading
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from test_probe_tmux import ROOT, ProbeHarness

CONTRACT = runpy.run_path(str(ROOT / "scripts/runpod_probe_lifecycle.py"))


class ProbeGuardTests(unittest.TestCase):
    def test_concurrent_run_guards_never_consume_other_runs_completion(self):
        harness = self.harness()
        harness.write_script(
            harness.bin / "python3",
            'if [[ "$1" == */runpod_rest_v2_control.py ]]; then\n'
            '  [[ "$2 $3" == "pod delete" ]] || exit 91\n'
            '  printf "%s\\n" "$4" >> ' + shlex.quote(str(harness.root / "scoped-deletes")) + "\n"
            '  printf "{}\\n"\n'
            "  exit 0\nfi\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
        )

        def publish(name, ready):
            marker = harness.volume / f"lifecycle/runs/run-{name}/training.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix(".pending")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "stage1-training",
                        "state": "ready" if ready else "preparing",
                        "pod_id": "pod-" + name,
                        "wandb_run_id": "run-" + name,
                        "training_completed": ready,
                    }
                )
            )
            temporary.replace(marker)

        publish("a", True)
        publish("b", False)

        def guard(name):
            env = {
                **harness.environment,
                "RUNPOD_TEST_MODE": "1",
                "RUNPOD_GUARD_RUN_ID": "run-" + name,
                "RUNPOD_GUARD_LIFECYCLE_KEY": f"lifecycle/runs/run-{name}/training.json",
                "RUNPOD_GUARD_VOLUME_ROOT": str(harness.volume),
                "RUNPOD_GUARD_POLL_SECONDS": "1",
                "RUNPOD_GUARD_MAX_ATTEMPTS": "1",
            }
            env.pop("RUNPOD_POD_ID")
            return subprocess.run(
                [
                    "bash",
                    str(harness.scripts / "terminate_runpod_after.sh"),
                    "pod-" + name,
                    "8",
                    str(harness.root / f"guard-{name}.log"),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=12,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.submit(guard, "a"), pool.submit(guard, "b")
            result = first.result(timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr + harness.read("guard-a.log"))
            self.assertFalse(second.done(), "B stopped when only A had completed")
            self.assertEqual(harness.read("scoped-deletes").splitlines(), ["pod-a"])
            publish("b", True)
            result = second.result(timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr + harness.read("guard-b.log"))
        self.assertEqual(harness.read("scoped-deletes").splitlines(), ["pod-a", "pod-b"])
        for name in ("a", "b"):
            self.assertIn(
                "termination triggered by lifecycle-ready", harness.read(f"guard-{name}.log")
            )

    def harness(self) -> ProbeHarness:
        harness = ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        for relative in (
            "scripts/terminate_runpod_after.sh",
            "scripts/runpodctl_project.sh",
            "scripts/runpod_rest_v2_control.py",
            "scripts/lib/runpod_project_env.sh",
            "scripts/runpod_readiness.py",
            "src/stock_forecasting/data/content_identity.py",
            "src/stock_forecasting/dataset_identity.py",
            "src/stock_forecasting/dataset_profiles.py",
            "src/stock_forecasting/training_stage_contract.py",
        ):
            target = harness.project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        env_file = harness.project / ".env"
        env_file.write_text(
            "RUNPOD_API_KEY=local-test-only\nRUNPOD_NETWORK_VOLUME_ID=fixture-volume\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        harness.write_script(
            harness.bin / "python3",
            r"""
            if [[ "$1" == */runpod_rest_v2_control.py ]]; then
                root="$(cd "$(dirname "$1")/../../.." && pwd)"
                [[ "${RUNPOD_API_KEY:-}" == local-test-only ]] || exit 91
                [[ "$2 $3 $4" == 'pod delete probe-fixture' ]] || exit 92
                printf '%s\n' "$2 $3 $4" >> "${root}/local-delete-calls"
                printf '{"id":"probe-fixture","deleted":true}\n'
                exit 0
            fi
            exec """
            + shlex.quote(sys.executable)
            + ' "$@"\n',
        )
        harness.write_script(harness.bin / "aws", "exit 90\n")
        harness.write_script(
            harness.scripts / "runpod_s3_project.sh",
            r"""
            [[ "$1" == s3 && "$2" == cp ]] || exit 93
            if [[ "$3" == - ]]; then
                printf '%s\n' "$4" >> "${HARNESS_ROOT}/s3-write-calls"
                cat >/dev/null
                exit 0
            fi
            key="${3#s3://fixture-volume/}"
            [[ "${key}" != "$3" ]] || exit 94
            if [[ "${HARNESS_DIAGNOSTIC_TRANSIENT:-0}" == 1 \
                && "${key}" == lifecycle/diagnostics/* ]]; then
                [[ ! -f "${HARNESS_ROOT}/diagnostic-read-once" ]] || exit 5
                : > "${HARNESS_ROOT}/diagnostic-read-once"
            fi
            cat "${HARNESS_ROOT}/volume/${key}"
            exit "${HARNESS_S3_READ_EXIT:-0}"
            """,
        )
        return harness

    def payload(self, harness: ProbeHarness, state: str = "succeeded", code: int = 0) -> dict:
        job_dir = harness.volume / "logs/tmux/stock-forecasting-probe-scales/launch-fixture"
        return {
            "schema_version": 1,
            "kind": "representation-scale-probe",
            "pod_id": "probe-fixture",
            "owner_run_id": "run-pod-owner",
            "launch_id": "launch-fixture",
            "gpu_lease_acquired": True,
            "state": state,
            "exit_code": code,
            "generated_at": "2026-09-14T08:00:00+00:00",
            "log_path": str(job_dir / "combined.log"),
            "status_path": str(job_dir / "status.json"),
        }

    def run_guard(
        self, harness: ProbeHarness, seconds: int = 5, lifecycle: str = "training",
        overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        environment = {
            **harness.environment,
            "RUNPOD_TEST_MODE": "1",
            "RUNPOD_GUARD_LIFECYCLE_KEY": f"lifecycle/stage1/{lifecycle}.json",
            "RUNPOD_GUARD_RUN_ID": (
                "run-pod-owner" if lifecycle in {"training", "validation"} else ""
            ),
            "RUNPOD_GUARD_VOLUME_ROOT": str(harness.volume),
            "RUNPOD_GUARD_POLL_SECONDS": "1",
            "RUNPOD_GUARD_MAX_ATTEMPTS": "1",
        }
        environment.update(overrides or {})
        environment.pop("RUNPOD_POD_ID")
        return subprocess.run(
            [
                "bash", str(harness.scripts / "terminate_runpod_after.sh"),
                "probe-fixture", str(seconds), str(harness.root / "local-guard.log"),
            ],
            env=environment, capture_output=True, text=True, check=False, timeout=12,
        )

    def test_section_safe_limit_waits_past_hard_limit_for_remote_section(self) -> None:
        harness = self.harness()
        config = harness.project / "checkpoint-config.yaml"
        config.write_text("checkpoint: test\n", encoding="utf-8")
        (harness.scripts / "runpod_guard_checkpoint.py").write_text(
            'raise SystemExit(3)\n',
            encoding="utf-8",
        )
        marker = harness.volume / "lifecycle/stage1/training.json"

        def complete_section() -> None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "schema_version": 1, "kind": "stage1-training", "state": "timed_out",
                "pod_id": "probe-fixture", "wandb_run_id": "run-pod-owner",
                "launch_id": "launch-section", "exit_code": 124,
            }), encoding="utf-8")

        timer = threading.Timer(4, complete_section)
        timer.start()
        self.addCleanup(timer.join)
        result = self.run_guard(
            harness,
            seconds=3,
            overrides={
                "RUNPOD_GUARD_SOFT_LIMIT_SECONDS": "1",
                "RUNPOD_GUARD_SECTION_SAFE": "1",
                "RUNPOD_GUARD_CHECKPOINT_RETRY_SECONDS": "1",
                "RUNPOD_GUARD_CHECKPOINT_CONFIG": str(config),
                "RUNPOD_GUARD_DATASET_REQUEST_SHA256": "0" * 64,
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("stop-request.json", harness.read("s3-write-calls"))
        self.assertIn(
            "termination triggered by lifecycle-timed_out",
            harness.read("local-guard.log"),
        )
        self.assertEqual(harness.read("local-delete-calls"), "pod delete probe-fixture\n")

    def test_legacy_guard_still_enforces_hard_limit(self) -> None:
        harness = self.harness()
        config = harness.project / "checkpoint-config.yaml"
        config.write_text("checkpoint: test\n", encoding="utf-8")
        (harness.scripts / "runpod_guard_checkpoint.py").write_text(
            'raise SystemExit(3)\n', encoding="utf-8"
        )
        result = self.run_guard(
            harness,
            seconds=3,
            overrides={
                "RUNPOD_GUARD_SOFT_LIMIT_SECONDS": "1",
                "RUNPOD_GUARD_CHECKPOINT_RETRY_SECONDS": "1",
                "RUNPOD_GUARD_CHECKPOINT_CONFIG": str(config),
                "RUNPOD_GUARD_DATASET_REQUEST_SHA256": "0" * 64,
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("termination triggered by hard-limit", harness.read("local-guard.log"))
        self.assertEqual(harness.read("local-delete-calls"), "pod delete probe-fixture\n")

    def test_complete_runner_to_local_guard_to_project_cli_chain(self) -> None:
        for environment, state, code in (
            ({}, "succeeded", 0),
            ({"HARNESS_JOB_EXIT": "7"}, "failed", 7),
            ({"HARNESS_TIMEOUT": "1"}, "timed_out", 124),
        ):
            with self.subTest(state=state):
                harness = self.harness()
                harness.environment.update(environment)
                launch = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(launch.returncode, 0, launch.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, code, runner.stdout + runner.stderr)
                self.assertFalse((harness.root / "shutdown-called").exists())
                result = self.run_guard(harness)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(harness.read("local-delete-calls"), "pod delete probe-fixture\n")
                log = harness.read("local-guard.log")
                self.assertIn("termination triggered by lifecycle-" + state, log)
                self.assertIn("lifecycle/diagnostics/representation-scales/probe-fixture.json", log)
                self.assertFalse((harness.root / "s3-write-calls").exists())
                self.assertEqual(list((harness.volume / "lifecycle/stage1").glob("*.json")), [])

    def test_running_diagnostic_waits_for_hard_limit_without_training_timeout_writes(self) -> None:
        harness = self.harness()
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness, "running")), encoding="utf-8")
        result = self.run_guard(harness, seconds=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = harness.read("local-guard.log")
        self.assertIn("termination triggered by hard-limit", log)
        self.assertNotIn("lifecycle-running", log)
        self.assertFalse((harness.root / "s3-write-calls").exists())

    def test_s3_failure_cannot_authorize_termination_even_after_valid_json(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_S3_READ_EXIT"] = "5"
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness)), encoding="utf-8")
        result = self.run_guard(harness, seconds=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("termination triggered by hard-limit", harness.read("local-guard.log"))

    def test_guard_keeps_diagnostic_ownership_across_transient_read_failures(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_DIAGNOSTIC_TRANSIENT"] = "1"
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness, "running")), encoding="utf-8")
        training = harness.volume / "lifecycle/stage1/training.json"
        training.parent.mkdir(parents=True)
        training.write_text(json.dumps({
            "schema_version": 1, "kind": "stage1-training", "state": "failed",
            "pod_id": "probe-fixture", "wandb_run_id": "run-pod-owner",
            "launch_id": "launch-training", "exit_code": 7,
        }))
        result = self.run_guard(harness, seconds=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = harness.read("local-guard.log")
        self.assertIn("termination triggered by hard-limit", log)
        self.assertNotIn("termination triggered by lifecycle-failed", log)
        self.assertFalse((harness.root / "s3-write-calls").exists())

    def test_wrong_pod_or_owner_cannot_trigger_diagnostic_termination(self) -> None:
        for field, value in (("pod_id", "other-pod"), ("owner_run_id", "historical-model-run")):
            with self.subTest(field=field):
                harness = self.harness()
                marker = harness.diagnostic_marker()
                marker.parent.mkdir(parents=True)
                marker.write_text(json.dumps({**self.payload(harness), field: value}))
                result = self.run_guard(harness, seconds=1)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = harness.read("local-guard.log")
                self.assertIn("termination triggered by hard-limit", log)
                self.assertNotIn("termination triggered by lifecycle-succeeded", log)

    def test_existing_workflow_markers_still_trigger_local_termination(self) -> None:
        for lifecycle in ("cpu-preparation", "mixed-finalization", "training", "validation"):
            with self.subTest(lifecycle=lifecycle):
                harness = self.harness()
                marker = harness.volume / "lifecycle/stage1" / (lifecycle + ".json")
                marker.parent.mkdir(parents=True)
                marker.write_text(json.dumps({
                    "schema_version": 1, "kind": "stage1-" + lifecycle, "state": "failed",
                    "pod_id": "probe-fixture", "wandb_run_id": "run-pod-owner",
                    "launch_id": "launch-existing", "exit_code": 7,
                }))
                result = self.run_guard(harness, lifecycle=lifecycle)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    "termination triggered by lifecycle-failed", harness.read("local-guard.log")
                )

    def test_marker_validation_rejects_untrusted_fields_and_inconsistent_states(self) -> None:
        harness = self.harness()
        base = self.payload(harness)
        for field, value in (
            ("schema_version", True), ("kind", "stage1-training"),
            ("pod_id", "another-pod"), ("owner_run_id", "run-checkpoint"),
            ("launch_id", "../unsafe"), ("gpu_lease_acquired", False),
            ("log_path", "/tmp/other.log"), ("status_path", "/tmp/other.json"),
            ("state", "complete"), ("state", "failed"), ("state", "timed_out"),
            ("exit_code", True), ("exit_code", -1), ("exit_code", 256),
            ("generated_at", "2026-09-14T08:00:00"), ("generated_at", "invalid"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                CONTRACT["validate"](
                    {**base, field: value}, str(harness.volume), "probe-fixture", "run-pod-owner"
                )

    def test_json_reads_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounded read limit"):
            CONTRACT["load_bounded"](io.BytesIO(b" " * 65537))

    def test_terminal_publication_requires_persisted_matching_status(self) -> None:
        harness = self.harness()
        arguments = Namespace(
            network_volume_root=str(harness.volume), pod_id="probe-fixture",
            owner_run_id="run-pod-owner", launch_id="launch-fixture",
            state="succeeded", exit_code=0,
        )
        with (
            patch.dict("os.environ", {"RUNPOD_GPU_WORKFLOW_LEASE_HELD": "1"}),
            patch("os.fstat"),
        ):
            with self.assertRaises(FileNotFoundError):
                CONTRACT["publish"](arguments)
            paths = CONTRACT["identity"](
                str(harness.volume), "probe-fixture", "run-pod-owner", "launch-fixture"
            )
            paths["status_path"].parent.mkdir(parents=True)
            paths["status_path"].write_text(json.dumps({
                "state": "failed", "exit_code": 7, "log_path": str(paths["log_path"]),
            }))
            with self.assertRaisesRegex(ValueError, "matching persisted tmux status"):
                CONTRACT["publish"](arguments)
            self.assertFalse(harness.diagnostic_marker().exists())


if __name__ == "__main__":
    unittest.main()
