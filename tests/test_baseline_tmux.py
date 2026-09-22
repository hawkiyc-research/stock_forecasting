"""Baseline lifecycle integration with local stubs, never a real cloud workload."""

from __future__ import annotations

import unittest

from test_probe_tmux import ProbeHarness


class BaselineTmuxTests(unittest.TestCase):
    def harness(self):
        harness = ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        harness.environment["WANDB_RUN_ID"] = "baseline-run-fixture"
        harness.environment["RUNPOD_ROLE"] = "gpu-baseline"
        harness.write_script(
            harness.scripts / "runpod_baseline.sh",
            r"""
            [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 && -e /dev/fd/9 ]] || exit 97
            printf '%s\n' "${WANDB_RUN_ID}" "${RUNPOD_RUN_KEY}" \
                > "${HARNESS_ROOT}/baseline-identity"
            exit "${HARNESS_JOB_EXIT:-0}"
        """,
        )
        return harness

    def test_baseline_runs_in_tmux_and_retains_lease_for_local_guard(self):
        for values, state, exit_code in (
            ({}, "succeeded", 0),
            ({"HARNESS_JOB_EXIT": "7"}, "failed", 7),
            ({"HARNESS_TIMEOUT": "1"}, "timed_out", 124),
        ):
            with self.subTest(state=state):
                harness = self.harness()
                harness.environment.update(values)
                launched = harness.command("runpod_tmux_launch.sh", "baseline")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-baseline", launched.stdout)
                result = harness.run_worker()
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.read("waiting-for-local-guard"), "lease-held\n")
                self.assertFalse((harness.root / "shutdown-called").exists())
                self.assertIn("stage1-baseline", harness.read("lifecycle-calls"))
                if not values:
                    self.assertEqual(
                        harness.read("baseline-identity").splitlines(), ["baseline-run-fixture"] * 2
                    )

    def test_baseline_lease_conflict_never_terminates_another_job(self):
        harness = self.harness()
        harness.environment["HARNESS_LEASE_EXIT"] = "1"
        self.assertEqual(harness.command("runpod_tmux_launch.sh", "baseline").returncode, 0)
        result = harness.run_worker()
        self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
        self.assertFalse((harness.root / "baseline-identity").exists())
        self.assertFalse((harness.root / "waiting-for-local-guard").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())

    def test_baseline_requires_control_host_run_identity(self):
        harness = self.harness()
        harness.environment["WANDB_RUN_ID"] = "run-main-model"
        result = harness.command("runpod_tmux_launch.sh", "baseline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("allocated by the local workflow", result.stderr)

    def test_verification_forwards_only_allowlisted_regression_option(self):
        harness = self.harness()
        harness.write_script(
            harness.scripts / "runpod_verify_full_workflow.sh",
            r"""
            [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 && -e /dev/fd/9 ]] || exit 97
            printf '%s\n' "$@" > "${HARNESS_ROOT}/verification-arguments"
            """,
        )
        result = harness.command(
            "runpod_tmux_launch.sh", "verify-full-workflow", "--regression-only"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        worker = harness.run_worker()
        self.assertEqual(worker.returncode, 0, worker.stdout + worker.stderr)
        self.assertEqual(harness.read("verification-arguments"), "--regression-only\n")
        for role in ("baseline", "verify-full-workflow"):
            rejected = harness.command("runpod_tmux_launch.sh", role, "--unsupported")
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
