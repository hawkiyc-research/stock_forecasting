"""Verify paired baseline ownership and failure boundaries without an ML runtime."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
CONTRACT = runpy.run_path(str(ROOT / "scripts/runpod_paired.py"))
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    RUNTIME = runpy.run_path(str(ROOT / "scripts/runpod_paired_baseline.py"))


class PairedBaselineRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paired-baseline-")
        self.addCleanup(temporary.cleanup)
        self.volume = Path(temporary.name).resolve()
        self.project = self.volume / "stock_forecasting"
        for folder in ("configs", "scripts", "src"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        selected = SELECTION["_build_selection"](argparse.Namespace(
            stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
            start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
            stocks=[], etfs=[], symbol_limit=None, feature_mode="combined"), self.project)
        self.plan = CONTRACT["build_plan"](self.project, selected, "a-signal64", seed=42,
                                            workflow="baseline", run_id="baseline-run-fixture")
        self.manifest = self.volume / CONTRACT["manifest_key"](self.plan["run_id"])
        CONTRACT["write_json"](self.manifest, self.plan)
        for child in self.plan["children"].values():
            CONTRACT["write_json"](self.volume / child["selection_path"], child["selection"])
        self.env = {
            "WANDB_RUN_ID": self.plan["run_id"], "RUNPOD_RUN_KEY": self.plan["run_id"],
            "RUNPOD_ROLE": "gpu-baseline", "RUNPOD_POD_ID": "fixture-pod",
            "RUNPOD_GPU_WORKFLOW_LEASE_HELD": "1", "RUNPOD_TMUX_LOG_FILE": "fixture.log",
            "MAX_RUNTIME_SECONDS": "86400", "RUNPOD_HARD_LIMIT_SECONDS": "90000",
            "RUNPOD_PAIRED_PLAN_PATH": str(self.manifest),
        }

    def run_fixture(self, fail_call=None):
        calls = []

        def execute(command, **kwargs):
            calls.append((command, kwargs))
            status = json.loads(self.manifest.with_name("pair-status.json").read_text())
            self.assertEqual(status["state"], "running")
            return subprocess.CompletedProcess(command, 17 if len(calls) == fail_call else 0)

        with contextlib.redirect_stdout(io.StringIO()):
            result = RUNTIME["run"](self.project, self.volume, self.manifest,
                                    environ=self.env, execute=execute)
        return result, calls, json.loads(self.manifest.with_name("pair-status.json").read_text())

    def test_two_tasks_share_parent_deadline_but_keep_own_selection(self):
        result, calls, status = self.run_fixture()
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(status["state"], "complete")
        self.assertTrue(all(item["state"] == "complete" for item in status["children"].values()))
        commands_by_task = {}
        for command, kwargs in calls:
            env = kwargs["env"]
            task = env["RUNPOD_PAIRED_CHILD_TASK"]
            commands_by_task.setdefault(task, []).append(command)
            self.assertEqual(env["WANDB_RUN_ID"], self.plan["run_id"])
            self.assertEqual(env["RUNPOD_RUN_KEY"], self.plan["run_id"])
            self.assertEqual(env["MAX_RUNTIME_SECONDS"], "86400")
            self.assertEqual(env["RUNPOD_HARD_LIMIT_SECONDS"], "90000")
            self.assertEqual(env["RUNPOD_TMUX_LOG_FILE"], "fixture.log")
            child = self.plan["children"][task]
            self.assertEqual(env["RUNPOD_CONFIG"], child["config_path"])
            self.assertEqual(env["RUNPOD_SELECTION_FILE"], str(self.volume / child["selection_path"]))
            self.assertEqual(env["RUNPOD_REMOTE_SELECTION_PATH"], env["RUNPOD_SELECTION_FILE"])
            self.assertEqual(env["RUNPOD_STAGE_CONFIG_SHA256"], child["selection"]["stage"]["config_sha256"])
            self.assertEqual(kwargs["cwd"], self.project)
        self.assertEqual(set(commands_by_task), {"alpha_point", "absolute_direction"})
        self.assertTrue(all(len(commands) == 2 for commands in commands_by_task.values()))

    def test_first_task_failure_never_starts_second_or_completes_parent(self):
        result, calls, status = self.run_fixture(fail_call=2)
        self.assertEqual(result, 17)
        self.assertEqual(len(calls), 2)
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["children"]["alpha_point"]["state"], "failed")
        self.assertEqual(status["children"]["absolute_direction"]["state"], "pending")

    def test_second_task_failure_keeps_first_result_without_false_completion(self):
        result, calls, status = self.run_fixture(fail_call=4)
        self.assertEqual(result, 17)
        self.assertEqual(len(calls), 4)
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["children"]["alpha_point"]["state"], "complete")
        self.assertEqual(status["children"]["absolute_direction"]["state"], "failed")

    def test_readiness_failure_prevents_model_build(self):
        result, calls, status = self.run_fixture(fail_call=1)
        self.assertEqual(result, 17)
        self.assertEqual(len(calls), 1)
        self.assertIn("verify_runpod_mounted_readiness.sh", calls[0][0][1])
        self.assertEqual(status["state"], "incomplete")

    def test_runtime_rejects_missing_outer_tmux_or_gpu_lease(self):
        for key in ("RUNPOD_GPU_WORKFLOW_LEASE_HELD", "RUNPOD_TMUX_LOG_FILE"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                RUNTIME["run"](self.project, self.volume, self.manifest,
                               environ={name: value for name, value in self.env.items() if name != key},
                               execute=lambda *args, **kwargs: self.fail("Must not start a subprocess"))

    def test_runtime_rejects_a_selection_from_another_task(self):
        alpha = self.plan["children"]["alpha_point"]
        direction = self.plan["children"]["absolute_direction"]
        CONTRACT["write_json"](self.volume / alpha["selection_path"], direction["selection"])
        with self.assertRaises(ValueError):
            RUNTIME["run"](self.project, self.volume, self.manifest, environ=self.env,
                           execute=lambda *args, **kwargs: self.fail("Must not start a subprocess"))
        status = json.loads(self.manifest.with_name("pair-status.json").read_text())
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["children"]["alpha_point"]["state"], "failed")

    def test_subprocess_launch_exception_does_not_leave_parent_running(self):
        def execute(command, **kwargs):
            raise OSError("Fixture executable unavailable")

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
            RUNTIME["run"](self.project, self.volume, self.manifest, environ=self.env, execute=execute)
        status = json.loads(self.manifest.with_name("pair-status.json").read_text())
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["children"]["alpha_point"], {"state": "failed", "error_type": "OSError"})
        self.assertEqual(status["children"]["absolute_direction"]["state"], "pending")


if __name__ == "__main__":
    unittest.main()
