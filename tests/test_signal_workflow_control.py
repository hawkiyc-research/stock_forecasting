"""Dependency-free CLI checks for independent-task baseline launches."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SignalWorkflowControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="signal-control-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for directory in ("scripts", "configs", "src"):
            shutil.copytree(
                ROOT / directory,
                self.root / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        self.selection = runpy.run_path(str(self.root / "scripts/runpod_selection.py"))
        arguments = self.selection["build_parser"]().parse_args(
            [
                "create",
                "--project-root",
                str(self.root),
                "--stage",
                "stage2",
                "--data-profile",
                "us_tw_eodhd",
                "--start",
                "2021-01-01",
                "--end",
                "2026-06-01",
                "--universe",
                "all",
            ]
        )
        self.original = self.selection["_build_selection"](arguments, self.root)
        self.selection["_activate_selection"](self.root, self.original)
        for name, event in (
            ("publish_runpod_selection.sh", "published"),
            ("create_runpod_pod.sh", "created"),
        ):
            (self.root / "scripts" / name).write_text(
                "#!/usr/bin/env bash\nset -eu\n" + "printf '%s\\n' " + event + " >> events.txt\n"
            )
        # A regression must never reach a live remote transport from this fixture.
        for name in ("runpodctl_project.sh", "runpod_s3_project.sh"):
            (self.root / "scripts" / name).write_text("#!/bin/bash\nexit 99\n")
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "FIN_TS_"))
        }

    def command(self, *args):
        return subprocess.run(
            ["bash", "scripts/runpod_workflow.sh", *args],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def active(self):
        pointer = json.loads((self.root / ".runpod/active-selection.json").read_text())
        return json.loads(
            (self.root / ".runpod/selections" / (pointer["selection_id"] + ".json")).read_text()
        )

    def test_baseline_selects_and_publishes_task_before_creation(self):
        result = self.command(
            "baseline",
            "--experiment",
            "a-alpha-point64",
            "--maxRuntime",
            "24h",
            "--gpuId",
            "NVIDIA GeForce RTX 4090",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.root / "events.txt").read_text().splitlines(), ["published", "created"]
        )
        self.assertEqual(
            self.active()["baseline_task"],
            {"prediction_task": "alpha_point", "point_loss_beta": 1.0},
        )
        self.assertEqual(
            self.active()["dataset_request_sha256"], self.original["dataset_request_sha256"]
        )

    def test_no_experiment_preserves_original_baseline_launch(self):
        result = self.command("baseline", "--maxRuntime", "24h")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "events.txt").read_text().splitlines(), ["created"])
        self.assertNotIn("baseline_task", self.active())

    def test_invalid_or_repeated_baseline_experiment_never_launches(self):
        for arguments in (
            ("--experiment", "unknown"),
            ("--experiment", "a-alpha-point64", "--experiment", "a-absolute-direction64"),
            ("--seed", "42"),
            ("--experiment",),
        ):
            with self.subTest(arguments=arguments):
                result = self.command("baseline", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "events.txt").exists())
                self.assertEqual(self.active()["selection_id"], self.original["selection_id"])


if __name__ == "__main__":
    unittest.main()
