"""Verify parent identity and resume/evaluation routing without paid compute."""

from __future__ import annotations

import copy
import os
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / "tests/test_runpod_paired_results_control.py"))
PAIR = FIXTURE["PAIRED"]
BASE = FIXTURE["BASE"]
RELAUNCH = runpy.run_path(str(ROOT / "scripts/runpod_paired_relaunch.py"))


class PairedContractRelaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paired-relaunch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for directory in ("configs", "scripts", "src"):
            shutil.copytree(ROOT / directory, self.root / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.plan, self.records = FIXTURE["pair_fixtures"]()
        self.reader = BASE["FakeReader"](self.records)
        self.reader.bucket = "fixture-volume"
        self.reader.root = self.root

    def test_contract_rejects_mixed_seeds_ownership_and_missing_task(self):
        for change in (
            lambda p: p["children"].pop("absolute_direction"),
            lambda p: p.update(seed=43),
            lambda p: p["children"]["alpha_point"].update(run_id="unowned-run"),
            lambda p: p["children"]["alpha_point"].update(selection=None),
            lambda p: p.update(created_at="2026-10-14T12:00:00"),
        ):
            candidate = copy.deepcopy(self.plan)
            change(candidate)
            with self.assertRaises(ValueError):
                PAIR["validate_manifest"](candidate)

    def test_plan_write_does_not_replace_existing_parent(self):
        path = PAIR["store_plan"](self.root, self.plan)
        self.assertEqual(PAIR["load_manifest"](path), self.plan)
        changed = copy.deepcopy(self.plan)
        changed["created_at"] = "2026-10-14T13:00:00Z"
        with self.assertRaisesRegex(ValueError, "must not be overwritten"):
            PAIR["store_plan"](self.root, changed)
        self.assertEqual(PAIR["load_manifest"](path), self.plan)

    def test_resume_skips_completed_model_and_allows_unstarted_sibling(self):
        child = self.plan["children"]["absolute_direction"]["run_id"]
        self.reader.payloads = {k: v for k, v in self.records.items() if child not in k}
        calls = []
        RELAUNCH["check_children"](self.root, self.reader, self.plan, "resume", execute=lambda *a, **kw: calls.append(a))
        self.assertEqual(calls, [])

    def test_validate_requires_both_trained_and_resume_rejects_complete(self):
        with self.assertRaisesRegex(ValueError, "already have final"):
            RELAUNCH["check_children"](self.root, self.reader, self.plan, "resume")
        child = self.plan["children"]["absolute_direction"]["run_id"]
        self.reader.payloads = {k: v for k, v in self.records.items() if child not in k}
        with self.assertRaisesRegex(ValueError, "requires completed absolute_direction"):
            RELAUNCH["check_children"](self.root, self.reader, self.plan, "validate")

    def test_forced_validate_checks_both_best_checkpoints_with_own_config(self):
        calls = []
        RELAUNCH["check_children"](self.root, self.reader, self.plan, "validate", force=True,
                                   execute=lambda command, **kw: calls.append((command, kw)))
        self.assertEqual(len(calls), 2)
        for command, kw in calls:
            self.assertEqual(command[-1], "best")
            run = command[command.index("--run-id") + 1]
            child = next(c for c in self.plan["children"].values() if c["run_id"] == run)
            self.assertEqual(command[command.index("--config") + 1], str(self.root / child["config_path"]))
            self.assertEqual(kw["env"]["RUNPOD_TRAINING_SEED"], "42")

    def test_completed_pair_skips_paid_validation_unless_forced(self):
        execute = Mock()
        self.assertFalse(RELAUNCH["check_children"](self.root, self.reader, self.plan, "validate", execute=execute))
        execute.assert_not_called()

    def test_partial_model_uses_latest_checkpoint_not_sibling(self):
        child = self.plan["children"]["absolute_direction"]["run_id"]
        self.reader.payloads = {k: v for k, v in self.records.items() if not (
            child in k and ("training-completed" in k or "training-result" in k or "evaluation" in k)
        )}
        self.reader.payloads[f"lifecycle/runs/{child}/training.json"].update(state="interrupted", training_completed=False)
        calls = []
        RELAUNCH["check_children"](self.root, self.reader, self.plan, "resume",
                                   execute=lambda command, **kw: calls.append(command))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1], "latest")
        self.assertIn(child, calls[0])

    def test_cli_relaunch_preserves_parent_flags_and_calls_one_launcher(self):
        reader = Mock()
        reader.pair.return_value = self.plan
        reader.choose.return_value = self.plan["run_id"]
        launch = Mock(return_value=0)
        paths = {
            "runpod_paired_control.py": {"_bucket": lambda root: "fixture", "launch_existing": launch},
            "runpod_runs.py": {"RunRecords": lambda *a, **kw: reader, "validate_run": lambda value: value},
        }
        checks = Mock()
        with patch.dict(RELAUNCH["main"].__globals__, {"ROOT": self.root, "check_children": checks}), \
             patch("runpy.run_path", side_effect=lambda path: paths[Path(path).name]), \
             patch("subprocess.check_output", return_value="259200\n"), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(RELAUNCH["main"](["validate", "--force", "--maxRuntime", "72h", self.plan["run_id"]]), 0)
            self.assertEqual(os.environ["VALIDATION_RESUME"], "0")
            self.assertEqual(os.environ["VALIDATION_FORCE_RECOMPUTE"], "1")
            self.assertEqual(os.environ["RUNPOD_CLI_MAX_RUNTIME_SECONDS"], "259200")
            launch.assert_called_once_with(self.root, self.plan, "validate")
            checks.assert_called_once_with(self.root, reader, self.plan, "validate", force=True)

    def test_legacy_command_is_forwarded_without_dropping_options(self):
        reader = Mock()
        reader.pair.return_value = None
        paths = {
            "runpod_paired_control.py": {"_bucket": lambda root: "fixture"},
            "runpod_runs.py": {"RunRecords": lambda *a, **kw: reader, "validate_run": lambda value: value},
        }
        arguments = ["--no-resume", "--maxRuntime", "24h", "run-legacy"]
        with patch.dict(RELAUNCH["main"].__globals__, {"ROOT": self.root}), \
             patch("runpy.run_path", side_effect=lambda path: paths[Path(path).name]), \
             patch("subprocess.check_output", return_value="86400\n"), \
             patch.dict(os.environ, {}, clear=True), \
             patch("os.execvp", side_effect=SystemExit(0)) as execute:
            with self.assertRaises(SystemExit):
                RELAUNCH["main"](["validate", *arguments])
            execute.assert_called_once_with("bash", ["bash", str(self.root / "scripts/create_runpod_validation_pod.sh"), *arguments])


if __name__ == "__main__":
    unittest.main()
