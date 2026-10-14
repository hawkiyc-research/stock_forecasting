"""Exercise paired launch admission without remote resources or ML imports."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONTROL = runpy.run_path(str(ROOT / "scripts/runpod_paired_control.py"))
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))


class PairedAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paired-control-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for directory in ("configs", "scripts", "src"):
            shutil.copytree(ROOT / directory, self.root / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.original = SELECTION["_build_selection"](argparse.Namespace(
            stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
            start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
            stocks=[], etfs=[], symbol_limit=None, feature_mode="combined"), self.root)
        SELECTION["_activate_selection"](self.root, self.original)
        self.api = runpy.run_path(str(self.root / "scripts/runpod_paired_control.py"))
        self.plan, self.path, self.env = self.api["prepare_launch"](
            self.root, self.original, "a-signal64", 42)

    def test_pair_preflight_checks_both_task_baselines_and_both_readiness(self):
        commands = []

        def run(root, command, env, **kwargs):
            commands.append((command, env.copy()))
            return '{"complete":true}' if "runpod_baseline_cache.py" in command[1] else ""

        globals_ = self.api["preflight"].__globals__
        with patch.dict(globals_, {"_run": run, "_bucket": lambda root: "fixture"}), contextlib.redirect_stdout(io.StringIO()):
            result = self.api["preflight"](self.root, self.path)
        self.assertTrue(result["complete"])
        caches = [(cmd, env) for cmd, env in commands if "runpod_baseline_cache.py" in cmd[1]]
        ready = [(cmd, env) for cmd, env in commands if "verify_runpod_stage_readiness.sh" in cmd[1]]
        self.assertEqual(len(caches), 2)
        self.assertEqual(len(ready), 2)
        self.assertTrue(all(cmd[2] == "require" for cmd, _ in caches))
        self.assertTrue(all(cmd[-2:] == ["--gpu", "--local-selection"] for cmd, _ in ready))
        self.assertEqual({json.loads(Path(env["RUNPOD_SELECTION_FILE"]).read_text())["baseline_task"]["prediction_task"] for _, env in caches},
                         {"alpha_point", "absolute_direction"})

    def test_either_missing_baseline_blocks_train_before_readiness_or_paid_create(self):
        commands = []

        def run(root, command, env, **kwargs):
            commands.append(command)
            child = json.loads(Path(env["RUNPOD_SELECTION_FILE"]).read_text())
            if child["baseline_task"]["prediction_task"] == "absolute_direction":
                raise RuntimeError("Missing direction baseline")
            return '{"complete":true}'

        with patch.dict(self.api["preflight"].__globals__, {"_run": run, "_bucket": lambda root: "fixture"}):
            with self.assertRaisesRegex(RuntimeError, "Missing direction"):
                self.api["preflight"](self.root, self.path)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("runpod_baseline_cache.py" in command[1] for command in commands))

    def test_baseline_complete_requires_both_children(self):
        for completed in ((True, True), (True, False), (False, True), (False, False)):
            by_task = dict(zip(("alpha_point", "absolute_direction"), completed))

            def run(root, command, env, **kwargs):
                child = json.loads(Path(env["RUNPOD_SELECTION_FILE"]).read_text())
                return json.dumps({"complete": by_task[child["baseline_task"]["prediction_task"]]})

            with patch.dict(self.api["baseline_status"].__globals__, {"_run": run, "_bucket": lambda root: "fixture"}):
                result = self.api["baseline_status"](self.root, self.plan)
            self.assertEqual(result["complete"], all(completed))
            self.assertEqual(set(result["tasks"]), set(by_task))

    def test_status_reports_both_tasks_when_one_baseline_is_not_complete(self):
        helper = self.root / "scripts/runpod_baseline_readiness.py"
        helper.write_text(
            "import os,json\np=json.load(open(os.environ['RUNPOD_SELECTION_FILE']))\n"
            "incomplete=p['baseline_task']['prediction_task']=='absolute_direction'\n"
            "print('Baseline: NOT COMPLETE' if incomplete else 'Baseline: COMPLETE')\n"
            "print('Eligible windows: train=1, validation=1, test=1')\n"
            "print('Next: bash scripts/runpod_workflow.sh baseline --maxRuntime DURATION --gpuId GPU_ID') if incomplete else None\n"
            "raise SystemExit(int(incomplete))\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = self.api["status"](self.root, "a-signal64")
        self.assertEqual(code, 1)
        self.assertIn("[a-signal64 / alpha_point]", output.getvalue())
        self.assertIn("[a-signal64 / absolute_direction]", output.getvalue())
        self.assertIn("Baseline: COMPLETE", output.getvalue())
        self.assertIn("Baseline: NOT COMPLETE", output.getvalue())
        self.assertEqual(output.getvalue().count("Eligible windows:"), 2)
        self.assertEqual(output.getvalue().count("Next:"), 1)
        self.assertIn("Next: bash scripts/runpod_workflow.sh baseline --experiment a-signal64\n", output.getvalue())
        self.assertNotIn("DURATION", output.getvalue())
        self.assertNotIn("baseline --maxRuntime", output.getvalue())

    def test_parent_launch_keeps_data_identity_and_does_not_change_active_selection(self):
        pointer = (self.root / ".runpod/active-selection.json").read_bytes()
        for task, child in self.plan["children"].items():
            self.assertEqual(child["selection"]["dataset_request_sha256"], self.original["dataset_request_sha256"])
            self.assertEqual(child["selection"]["baseline_task"]["prediction_task"], task)
        self.assertEqual((self.root / ".runpod/active-selection.json").read_bytes(), pointer)
        self.assertEqual(self.env["WANDB_RUN_ID"], self.plan["run_id"])
        self.assertEqual(self.env["RUNPOD_POD_NAME"], "stock-forecasting-a-signal64-seed42")

    def test_paid_envelope_requires_canonical_plan_path_and_owned_primary_selection(self):
        primary = self.api["child_environments"](self.root, self.plan)[0][1]
        env = {**primary, **self.env}
        self.api["validate_launch"](self.plan, env)
        for key, value in (
            ("RUNPOD_PAIRED_PLAN_PATH", "/runpod-volume/lifecycle/runs/another/pair.json"),
            ("RUNPOD_SELECTION_SHA256", "0" * 64),
            ("RUNPOD_CONFIG", "configs/experiments/b_alpha_point64.yaml"),
            ("RUNPOD_STAGE_CONFIG_SHA256", "1" * 64),
            ("RUNPOD_DATASET_REQUEST_SHA256", "2" * 64),
            ("WANDB_RUN_ID", "run-another"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api["validate_launch"](self.plan, {**env, key: value})

    def test_custom_mount_and_either_owned_primary_are_validated_without_guessing(self):
        for _task, primary in self.api["child_environments"](self.root, self.plan):
            env = {**primary, **self.env,
                   "RUNPOD_VOLUME_MOUNT_PATH": "/mnt/task-volume",
                   "RUNPOD_PAIRED_PLAN_PATH": f"/mnt/task-volume/lifecycle/runs/{self.plan['run_id']}/pair.json"}
            self.api["validate_launch"](self.plan, env)

    def test_workflow_routes_paired_baseline_once_and_leaves_single_task_supported(self):
        helper = self.root / "scripts/runpod_paired_control.py"
        helper.write_text("import json,sys,os; print(json.dumps({'args':sys.argv[1:], 'runtime':os.environ['RUNPOD_CLI_MAX_RUNTIME_SECONDS']}))\n")
        result = subprocess.run(["bash", str(self.root / "scripts/runpod_workflow.sh"),
                                 "baseline", "--experiment", "a-signal64", "--maxRuntime", "24h",
                                 "--gpuId", "NVIDIA GeForce RTX 4090"], capture_output=True, text=True, timeout=10, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"], ["launch-baseline", "--experiment", "a-signal64"])
        self.assertEqual(payload["runtime"], "86400")

    def test_baseline_creator_skips_both_complete_or_creates_one_pod_for_missing_tasks(self):
        env_file = self.root / ".env"
        env_file.write_text("RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_ACCESS_KEY_ID=fixture-key\n"
                            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture-secret\nRUNPOD_S3_REGION=fixture-region\n")
        env_file.chmod(0o600)
        scripts = self.root / "scripts"
        (scripts / "verify_runpod_stage_readiness.sh").write_text("#!/bin/bash\nexit 0\n")
        (scripts / "runpod_s3_project.sh").write_text("#!/bin/bash\nexit 0\n")
        (scripts / "runpod_baseline_cache.py").write_text(
            "import json,os\np=json.load(open(os.environ['RUNPOD_SELECTION_FILE']))\n"
            "task=p['baseline_task']['prediction_task']\n"
            "print(json.dumps({'complete':task not in os.environ.get('FIXTURE_MISSING','').split(',')}))\n")
        transport = (
            "import json,os,pathlib,sys\n"
            "if sys.argv[1:3]==['pod','list']: print('[]')\n"
            "elif sys.argv[1:3]==['pod','create-gpu']:\n"
            " p=json.load(sys.stdin); p['id']='fixture-pod'; "
            "pathlib.Path(os.environ['FIXTURE_OUTPUT']).write_text(json.dumps(p)); print(json.dumps(p))\n"
            "else: raise SystemExit(93)\n")
        (scripts / "runpodctl_project.sh").write_text(
            "#!/bin/bash\nexec python3 -c " + shlex.quote(transport) + ' "$@"\n')
        (scripts / "launch_runpod_guard.sh").write_text("#!/bin/bash\nprintf '12345\\n'\n")
        base_env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("RUNPOD_", "WANDB_", "VALIDATION_", "FIN_TS_"))}
        base_env.update(RUNPOD_ENV_FILE=str(env_file), RUNPOD_GUARD_LOG_DIR=str(self.root / "guards"))
        for missing in ("", "alpha_point", "absolute_direction", "alpha_point,absolute_direction"):
            output = self.root / ("pod-" + (missing or "none") + ".json")
            with patch.dict(os.environ, {**base_env, "FIXTURE_MISSING": missing, "FIXTURE_OUTPUT": str(output)}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.api["launch_baseline"](self.root, "a-signal64"), 0)
            self.assertEqual(output.exists(), bool(missing))
            if output.exists():
                payload = json.loads(output.read_text())
                self.assertEqual(payload["env"]["RUNPOD_ROLE"], "gpu-baseline")
                run_id = payload["env"]["WANDB_RUN_ID"]
                self.assertTrue(run_id.startswith("baseline-run-"))
                manifest = json.loads((self.root / ".runpod/paired" / f"{run_id}.json").read_text())
                self.assertEqual(len(manifest["children"]), 2)
                self.assertEqual(payload["name"], "stock-forecasting-a-signal64-baseline")


if __name__ == "__main__":
    unittest.main()
