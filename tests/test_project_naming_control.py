"""Dependency-free project naming and non-invalidating deployment regressions."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "stock_forecasting"
SLUG = NAME.replace("_", "-")
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
BASELINE = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
CONTENT = runpy.run_path(str(ROOT / "src/stock_forecasting/data/content_identity.py"))
PROBE = runpy.run_path(str(ROOT / "scripts/runpod_probe_lifecycle.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))


class ProjectNamingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-names-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        for folder in ("scripts", "src", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pyproject.toml", self.project / "pyproject.toml")
        env_file = self.project / ".env"
        env_file.write_text("RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        env_file.chmod(0o600)
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "python3").symlink_to(sys.executable)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "FIN_TS_", "WANDB_", "VALIDATION_", "AWS_"))
            and key not in {"PROJECT_ROOT", "DATA_ROOT", "SAVED_MODEL_ROOT", "RESUME_CHECKPOINT"}
        }
        self.environment.update({
            "PATH": f"{binary}:{os.defpath}", "RUNPOD_ENV_FILE": str(env_file),
            "RUNPOD_TEST_MODE": "1", "RUNPOD_TEST_READINESS_READY": "1",
        })
        # Both transports fail closed even if a regression bypasses dry-run handling.
        for name in ("runpod_s3_project.sh", "runpodctl_project.sh"):
            (self.project / "scripts" / name).write_text(
                "#!/usr/bin/env bash\necho 'Cloud access is forbidden in this test' >&2\nexit 93\n"
            )
        arguments = SELECTION["build_parser"]().parse_args([
            "create", "--project-root", str(self.project), "--stage", "stage2",
            "--data-profile", "us_tw_eodhd", "--start", "2021-01-01",
            "--end", "2026-06-01", "--universe", "all",
        ])
        self.selection = SELECTION["_build_selection"](arguments, self.project)
        SELECTION["_activate_selection"](self.project, self.selection)

    def command(self, *arguments, env=None):
        return subprocess.run(
            ["/bin/bash", str(self.project / "scripts/runpod_workflow.sh"), *arguments],
            env={**self.environment, **(env or {})}, cwd=self.project,
            capture_output=True, text=True, timeout=30,
        )

    def test_package_and_every_console_entry_use_the_project_name(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
        self.assertEqual(project["project"]["name"], NAME)
        self.assertEqual(project["tool"]["poetry"]["packages"], [
            {"include": NAME, "from": "src"},
        ])
        entries = project["project"]["scripts"]
        self.assertEqual(len(entries), 14)
        for name, target in entries.items():
            self.assertTrue(name.startswith(SLUG + "-"), name)
            self.assertTrue(target.startswith(NAME + "."), target)

    def test_all_gpu_launches_emit_project_names_and_preserve_dataset_paths(self):
        for workflow, role in (("train", "train"), ("validate", "validation"),
                               ("baseline", "baseline")):
            with self.subTest(workflow=workflow):
                result = self.command(workflow, "--maxRuntime", "72h")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                command = shlex.split(result.stdout.strip().removeprefix("DRY RUN:"))
                self.assertEqual(command[command.index("--name") + 1], SLUG + "-" + role)
                env = json.loads(command[command.index("--env") + 1])
                self.assertEqual(env["PROJECT_ROOT"], "/runpod-volume/" + NAME)
                self.assertEqual(env["WANDB_PROJECT"], NAME)
                self.assertEqual(env["DATA_ROOT"], "/runpod-volume/datasets/"
                                 + self.selection["dataset_request_sha256"])
                self.assertEqual(env["RUNPOD_REQUESTED_RUNTIME_SECONDS"], "259200")
                self.assertEqual(env["RUNPOD_HARD_LIMIT_SECONDS"], "262800")
                self.assertEqual(
                    env["MAX_RUNTIME_SECONDS"],
                    "262800" if workflow == "train" else "259200",
                )

    def test_cpu_launch_name_changes_without_changing_preparation_identity(self):
        result = self.command("cpu", "prepare", "--max-api-calls", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line)
        self.assertEqual(command[command.index("--name") + 1], SLUG + "-cpu-prepare")
        env = json.loads(command[command.index("--env") + 1])
        self.assertEqual(env["RUNPOD_DATASET_REQUEST_SHA256"],
                         self.selection["dataset_request_sha256"])

    def test_cpu_launch_accepts_only_rest_v2_vcpu_counts(self):
        for count in ("2", "4", "8", "16", "32"):
            with self.subTest(count=count):
                result = self.command(
                    "cpu", "prepare", "--max-api-calls", "1", "--cpuNumber", count
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("https://api.runpod.io/v2/pods", result.stdout)
        for count in ("1", "3", "6", "33"):
            with self.subTest(count=count):
                result = self.command(
                    "cpu", "prepare", "--max-api-calls", "1", "--cpuNumber", count
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--cpuNumber must be one of: 2, 4, 8, 16, 32", result.stderr)
                self.assertNotIn("Cloud access", result.stderr)

    def test_stale_resource_name_overrides_are_rejected_before_cloud_access(self):
        for workflow, variable in (("train", "RUNPOD_POD_NAME"),
                                   ("baseline", "RUNPOD_POD_NAME"),
                                   ("validate", "RUNPOD_POD_NAME")):
            result = self.command(workflow, env={variable: "another-project-train"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must use stock-forecasting", result.stderr)
            self.assertNotIn("Cloud access", result.stderr)
        accepted = self.command("train", env={"RUNPOD_POD_NAME": SLUG + "-experiment-b"})
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn(SLUG + "-experiment-b", accepted.stdout)

    def test_wandb_defaults_match_the_project_name(self):
        tree = ast.parse((ROOT / "src" / NAME / "config.py").read_text())
        config = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "WandbConfig")
        project = next(node for node in config.body
                       if isinstance(node, ast.AnnAssign)
                       and isinstance(node.target, ast.Name) and node.target.id == "project")
        self.assertEqual(ast.literal_eval(project.value), NAME)
        paths = sorted((ROOT / "configs").glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                defaults = re.findall(r"(?m)^  project:[ \t]*(\S+)[ \t]*$", path.read_text())
                self.assertEqual(len(defaults), 1)
                self.assertIn(defaults[0], (NAME, "${WANDB_PROJECT:-" + NAME + "}"))

    def test_new_tmux_sessions_use_the_project_prefix(self):
        script = (ROOT / "scripts/runpod_tmux_launch.sh").read_text()
        sessions = re.findall(r"(?m)^[ \t]*SESSION_NAME=(\S+)[ \t]*$", script)
        roles = ("cpu-prepare", "cpu-finalize", "train", "baseline", "validation", "probe-scales")
        self.assertEqual(sorted(sessions), sorted(SLUG + "-" + role for role in roles))

    def test_gpu_workflows_reuse_the_interpreter_and_import_namespace(self):
        workflows = {
            "runpod_entrypoint.sh": ("${PROJECT_VENV}/bin/python", "runpod.supervisor"),
            "runpod_train_then_validate.sh": ("${PROJECT_VENV}/bin/python", "cli.train"),
            "runpod_validation.sh": ("${PROJECT_VENV}/bin/python", "cli.validate_benchmarks"),
            "runpod_baseline.sh": ("${PROJECT_PYTHON}", "cli.build_baselines"),
            "runpod_probe_scales.sh": ("${PROJECT_ROOT}/.venv/bin/python", "cli.probe_scales"),
        }
        for script, (interpreter, module) in workflows.items():
            with self.subTest(script=script):
                content = (ROOT / "scripts" / script).read_text()
                self.assertIn(f'"{interpreter}" -m {NAME}.{module}', content)
                self.assertIn('/stock_forecasting}"', content)
                self.assertNotRegex(content, r'(?m)^\s*[^#\n]*\brun stock-forecasting-')
                self.assertNotRegex(content, r'(?m)^\s*[^#\n]*-m (?:venv|pip)\b')

    def test_names_and_launch_wrappers_do_not_invalidate_data_baseline_or_resume(self):
        baseline = BASELINE["baseline_contract"](self.project, self.selection)
        content = CONTENT["code_content_identity"](package_root=self.project / "src" / NAME)
        selection_digest = self.selection["dataset_request_sha256"]
        tree = ast.parse((self.project / "src" / NAME / "run_contract.py").read_text())
        paths = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name)
                         and target.id == "TRAINING_IMPLEMENTATION_PATHS" for target in node.targets
                     ))
        def resume_files():
            package = self.project / "src" / NAME
            return {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in paths}
        resume = resume_files()
        for relative in ("pyproject.toml", "scripts/create_runpod_pod.sh",
                         "scripts/runpod_cpu_prepare.sh", "scripts/runpod_tmux_launch.sh"):
            path = self.project / relative
            updated_source = path.read_text().replace(SLUG, "fixture-renamed-project")
            if relative == "pyproject.toml":
                updated_source = updated_source.replace(
                    f'name = "{NAME}"', 'name = "fixture_renamed_project"'
                )
            path.write_text(updated_source)
        self.assertEqual(BASELINE["baseline_contract"](self.project, self.selection), baseline)
        self.assertEqual(CONTENT["code_content_identity"](
            package_root=self.project / "src" / NAME), content)
        request_core = SELECTION["_dataset_request_core"](self.selection["dataset_request"])
        self.assertEqual(SELECTION["_payload_sha256"](request_core), selection_digest)
        self.assertEqual(resume_files(), resume)

    def test_existing_pods_remain_identifiable_without_requiring_a_rename(self):
        for name in (SLUG + "-train", "historical-display-name"):
            pod = {"id": "fixture-pod", "name": name, "networkVolumeId": "fixture-volume",
                   "env": {"RUNPOD_ROLE": "gpu-train", "NETWORK_VOLUME_ROOT": "/runpod-volume",
                           "PROJECT_ROOT": "/runpod-volume/" + NAME}}
            self.assertTrue(RECOVERY["project_pod"](pod, "fixture-volume"))

    def test_probe_guard_reads_old_paths_but_new_identity_only_writes_project_paths(self):
        canonical = PROBE["identity"](
            "/runpod-volume", "fixture-pod", "run-fixture", "launch-fixture"
        )
        self.assertIn(SLUG + "-probe-scales", str(canonical["log_path"]))
        for session in (PROBE["PROBE_SESSION_NAME"], PROBE["LEGACY_PROBE_SESSION_NAME"]):
            paths = PROBE["identity"]("/runpod-volume", "fixture-pod", "run-fixture",
                                       "launch-fixture", session=session)
            payload = {"schema_version": 1, "kind": "representation-scale-probe",
                       "pod_id": "fixture-pod", "owner_run_id": "run-fixture",
                       "launch_id": "launch-fixture", "gpu_lease_acquired": True,
                       "state": "succeeded", "exit_code": 0,
                       "generated_at": "2026-09-22T00:00:00+00:00",
                       "log_path": str(paths["log_path"]), "status_path": str(paths["status_path"])}
            self.assertEqual(PROBE["validate"](payload, "/runpod-volume", "fixture-pod",
                                               "run-fixture"), "succeeded")
            payload["status_path"] = "/runpod-volume/unrelated/status.json"
            with self.assertRaisesRegex(ValueError, "identity or paths"):
                PROBE["validate"](payload, "/runpod-volume", "fixture-pod", "run-fixture")

    def test_cpu_log_reader_preserves_historical_paths_and_rejects_path_escape(self):
        script = (ROOT / "scripts/download_runpod_cpu_logs.sh").read_text()
        source = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        marker = self.root / "cpu-preparation.json"
        for session in (SLUG + "-cpu-prepare", "fin-ts-cpu-prepare", "../../other"):
            path = f"/runpod-volume/logs/tmux/{session}/launch-fixture"
            marker.write_text(json.dumps({
                "state": "ready", "launch_id": "launch-fixture", "log_path": path,
            }))
            result = subprocess.run(
                [sys.executable, "-", str(marker)], input=source,
                capture_output=True, text=True, timeout=10,
            )
            if session.startswith("../"):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unexpected CPU log_path", result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip().split("\t")[-1], path)


if __name__ == "__main__":
    unittest.main()
