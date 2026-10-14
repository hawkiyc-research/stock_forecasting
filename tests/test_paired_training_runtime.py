"""Exercise paired orchestration without importing torch or starting paid Pods."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_training_runtime", ROOT / "scripts/runpod_paired_train.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
PAIR = runpy.run_path(str(ROOT / "scripts/runpod_paired.py"))
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))


class PairedTrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paired-runtime-")
        self.addCleanup(temporary.cleanup)
        self.volume = Path(temporary.name).resolve()
        self.project = self.volume / "stock_forecasting"
        self.project.mkdir()
        shutil.copytree(ROOT / "configs", self.project / "configs")
        (self.project / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        selected = SELECTION["_build_selection"](
            argparse.Namespace(stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
                               start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
                               stocks=[], etfs=[], symbol_limit=None, feature_mode="combined"),
            self.project,
        )
        self.plan = PAIR["build_plan"](self.project, selected, "a-signal64", seed=43, run_id="run-pair-fixture")
        self.path = self.volume / "lifecycle/runs/run-pair-fixture/pair.json"
        self.write(self.path, self.plan)
        for child in self.plan["children"].values():
            self.write(self.volume / child["selection_path"], child["selection"])
        self.environment = {
            "NETWORK_VOLUME_ROOT": str(self.volume), "PROJECT_ROOT": str(self.project),
            "RUNPOD_RUN_KEY": self.plan["run_id"], "WANDB_RUN_ID": self.plan["run_id"],
            "RUNPOD_LAUNCH_ID": "launch-fixture", "RUNPOD_POD_ID": "pod-fixture",
            "RUNPOD_SCOPED_LIFECYCLE": "1", "RUNPOD_PAIRED_PLAN_PATH": str(self.path),
            "MAX_RUNTIME_SECONDS": "7200", "RUNPOD_HARD_LIMIT_SECONDS": "10800",
            "RUNPOD_SHUTDOWN_DIR": str(self.volume / "logs/run-pair-fixture/launcher/launch-fixture"),
            "RUNPOD_TRAINING_COMPLETED_MARKER": str(self.volume / "logs/run-pair-fixture/launcher/launch-fixture/training-completed.json"),
            "RESUME_CHECKPOINT": "/stale-sibling/checkpoint-000999", "VALIDATION_CHECKPOINT": "/stale-evaluation",
        }
        self.calls = []
        self.lifecycles = []

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def complete(self, task, *, holdout=True):
        child = self.plan["children"][task]
        run_id = child["run_id"]
        manifest = self.volume / "savedModel" / run_id / "run-manifest.json"
        self.write(manifest, {"run_id": run_id, "run_key": run_id,
                              "training_resume_contract": {"model": {"prediction_task": task}},
                              "selection_provenance": {"selection_id": child["selection"]["selection_id"]}})
        self.write(self.volume / "lifecycle/runs" / run_id / "training-completed.json", {
            "schema_version": "1.0", "kind": "stage1-training-completion", "state": "ready",
            "training_completed": True, "run_id": run_id, "run_key": run_id,
            "run_manifest_path": str(manifest), "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        })
        if holdout:
            self.write(self.volume / "evaluations" / run_id / "validation-benchmark.json", {
                "run_id": run_id, "prediction_task": task, "state": "ready", "evaluation_split": "test",
                "completed_at": "2026-10-14T00:00:00+00:00",
            })

    def subprocess(self, command, **kwargs):
        env = kwargs["env"]
        if Path(command[1]).name == "verify_runpod_mounted_readiness.sh":
            self.calls.append(("readiness", env["WANDB_RUN_ID"], env.copy()))
            return subprocess.CompletedProcess(command, 0)
        script = Path(command[1]).name
        self.assertIn(script, ("runpod_train_then_validate.sh", "runpod_validation.sh"))
        task = next(task for task, child in self.plan["children"].items() if child["run_id"] == env["WANDB_RUN_ID"])
        self.assertFalse(Path(self.environment["RUNPOD_TRAINING_COMPLETED_MARKER"]).exists())
        self.assertNotIn("VALIDATION_CHECKPOINT", env)
        self.calls.append((script, task, env.copy()))
        self.complete(task)
        return subprocess.CompletedProcess(command, 0)

    def lifecycle(self, project, volume, env, phase, state, **kwargs):
        self.lifecycles.append((env["WANDB_RUN_ID"], phase, state, kwargs))

    def run_pair(self, **kwargs):
        with patch.object(RUNTIME.subprocess, "run", side_effect=self.subprocess), patch.object(RUNTIME, "lifecycle", side_effect=self.lifecycle):
            return RUNTIME.run_pair(self.path, environment=self.environment, **kwargs)

    def test_one_outer_run_executes_both_models_with_separate_state_then_completes(self):
        self.assertEqual(self.run_pair(), 0)
        work = [call for call in self.calls if call[0] != "readiness"]
        self.assertEqual([call[1] for call in work], ["alpha_point", "absolute_direction"])
        self.assertEqual(len([call for call in self.calls if call[0] == "readiness"]), 2)
        for _, task, env in work:
            self.assertEqual(env["WANDB_RUN_ID"], self.plan["children"][task]["run_id"])
            self.assertEqual(env["RUNPOD_TRAINING_SEED"], "43")
            self.assertEqual(env["MAX_RUNTIME_SECONDS"], "7200")
            self.assertNotIn("RESUME_CHECKPOINT", env)
        self.assertNotEqual(work[0][2]["RUNPOD_CONFIG"], work[1][2]["RUNPOD_CONFIG"])
        self.assertTrue(Path(self.environment["RUNPOD_TRAINING_COMPLETED_MARKER"]).exists())
        self.assertEqual(self.lifecycles[-2][0:3], (self.plan["run_id"], "training", "finalizing"))
        self.assertEqual(self.lifecycles[-1][0:3], (self.plan["run_id"], "validation", "finalizing"))

    def test_second_task_failure_never_marks_parent_complete(self):
        original = self.subprocess
        def fail_second(command, **kwargs):
            if Path(command[1]).name == "runpod_train_then_validate.sh" and kwargs["env"]["WANDB_RUN_ID"].endswith("-direction"):
                raise subprocess.CalledProcessError(7, command)
            return original(command, **kwargs)
        with patch.object(RUNTIME.subprocess, "run", side_effect=fail_second), patch.object(RUNTIME, "lifecycle", side_effect=self.lifecycle):
            with self.assertRaises(subprocess.CalledProcessError):
                RUNTIME.run_pair(self.path, environment=self.environment)
        self.assertFalse(Path(self.environment["RUNPOD_TRAINING_COMPLETED_MARKER"]).exists())
        state = json.loads(self.path.with_name("pair-status.json").read_text())
        self.assertEqual(state["children"]["alpha_point"]["state"], "complete")
        self.assertEqual(state["children"]["absolute_direction"]["state"], "failed")
        self.assertFalse(any(row[0] == self.plan["run_id"] for row in self.lifecycles))

    def test_resume_skips_completed_sibling_and_uses_only_child_checkpoint(self):
        self.complete("alpha_point")
        checkpoint = str(self.volume / "savedModel" / self.plan["children"]["absolute_direction"]["run_id"] / "checkpoint-000123")
        with patch.object(RUNTIME, "resume_checkpoint", return_value=checkpoint):
            self.run_pair(action="resume")
        work = [call for call in self.calls if call[0] != "readiness"]
        self.assertEqual([call[1] for call in work], ["absolute_direction"])
        self.assertEqual(work[0][2]["RESUME_CHECKPOINT"], checkpoint)

    def test_trained_child_only_retries_holdout(self):
        self.complete("alpha_point", holdout=False)
        self.run_pair(action="resume")
        work = [call for call in self.calls if call[0] != "readiness"]
        self.assertEqual(work[0][0], "runpod_validation.sh")
        self.assertEqual(work[1][0], "runpod_train_then_validate.sh")

    def test_validation_only_requires_both_training_completions(self):
        self.complete("alpha_point", holdout=False)
        self.complete("absolute_direction", holdout=False)
        self.run_pair(action="validate")
        work = [call for call in self.calls if call[0] != "readiness"]
        self.assertEqual([call[0] for call in work], ["runpod_validation.sh"] * 2)

    def test_missing_sibling_training_blocks_validation_before_any_work(self):
        self.complete("alpha_point", holdout=False)
        with self.assertRaisesRegex(ValueError, "completed absolute_direction training"):
            self.run_pair(action="validate")
        self.assertEqual(self.calls, [])

    def test_forced_validation_keeps_policy_environment_without_overriding_flags(self):
        for task in self.plan["children"]:
            self.complete(task)
        self.environment.update(VALIDATION_RESUME="0", VALIDATION_FORCE_RECOMPUTE="1")
        original = self.subprocess
        def verify_flags(command, **kwargs):
            if Path(command[1]).name == "runpod_validation.sh":
                self.assertNotIn("--resume", command)
                self.assertEqual(kwargs["env"]["VALIDATION_RESUME"], "0")
                self.assertEqual(kwargs["env"]["VALIDATION_FORCE_RECOMPUTE"], "1")
            return original(command, **kwargs)
        with patch.object(RUNTIME.subprocess, "run", side_effect=verify_flags), patch.object(RUNTIME, "lifecycle", side_effect=self.lifecycle):
            RUNTIME.run_pair(self.path, environment=self.environment, action="validate")
        self.assertEqual(len([call for call in self.calls if call[0] == "runpod_validation.sh"]), 2)

    def test_force_failure_then_resume_retries_old_report_and_finalizes_new_success(self):
        for task in self.plan["children"]:
            self.complete(task)
        alpha = self.plan["children"]["alpha_point"]
        self.environment["VALIDATION_FORCE_RECOMPUTE"] = "1"

        def persist_lifecycle(project, volume, env, phase, state, **kwargs):
            self.lifecycle(project, volume, env, phase, state, **kwargs)
            self.write(volume / "lifecycle/runs" / env["WANDB_RUN_ID"] / (phase + ".json"), {
                "state": state, "wandb_run_id": env["WANDB_RUN_ID"], "kind": "stage1-" + phase,
                "generated_at": "2026-10-16T00:00:02+00:00" if state == "ready" else "2026-10-15T00:00:00+00:00",
            })

        def fail_before_rewriting_report(command, **kwargs):
            if Path(command[1]).name == "runpod_validation.sh":
                raise subprocess.CalledProcessError(7, command)
            return self.subprocess(command, **kwargs)

        with patch.object(RUNTIME.subprocess, "run", side_effect=fail_before_rewriting_report), patch.object(RUNTIME, "lifecycle", side_effect=persist_lifecycle):
            with self.assertRaises(subprocess.CalledProcessError):
                RUNTIME.run_pair(self.path, environment=self.environment, action="validate")
        self.assertEqual(RUNTIME.completed_child(alpha, self.volume, self.project), (True, False))
        self.assertFalse(Path(self.environment["RUNPOD_TRAINING_COMPLETED_MARKER"]).exists())

        self.calls.clear()
        self.environment["VALIDATION_FORCE_RECOMPUTE"] = "0"

        def successful_deferred_validation(command, **kwargs):
            result = self.subprocess(command, **kwargs)
            if Path(command[1]).name == "runpod_validation.sh":
                report_path = self.volume / "evaluations" / alpha["run_id"] / "validation-benchmark.json"
                report = json.loads(report_path.read_text())
                report["completed_at"] = "2026-10-16T00:00:00+00:00"
                self.write(report_path, report)
                self.write(self.volume / "lifecycle/runs" / alpha["run_id"] / "validation.json", {
                    "state": "finalizing", "generated_at": "2026-10-16T00:00:01+00:00",
                    "kind": "stage1-validation", "wandb_run_id": alpha["run_id"],
                })
                # A future startup cannot trust this marker without the
                # successful subprocess boundary retained by its current owner.
                self.assertEqual(RUNTIME.completed_child(alpha, self.volume, self.project), (True, False))
            return result

        with patch.object(RUNTIME.subprocess, "run", side_effect=successful_deferred_validation), patch.object(RUNTIME, "lifecycle", side_effect=persist_lifecycle):
            RUNTIME.run_pair(self.path, environment=self.environment, action="resume")
        work = [call for call in self.calls if call[0] != "readiness"]
        self.assertEqual([(call[0], call[1]) for call in work], [("runpod_validation.sh", "alpha_point")])
        self.assertEqual(RUNTIME.completed_child(alpha, self.volume, self.project), (True, True))
        self.assertTrue(Path(self.environment["RUNPOD_TRAINING_COMPLETED_MARKER"]).exists())

    def test_validation_scores_do_not_count_as_final_holdout(self):
        self.complete("alpha_point")
        child = self.plan["children"]["alpha_point"]
        report_path = self.volume / "evaluations" / child["run_id"] / "validation-benchmark.json"
        report = json.loads(report_path.read_text())
        report["evaluation_split"] = "validation"
        self.write(report_path, report)
        self.assertEqual(RUNTIME.completed_child(child, self.volume, self.project), (True, False))

    def test_source_selection_mismatch_fails_before_any_model(self):
        child = self.plan["children"]["absolute_direction"]
        modified = dict(child["selection"], created_at="2000-01-01T00:00:00+00:00")
        self.write(self.volume / child["selection_path"], modified)
        with self.assertRaisesRegex(ValueError, "Mounted child selection"):
            self.run_pair()
        self.assertEqual(self.calls, [])

    def test_sibling_run_lease_contention_prevents_both_models_starting(self):
        child = self.plan["children"]["absolute_direction"]
        lock_path = self.volume / "lifecycle/runs" / child["run_id"] / "gpu-workflow.lock"
        lock_path.parent.mkdir(parents=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already has a workflow owner"):
                self.run_pair()
        self.assertFalse(any(call[0] != "readiness" for call in self.calls))

    def test_soft_stop_between_children_does_not_start_another_model(self):
        original = self.subprocess
        def finish_then_stop(command, **kwargs):
            result = original(command, **kwargs)
            if Path(command[1]).name == "runpod_train_then_validate.sh":
                self.write(self.volume / "lifecycle/runs" / self.plan["run_id"] / "pods/pod-fixture/stop-request.json", {
                    "schema_version": 1, "kind": "runtime-stop-request", "pod_id": "pod-fixture", "run_id": self.plan["run_id"],
                })
            return result
        with patch.object(RUNTIME.subprocess, "run", side_effect=finish_then_stop), patch.object(RUNTIME, "lifecycle", side_effect=self.lifecycle):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                RUNTIME.run_pair(self.path, environment=self.environment)
        self.assertEqual(error.exception.returncode, 124)
        self.assertEqual(len([call for call in self.calls if call[0] == "runpod_train_then_validate.sh"]), 1)

    def test_pid1_import_and_entrypoint_preserve_single_model_fallback(self):
        importer = runpy.run_path(str(ROOT / "scripts/runpod_reexec_with_pid1_env.py"))
        self.assertIn("RUNPOD_PAIRED_PLAN_PATH", importer["ALLOWED_NAMES"])
        self.assertIn("RUNPOD_PAIRED_ACTION", importer["ALLOWED_NAMES"])
        source = (ROOT / "scripts/runpod_entrypoint.sh").read_text()
        self.assertIn('"${SCRIPT_DIR}/runpod_paired_train.py"', source)
        self.assertIn('set -- bash "${SCRIPT_DIR}/runpod_train_then_validate.sh" "${RUNPOD_CONFIG}"', source)

    def test_real_tmux_entrypoint_child_shells_and_guard_parent_completion(self):
        # This is a fixture executable, not a local Python/ML environment. Only
        # numerical CLI calls are replaced; shell handoff and lifecycle are real.
        (self.project / "scripts").unlink()
        shutil.copytree(ROOT / "scripts", self.project / "scripts")
        (self.project / "src").symlink_to(ROOT / "src", target_is_directory=True)
        for name in ("verify_runpod_mounted_readiness.sh", "ensure_runpod_data_dependencies.sh", "ensure_runpod_tmux.sh"):
            script = self.project / "scripts" / name
            script.write_text("#!/usr/bin/env bash\nexit 0\n")
            script.chmod(0o700)
        executable = self.project / ".venv/bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!" + sys.executable + "\n" + textwrap.dedent('''\
            import hashlib, json, os, subprocess, sys
            from pathlib import Path
            from datetime import datetime, timezone
            if sys.argv[1] == '-c':
                print('1' if 'ExperimentConfig' in sys.argv[2] else 'fixture numpy')
                raise SystemExit(0)
            if sys.argv[1].endswith('runpod_paired_train.py'):
                os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            if sys.argv[1] != '-m':
                raise SystemExit(81)
            module = sys.argv[2]
            if module == 'stock_forecasting.runpod.supervisor':
                command = sys.argv[sys.argv.index('--') + 1:]
                os.execvpe(command[0], command, os.environ)
            volume = Path(os.environ['NETWORK_VOLUME_ROOT'])
            run_id = os.environ['WANDB_RUN_ID']
            selection = json.loads(Path(os.environ['RUNPOD_REMOTE_SELECTION_PATH']).read_text())
            task = selection['baseline_task']['prediction_task']
            parent_marker = volume / 'lifecycle/runs' / os.environ['RUNPOD_PARENT_RUN_ID'] / 'training.json'
            if json.loads(parent_marker.read_text())['state'] in ('ready', 'failed', 'timed_out'):
                raise SystemExit(83)
            if (volume / 'fixture-termination-calls').exists():
                raise SystemExit(84)
            with (volume / 'model-cli-calls.jsonl').open('a') as stream:
                stream.write(json.dumps({'module': module, 'run_id': run_id, 'task': task}) + '\\n')
            if module == 'stock_forecasting.cli.train':
                manifest = volume / 'savedModel' / run_id / 'run-manifest.json'
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({
                    'schema_version': '4.0', 'run_id': run_id, 'run_key': run_id,
                    'training_resume_contract_sha256': 'a' * 64,
                    'training_resume_contract': {'model': {'prediction_task': task}},
                    'selection_provenance': {'selection_id': selection['selection_id']},
                }))
            elif module == 'stock_forecasting.cli.validate_benchmarks':
                report = volume / 'evaluations' / run_id / 'validation-benchmark.json'
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({
                    'run_id': run_id, 'prediction_task': task, 'state': 'ready',
                    'completed_at': datetime.now(timezone.utc).isoformat(), 'evaluation_split': 'test',
                }))
                subprocess.run([
                    sys.executable, str(Path(os.environ['PROJECT_ROOT']) / 'scripts/runpod_readiness.py'),
                    'write-state', '--output', os.environ['RUNPOD_VALIDATION_LIFECYCLE_MARKER'],
                    '--network-volume-root', str(volume), '--kind', 'stage1-validation',
                    '--state', 'finalizing', '--launch-id', os.environ['RUNPOD_LAUNCH_ID'],
                    '--wandb-run-id', run_id, '--log-path', os.environ['RUNPOD_SHUTDOWN_DIR'],
                    '--max-runtime-seconds', os.environ['MAX_RUNTIME_SECONDS'],
                ], check=True)
            else:
                raise SystemExit(82)
            '''))
        executable.chmod(0o700)
        (self.project / ".venv/pyvenv.cfg").write_text("include-system-site-packages = true\n")
        (self.project / "pyproject.toml").write_text("# Fixture metadata only\n")
        poetry = self.volume / "tools/poetry/2.4.0/bin/poetry"
        poetry.parent.mkdir(parents=True)
        poetry.write_text("#!/usr/bin/env bash\nexit 0\n")
        poetry.chmod(0o700)
        external_bin = tempfile.TemporaryDirectory(prefix="paired-image-fixture-")
        self.addCleanup(external_bin.cleanup)
        bin_root = Path(external_bin.name).resolve()
        image_python = bin_root / "image-python"
        image_python.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == -c ]]; then exit 0; fi\n'
            "exec " + shlex.quote(sys.executable) + ' "$@"\n'
        )
        image_python.chmod(0o700)
        tmux = bin_root / "tmux"
        tmux.write_text(textwrap.dedent('''\
            #!/usr/bin/env bash
            shift 2
            case "$1" in
                has-session) [[ -f "${NETWORK_VOLUME_ROOT}/fixture-session" ]] ;;
                new-session) touch "${NETWORK_VOLUME_ROOT}/fixture-session" ;;
                set-option|send-keys) exit 0 ;;
                *) exit 91 ;;
            esac
            '''))
        tmux.chmod(0o700)
        timeout = bin_root / "timeout"
        timeout.write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n')
        timeout.chmod(0o700)
        terminate = self.project / "scripts/runpod_self_terminate.sh"
        terminate.write_text('#!/usr/bin/env bash\nprintf "terminated\\n" >> "${NETWORK_VOLUME_ROOT}/fixture-termination-calls"\n')
        terminate.chmod(0o700)
        env = dict(os.environ, **self.environment, RUNPOD_TEST_MODE="1", RUNPOD_ROLE="gpu-train",
                   RUNPOD_PYTHON_BIN=str(image_python), RUNPOD_GPU_WORKFLOW_LEASE_HELD="1",
                   RUNPOD_SSH_ENV_IMPORTED="1", RUNPOD_TEST_READINESS_READY="1", WANDB_API_KEY="fixture-only",
                   RUNPOD_CONFIG=self.plan["children"]["alpha_point"]["config_path"],
                   PATH=str(bin_root) + os.pathsep + os.defpath)
        env.pop("RESUME_CHECKPOINT")
        env.pop("VALIDATION_CHECKPOINT")
        completed = subprocess.run(
            ["bash", str(self.project / "scripts/runpod_tmux_launch.sh"), "stage1-train"],
            env=env, cwd=self.project, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        runners = list((self.volume / "logs" / self.plan["run_id"] / "tmux").glob("*/*/runner.sh"))
        self.assertEqual(len(runners), 1)
        completed = subprocess.run(["bash", str(runners[0])], env=env, cwd=self.project,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        calls = [json.loads(line) for line in (self.volume / "model-cli-calls.jsonl").read_text().splitlines()]
        self.assertEqual([(call["task"], call["module"]) for call in calls], [
            (task, "stock_forecasting.cli." + cli)
            for task in ("alpha_point", "absolute_direction") for cli in ("train", "validate_benchmarks")
        ])
        for child in self.plan["children"].values():
            for phase in ("training", "validation"):
                marker = self.volume / "lifecycle/runs" / child["run_id"] / (phase + ".json")
                self.assertEqual(json.loads(marker.read_text())["state"], "ready")
        for phase in ("training", "validation"):
            marker = self.path.with_name(phase + ".json")
            self.assertEqual(json.loads(marker.read_text())["state"], "ready")
            checked = subprocess.run([
                sys.executable, str(ROOT / "scripts/runpod_readiness.py"), "guard-lifecycle-state",
                "--expected-kind", "stage1-" + phase, "--expected-pod-id", "pod-fixture",
                "--active-run-id", self.plan["run_id"],
            ], input=marker.read_text(), capture_output=True, text=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout.strip(), "ready")
        self.assertEqual((self.volume / "fixture-termination-calls").read_text().splitlines(), ["terminated"])


if __name__ == "__main__":
    unittest.main()
