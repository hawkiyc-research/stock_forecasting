"""Dependency-free multi-Pod launch, ownership, discovery and lease regression."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONTROL = runpy.run_path(str(ROOT / "scripts/runpod_concurrency.py"))
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
RUNS = runpy.run_path(str(ROOT / "scripts/runpod_runs.py"))
PATHS = runpy.run_path(str(ROOT / "src/stock_forecasting/run_paths.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))


def pod(run_id="run-a", pod_id="pod-a", role="gpu-train", scoped="1"):
    return {
        "id": pod_id,
        "networkVolumeId": "fixture-volume",
        "env": {
            "RUNPOD_ROLE": role,
            "RUNPOD_SCOPED_LIFECYCLE": scoped,
            "WANDB_RUN_ID": run_id,
        },
    }


def write_script(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\nset -eu\n" + content)
    path.chmod(0o700)


class ParallelControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parallel-control-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def selection(self):
        shutil.copytree(ROOT / "configs", self.root / "configs")
        shutil.copytree(
            ROOT / "src", self.root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        (self.root / "scripts").mkdir()
        shutil.copy2(
            ROOT / "scripts/runpod_selection.py", self.root / "scripts/runpod_selection.py"
        )
        args = argparse.Namespace(
            stage="stage2",
            data_profile="us_tw_eodhd",
            dataset_revision="v1",
            start="2016-01-01",
            end="2026-06-01",
            h_start=1,
            universe="all",
            stocks=[],
            etfs=[],
            symbol_limit=None,
            feature_mode="combined",
        )
        selected = SELECTION["_build_selection"](args, self.root)
        SELECTION["_activate_selection"](self.root, selected)
        return selected

    def test_only_distinct_scoped_consumers_can_share_volume(self):
        admit = CONTROL["check_admission"]
        admit([pod()], "fixture-volume", "train", "run-b")
        admit([pod(role="gpu-validation")], "fixture-volume", "validation", "run-b")
        admit([pod()], "another-volume", "sync", "")
        for mode, other in (
            ("train", pod()),
            ("validation", pod()),
            ("sync", pod()),
            ("exclusive", pod()),
            ("train", pod(scoped="0")),
            ("train", pod(role="gpu-baseline")),
            ("train", pod(role="cpu-prep")),
        ):
            with self.subTest(mode=mode, other=other), self.assertRaises(ValueError):
                admit([other], "fixture-volume", mode, "run-a")
        with self.assertRaises(ValueError):
            admit({}, "fixture-volume", "train", "run-a")

    def test_three_launches_pin_config_and_do_not_change_active_selection(self):
        self.selection()
        pointer = self.root / ".runpod/active-selection.json"
        before = pointer.read_bytes()
        write_script(
            self.root / "scripts/publish_runpod_selection.sh",
            'test -f "$RUNPOD_LAUNCH_SELECTION_FILE"\n',
        )
        write_script(
            self.root / "scripts/create_runpod_pod.sh",
            f"exec {shlex.quote(sys.executable)} -c "
            + shlex.quote(
                "import json,os,pathlib,time; "
                "p=json.load(open(os.environ['RUNPOD_LAUNCH_SELECTION_FILE'])); "
                "time.sleep(0.03); "
                "root=pathlib.Path(os.environ['FIXTURE_ROOT']); "
                "name=p['stage']['experiment']; "
                "suffix='-preflight' if os.environ.get('RUNPOD_PREFLIGHT_ONLY') else '-launch'; "
                "(root/(name+suffix)).write_text(json.dumps(p))"
            )
            + "\n",
        )
        with (
            patch.dict(os.environ, {"FIXTURE_ROOT": str(self.root)}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = CONTROL["launch"](self.root, ["a-lora32", "a-lora64", "b-partial"], 2)
        self.assertEqual(result, 0)
        self.assertEqual(pointer.read_bytes(), before)
        for name in ("a-lora32", "a-lora64", "b-partial"):
            saved = json.loads((self.root / (name + "-launch")).read_text())
            self.assertEqual(saved["stage"]["experiment"], name)
            self.assertEqual(
                saved["stage"]["config_path"],
                "configs/experiments/" + name.replace("-", "_") + ".yaml",
            )
            self.assertEqual(json.loads((self.root / (name + "-preflight")).read_text()), saved)

    def test_full_creator_binds_two_pods_guards_and_selections(self):
        self.selection()
        shutil.copytree(
            ROOT / "scripts",
            self.root / "scripts",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        env_file = self.root / ".env"
        env_file.write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "RUNPOD_S3_ACCESS_KEY_ID=fixture-key\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture-secret\n"
            "RUNPOD_S3_REGION=fixture-region\n"
        )
        env_file.chmod(0o600)
        writes = self.root / "cloud"
        writes.mkdir()
        write_script(self.root / "scripts/verify_runpod_stage_readiness.sh", "exit 0\n")
        (self.root / "scripts/runpod_baseline_cache.py").write_text("raise SystemExit(0)\n")
        write_script(
            self.root / "scripts/runpod_s3_project.sh", '[[ "$1 $2" == "s3 cp" ]] || exit 93\n'
        )
        transport = (
            "import json,os,pathlib,sys; root=pathlib.Path(os.environ['FIXTURE_ROOT'])/'cloud'; "
            "args=sys.argv[1:]; "
            "command=args[1]; "
            "payload=json.load(sys.stdin) if command=='create-gpu' else None; "
            "run=payload['env']['WANDB_RUN_ID'] if payload else ''; "
            "pod_id='pod-'+run if run else ''; "
            "record=dict(payload,id=pod_id) if payload else None; "
            "(root/(pod_id+'.json')).write_text(json.dumps(record)) if payload else None; "
            "print(json.dumps(record if payload else "
            "[json.loads(p.read_text()) for p in root.glob('pod-*.json')] if command=='list' "
            "else json.loads((root/(args[2]+'.json')).read_text())))"
        )
        write_script(
            self.root / "scripts/runpodctl_project.sh",
            f'exec {shlex.quote(sys.executable)} -c {shlex.quote(transport)} "$@"\n',
        )
        guard = (
            "import json,os,pathlib,sys; "
            "path=pathlib.Path(os.environ['FIXTURE_ROOT'])/'cloud'/('guard-'+sys.argv[1]+'.json'); "
            "record=dict(pod=sys.argv[1],key=sys.argv[3],run=os.environ['RUNPOD_GUARD_RUN_ID']); "
            "path.write_text(json.dumps(record)); print(12345)"
        )
        write_script(
            self.root / "scripts/launch_runpod_guard.sh",
            f'exec {shlex.quote(sys.executable)} -c {shlex.quote(guard)} "$@"\n',
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("RUNPOD_", "WANDB_", "VALIDATION_", "FIN_TS_"))
        }
        env.update(
            FIXTURE_ROOT=str(self.root),
            RUNPOD_ENV_FILE=str(env_file),
            RUNPOD_GUARD_LOG_DIR=str(self.root / "guards"),
        )
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = CONTROL["launch"](self.root, ["a-lora32", "b-partial"], 2)
        self.assertEqual(result, 0)
        pods = [json.loads(p.read_text()) for p in writes.glob("pod-*.json")]
        self.assertEqual(len(pods), 2)
        self.assertEqual(len({p["env"]["WANDB_RUN_ID"] for p in pods}), 2)
        self.assertEqual(
            {p["name"] for p in pods}, {"stock-forecasting-a-lora32", "stock-forecasting-b-partial"}
        )
        for record in pods:
            run_id = record["env"]["WANDB_RUN_ID"]
            guard = json.loads((writes / ("guard-" + record["id"] + ".json")).read_text())
            self.assertEqual(record["env"]["RUNPOD_SCOPED_LIFECYCLE"], "1")
            self.assertEqual(
                guard,
                {
                    "pod": record["id"],
                    "run": run_id,
                    "key": f"lifecycle/runs/{run_id}/training.json",
                },
            )

    def test_real_lifecycle_writer_keeps_two_runs_separate(self):
        volume = self.root / "volume"
        for name in ("a", "b"):
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/runpod_readiness.py"),
                    "write-state",
                    "--output",
                    str(volume / f"lifecycle/runs/run-{name}/training.json"),
                    "--network-volume-root",
                    str(volume),
                    "--kind",
                    "stage1-training",
                    "--state",
                    "preparing",
                    "--wandb-run-id",
                    f"run-{name}",
                    "--launch-id",
                    "launch-" + name,
                ],
                env={**os.environ, "RUNPOD_POD_ID": "pod-" + name},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((volume / "lifecycle/stage1/training.json").exists())
        for name in ("a", "b"):
            payload = json.loads((volume / f"lifecycle/runs/run-{name}/training.json").read_text())
            self.assertEqual(payload["wandb_run_id"], "run-" + name)
            self.assertEqual(payload["pod_id"], "pod-" + name)

    def test_failed_preflight_prevents_every_paid_create(self):
        self.selection()
        write_script(self.root / "scripts/publish_runpod_selection.sh", "exit 0\n")
        write_script(
            self.root / "scripts/create_runpod_pod.sh",
            '[[ "${RUNPOD_PREFLIGHT_ONLY:-0}" == 1 ]] || exit 99\nexit 2\n',
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CONTROL["launch"](self.root, ["a-lora32", "b-lora64"], 2), 2)
        with self.assertRaises(ValueError):
            CONTROL["launch"](self.root, ["a-lora32", "a-lora32"], 2)

    def test_partial_launch_reports_each_result_without_retrying_success(self):
        self.selection()
        write_script(self.root / "scripts/publish_runpod_selection.sh", "exit 0\n")
        write_script(
            self.root / "scripts/create_runpod_pod.sh",
            r"""
            if [[ "${RUNPOD_PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
            printf '%s\n' "$RUNPOD_POD_NAME" >> "$FIXTURE_ROOT/created-once"
            if [[ "$RUNPOD_POD_NAME" == stock-forecasting-a-lora64 ]]; then exit 23; fi
            echo 'Pod created with an independent guard'
        """,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict(os.environ, {"FIXTURE_ROOT": str(self.root)}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(CONTROL["launch"](self.root, ["a-lora32", "a-lora64"], 2), 2)
        self.assertCountEqual(
            (self.root / "created-once").read_text().splitlines(),
            ["stock-forecasting-a-lora32", "stock-forecasting-a-lora64"],
        )
        self.assertIn("[a-lora32] created", stdout.getvalue())
        self.assertIn("[a-lora64] launch failed", stdout.getvalue())
        self.assertIn("No automatic retries or deletions", stderr.getvalue())

    def test_resume_selection_does_not_follow_or_replace_active_experiment(self):
        original = self.selection()
        pointer = self.root / ".runpod/active-selection.json"
        before = pointer.read_bytes()
        selected = SELECTION["_with_experiment"](original, "a-lora64", self.root)
        reader = RUNS["RunRecords"](self.root, "fixture-volume")
        with patch.object(reader, "read", return_value=selected) as read:
            restored = reader.selection("run-previous-a")
        read.assert_called_once_with("lifecycle/runs/run-previous-a/selection.json", optional=True)
        self.assertEqual(json.loads(restored.read_text())["stage"]["experiment"], "a-lora64")
        self.assertEqual(pointer.read_bytes(), before)

    def test_sync_and_launch_mutex(self):
        with CONTROL["control_lock"](self.root, exclusive=False):
            with CONTROL["control_lock"](self.root, exclusive=False):
                pass
            with self.assertRaises(ValueError), CONTROL["control_lock"](self.root, exclusive=True):
                pass

    def test_recovery_listing_is_paginated_and_includes_all_runs(self):
        function = RECOVERY["list_lifecycle_keys"]
        pages = [
            {
                "Contents": [{"Key": "lifecycle/runs/run-a/training.json"}],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            },
            {"Contents": [{"Key": "lifecycle/runs/run-b/training.json"}]},
        ]
        with patch.dict(function.__globals__, {"run_json": lambda cmd: pages.pop(0)}):
            self.assertEqual(
                function("fixture-volume"),
                {
                    "lifecycle/runs/run-a/training.json",
                    "lifecycle/runs/run-b/training.json",
                },
            )
        with (
            patch.dict(
                function.__globals__,
                {
                    "run_json": lambda cmd: {
                        "IsTruncated": True,
                        "NextContinuationToken": "same",
                    }
                },
            ),
            self.assertRaisesRegex(RuntimeError, "continuation"),
        ):
            function("fixture-volume")

    def test_dependency_waiter_rechecks_under_read_lease_after_another_updater(self):
        scripts, binary = self.root / "scripts", self.root / "bin"
        scripts.mkdir()
        binary.mkdir()
        shutil.copy2(ROOT / "scripts/ensure_runpod_data_dependencies.sh", scripts)
        # Executable transport fixtures, not a Python interpreter or environment.
        write_script(
            self.root / ".venv/bin/python",
            r"""
            if [[ ! -f "$PROJECT_ROOT/version-checked" ]]; then
                touch "$PROJECT_ROOT/version-checked"
                exit 1
            fi
            exit 0
        """,
        )
        write_script(
            binary / "flock",
            r"""
            printf '%s\n' "$*" >> "$PROJECT_ROOT/locks"
            [[ "$*" != "-n 9" ]]
        """,
        )
        write_script(scripts / "setup_runpod_environment.sh", "exit 93\n")
        result = subprocess.run(
            ["bash", str(scripts / "ensure_runpod_data_dependencies.sh")],
            env={
                **os.environ,
                "PATH": str(binary) + os.pathsep + os.defpath,
                "PROJECT_ROOT": str(self.root),
                "RUNPOD_POD_ID": "fixture-pod",
                "RUNPOD_GPU_WORKFLOW_LEASE_MODE": "shared",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.root / "locks").read_text().splitlines(), ["-u 9", "-n 9", "-s -n 9"]
        )

    def test_run_paths_reject_other_run_and_recovery_uses_pod_binding(self):
        for kind in ("training", "validation"):
            validate = PATHS["validate_" + kind + "_lifecycle_path"]
            scoped = self.root / "lifecycle/runs/run-a" / (kind + ".json")
            self.assertEqual(
                validate(scoped, network_volume_root=self.root, run_id="run-a"), scoped
            )
            with self.assertRaises(ValueError):
                validate(scoped, network_volume_root=self.root, run_id="run-b")
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod()),
            ("stage1-training", "lifecycle/runs/run-a/training.json"),
        )
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod(scoped="0")),
            ("stage1-training", "lifecycle/stage1/training.json"),
        )

    def test_latest_uses_completion_time_and_resume_rejects_ambiguity(self):
        reader = RUNS["RunRecords"](self.root, "fixture-volume")
        records = [
            (
                f"lifecycle/runs/run-{name}/training-completed.json",
                {
                    "run_id": "run-" + name,
                    "kind": "stage1-training-completion",
                    "state": "ready",
                    "training_completed": True,
                    "completed_at": f"2026-10-06T{hour}:00:00+00:00",
                },
            )
            for name, hour in (("a", "09"), ("b", "08"))
        ]
        with (
            patch.object(reader, "records", return_value=iter(records)),
            patch.object(reader, "read", return_value=None),
        ):
            self.assertEqual(reader.choose("completed"), "run-a")
        interrupted = [
            (
                f"lifecycle/runs/run-{name}/training.json",
                {
                    "wandb_run_id": f"run-{name}",
                    "kind": "stage1-training",
                    "state": "timed_out",
                    "generated_at": "2026-10-06T08:00:00Z",
                },
            )
            for name in ("a", "b")
        ]
        with (
            patch.object(reader, "records", return_value=iter(interrupted)),
            patch.object(reader, "read", return_value=None),
            self.assertRaisesRegex(ValueError, "Multiple interrupted"),
        ):
            reader.choose("resume")

    def test_lifecycle_never_borrows_another_runs_marker(self):
        reader = RUNS["RunRecords"](self.root, "fixture-volume")
        wrong = {"wandb_run_id": "run-b", "kind": "stage1-training", "state": "ready"}
        with patch.object(reader, "read", side_effect=[None, wrong]):
            self.assertEqual(reader.lifecycle("run-a", "training"), {})
        with patch.object(reader, "read", return_value=wrong), self.assertRaises(ValueError):
            reader.lifecycle("run-a", "training")

    def test_path_routing_is_not_a_numerical_resume_dependency(self):
        tree = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        paths = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "TRAINING_IMPLEMENTATION_PATHS"
                for t in node.targets
            )
        )
        self.assertNotIn("run_paths.py", paths)
        for name in (
            "training.py",
            "config.py",
            "date_market_sampler.py",
            "models/quant.py",
            "data/sample_universe.py",
        ):
            self.assertIn(name, paths)
        names = {
            "_canonical_payload_digest",
            "_validated_implementation_files",
            "_checkpoint_retention_migration_matches",
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            ]
            + [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ],
            type_ignores=[],
        )
        namespace = {
            "hashlib": hashlib,
            "json": json,
            "TRAINING_IMPLEMENTATION_PATHS": paths,
            "CHECKPOINT_RETENTION_MIGRATIONS": (),
        }
        exec(compile(ast.fix_missing_locations(module), "<contract-test>", "exec"), namespace)
        files = dict.fromkeys(paths, "a" * 64)

        def contract(values):
            return {
                "training_implementation": {
                    "files": values,
                    "sha256": namespace["_canonical_payload_digest"](values),
                }
            }

        old = contract({**files, "run_paths.py": "b" * 64})
        self.assertTrue(namespace["_checkpoint_retention_migration_matches"](old, contract(files)))
        self.assertFalse(
            namespace["_checkpoint_retention_migration_matches"](
                old,
                contract({**files, "training.py": "c" * 64}),
            )
        )

    def test_existing_static_workflow_boundaries(self):
        tree = ast.parse((ROOT / "tests/test_runpod_quant_contract.py").read_text())
        names = {
            "test_stable_runpod_lifecycle_and_synchronization_boundaries_remain",
            "test_workflow_exposes_bounded_cpu_gpu_resume_and_validation_options",
            "test_download_defaults_to_all_retained_checkpoints_and_can_select_best",
        }
        module = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ],
            type_ignores=[],
        )
        namespace = {"ROOT": ROOT}
        exec(compile(module, "<static-control-contracts>", "exec"), namespace)
        for name in sorted(names):
            with self.subTest(name=name):
                namespace[name]()

    def test_real_shell_shared_and_per_run_leases(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        # Use the host OS flock via stdlib; no Python project environment is used.
        flock = bin_dir / "flock"
        flock.write_text(
            f"#!{sys.executable}\nimport fcntl,sys\n"
            "mode=fcntl.LOCK_SH if '-s' in sys.argv else fcntl.LOCK_EX\n"
            "try: fcntl.flock(int(sys.argv[-1]), mode | fcntl.LOCK_NB)\n"
            "except BlockingIOError: sys.exit(1)\n"
        )
        flock.chmod(0o700)
        env = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "RUNPOD_ROLE": "gpu-train",
            "RUNPOD_SCOPED_LIFECYCLE": "1",
            "WANDB_RUN_ID": "run-a",
        }
        code = (
            'source "$1"; runpod_acquire_gpu_workflow_lease "$2" || exit $?; '
            "echo held; read -r release"
        )
        holder = subprocess.Popen(
            [
                "bash",
                "-c",
                code,
                "fixture",
                str(ROOT / "scripts/lib/runpod_paths.sh"),
                str(self.root / "volume"),
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "held")
            command = [
                "bash",
                "-c",
                'source "$1"; runpod_acquire_gpu_workflow_lease "$2"',
                "fixture",
                str(ROOT / "scripts/lib/runpod_paths.sh"),
                str(self.root / "volume"),
            ]
            for run_id, scoped, expected in (
                ("run-b", "1", 0),
                ("run-a", "1", 75),
                ("run-c", "0", 75),
            ):
                response = subprocess.run(
                    command,
                    env={**env, "WANDB_RUN_ID": run_id, "RUNPOD_SCOPED_LIFECYCLE": scoped},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(response.returncode, expected, response.stderr)
        finally:
            holder.communicate("release\n", timeout=10)


if __name__ == "__main__":
    unittest.main()
