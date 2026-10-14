"""Real creator-shell checks with fake provider, object store and guard boundaries."""

from __future__ import annotations

import json
import contextlib
import io
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]

# Every remote boundary is replaced. Neither RUNPOD_TEST_MODE nor dry-run is used:
# the production creator, selection validation, paired preflight/publication and
# local receipt writer execute through their real shell/subprocess entry points.
BOUNDARY = r'''
import json, os, pathlib, sys
root = pathlib.Path(os.environ["FIXTURE_ROOT"])
mode, args = sys.argv[1], sys.argv[2:]
def event(kind, **fields):
    data = (json.dumps(dict(kind=kind, **fields)) + "\n").encode()
    fd = os.open(root / "events.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
if mode == "api":
    if args == ["pod", "list", "--output", "json"]:
        event("inventory")
        print("[]")
    elif args == ["pod", "create-gpu"]:
        payload = json.load(sys.stdin)
        event("create", payload=payload)
        if os.environ.get("FIXTURE_API") == "rejected":
            print('{"status":400,"error":"No capacity available for this GPU"}', file=sys.stderr)
            sys.exit(7)
        if os.environ.get("FIXTURE_API") == "malformed":
            print('{"unexpected":"response"}')
        else:
            print('{"id":"fixture-pod"}')
    else:
        raise AssertionError("Unexpected provider request: " + repr(args))
elif mode == "s3":
    if args[:2] == ["s3api", "list-objects-v2"]:
        event("inventory-read")
        print("None")
    elif args[:2] == ["s3", "cp"]:
        source, destination = args[2:4]
        if destination.startswith("s3://"):
            event("write", key=destination, payload=json.loads(pathlib.Path(source).read_text()))
            if os.environ.get("FIXTURE_PUBLICATION_FAIL") == "1":
                print("fixture metadata upload failure", file=sys.stderr)
                sys.exit(8)
        elif source.startswith("s3://") and destination == "-":
            event("read", key=source)
            if source.endswith("/code.json"):
                print('{"release_digest":"fixture-release"}')
            elif "/lifecycle/selections/" in source:
                print(pathlib.Path(os.environ["RUNPOD_SELECTION_FILE"]).read_text())
            else:
                raise AssertionError("Unexpected remote read: " + source)
        else:
            raise AssertionError("Unexpected object copy: " + repr(args))
    else:
        raise AssertionError("Unexpected object-store request: " + repr(args))
elif mode == "guard":
    event("guard", pod=args[0], key=args[2], run=os.environ["RUNPOD_GUARD_RUN_ID"])
    if os.environ.get("FIXTURE_GUARD_FAIL") == "1":
        print("fixture guard startup failure", file=sys.stderr)
        sys.exit(8)
    print(45678)
elif mode == "baseline":
    event("baseline", args=args, config=os.environ.get("RUNPOD_CONFIG"))
    if os.environ.get("FIXTURE_BASELINE_FAIL") == "1":
        print("fixture baseline is incomplete", file=sys.stderr)
        sys.exit(9)
    print(json.dumps({"complete": args[0] == "require"}))
elif mode == "readiness":
    event("readiness", args=args)
    assert "--local-selection" in args
elif mode == "check-code":
    event("check-code", args=args)
    json.load(sys.stdin)
    if os.environ.get("FIXTURE_GATE_FAIL") == "code":
        sys.exit(9)
elif mode in {"training-data", "baseline-data"}:
    event(mode, args=args)
    if os.environ.get("FIXTURE_GATE_FAIL") == mode:
        sys.exit(9)
else:
    raise AssertionError("Unexpected boundary: " + mode)
'''


class CreatePublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="create-publication-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for directory in ("scripts", "configs", "src"):
            shutil.copytree(
                ROOT / directory,
                self.root / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        self.selection = runpy.run_path(str(self.root / "scripts/runpod_selection.py"))
        args = self.selection["build_parser"]().parse_args([
            "create", "--project-root", str(self.root), "--stage", "stage2",
            "--data-profile", "us_tw_eodhd", "--start", "2021-01-01",
            "--end", "2026-06-01", "--universe", "all",
        ])
        self.original = self.selection["_build_selection"](args, self.root)
        self.selection["_activate_selection"](self.root, self.original)
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "WANDB_", "FIN_TS_", "VALIDATION_", "FIXTURE_"))
            and key not in {"RESUME_CHECKPOINT", "PROJECT_ROOT", "SAVED_MODEL_ROOT", "DATA_ROOT"}
        }
        env_file = self.root / ".env"
        env_file.write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "RUNPOD_S3_ACCESS_KEY_ID=fixture-key\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture-secret\n"
            "RUNPOD_S3_REGION=fixture-region\n"
        )
        env_file.chmod(0o600)
        self.env.update(
            FIXTURE_ROOT=str(self.root), RUNPOD_ENV_FILE=str(env_file),
            RUNPOD_GUARD_LOG_DIR=str(self.root / "guards"),
            RUNPOD_LAUNCH_CONTROL_LOCK_HELD="1",
        )
        (self.root / "fixture_boundary.py").write_text(BOUNDARY)
        for name, mode in (
            ("runpodctl_project.sh", "api"), ("runpod_s3_project.sh", "s3"),
            ("launch_runpod_guard.sh", "guard"),
            ("verify_runpod_stage_readiness.sh", "readiness"),
            ("runpod_baseline_cache.py", "baseline"),
        ):
            self.boundary(name, mode)

    def boundary(self, name, mode):
        path = self.root / "scripts" / name
        if name.endswith(".py"):
            path.write_text(
                "import os,sys\n"
                f"os.execv(sys.executable, [sys.executable, {str(self.root / 'fixture_boundary.py')!r}, "
                f"{mode!r}, *sys.argv[1:]])\n"
            )
        else:
            path.write_text(
                "#!/bin/bash\nset -eu\nexec " + shlex.quote(sys.executable) + " "
                + shlex.quote(str(self.root / "fixture_boundary.py")) + " "
                + shlex.quote(mode) + ' "$@"\n'
            )
        path.chmod(0o700)

    def configure(self, *, paired=False, workflow="train"):
        if paired:
            paired_control = runpy.run_path(str(self.root / "scripts/runpod_paired_control.py"))
            with patch.dict(os.environ, self.env, clear=True):
                self.plan, _, environment = paired_control["prepare_launch"](
                    self.root, self.original, "a-signal64", 42, workflow=workflow,
                )
            self.env.update(environment)
            return
        selected = self.selection["_with_experiment"](
            self.original, "a-alpha-point64", self.root,
        )
        self.selection["_activate_selection"](self.root, selected)
        self.env["RUNPOD_GPU_WORKFLOW"] = workflow
        self.plan = None

    def create(self, **overrides):
        return subprocess.run(
            ["bash", "scripts/create_runpod_pod.sh"], cwd=self.root,
            env={**self.env, **overrides}, capture_output=True, text=True, timeout=30,
        )

    def events(self):
        path = self.root / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def receipt(self):
        paths = list((self.root / ".runpod/launch-attempts").glob("*.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)
        text = paths[0].read_text()
        for forbidden in ("fixture-secret", "WANDB_API_KEY", "RUNPOD_S3_SECRET_ACCESS_KEY", "env"):
            self.assertNotIn(forbidden, text)
        return json.loads(text)

    def assert_refused(self, *, paired, workflow):
        self.configure(paired=paired, workflow=workflow)
        result = self.create(FIXTURE_API="rejected")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn('{"status":400,"error":"No capacity available for this GPU"}', result.stderr)
        events = self.events()
        self.assertEqual(len([item for item in events if item["kind"] == "create"]), 1)
        self.assertFalse(any(item["kind"] in {"write", "guard"} for item in events))
        self.assertEqual(self.receipt()["state"], "rejected")
        self.assertEqual(self.receipt()["http_status"], 400)

    def test_individual_training_capacity_refusal_has_no_remote_writes(self):
        self.assert_refused(paired=False, workflow="train")

    def test_individual_baseline_capacity_refusal_has_no_remote_writes(self):
        self.assert_refused(paired=False, workflow="baseline")

    def test_paired_training_capacity_refusal_has_no_remote_writes(self):
        self.assert_refused(paired=True, workflow="train")

    def test_paired_baseline_capacity_refusal_has_no_remote_writes(self):
        self.assert_refused(paired=True, workflow="baseline")

    def assert_success(self, *, paired, workflow):
        self.configure(paired=paired, workflow=workflow)
        result = self.create()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = self.events()
        kinds = [item["kind"] for item in events]
        self.assertLess(kinds.index("create"), kinds.index("guard"))
        writes = [item for item in events if item["kind"] == "write"]
        self.assertTrue(writes)
        self.assertTrue(all(index > kinds.index("guard") for index, kind in enumerate(kinds) if kind == "write"))
        request = next(item["payload"] for item in events if item["kind"] == "create")
        run_id = request["env"]["WANDB_RUN_ID"]
        selection_id = request["env"]["RUNPOD_SELECTION_ID"]
        base = "s3://fixture-volume/"
        expected = {base + f"lifecycle/selections/{selection_id}.json"}
        if paired:
            expected = {base + child["selection_path"] for child in self.plan["children"].values()}
            expected.add(base + f"lifecycle/runs/{run_id}/pair.json")
            published = next(item["payload"] for item in writes if item["key"].endswith("/pair.json"))
            self.assertEqual(published, self.plan)
        if workflow == "train":
            expected.add(base + f"lifecycle/runs/{run_id}/selection.json")
        self.assertEqual({item["key"] for item in writes}, expected)
        self.assertEqual(len(writes), len(expected))
        guard = next(item for item in events if item["kind"] == "guard")
        self.assertEqual(guard["run"], run_id)
        self.assertEqual(guard["pod"], "fixture-pod")
        self.assertIn("Created Pod: fixture-pod", result.stdout)
        self.assertIn("Run ID: " + run_id, result.stdout)
        self.assertEqual(self.receipt()["state"], "created")
        self.assertEqual(self.receipt()["pod_id"], "fixture-pod")

    def test_individual_training_publishes_only_after_guard(self):
        self.assert_success(paired=False, workflow="train")

    def test_individual_baseline_publishes_only_after_guard(self):
        self.assert_success(paired=False, workflow="baseline")

    def test_paired_training_publishes_both_selections_and_parent_only_after_guard(self):
        self.assert_success(paired=True, workflow="train")

    def test_paired_baseline_publishes_both_selections_and_parent_only_after_guard(self):
        self.assert_success(paired=True, workflow="baseline")

    def test_guard_failure_keeps_created_receipt_and_does_not_publish(self):
        self.configure(paired=True)
        result = self.create(FIXTURE_GUARD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(item["kind"] == "write" for item in self.events()))
        self.assertEqual(self.receipt()["state"], "created")
        self.assertIn("fixture-pod", result.stderr)
        self.assertIn("CREATED", result.stderr)
        self.assertIn("guard", result.stderr.lower())

    def test_publication_failure_keeps_created_receipt_and_armed_guard(self):
        self.configure(paired=True)
        result = self.create(FIXTURE_PUBLICATION_FAIL="1")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        kinds = [item["kind"] for item in self.events()]
        self.assertLess(kinds.index("guard"), kinds.index("write"))
        self.assertEqual(self.receipt()["state"], "created")
        self.assertIn("Pod fixture-pod was CREATED", result.stderr)
        self.assertIn("Guard 45678 remains armed", result.stderr)
        self.assertIn("Do not start training", result.stderr)

    def test_malformed_success_retains_unknown_receipt_without_publishing(self):
        self.configure()
        result = self.create(FIXTURE_API="malformed")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.receipt()["state"], "unknown")
        self.assertFalse(any(item["kind"] in {"write", "guard"} for item in self.events()))

    def test_baseline_gate_failure_prevents_even_the_capacity_request(self):
        self.configure(paired=True)
        result = self.create(FIXTURE_BASELINE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(item["kind"] in {"write", "guard", "create"} for item in self.events()))
        self.assertFalse((self.root / ".runpod/launch-attempts").exists())

    def restore_readiness_shell(self):
        shutil.copy2(ROOT / "scripts/verify_runpod_stage_readiness.sh", self.root / "scripts/verify_runpod_stage_readiness.sh")
        self.boundary("runpod_readiness.py", "check-code")
        self.boundary("runpod_training_readiness.py", "training-data")
        self.boundary("runpod_baseline_readiness.py", "baseline-data")

    def test_local_selection_keeps_code_and_data_model_gates_before_capacity(self):
        self.configure()
        self.restore_readiness_shell()
        result = self.create(FIXTURE_API="rejected")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        kinds = [item["kind"] for item in self.events()]
        for gate in ("baseline", "check-code", "training-data"):
            self.assertLess(kinds.index(gate), kinds.index("create"))
        reads = [item["key"] for item in self.events() if item["kind"] == "read"]
        self.assertEqual(reads, ["s3://fixture-volume/lifecycle/stage1/code.json"])

    def test_local_selection_keeps_baseline_cleaning_gate_before_capacity(self):
        self.configure(workflow="baseline")
        self.restore_readiness_shell()
        result = self.create(FIXTURE_API="rejected")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        kinds = [item["kind"] for item in self.events()]
        for gate in ("baseline", "check-code", "baseline-data"):
            self.assertLess(kinds.index(gate), kinds.index("create"))
        self.assertNotIn("training-data", kinds)

    def test_local_selection_rejects_failed_code_or_data_gate(self):
        self.configure()
        self.restore_readiness_shell()
        for gate in ("code", "training-data"):
            with self.subTest(gate=gate):
                result = self.create(FIXTURE_GATE_FAIL=gate)
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(item["kind"] in {"write", "guard", "create"} for item in self.events()))

    def test_local_selection_does_not_bypass_local_config_validation(self):
        self.configure()
        self.restore_readiness_shell()
        config = self.root / "configs/experiments/a_alpha_point64.yaml"
        config.write_text(config.read_text() + "\n# Fixture post-selection change.\n")
        result = self.create()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(item["kind"] in {"write", "guard", "create"} for item in self.events()))

    def test_default_readiness_still_compares_published_selection(self):
        self.configure()
        self.restore_readiness_shell()
        for mode in ("--gpu", "--baseline"):
            with self.subTest(mode=mode):
                before = len(self.events())
                result = subprocess.run(
                    ["bash", "scripts/verify_runpod_stage_readiness.sh", mode],
                    cwd=self.root, env=self.env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                reads = [item["key"] for item in self.events()[before:] if item["kind"] == "read"]
                self.assertEqual(len(reads), 2)
                self.assertIn("/lifecycle/selections/", reads[1])
        self.assertFalse(any(item["kind"] in {"write", "guard", "create"} for item in self.events()))


class CreateReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="create-receipt-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = runpy.run_path(str(ROOT / "scripts/runpod_create_request.py"))
        self.payload = {
            "name": "stock-forecasting-fixture", "gpuId": "fixture-gpu",
            "networkVolumeId": "fixture-volume",
            "env": {"WANDB_RUN_ID": "run-fixture", "API_TOKEN": "never-copy-this-secret"},
        }

    def record(self):
        paths = list((self.root / ".runpod/launch-attempts").glob("*.json"))
        self.assertEqual(len(paths), 1)
        text = paths[0].read_text()
        self.assertNotIn("never-copy-this-secret", text)
        self.assertNotIn("API_TOKEN", text)
        return json.loads(text)

    def test_provider_failure_preserves_raw_diagnostic_and_exit_code(self):
        raw_error = '{"status":400,"error":{"capacity":"GPU unavailable","details":[1,2]}}\n'
        response = subprocess.CompletedProcess([], 7, "provider stdout\n", raw_error)
        out, err = io.StringIO(), io.StringIO()
        with patch.object(self.module["subprocess"], "run", return_value=response), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.module["create"](self.root, self.payload)
        self.assertEqual(code, 7)
        self.assertEqual(out.getvalue(), response.stdout)
        self.assertEqual(err.getvalue(), raw_error)
        self.assertEqual(self.record()["state"], "rejected")

    def test_request_timeout_leaves_unknown_receipt(self):
        with patch.object(self.module["subprocess"], "run", side_effect=subprocess.TimeoutExpired("fixture", 90)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.module["create"](self.root, self.payload)
        self.assertEqual(self.record()["state"], "unknown")
        self.assertIn("finished_at", self.record())

    def test_failed_final_receipt_write_does_not_mask_created_pod_id(self):
        namespace = self.module["create"].__globals__
        save = namespace["save"]
        calls = []

        def save_then_fail(path, record):
            calls.append(record["state"])
            if len(calls) == 2:
                raise OSError("fixture disk full")
            save(path, record)

        response = subprocess.CompletedProcess([], 0, '{"id":"fixture-pod"}\n', "")
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(namespace, save=save_then_fail), \
                patch.object(self.module["subprocess"], "run", return_value=response), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.module["create"](self.root, self.payload)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["id"], "fixture-pod")
        self.assertEqual(calls, ["requesting", "created"])
        self.assertIn("could not update local launch receipt", err.getvalue())
        self.assertEqual(self.record()["state"], "requesting")

    def test_initial_receipt_failure_prevents_paid_request(self):
        namespace = self.module["create"].__globals__
        provider = Mock()
        with patch.dict(namespace, save=Mock(side_effect=OSError("fixture disk full"))), \
                patch.object(self.module["subprocess"], "run", provider):
            with self.assertRaises(OSError):
                self.module["create"](self.root, self.payload)
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
