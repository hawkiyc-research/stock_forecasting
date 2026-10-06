"""Run the result download shell against a bounded fake object store."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrainingResultDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        (scripts / "lib").mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/download_runpod_results.sh", scripts / "download.sh")
        shutil.copyfile(ROOT / "scripts/runpod_runs.py", scripts / "runpod_runs.py")
        (scripts / "lib/runpod_project_env.sh").write_text(
            "runpod_load_s3_env() { export RUNPOD_NETWORK_VOLUME_ID=fixture-volume; }\n"
        )
        (scripts / "runpod_readiness.py").write_text('print("checkpoint-000001")\n')
        (scripts / "runpod_s3_project.sh").write_text(
            'exec python3 "$(dirname "$0")/fake_s3.py" "$@"\n'
        )
        (scripts / "fake_s3.py").write_text('''
import json
import os
import pathlib
import shutil
import sys
root = pathlib.Path(__file__).resolve().parents[1]
args = sys.argv[1:]
denied = root / "denied"
if denied.exists() and denied.read_text().strip() in ("", args[1]):
    sys.stderr.write("An error occurred (403): Access denied")
    sys.exit(1)
if args[:2] == ["s3api", "list-objects-v2"]:
    cloud = root / "cloud"
    prefix = args[args.index("--prefix") + 1]
    keys = sorted(str(path.relative_to(cloud)) for path in cloud.rglob("*.json"))
    print(json.dumps({"IsTruncated": False, "Contents": [
        {"Key": key} for key in keys if key.startswith(prefix)
    ]}))
elif args[:2] == ["s3api", "head-object"]:
    key = args[args.index("--key") + 1]
    if not (root / "cloud" / key).is_file():
        sys.stderr.write("An error occurred (404): Not Found")
        sys.exit(1)
    print("{}")
else:
    assert args[:2] == ["s3", "cp"]
    source = root / "cloud" / args[2].removeprefix("s3://fixture-volume/")
    if "--recursive" in args:
        if source.is_dir():
            shutil.copytree(source, args[3], dirs_exist_ok=True)
    elif not source.is_file():
        sys.stderr.write("An error occurred (404): Not Found: " + str(source))
        sys.exit(1)
    elif args[3] == "-":
        sys.stdout.buffer.write(source.read_bytes())
    else:
        shutil.copyfile(source, args[3])
''')
        self.saved = self.root / "cloud/savedModel/run-fixture"
        self.saved.mkdir(parents=True)
        for name in ("run-manifest.json", "checkpoint-leaderboard.json", "best-checkpoint.json"):
            (self.saved / name).write_text("{}")
        (self.saved / "resolved-config.yaml").write_text("experiment_name: fixture\n")
        self.report = self.root / "cloud/evaluations/run-fixture"
        self.report.mkdir(parents=True)
        (self.report / "validation-benchmark.json").write_text("{}")
        self.write_lifecycle("stage1", "run-fixture")
        self.write_record("lifecycle/runs/run-fixture/training-completed.json", {
            "kind": "stage1-training-completion", "state": "ready",
            "run_id": "run-fixture", "training_completed": True,
            "completed_at": "2026-10-06T00:00:00+00:00",
        })
        (self.root / "bin").mkdir()
        (self.root / "bin/python3").symlink_to(sys.executable)

    def write_record(self, relative, record):
        path = self.root / "cloud" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))

    def write_lifecycle(self, namespace, run_id, stamp="2026-10-06T00:00:00+00:00"):
        for kind in ("training", "validation"):
            self.write_record(f"lifecycle/{namespace}/{kind}.json", {
                "kind": f"stage1-{kind}", "state": "ready", "wandb_run_id": run_id,
                f"{kind}_completed": True, "generated_at": stamp,
            })

    def command(self, *arguments):
        environment = dict(os.environ)
        environment["PATH"] = str(self.root / "bin") + os.pathsep + os.defpath
        return subprocess.run(
            ["/bin/bash", str(self.root / "scripts/download.sh"), *arguments],
            env=environment, capture_output=True, text=True, timeout=30,
        )

    def test_old_run_without_new_artifacts_downloads(self):
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Historical run has no metrics.jsonl", result.stdout)

    def test_existing_historical_logs_are_downloaded(self):
        for name in ("metrics.jsonl", "summary.json"):
            (self.saved / name).write_text("{}")
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "artifacts/runpod/run-fixture/metrics.jsonl").is_file())

    def test_permission_error_is_not_treated_as_missing(self):
        (self.root / "denied").write_text("head-object")
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Access denied", result.stderr)
        self.assertIn("Unable to inspect historical log", result.stderr)

    def test_run_discovery_permission_error_is_not_treated_as_absent(self):
        (self.root / "denied").write_text("list-objects-v2")
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run record request failed", result.stderr)
        self.assertIn("Access denied", result.stderr)
        self.assertFalse((self.root / "artifacts").exists())

    def test_new_run_requires_durable_logs(self):
        (self.saved / "run-manifest.json").write_text(
            json.dumps({"training_resume_contract": {"data_cleaning": {}}})
        )
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metrics.jsonl", result.stderr)

    def test_enabled_calibration_is_required_and_downloaded(self):
        (self.report / "validation-benchmark.json").write_text(
            json.dumps({"config": {"validation": {"calibrate_intervals": True}}})
        )
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("interval-calibration.json", result.stderr)
        (self.report / "interval-calibration.json").write_text("{}")
        # Resume the same local download without overwriting any remote objects.
        result = self.command("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "artifacts/runpod/run-fixture/interval-calibration.json"
        self.assertTrue(target.is_file())

    def test_explicit_run_uses_its_own_lifecycle_not_another_runs(self):
        self.write_lifecycle("runs/run-fixture", "run-fixture")
        self.write_lifecycle("stage1", "run-other", "2026-10-07T00:00:00+00:00")
        result = self.command("run-fixture")
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "artifacts/runpod/run-fixture"
        for kind in ("training", "validation"):
            payload = json.loads((target / f"{kind}-lifecycle.json").read_text())
            self.assertEqual(payload["wandb_run_id"], "run-fixture")

    def test_latest_run_is_selected_from_scoped_records(self):
        self.write_lifecycle("runs/run-fixture", "run-fixture")
        self.write_lifecycle("runs/run-older", "run-older", "2026-10-04T00:00:00+00:00")
        self.write_lifecycle("stage1", "run-other", "2026-10-05T00:00:00+00:00")
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_id=run-fixture", result.stdout)

    def test_current_run_downloads_all_required_artifacts(self):
        self.write_lifecycle("runs/run-fixture", "run-fixture")
        self.write_record("savedModel/run-fixture/run-manifest.json", {
            "training_resume_contract": {"data_cleaning": {}},
        })
        for name in ("metrics.jsonl", "summary.json"):
            (self.saved / name).write_text("{}")
        (self.saved / "resolved-config.yaml").write_text("evaluations_per_epoch: 5\n")
        completion = self.saved / "completion-result"
        completion.mkdir()
        for name in ("adapter.safetensors", "resolved-config.yaml", "training-result.json"):
            (completion / name).write_text("fixture")
        checkpoint = self.saved / "checkpoint-000001"
        checkpoint.mkdir()
        (checkpoint / "adapter.safetensors").write_text("fixture weights")
        self.write_record("evaluations/run-fixture/validation-benchmark.json", {
            "config": {"validation": {"calibrate_intervals": True}},
        })
        (self.report / "interval-calibration.json").write_text("{}")
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "artifacts/runpod/run-fixture"
        for relative in (
            "metrics.jsonl", "summary.json", "interval-calibration.json",
            "checkpoint-000001/adapter.safetensors", "training-completed.json",
            "completion-result/adapter.safetensors", "completion-result/resolved-config.yaml",
            "completion-result/training-result.json",
        ):
            with self.subTest(artifact=relative):
                self.assertTrue((target / relative).is_file())


if __name__ == "__main__":
    unittest.main()
