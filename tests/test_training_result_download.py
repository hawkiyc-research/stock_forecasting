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
if args[:2] == ["s3api", "head-object"]:
    if (root / "denied").exists():
        sys.stderr.write("An error occurred (403): Access denied")
        sys.exit(1)
    key = args[args.index("--key") + 1]
    if not (root / "cloud" / key).is_file():
        sys.stderr.write("An error occurred (404): Not Found")
        sys.exit(1)
    print("{}")
else:
    assert args[:2] == ["s3", "cp"]
    source = root / "cloud" / args[2].removeprefix("s3://fixture-volume/")
    if args[3] == "-":
        sys.stdout.buffer.write(source.read_bytes())
    elif "--recursive" in args:
        if source.is_dir():
            shutil.copytree(source, args[3], dirs_exist_ok=True)
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
        for relative in (
            "lifecycle/stage1/training.json", "lifecycle/stage1/validation.json",
            "lifecycle/runs/run-fixture/training-completed.json",
        ):
            path = self.root / "cloud" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"state": "ready", "wandb_run_id": "run-fixture"}))
        (self.root / "bin").mkdir()
        (self.root / "bin/python3").symlink_to(sys.executable)

    def command(self):
        environment = dict(os.environ)
        environment["PATH"] = str(self.root / "bin") + os.pathsep + os.defpath
        return subprocess.run(
            ["/bin/bash", str(self.root / "scripts/download.sh")],
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
        (self.root / "denied").touch()
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Access denied", result.stderr)

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
        (self.report / "interval-calibration.json").write_text("{}")
        # Resume the same local download without overwriting any remote objects.
        environment = dict(os.environ)
        environment["PATH"] = str(self.root / "bin") + os.pathsep + os.defpath
        result = subprocess.run(
            ["/bin/bash", str(self.root / "scripts/download.sh"), "--resume"],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "artifacts/runpod/run-fixture/interval-calibration.json"
        self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
