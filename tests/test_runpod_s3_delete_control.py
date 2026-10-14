"""Only an exact selection-record object may use the S3 delete API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class S3DeleteControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="s3-delete-control-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("scripts/runpod_s3_project.sh", "scripts/lib/runpod_project_env.sh"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (self.root / ".env").write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_REGION=EU-RO-1\n"
            "RUNPOD_S3_ACCESS_KEY_ID=fixture-access-only\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture-secret-only\n")
        (self.root / ".env").chmod(0o600)
        (self.root / "bin").mkdir()
        aws = self.root / "bin/aws"
        aws.write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\n"
                       "Path(__file__).resolve().parents[1].joinpath('request.json').write_text(json.dumps(sys.argv[1:]))\n"
                       "print('{}')\n")
        aws.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                    "RUNPOD_ENV_FILE": str(self.root / ".env")}
        self.key = "lifecycle/runs/run-20261014T080044Z-012345/selection.json"

    def invoke(self, arguments):
        return subprocess.run(["bash", str(self.root / "scripts/runpod_s3_project.sh"), *arguments],
                              env=self.env, capture_output=True, text=True, check=False)

    def test_exact_configured_bucket_and_selection_key_are_forwarded(self):
        args = ["s3api", "delete-object", "--bucket", "fixture-volume", "--key", self.key]
        result = self.invoke(args)
        self.assertEqual(result.returncode, 0, result.stderr)
        request = json.loads((self.root / "request.json").read_text())
        self.assertEqual(request[:6], args)
        self.assertEqual(request[-2:], ["--endpoint-url", "https://s3api-eu-ro-1.runpod.io/"])

    def test_other_objects_and_traversal_are_rejected_without_calling_aws(self):
        for key in ("savedModel/run-example/selection.json", "data/selection.json",
                    "lifecycle/runs/run-example/pair.json", "lifecycle/runs/run-example/training.json",
                    "lifecycle/runs/../selection.json", "lifecycle/runs/run-example/../selection.json",
                    "lifecycle/runs/run--example/selection.json", "lifecycle/runs/run..example/selection.json",
                    "lifecycle/runs/run-example/selection.json/child", "lifecycle/runs/*/selection.json",
                    "lifecycle/runs/run-example%2Felsewhere/selection.json", "lifecycle/runs/./selection.json"):
            with self.subTest(key=key):
                result = self.invoke(["s3api", "delete-object", "--bucket", "fixture-volume", "--key", key])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "request.json").exists())

    def test_other_bucket_extra_flags_and_other_delete_operations_are_rejected(self):
        valid = ["s3api", "delete-object", "--bucket", "fixture-volume", "--key", self.key]
        cases = [valid[:3] + ["another-volume"] + valid[4:],
                 ["s3api", "delete-objects", "--bucket", "fixture-volume"],
                 ["s3", "rm", "s3://fixture-volume/lifecycle/runs/", "--recursive"],
                 ["s3api", "delete-object", "--key", self.key, "--bucket", "fixture-volume"],
                 valid[:-1], valid + ["--recursive"], valid + ["--version-id", "1"],
                 valid + ["--output", "json"], valid + ["--endpoint-url", "https://example.com"],
                 valid + ["--bucket", "another-volume"]]
        for args in cases:
            with self.subTest(args=args):
                result = self.invoke(args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "request.json").exists())

    def test_existing_read_and_copy_operations_remain_available(self):
        for args in (["s3api", "list-objects-v2", "--bucket", "fixture-volume"],
                     ["s3api", "head-object", "--bucket", "fixture-volume", "--key", "data/file"],
                     ["s3", "cp", "s3://fixture-volume/data/file", "-"],
                     ["s3", "cp", "local-file", "s3://fixture-volume/code/file"]):
            with self.subTest(args=args):
                result = self.invoke(args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((self.root / "request.json").read_text())[:len(args)], args)


if __name__ == "__main__":
    unittest.main()
