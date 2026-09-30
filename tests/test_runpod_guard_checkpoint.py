"""Verify bounded local checkpoint checks used by the paid GPU guard."""

import shutil
import runpy
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GuardCheckpointTests(unittest.TestCase):
    def run_check(self, preflight_source: str, timeout: int = 2) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory)
            shutil.copyfile(ROOT / "scripts/runpod_guard_checkpoint.py", scripts / "runpod_guard_checkpoint.py")
            (scripts / "runpod_remote_checkpoint_preflight.py").write_text(
                preflight_source, encoding="utf-8"
            )
            return subprocess.run(
                [
                    sys.executable, str(scripts / "runpod_guard_checkpoint.py"),
                    "--s3-wrapper", str(scripts / "s3.sh"),
                    "--bucket", "fixture-volume",
                    "--run-id", "fixture-run",
                    "--config", str(scripts / "config.yaml"),
                    "--dataset-request-sha256", "0" * 64,
                    "--created-after", "2026-01-01T00:00:00Z",
                    "--timeout-seconds", str(timeout),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )

    def test_reports_only_a_successful_preflight(self) -> None:
        successful = self.run_check('print("checkpoint-000123")\n')
        self.assertEqual(successful.returncode, 0)
        self.assertEqual(successful.stdout.strip(), "checkpoint-000123")
        failed = self.run_check('raise SystemExit(3)\n')
        self.assertEqual(failed.returncode, 3)

    def test_preflight_timeout_is_bounded(self) -> None:
        result = self.run_check('import time\ntime.sleep(10)\n', timeout=1)
        self.assertEqual(result.returncode, 3)
        self.assertIn("timed out", result.stderr)

    def test_resume_checkpoint_must_be_newer_than_this_pod_guard(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            require_new = runpy.run_path(
                str(ROOT / "scripts/runpod_remote_checkpoint_preflight.py")
            )["_require_checkpoint_created_after"]
        finally:
            sys.path.pop(0)
        started = datetime(2026, 10, 1, tzinfo=UTC)
        require_new("2026-10-01T00:00:01+00:00", started)
        with self.assertRaisesRegex(ValueError, "predates"):
            require_new("2026-09-30T23:59:59+00:00", started)
        with self.assertRaisesRegex(ValueError, "predates"):
            require_new("2026-10-01T00:00:01", started)


if __name__ == "__main__":
    unittest.main()
