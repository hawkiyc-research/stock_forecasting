"""Dependency-free tests for cloud regression evidence, not local model execution."""

from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "scripts/verify_cloud_regression.py"))


class CloudRegressionControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.journal = MODULE["RegressionJournal"](self.root / "output", max_files=2, max_bytes=128)
        self.addCleanup(self.journal.close)

    def test_builder_runs_first_without_deselection(self):
        names = [
            "tests/a.py::test_other",
            "tests/b.py::test_complete_baseline_builder_and_cache_reuse",
        ]
        items = [SimpleNamespace(nodeid=name) for name in names]
        self.journal.pytest_collection_modifyitems(items)
        self.assertEqual([item.nodeid for item in items], names[::-1])
        self.assertEqual(json.loads((self.root / "output/collected.json").read_text()), names[::-1])

    def test_bounded_state_and_per_test_events(self):
        fixture = self.root / "fixture"
        job = fixture / "baselines/baseline-fixture/jobs/gru-42"
        job.mkdir(parents=True)
        (job / "progress.json").write_text('{"samples":32}')
        (job / "complete.json").write_text(json.dumps({"too_large": "x" * 200}))
        (job / "model.pt").write_bytes(b"never copy weights")
        self.journal.pytest_runtest_logstart("tests/a.py::test_x")
        self.journal.pytest_runtest_call(
            SimpleNamespace(nodeid="tests/a.py::test_x", funcargs={"tmp_path": fixture})
        )
        state = json.loads((self.root / "output/active.json").read_text())
        self.assertEqual(len(state["files"]), 1)
        self.assertEqual(len(state["excluded"]), 1)
        self.assertNotIn("model.pt", json.dumps(state))
        self.journal.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="tests/a.py::test_x", when="call", outcome="passed", duration=1.2
            )
        )
        self.journal.pytest_runtest_teardown()
        self.assertIsNone(self.journal.active)
        events = [
            json.loads(line)
            for line in (self.root / "output/events.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["event"] for row in events], ["start", "report"])

    def test_observer_failure_invalidates_acceptance(self):
        self.journal.errors.append("fixture observer failed")
        session = SimpleNamespace(exitstatus=0)
        self.journal.pytest_sessionfinish(session)
        self.assertEqual(session.exitstatus, 3)

    def test_source_verification_fails_on_stale_test_file(self):
        project = self.root / "project"
        project.mkdir()
        source = project / "test_sample.py"
        source.write_text("VALUE = 1\n")
        marker = self.root / "code.json"
        marker.write_text(
            json.dumps(
                {
                    "state": "ready",
                    "files": [
                        {
                            "path": source.name,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        MODULE["verify_sources"](project, marker, self.root / "output")
        source.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "Deployed source mismatch"):
            MODULE["verify_sources"](project, marker, self.root / "output")


if __name__ == "__main__":
    unittest.main()
