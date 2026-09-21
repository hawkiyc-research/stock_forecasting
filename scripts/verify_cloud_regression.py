#!/usr/bin/env python3
"""Run cloud-only regression with durable per-test and bounded child-job evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def verify_sources(project, marker, output):
    payload = json.loads(marker.read_text())
    verified = {}
    if payload.get("state") != "ready" or not payload.get("files"):
        raise ValueError("A ready, nonempty source manifest is required")
    for row in payload["files"]:
        path = project / row["path"]
        if not path.resolve().is_relative_to(project.resolve()) or path.is_symlink():
            raise ValueError("Unsafe source manifest path")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Deployed source mismatch: {row['path']}")
        verified[row["path"]] = digest
    write_json(output / "verified-sources.json", verified)


class RegressionJournal:
    """Observe small JSON state only; never copy fixtures, datasets or checkpoints."""

    STATE_NAMES = frozenset(
        {
            "resource-plan.json",
            "live-resources.json",
            "progress.json",
            "execution.json",
            "failure.json",
            "runtime-plan.json",
            "complete.json",
        }
    )

    def __init__(self, output, interval=5.0, max_files=64, max_bytes=262144):
        if interval <= 0 or max_files < 1 or max_bytes < 1:
            raise ValueError("Snapshot limits must be positive")
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.interval, self.max_files, self.max_bytes = interval, max_files, max_bytes
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active = None
        self.errors = []
        self.thread = threading.Thread(target=self.monitor, name="qa-journal", daemon=True)

    def record(self, event, **values):
        with self.lock, (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "event": event, **values}) + "\n")

    def pytest_collection_modifyitems(self, items):
        # Diagnose the previous timeout first without deselecting any tests.
        items.sort(
            key=lambda item: "::test_complete_baseline_builder_and_cache_reuse" not in item.nodeid
        )
        write_json(self.output / "collected.json", [item.nodeid for item in items])

    def pytest_runtest_logstart(self, nodeid):
        self.record("start", nodeid=nodeid)

    def pytest_runtest_call(self, item):
        with self.lock:
            self.active = (item.nodeid, item.funcargs.get("tmp_path"))
        self.snapshot()

    def pytest_runtest_logreport(self, report):
        self.record(
            "report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            duration=report.duration,
        )
        if report.when == "call":
            self.snapshot()

    def pytest_runtest_teardown(self):
        with self.lock:
            self.snapshot()
            self.active = None

    def pytest_sessionfinish(self, session):
        if self.errors:
            session.exitstatus = 3

    def snapshot(self):
        with self.lock:
            if self.active is None:
                return
            nodeid, root = self.active
            state = {"nodeid": nodeid, "time": time.time(), "files": {}, "excluded": []}
            if root is not None:
                root = Path(root)
                paths = sorted(
                    path
                    for pattern in ("baselines/*/*.json", "baselines/*/jobs/*/*.json")
                    for path in root.glob(pattern)
                    if path.name in self.STATE_NAMES
                )
                state["candidate_files"] = len(paths)
                for path in paths[: self.max_files]:
                    try:
                        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                            raise ValueError("Snapshot must stay within the current test fixture")
                        relative = str(path.relative_to(root))
                        with path.open("rb") as stream:
                            content = stream.read(self.max_bytes + 1)
                        if len(content) > self.max_bytes:
                            state["excluded"].append(relative)
                            continue
                        state["files"][relative] = json.loads(content)
                    except FileNotFoundError:
                        # Fixture cleanup or atomic publication can remove a candidate.
                        continue
            key = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
            write_json(self.output / f"state-{key}.json", state)
            write_json(self.output / "active.json", state)

    def monitor(self):
        while not self.stop.wait(self.interval):
            try:
                self.snapshot()
            except Exception as error:
                with self.lock:
                    self.errors.append(f"{type(error).__name__}: {error}")
                    write_json(self.output / "observer-errors.json", self.errors)
                return

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Regression journal failed to stop within 30 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-seconds", type=float, default=5)
    parser.add_argument("--max-snapshot-files", type=int, default=64)
    parser.add_argument("--max-snapshot-bytes", type=int, default=262144)
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_ROLE") != "gpu-baseline":
        raise RuntimeError("Regression must run on an authorized verification GPU Pod")
    project = Path(__file__).resolve().parents[1]
    volume = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    if not args.output.resolve().is_relative_to((volume / "diagnostics").resolve()):
        raise ValueError("Verification output must remain under volume diagnostics")
    args.output.mkdir(parents=True, exist_ok=True)
    verify_sources(project, volume / "lifecycle/stage1/code.json", args.output)
    # Import pytest only after cloud and source-identity checks.
    import pytest

    journal = RegressionJournal(
        args.output / "journal",
        args.snapshot_seconds,
        args.max_snapshot_files,
        args.max_snapshot_bytes,
    )
    journal.thread.start()
    try:
        result = int(
            pytest.main(
                [
                    "tests",
                    "-vv",
                "-ra",
                "--capture=tee-sys",
                    "--durations=25",
                    "-o",
                    "addopts=",
                    "-o",
                    "faulthandler_timeout=180",
                    f"--junitxml={args.output / 'pytest.xml'}",
                ],
                plugins=[journal],
            )
        )
    finally:
        journal.close()
    if journal.errors:
        result = 3
    write_json(
        args.output / "pytest-status.json", {"exit_code": result, "observer_errors": journal.errors}
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
