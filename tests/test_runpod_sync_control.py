"""Dependency-free Bash and source-sync regression tests; no cloud access."""

from __future__ import annotations

import json
import os
import runpy
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = "/bin/bash"


def fixture_s3(arguments):
    """Implement only the source-sync S3 operations against temporary storage."""
    volume = Path(os.environ["SYNC_FIXTURE_VOLUME"])
    log = Path(os.environ["SYNC_FIXTURE_LOG"])

    def option(name):
        return arguments[arguments.index(name) + 1]

    def remote_path(uri):
        prefix = "s3://fixture-volume/"
        if not uri.startswith(prefix):
            raise ValueError("Unexpected fixture bucket")
        relative = Path(uri[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected fixture object path")
        return volume / relative

    if arguments[:2] == ["s3api", "head-bucket"]:
        assert option("--bucket") == "fixture-volume"
    elif arguments[:2] == ["s3", "cp"]:
        source, target = arguments[2:4]
        if source.startswith("s3://") and target == "-":
            sys.stdout.buffer.write(remote_path(source).read_bytes())
        elif target.startswith("s3://"):
            data = sys.stdin.buffer.read() if source == "-" else Path(source).read_bytes()
            path = remote_path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            event = {"key": str(path.relative_to(volume))}
            if event["key"] == "lifecycle/stage1/code.json":
                event["state"] = json.loads(data)["state"]
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
        else:
            raise ValueError("Unsupported fixture copy")
    elif arguments[:2] == ["s3api", "head-object"]:
        assert option("--bucket") == "fixture-volume"
        print(remote_path("s3://fixture-volume/" + option("--key")).stat().st_size)
    elif arguments[:2] == ["s3api", "list-objects-v2"]:
        assert option("--bucket") == "fixture-volume"
        assert option("--delimiter") == "/"
        prefix = option("--prefix")
        directory = remote_path("s3://fixture-volume/" + prefix)
        contents, prefixes = [], []
        for path in sorted(directory.iterdir()):
            key = str(path.relative_to(volume))
            if path.is_dir():
                prefixes.append({"Prefix": key + "/"})
            else:
                contents.append({"Key": key})
        print(json.dumps({"Contents": contents, "CommonPrefixes": prefixes}))
    else:
        raise ValueError("Unexpected S3 operation; cloud access is forbidden")


@unittest.skipUnless(Path(BASH).is_file(), "The workflow requires Bash")
class RunPodSyncControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runpod-sync-control-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.project = root / "project"
        self.volume = root / "volume"
        self.volume.mkdir()
        self.log = root / "s3-writes.jsonl"
        for name in ("src", "scripts", "configs", "cloudrun/tpex-relay"):
            shutil.copytree(
                ROOT / name, self.project / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"),
            )
        for name in (
            "README.md", "pyproject.toml", "LICENSE", "MODEL_LICENSE",
            "THIRD_PARTY_NOTICES.md", "RELEASES.md",
        ):
            shutil.copy2(ROOT / name, self.project / name)
        (self.project / "tests").mkdir()
        transport = self.project / "tests" / Path(__file__).name
        shutil.copy2(__file__, transport)
        (self.project / "scripts/runpod_s3_project.sh").write_text(
            "#!/usr/bin/env bash\nexec python3 " + shlex.quote(str(transport))
            + ' --fixture-s3 "$@"\n', encoding="utf-8",
        )
        environment_file = self.project / ".env"
        # Generate credentials so the real secret scan cannot match fixture source literals.
        environment_file.write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_REGION=fixture-region\n"
            + "RUNPOD_S3_ACCESS_KEY_ID=" + secrets.token_hex(16) + "\n"
            + "RUNPOD_S3_SECRET_ACCESS_KEY=" + secrets.token_hex(16) + "\n",
            encoding="utf-8",
        )
        environment_file.chmod(0o600)
        binary = root / "bin"
        binary.mkdir()
        (binary / "python3").symlink_to(sys.executable)
        # Exercise macOS Bash 3.2 explicitly, even when PATH normally selects newer Bash.
        (binary / "bash").symlink_to(BASH)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "FIN_TS_", "AWS_"))
        }
        self.environment.update({
            "PATH": f"{binary}:{os.defpath}", "RUNPOD_ENV_FILE": str(environment_file),
            "SYNC_FIXTURE_VOLUME": str(self.volume), "SYNC_FIXTURE_LOG": str(self.log),
            "RUNPOD_S3_RETRY_MAX_ATTEMPTS": "1",
        })
        self.selection = runpy.run_path(str(self.project / "scripts/runpod_selection.py"))
        self.activate()

    def activate(self, start="2021-01-01"):
        arguments = self.selection["build_parser"]().parse_args([
            "create", "--project-root", str(self.project),
            "--stage", "stage2", "--data-profile", "us_tw_eodhd",
            "--start", start, "--end", "2026-06-01", "--universe", "all",
        ])
        self.selected = self.selection["_build_selection"](arguments, self.project)
        self.selection_path = self.selection["_activate_selection"](self.project, self.selected)

    def load_selection(self, consumer=None):
        command = 'set -Eeuo pipefail; source "$1/scripts/lib/runpod_selection.sh"; '
        command += 'runpod_load_active_selection "$1"'
        if consumer is not None:
            command += ' "$2"'
        command += '; printf "%s\\n%s\\n" "$RUNPOD_SELECTION_ID" "$RUNPOD_DATASET_REQUEST_SHA256"'
        return subprocess.run(
            [BASH, "-c", command, "selection-test", str(self.project)]
            + ([] if consumer is None else [consumer]),
            env=self.environment, capture_output=True, text=True, timeout=30,
        )

    def sync(self):
        return subprocess.run(
            [BASH, str(self.project / "scripts/sync_project_to_runpod_volume.sh"), "--apply"],
            env=self.environment, capture_output=True, text=True, timeout=120,
        )

    def writes(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def change_main_config(self):
        path = self.project / self.selected["stage"]["config_path"]
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n# A training-only fixture change.\n")

    def test_default_training_and_baseline_load_both_ab_selections_under_nounset(self):
        for start in ("2021-01-01", "2016-01-01"):
            self.activate(start)
            for consumer in (None, "training", "baseline"):
                with self.subTest(start=start, consumer=consumer):
                    result = self.load_selection(consumer)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(result.stdout.splitlines(), [
                        self.selected["selection_id"], self.selected["dataset_request_sha256"],
                    ])

    def test_default_consumer_retains_main_config_validation(self):
        self.change_main_config()
        for consumer in (None, "training"):
            with self.subTest(consumer=consumer):
                result = self.load_selection(consumer)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No valid active RunPod selection", result.stderr)
                self.assertNotIn("unbound variable", result.stderr)
        result = self.load_selection("baseline")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tampered_selection_is_rejected_for_all_consumers(self):
        payload = json.loads(self.selection_path.read_text())
        payload["dataset_request"]["date_range"]["start_inclusive"] = "1990-01-01"
        self.selection_path.write_text(json.dumps(payload), encoding="utf-8")
        # Existing parent-shell values must not conceal a failed selection export.
        self.environment.update(self.selection["_selection_exports"](
            self.selection_path, self.selected
        ))
        for consumer in (None, "training", "baseline"):
            with self.subTest(consumer=consumer):
                result = self.load_selection(consumer)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No valid active RunPod selection", result.stderr)

    def test_sync_apply_publishes_selection_before_ready_without_dataset_writes(self):
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("code readiness publication completed successfully", result.stdout)
        writes = self.writes()
        markers = [event for event in writes if "state" in event]
        self.assertEqual([event["state"] for event in markers], ["syncing", "ready"])
        selection_key = f"lifecycle/selections/{self.selected['selection_id']}.json"
        self.assertEqual(writes[-2], {"key": selection_key})
        self.assertEqual(writes[-1], markers[-1])
        self.assertEqual(json.loads((self.volume / selection_key).read_text()), self.selected)
        for event in writes:
            self.assertTrue(
                event["key"].startswith("stock_forecasting/")
                or event["key"] in {"lifecycle/stage1/code.json", selection_key}, event,
            )
        marker = json.loads((self.volume / "lifecycle/stage1/code.json").read_text())
        source_writes = [event for event in writes if event["key"].startswith("stock_forecasting/")]
        self.assertEqual(marker["file_count"], len(source_writes))

    def test_sync_apply_keeps_syncing_when_selection_is_invalid(self):
        self.change_main_config()
        result = self.sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No valid active RunPod selection", result.stderr)
        self.assertNotIn("unbound variable", result.stderr)
        self.assertEqual([event["state"] for event in self.writes() if "state" in event], ["syncing"])
        self.assertFalse((self.volume / "lifecycle/selections").exists())


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fixture-s3"]:
        fixture_s3(sys.argv[2:])
    else:
        unittest.main()
