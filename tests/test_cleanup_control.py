"""Dependency-free checks of the cleanup tools: delete guard, classification, inventories."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME_WEIGHTS = runpy.run_path(str(ROOT / "scripts/runpod_volume_weights.py"))
WANDB_CLEANUP = runpy.run_path(str(ROOT / "scripts/wandb_cleanup.py"))
RUN = "run-20261021T064937Z-c1cf693ab51744a6"
BASELINE = "baseline-" + "5" * 64


class ClassificationTests(unittest.TestCase):
    def test_weights_and_exports_are_deletable_and_records_are_kept(self) -> None:
        classify = VOLUME_WEIGHTS["classify"]
        for key, expected in (
            (f"savedModel/{RUN}/checkpoint-003680/adapter.safetensors", ("weights", RUN)),
            (f"savedModel/{RUN}/checkpoint-003680/optimizer.pt", ("weights", RUN)),
            (f"savedModel/{RUN}/checkpoint-003680/scheduler.pt", ("weights", RUN)),
            (f"baselines/{BASELINE}/jobs/gbdt-42/model.pkl", ("weights", BASELINE)),
            (f"baselines/{BASELINE}/jobs/gru-42/model.pt", ("weights", BASELINE)),
            (f"diagnostics/signal-predictions/{RUN}/checkpoint-003680/export-x/test/predictions.npy",
             ("exports", RUN)),
            (f"diagnostics/forward-predictions/2026-10-20/{RUN}/checkpoint-1/export-x/forward/predictions.npy",
             ("exports", "2026-10-20")),
        ):
            with self.subTest(key=key):
                self.assertEqual(classify(key), expected)
        for key in (
            f"savedModel/{RUN}/checkpoint-003680/trainer-state.json",
            f"savedModel/{RUN}/checkpoint-003680/resolved-config.yaml",
            f"savedModel/{RUN}/checkpoint-leaderboard.json",
            f"savedModel/{RUN}/metrics.jsonl",
            f"baselines/{BASELINE}/complete.json",
            f"baselines/{BASELINE}/jobs/patchtst-42/validation/predictions.npy",
            f"baselines/{BASELINE}/inputs/test/targets.npy",
            f"diagnostics/signal-predictions/{RUN}/checkpoint-003680/export-x/export.json",
            f"diagnostics/signal-predictions/{RUN}/checkpoint-003680/export-x/test/split.json",
            "datasets/" + "a" * 64 + "/prepared/bar-store/shards/bucket-0000/shard.parquet",
            "third_party/Kronos/model.safetensors",
            "models/NeoQuasar/Kronos-base/model.safetensors",
        ):
            with self.subTest(key=key):
                self.assertIsNone(classify(key))

    def test_owner_groups_runs_baselines_and_exports(self) -> None:
        owner_of = VOLUME_WEIGHTS["owner_of"]
        self.assertEqual(owner_of(f"savedModel/{RUN}/checkpoint-1/adapter.safetensors"),
                         f"savedModel/{RUN}")
        self.assertEqual(owner_of(f"baselines/{BASELINE}/jobs/gbdt-42/model.pkl"),
                         f"baselines/{BASELINE}")
        self.assertEqual(owner_of(f"diagnostics/signal-predictions/{RUN}/c/e/test/predictions.npy"),
                         f"diagnostics/signal-predictions/{RUN}")


class WrapperDeleteGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cleanup-guard-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.env_file = root / ".env"
        self.env_file.write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_ACCESS_KEY_ID=fixture\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture\nRUNPOD_S3_REGION=EU-RO-1\n",
            encoding="utf-8",
        )
        self.env_file.chmod(0o600)
        binary = root / "bin"
        binary.mkdir()
        self.calls = root / "aws-calls"
        fake = binary / "aws"
        # The wrapper clears the environment before exec, so the record path is baked in.
        fake.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> '{self.calls}'\n",
                        encoding="utf-8")
        fake.chmod(0o700)
        self.environment = {
            "PATH": f"{binary}:{os.defpath}", "RUNPOD_ENV_FILE": str(self.env_file),
            "HOME": str(root),
        }

    def wrapper(self, *arguments: str, allow: bool) -> subprocess.CompletedProcess:
        env = dict(self.environment)
        if allow:
            env["RUNPOD_S3_ALLOW_DELETE"] = "1"
        return subprocess.run(
            ["bash", str(ROOT / "scripts/runpod_s3_project.sh"), *arguments],
            env=env, capture_output=True, text=True, check=False, timeout=30,
        )

    def test_rm_needs_the_opt_in_and_an_allowed_key(self) -> None:
        allowed = f"s3://fixture-volume/savedModel/{RUN}/checkpoint-003680/adapter.safetensors"
        result = self.wrapper("s3", "rm", allowed, allow=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("RUNPOD_S3_ALLOW_DELETE=1", result.stderr)
        for url in (
            f"s3://fixture-volume/savedModel/{RUN}/checkpoint-003680/trainer-state.json",
            f"s3://fixture-volume/savedModel/{RUN}/metrics.jsonl",
            f"s3://fixture-volume/baselines/{BASELINE}/complete.json",
            f"s3://fixture-volume/baselines/{BASELINE}/jobs/gru-42/validation/predictions.npy",
            "s3://fixture-volume/datasets/" + "a" * 64 + "/prepared/bar-store/bar-store.json",
            f"s3://other-volume/savedModel/{RUN}/checkpoint-003680/adapter.safetensors",
            f"s3://fixture-volume/savedModel/../{RUN}/checkpoint-003680/adapter.safetensors",
        ):
            with self.subTest(url=url):
                result = self.wrapper("s3", "rm", url, allow=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.calls.exists())
        result = self.wrapper("s3", "rm", allowed, "--recursive", allow=True)
        self.assertEqual(result.returncode, 2)
        result = self.wrapper("s3", "rm", allowed, allow=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"s3 rm {allowed}", self.calls.read_text(encoding="utf-8"))


class InventoryToolTests(unittest.TestCase):
    def test_volume_delete_refuses_a_foreign_or_unlisted_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inventory = Path(temporary) / "inventory.json"
            inventory.write_text(json.dumps({
                "kind": VOLUME_WEIGHTS["INVENTORY_KIND"], "volume_id": "other-volume",
                "owners": {}, "deletable": {"weights": [], "exports": []},
            }))
            result = subprocess.run(
                ["python3", str(ROOT / "scripts/runpod_volume_weights.py"), "delete",
                 "--inventory", str(inventory), "--category", "weights"],
                env={**os.environ, "RUNPOD_NETWORK_VOLUME_ID": "fixture-volume"},
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("does not belong", result.stderr)
            inventory.write_text(json.dumps({
                "kind": VOLUME_WEIGHTS["INVENTORY_KIND"], "volume_id": "fixture-volume",
                "owners": {},
                "deletable": {"weights": [
                    {"key": f"savedModel/{RUN}/checkpoint-1/adapter.safetensors", "bytes": 10},
                    {"key": f"savedModel/{RUN}/checkpoint-1/trainer-state.json", "bytes": 10},
                ], "exports": []},
            }))
            # Without --apply nothing is deleted and records in the list are ignored.
            result = subprocess.run(
                ["python3", str(ROOT / "scripts/runpod_volume_weights.py"), "delete",
                 "--inventory", str(inventory), "--category", "weights"],
                env={**os.environ, "RUNPOD_NETWORK_VOLUME_ID": "fixture-volume"},
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Would delete 1 weights files", result.stdout)

    def test_wandb_cleanup_reads_dotenv_values_without_executing_them(self) -> None:
        dotenv_value = WANDB_CLEANUP["dotenv_value"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(textwrap.dedent("""
                export WANDB_API_KEY="fixture-key"
                WANDB_PROJECT=stock_forecasting
                WANDB_ENTITY=
                OTHER=$(echo injected)
            """), encoding="utf-8")
            self.assertEqual(dotenv_value(path, "WANDB_API_KEY"), "fixture-key")
            self.assertEqual(dotenv_value(path, "WANDB_PROJECT"), "stock_forecasting")
            self.assertEqual(dotenv_value(path, "WANDB_ENTITY"), "")
            self.assertEqual(dotenv_value(path, "OTHER"), "$(echo injected)")
        parser = WANDB_CLEANUP["build_parser"]()
        arguments = parser.parse_args(["delete", "--inventory", "x.json"])
        self.assertFalse(arguments.apply)
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
        self.assertIn("volume-weights)", workflow)

    def test_wandb_artifact_uploads_are_off_by_default(self) -> None:
        config = (ROOT / "src/stock_forecasting/config.py").read_text(encoding="utf-8")
        self.assertIn("log_model_artifact: bool = False", config)
        for name in ("stage1_kronos_base_lora.yaml", "stage2_kronos_base_lora.yaml",
                     "local_mock.yaml"):
            self.assertIn("log_model_artifact: false", (ROOT / "configs" / name).read_text())
        guard = "if config.wandb.log_model_artifact:"
        for path in ("src/stock_forecasting/validation_benchmark.py",
                     "src/stock_forecasting/cli/benchmark_baselines.py"):
            source = (ROOT / path).read_text(encoding="utf-8")
            self.assertEqual(source.count("log_artifact("), source.count(guard), path)
        training = (ROOT / "src/stock_forecasting/training.py").read_text(encoding="utf-8")
        self.assertEqual(training.count("tracking.log_model_artifact("), training.count(guard))


if __name__ == "__main__":
    unittest.main()
