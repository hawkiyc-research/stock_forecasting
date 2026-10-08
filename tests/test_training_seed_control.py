"""Exercise seed launch/configuration boundaries without a local ML environment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
CONFIGURATION = runpy.run_path(str(ROOT / "src/stock_forecasting/runpod/configuration.py"))


class TrainingSeedControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="seed-control-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        shutil.copytree(ROOT / "configs", self.root / "configs")
        self.original = SELECTION["_build_selection"](
            argparse.Namespace(
                stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
                start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
                stocks=[], etfs=[], symbol_limit=None, feature_mode="combined",
            ), self.root,
        )

    def pin(self, seed):
        selected = SELECTION["_with_training_seed"](self.original, seed, self.root)
        path = SELECTION["_store_selection"](self.root, selected)
        env = SELECTION["_selection_exports"](path, selected)
        env.update(PROJECT_ROOT=str(self.root), NETWORK_VOLUME_ROOT="/runpod-volume",
                   RUNPOD_REMOTE_SELECTION_PATH=str(path))
        config = self.root / selected["stage"]["config_path"]
        return config, selected, path, env

    def test_three_seeds_and_uint32_boundaries_resolve_without_editing_yaml(self):
        for seed in (0, 42, 43, 44, 2**32 - 1):
            with self.subTest(seed=seed):
                config, selected, path, env = self.pin(seed)
                before = config.read_bytes()
                SELECTION["_verify_environment"](path, selected, env)
                self.assertEqual(CONFIGURATION["training_seed_override"](config, env), seed)
                self.assertEqual(config.read_bytes(), before)

    def test_legacy_selection_does_not_require_or_override_seed(self):
        path = SELECTION["_store_selection"](self.root, self.original)
        env = SELECTION["_selection_exports"](path, self.original)
        env.update(NETWORK_VOLUME_ROOT="/runpod-volume", RUNPOD_REMOTE_SELECTION_PATH=str(path))
        del env["RUNPOD_TRAINING_SEED"]
        SELECTION["_verify_environment"](path, self.original, env)
        self.assertIsNone(CONFIGURATION["training_seed_override"](self.root / "not-read.yaml", env))

    def test_effective_config_changes_training_seed_only(self):
        path, _, _, env = self.pin(43)
        fixture = types.SimpleNamespace(
            training=types.SimpleNamespace(seed=42), data=types.SimpleNamespace(calibration_seed=59)
        )
        module = types.ModuleType("stock_forecasting.config")
        module.ExperimentConfig = types.SimpleNamespace(
            from_yaml=lambda path: copy.deepcopy(fixture)
        )
        with (
            patch.dict(sys.modules, {"stock_forecasting.config": module}),
            patch.dict(os.environ, env),
        ):
            resolved = CONFIGURATION["load_run_config"](path)
        self.assertEqual(resolved.training.seed, 43)
        self.assertEqual(resolved.data.calibration_seed, 59)
        self.assertEqual(fixture.training.seed, 42)

    def test_saved_resolved_config_is_never_overridden_by_another_runs_environment(self):
        _, _, _, env = self.pin(43)
        saved = self.root / "savedModel/run-previous/checkpoint-000001/resolved-config.yaml"
        self.assertIsNone(CONFIGURATION["training_seed_override"](saved, env))

    def test_seed_env_must_match_pinned_selection_and_source_config(self):
        config, selected, path, env = self.pin(43)
        for key, value in (
            ("RUNPOD_TRAINING_SEED", "42"), ("RUNPOD_TRAINING_SEED", "-1"),
            ("RUNPOD_TRAINING_SEED", "4294967296"), ("RUNPOD_TRAINING_SEED", "4.3"),
            ("RUNPOD_SELECTION_ID", "different"), ("RUNPOD_SELECTION_SHA256", "0" * 64),
            ("RUNPOD_CONFIG", ""), ("RUNPOD_REMOTE_SELECTION_PATH", str(self.root / "missing")),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CONFIGURATION["training_seed_override"](config, {**env, key: value})
        for seed in ("", "42"):
            with self.assertRaises(ValueError):
                SELECTION["_verify_environment"](
                    path, selected, {**env, "RUNPOD_TRAINING_SEED": seed}
                )
        config.write_bytes(config.read_bytes() + b"\n# Changed fixture\n")
        with self.assertRaises(ValueError):
            CONFIGURATION["training_seed_override"](config, env)

    def test_invalid_selection_cannot_supply_seed(self):
        config, selected, path, env = self.pin(43)
        broken = copy.deepcopy(selected)
        broken["stage"]["training_seed"] = 44
        for payload in ([], {"stage": []}, broken):
            path.write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                CONFIGURATION["training_seed_override"](config, env)

    def test_ssh_environment_keeps_pinned_seed_and_clears_a_stale_seed(self):
        source = self.root / "pid1"
        base = b"RUNPOD_POD_ID=fixture\0RUNPOD_ROLE=gpu-train\0"
        for seed in ("0", "43", ""):
            entry = b"RUNPOD_TRAINING_SEED=" + seed.encode() + b"\0" if seed else b""
            source.write_bytes(base + entry)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/runpod_reexec_with_pid1_env.py"),
                 "--source", str(source), "--", sys.executable, "-c",
                 "import os; print(os.environ.get('RUNPOD_TRAINING_SEED', 'absent'))"],
                env={**os.environ, "RUNPOD_TEST_MODE": "1", "RUNPOD_TRAINING_SEED": "999"},
                capture_output=True, text=True, timeout=10, check=True,
            )
            self.assertEqual(result.stdout.strip(), seed or "absent")

    def test_workflow_passes_repeated_seeds_without_changing_other_arguments(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy2(ROOT / "scripts/runpod_workflow.sh", scripts)
        shutil.copytree(ROOT / "scripts/lib", scripts / "lib")
        (scripts / "runpod_concurrency.py").write_text(
            "import json,os,sys\nprint(json.dumps({'args':sys.argv[1:],"
            "'gpu':os.environ['RUNPOD_CLI_GPU_ID'],'runtime':os.environ['RUNPOD_CLI_MAX_RUNTIME_SECONDS']}))\n"
        )
        result = subprocess.run(
            ["bash", str(scripts / "runpod_workflow.sh"), "train", "--seed", "42", "--seed", "43",
             "--seed", "44", "--experiment", "a-adaptive64", "--maxRuntime", "48h",
             "--gpuId", "NVIDIA GeForce RTX 5090", "--launchWorkers", "3"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["args"], ["launch", "--seed", "42", "--seed", "43", "--seed", "44",
                                          "--experiment", "a-adaptive64", "--launchWorkers", "3"])
        self.assertEqual(output["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(output["runtime"], "172800")
        for args in (("train", "--seed"), ("baseline", "--seed", "42")):
            invalid = subprocess.run(["bash", str(scripts / "runpod_workflow.sh"), *args],
                                     capture_output=True, text=True, timeout=10)
            self.assertNotEqual(invalid.returncode, 0)

    def test_checkpoint_preflight_checks_pinned_seed_without_changing_old_runs(self):
        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            preflight = runpy.run_path(str(ROOT / "scripts/runpod_remote_checkpoint_preflight.py"))
        contract = {"training": {"seed": 43}, "dataset_artifacts": {}}
        digest = SELECTION["_payload_sha256"](contract)
        manifest = {
            "run_id": "run-fixture", "run_key": "run-fixture",
            "schema_version": preflight["readiness"].CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
            "source_config_sha256": "a" * 64,
            "resolved_config_sha256": hashlib.sha256(b"resolved").hexdigest(),
            "training_resume_contract": contract, "training_resume_contract_sha256": digest,
        }
        validate = preflight["_validate_run_manifest"]
        args = (manifest, "run-fixture", digest, "a" * 64, b"resolved", "dataset", {}, "b" * 64)
        with patch.dict(validate.__globals__, _dataset_contract=lambda *args: {}):
            for seed in ("", "43"):
                with patch.dict(os.environ, {"RUNPOD_TRAINING_SEED": seed}):
                    validate(*args)
            with patch.dict(os.environ, {"RUNPOD_TRAINING_SEED": "42"}), self.assertRaisesRegex(
                ValueError, "training seed differs"
            ):
                validate(*args)


if __name__ == "__main__":
    unittest.main()
