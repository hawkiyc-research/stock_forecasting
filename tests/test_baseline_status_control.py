"""Exercise baseline completion and cleaning together without an ML environment or Pod."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = runpy.run_path(str(ROOT / "tests/test_baseline_readiness_control.py"))
BASE = FIXTURES["BASE"]
GATE = FIXTURES["GATE"]
CONTRACT = BASE["CONTRACT"]
encode = BASE["encode"]


def completed_fixture(project, selected, objects):
    identity = CONTRACT["baseline_contract"](project, selected)
    _, _, splits = GATE["baseline_sources"](project, selected)
    counts, data_sources = {}, {}
    for split, dataset_id in splits.items():
        prefix = f"datasets/{dataset_id}/prepared/"
        raw = objects[prefix + "bar-store/bar-store.json"]
        source_hash = hashlib.sha256(raw).hexdigest()
        expected = GATE["cleaning_identity"](
            project, json.loads(raw), source_hash, identity["contract"]["data_cleaning"]
        )
        key = GATE["digest"](expected)
        universe = json.loads(objects[prefix + f"sample-universes/{key}/universe.json"])
        counts[split] = universe["split_counts"][split]
        data_sources[split] = {
            "manifest_sha256": source_hash,
            "sample_universe": key,
            "samples": counts[split],
        }
    membership = {"samples": counts["test"], "sha256": "fixture-membership"}
    metrics = {
        "sample_membership": membership,
        "samples": counts["test"],
        "evaluation_robust_scales": [1.0] * 14,
    }
    payload = {
        "state": "complete",
        "identity": identity,
        "sample_counts": counts,
        "evaluation_membership": membership,
        "validation_membership": {"samples": counts["validation"]},
        "robust_scales": [1.0] * 14,
        "models": {},
        "artifacts": {},
        "data_identity": {"split_sources": data_sources},
        "evaluation_data": CONTRACT["shared_evaluation_paths"](),
    }
    prefix = f"baselines/{identity['baseline_id']}/"

    def artifact(relative):
        content = b"fixture-saved-artifact"
        objects[prefix + relative] = content
        payload["artifacts"][relative] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    for name in identity["contract"]["parameters"]["models"]:
        learned = name in {"gbdt", "gru", "dlinear", "patchtst"}
        seeds = identity["contract"]["parameters"]["seeds"] if learned else [None]
        result = {"state": "complete"}
        result.update(
            {"seed_results": {str(seed): metrics for seed in seeds}}
            if learned
            else {"metrics": metrics}
        )
        payload["models"][name] = result
        for seed in seeds:
            job = f"jobs/{name}-{seed}" if learned else f"jobs/rules/{name}"
            weight = "model.pkl" if name == "gbdt" else "model.pt" if learned else "model.json"
            for suffix in (
                weight,
                "validation-metrics.json",
                "validation/predictions.npy",
                "test/predictions.npy",
            ):
                artifact(f"{job}/{suffix}")
    for paths in payload["evaluation_data"].values():
        for relative in paths.values():
            artifact(relative)
    key = prefix + "complete.json"
    objects[key] = encode(payload)
    return key, payload


class BaselineStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="baseline-status-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve() / "project"
        for name in ("src", "scripts", "configs"):
            shutil.copytree(
                ROOT / name, self.project / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        self.a, self.b = BASE["selection"](), BASE["selection"]("2016-01-01")
        self.objects = {**BASE["fixture"](self.a), **BASE["fixture"](self.b)}
        for selected, train_count in ((self.a, 180), (self.b, 220)):
            self.objects.update(
                FIXTURES["clean_fixture"](
                    selected,
                    self.objects,
                    counts={"train": train_count, "validation": 60, "test": 70},
                )
            )
        self.objects["lifecycle/stage1/code.json"] = encode({"release_digest": "a" * 64})
        self.key, self.payload = completed_fixture(self.project, self.a, self.objects)
        self.calls = []
        self.output = io.StringIO()

    def transport(self, command, **kwargs):
        self.calls.append(command)
        args = command[2:]
        reader = FIXTURES["CacheReader"](self.objects)
        try:
            content = reader.s3(*args)
            return subprocess.CompletedProcess(
                command, 0, content.decode() if kwargs.get("text") else content, ""
            )
        except KeyError:
            return subprocess.CompletedProcess(
                command, 1, "" if kwargs.get("text") else b"", "NoSuchKey"
            )

    def status(self, selected=None):
        reader = FIXTURES["CacheReader"](self.objects)
        reader.project = self.project
        self.output = io.StringIO()
        with (
            patch.object(subprocess, "run", side_effect=self.transport),
            patch.dict(
                os.environ,
                {
                    "RUNPOD_NETWORK_VOLUME_ID": "fixture-volume",
                    "RUNPOD_BASELINE_PREFLIGHT_WORKERS": "2",
                },
            ),
            contextlib.redirect_stdout(self.output),
        ):
            code = GATE["check_status"](self.project, selected or self.a, reader, workers=2)
        return code, self.output.getvalue(), reader

    def test_complete_requires_both_clean_inputs_and_all_saved_results(self):
        code, text, reader = self.status()
        self.assertEqual(code, 0)
        self.assertIn("Baseline: COMPLETE (12 models; 52 artifacts verified).", text)
        self.assertIn("Data rules: PASS", text)
        self.assertIn("train=180, validation=60, test=70", text)
        self.assertLessEqual(len(text.splitlines()), 6)
        self.assertNotIn('"sources":', text)
        self.assertEqual(len([call for call in self.calls if "head-object" in call]), 52)
        self.assertTrue(FIXTURES["ReadinessReuseTests"].artifact_heads(reader))

    def test_missing_completion_is_not_reported_as_successful_training(self):
        self.objects.pop(self.key)
        code, text, _ = self.status()
        self.assertEqual(code, 1)
        self.assertIn("Baseline: NOT COMPLETE", text)
        self.assertIn("Data rules: PASS", text)
        self.assertNotIn("Baseline: COMPLETE", text)

    def test_main_training_still_refuses_missing_baselines(self):
        self.objects.pop(self.key)
        cache = runpy.run_path(str(self.project / "scripts/runpod_baseline_cache.py"))
        with (
            patch.object(subprocess, "run", side_effect=self.transport),
            patch.dict(os.environ, {"RUNPOD_NETWORK_VOLUME_ID": "fixture-volume"}),
            self.assertRaisesRegex(ValueError, "Matching full-data baselines are missing"),
        ):
            cache["check_baseline"](self.project, self.a, command="require")

    def test_interrupted_cleanup_is_distinct_from_missing_training(self):
        prefix = self.key.removesuffix("complete.json")
        self.objects[prefix + "jobs/gru-42/test/membership.npy"] = b"old duplicate"
        code, text, _ = self.status()
        self.assertEqual(code, 1)
        self.assertIn("STORAGE FINALIZATION REQUIRED", text)
        self.assertIn("do not need retraining", text)
        self.assertIn("Data rules: PASS", text)

    def test_no_code_marker_still_checks_data_and_completed_results(self):
        self.objects.pop("lifecycle/stage1/code.json")
        self.assertEqual(self.status()[0], 0)

    def test_pending_cleaning_never_uses_prepared_candidate_counts(self):
        self.objects = {
            key: value
            for key, value in self.objects.items()
            if "sample-universes/" not in key and key != self.key
        }
        code, text, _ = self.status()
        self.assertEqual(code, 1)
        self.assertIn("PENDING INDEX BUILD", text)
        self.assertIn("train=pending, validation=pending, test=pending", text)
        self.assertNotIn("train=300", text)

    def test_completed_marker_cannot_hide_a_corrupt_cleaned_index(self):
        index = next(key for key in self.objects if key.endswith("universe.json"))
        payload = json.loads(self.objects[index])
        payload["audit"]["train"]["accepted"] += 1
        self.objects[index] = encode(payload)
        with self.assertRaisesRegex(ValueError, "accepted-window audit"):
            self.status()
        self.assertNotIn("Baseline: COMPLETE", self.output.getvalue())

    def test_completed_marker_cannot_hide_a_missing_weight_or_prediction(self):
        prefix = self.key.removesuffix("complete.json")
        for relative in ("jobs/gru-42/model.pt", "jobs/gru-42/test/predictions.npy"):
            with self.subTest(relative=relative):
                original = self.objects.pop(prefix + relative)
                with self.assertRaisesRegex(ValueError, "missing or truncated"):
                    self.status()
                self.assertNotIn("Baseline: COMPLETE", self.output.getvalue())
                self.objects[prefix + relative] = original

    def test_cached_input_success_does_not_hide_a_truncated_shard(self):
        self.status()
        shard = next(key for key in self.objects if key.endswith("shard.parquet"))
        self.objects[shard] = b""
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.status()

    def test_policy_changes_require_new_cleaning_and_a_matching_baseline(self):
        path = self.project / "configs/data_cleaning.json"
        policy = json.loads(path.read_text())
        policy["minimum_median_daily_turnover"]["USD"] *= 2
        path.write_text(json.dumps(policy))
        code, text, _ = self.status()
        self.assertEqual(code, 1)
        self.assertIn("NOT COMPLETE", text)
        self.assertIn("PENDING INDEX BUILD", text)
        self.assertNotIn(self.payload["identity"]["baseline_id"], text)

    def test_ab_switching_never_reuses_the_other_groups_completion(self):
        self.assertEqual(self.status(self.b)[0], 1)
        completed_fixture(self.project, self.b, self.objects)
        for selected, count in ((self.a, 180), (self.b, 220), (self.a, 180)):
            code, text, _ = self.status(selected)
            self.assertEqual(code, 0)
            self.assertIn(f"train={count}, validation=60, test=70", text)

    def test_all_six_experiments_share_only_their_matching_baseline(self):
        completed_fixture(self.project, self.b, self.objects)
        for group, selected in (("a", self.a), ("b", self.b)):
            expected = CONTRACT["baseline_contract"](self.project, selected)
            for variant in ("lora32", "lora64", "partial"):
                experiment = BASE["SELECTION"]["_with_experiment"](
                    selected, f"{group}-{variant}", self.project
                )
                self.assertEqual(CONTRACT["baseline_contract"](self.project, experiment), expected)
                self.assertEqual(self.status(experiment)[0], 0)

    def test_status_and_launch_still_share_the_data_preflight_receipt(self):
        self.status()
        reader = FIXTURES["CacheReader"](self.objects)
        reader.project = self.project
        result, reused = GATE["verify_with_receipt"](
            self.project,
            self.a,
            reader,
            workers=2,
            release_digest="a" * 64,
        )
        self.assertTrue(reused)
        self.assertEqual(result["cleaning_state"], "ready")
        self.assertFalse(FIXTURES["ReadinessReuseTests"].artifact_heads(reader))

    def test_control_docs_and_test_edits_do_not_invalidate_baseline(self):
        before = CONTRACT["baseline_contract"](self.project, self.a)
        registry = self.project / "configs/baseline_execution_compatibility.json"
        original_registry = registry.read_bytes()
        for relative in (
            "scripts/runpod_workflow.sh",
            "scripts/runpod_baseline_cache.py",
            "scripts/runpod_baseline_readiness.py",
            "README.md",
            "tests/new_test.py",
        ):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((path.read_text() if path.exists() else "") + "\n# Control-only edit\n")
        self.assertEqual(CONTRACT["baseline_contract"](self.project, self.a), before)
        self.assertEqual(self.status()[0], 0)
        self.assertEqual(registry.read_bytes(), original_registry)

    def test_failed_completed_artifact_read_is_not_reported_as_unbuilt(self):
        transport = self.transport

        def denied(command, **kwargs):
            if any(str(arg).endswith("complete.json") for arg in command):
                return subprocess.CompletedProcess(command, 1, "", "AccessDenied")
            return transport(command, **kwargs)

        with (
            patch.object(self, "transport", side_effect=denied),
            self.assertRaisesRegex(RuntimeError, "Unable to verify"),
        ):
            self.status()
        self.assertNotIn("NOT COMPLETE", self.output.getvalue())

    def test_incomplete_model_or_wrong_test_population_fails(self):
        for change in ("model", "count"):
            payload = copy.deepcopy(self.payload)
            if change == "model":
                payload["models"]["gru"]["state"] = "training"
            else:
                payload["sample_counts"]["test"] += 1
            self.objects[self.key] = encode(payload)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.status()

    def test_public_shell_command_needs_no_new_flag_or_source_upload(self):
        volume = Path(self.temp.name) / "volume"
        for key, content in self.objects.items():
            path = volume / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        BASE["SELECTION"]["_activate_selection"](self.project, self.a)
        env_file = self.project / ".env"
        env_file.write_text("RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        env_file.chmod(0o600)
        transport = self.project / "fixture_transport.py"
        transport.write_text("""import hashlib, json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_VOLUME"])
args = sys.argv[1:]
if args[:2] == ["s3", "cp"] and args[3] == "-":
    path = root / args[2].split("/", 3)[3]
    if not path.is_file():
        print("NoSuchKey", file=sys.stderr)
        raise SystemExit(1)
    sys.stdout.buffer.write(path.read_bytes())
elif args[:2] == ["s3api", "head-object"]:
    print((root / args[args.index("--key") + 1]).stat().st_size)
elif args[:2] == ["s3api", "list-objects-v2"]:
    target = root / args[args.index("--prefix") + 1]
    paths = sorted(target.rglob("*")) if target.is_dir() else [target]
    entries = [{"Key": str(p.relative_to(root)), "Size": p.stat().st_size,
                "ETag": hashlib.sha256(p.read_bytes()).hexdigest(),
                "LastModified": str(p.stat().st_mtime_ns)} for p in paths if p.is_file()]
    print(json.dumps({"Contents": entries, "IsTruncated": False}))
else:
    raise RuntimeError("Unexpected write or Pod operation")
""")
        (self.project / "scripts/runpod_s3_project.sh").write_text(
            "#!/usr/bin/env bash\nexec "
            + shlex.quote(sys.executable)
            + " "
            + shlex.quote(str(transport))
            + ' "$@"\n'
        )
        for complete in (True, False):
            if not complete:
                (volume / self.key).unlink()
            result = subprocess.run(
                [
                    "bash",
                    str(self.project / "scripts/runpod_workflow.sh"),
                    "readiness",
                    "--baseline",
                ],
                env={
                    **os.environ,
                    "FAKE_VOLUME": str(volume),
                    "RUNPOD_ENV_FILE": str(env_file),
                    "RUNPOD_BASELINE_PREFLIGHT_WORKERS": "2",
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0 if complete else 1, result.stdout + result.stderr)
            self.assertIn("Data rules: PASS", result.stdout)
            self.assertIn(
                "Baseline: COMPLETE" if complete else "Baseline: NOT COMPLETE", result.stdout
            )
        index = next(volume.glob("datasets/*/prepared/sample-universes/*/cutoff-ranges.parquet"))
        index.write_bytes(b"")
        result = subprocess.run(
            ["bash", str(self.project / "scripts/runpod_workflow.sh"), "readiness", "--baseline"],
            env={
                **os.environ,
                "FAKE_VOLUME": str(volume),
                "RUNPOD_ENV_FILE": str(env_file),
                "RUNPOD_BASELINE_PREFLIGHT_WORKERS": "2",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("Baseline: COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
