"""Dependency-free publication, interrupted cleanup and numerical-identity regressions."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_forecasting import baseline_result_storage as storage  # noqa: E402
from stock_forecasting.baseline_contract import (  # noqa: E402
    shared_evaluation_paths,
    validate_complete,
)


def metadata(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fixture(root):
    root.mkdir(parents=True)
    membership = {"samples": 2, "sha256": "fixture"}
    metrics = {"samples": 2, "sample_membership": membership, "evaluation_robust_scales": [1.0]}
    payload = {
        "state": "complete",
        "identity": {"baseline_id": root.name, "contract": {"parameters": {
            "models": ["zero_return", "gru"], "seeds": [42],
        }}},
        "sample_counts": {"train": 4, "validation": 2, "test": 2},
        "evaluation_membership": membership, "validation_membership": membership,
        "robust_scales": [1.0], "artifacts": {},
        "models": {
            "zero_return": {"state": "complete", "metrics": metrics},
            "gru": {"state": "complete", "seed_results": {"42": metrics}},
        },
    }

    def save(relative, content, artifact=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if artifact:
            payload["artifacts"][relative] = metadata(content)

    for split, paths in shared_evaluation_paths().items():
        for kind, path in paths.items():
            save(path, f"{split}:{kind}:canonical".encode(), False)
    save("inputs/train/targets.npy", b"protected training labels", False)
    for job, weight in (("jobs/rules/zero_return", "model.json"), ("jobs/gru-42", "model.pt")):
        save(f"{job}/{weight}", b"protected weight")
        save(f"{job}/validation-metrics.json", b"protected metrics")
        for split in ("validation", "test"):
            save(f"{job}/{split}/predictions.npy", f"{job}:{split}:predictions".encode())
            for kind in ("membership", "targets"):
                save(f"{job}/{split}/{kind}.npy", f"{split}:{kind}:canonical".encode())
    (root / "complete.json").write_text(json.dumps(payload))
    return payload


class BaselineStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "baselines/baseline-fixture"
        self.payload = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def finalize(self, **kwargs):
        with (
            patch.object(storage._ReadBudget, "reserve"),
            patch.object(storage, "_workers", return_value=2),
        ):
            return storage.finalize_baseline_storage(self.root, workers=2, progress=None, **kwargs)

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_dry_run_does_not_change_any_file(self):
        before = self.snapshot()
        result = self.finalize()
        self.assertEqual(result["duplicate_files"], 8)
        self.assertEqual(self.snapshot(), before)

    def test_only_duplicate_arrays_are_removed_and_identity_never_changes(self):
        before = self.snapshot()
        result = self.finalize(apply=True)
        updated = result["payload"]
        self.assertEqual(result["deleted_files"], 8)
        self.assertEqual(updated["identity"], self.payload["identity"])
        self.assertEqual(updated["models"], self.payload["models"])
        validate_complete(updated, self.payload["identity"], require_shared=True)
        for relative, content in before.items():
            if storage.DUPLICATE_PATH.fullmatch(relative):
                self.assertFalse((self.root / relative).exists())
            elif relative != "complete.json":
                self.assertEqual((self.root / relative).read_bytes(), content)
        self.assertEqual(updated["evaluation_data"], shared_evaluation_paths())
        self.assertEqual(len(updated["artifacts"]), len(self.payload["artifacts"]) - 8 + 4)
        second = self.finalize(apply=True)
        self.assertEqual(second["deleted_files"], 0)
        self.assertEqual(second["payload"], updated)

    def test_corrupt_duplicate_aborts_before_publication_or_deletion(self):
        path = self.root / "jobs/gru-42/test/targets.npy"
        path.write_bytes(b"x" * path.stat().st_size)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.finalize(apply=True)
        self.assertEqual(self.snapshot(), before)

    def test_corrupt_shared_input_is_not_silently_rebuilt(self):
        path = self.root / "inputs/test/metadata.npy"
        path.write_bytes(b"x" * path.stat().st_size)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.finalize(apply=True)
        self.assertEqual(self.snapshot(), before)

    def test_disagreeing_recorded_populations_are_rejected(self):
        self.payload["artifacts"]["jobs/gru-42/test/targets.npy"]["sha256"] = "0" * 64
        (self.root / "complete.json").write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.finalize(apply=True)

    def test_interrupted_cleanup_can_resume_without_retraining_or_references_to_deleted_files(self):
        unlink = Path.unlink
        deleted = []

        def interrupted(path, *args, **kwargs):
            if path.name in ("targets.npy", "membership.npy"):
                if deleted:
                    raise OSError("simulated interruption")
                deleted.append(str(path))
            return unlink(path, *args, **kwargs)

        with (
            patch.object(Path, "unlink", interrupted),
            self.assertRaisesRegex(OSError, "simulated interruption"),
        ):
            self.finalize(apply=True)
        published = json.loads((self.root / "complete.json").read_text())
        validate_complete(published, self.payload["identity"], require_shared=True)
        self.assertTrue(all((self.root / path).is_file() for path in published["artifacts"]))
        result = self.finalize(apply=True)
        self.assertEqual(result["deleted_files"], 7)
        self.assertEqual(result["payload"], published)

    def test_crash_after_last_unlink_does_not_require_a_receipt_to_reuse_results(self):
        publish = storage._atomic_json

        def interrupted(path, payload):
            if path.name == "storage-finalization.json" and payload["state"] == "complete":
                raise OSError("simulated final receipt interruption")
            publish(path, payload)

        with patch.object(storage, "_atomic_json", interrupted), self.assertRaises(OSError):
            self.finalize(apply=True)
        payload = json.loads((self.root / "complete.json").read_text())
        validate_complete(payload, self.payload["identity"], require_shared=True)
        self.assertEqual(self.finalize(apply=True)["deleted_files"], 0)
        receipt = self.root / "storage-finalization.json"
        self.assertEqual(json.loads(receipt.read_text())["state"], "complete")
        before = receipt.read_bytes()
        self.finalize(apply=True)
        self.assertEqual(receipt.read_bytes(), before)

    def test_symlinked_data_is_never_deleted(self):
        path = self.root / "jobs/gru-42/test/targets.npy"
        path.unlink()
        path.symlink_to(self.root / "inputs/test/targets.npy")
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            self.finalize(apply=True)
        self.assertTrue(path.is_symlink())

    def test_incomplete_results_are_never_cleaned(self):
        self.payload["state"] = "running"
        (self.root / "complete.json").write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError, "No complete baseline"):
            self.finalize(apply=True)

    def test_published_reader_rejects_staging_output_and_wrong_shared_paths(self):
        with self.assertRaisesRegex(ValueError, "not finalized"):
            validate_complete(self.payload, self.payload["identity"], require_shared=True)
        updated = self.finalize(apply=True)["payload"]
        updated["evaluation_data"]["test"]["targets"] = "inputs/validation/targets.npy"
        with self.assertRaisesRegex(ValueError, "shared evaluation paths"):
            validate_complete(updated, self.payload["identity"], require_shared=True)


class StorageIdentityTests(unittest.TestCase):
    def test_control_plane_storage_edits_do_not_change_numerical_resume_identity(self):
        tree = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        selected = []
        names = {
            "_validated_implementation_files", "_canonical_payload_digest",
            "_checkpoint_retention_migration_matches",
        }
        for node in tree.body:
            if getattr(node, "name", None) in names or (
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "TRAINING_IMPLEMENTATION_PATHS"
                    for target in node.targets
                )
            ):
                selected.append(node)
        scope = {"hashlib": hashlib, "json": json, "Any": object,
                 "CHECKPOINT_RETENTION_MIGRATIONS": ()}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "resume", "exec"), scope)
        digest = scope["_canonical_payload_digest"]
        files = {name: "1" * 64 for name in scope["TRAINING_IMPLEMENTATION_PATHS"]}
        self.assertNotIn("baseline_contract.py", files)
        self.assertNotIn("run_contract.py", files)
        current = {"data": "same", "model": "same", "training": "same",
                   "training_implementation": {"files": files, "sha256": digest(files)}}
        stored = copy.deepcopy(current)
        old_files = stored["training_implementation"]["files"]
        old_files.update({"baseline_contract.py": "2" * 64, "run_contract.py": "3" * 64})
        stored["training_implementation"]["sha256"] = digest(old_files)
        matches = scope["_checkpoint_retention_migration_matches"]
        self.assertTrue(matches(stored, current))
        self.assertFalse(matches(stored, {**current, "data": "different"}))
        self.assertFalse(matches(stored, {**current, "training": "different"}))
        changed = copy.deepcopy(current)
        changed["training_implementation"]["files"]["models/quant.py"] = "9" * 64
        changed["training_implementation"]["sha256"] = digest(
            changed["training_implementation"]["files"]
        )
        self.assertFalse(matches(stored, changed))
        stored["training_implementation"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            matches(stored, current)

    def test_storage_modules_do_not_enter_baseline_numerical_identity(self):
        contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        for path in (
            "src/stock_forecasting/baseline_result_storage.py",
            "src/stock_forecasting/cli/build_baselines.py",
        ):
            self.assertNotIn(path, contract["BASELINE_SOURCES"])

    def test_downloaded_ab_baselines_keep_their_existing_numerical_identity(self):
        contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        directory = ROOT / ".runpod/diagnostics/storage-20260923"
        for group in ("A", "B"):
            path = directory / f"baseline-{group}-complete.json"
            if not path.is_file():
                self.skipTest("Downloaded baseline records are optional verification artifacts")
            with self.subTest(group=group):
                saved = json.loads(path.read_text())
                selected = {"dataset_request": saved["identity"]["contract"]["data"]}
                current = contract["baseline_contract"](ROOT, selected)
                self.assertEqual(current, saved["identity"])
                contract["validate_complete"](saved, current)


class BaselineStorageEntryTests(unittest.TestCase):
    """Run the real local cache CLI and finalizer over a filesystem-backed S3 fixture."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "baselines/baseline-fixture"
        self.payload = fixture(self.root)
        parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        parameters.update(self.payload["identity"]["contract"]["parameters"])
        self.payload["identity"]["contract"]["parameters"] = parameters
        self.manifest = json.dumps({"split_counts": self.payload["sample_counts"]}).encode()
        self.payload["data_identity"] = {
            "manifest_sha256": hashlib.sha256(self.manifest).hexdigest(),
        }
        (self.root / "complete.json").write_text(json.dumps(self.payload))
        self.cache = runpy.run_path(str(ROOT / "scripts/runpod_baseline_cache.py"))
        self.contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        self.calls = []
        self.page_size = 3
        self.fail_listing = False

    def tearDown(self):
        self.temporary.cleanup()

    def transport(self, command, **kwargs):
        self.calls.append(command)
        if "list-objects-v2" in command:
            if self.fail_listing:
                return SimpleNamespace(returncode=1, stdout="", stderr="AccessDenied")
            prefix = command[command.index("--prefix") + 1]
            keys = sorted("baselines/baseline-fixture/" + str(path.relative_to(self.root))
                          for path in self.root.rglob("*") if path.is_file()
                          and str(path.relative_to(self.root)).startswith("jobs/"))
            start = (int(command[command.index("--continuation-token") + 1])
                     if "--continuation-token" in command else 0)
            keys = [key for key in keys if key.startswith(prefix)]
            stop = start + self.page_size
            page = {"Contents": [{"Key": key} for key in keys[start:stop]],
                    "IsTruncated": stop < len(keys)}
            if page["IsTruncated"]:
                page["NextContinuationToken"] = str(stop)
            return SimpleNamespace(returncode=0, stdout=json.dumps(page))
        if "head-object" in command:
            key = command[command.index("--key") + 1]
            relative = key.removeprefix("baselines/baseline-fixture/")
            path = self.root / relative
            return SimpleNamespace(returncode=0 if path.is_file() else 1,
                                   stdout=str(path.stat().st_size) if path.is_file() else "")
        if "cp" in command:
            key = command[command.index("cp") + 1]
            content = ((self.root / "complete.json").read_bytes()
                       if key.endswith("/complete.json") else self.manifest)
            return SimpleNamespace(returncode=0, stdout=content.decode() if kwargs.get("text")
                                   else content)
        raise AssertionError(f"Unexpected write, training or Pod creation: {command}")

    def gate(self, command="check"):
        real_run_path = runpy.run_path
        selected = {"dataset_request_sha256": "fixture-data", "dataset_request": {
            "preparation": {"h_start": 1, "window_size": 128}}}

        def modules(path):
            if path.endswith("runpod_selection.py"):
                return {"_resolve_selection_path": lambda *args, **kwargs: (None, selected)}
            if path.endswith("baseline_contract.py"):
                return {
                    **self.contract, "baseline_contract": lambda *args: self.payload["identity"],
                }
            return real_run_path(path)

        output = io.StringIO()
        with (
            patch.object(runpy, "run_path", side_effect=modules),
            patch.object(subprocess, "run", side_effect=self.transport),
            patch.dict(os.environ, {"RUNPOD_NETWORK_VOLUME_ID": "fixture-volume",
                                    "RUNPOD_TEST_MODE": "1"}),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(self.cache["main"]([command, "--project-root", str(ROOT)]), 0)
        return json.loads(output.getvalue())

    def finalize(self):
        with (
            patch.object(storage._ReadBudget, "reserve"),
            patch.object(storage, "_workers", return_value=2),
        ):
            return storage.finalize_baseline_storage(self.root, apply=True, progress=None)

    def test_unpublished_results_request_finalization_not_a_new_baseline_identity(self):
        result = self.gate()
        self.assertFalse(result["complete"])
        self.assertTrue(result["storage_finalization_pending"])
        self.assertEqual(result["baseline_id"], self.payload["identity"]["baseline_id"])
        self.finalize()
        self.assertTrue(self.gate()["complete"])

    def test_normal_cli_detects_interrupted_cleanup_but_training_can_reuse_shared_data(self):
        unlink = Path.unlink
        count = 0

        def interrupted(path, *args, **kwargs):
            nonlocal count
            if path.name in ("membership.npy", "targets.npy"):
                count += 1
                if count == 2:
                    raise OSError("interrupted removal")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", interrupted), self.assertRaises(OSError):
            self.finalize()
        self.assertTrue(self.gate("require")["complete"])
        self.assertFalse(any("list-objects-v2" in call for call in self.calls))
        pending = self.gate()
        self.assertFalse(pending["complete"])
        self.assertTrue(pending["storage_finalization_pending"])
        self.assertEqual(self.finalize()["deleted_files"], 7)
        self.assertTrue(self.gate()["complete"])
        self.assertTrue(any("--continuation-token" in call for call in self.calls))
        (self.root / "storage-finalization.json").unlink()
        self.assertTrue(self.gate()["complete"])
        self.assertTrue(self.gate("require")["complete"])

    def test_stale_receipt_does_not_force_a_paid_pod(self):
        self.finalize()
        (self.root / "storage-finalization.json").write_text('{"state": "verified"}')
        self.assertTrue(self.gate()["complete"])

    def test_listing_errors_fail_before_any_paid_pod_but_do_not_block_training(self):
        self.finalize()
        self.fail_listing = True
        with self.assertRaisesRegex(RuntimeError, "no Pod will be created"):
            self.gate()
        self.assertTrue(self.gate("require")["complete"])

    def test_invalid_pagination_cannot_silently_skip_remaining_copies(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout='{"Contents": [], "IsTruncated": true}',
        )), self.assertRaisesRegex(ValueError, "Incomplete"):
            self.cache["_has_duplicate_arrays"]("wrapper", "bucket", "baselines/fixture/", [])


if __name__ == "__main__":
    unittest.main()
