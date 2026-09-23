"""Dependency-free publication, interrupted cleanup and numerical-identity regressions."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
