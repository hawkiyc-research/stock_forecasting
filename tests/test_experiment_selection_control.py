"""Dependency-free experiment selection and baseline/data isolation regression."""

import argparse
import copy
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
BASELINE = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))


class ExperimentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve()
        shutil.copytree(ROOT / "configs", self.project / "configs")
        args = argparse.Namespace(
            stage="stage2",
            data_profile="us_tw_eodhd",
            dataset_revision="v1",
            start="2016-01-01",
            end="2026-06-01",
            h_start=1,
            universe="all",
            stocks=[],
            etfs=[],
            symbol_limit=None,
            feature_mode="combined",
        )
        self.original = SELECTION["_build_selection"](args, self.project)

    def test_six_presets_have_only_two_data_and_baseline_identities(self):
        data, baseline = {}, {}
        for name in SELECTION["EXPERIMENT_CONFIGS"]:
            selected = SELECTION["_with_experiment"](self.original, name, self.project)
            group = name[0]
            data.setdefault(group, set()).add(selected["dataset_request_sha256"])
            if "baseline_task" not in selected:
                baseline.setdefault(group, set()).add(
                    BASELINE["baseline_contract"](ROOT, selected)["baseline_id"]
                )
            self.assertEqual(selected["stage"]["experiment"], name)
        self.assertEqual([len(v) for v in data.values()], [1, 1])
        self.assertEqual([len(v) for v in baseline.values()], [1, 1])
        self.assertNotEqual(data["a"], data["b"])
        self.assertEqual(data["b"], {self.original["dataset_request_sha256"]})

    def test_a_b_a_switch_restores_exact_selection(self):
        a = SELECTION["_with_experiment"](self.original, "a-lora32", self.project)
        b = SELECTION["_with_experiment"](a, "b-partial", self.project)
        restored = SELECTION["_with_experiment"](b, "a-lora32", self.project)
        self.assertEqual(a["selection_sha256"], restored["selection_sha256"])
        for selected in (a, b, restored):
            SELECTION["_activate_selection"](self.project, selected)
        pointer = json.loads((self.project / ".runpod/active-selection.json").read_text())
        self.assertEqual(pointer["selection_id"], a["selection_id"])

    def test_independent_tasks_share_prepared_data_but_not_baseline_labels(self):
        for group in ("a", "b"):
            reference = SELECTION["_with_experiment"](
                self.original, group + "-lora64", self.project
            )
            identities = {BASELINE["baseline_contract"](ROOT, reference)["baseline_id"]}
            for variant, task in (
                ("alpha-point64", {"prediction_task": "alpha_point", "point_loss_beta": 1.0}),
                (
                    "absolute-direction64",
                    {"prediction_task": "absolute_direction", "neutral_return_threshold": 0.01},
                ),
            ):
                selected = SELECTION["_with_experiment"](
                    self.original, group + "-" + variant, self.project
                )
                self.assertEqual(selected["baseline_task"], task)
                self.assertEqual(
                    selected["dataset_request_sha256"], reference["dataset_request_sha256"]
                )
                identity = BASELINE["baseline_contract"](ROOT, selected)["baseline_id"]
                identities.add(identity)
                for seed in (42, 43, 44):
                    pinned = SELECTION["_with_training_seed"](selected, seed, self.project)
                    self.assertEqual(pinned["baseline_task"], task)
                    self.assertEqual(
                        BASELINE["baseline_contract"](ROOT, pinned)["baseline_id"], identity
                    )
                restored = SELECTION["_with_experiment"](selected, group + "-lora64", self.project)
                self.assertNotIn("baseline_task", restored)
                self.assertEqual(restored["selection_sha256"], reference["selection_sha256"])
            self.assertEqual(len(identities), 3)

    def test_task_parameters_follow_yaml_and_are_part_of_immutable_selection(self):
        original = SELECTION["_with_experiment"](self.original, "a-alpha-point64", self.project)
        path = self.project / original["stage"]["config_path"]
        path.write_text(path.read_text().replace("point_loss_beta: 1.0", "point_loss_beta: 0.5"))
        updated = SELECTION["_with_experiment"](self.original, "a-alpha-point64", self.project)
        self.assertEqual(updated["baseline_task"]["point_loss_beta"], 0.5)
        self.assertEqual(updated["dataset_request_sha256"], original["dataset_request_sha256"])
        self.assertNotEqual(updated["selection_sha256"], original["selection_sha256"])
        tampered = copy.deepcopy(updated)
        tampered["baseline_task"]["point_loss_beta"] = 1.0
        with self.assertRaisesRegex(ValueError, "parameters disagree"):
            SELECTION["_validate_selection"](tampered, project_root=self.project)
        with self.assertRaises(ValueError):
            SELECTION["_validate_selection"](tampered)

    def test_task_cannot_be_missing_or_attached_to_legacy_experiment(self):
        point = SELECTION["_with_experiment"](self.original, "a-alpha-point64", self.project)
        del point["baseline_task"]
        with self.assertRaisesRegex(ValueError, "semantics disagree"):
            SELECTION["_validate_selection"](point)
        legacy = SELECTION["_with_experiment"](self.original, "a-lora64", self.project)
        legacy["baseline_task"] = {"prediction_task": "alpha_point", "point_loss_beta": 1.0}
        with self.assertRaisesRegex(ValueError, "semantics disagree"):
            SELECTION["_validate_selection"](legacy)

    def test_seed_overrides_change_only_run_identity_not_data_or_baseline(self):
        before = {path: path.read_bytes() for path in self.project.glob("configs/**/*.yaml")}
        for name in ("a-adaptive64", "b-adaptive64"):
            selected = SELECTION["_with_experiment"](self.original, name, self.project)
            baseline = BASELINE["baseline_contract"](ROOT, selected)
            identities = set()
            for seed in (0, 42, 43, 44, 2**32 - 1):
                pinned = SELECTION["_with_training_seed"](selected, seed, self.project)
                self.assertEqual(pinned["stage"]["training_seed"], seed)
                self.assertEqual(
                    pinned["dataset_request_sha256"], selected["dataset_request_sha256"]
                )
                self.assertEqual(pinned["dataset_request"], selected["dataset_request"])
                self.assertEqual(BASELINE["baseline_contract"](ROOT, pinned), baseline)
                identities.add(pinned["selection_id"])
                exports = SELECTION["_selection_exports"](Path("selection.json"), pinned)
                self.assertEqual(exports["RUNPOD_TRAINING_SEED"], str(seed))
            self.assertEqual(len(identities), 5)
            self.assertNotIn("training_seed", selected["stage"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_training_seed_validation_and_legacy_default(self):
        exports = SELECTION["_selection_exports"](Path("selection.json"), self.original)
        self.assertEqual(exports["RUNPOD_TRAINING_SEED"], "")
        for invalid in (-1, 2**32, True, None, 1.5, "42"):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                SELECTION["_with_training_seed"](self.original, invalid, self.project)

    def test_reject_wrong_dates_and_config_path(self):
        selected = SELECTION["_with_experiment"](self.original, "a-lora64", self.project)
        for key, value in (("experiment", "b-lora64"), ("config_path", "../outside.yaml")):
            broken = copy.deepcopy(selected)
            broken["stage"][key] = value
            with self.assertRaises(ValueError):
                SELECTION["_validate_selection"](broken, project_root=self.project)

    def test_log_download_and_diagnostic_fields(self):
        upload = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text()
        self.assertIn('append_manifest_file "configs/data_cleaning.json"', upload)
        download = (ROOT / "scripts/download_runpod_results.sh").read_text()
        for filename in ("metrics.jsonl", "summary.json"):
            self.assertIn(f'download_file "savedModel/${{RUN_ID}}/{filename}"', download)
        self.assertIn(
            'download_file "evaluations/${RUN_ID}/interval-calibration.json"',
            download,
        )
        source = (ROOT / "src/stock_forecasting/training.py").read_text()
        for metric in (
            "train/loss",
            "train/unfrozen_learning_rate",
            "train/processed_samples",
            "validation/loss",
            "validation/processed_train_samples",
        ):
            self.assertIn('"' + metric + '"', source)
        # Task-specific losses retain their own names instead of mislabelling point losses.
        self.assertIn('f"train/{name}"', source)
        for component in (
            "pinball_loss",
            "ranking_loss",
            "normalized_smooth_l1",
            "cross_entropy",
        ):
            self.assertIn('"' + component + '"', source)


if __name__ == "__main__":
    unittest.main()
