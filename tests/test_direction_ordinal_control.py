"""Stdlib-only controls for ordinal presets, public docs and unchanged data contracts."""

from __future__ import annotations

import argparse
import ast
import json
import runpy
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "selection_class_support_min",
    "selection_prediction_share_min",
    "selection_collapse_weight",
    "selection_opposite_weight",
)


class OrdinalControlTests(unittest.TestCase):
    def test_presets_are_additive_without_changing_baseline_task_or_data(self):
        selection = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
        registry = selection["EXPERIMENT_CONFIGS"]
        for group in ("a", "b"):
            for variant in (
                "lora32",
                "lora64",
                "partial",
                "adaptive64",
                "adaptive128",
                "direction64",
                "direction64-v2",
                "direction64-v3",
            ):
                self.assertIn(f"{group}-{variant}", registry)
                self.assertTrue((ROOT / registry[f"{group}-{variant}"]["config_path"]).is_file())
        contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        original = selection["_build_selection"](
            argparse.Namespace(
                stage="stage2",
                data_profile="us_tw_eodhd",
                dataset_revision="v1",
                start="2021-01-01",
                end="2026-06-01",
                h_start=1,
                universe="all",
                stocks=[],
                etfs=[],
                symbol_limit=None,
                feature_mode="combined",
            ),
            ROOT,
        )
        expected_parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        expected_parameters.pop("resources", None)
        # An unrelated new baseline task may edit shared files. Test the enduring
        # isolation contract, not which files happen to be dirty before commit.
        for group in ("a", "b"):
            previous = selection["_with_experiment"](original, f"{group}-direction64-v2", ROOT)
            ordinal = selection["_with_experiment"](original, f"{group}-direction64-v3", ROOT)
            self.assertEqual(previous["dataset_request"], ordinal["dataset_request"])
            self.assertEqual(previous["dataset_request_sha256"], ordinal["dataset_request_sha256"])
            self.assertNotIn("baseline_task", ordinal)
            baseline = contract["baseline_contract"](ROOT, previous)
            self.assertEqual(baseline, contract["baseline_contract"](ROOT, ordinal))
            self.assertEqual(baseline["contract"]["parameters"], expected_parameters)
            for seed in (42, 43, 44):
                seeded = selection["_with_training_seed"](ordinal, seed, ROOT)
                self.assertEqual(baseline, contract["baseline_contract"](ROOT, seeded))

    @unittest.skipUnless((ROOT / ".git").exists(), "Source preservation requires the Git checkout")
    def test_shared_direction_helpers_are_identical_to_retained_version(self):
        relative = "src/stock_forecasting/models/direction.py"
        current = (ROOT / relative).read_text()
        previous = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT, text=True)
        for name in ("direction_labels", "direction_cost_features", "classification_loss"):

            def extract(source, name=name):
                node = next(
                    n
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                return ast.get_source_segment(source, node)

            self.assertEqual(extract(current), extract(previous))

    @unittest.skipUnless((ROOT / ".git").exists(), "Source preservation requires the Git checkout")
    def test_shared_baseline_sampling_and_training_definitions_are_unchanged(self):
        contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        relative = "src/stock_forecasting/training.py"
        previous = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT, text=True)
        current = (ROOT / relative).read_text()
        names = contract["SHARED_TRAINING_DEFINITIONS"]

        def extract(source):
            nodes = [n for n in ast.parse(source).body if getattr(n, "name", None) in names]
            self.assertEqual(len(nodes), len(names))
            return [ast.dump(node, include_attributes=False) for node in nodes]

        self.assertEqual(extract(current), extract(previous))

    def test_bilingual_docs_describe_every_new_parameter_and_no_accuracy_guarantee(self):
        for relative in ("README.md", "docs/experiment_protocol.md"):
            source = (ROOT / relative).read_text()
            self.assertEqual(source.count("\n## English\n"), 1)
            for part in source.split("\n## English\n"):
                for marker in (
                    "direction64-v3",
                    "market_ordinal_v3",
                    "CORN",
                    "MASTER",
                    "Brier",
                    "population_macro_F1",
                    "https://arxiv.org/abs/2111.08851",
                ):
                    self.assertIn(marker, part)
                if relative == "README.md":
                    for field in FIELDS:
                        self.assertIn(f"| `{field}` |", part)

    def test_no_local_ml_environment_required_for_control(self):
        tree = ast.parse(Path(__file__).read_text())
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("torch", imports)
        self.assertNotIn("stock_forecasting.config", imports)


if __name__ == "__main__":
    unittest.main()
