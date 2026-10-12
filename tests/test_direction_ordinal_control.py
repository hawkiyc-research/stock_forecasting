"""Stdlib-only controls for ordinal presets, public docs and unchanged data contracts."""

from __future__ import annotations

import ast
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
    @unittest.skipUnless((ROOT / ".git").exists(), "Source preservation requires the Git checkout")
    def test_presets_are_additive_and_baseline_sources_are_untouched(self):
        registry = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))["EXPERIMENT_CONFIGS"]
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
        changed = set(
            subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True
            ).splitlines()
        )
        self.assertFalse(changed.intersection(contract["BASELINE_SOURCES"]))
        self.assertFalse(
            changed.intersection(
                {
                    "configs/baseline.json",
                    "configs/data_cleaning.json",
                    "src/stock_forecasting/dataset_identity.py",
                }
            )
        )

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
