"""Dependency-free control regressions; no local ML environment is required."""

import argparse
import ast
import copy
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SamplePlateauScheduler = runpy.run_path(
    str(ROOT / "src/stock_forecasting/forecast_optimization.py")
)["SamplePlateauScheduler"]

SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
BASELINE = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))


class IntegratedAdaptationControlTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        return SamplePlateauScheduler(
            SimpleNamespace(param_groups=[{"lr": 1e-5}, {"lr": 5e-6}]), 2,
            warmup_samples=32, decay_start=64, decay_end=128, **kwargs,
        )

    def test_sample_clock_matches_across_batch_sizes(self):
        first, second = self.scheduler(), self.scheduler()
        for samples in (16, 16, 16, 16, 16, 16):
            first.step(samples)
        for samples in (32, 32, 32):
            second.step(samples)
        self.assertEqual(first.processed_samples, 96)
        self.assertEqual(first.get_last_lr(), second.get_last_lr())
        self.assertAlmostEqual(first.get_last_lr()[0], 1e-5 * (1 + 0.09) / 2)

    def test_resume_realign_preserves_sample_clock_and_plateau(self):
        first = self.scheduler()
        for _ in range(3):
            first.step(32)
            first.observe(0.4)
        resumed = self.scheduler()
        resumed.load_state_dict(copy.deepcopy(first.state_dict()))
        resumed.realign(6, 4)
        self.assertEqual(resumed.processed_samples, 96)
        self.assertEqual(resumed.get_last_lr(), first.get_last_lr())
        for scheduler in (first, resumed):
            scheduler.step(32)
            scheduler.observe(0.5)
            self.assertFalse(scheduler.permits_early_stop)
            scheduler.observe(0.5)
            self.assertEqual(scheduler.low_evaluations, 0)
            for _ in range(2):
                scheduler.step(16)
                scheduler.observe(0.5)
            self.assertTrue(scheduler.permits_early_stop)
        self.assertEqual(resumed.get_last_lr(), first.get_last_lr())

    def test_invalid_samples_and_validation_are_rejected(self):
        scheduler = self.scheduler()
        for value in (None, 0, -1, True, 3.5):
            with self.assertRaises(ValueError):
                scheduler.step(value)
        with self.assertRaises(ValueError):
            scheduler.observe(float("nan"))

    def test_new_presets_keep_old_presets_and_baseline_identity(self):
        names = SELECTION["EXPERIMENT_CONFIGS"]
        expected = {f"{g}-{v}" for g in ("a", "b")
                    for v in ("lora32", "lora64", "partial", "adaptive64", "adaptive128")}
        self.assertEqual(set(names), expected)
        args = argparse.Namespace(stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
                                  start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
                                  stocks=[], etfs=[], symbol_limit=None, feature_mode="combined")
        original = SELECTION["_build_selection"](args, ROOT)
        for group in ("a", "b"):
            identities, datasets = set(), set()
            for variant in ("lora32", "lora64", "partial", "adaptive64", "adaptive128"):
                selected = SELECTION["_with_experiment"](original, f"{group}-{variant}", ROOT)
                identities.add(BASELINE["baseline_contract"](ROOT, selected)["baseline_id"])
                datasets.add(selected["dataset_request_sha256"])
            self.assertEqual(len(identities), 1)
            self.assertEqual(len(datasets), 1)

    def test_main_model_policy_does_not_enter_baseline_or_prepare_identity(self):
        self.assertFalse(any("forecast_optimization" in p or "interval_calibration" in p
                             for p in BASELINE["BASELINE_SOURCES"]))
        tree = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        paths = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == "TRAINING_IMPLEMENTATION_PATHS"
                         for t in node.targets))
        self.assertIn("forecast_optimization.py", paths)
        self.assertIn("interval_calibration.py", paths)


if __name__ == "__main__":
    unittest.main()
