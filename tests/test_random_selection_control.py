"""Read-only selection tests for the new presets; never call remote tools."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RandomSelectionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
        cls.contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        arguments = cls.selection["build_parser"]().parse_args([
            "create", "--project-root", str(ROOT), "--stage", "stage2",
            "--data-profile", "us_tw_eodhd", "--start", "2021-01-01",
            "--end", "2026-06-01", "--universe", "all",
        ])
        cls.base = cls.selection["_build_selection"](arguments, ROOT)

    def test_all_new_experiments_pin_correct_config_task_and_preparation(self):
        for group in ("a", "b"):
            for task in ("alpha-point", "absolute-direction"):
                legacy = self.selection["_with_experiment"](self.base, f"{group}-{task}64", ROOT)
                for backend in ("random", "chronos2"):
                    name = f"{group}-{backend}-{task}64"
                    with self.subTest(experiment=name):
                        payload = self.selection["_with_experiment"](self.base, name, ROOT)
                        self.assertEqual(payload["stage"]["experiment"], name)
                        self.assertEqual(payload["baseline_task"], legacy["baseline_task"])
                        self.assertEqual(payload["dataset_request_sha256"],
                                         legacy["dataset_request_sha256"])
                        self.assertNotEqual(payload["stage"]["config_sha256"],
                                            legacy["stage"]["config_sha256"])
                        self.assertEqual(payload["stage"]["config_path"],
                                         "configs/experiments/" + name.replace("-", "_") + ".yaml")

    def test_baseline_numerical_contract_is_reused_across_backbones_and_sampling(self):
        for group in ("a", "b"):
            for task in ("alpha-point", "absolute-direction"):
                legacy = self.selection["_with_experiment"](self.base, f"{group}-{task}64", ROOT)
                expected = self.contract["baseline_contract"](ROOT, legacy)
                for backend in ("random", "chronos2"):
                    payload = self.selection["_with_experiment"](
                        self.base, f"{group}-{backend}-{task}64", ROOT
                    )
                    self.assertEqual(self.contract["baseline_contract"](ROOT, payload), expected)

    def test_seed_override_changes_run_selection_and_preserves_data_identity(self):
        payload = self.selection["_with_experiment"](
            self.base, "a-chronos2-alpha-point64", ROOT
        )
        seeded = self.selection["_with_training_seed"](payload, 347, ROOT)
        self.assertEqual(seeded["stage"]["training_seed"], 347)
        self.assertEqual(seeded["dataset_request_sha256"], payload["dataset_request_sha256"])
        self.assertNotEqual(seeded["selection_sha256"], payload["selection_sha256"])


if __name__ == "__main__":
    unittest.main()
