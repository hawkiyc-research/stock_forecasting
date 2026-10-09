"""Local task admission and identity checks without a Python ML environment."""

import copy
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))


class DirectionContractTests(unittest.TestCase):
    def setUp(self):
        self.parameters = json.loads((ROOT / "configs/baseline.json").read_text())

    def test_all_presets_keep_independent_baseline_identity_and_matching_costs(self):
        for entry in SELECTION["EXPERIMENT_CONFIGS"].values():
            CONTRACT["validate_direction_comparison_config"](
                ROOT,
                {"stage": entry},
                self.parameters,
            )

    def test_cost_mismatch_fails_main_admission_but_does_not_require_main_model_for_baseline(self):
        selected = {"stage": {"config_path": "configs/experiments/a_direction64.yaml"}}
        for change in ("cost", "disabled"):
            parameters = copy.deepcopy(self.parameters)
            if change == "cost":
                parameters["direction"]["costs"]["us_shares"] = 100
            else:
                parameters["direction"]["enabled"] = False
            with self.assertRaisesRegex(ValueError, "cost scenarios differ"):
                CONTRACT["validate_direction_comparison_config"](ROOT, selected, parameters)
        # Alpha-only controls are not silently removed by the new task.
        selected["stage"]["config_path"] = "configs/experiments/a_lora64.yaml"
        CONTRACT["validate_direction_comparison_config"](ROOT, selected, parameters)

    def test_actual_task_parameters_change_identity_but_model_capacity_and_guard_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for name in ("src", "configs", "scripts"):
                shutil.copytree(
                    ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            selected = {
                "dataset_request": {
                    "date_range": {
                        "start_inclusive": "2021-01-01",
                        "end_exclusive": "2026-06-01",
                    }
                }
            }
            before = CONTRACT["baseline_contract"](project, selected)
            for name in ("scripts/runpod_gpu_monitor.py", "configs/experiments/a_direction64.yaml"):
                (project / name).write_text("fixture changed unrelated code\n")
            self.assertEqual(CONTRACT["baseline_contract"](project, selected), before)
            self.parameters["direction"]["alpha_weight"] = 0.5
            (project / "configs/baseline.json").write_text(json.dumps(self.parameters))
            self.assertNotEqual(CONTRACT["baseline_contract"](project, selected), before)

    def test_quantile_only_wrong_cost_partial_horizons_and_partial_counts_are_rejected(self):
        direction = self.parameters["direction"]
        valid = {
            "samples": 8,
            "class_order": ["down", "neutral", "up"],
            "task_order": ["absolute", "alpha"],
            "cost_scenario": direction["costs"],
            "overall": {
                task: {"horizons": {str(h): {"count": 8} for h in range(1, 15)}}
                for task in ("absolute", "alpha")
            },
        }
        check = CONTRACT["validate_direction_metrics"]
        check(valid, direction, 8, horizons=range(1, 15))
        invalid = [None]
        changed = copy.deepcopy(valid)
        changed["cost_scenario"]["us_shares"] = 100
        invalid.append(changed)
        changed = copy.deepcopy(valid)
        changed["overall"]["alpha"]["horizons"].pop("14")
        invalid.append(changed)
        changed = copy.deepcopy(valid)
        changed["overall"]["absolute"]["horizons"]["1"]["count"] = 7
        invalid.append(changed)
        for metrics in invalid:
            with self.assertRaises(ValueError):
                check(metrics, direction, 8, horizons=range(1, 15))


if __name__ == "__main__":
    unittest.main()
