"""Dependency-free checks of the pre-registered Phase 1 information-test settings."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/research/tw_information_test.json").read_text())
BLOCKS = {
    "technical", "technical_short", "long_lookback", "flows", "market", "us_evening",
    "us_pre_open",
}


class InformationTestConfigTests(unittest.TestCase):
    def test_pre_registered_horizons_criteria_and_costs(self) -> None:
        self.assertEqual(CONFIG["horizons"], [1, 2, 3, 5])
        self.assertEqual(CONFIG["comparison_horizons"], [10, 14])
        self.assertEqual(CONFIG["criteria"], {
            "decision_horizons": [1, 2, 3],
            "min_passing_horizons": 2,
            "min_relative_improvement": 0.01,
        })
        # Undiscounted round trip: 0.1425% commission each side plus transaction tax.
        self.assertAlmostEqual(CONFIG["cost_floor"]["stock"], 2 * 0.001425 + 0.003)
        self.assertAlmostEqual(CONFIG["cost_floor"]["etf"], 2 * 0.001425 + 0.001)
        self.assertAlmostEqual(CONFIG["neutral_share_target"], 1 / 3)
        self.assertEqual(CONFIG["markets"], ["TWSE", "TPEX"])

    def test_feature_sets_reference_known_blocks_and_keep_the_baseline(self) -> None:
        sets = CONFIG["feature_sets"]
        self.assertEqual(sets["F0"], ["technical"])
        for name, blocks in sets.items():
            with self.subTest(name=name):
                self.assertTrue(set(blocks) <= BLOCKS)
        self.assertTrue(set(CONFIG["fixed_band_sets"]) <= set(sets))

    def test_bootstrap_and_model_keys_match_their_functions(self) -> None:
        tree = ast.parse((ROOT / "src/stock_forecasting/research/tw_evaluation.py").read_text())
        bootstrap = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "block_bootstrap"
        )
        keyword_only = {argument.arg for argument in bootstrap.args.kwonlyargs}
        self.assertEqual(set(CONFIG["bootstrap"]), keyword_only)
        self.assertLess(CONFIG["model"]["holdout_fraction"], 0.5)
        self.assertGreaterEqual(
            CONFIG["model"]["holdout_embargo_sessions"], max(CONFIG["comparison_horizons"])
        )

    def test_models_only_fit_train_rows(self) -> None:
        source = (ROOT / "src/stock_forecasting/research/tw_information.py").read_text()
        self.assertEqual(source.count("usable = train & "), 2)
        self.assertEqual(source.count("fit_classifier(\n"), 1)
        self.assertEqual(source.count("fit_regressor(\n"), 1)


if __name__ == "__main__":
    unittest.main()
