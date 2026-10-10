"""Dependency-free control-host checks; no local ML environment is needed."""

import ast
import json
import math
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_COST_MODULE = runpy.run_path(str(ROOT / "src/stock_forecasting/direction_costs.py"))
COST = _COST_MODULE["direction_cost_band"]
BANDS = _COST_MODULE["direction_band_values"]


def defaults():
    tree = ast.parse((ROOT / "src/stock_forecasting/config.py").read_text())
    definition = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DirectionCostConfig"
    )
    result = {}
    for node in definition.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            keywords = {k.arg: k.value for k in value.keywords}
            value = keywords.get("default") or keywords["default_factory"].body
        result[node.target.id] = ast.literal_eval(value)
    return SimpleNamespace(**result)


class DirectionControlTests(unittest.TestCase):
    def test_taiwan_round_trip_and_minimum_fee(self):
        config = defaults()
        regular = COST(100, "TWSE", "stock", "2330.TW", config)
        self.assertEqual(regular["shares"], 1000)
        self.assertAlmostEqual(regular["round_trip_fraction"], 0.00685)
        low = COST(10, "TPEX", "stock", "1234.TWO", config)
        self.assertAlmostEqual(low["round_trip_fraction"], 0.008)
        for kind, symbol in (
            ("etf", "0050.TW"),
            ("stock", "910322.TW"),
            ("stock", "9103.TW"),
            ("stock", "9105.TW"),
            ("stock", "9110.TW"),
            ("stock", "9136.TW"),
        ):
            value = COST(100, "TWSE", kind, symbol, config)
            self.assertAlmostEqual(value["round_trip_fraction"], 0.00485)

    def test_us_direct_account_fifty_shares_and_minimum(self):
        result = COST(20, "US", "stock", "AAPL.US", defaults())
        self.assertEqual(result["shares"], 50)
        # Average commission is (IBKR minimum $1 + Schwab $0)/2 per leg.
        self.assertAlmostEqual(result["round_trip_fraction"], (1 + 0.03 + 0.0003) / 1000 + 0.001)
        self.assertAlmostEqual(
            math.expm1(result["upper_log_return"]), result["round_trip_fraction"]
        )
        self.assertAlmostEqual(
            math.expm1(result["lower_log_return"]), -result["round_trip_fraction"]
        )

    def test_unknown_market_and_nonfinite_prices_fail_explicitly(self):
        for price in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                COST(price, "US", "stock", "X.US", defaults())
        with self.assertRaises(ValueError):
            COST(20, "OTHER", "stock", "X", defaults())

    def test_cost_above_notional_keeps_exact_upward_hurdle_and_json(self):
        for price, market, symbol in ((0.000001, "US", "LOW.US"), (0.001, "TWSE", "LOW.TW")):
            with self.subTest(market=market):
                result = COST(price, market, "stock", symbol, defaults())
                self.assertGreater(result["round_trip_fraction"], 1)
                self.assertIsNone(result["lower_log_return"])
                self.assertEqual(BANDS(result)[0], -math.inf)
                self.assertAlmostEqual(math.expm1(BANDS(result)[1]), result["round_trip_fraction"])
                self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_exactly_full_notional_cost_has_unreachable_down_boundary(self):
        config = defaults()
        for name in (
            "us_ibkr_minimum",
            "us_ibkr_per_share",
            "us_ibkr_maximum_rate",
            "us_schwab_commission",
            "us_sec_sell_rate",
            "us_taf_per_share",
            "slippage_bps_per_side",
        ):
            setattr(config, name, 0.0)
        config.us_cat_per_share = 0.01
        result = COST(0.02, "US", "stock", "LOW.US", config)
        self.assertEqual(result["round_trip_fraction"], 1.0)
        self.assertEqual(BANDS(result), (-math.inf, math.log(2)))

    def test_direction_baselines_remain_independent_of_kronos_capacity(self):
        contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        self.assertIn("src/stock_forecasting/baseline_direction.py", contract["BASELINE_SOURCES"])
        self.assertNotIn("src/stock_forecasting/models/quant.py", contract["BASELINE_SOURCES"])
        self.assertNotIn("calibrate_direction_references", contract["SHARED_TRAINING_DEFINITIONS"])
        configs = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))["EXPERIMENT_CONFIGS"]
        for group in ("a", "b"):
            for variant in (
                "lora32",
                "lora64",
                "partial",
                "adaptive64",
                "adaptive128",
                "direction64",
            ):
                self.assertIn(f"{group}-{variant}", configs)
                self.assertTrue((ROOT / configs[f"{group}-{variant}"]["config_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
