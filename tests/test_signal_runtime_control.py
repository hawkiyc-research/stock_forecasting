"""Dependency-free task/output and final-holdout reporting contracts."""

from __future__ import annotations

import ast
import copy
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative, names, scope=None):
    source = ROOT / "src/stock_forecasting" / relative
    tree = ast.parse(source.read_text())
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    output = dict(scope or {})
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), output)
    return output


class SignalRuntimeControlTests(unittest.TestCase):
    def setUp(self):
        helper = runpy.run_path(str(ROOT / "tests/test_evaluation_storage_control.py"))
        self.scope = helper["production_scope"]()

    def test_schema_pair_is_strict_and_legacy_is_preserved(self):
        scope = definitions(
            "checkpointing.py", {"_valid_output_schema"}, {"MODEL_OUTPUT_SCHEMA_VERSION": "5.0"}
        )
        valid = scope["_valid_output_schema"]
        self.assertTrue(valid({"model_output_schema_version": "5.0"}))
        for task in ("alpha_point", "absolute_direction"):
            self.assertTrue(
                valid({"model_output_schema_version": "signal-1.0", "prediction_task": task})
            )
            self.assertFalse(valid({"model_output_schema_version": "5.0", "prediction_task": task}))
        self.assertFalse(valid({"model_output_schema_version": "signal-1.0"}))
        self.assertFalse(
            valid({"model_output_schema_version": "signal-1.0", "prediction_task": "other"})
        )

    def test_signal_comparison_does_not_publish_quantile_or_trading_metrics(self):
        metrics = {
            "aggregate": {"selection_score": 0.3, "normalized_smooth_l1": 0.3},
            "primary_5d": {"mae": 0.02, "rmse": 0.03, "correlation": 0.1},
            "cross_sectional": {"by_market": {"US": {"5d": {"rank_ic_mean": 0.12}}}},
        }
        payload = {"kronos_full": {"state": "complete", "metrics": metrics}}
        row = self.scope["_comparison_summary"](payload, "alpha_point")["rows"][0]
        self.assertEqual(row["mae"], 0.02)
        self.assertEqual(row["rank_ic_by_market_5d"], {"US": 0.12})
        self.assertNotIn("normalized_pinball", row)
        self.assertNotIn("interval_coverage", row)
        self.assertNotIn("net_long_short_sharpe", row)
        legacy = self.scope["_comparison_summary"](payload)["rows"][0]
        self.assertIn("normalized_pinball", legacy)

    def test_constructed_protocol_matches_each_real_task_and_preserves_legacy(self):
        helper = runpy.run_path(str(ROOT / "tests/test_evaluation_storage_control.py"))
        case = helper["EvaluationStorageControlTests"]()
        case.setUp()
        try:
            for task in ("alpha_point", "absolute_direction", "quantiles"):
                with self.subTest(task=task):
                    case.config.model.prediction_task = task
                    payload = case.runner().payload
                    protocol = payload["protocol"]
                    if task == "quantiles":
                        self.assertNotIn("prediction_task", payload)
                        self.assertIn("normalized pinball", protocol["model_selection"])
                        self.assertIn("distribution post-processing", protocol["classification"])
                        continue
                    self.assertEqual(payload["prediction_task"], task)
                    self.assertNotIn("pinball", protocol["model_selection"])
                    self.assertNotIn("distribution post-processing", protocol["classification"])
                    self.assertIn("Validation-only", protocol["model_selection"])
                    if task == "alpha_point":
                        self.assertIn(
                            "equal-horizon normalized SmoothL1", protocol["model_selection"]
                        )
                        self.assertIn("No classifier", protocol["classification"])
                    else:
                        self.assertIn(
                            "equal-horizon absolute-direction cross entropy",
                            protocol["model_selection"],
                        )
                        self.assertIn("Independent down/neutral/up", protocol["classification"])
                        self.assertIn(
                            "no transaction-cost labels or inputs", protocol["classification"]
                        )
                    for definition in payload["model_definitions"].values():
                        self.assertIn(task, definition)
        finally:
            case.tearDown()

    def _runner(self, task):
        runner = self.scope["ValidationBenchmark"].__new__(self.scope["ValidationBenchmark"])
        runner.config = SimpleNamespace(model=SimpleNamespace(prediction_task=task))
        membership = {"samples": 3, "ordered_symbol_dates_sha256": "fixture"}
        key = "point_loss_beta" if task == "alpha_point" else "neutral_return_threshold"
        metrics = {
            "prediction_task": task,
            key: 1.0 if task == "alpha_point" else 0.01,
            "sample_membership": membership,
            "evaluation_robust_scales": [0.1] * 14,
            "daily_signal_loss": {"2026-01-02": 0.3},
        }
        runner.payload = {
            "evaluation_membership": membership,
            "comparison": {},
            "models": {
                "kronos_full": {"metrics": copy.deepcopy(metrics)},
                "gru": {"seed_results": {"42": copy.deepcopy(metrics)}},
            },
        }
        return runner

    def test_prebuilt_fraction_splits_report_actual_holdout_not_validation(self):
        helper = runpy.run_path(str(ROOT / "tests/test_evaluation_storage_control.py"))
        case = helper["EvaluationStorageControlTests"]()
        case.setUp()
        try:
            for task in ("quantiles", "alpha_point", "absolute_direction"):
                for fixed in (False, True):
                    for prebuilt in (False, True):
                        with self.subTest(task=task, fixed=fixed, prebuilt=prebuilt):
                            case.config.model.prediction_task = task
                            case.config.data.fixed_split = fixed
                            case.config.validation.require_prebuilt_baselines = prebuilt
                            expected = (
                                "test" if fixed or prebuilt or task != "quantiles" else "validation"
                            )
                            runner = case.runner()
                            self.assertEqual(runner.evaluation_split, expected)
                            self.assertEqual(runner.payload["evaluation_split"], expected)
                            protocol = runner.evaluation_contract["inputs"]["evaluation_protocol"]
                            self.assertEqual(protocol["split"], expected)
                            self.assertEqual(protocol["fixed_split"], fixed)
                            policy = runner.payload["protocol"]["test_policy"]
                            if expected == "test":
                                self.assertIn("Holdout is scored only after validation", policy)
                            else:
                                self.assertIn("validation only, not a holdout result", policy)
        finally:
            case.tearDown()

    def test_pairing_compares_real_task_losses_and_rejects_different_tasks_or_thresholds(self):
        for task in ("alpha_point", "absolute_direction"):
            runner = self._runner(task)
            runner._verify_paired_results([0.1] * 14)
            self.scope["paired_block_comparison"].assert_called_with(
                {"2026-01-02": 0.3}, {"2026-01-02": 0.3}
            )
            runner.payload["models"]["gru"]["seed_results"]["42"]["prediction_task"] = "quantiles"
            with self.assertRaisesRegex(ValueError, "different prediction task"):
                runner._verify_paired_results([0.1] * 14)
        runner = self._runner("absolute_direction")
        runner.payload["models"]["gru"]["seed_results"]["42"]["neutral_return_threshold"] = 0.025
        with self.assertRaisesRegex(ValueError, "different neutral_return_threshold"):
            runner._verify_paired_results([0.1] * 14)

    def test_signal_source_identity_is_additive_and_task_specific(self):
        source = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        constants = {}
        for node in source.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("IMPLEMENTATION_PATHS"):
                        constants[target.id] = ast.literal_eval(node.value)
        self.assertIn("models/signal.py", constants["SIGNAL_IMPLEMENTATION_PATHS"])
        self.assertIn("signal_evaluation.py", constants["SIGNAL_IMPLEMENTATION_PATHS"])
        self.assertIn("signal_training.py", constants["SIGNAL_IMPLEMENTATION_PATHS"])
        self.assertNotIn("models/signal.py", constants["TRAINING_IMPLEMENTATION_PATHS"])
        for path in constants["SIGNAL_IMPLEMENTATION_PATHS"]:
            self.assertIn(path, self.scope["SIGNAL_EVALUATION_IMPLEMENTATION_PATHS"])
            self.assertNotIn(path, self.scope["EVALUATION_NUMERICAL_IMPLEMENTATION_PATHS"])
        schema = definitions(
            "run_contract.py",
            {"model_output_schema_version"},
            {"MODEL_OUTPUT_SCHEMA_VERSION": "5.0"},
        )["model_output_schema_version"]
        self.assertEqual(
            schema(SimpleNamespace(model=SimpleNamespace(prediction_task="quantiles"))), "5.0"
        )
        self.assertEqual(
            schema(SimpleNamespace(model=SimpleNamespace(prediction_task="alpha_point"))),
            "signal-1.0",
        )


if __name__ == "__main__":
    unittest.main()
