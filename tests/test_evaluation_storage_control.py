"""Run the testing constructor/resume path without a local ML environment."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import runpy
import statistics
import sys
import tempfile
import time
import unittest
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/stock_forecasting/validation_benchmark.py"
sys.path.insert(0, str(ROOT / "src"))


def production_scope():
    """Compile unchanged production definitions, excluding only ML imports.

    Device planning and numerical inference are explicit test doubles. Path
    checks, contract construction/comparison, constructor, JSON publication,
    completion decisions, and the run/prebuilt routing execute real code.
    """
    scope = runpy.run_path(str(ROOT / "src/stock_forecasting/run_paths.py"))
    scope.update(
        runpy.run_path(str(ROOT / "src/stock_forecasting/evaluation_resume_migrations.py"))
    )
    runtime_stop = runpy.run_path(str(ROOT / "src/stock_forecasting/runpod/runtime_stop.py"))
    scope["RuntimeStopRequested"] = runtime_stop["RuntimeStopRequested"]
    scope["stop_at_saved_boundary"] = Mock()
    baselines = ast.parse((ROOT / "src/stock_forecasting/baselines.py").read_text())
    rules = next(
        node.value
        for node in baselines.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RULE_BASELINE_NAMES"
            for target in node.targets
        )
    )
    protocol = ast.parse((ROOT / "src/stock_forecasting/evaluation_protocol.py").read_text())
    version = next(
        node.value
        for node in protocol.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EVALUATION_PROTOCOL_VERSION"
            for target in node.targets
        )
    )
    scope.update(
        {
            "__file__": str(SOURCE),
            "hashlib": hashlib,
            "json": json,
            "math": math,
            "os": os,
            "time": time,
            "uuid": uuid,
            "datetime": datetime,
            "UTC": UTC,
            "Path": Path,
            "RULE_BASELINE_NAMES": ast.literal_eval(rules),
            "EVALUATION_PROTOCOL_VERSION": ast.literal_eval(version),
            "np": SimpleNamespace(median=statistics.median),
            "_requested_dataloader_workers": lambda _config: (2, "fixture"),
            "plan_dataloader_workers": lambda *args, **kwargs: SimpleNamespace(effective_workers=2),
            "_loader_process_options": lambda *args, **kwargs: {"num_workers": 2},
            "paired_block_comparison": Mock(return_value={"fixture": "paired"}),
        }
    )
    definitions = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope


class EvaluationStorageControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.scope = production_scope()
        self.run_id = "run-testing-storage"
        self.checkpoint = self.root / "savedModel" / self.run_id / "checkpoint-000001"
        self.checkpoint.mkdir(parents=True)
        self.output = self.root / "evaluations" / self.run_id / "validation-benchmark.json"
        self.scales = [0.03]
        for name in ("adapter.safetensors", "resolved-config.yaml"):
            (self.checkpoint / name).write_bytes(b"fixture-checkpoint")
        (self.checkpoint / "trainer-state.json").write_text(
            json.dumps(
                {
                    "runtime_robust_scales": self.scales,
                }
            )
        )
        self.numeric_config = {"baseline_max_samples_per_split": None, "neural_epochs": 2}
        self.config = SimpleNamespace(
            training=SimpleNamespace(
                output_root=self.root / "savedModel",
                evaluation_max_samples=None,
            ),
            validation=SimpleNamespace(
                output_root=self.root / "evaluations",
                require_prebuilt_baselines=True,
                model_dump=lambda **kwargs: dict(self.numeric_config),
            ),
            data=SimpleNamespace(
                fixed_split=True,
                h_start=1,
                max_horizon=14,
                dataset_profile="fixture",
                selected_datasets=["fixture"],
            ),
            model=SimpleNamespace(time_series_backend="kronos"),
            model_architecture_digest=lambda: "architecture-fixture",
            as_dict=lambda: {},
        )
        self.scope["validate_training_resume_contract"] = lambda *args, **kwargs: "training-fixture"
        self.scope["training_resume_contract"] = lambda config: {
            "dataset_artifacts": {"data": "fixture"},
        }
        self.metrics = {
            "samples": 2,
            "sample_membership": {"samples": 2, "ordered_symbol_dates_sha256": "rows"},
            "evaluation_robust_scales": self.scales,
            "daily_normalized_pinball": {"2026-01-02": 0.4},
            "aggregate": {"normalized_pinball": 0.4},
            "primary_5d": {"selection_score": 0.4},
        }
        self.cached = {
            "identity": {"baseline_id": "fixture"},
            "robust_scales": self.scales,
            "sample_counts": {"train": 4, "validation": 2, "test": 2},
            "evaluation_membership": self.metrics["sample_membership"],
            "models": {"zero_return": {"state": "complete", "metrics": self.metrics}},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self):
        return self.scope["ValidationBenchmark"](
            self.config,
            run_id=self.run_id,
            checkpoint=self.checkpoint,
            output=self.output,
            lifecycle=self.root / "lifecycle/stage1/validation.json",
            models=["zero_return", "kronos_full"],
            seeds=[42],
            resume=True,
            recompute_full_model=True,
        )

    def saved_output(self):
        first = self.runner()
        payload = copy.deepcopy(first.payload)
        payload.update(
            state="ready",
            models={
                **self.cached["models"],
                "kronos_full": {
                    "state": "complete",
                    "source": "checkpoint_recomputed",
                    "metrics": self.metrics,
                },
            },
        )
        implementation = payload["evaluation_contract"]["inputs"]["evaluation_implementation"]
        for name in self.scope["EVALUATION_CONTROL_IMPLEMENTATION_PATHS"]:
            implementation[name] = {"sha256": "old-reader-provenance", "size_bytes": 123}
        self.write(payload)
        return payload

    def write(self, payload):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload))

    def test_constructor_reuses_old_output_without_baseline_retraining_or_inference(self):
        before = self.saved_output()
        forbidden = Mock(side_effect=AssertionError("Unexpected training or inference"))
        runner = self.runner()
        self.assertEqual(runner.payload["models"], before["models"])
        runner._run_learned_baseline = forbidden
        self.scope.update(evaluate_checkpoint=forbidden, _lazy_baseline_arrays=forbidden)
        with patch(
            "stock_forecasting.baseline_contract.require_baselines",
            return_value=self.cached,
        ):
            result = runner.run()
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["models"], before["models"])
        self.assertTrue(result["baseline_reused"])
        forbidden.assert_not_called()

    def test_new_testing_entry_evaluates_only_main_model_once_and_then_resumes(self):
        evaluate = Mock(return_value={"metrics": self.metrics})
        forbidden = Mock(side_effect=AssertionError("Baseline training or preparation ran"))
        self.scope.update(evaluate_checkpoint=evaluate, _lazy_baseline_arrays=forbidden)
        with patch(
            "stock_forecasting.baseline_contract.require_baselines",
            return_value=self.cached,
        ):
            self.assertEqual(self.runner().run()["state"], "ready")
            self.assertEqual(self.runner().run()["state"], "ready")
        evaluate.assert_called_once_with(self.config, self.checkpoint, split="test")
        self.scope["stop_at_saved_boundary"].assert_called_once_with(
            section_kind="validation-model",
            section_id="kronos_full",
            artifact=self.output,
        )
        forbidden.assert_not_called()

    def test_numerical_source_changes_missing_sources_and_unknown_sources_are_rejected(self):
        original = self.saved_output()
        for name in self.scope["EVALUATION_NUMERICAL_IMPLEMENTATION_PATHS"]:
            with self.subTest(source=name):
                changed = copy.deepcopy(original)
                files = changed["evaluation_contract"]["inputs"]["evaluation_implementation"]
                files[name]["sha256"] = "changed"
                self.write(changed)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.runner()
        for missing in (True, False):
            changed = copy.deepcopy(original)
            implementation = changed["evaluation_contract"]["inputs"]["evaluation_implementation"]
            if missing:
                del implementation["metrics.py"]
            else:
                implementation["unexpected.py"] = {"sha256": "unknown"}
            self.write(changed)
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.runner()

    def test_actual_constructor_still_rejects_changed_numerical_inputs(self):
        original = self.saved_output()
        for name in (
            "run_id",
            "training_resume_contract_sha256",
            "dataset_artifacts",
            "checkpoint",
            "evaluation_protocol",
            "evaluation_schema",
            "models",
            "seeds",
            "model_architecture_sha256",
            "validation_numerical_config",
        ):
            with self.subTest(input=name):
                changed = copy.deepcopy(original)
                changed["evaluation_contract"]["inputs"][name] = {"changed": True}
                self.write(changed)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.runner()

    def test_old_pre_holdout_results_are_not_relabeled(self):
        payload = self.saved_output()
        payload["evaluation_contract"]["version"] = "5.0"
        self.write(payload)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.runner()

    def test_downloaded_report_is_preserved_but_not_reused_after_numerical_changes(self):
        report = ROOT / "artifacts/runpod/run-20260922T182053Z-1388222581/validation-benchmark.json"
        if not report.is_file():
            self.skipTest("Downloaded report is optional local verification evidence")
        original = report.read_bytes()
        payload = json.loads(original)
        inputs = payload["evaluation_contract"]["inputs"]
        self.run_id, self.checkpoint = inputs["run_id"], Path(inputs["checkpoint"]["path"])
        self.output = self.root / "evaluations" / self.run_id / "validation-benchmark.json"
        self.config.training.output_root = self.checkpoint.parent.parent
        self.config.data.fixed_split = inputs["evaluation_protocol"]["fixed_split"]
        self.config.training.evaluation_max_samples = inputs["evaluation_protocol"]["max_samples"]
        self.config.model_architecture_digest = lambda: inputs["model_architecture_sha256"]
        self.numeric_config = inputs["validation_numerical_config"]
        self.scope["validate_training_resume_contract"] = lambda *args, **kwargs: inputs[
            "training_resume_contract_sha256"
        ]
        self.scope["training_resume_contract"] = lambda config: {
            "dataset_artifacts": inputs["dataset_artifacts"],
        }
        fingerprint = self.scope["_file_fingerprint"]
        self.scope["_file_fingerprint"] = lambda path: (
            inputs["checkpoint"]["files"][path.name]
            if path.parent == self.checkpoint
            else fingerprint(path)
        )
        self.write(payload)
        with self.assertRaisesRegex(ValueError, "does not match this run contract"):
            self.scope["ValidationBenchmark"](
                self.config,
                run_id=self.run_id,
                checkpoint=self.checkpoint,
                output=self.output,
                lifecycle=self.root / "lifecycle/stage1/validation.json",
                models=inputs["models"],
                seeds=inputs["seeds"],
                resume=True,
                recompute_full_model=True,
            )
        self.assertEqual(report.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
