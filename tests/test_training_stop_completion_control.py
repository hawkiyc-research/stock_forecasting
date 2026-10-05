"""Execute production completion control flow without a local ML environment."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "src/stock_forecasting/training.py"
spec = importlib.util.spec_from_file_location(
    "runtime_stop_control", ROOT / "src/stock_forecasting/runpod/runtime_stop.py"
)
STOP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STOP)


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def production_flow(*, resume):
    """Extract actual loop decisions and completion checks, not rewritten logic."""
    tree = ast.parse(TRAINING.read_text())
    train = next(node for node in tree.body if getattr(node, "name", None) == "_train_with_lease")
    body = next(node.body for node in train.body if isinstance(node, ast.Try))
    loop = next(node for node in body if isinstance(node, ast.For))
    completion_start = body.index(loop) + 1
    completion_end = next(i for i, node in enumerate(body) if assigned(node, "completion_result"))
    if resume:
        start = next(i for i, node in enumerate(body) if assigned(node, "stop_training"))
        decisions = body[start:completion_start]
    else:
        batches = next(node for node in loop.body if isinstance(node, ast.For))
        validation = next(
            node
            for node in batches.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "runtime_global_step in evaluation_alignment"
        )
        start = next(i for i, node in enumerate(validation.body) if assigned(node, "best_path"))
        # The checkpoint has already been saved. Execute its real boundary decisions
        # inside one loop iteration, including the original break statements.
        decisions = [
            ast.For(
                target=ast.Name(id="_fixture_iteration", ctx=ast.Store()),
                iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                body=validation.body[start:] + batches.body[-2:],
                orelse=[],
            )
        ]
    nodes = decisions + body[completion_start : completion_end + 1]
    nodes.append(
        ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=ast.Name(id="completion_result", ctx=ast.Load()),
        )
    )
    return compile(
        ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(TRAINING), "exec"
    )


class TrainingStopCompletionControlTests(unittest.TestCase):
    def run_flow(self, *, early, final=False, request=True, resume=False):
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            step = 590320 if final else 165290
            checkpoint = volume / "savedModel/run-fixture" / f"checkpoint-{step:06d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer-state.json").write_text(json.dumps({"global_step": step}))
            signal = volume / "lifecycle/runs/run-fixture/pods/pod-fixture"
            signal.mkdir(parents=True)
            if request:
                (signal / "stop-request.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "runtime-stop-request",
                            "pod_id": "pod-fixture",
                            "run_id": "run-fixture",
                        }
                    )
                )
            calls = []

            def save_completion(**kwargs):
                calls.append(kwargs)
                return kwargs

            def unexpected_training(*_args, **_kwargs):
                raise AssertionError("A terminal checkpoint must not run training or validation")

            scope = {
                "Path": Path,
                "time": time,
                "checkpoint": checkpoint,
                "last_ranking": {"best_checkpoint": str(checkpoint)},
                "best_checkpoint": checkpoint,
                "global_step": step,
                "runtime_global_step": step,
                "last_evaluation_step": step,
                "last_checkpoint_step": step,
                "last_runtime_evaluation_step": step,
                "last_runtime_checkpoint_step": step,
                "runtime_configured_steps": 590320,
                "completed_epochs_at_step": 5 if final else 1,
                "completed_epochs": 5 if final else 1,
                "starting_epoch": 5 if final else 1,
                "stop_training": False,
                "stop_reason": "epochs_completed",
                "early_stopping": SimpleNamespace(
                    triggered=early,
                    evaluation_count=7,
                    as_dict=lambda: {"triggered": early},
                ),
                "canonical_schedule": SimpleNamespace(configured_optimizer_steps=590320),
                "config": SimpleNamespace(
                    training=SimpleNamespace(epochs=5),
                    model_architecture_digest=lambda: "unchanged",
                ),
                "tracking": SimpleNamespace(directory=checkpoint.parent),
                "bundle": SimpleNamespace(model=object()),
                "selected_train_samples": 30224384,
                "processed_train_samples": 42314112,
                "last_validation_flat_metrics": {"samples": 1232972, "loss": 0.30386137377579187},
                "pipeline_timings": {},
                "stop_at_saved_boundary": STOP.stop_at_saved_boundary,
                "save_training_completion_result": save_completion,
                "iter_device_batches": unexpected_training,
                "evaluate_loader": unexpected_training,
                "forward_batch": unexpected_training,
                "optimizer": SimpleNamespace(step=unexpected_training),
            }
            environment = {
                "NETWORK_VOLUME_ROOT": str(volume),
                "RUNPOD_RUN_KEY": "run-fixture",
                "RUNPOD_POD_ID": "pod-fixture",
            }
            with patch.dict(os.environ, environment):
                if not early and not final:
                    with self.assertRaises(STOP.RuntimeStopRequested):
                        exec(production_flow(resume=False), scope)
                    self.assertEqual(calls, [])
                    self.assertTrue((signal / "stop-ready.json").is_file())
                    return
                exec(production_flow(resume=resume), scope)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                scope["result"]["stop_reason"], "early_stopping" if early else "epochs_completed"
            )
            self.assertEqual(scope["result"]["global_step"], step)
            self.assertEqual(scope["result"]["metrics"]["samples"], 1232972)
            # A cutoff acknowledgement must not precede formal completion.
            self.assertFalse((signal / "stop-ready.json").exists())

    def test_early_stop_and_cutoff_publish_completion(self):
        self.run_flow(early=True)

    def test_final_step_and_cutoff_publish_completion(self):
        self.run_flow(early=False, final=True)

    def test_unfinished_run_still_acknowledges_cutoff_after_checkpoint(self):
        self.run_flow(early=False)

    def test_resume_early_stopped_checkpoint_skips_training_and_validation(self):
        self.run_flow(early=True, resume=True)

    def test_resume_final_checkpoint_skips_training_and_validation(self):
        self.run_flow(early=False, final=True, resume=True)

    def test_natural_completion_without_cutoff_remains_unchanged(self):
        self.run_flow(early=True, request=False)


if __name__ == "__main__":
    unittest.main()
