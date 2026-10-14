"""Exercise the durable section boundary used for a local runtime stop."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/stock_forecasting/runpod/runtime_stop.py"
SPEC = importlib.util.spec_from_file_location("runpod_runtime_stop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STOP)


class RuntimeStopTests(unittest.TestCase):
    def test_paired_child_acknowledges_parent_stop_only_for_its_own_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory).resolve()
            parent_id, child_id = "run-pair", "run-pair-direction"
            signal_dir = volume / "lifecycle/runs" / parent_id / "pods/pod-fixture"
            signal_dir.mkdir(parents=True)
            plan_path = signal_dir.parents[1] / "pair.json"
            plan_path.write_text(json.dumps({
                "schema_version": 1, "kind": "paired-training", "workflow": "train", "run_id": parent_id,
                "children": {"alpha_point": {"run_id": parent_id + "-alpha"},
                             "absolute_direction": {"run_id": child_id}},
            }))
            (signal_dir / "stop-request.json").write_text(json.dumps({
                "schema_version": 1, "kind": "runtime-stop-request", "pod_id": "pod-fixture", "run_id": parent_id,
            }))
            checkpoint = volume / "savedModel" / child_id / "checkpoint-000123"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer-state.json").write_text('{"global_step":123}')
            env = {"NETWORK_VOLUME_ROOT": str(volume), "RUNPOD_RUN_KEY": child_id,
                   "RUNPOD_POD_ID": "pod-fixture", "RUNPOD_PARENT_RUN_ID": parent_id,
                   "RUNPOD_PAIRED_PLAN_PATH": str(plan_path)}
            with patch.dict(os.environ, env):
                with self.assertRaises(STOP.RuntimeStopRequested):
                    STOP.stop_at_saved_boundary(section_kind="training-checkpoint", section_id=checkpoint.name, artifact=checkpoint)
            acknowledgement = json.loads((signal_dir / "stop-ready.json").read_text())
            self.assertEqual(acknowledgement["run_id"], parent_id)
            self.assertEqual(acknowledgement["child_run_id"], child_id)
            self.assertEqual(acknowledgement["artifact"], str(checkpoint))
            with patch.dict(os.environ, dict(env, RUNPOD_RUN_KEY="run-unrelated")):
                with self.assertRaisesRegex(ValueError, "does not own"):
                    STOP.stop_at_saved_boundary(section_kind="training-checkpoint", section_id=checkpoint.name, artifact=checkpoint)

    def test_training_waits_for_matching_request_and_committed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            checkpoint = volume / "savedModel/run-fixture/checkpoint-000123"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer-state.json").write_text(
                '{"global_step":123}', encoding="utf-8"
            )
            signal_dir = volume / "lifecycle/runs/run-fixture/pods/pod-fixture"
            signal_dir.mkdir(parents=True)
            environment = {
                "NETWORK_VOLUME_ROOT": str(volume),
                "RUNPOD_RUN_KEY": "run-fixture",
                "RUNPOD_POD_ID": "pod-fixture",
            }
            with patch.dict(os.environ, environment):
                STOP.stop_at_saved_boundary(
                    section_kind="training-checkpoint",
                    section_id="checkpoint-000123",
                    artifact=checkpoint,
                )
                self.assertFalse((signal_dir / "stop-ready.json").exists())
                (signal_dir / "stop-request.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "kind": "runtime-stop-request",
                        "pod_id": "pod-other",
                        "run_id": "run-fixture",
                    }),
                    encoding="utf-8",
                )
                STOP.stop_at_saved_boundary(
                    section_kind="training-checkpoint",
                    section_id="checkpoint-000123",
                    artifact=checkpoint,
                )
                self.assertFalse((signal_dir / "stop-ready.json").exists())
                (signal_dir / "stop-request.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "kind": "runtime-stop-request",
                        "pod_id": "pod-fixture",
                        "run_id": "run-fixture",
                    }),
                    encoding="utf-8",
                )
                with self.assertRaises(STOP.RuntimeStopRequested):
                    STOP.stop_at_saved_boundary(
                        section_kind="training-checkpoint",
                        section_id="checkpoint-000123",
                        artifact=checkpoint,
                    )
            acknowledged = json.loads((signal_dir / "stop-ready.json").read_text())
            self.assertEqual(acknowledged["section_id"], "checkpoint-000123")
            self.assertEqual(acknowledged["artifact"], str(checkpoint.resolve()))

    def test_validation_seed_ack_requires_persisted_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            result = volume / "evaluations/run-fixture/validation-benchmark.json"
            result.parent.mkdir(parents=True)
            signal_dir = volume / "lifecycle/runs/run-fixture/pods/pod-fixture"
            signal_dir.mkdir(parents=True)
            (signal_dir / "stop-request.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "kind": "runtime-stop-request",
                    "pod_id": "pod-fixture",
                    "run_id": "run-fixture",
                }),
                encoding="utf-8",
            )
            environment = {
                "NETWORK_VOLUME_ROOT": str(volume),
                "RUNPOD_RUN_KEY": "run-fixture",
                "RUNPOD_POD_ID": "pod-fixture",
            }
            with patch.dict(os.environ, environment):
                with self.assertRaises(FileNotFoundError):
                    STOP.stop_at_saved_boundary(
                        section_kind="validation-seed",
                        section_id="gru.3",
                        artifact=result,
                    )
                result.write_text(
                    '{"run_id":"run-fixture","models":{"gru":{"seed_results":{"3":{"score":1}}}}}',
                    encoding="utf-8",
                )
                with self.assertRaises(STOP.RuntimeStopRequested):
                    STOP.stop_at_saved_boundary(
                        section_kind="validation-seed",
                        section_id="gru.3",
                        artifact=result,
                    )
            acknowledged = json.loads((signal_dir / "stop-ready.json").read_text())
            self.assertEqual(acknowledged["section_kind"], "validation-seed")
            self.assertEqual(acknowledged["section_id"], "gru.3")


if __name__ == "__main__":
    unittest.main()
