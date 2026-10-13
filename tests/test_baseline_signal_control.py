"""Dependency-free admission and identity tests for isolated prediction tasks."""

import copy
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_signal_contract.py"))
CONTRACT = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
CACHE = runpy.run_path(str(ROOT / "scripts/runpod_baseline_cache.py"))


class BaselineSignalControlTests(unittest.TestCase):
    def test_quantile_parameters_are_unchanged_and_independent_tasks_drop_costs(self):
        original = json.loads((ROOT / "configs/baseline.json").read_text())
        self.assertEqual(TOOLS["parameters_for_task"](original, {}), original)
        for task in ("alpha_point", "absolute_direction"):
            result = TOOLS["parameters_for_task"](
                original, {"baseline_task": {"prediction_task": task}}
            )
            self.assertEqual(result["direction"], {"enabled": False})
            self.assertEqual(result["models"], original["models"])
            self.assertNotIn("costs", result["signal_task"])
        self.assertIn("costs", original["direction"])

    def test_simple_percent_log_boundaries_are_asymmetric(self):
        lower, upper = TOOLS["log_thresholds"](0.01)
        self.assertAlmostEqual(math.expm1(lower), -0.01)
        self.assertAlmostEqual(math.expm1(upper), 0.01)
        self.assertNotEqual(-lower, upper)
        for invalid in (0, 1, -0.1, float("nan"), True):
            with self.assertRaises(ValueError):
                TOOLS["log_thresholds"](invalid)

    def test_task_relevant_fields_only_affect_identity(self):
        point = {
            "prediction_task": "alpha_point",
            "point_loss_beta": 1,
            "neutral_return_threshold": 0.5,
            "lora_rank": 512,
            "seed": 12,
        }
        self.assertEqual(
            TOOLS["signal_task"]({"baseline_task": point}),
            {"prediction_task": "alpha_point", "point_loss_beta": 1.0},
        )
        direction = {"prediction_task": "absolute_direction", "point_loss_beta": 10}
        self.assertEqual(
            TOOLS["signal_task"]({"baseline_task": direction}),
            {"prediction_task": "absolute_direction", "neutral_return_threshold": 0.01},
        )

    def test_full_class_audit_rejects_missing_classes_and_reports_imbalance(self):
        task = {"prediction_task": "absolute_direction", "neutral_return_threshold": 0.01}
        audit = {
            **task,
            "samples": 1000,
            "counts": {"overall": {"1": [1, 998, 1]}, "US": {"1": [1, 998, 1]}},
        }
        warnings = TOOLS["validate_class_audit"](audit, task, 1000, [1])
        self.assertEqual(len(warnings), 2)
        invalid = copy.deepcopy(audit)
        invalid["counts"]["US"]["1"] = [0, 999, 1]
        with self.assertRaisesRegex(ValueError, "Missing training class"):
            TOOLS["validate_class_audit"](invalid, task, 1000, [1])
        with self.assertRaisesRegex(ValueError, "every training window"):
            TOOLS["validate_class_audit"](audit, task, 999, [1])

    def test_baseline_identity_separates_labels_and_not_main_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for directory in ("src", "configs"):
                shutil.copytree(
                    ROOT / directory,
                    project / directory,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            selected = {
                "dataset_request": {
                    "date_range": {"start_inclusive": "2021-01-01", "end_exclusive": "2026-06-01"}
                },
                "baseline_task": {"prediction_task": "alpha_point", "point_loss_beta": 1.0},
            }
            before = CONTRACT["baseline_contract"](project, selected)
            self.assertEqual(before["contract"]["parameters"]["direction"], {"enabled": False})
            (project / "configs/experiments/a_lora64.yaml").write_text(
                "unrelated capacity change\n"
            )
            selected["stage"] = {"experiment_id": "different-capacity"}
            self.assertEqual(CONTRACT["baseline_contract"](project, selected), before)
            selected["baseline_task"]["point_loss_beta"] = 0.5
            self.assertNotEqual(CONTRACT["baseline_contract"](project, selected), before)
            selected["baseline_task"] = {
                "prediction_task": "absolute_direction",
                "neutral_return_threshold": 0.01,
            }
            self.assertNotEqual(CONTRACT["baseline_contract"](project, selected), before)


class PendingClassAuditGateTests(unittest.TestCase):
    def setUp(self):
        fixtures = runpy.run_path(str(ROOT / "tests/test_baseline_readiness_control.py"))
        base, gate = fixtures["BASE"], fixtures["GATE"]
        self.selection = base["selection"]()
        self.selection["baseline_task"] = {
            "prediction_task": "absolute_direction",
            "neutral_return_threshold": 0.01,
        }
        evaluation = base["selection"]("2016-01-01")
        self.objects = {**base["fixture"](self.selection), **base["fixture"](evaluation)}
        for selected in (self.selection, evaluation):
            self.objects.update(fixtures["clean_fixture"](selected, self.objects))
        self.identity = CONTRACT["baseline_contract"](ROOT, self.selection)
        _, _, dataset_ids = gate["baseline_sources"](ROOT, self.selection)
        counts, sources = {}, {}
        for split, dataset_id in dataset_ids.items():
            root = f"datasets/{dataset_id}/prepared/"
            raw = self.objects[root + "bar-store/bar-store.json"]
            sha = hashlib.sha256(raw).hexdigest()
            universe_identity = gate["cleaning_identity"](
                ROOT, json.loads(raw), sha, self.identity["contract"]["data_cleaning"]
            )
            key = gate["digest"](universe_identity)
            universe = json.loads(self.objects[root + f"sample-universes/{key}/universe.json"])
            counts[split] = universe["split_counts"][split]
            sources[split] = {
                "manifest_sha256": sha,
                "sample_universe": key,
                "samples": counts[split],
            }
        self.cache = {
            "identity": {
                "schema": 3,
                "signal_task": self.selection["baseline_task"],
                "horizons": list(range(1, 15)),
                "split_sources": sources,
                "manifest_sha256": sources["train"]["manifest_sha256"],
            },
            "counts": counts,
            "class_label_audit": {
                **self.selection["baseline_task"],
                "samples": counts["train"],
                "counts": {
                    market: {str(h): values[:] for h in range(1, 15)}
                    for market, values in (("overall", [60, 60, 60]), ("US", [60, 60, 60]))
                },
            },
        }
        self.key = f"baselines/{self.identity['baseline_id']}/inputs/complete.json"
        self.calls = []

    def check(self, cache=True):
        if cache:
            self.objects[self.key] = json.dumps(self.cache).encode()

        def transport(command, **kwargs):
            self.calls.append(command)
            self.assertEqual(command[2:4], ["s3", "cp"])
            key = command[4].split("/", 3)[3]
            content = self.objects.get(key)
            if content is None:
                return subprocess.CompletedProcess(
                    command, 1, "" if kwargs.get("text") else b"", "NoSuchKey"
                )
            return subprocess.CompletedProcess(
                command, 0, content.decode() if kwargs.get("text") else content, ""
            )

        with (
            patch.object(subprocess, "run", side_effect=transport),
            patch.dict(
                os.environ, {"RUNPOD_NETWORK_VOLUME_ID": "fixture-volume", "RUNPOD_TEST_MODE": "1"}
            ),
        ):
            return CACHE["check_baseline"](ROOT, self.selection)

    def test_fresh_and_passing_partial_build_remain_incomplete(self):
        self.assertFalse(self.check(cache=False)["complete"])
        self.assertFalse(self.check()["complete"])

    def test_failed_existing_full_audit_blocks_before_paid_retry(self):
        for market in ("overall", "US"):
            self.cache["class_label_audit"]["counts"][market]["1"] = [1, 178, 1]
        with self.assertRaisesRegex(ValueError, "extreme class imbalance"):
            self.check()
        self.assertTrue(all(command[2:4] == ["s3", "cp"] for command in self.calls))

    def test_stale_or_incomplete_audit_cannot_satisfy_gate(self):
        original = copy.deepcopy(self.cache)
        for change in (
            lambda payload: payload["identity"]["signal_task"].update(
                neutral_return_threshold=0.025
            ),
            lambda payload: payload["identity"]["horizons"].pop(),
            lambda payload: payload["class_label_audit"]["counts"]["US"].pop("1"),
            lambda payload: payload["identity"]["split_sources"]["train"].update(
                manifest_sha256="a" * 64
            ),
        ):
            self.cache = copy.deepcopy(original)
            change(self.cache)
            with self.assertRaises(ValueError):
                self.check()

    def committed_resume(self):
        return {
            "schema_version": 1,
            "identity": copy.deepcopy(self.cache["identity"]),
            "count": self.cache["counts"]["train"],
            "committed_rows": self.cache["counts"]["train"],
            "class_label_counts": copy.deepcopy(self.cache["class_label_audit"]["counts"]),
        }

    def save_resume(self, payload):
        key = self.key.removesuffix("complete.json") + "train/resume.json"
        self.objects[key] = json.dumps(payload).encode()

    def test_full_committed_missing_class_blocks_without_completion_file(self):
        resume = self.committed_resume()
        for market in ("overall", "US"):
            resume["class_label_counts"][market]["1"] = [0, 120, 60]
        self.save_resume(resume)
        with self.assertRaisesRegex(ValueError, "Missing training class"):
            self.check(cache=False)
        self.assertNotIn(self.key, self.objects)
        self.assertTrue(all(command[2:4] == ["s3", "cp"] for command in self.calls))

    def test_full_committed_passing_audit_is_not_baseline_completion(self):
        self.save_resume(self.committed_resume())
        self.assertFalse(self.check(cache=False)["complete"])
        self.assertNotIn(self.key, self.objects)

    def test_partial_committed_counts_do_not_trigger_class_rejection(self):
        resume = self.committed_resume()
        resume["committed_rows"] = 60
        resume["class_label_counts"] = {
            market: {str(h): [0, 60, 0] for h in range(1, 15)} for market in ("overall", "US")
        }
        self.save_resume(resume)
        self.assertFalse(self.check(cache=False)["complete"])
        self.assertNotIn(self.key, self.objects)

    def test_invalid_resume_cursor_is_rejected(self):
        for cursor in (-1, 181, 180.0, True, None):
            with self.subTest(cursor=cursor):
                resume = self.committed_resume()
                resume["committed_rows"] = cursor
                self.save_resume(resume)
                with self.assertRaisesRegex(ValueError, "resume cursor"):
                    self.check(cache=False)

    def test_full_resume_requires_current_identity_and_complete_class_counts(self):
        for change in (
            lambda payload: payload["identity"]["signal_task"].update(
                neutral_return_threshold=0.025
            ),
            lambda payload: payload["identity"]["horizons"].pop(),
            lambda payload: payload["identity"]["split_sources"]["train"].update(
                manifest_sha256="a" * 64
            ),
            lambda payload: payload["identity"]["split_sources"]["train"].update(samples=181),
            lambda payload: payload["class_label_counts"]["US"].pop("1"),
            lambda payload: payload.pop("class_label_counts"),
        ):
            resume = self.committed_resume()
            change(resume)
            self.save_resume(resume)
            with self.assertRaises(ValueError):
                self.check(cache=False)


if __name__ == "__main__":
    unittest.main()
