"""Exercise paired status/discovery/download without training or cloud access."""

from __future__ import annotations

import argparse
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "tests/test_runpod_run_status_control.py"))
PAIRED = runpy.run_path(str(ROOT / "scripts/runpod_paired.py"))
QUERY = BASE["QUERY"]
DOWNLOAD = runpy.run_path(str(ROOT / "scripts/download_runpod_pair.py"))
DOWNLOAD_TESTS = runpy.run_path(str(ROOT / "tests/test_training_result_download.py"))
PARENT = "run-20261014T120000Z-pair"


def pair_fixtures(*, second_complete=True, workflow="train"):
    selection = PAIRED["selection_api"]()["_build_selection"](argparse.Namespace(
        stage="stage2", data_profile="us_tw_eodhd", dataset_revision="v1",
        start="2021-01-01", end="2026-06-01", h_start=1, universe="all",
        stocks=[], etfs=[], symbol_limit=None,
    ), ROOT)
    pair = PAIRED["build_plan"](ROOT, selection, "a-signal64", 42, workflow, PARENT)
    pair["created_at"] = "2026-10-14T12:00:00Z"
    payloads = {f"lifecycle/runs/{PARENT}/pair.json": pair}
    for task, child in pair["children"].items():
        run = child["run_id"]
        records = BASE["fixtures"](
            run, child["experiment"], complete=second_complete or task == "alpha_point",
        )
        records[f"lifecycle/runs/{run}/selection.json"] = child["selection"]
        manifest = records[f"savedModel/{run}/run-manifest.json"]
        manifest["selection_provenance"]["selection_id"] = child["selection"]["selection_id"]
        manifest["training_resume_contract"]["training"] = {"seed": 42}
        manifest["training_resume_contract"]["model"]["prediction_task"] = task
        if f"evaluations/{run}/validation-benchmark.json" in records:
            records[f"evaluations/{run}/validation-benchmark.json"]["prediction_task"] = task
        payloads.update(records)
    return pair, payloads


class PairedResultsTests(unittest.TestCase):
    def test_status_complete_requires_both_models_and_test_not_validation(self):
        pair, data = pair_fixtures()
        result = QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["configuration"]["experiment"], "a-signal64")
        self.assertEqual(set(result["children"]), {"alpha_point", "absolute_direction"})
        child = pair["children"]["absolute_direction"]["run_id"]
        data[f"evaluations/{child}/validation-benchmark.json"]["evaluation_split"] = "validation"
        self.assertNotEqual(QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)["state"], "complete")

    def test_parent_failure_cannot_be_hidden_by_stale_running_child(self):
        _, data = pair_fixtures(second_complete=False)
        data[f"lifecycle/runs/{PARENT}/training.json"] = {
            "kind": "stage1-training", "wandb_run_id": PARENT, "state": "failed",
            "generated_at": "2026-10-14T13:00:00Z",
        }
        result = QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(result["children"]["alpha_point"]["state"], "complete")

    def test_unstarted_second_task_is_visible_and_resume_chooses_parent(self):
        pair, data = pair_fixtures()
        child = pair["children"]["absolute_direction"]["run_id"]
        data = {key: value for key, value in data.items() if child not in key}
        result = QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(result["children"]["absolute_direction"]["state"], "not_started")
        self.assertEqual(BASE["FakeReader"](data).choose("resume"), PARENT)
        with self.assertRaisesRegex(ValueError, "No eligible run"):
            BASE["FakeReader"](data).choose("results")

    def test_complete_parent_marker_never_substitutes_for_child_artifacts(self):
        pair, _ = pair_fixtures()
        data = {f"lifecycle/runs/{PARENT}/pair.json": pair}
        data[f"lifecycle/runs/{PARENT}/training.json"] = {
            "kind": "stage1-training", "wandb_run_id": PARENT,
            "state": "ready", "training_completed": True,
        }
        result = QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)
        self.assertNotEqual(result["state"], "complete")
        self.assertEqual(result["configuration"]["date_range"]["start_inclusive"], "2021-01-01")

    def test_wrong_task_artifact_never_counts_as_the_second_model(self):
        for artifact in ("manifest", "report"):
            pair, data = pair_fixtures()
            child = pair["children"]["absolute_direction"]["run_id"]
            if artifact == "manifest":
                data[f"savedModel/{child}/run-manifest.json"]["training_resume_contract"]["model"]["prediction_task"] = "alpha_point"
            else:
                data[f"evaluations/{child}/validation-benchmark.json"]["prediction_task"] = "alpha_point"
            with self.subTest(artifact=artifact), self.assertRaisesRegex(ValueError, "prediction task mismatch"):
                QUERY["inspect_run"](BASE["FakeReader"](data), PARENT)

    def test_resume_can_finish_evaluation_after_both_models_trained(self):
        pair, data = pair_fixtures()
        child = pair["children"]["absolute_direction"]["run_id"]
        del data[f"evaluations/{child}/validation-benchmark.json"]
        del data[f"lifecycle/runs/{child}/validation.json"]
        data[f"lifecycle/runs/{PARENT}/training-completed.json"] = {
            "kind": "stage1-training-completion", "run_id": PARENT, "state": "ready",
            "training_completed": True, "completed_at": "2026-10-14T14:00:00Z",
        }
        self.assertEqual(BASE["FakeReader"](data).choose("resume"), PARENT)

    def test_default_inventory_one_row_task_filter_still_available(self):
        pair, data = pair_fixtures()
        reader = BASE["FakeReader"](data)
        self.assertEqual(QUERY["discover"](reader), [PARENT])
        discovered = QUERY["discover"](reader, experiment="a-alpha-point64")
        self.assertEqual(set(discovered), {PARENT, pair["children"]["alpha_point"]["run_id"]})
        self.assertEqual(reader.choose("results"), PARENT)

    def test_baseline_pair_never_masquerades_as_a_training_result(self):
        _, data = pair_fixtures(workflow="baseline")
        reader = BASE["FakeReader"](data)
        self.assertEqual(QUERY["discover"](reader), [])
        result = QUERY["inspect_run"](reader, PARENT)
        self.assertEqual(result["kind"], "paired-baseline")
        self.assertEqual(result["state"], "unknown")
        with self.assertRaisesRegex(ValueError, "No eligible run"):
            reader.choose("results")

    def test_invalid_pair_ownership_fails_without_hiding_children(self):
        pair, data = pair_fixtures()
        data[f"lifecycle/runs/{PARENT}/pair.json"]["children"]["alpha_point"]["run_id"] = "other-run"
        with self.assertRaisesRegex(ValueError, "owned"):
            QUERY["discover"](BASE["FakeReader"](data))

    def test_pair_reader_does_not_validate_against_current_config_files(self):
        pair, data = pair_fixtures()
        reader = BASE["FakeReader"](data)
        with patch.object(Path, "read_text", side_effect=AssertionError("No current YAML reads")):
            result = QUERY["inspect_pair"](reader, PARENT, pair)
        self.assertEqual(result["state"], "complete")

    def download_fixture(self, data):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        reader = BASE["FakeReader"](data)
        reader.root = Path(temporary.name)
        (reader.root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/runpod_run_status.py", reader.root / "scripts/runpod_run_status.py")
        (reader.root / "src/stock_forecasting").mkdir(parents=True)
        shutil.copyfile(ROOT / "src/stock_forecasting/runtime_resources.py", reader.root / "src/stock_forecasting/runtime_resources.py")
        reader.validate_pair = lambda payload, run: PAIRED["validate_manifest"](payload, run_id=run)
        return reader

    def test_download_runs_two_children_preserves_scope_resume_and_no_weight_copies(self):
        pair, data = pair_fixtures()
        reader = self.download_fixture(data)
        with (
            patch.object(DOWNLOAD["subprocess"], "run", return_value=argparse.Namespace(returncode=0)) as call,
            patch.object(DOWNLOAD["subprocess"], "check_output", return_value="page size of 4096 bytes\nPages free: 500000.\nPages inactive: 500000."),
        ):
            result = DOWNLOAD["download_pair"](reader, pair, script="download.sh", resume=True, checkpoint_scope="best")
        self.assertEqual(call.call_count, 2)
        for request in call.call_args_list:
            self.assertIn("--resume", request.args[0])
            self.assertIn("best", request.args[0])
        target = reader.root / "artifacts/runpod" / PARENT
        self.assertEqual({file.name for file in target.iterdir()}, {"pair.json", "download-result.json"})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(set(result["children"]), {"alpha_point", "absolute_direction"})

    def test_failed_or_unfinished_download_never_writes_complete_result(self):
        pair, data = pair_fixtures()
        reader = self.download_fixture(data)
        with (
            patch.object(DOWNLOAD["subprocess"], "run", return_value=argparse.Namespace(returncode=1)),
            patch.object(DOWNLOAD["subprocess"], "check_output", return_value="page size of 4096 bytes\nPages free: 500000.\nPages inactive: 500000."),
        ):
            with self.assertRaisesRegex(ValueError, "retry with --resume"):
                DOWNLOAD["download_pair"](reader, pair, script="download.sh")
        self.assertFalse((reader.root / "artifacts/runpod" / PARENT / "download-result.json").exists())
        pair, data = pair_fixtures(second_complete=False)
        reader = self.download_fixture(data)
        with patch.object(DOWNLOAD["subprocess"], "run") as call:
            with self.assertRaisesRegex(ValueError, "not completed"):
                DOWNLOAD["download_pair"](reader, pair, script="download.sh")
            call.assert_not_called()

    def test_public_download_shell_fetches_both_models_from_fake_object_store(self):
        fixture = DOWNLOAD_TESTS["TrainingResultDownloadTests"]()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for name in ("runpod_paired.py", "runpod_selection.py", "runpod_run_status.py", "download_runpod_pair.py"):
            shutil.copyfile(ROOT / "scripts" / name, fixture.root / "scripts" / name)
        source = fixture.root / "src/stock_forecasting"
        source.mkdir(parents=True)
        for name in ("dataset_identity.py", "dataset_profiles.py", "runtime_resources.py"):
            shutil.copyfile(ROOT / "src/stock_forecasting" / name, source / name)
        pair, data = pair_fixtures()
        for key, value in data.items():
            fixture.write_record(key, value)
        for child in pair["children"].values():
            saved = fixture.root / "cloud/savedModel" / child["run_id"]
            (saved / "resolved-config.yaml").write_text("experiment_name: task-fixture\n")
            for name in ("checkpoint-leaderboard.json", "best-checkpoint.json"):
                (saved / name).write_text("{}")
            (saved / "checkpoint-000001").mkdir()
            (saved / "checkpoint-000001/adapter.safetensors").write_text("task fixture weights")
        result = fixture.command("--checkpointScope", "best", PARENT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Downloaded paired RunPod result", result.stdout)
        for child in pair["children"].values():
            target = fixture.root / "artifacts/runpod" / child["run_id"]
            self.assertTrue((target / "checkpoint-000001/adapter.safetensors").is_file())
            self.assertTrue((target / "validation-benchmark.json").is_file())
        self.assertTrue((fixture.root / "artifacts/runpod" / PARENT / "download-result.json").is_file())
        retry = fixture.command("--resume", "--checkpointScope", "best")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertIn(f"run_id={PARENT}", retry.stdout)


if __name__ == "__main__":
    unittest.main()
