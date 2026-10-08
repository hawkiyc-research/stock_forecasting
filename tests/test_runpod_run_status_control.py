"""Exercise read-only run discovery without a Python training environment or Pod."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import runpy
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
QUERY = runpy.run_path(str(ROOT / "scripts/runpod_run_status.py"))
RUNS = runpy.run_path(str(ROOT / "scripts/runpod_runs.py"))
RUN_A = "run-20261001T120000Z-1"
RUN_B = "run-20261002T130000Z-2"


def fixtures(run=RUN_A, experiment="a-lora32", complete=True):
    selection = {
        "selection_id": "selection-0123456789abcdef",
        "created_at": "2026-01-01T00:00:00Z",
        "stage": {
            "name": "stage2",
            "experiment": experiment,
            "config_path": "configs/experiments/a_lora32.yaml",
            "feature_mode": "combined",
        },
        "dataset_request": {
            "profile": "us_tw_eodhd",
            "revision": "v1",
            "date_range": {"start_inclusive": "2021-01-01", "end_exclusive": "2026-06-01"},
            "preparation": {
                "h_start": 1,
                "window_size": 128,
                "fixed_split": {
                    "train_end_exclusive": "2025-06-01",
                    "validation_end_exclusive": "2025-12-01",
                    "test_end_exclusive": "2026-06-01",
                },
            },
            "universe": {"mode": "all", "us_stocks": [], "us_etfs": ["SPY"], "symbol_limit": 0},
        },
    }
    payloads = {
        f"lifecycle/runs/{run}/selection.json": selection,
        f"savedModel/{run}/run-manifest.json": {
            "run_id": run,
            "created_at": "2026-10-02T13:05:00Z",
            "selection_provenance": {
                "selection_id": selection["selection_id"],
                "requested_dataset": selection["dataset_request"],
                "selected_stage": "stage2",
                "stage_config_path": "old.yaml",
            },
            "training_resume_contract": {"model": {"lora": {"rank": 32}}},
        },
        f"lifecycle/runs/{run}/training.json": {
            "kind": "stage1-training",
            "wandb_run_id": run,
            "state": "ready" if complete else "running",
            "training_completed": complete,
            "generated_at": "2026-10-03T02:00:00Z",
            "pod_id": "pod-a",
        },
        f"lifecycle/runs/{run}/wandb.json": {
            "run_id": run,
            "state": "offline_pending",
            "components": {"training": {"state": "offline_pending"}},
        },
    }
    if complete:
        payloads.update(
            {
                f"lifecycle/runs/{run}/training-completed.json": {
                    "kind": "stage1-training-completion",
                    "run_id": run,
                    "state": "ready",
                    "training_completed": True,
                    "completed_at": "2026-10-03T01:00:00Z",
                },
                f"lifecycle/runs/{run}/validation.json": {
                    "kind": "stage1-validation",
                    "wandb_run_id": run,
                    "state": "ready",
                    "validation_completed": True,
                    "generated_at": "2026-10-03T02:00:00Z",
                },
                f"evaluations/{run}/validation-benchmark.json": {
                    "run_id": run,
                    "state": "ready",
                    "completed_at": "2026-10-03T02:00:00Z",
                    "evaluation_split": "test",
                    "selection_split": "validation",
                },
            }
        )
    return payloads


class FakeReader(RUNS["RunRecords"]):
    def __init__(self, payloads):
        super().__init__(ROOT, "fixture-volume")
        self.payloads = copy.deepcopy(payloads)
        self.calls = []

    def read(self, key, *, optional=False):
        self.calls.append(key)
        if key not in self.payloads and not optional:
            raise ValueError("Missing fixture: " + key)
        return copy.deepcopy(self.payloads.get(key))

    def call(self, *args, **kwargs):
        self.calls.append(args)
        prefix = args[args.index("--prefix") + 1]
        if prefix == "lifecycle/runs/":
            return {"Contents": [{"Key": key} for key in self.payloads if key.startswith(prefix)]}
        return {
            "CommonPrefixes": [
                {"Prefix": f"savedModel/{run}/"}
                for run in sorted(
                    {key.split("/")[1] for key in self.payloads if key.startswith("savedModel/")}
                )
            ]
        }


class RunStatusTests(unittest.TestCase):
    def inspect(self, payloads=None, run=RUN_A):
        return QUERY["inspect_run"](FakeReader(fixtures() if payloads is None else payloads), run)

    def invoke(self, command, args, payloads=None):
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            patch.dict(QUERY["main"].__globals__, worker_count=lambda reader, requested: requested),
        ):
            code = QUERY["main"](
                FakeReader(fixtures() if payloads is None else payloads), command, args
            )
        return code, out.getvalue(), err.getvalue()

    def test_complete_requires_training_and_final_evaluation_not_wandb(self):
        result = self.inspect()
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["evaluation"]["split"], "test")
        self.assertEqual(result["wandb"]["state"], "offline_pending")

    def test_training_done_does_not_certify_missing_evaluation(self):
        data = fixtures()
        del data[f"evaluations/{RUN_A}/validation-benchmark.json"]
        result = self.inspect(data)
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(result["evaluation"]["state"], "missing_result")
        del data[f"lifecycle/runs/{RUN_A}/validation.json"]
        self.assertEqual(self.inspect(data)["state"], "trained")

    def test_completion_artifacts_override_cutoff_race_but_retain_raw_state(self):
        data = fixtures()
        data[f"lifecycle/runs/{RUN_A}/training.json"].update(
            state="timed_out", training_completed=False
        )
        self.assertEqual(self.inspect(data)["state"], "complete")
        self.assertEqual(self.inspect(data)["training"]["lifecycle_state"], "timed_out")

    def test_legacy_completion_result_is_read_without_rewriting_any_contract(self):
        data = fixtures()
        del data[f"lifecycle/runs/{RUN_A}/training-completed.json"]
        data[f"lifecycle/runs/{RUN_A}/training.json"].update(
            state="timed_out", training_completed=False
        )
        data[f"savedModel/{RUN_A}/completion-result/training-result.json"] = {
            "kind": "training-completion-result",
            "run_id": RUN_A,
            "created_at": "2026-10-03T01:00:00Z",
        }
        self.assertEqual(self.inspect(data)["state"], "complete")

    def test_running_and_failed_training_are_not_completed(self):
        data = fixtures(complete=False)
        self.assertEqual(self.inspect(data)["state"], "training")
        data[f"lifecycle/runs/{RUN_A}/training.json"]["state"] = "failed"
        self.assertEqual(self.inspect(data)["state"], "incomplete")

    def test_new_evaluation_does_not_hide_behind_old_success(self):
        data = fixtures()
        data[f"lifecycle/runs/{RUN_A}/validation.json"].update(
            state="running", generated_at="2026-10-04T02:00:00Z"
        )
        self.assertEqual(self.inspect(data)["state"], "evaluating")

    def test_current_workflow_finalizing_state_remains_active(self):
        data = fixtures(complete=False)
        data[f"lifecycle/runs/{RUN_A}/training.json"]["state"] = "finalizing"
        self.assertEqual(self.inspect(data)["state"], "training")
        data[f"lifecycle/runs/{RUN_A}/validation.json"] = {
            "kind": "stage1-validation",
            "wandb_run_id": RUN_A,
            "state": "finalizing",
        }
        self.assertEqual(self.inspect(data)["state"], "evaluating")

    def test_historical_validation_is_not_called_holdout_test(self):
        data = fixtures()
        del data[f"evaluations/{RUN_A}/validation-benchmark.json"]["evaluation_split"]
        self.assertEqual(self.inspect(data)["evaluation"]["split"], "validation")

    def test_current_config_never_overrides_frozen_run_settings(self):
        data = fixtures()
        result = self.inspect(data)
        self.assertEqual(result["configuration"]["date_range"]["start_inclusive"], "2021-01-01")
        self.assertEqual(result["configuration"]["experiment"], "a-lora32")
        self.assertEqual(result["initialized_at"], "2026-10-02T13:05:00Z")
        self.assertNotEqual(
            result["initialized_at"], data[f"lifecycle/runs/{RUN_A}/selection.json"]["created_at"]
        )

    def test_old_selection_is_read_without_current_yaml_validation(self):
        data = fixtures()
        selection = data.pop(f"lifecycle/runs/{RUN_A}/selection.json")
        selection["stage"]["config_sha256"] = "obsolete-yaml-digest"
        data[f"lifecycle/selections/{selection['selection_id']}.json"] = selection
        self.assertEqual(self.inspect(data)["configuration"]["feature_mode"], "combined")

    def test_missing_historical_selection_uses_manifest_not_active_selection(self):
        data = fixtures()
        del data[f"lifecycle/runs/{RUN_A}/selection.json"]
        config = self.inspect(data)["configuration"]
        self.assertEqual(config["source"], "run_manifest")
        self.assertEqual(config["config_path"], "old.yaml")
        self.assertIsNone(config["feature_mode"])

    def test_mismatched_manifest_or_lifecycle_never_borrows_another_run(self):
        for key, field in (
            (f"savedModel/{RUN_A}/run-manifest.json", "run_id"),
            (f"lifecycle/runs/{RUN_A}/training.json", "wandb_run_id"),
            (f"evaluations/{RUN_A}/validation-benchmark.json", "run_id"),
            (f"lifecycle/runs/{RUN_A}/training-completed.json", "run_id"),
        ):
            with self.subTest(key=key):
                data = fixtures()
                data[key][field] = RUN_B
                with self.assertRaisesRegex(ValueError, "[Oo]wnership"):
                    self.inspect(data)

    def test_unknown_id_and_unsafe_id_fail(self):
        with self.assertRaisesRegex(ValueError, "Run not found"):
            self.inspect({}, RUN_A)
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            self.inspect({}, "../run")
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            self.invoke("status", ["--run-id", ""])

    def test_selection_without_training_is_not_started(self):
        data = fixtures()
        self.assertEqual(
            self.inspect(
                {
                    f"lifecycle/runs/{RUN_A}/selection.json": data[
                        f"lifecycle/runs/{RUN_A}/selection.json"
                    ]
                }
            )["state"],
            "not_started",
        )

    def test_list_groups_runs_and_pages_without_duplicates(self):
        data = {**fixtures(), **fixtures(RUN_B, "b-lora64", complete=False)}
        code, output, _ = self.invoke("list", ["--limit", "1", "--output", "json"], data)
        first = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(first["runs"][0]["run_id"], RUN_B)
        self.assertEqual(first["next_offset"], 1)
        _, output, _ = self.invoke(
            "list", ["--limit", "1", "--offset", "1", "--output", "json"], data
        )
        self.assertEqual(json.loads(output)["runs"][0]["run_id"], RUN_A)

    def test_filters_are_from_own_configuration_and_state(self):
        data = {**fixtures(), **fixtures(RUN_B, "b-lora64", complete=False)}
        _, output, _ = self.invoke(
            "list", ["--experiment", "a-lora32", "--state", "complete", "--output", "json"], data
        )
        self.assertEqual([row["run_id"] for row in json.loads(output)["runs"]], [RUN_A])

    def test_seed_display_and_filter_use_run_records_including_zero(self):
        data = {**fixtures(), **fixtures(RUN_B, "a-lora32", complete=False)}
        data[f"savedModel/{RUN_A}/run-manifest.json"]["training_resume_contract"]["training"] = {
            "seed": 42,
        }
        data[f"lifecycle/runs/{RUN_B}/selection.json"]["stage"]["training_seed"] = 0
        for seed, run in ((42, RUN_A), (0, RUN_B)):
            _, output, _ = self.invoke("list", ["--seed", str(seed), "--output", "json"], data)
            self.assertEqual([row["run_id"] for row in json.loads(output)["runs"]], [run])
            _, output, _ = self.invoke("status", [run], data)
            self.assertIn(f"seed={seed}", output)

    def test_conflicting_recorded_seed_reports_an_error(self):
        data = fixtures()
        data[f"lifecycle/runs/{RUN_A}/selection.json"]["stage"]["training_seed"] = 43
        data[f"savedModel/{RUN_A}/run-manifest.json"]["training_resume_contract"]["training"] = {
            "seed": 42,
        }
        code, output, _ = self.invoke("list", ["--output", "json"], data)
        self.assertEqual(code, 2)
        self.assertIn("seed does not match", json.loads(output)["errors"][0]["error"])

    def test_list_reports_errors_instead_of_silently_hiding_failed_reads(self):
        data = fixtures()
        data[f"savedModel/{RUN_A}/run-manifest.json"]["run_id"] = RUN_B
        code, output, _ = self.invoke("list", ["--experiment", "other", "--output", "json"], data)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)["errors"][0]["run_id"], RUN_A)

    def test_transport_errors_are_not_missing_runs_or_success(self):
        reader = FakeReader(fixtures())
        with patch.object(reader, "read", side_effect=ValueError("S3 returned 403")):
            result = QUERY["inventory"](
                reader, limit=20, offset=0, experiment=None, state=None, workers=2
            )
        self.assertEqual(result["runs"][0]["state"], "error")
        self.assertIn("403", result["errors"][0]["error"])

    def test_run_inspection_uses_only_canonical_read_keys(self):
        reader = FakeReader(fixtures())
        QUERY["inspect_run"](reader, RUN_A)
        self.assertTrue(all(isinstance(call, str) for call in reader.calls))
        self.assertFalse(any("active.json" in call for call in reader.calls))
        for call in reader.calls:
            self.assertTrue(
                call.startswith(
                    (f"savedModel/{RUN_A}/", f"evaluations/{RUN_A}/", f"lifecycle/runs/{RUN_A}/")
                )
            )

    def test_output_contains_settings_times_and_per_run_wandb(self):
        code, output, _ = self.invoke("status", [RUN_A])
        self.assertEqual(code, 0)
        for fragment in (
            RUN_A,
            "COMPLETE",
            "2026-10-02 21:05:00",
            "a-lora32",
            "stage=stage2",
            "data-profile=us_tw_eodhd",
            "dataset-revision=v1",
            "2021-01-01",
            "2026-06-01",
            "h-start=1",
            "feature-mode=combined",
            "universe=all",
            "etfs=SPY",
            "Training completed",
            "Final evaluation",
            "offline_pending",
            "pod-a",
        ):
            self.assertIn(fragment, output)
        _, utc, _ = self.invoke("status", [RUN_A, "--timezone", "UTC"])
        self.assertIn("2026-10-02 13:05:00 UTC", utc)

    def test_exit_codes_and_explicit_run_id_alias(self):
        self.assertEqual(self.invoke("status", ["--run-id", RUN_A])[0], 0)
        self.assertEqual(self.invoke("status", [RUN_A], fixtures(complete=False))[0], 1)
        self.assertEqual(self.invoke("list", [], {})[0], 0)

    def test_cli_rejects_unbounded_input(self):
        for options in (
            ["--limit", "0"],
            ["--limit", "201"],
            ["--workers", "0"],
            ["--workers", "9"],
            ["--offset", "-1"],
            ["--seed", "-1"],
            ["--seed", "4294967296"],
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.invoke("list", options)

    def test_discovery_includes_historical_saved_models_not_nested_logs(self):
        data = fixtures()
        data[f"savedModel/{RUN_B}/run-manifest.json"] = {"run_id": RUN_B}
        data["lifecycle/runs/not-a-run/pods/other/training.json"] = {}
        self.assertEqual(QUERY["discover"](FakeReader(data)), [RUN_B, RUN_A])

    def test_pagination_checks_continuation_and_preserves_all_pages(self):
        reader = FakeReader({})
        responses = iter(
            [{"IsTruncated": True, "NextContinuationToken": "page2"}, {"IsTruncated": False}]
        )
        with patch.object(reader, "call", side_effect=lambda *args: next(responses)) as calls:
            self.assertEqual(len(list(QUERY["pages"](reader, "savedModel/", delimiter=True))), 2)
            self.assertIn("page2", calls.call_args.args)
        with (
            patch.object(
                reader, "call", return_value={"IsTruncated": True, "NextContinuationToken": "loop"}
            ),
            self.assertRaisesRegex(ValueError, "continuation"),
        ):
            list(QUERY["pages"](reader, "savedModel/"))

    def test_worker_limits_account_for_memory_cpu_and_service(self):
        resource = {
            "detect_available_memory": lambda: type("Memory", (), {"available_bytes": 1024**3})(),
            "detect_visible_cpu_count": lambda: 16,
        }
        with (
            patch.object(QUERY["runpy"], "run_path", return_value=resource),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(QUERY["worker_count"](FakeReader({}), 8), 2)

    def test_no_argument_status_retains_global_capabilities(self):
        script = (ROOT / "scripts/show_runpod_status.sh").read_text()
        for existing in (
            "summarize_selected_dataset",
            "summarize_download_progress",
            "cpu-preparation.json",
            "code.json",
            "baseline.json",
            "summarize_wandb",
            'runpod_runs.py" status',
        ):
            self.assertIn(existing, script)

    def test_public_wrapper_routes_read_only_commands_and_preserves_no_arg_status(self):
        with tempfile.TemporaryDirectory(prefix="run-status-control-") as temporary:
            root = Path(temporary)
            (root / "scripts/lib").mkdir(parents=True)
            for name in ("runpod_workflow.sh", "lib/runpod_cli.sh"):
                shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
            (root / "scripts/lib/runpod_project_env.sh").write_text("runpod_load_s3_env() { :; }\n")
            (root / "scripts/runpod_runs.py").write_text(
                "import json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            (root / "scripts/show_runpod_status.sh").write_text("echo existing-global-status\n")
            for args, expected in (
                (["runs", "--experiment", "a-lora64"], ["list", "--experiment", "a-lora64"]),
                (["status", RUN_A], ["status", RUN_A]),
            ):
                result = subprocess.run(
                    ["bash", str(root / "scripts/runpod_workflow.sh"), *args],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                    env={**os.environ, "RUNPOD_NETWORK_VOLUME_ID": "fixture-volume"},
                )
                self.assertEqual(json.loads(result.stdout), expected)
            result = subprocess.run(
                ["bash", str(root / "scripts/runpod_workflow.sh"), "status"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            self.assertEqual(result.stdout.strip(), "existing-global-status")


if __name__ == "__main__":
    unittest.main()
