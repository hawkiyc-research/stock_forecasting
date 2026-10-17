"""Dependency-free controls for signal-prediction export, download and decomposition."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import runpy
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = runpy.run_path(str(ROOT / "scripts/analyze_signal_decomposition.py"))
DOWNLOAD = runpy.run_path(str(ROOT / "scripts/download_runpod_signal_predictions.py"))
HORIZONS = 14


def probe_harness():
    """Reuse the existing launcher harness without importing tests as a package."""
    spec = importlib.util.spec_from_file_location("probe_tmux", ROOT / "tests/test_probe_tmux.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProbeHarness


def write_npy(path: Path, descr, shape: tuple[int, ...], payload: bytes) -> None:
    header = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': {shape!r}, }}"
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + payload
    )


def floats(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


META_FIELDS = (("symbol", 48), ("date", 40), ("market", 24), ("asset_type", 24), ("provider", 32))


def metadata(rows) -> bytes:
    out = b""
    for row in rows:
        for (name, width), value in zip(META_FIELDS, row, strict=True):
            out += value.ljust(width, "\x00").encode("utf-32-le")
    return out


class Fixture:
    """Tiny shared evaluation arrays: 2 markets x 2 dates x 4 stocks per split."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.baselines = root / "baselines"
        self.predictions = root / "exports"
        self.rows = [
            (f"S{market}{stock}", f"2025-06-0{day}T00:00:00+00:00", market, "stock", "fixture")
            for market in ("TWSE", "US")
            for day in (2, 3)
            for stock in range(4)
        ]
        self.count = len(self.rows)
        alpha, absolute = [], []
        for index, (_symbol, date, market, _a, _p) in enumerate(self.rows):
            common = (0.02 if market == "US" else -0.03) * (1 if date.endswith("02") else -1)
            for h in range(HORIZONS):
                stock = 0.01 * ((index % 4) - 1.5)
                alpha.append(common + stock)
                absolute.append(2 * common + stock)
        self.alpha, self.absolute = alpha, absolute
        scales = [0.05] * HORIZONS
        for task_id in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]):
            base = self.baselines / task_id
            membership = {"ordered_symbol_dates_sha256": "f" * 64, "samples": self.count}
            base.mkdir(parents=True)
            (base / "complete.json").write_text(json.dumps({
                "robust_scales": scales,
                "validation_membership": membership,
                "evaluation_membership": membership,
            }))
            for split in ("validation", "test"):
                write_npy(
                    base / "inputs" / split / "metadata.npy",
                    [(name, f"<U{width}") for name, width in META_FIELDS],
                    (self.count,),
                    metadata(self.rows),
                )
                write_npy(base / "inputs" / split / "targets.npy", "<f4",
                          (self.count, HORIZONS), floats(alpha))
                write_npy(base / "inputs" / split / "absolute_targets.npy", "<f4",
                          (self.count, HORIZONS), floats(absolute))
                for model, scale in (("gru-42", 0.5), ("gbdt-42", 0.25)):
                    write_npy(base / "jobs" / model / split / "predictions.npy", "<f4",
                              (self.count, HORIZONS), floats([scale * v for v in alpha]))
            if task_id == ANALYSIS["DIRECTION_ID"]:
                audit = {"counts": {m: {str(h): [3, 2, 3] for h in range(1, 15)}
                                    for m in ("TWSE", "US", "overall")}}
                (base / "inputs/train").mkdir(parents=True, exist_ok=True)
                (base / "inputs/train/class-label-audit.json").write_text(json.dumps(audit))
                for model in ("gru-42", "gbdt-42"):
                    for split in ("validation", "test"):
                        write_npy(base / "jobs" / model / split / "predictions.npy", "<f4",
                                  (self.count, HORIZONS, 3), floats([0.0] * self.count * 42))

    def export(self, run_id: str, task: str, values, digest="f" * 64,
               checkpoint="checkpoint-003680") -> None:
        export = "export-20261017T000000Z-0a1b2c3d"
        directory = self.predictions / run_id / checkpoint / export
        tail = (HORIZONS,) if task == "alpha_point" else (HORIZONS, 3)
        splits = {}
        for split in ("validation", "test"):
            write_npy(directory / split / "predictions.npy", "<f4", (self.count, *tail),
                      floats(values))
            splits[split] = {"sample_membership": {"ordered_symbol_dates_sha256": digest}}
        (directory / "status.json").write_text(json.dumps({"state": "complete"}))
        (directory / "export.json").write_text(json.dumps({
            "kind": "signal-prediction-export", "run_id": run_id, "prediction_task": task,
            "created_at": "2026-10-17T00:00:00+00:00", "splits": splits,
        }))


class DecompositionAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def run_analysis(self, *arguments: str) -> dict:
        output = self.fixture.root / "result.json"
        with contextlib.redirect_stdout(io.StringIO()):
            ANALYSIS["main"]([
                "--baselines", str(self.fixture.baselines), "--output", str(output),
                "--workers", "1", *arguments,
            ])
        payload = json.loads(output.read_text())
        return {(item["task"], item["split"]): item for item in payload["results"]}

    def test_oracle_date_mean_removes_the_common_component_exactly(self) -> None:
        alpha = self.run_analysis()[("alpha_point", "validation")]["metrics"]
        zero = alpha["zero_return"]["aggregate"]
        oracle = alpha["oracle_date_mean"]["aggregate"]
        # Within-group alpha deviations are +/-0.005 and +/-0.015 over scale 0.05.
        expected = sum(0.5 * (d / 0.05) ** 2 for d in (0.015, 0.005, 0.005, 0.015)) / 4
        self.assertAlmostEqual(oracle, expected, places=6)
        self.assertLess(oracle, zero)
        self.assertAlmostEqual(alpha["gru-42_date_only"]["aggregate"],
                               alpha["gru-42_date_only"]["aggregate"])

    def test_cross_sectional_prior_normalization_never_uses_labels(self) -> None:
        direction = self.run_analysis()[("absolute_direction", "test")]["metrics"]
        # Uniform logits whose date average is uniform become exactly the train prior.
        self.assertAlmostEqual(
            direction["gru-42_train_prior_adjusted"]["aggregate"],
            direction["train_prior"]["aggregate"],
            places=9,
        )
        self.assertLess(
            direction["oracle_date_prior"]["aggregate"], direction["train_prior"]["aggregate"]
        )

    def test_exported_models_and_ensembles_join_the_same_decomposition(self) -> None:
        alpha_values = [0.5 * v for v in self.fixture.alpha]
        self.fixture.export("run-a-alpha", "alpha_point", alpha_values)
        self.fixture.export("run-b-alpha", "alpha_point", alpha_values)
        results = self.run_analysis(
            "--predictions", str(self.fixture.predictions),
            "--model", "A=run-a-alpha", "--model", "B=run-b-alpha", "--ensemble", "AB=A,B",
        )
        metrics = results[("alpha_point", "test")]["metrics"]
        for name in ("A", "B", "AB"):
            self.assertAlmostEqual(metrics[name]["aggregate"], metrics["gru-42"]["aggregate"])
        self.assertIn("A_within_plus_split_level", metrics)

    def test_checkpoint_selectors_keep_two_checkpoints_of_one_run_apart(self) -> None:
        self.fixture.export("run-a-alpha", "alpha_point", [0.5 * v for v in self.fixture.alpha])
        self.fixture.export("run-a-alpha", "alpha_point", [0.25 * v for v in self.fixture.alpha],
                            checkpoint="checkpoint-022079")
        predictions = str(self.fixture.predictions)
        with self.assertRaisesRegex(ValueError, "checkpoint-003680, checkpoint-022079"):
            self.run_analysis("--predictions", predictions, "--model", "A=run-a-alpha")
        for selector in ("run-a-alpha/checkpoint-12", "run-a-alpha/../x", "run--a"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                ANALYSIS["latest_export"](self.fixture.predictions, selector)
        results = self.run_analysis(
            "--predictions", predictions,
            "--model", "Best=run-a-alpha/checkpoint-003680",
            "--model", "Late=run-a-alpha/checkpoint-022079",
        )
        metrics = results[("alpha_point", "validation")]["metrics"]
        self.assertAlmostEqual(metrics["Best"]["aggregate"], metrics["gru-42"]["aggregate"])
        self.assertAlmostEqual(metrics["Late"]["aggregate"], metrics["gbdt-42"]["aggregate"])
        models = ANALYSIS["resolve_models"](
            self.fixture.baselines, self.fixture.predictions,
            ["Late=run-a-alpha/checkpoint-022079"],
        )
        self.assertEqual(models["Late"]["run_id"], "run-a-alpha")
        self.assertEqual(models["Late"]["checkpoint"], "checkpoint-022079")

    def test_membership_mismatch_or_bad_ensemble_is_rejected(self) -> None:
        self.fixture.export("run-a-alpha", "alpha_point", [0.0] * self.fixture.count * HORIZONS,
                            digest="0" * 64)
        with self.assertRaises(ValueError):
            self.run_analysis("--predictions", str(self.fixture.predictions),
                              "--model", "A=run-a-alpha")
        with self.assertRaises(ValueError):
            ANALYSIS["resolve_ensembles"]({"A": {"task": "alpha"}}, ["E=A"])
        with self.assertRaises(ValueError):
            ANALYSIS["resolve_ensembles"](
                {"A": {"task": "alpha"}, "B": {"task": "direction"}}, ["E=A,B"]
            )

    def test_explicit_worker_count_is_bounded_and_respected(self) -> None:
        self.assertEqual(ANALYSIS["worker_count"](1, 4, 1), 1)
        self.assertEqual(ANALYSIS["worker_count"](8, 4, 1), 4)
        with self.assertRaises(ValueError):
            ANALYSIS["worker_count"](0, 4, 1)


class FakeReader:
    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents
        self.wrapper, self.volume_id = Path("/nonexistent"), "volume"

    def read_json(self, key, limit=None):
        payload = self.documents[key]
        raw = json.dumps(payload).encode()
        return payload, hashlib.sha256(raw).hexdigest()

    def status_keys(self, prefix):
        yield from (key for key in self.documents if key.startswith(prefix)
                    and key.endswith("/status.json"))


class PredictionDownloadTests(unittest.TestCase):
    def documents(self, run_id: str, export: str, created: str, state="complete",
                  name="checkpoint-003680"):
        prefix = f"diagnostics/signal-predictions/{run_id}/{name}/{export}/"
        checkpoint = f"/runpod-volume/savedModel/{run_id}/{name}"
        return {
            prefix + "status.json": {"state": state, "checkpoint": checkpoint},
            prefix + "export.json": {
                "schema_version": "1.0", "kind": "signal-prediction-export", "run_id": run_id,
                "checkpoint": {"path": checkpoint}, "created_at": created,
                "splits": {"validation": {"predictions_sha256": "a" * 64},
                           "test": {"predictions_sha256": "b" * 64}},
            },
        }

    def test_selects_the_newest_completed_export_and_skips_running_ones(self) -> None:
        documents = {
            **self.documents("run-x-alpha", "export-20261017T000000Z-00000001",
                             "2026-10-17T00:00:00+00:00"),
            **self.documents("run-x-alpha", "export-20261018T000000Z-00000002",
                             "2026-10-18T00:00:00+00:00"),
            **self.documents("run-x-alpha", "export-20261019T000000Z-00000003",
                             "2026-10-19T00:00:00+00:00", state="running"),
        }
        result = DOWNLOAD["select_export"](FakeReader(documents), "run-x-alpha")
        self.assertEqual(result["relative"].name, "export-20261018T000000Z-00000002")
        self.assertEqual(result["splits"], {"validation": "a" * 64, "test": "b" * 64})

    def test_named_checkpoints_are_selected_and_bare_ids_must_be_unambiguous(self) -> None:
        documents = {
            **self.documents("run-x-alpha", "export-20261017T000000Z-00000001",
                             "2026-10-17T00:00:00+00:00"),
            **self.documents("run-x-alpha", "export-20261018T000000Z-00000002",
                             "2026-10-18T00:00:00+00:00", name="checkpoint-022079"),
        }
        reader = FakeReader(documents)
        with self.assertRaisesRegex(ValueError, "checkpoint-003680, checkpoint-022079"):
            DOWNLOAD["select_export"](reader, "run-x-alpha")
        for name, export in (("checkpoint-003680", "export-20261017T000000Z-00000001"),
                             ("checkpoint-022079", "export-20261018T000000Z-00000002")):
            result = DOWNLOAD["select_export"](reader, f"run-x-alpha/{name}")
            self.assertEqual(result["relative"], Path("run-x-alpha", name, export))
        with self.assertRaises(FileNotFoundError):
            DOWNLOAD["select_export"](reader, "run-x-alpha/checkpoint-011040")
        self.assertEqual(DOWNLOAD["safe_selector"]("run-x-alpha"), ("run-x-alpha", None))
        for value in ("run-x-alpha/", "run-x-alpha/checkpoint-1", "run-x-alpha/a/b", "/x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DOWNLOAD["safe_selector"](value)

    def test_baseline_inputs_match_the_analysis_and_verify_digests(self) -> None:
        files = DOWNLOAD["baseline_files"]()
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(len(files), 15)
        for baseline in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]):
            self.assertTrue(any(name.startswith(baseline + "/") for name in files))
        payload = b"fixture-array"
        artifacts = {name.partition("/")[2]: {"sha256": hashlib.sha256(payload).hexdigest()}
                     for name in files}
        reader = FakeReader({
            f"baselines/{baseline}/complete.json": {"state": "complete", "artifacts": artifacts}
            for baseline in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"])
        })
        calls = []

        def fake_download(_reader, key, destination, _timeout):
            calls.append(key)
            destination.write_bytes(payload if "corrupt" not in str(destination) else b"x")

        download = DOWNLOAD["download_baseline_arrays"]
        with tempfile.TemporaryDirectory() as temporary, \
                DOWNLOAD["ThreadPoolExecutor"](max_workers=2) as executor:
            project = Path(temporary).resolve()
            first = download(reader, project, executor, 2, 60, fetch=fake_download)
            self.assertEqual(first[1:], (15, 0))
            second = download(reader, project, executor, 2, 60, fetch=fake_download)
            self.assertEqual(second[1:], (0, 15))
            self.assertEqual(len(calls), 15)
            (first[0] / files[1]).write_bytes(b"tampered")
            third = download(reader, project, executor, 2, 60, fetch=fake_download)
            self.assertEqual(third[1:], (1, 14))
            artifacts[files[2].partition("/")[2]]["sha256"] = "0" * 64
            (first[0] / files[2]).write_bytes(b"stale")
            with self.assertRaises(ValueError):
                download(reader, project, executor, 2, 60, fetch=fake_download)

    def test_identity_mismatch_and_unsafe_ids_are_rejected(self) -> None:
        documents = self.documents("run-x-alpha", "export-20261017T000000Z-00000001",
                                   "2026-10-17T00:00:00+00:00")
        export_key = next(key for key in documents if key.endswith("export.json"))
        documents[export_key]["run_id"] = "run-other"
        with self.assertRaises(ValueError):
            DOWNLOAD["select_export"](FakeReader(documents), "run-x-alpha")
        for value in ("../x", "--force", "a/b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DOWNLOAD["safe_run_id"](value)


class ExportWorkflowTests(unittest.TestCase):
    def harness(self):
        harness = probe_harness()()
        self.addCleanup(harness.temporary.cleanup)
        # Here-documents need a writable temporary directory inside the fixture.
        harness.environment["TMPDIR"] = str(harness.root)
        harness.write_script(
            harness.scripts / "runpod_export_signal_predictions.sh",
            "printf '%s\\0' \"$@\" > \"${HARNESS_ROOT}/worker-args\"\n",
        )
        return harness

    def test_launcher_registers_a_gpu_probe_diagnostic_session(self) -> None:
        harness = self.harness()
        arguments = ("--checkpoint", "run-x-alpha", "--splits", "validation")
        result = harness.command("runpod_tmux_launch.sh", "export-signal-predictions", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        runners = harness.runners()
        self.assertEqual(len(runners), 1)
        self.assertIn("stock-forecasting-export-predictions", str(runners[0]))
        runner = runners[0].read_text(encoding="utf-8")
        self.assertIn("runpod_export_signal_predictions.sh", runner)
        self.assertIn("publish_probe_state", runner)
        self.assertIn("RUNPOD_SCALE_PROBE_TMUX_WORKER=1", runner)
        self.assertNotIn("runpod_probe_scales.sh", runner)

    def test_launcher_help_needs_no_pod(self) -> None:
        harness = self.harness()
        (harness.scripts / "runpod_export_signal_predictions.sh").write_text(
            (ROOT / "scripts/runpod_export_signal_predictions.sh").read_text()
        )
        harness.environment.pop("RUNPOD_POD_ID")
        harness.environment.pop("RUNPOD_SSH_ENV_IMPORTED")
        result = harness.command("runpod_tmux_launch.sh", "export-signal-predictions", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("diagnostics/signal-predictions", result.stdout)
        self.assertIn("--confirm-test", result.stdout)
        self.assertFalse((harness.root / "tmux-calls").exists())

    def test_job_script_requires_a_pod_and_one_safe_run_id(self) -> None:
        harness = self.harness()
        (harness.scripts / "runpod_export_signal_predictions.sh").write_text(
            (ROOT / "scripts/runpod_export_signal_predictions.sh").read_text()
        )
        harness.environment["RUNPOD_TEST_MODE"] = "1"
        too_many = tuple(item for i in range(17) for item in ("--checkpoint", f"run-{i}"))
        for arguments, message in (
            ((), "1-16 times"),
            (("--checkpoint", "/runpod-volume/savedModel/x"), "safe RUN_ID"),
            (("--checkpoint", "run-a", "--checkpoint=../b"), "safe RUN_ID"),
            (("--checkpoint", "run-a/checkpoint-12"), "safe RUN_ID"),
            (("--checkpoint", "run-a/checkpoint-022079/x"), "safe RUN_ID"),
            (("--checkpoint=run-a/../checkpoint-022079",), "safe RUN_ID"),
            (too_many, "1-16 times"),
        ):
            with self.subTest(arguments=arguments):
                result = harness.command("runpod_export_signal_predictions.sh", *arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
        harness.environment.pop("RUNPOD_POD_ID")
        result = harness.command("runpod_export_signal_predictions.sh", "--checkpoint", "a")
        self.assertEqual(result.returncode, 2)
        self.assertIn("inside an existing CUDA RunPod Pod", result.stderr)
        self.assertEqual(harness.runners(), [])

    def test_job_script_forwards_named_leaderboard_checkpoints(self) -> None:
        harness = self.harness()
        (harness.scripts / "runpod_export_signal_predictions.sh").write_text(
            (ROOT / "scripts/runpod_export_signal_predictions.sh").read_text()
        )
        result = harness.command(
            "runpod_export_signal_predictions.sh",
            "--checkpoint", "run-a-alpha",
            "--checkpoint=run-a-alpha/checkpoint-022079",
            "--splits", "validation",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runners = harness.runners()
        self.assertEqual(len(runners), 1)
        self.assertIn("run-a-alpha/checkpoint-022079", runners[0].read_text(encoding="utf-8"))

    def test_export_module_never_writes_lifecycle_or_training_state(self) -> None:
        source = (ROOT / "src/stock_forecasting/cli/export_signal_predictions.py").read_text()
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn("load_checkpoint", imported)
        self.assertTrue({"save_checkpoint", "write_state", "AdamW"}.isdisjoint(imported))
        self.assertIn("allow_historical_inference=True", source)
        self.assertIn("--confirm-test", source)
        self.assertEqual(source.count("diagnostics\" / \"signal-predictions\""), 1)

    def test_export_is_not_part_of_any_numerical_identity(self) -> None:
        contract = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        listed = set()
        for node in contract.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
                for item in node.value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        listed.add(item.value)
        self.assertNotIn("cli/export_signal_predictions.py", listed)
        baseline = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        self.assertFalse(any("export_signal" in path for path in baseline["BASELINE_SOURCES"]))


if __name__ == "__main__":
    unittest.main()
