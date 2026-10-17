"""Standard-library numerical and integration tests; no ML environment or cloud calls."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_forecasting.signal_diagnostic_io import (  # noqa: E402
    NpyReader,
    inspect_membership,
    sha256,
    temporal_guard,
)
from stock_forecasting.signal_diagnostic_math import (  # noqa: E402
    ProbabilityMoments,
    distribution,
    fit_calibrator,
    objective_gradient,
    probabilities,
)


def module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "analyze_signal_calibration", ROOT / "scripts/analyze_signal_calibration.py"
    )
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def write_npy(path, descr, shape, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = repr({"descr": descr, "fortran_order": False, "shape": shape})
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + payload
    )


def float_file(path, shape, values):
    write_npy(path, "<f4", shape, struct.pack("<" + "f" * len(values), *values))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.base = root / "baseline"
        self.rows = []
        day = date(2025, 6, 2)
        days = []
        while len(days) < 40:
            if day.weekday() < 5:
                days.append(day.isoformat())
            day += timedelta(days=1)
        for day in days:
            for market in ("US", "TWSE", "TPEX"):
                self.rows.extend(
                    (f"{market}{i}", day + "T00:00:00+00:00", market) for i in range(3)
                )
        self.count = len(self.rows)
        self.meta = self.base / "inputs/validation/metadata.npy"
        fields = [("symbol", "<U48"), ("date", "<U40"), ("market", "<U24")]
        payload = b"".join(
            value.ljust(width, "\0").encode("utf-32-le")
            for row in self.rows
            for value, width in zip(row, (48, 40, 24), strict=True)
        )
        write_npy(self.meta, fields, (self.count,), payload)
        self.targets = self.base / "inputs/validation/absolute_targets.npy"
        self.labels = [(-0.02, 0.0, 0.02)[i % 3] for i in range(self.count) for _ in range(14)]
        float_file(self.targets, (self.count, 14), self.labels)
        with NpyReader(self.meta) as reader:
            digest, _ = inspect_membership(reader)
        self.membership = {
            "ordered_symbol_dates_sha256": digest,
            "samples": self.count,
            "unique_dates": 40,
            "cutoff_start": self.rows[0][1],
            "cutoff_end": self.rows[-1][1],
        }
        audit = {
            "source": "full_train",
            "samples": 240,
            "class_order": ["down", "neutral", "up"],
            "neutral_return_threshold": 0.01,
            "counts": {
                market: {str(h): [30, 20, 30] for h in range(1, 15)}
                for market in ("US", "TWSE", "TPEX")
            },
        }
        self.manifest = {
            "state": "complete",
            "validation_membership": self.membership,
            "class_label_audit": audit,
            "artifacts": {},
        }
        for name in ("gru-42", "gbdt-42"):
            path = self.base / f"jobs/{name}/validation/predictions.npy"
            float_file(
                path,
                (self.count, 14, 3),
                [v for _ in range(self.count * 14) for v in (1.5, 0.0, -0.3)],
            )
        for path in self.base.rglob("*.npy"):
            self.manifest["artifacts"][str(path.relative_to(self.base))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        (self.base / "complete.json").write_text(json.dumps(self.manifest))
        self.export = root / "export"
        path = self.export / "validation/predictions.npy"
        float_file(
            path,
            (self.count, 14, 3),
            [v for _ in range(self.count * 14) for v in (3.0, -1.0, -1.0)],
        )
        self.report = {
            "kind": "signal-prediction-export",
            "prediction_task": "absolute_direction",
            "horizons": list(range(1, 15)),
            "checkpoint": {
                "selection": "validation_best",
                "path": "/runpod-volume/savedModel/run-fixture/checkpoint-000001",
            },
            "splits": {
                "validation": {
                    "sample_membership": self.membership,
                    "predictions_sha256": sha256(path),
                }
            },
        }
        (self.export / "export.json").write_text(json.dumps(self.report))
        (self.export / "status.json").write_text(
            json.dumps({"state": "complete", "checkpoint": self.report["checkpoint"]["path"]})
        )


class MathTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        points = [((1.0, -0.4, 0.2), 0), ((-0.2, 0.4, 1.0), 1), ((0.2, 1.0, -0.5), 2)]
        theta = [1.2, 0.1, -0.2, 0.1]
        _, gradient = objective_gradient(points, theta, 0.1, True)
        for k in range(4):
            plus, minus = theta.copy(), theta.copy()
            plus[k] += 1e-6
            minus[k] -= 1e-6
            finite = (
                objective_gradient(points, plus, 0.1, True)[0]
                - objective_gradient(points, minus, 0.1, True)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[k], finite, places=6)

    def test_temperature_preserves_argmax_and_bias_can_change_it(self):
        z = [2.0, 1.0, 0.0]
        for beta in (0.05, 0.5, 1.0, 20.0):
            self.assertEqual(max(range(3), key=lambda k: probabilities(z, beta)[k]), 0)
        self.assertEqual(max(range(3), key=lambda k: probabilities(z, 1.0, (-3.0, 2.0, 1.0))[k]), 1)

    def test_fitted_shrunk_bias_is_zero_sum_and_improves_fit_objective(self):
        points = [((4.0, 0.0, 0.0), k) for _ in range(10) for k in (0, 1, 2)]
        fitted = fit_calibrator(points, with_bias=True, max_iterations=60)
        self.assertLess(fitted["final_fit_objective"], fitted["initial_fit_objective"])
        self.assertAlmostEqual(sum(fitted["bias"]), 0.0, places=10)
        self.assertGreaterEqual(fitted["inverse_temperature"], 0.05)
        self.assertLessEqual(fitted["inverse_temperature"], 20.0)

    def test_neutral_can_be_calibrated_without_any_neutral_argmax(self):
        moments = ProbabilityMoments()
        for k, count in enumerate((9, 2, 9)):
            for _ in range(count):
                moments.add([0.45, 0.1, 0.45], k)
        result = moments.result()
        self.assertEqual(result["predicted_class_share"][1], 0.0)
        self.assertAlmostEqual(result["classwise_ece"][1], 0.0)

    def test_extreme_wrong_logits_keep_exact_log_loss(self):
        p, logp = distribution([1000.0, 0.0, -1000.0])
        moments = ProbabilityMoments()
        moments.add(p, 2, logp[2])
        self.assertAlmostEqual(moments.result()["cross_entropy"], 2000.0)

    def test_reliability_and_auc_detect_separated_classes(self):
        moments = ProbabilityMoments()
        moments.add([0.9, 0.05, 0.05], 0)
        moments.add([0.05, 0.9, 0.05], 1)
        moments.add([0.05, 0.05, 0.9], 2)
        self.assertEqual(moments.result()["classwise_auc_15bin"], [1.0, 1.0, 1.0])

    def test_nonfinite_inputs_and_invalid_bounds_fail(self):
        for z in ([math.nan, 0.0, 0.0], [math.inf, 1.0, 0.0]):
            with self.assertRaises(ValueError):
                fit_calibrator([(z, 0)])
        with self.assertRaises(ValueError):
            fit_calibrator([])
        with self.assertRaises(ValueError):
            fit_calibrator([((0.0, 0.0, 0.0), 0)], penalty=0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))
        self.analysis = module()

    def test_guard_refuses_overlapping_or_short_market_gap(self):
        with NpyReader(self.fixture.meta) as reader:
            _, calendars = inspect_membership(reader)
        with self.assertRaises(ValueError):
            temporal_guard(calendars, "2025-06-13", "2025-06-20")
        self.assertEqual(
            set(temporal_guard(calendars, "2025-06-13", "2025-07-09")), {"US", "TWSE", "TPEX"}
        )

    def test_worker_exception_propagates_and_processes_are_reaped(self):
        with self.assertRaises(FileNotFoundError):
            self.analysis.run_jobs([{"metadata": str(Path(self.temp.name) / "missing.npy")}], 2, 5)
        self.assertFalse(self.analysis.multiprocessing.active_children())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "Requires POSIX for a deliberately blocked reader")
    def test_worker_timeout_terminates_blocked_readers(self):
        path = Path(self.temp.name) / "blocked.npy"
        os.mkfifo(path)
        with self.assertRaises(self.analysis.multiprocessing.TimeoutError):
            self.analysis.run_jobs([{"metadata": str(path)}], 2, 1)
        self.assertFalse(self.analysis.multiprocessing.active_children())

    def test_npy_refuses_object_dtype_and_truncated_payload(self):
        path = Path(self.temp.name) / "bad.npy"
        write_npy(path, "|O", (1,), b"payload")
        with self.assertRaises(ValueError):
            NpyReader(path, (14, 3))
        float_file(path, (2, 14), [0.0] * 28)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            NpyReader(path, (14,))

    def test_fit_sampling_is_reproducible_bounded_and_early_only(self):
        with NpyReader(self.fixture.meta) as reader:
            a, _ = self.analysis.fit_indices(reader, "2025-06-13", 16, 59)
            b, _ = self.analysis.fit_indices(reader, "2025-06-13", 16, 59)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 16)
            self.assertTrue(all(reader.text(i, "date")[:10] <= "2025-06-13" for i in a))

    def test_later_validation_labels_cannot_change_fitted_parameters(self):
        with NpyReader(self.fixture.meta) as reader:
            indices, _ = self.analysis.fit_indices(reader, "2025-06-13", 16, 59)
        job = {
            "name": "fixture",
            "metadata": str(self.fixture.meta),
            "targets": str(self.fixture.targets),
            "predictions": str(self.fixture.export / "validation/predictions.npy"),
            "fit_indices": indices,
            "priors": self.analysis.validate_prior(self.fixture.manifest["class_label_audit"]),
            "score_start": "2025-07-09",
            "max_iterations": 4,
            "bias_penalty": 0.1,
        }
        before = self.analysis.calibrate_job(job)
        labels = self.fixture.labels.copy()
        for i, row in enumerate(self.fixture.rows):
            if row[1][:10] >= "2025-07-09":
                labels[i * 14 : (i + 1) * 14] = [0.02] * 14
        float_file(self.fixture.targets, (self.fixture.count, 14), labels)
        after = self.analysis.calibrate_job(job)
        self.assertEqual(before["parameters"], after["parameters"])
        self.assertNotEqual(
            before["methods"]["raw"]["cross_entropy"], after["methods"]["raw"]["cross_entropy"]
        )

    def test_test_or_named_checkpoint_exports_are_rejected(self):
        self.fixture.report["splits"] = {"test": self.fixture.report["splits"]["validation"]}
        (self.fixture.export / "export.json").write_text(json.dumps(self.fixture.report))
        with self.assertRaises(ValueError):
            self.analysis.exported_model(
                "model=" + str(self.fixture.export), self.fixture.membership
            )

    def test_cli_spawn_compares_every_model_and_freezes_protocol(self):
        output = Path(self.temp.name) / "study"
        command = [
            sys.executable,
            "-I",
            str(ROOT / "scripts/analyze_signal_calibration.py"),
            "--baseline-root",
            str(self.fixture.base),
            "--model",
            "candidate=" + str(self.fixture.export),
            "--calibration-end",
            "2025-06-13",
            "--score-start",
            "2025-07-09",
            "--fit-rows",
            "16",
            "--max-iterations",
            "4",
            "--workers",
            "2",
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads((output / "protocol.json").read_text())
        report = json.loads((output / "comparison.json").read_text())
        self.assertFalse(protocol["test_opened"])
        self.assertIsNone(report["automatic_winner"])
        self.assertEqual({r["name"] for r in report["results"]}, {"gru-42", "gbdt-42", "candidate"})
        fixed = [r["methods"]["fixed_train_prior"]["cross_entropy"] for r in report["results"]]
        self.assertEqual(fixed, [fixed[0]] * 3)
        self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)

    def test_checksum_corruption_fails_before_any_output(self):
        self.fixture.targets.write_bytes(self.fixture.targets.read_bytes() + b"X")
        with self.assertRaises(ValueError):
            self.analysis.manifest_file(
                self.fixture.base, self.fixture.manifest, "inputs/validation/absolute_targets.npy"
            )


class ExportControlTests(unittest.TestCase):
    def test_export_launcher_uses_its_own_diagnostic_session(self):
        spec = importlib.util.spec_from_file_location(
            "export_probe_harness", ROOT / "tests/test_probe_tmux.py"
        )
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        harness = loaded.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        harness.write_script(harness.scripts / "runpod_export_signal_predictions.sh", "exit 0\n")
        result = harness.command(
            "runpod_tmux_launch.sh",
            "export-signal-predictions",
            "--checkpoint",
            "run-fixture",
            "--splits",
            "train",
            "validation",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("stock-forecasting-export-predictions", harness.read("tmux-calls"))
        self.assertFalse(
            (harness.volume / "savedModel/run-fixture/training-completed.json").exists()
        )

    def test_downloader_rejects_unsafe_artifact_names(self):
        download = runpy.run_path(str(ROOT / "scripts/download_runpod_signal_predictions.py"))

        class Reader:
            def read_json(self, key, *_args):
                checkpoint = "/runpod-volume/savedModel/run-fixture/checkpoint-000001"
                if key.endswith("status.json"):
                    return {"state": "complete", "checkpoint": checkpoint}, "a" * 64
                return {
                    "schema_version": "1.0",
                    "kind": "signal-prediction-export",
                    "run_id": "run-fixture",
                    "checkpoint": {"path": checkpoint},
                    "created_at": "2026-10-18T00:00:00+00:00",
                    "splits": {
                        "validation": {
                            "predictions_sha256": "b" * 64,
                            "artifacts": {"../escape": {"sha256": "c" * 64}},
                        }
                    },
                }, "d" * 64

        key = (
            "diagnostics/signal-predictions/run-fixture/checkpoint-000001/"
            "export-20261018T000000Z-abcdef01/status.json"
        )
        with self.assertRaises(ValueError):
            download["candidate"](Reader(), key)

    def test_failed_download_never_publishes_complete_status(self):
        download = runpy.run_path(str(ROOT / "scripts/download_runpod_signal_predictions.py"))
        good = b"expected prediction bytes"
        digest = hashlib.sha256(good).hexdigest()
        result = {
            "relative": Path("run-fixture/checkpoint-000001/export-20261018T000000Z-abcdef01"),
            "prefix": "fixture/",
            "splits": {"validation": digest},
            "hashes": {},
            "artifacts": {
                "validation": {"predictions.npy": {"sha256": digest, "bytes": len(good)}}
            },
        }
        original = download["download_export"].__globals__
        previous = original["download_file"]

        def fake(_reader, key, path, _timeout):
            path.write_bytes(b"tampered" if key.endswith("predictions.npy") else b"{}")

        original["download_file"] = fake
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                with self.assertRaises(ValueError):
                    download["download_export"](None, result, Path(directory), pool, 2, 10)
                destination = Path(directory) / download["LOCAL_ROOT"] / result["relative"]
                self.assertFalse((destination / "status.json").exists())
        finally:
            original["download_file"] = previous

    def test_shell_help_never_imports_models_or_runs_cloud_commands(self):
        for arguments in [
            ("runpod_export_signal_predictions.sh", "--help"),
            ("runpod_tmux_launch.sh", "export-signal-predictions", "--help"),
            ("download_runpod_signal_predictions.sh", "--help"),
        ]:
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / arguments[0]), *arguments[1:]],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("torch", sys.modules)

    def test_sampler_and_training_defaults_are_unchanged(self):
        for relative in (
            "src/stock_forecasting/training.py",
            "src/stock_forecasting/date_market_sampler.py",
            "src/stock_forecasting/data/dataset.py",
            "src/stock_forecasting/models/signal.py",
        ):
            old = subprocess.check_output(["git", "show", "9380188:" + relative], cwd=ROOT)
            self.assertEqual(old, (ROOT / relative).read_bytes())

    def test_export_defaults_do_not_open_test_and_bound_samples(self):
        tree = ast.parse(
            (ROOT / "src/stock_forecasting/cli/export_signal_predictions.py").read_text()
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
        )
        ns = {
            "argparse": __import__("argparse"),
            "__doc__": "fixture",
            "SPLITS": ("train", "validation", "test"),
            "MAX_CHECKPOINTS": 16,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<export parser>", "exec"), ns)
        args = ns["build_parser"]().parse_args(["--checkpoint", "run-fixture"])
        self.assertEqual(args.splits, ["validation"])
        self.assertFalse(args.confirm_test)
        self.assertEqual(args.train_samples, 16384)
        self.assertLessEqual(args.audit_samples, 32)


if __name__ == "__main__":
    unittest.main()
