"""Dependency-free controls for signal-prediction export, download and decomposition."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import random
import runpy
import statistics
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = runpy.run_path(str(ROOT / "scripts/analyze_signal_decomposition.py"))
DOWNLOAD = runpy.run_path(str(ROOT / "scripts/download_runpod_signal_predictions.py"))
HORIZONS = 14

# The storage fields the data workflow hashes into a dataset ID (as in baseline records).
EVALUATION_REQUEST = {
    "profile": "us_tw_eodhd",
    "revision": "v1",
    "schema_version": 3,
    "selected_datasets": ["twse_official", "eodhd_us", "tpex_official"],
    "date_range": {"start_inclusive": "2016-01-01", "end_exclusive": "2026-06-01"},
    "universe": {
        "include_delisted_us": True,
        "mode": "all",
        "symbol_limit": None,
        "us_etfs": [],
        "us_stocks": [],
    },
    "preparation": {
        "alpha_horizons": list(range(1, 15)),
        "bar_store_schema_version": "1.0",
        "benchmark_mapping_sha256": "4" * 64,
        "diagnostic_horizons": [1, 20],
        "effective_embargo_bars": 14,
        "effective_sample_stride": 1,
        "embargo_bars": 5,
        "entry_day_counts_as_holding_day_one": True,
        "entry_timing": "regular_session_open_t_plus_1",
        "exit_timing": "regular_session_close_t_plus_h",
        "fixed_split": {
            "test_end": "2026-06-01",
            "train_end": "2025-06-01",
            "validation_end": "2025-12-01",
        },
        "flat_volatility_multiplier": 0.25,
        "h_start": 1,
        "label_kind": "benchmark_relative_adjusted_log_return",
        "labels_materialized": False,
        "max_abs_log_return": 0.5,
        "max_horizon": 14,
        "processed_schema_version": "4.0",
        "purge_bars": 20,
        "schema_version": 5,
        "signal_timing": "after_close_t",
        "split_policy": "fixed_dates_with_exact_label_end_guard_v1",
        "storage_kind": "symbol-oriented-ohlcv-bar-store",
        "stride": 5,
        "target_horizon": 5,
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "window_materialized": False,
        "window_size": 128,
    },
}


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
        for (_name, width), value in zip(META_FIELDS, row, strict=True):
            out += value.ljust(width, "\x00").encode("utf-32-le")
    return out


class Fixture:
    """Tiny shared evaluation arrays: 2 markets x 2 dates x 4 stocks per split."""

    def __init__(self, root: Path, markets=("TWSE", "US")) -> None:
        self.root = root
        self.baselines = root / "baselines"
        self.predictions = root / "exports"
        self.rows = [
            (f"S{market}{stock}", f"2025-06-0{day}T00:00:00+00:00", market, "stock", "fixture")
            for market in markets
            for day in (2, 3)
            for stock in range(4)
        ]
        self.count = len(self.rows)
        alpha, absolute = [], []
        for index, (_symbol, date, market, _a, _p) in enumerate(self.rows):
            common = (0.02 if market == "US" else -0.03) * (1 if date.endswith("02") else -1)
            for _h in range(HORIZONS):
                stock = 0.01 * ((index % 4) - 1.5)
                alpha.append(common + stock)
                absolute.append(2 * common + stock)
        self.alpha, self.absolute = alpha, absolute
        scales = [0.05] * HORIZONS
        for task_id in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]):
            base = self.baselines / task_id
            membership = {"ordered_symbol_dates_sha256": "f" * 64, "samples": self.count}
            base.mkdir(parents=True)
            (base / "complete.json").write_text(
                json.dumps(
                    {
                        "robust_scales": scales,
                        "validation_membership": membership,
                        "evaluation_membership": membership,
                    }
                )
            )
            for split in ("validation", "test"):
                write_npy(
                    base / "inputs" / split / "metadata.npy",
                    [(name, f"<U{width}") for name, width in META_FIELDS],
                    (self.count,),
                    metadata(self.rows),
                )
                write_npy(
                    base / "inputs" / split / "targets.npy",
                    "<f4",
                    (self.count, HORIZONS),
                    floats(alpha),
                )
                write_npy(
                    base / "inputs" / split / "absolute_targets.npy",
                    "<f4",
                    (self.count, HORIZONS),
                    floats(absolute),
                )
                for model, scale in (("gru-42", 0.5), ("gbdt-42", 0.25)):
                    write_npy(
                        base / "jobs" / model / split / "predictions.npy",
                        "<f4",
                        (self.count, HORIZONS),
                        floats([scale * v for v in alpha]),
                    )
            if task_id == ANALYSIS["DIRECTION_ID"]:
                audit = {
                    "counts": {
                        m: {str(h): [3, 2, 3] for h in range(1, 15)}
                        for m in ("TWSE", "US", "overall")
                    }
                }
                (base / "inputs/train").mkdir(parents=True, exist_ok=True)
                (base / "inputs/train/class-label-audit.json").write_text(json.dumps(audit))
                for model in ("gru-42", "gbdt-42"):
                    for split in ("validation", "test"):
                        write_npy(
                            base / "jobs" / model / split / "predictions.npy",
                            "<f4",
                            (self.count, HORIZONS, 3),
                            floats([0.0] * self.count * 42),
                        )

    def export(
        self,
        run_id: str,
        task: str,
        values,
        digest="f" * 64,
        checkpoint="checkpoint-003680",
        records=None,
    ) -> None:
        export = "export-20261017T000000Z-0a1b2c3d"
        directory = self.predictions / run_id / checkpoint / export
        tail = (HORIZONS,) if task == "alpha_point" else (HORIZONS, 3)
        splits = {}
        for split in ("validation", "test"):
            write_npy(
                directory / split / "predictions.npy", "<f4", (self.count, *tail), floats(values)
            )
            splits[split] = {"sample_membership": {"ordered_symbol_dates_sha256": digest}}
            if records is not None:
                splits[split]["cross_sectional"] = records
        (directory / "status.json").write_text(json.dumps({"state": "complete"}))
        (directory / "export.json").write_text(
            json.dumps(
                {
                    "kind": "signal-prediction-export",
                    "run_id": run_id,
                    "prediction_task": task,
                    "created_at": "2026-10-17T00:00:00+00:00",
                    "splits": splits,
                }
            )
        )


class DecompositionAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def run_analysis(self, *arguments: str) -> dict:
        output = self.fixture.root / "result.json"
        with contextlib.redirect_stdout(io.StringIO()):
            ANALYSIS["main"](
                [
                    "--baselines",
                    str(self.fixture.baselines),
                    "--output",
                    str(output),
                    "--workers",
                    "1",
                    *arguments,
                ]
            )
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
        self.assertAlmostEqual(
            alpha["gru-42_date_only"]["aggregate"], alpha["gru-42_date_only"]["aggregate"]
        )

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
            "--predictions",
            str(self.fixture.predictions),
            "--model",
            "A=run-a-alpha",
            "--model",
            "B=run-b-alpha",
            "--ensemble",
            "AB=A,B",
        )
        metrics = results[("alpha_point", "test")]["metrics"]
        for name in ("A", "B", "AB"):
            self.assertAlmostEqual(metrics[name]["aggregate"], metrics["gru-42"]["aggregate"])
        self.assertIn("A_within_plus_split_level", metrics)

    def test_checkpoint_selectors_keep_two_checkpoints_of_one_run_apart(self) -> None:
        self.fixture.export("run-a-alpha", "alpha_point", [0.5 * v for v in self.fixture.alpha])
        self.fixture.export(
            "run-a-alpha",
            "alpha_point",
            [0.25 * v for v in self.fixture.alpha],
            checkpoint="checkpoint-022079",
        )
        predictions = str(self.fixture.predictions)
        with self.assertRaisesRegex(ValueError, "checkpoint-003680, checkpoint-022079"):
            self.run_analysis("--predictions", predictions, "--model", "A=run-a-alpha")
        for selector in ("run-a-alpha/checkpoint-12", "run-a-alpha/../x", "run--a"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                ANALYSIS["latest_export"](self.fixture.predictions, selector)
        results = self.run_analysis(
            "--predictions",
            predictions,
            "--model",
            "Best=run-a-alpha/checkpoint-003680",
            "--model",
            "Late=run-a-alpha/checkpoint-022079",
        )
        metrics = results[("alpha_point", "validation")]["metrics"]
        self.assertAlmostEqual(metrics["Best"]["aggregate"], metrics["gru-42"]["aggregate"])
        self.assertAlmostEqual(metrics["Late"]["aggregate"], metrics["gbdt-42"]["aggregate"])
        models = ANALYSIS["resolve_models"](
            self.fixture.baselines,
            self.fixture.predictions,
            ["Late=run-a-alpha/checkpoint-022079"],
        )
        self.assertEqual(models["Late"]["run_id"], "run-a-alpha")
        self.assertEqual(models["Late"]["checkpoint"], "checkpoint-022079")

    def test_membership_mismatch_or_bad_ensemble_is_rejected(self) -> None:
        self.fixture.export(
            "run-a-alpha", "alpha_point", [0.0] * self.fixture.count * HORIZONS, digest="0" * 64
        )
        with self.assertRaises(ValueError):
            self.run_analysis(
                "--predictions", str(self.fixture.predictions), "--model", "A=run-a-alpha"
            )
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


def reference_rank_ic(fixture: Fixture, values) -> dict:
    """Independent per-market Rank IC built on statistics.correlation(method="ranked")."""
    f32 = ANALYSIS["float32"]
    cells: dict = {}
    groups: dict = {}
    for index, (_symbol, date, market, _a, _p) in enumerate(fixture.rows):
        groups.setdefault((market, date), []).append(index)
    for (market, _date), rows in groups.items():
        for h in range(HORIZONS):
            ys = [f32(fixture.alpha[r * HORIZONS + h]) for r in rows]
            xs = [f32(values[r * HORIZONS + h]) for r in rows]
            pairs = [
                (y, x)
                for y, x in zip(ys, xs, strict=False)
                if math.isfinite(y) and math.isfinite(x)
            ]
            ics = cells.setdefault(market, {}).setdefault(f"{h + 1}d", [])
            if (
                len(rows) >= 3
                and len(pairs) >= 3
                and len({x for _y, x in pairs}) > 1
                and len({y for y, _x in pairs}) > 1
            ):
                ics.append(
                    statistics.correlation(
                        [y for y, _x in pairs], [x for _y, x in pairs], method="ranked"
                    )
                )
    out = {}
    for market, horizons in cells.items():
        for horizon, ics in horizons.items():
            mean = statistics.fmean(ics) if ics else 0.0
            std = statistics.stdev(ics) if len(ics) > 1 else 0.0
            out.setdefault(market, {})[horizon] = {
                "rank_ic_mean": mean,
                "rank_ic_ir": mean / std if std > 0 else 0.0,
                "rank_ic_evaluated_dates": float(len(ics)),
            }
    return out


class RankICTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def analyze(self, fixture: Fixture, *arguments: str) -> tuple[dict, str]:
        output, stdout = fixture.root / "result.json", io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ANALYSIS["main"](
                [
                    "--baselines",
                    str(fixture.baselines),
                    "--output",
                    str(output),
                    "--predictions",
                    str(fixture.predictions),
                    "--workers",
                    "1",
                    *arguments,
                ]
            )
        return json.loads(output.read_text())["rank_ic"], stdout.getvalue()

    def assert_matches(self, computed: dict, expected: dict) -> None:
        for market, horizons in expected.items():
            for horizon, fields in horizons.items():
                for key, value in fields.items():
                    with self.subTest(market=market, horizon=horizon, key=key):
                        self.assertAlmostEqual(computed[market][horizon][key], value, places=12)

    def noisy(self, fixture: Fixture, seed: int) -> list[float]:
        # One-decimal values create ties; one date/horizon cell is constant.
        rng = random.Random(seed)
        values = [round(rng.gauss(0.0, 1.0), 1) for _ in range(fixture.count * HORIZONS)]
        for index, row in enumerate(fixture.rows):
            if row[2] == "TWSE" and row[1].startswith("2025-06-02"):
                values[index * HORIZONS] = 0.5
        return values

    def test_every_alpha_source_matches_an_independent_spearman_reference(self) -> None:
        fixture = Fixture(Path(self.temporary.name))
        values = self.noisy(fixture, 11)
        fixture.export("run-n-alpha", "alpha_point", values)
        rank_ic, _ = self.analyze(fixture, "--model", "N=run-n-alpha")
        self.assertEqual(set(rank_ic), {"validation", "test"})
        self.assertEqual(set(rank_ic["test"]), {"gru-42", "gbdt-42", "N"})
        computed = rank_ic["validation"]["N"]
        self.assert_matches(computed["by_market"], reference_rank_ic(fixture, values))
        self.assertEqual(computed["by_market"]["TWSE"]["1d"]["rank_ic_evaluated_dates"], 1.0)
        self.assertEqual(computed["by_market"]["TWSE"]["1d"]["rank_ic_ir_status"], "unavailable")
        self.assertFalse(computed["market_summary"]["TWSE"]["complete"])
        self.assertIsNone(computed["market_equal"])  # No TPEX in this fixture.
        self.assertIsNone(computed["recorded_check"])  # Export predates the record.
        # A monotone baseline ranks every date perfectly.
        gru = rank_ic["validation"]["gru-42"]["by_market"]["US"]["5d"]
        self.assertEqual((gru["rank_ic_mean"], gru["rank_ic_evaluated_dates"]), (1.0, 2.0))
        self.assertEqual(
            set(rank_ic["validation"]["N"]["daily"]["US"]), {"2025-06-02", "2025-06-03"}
        )

    def test_ensembles_rank_the_float32_member_mean(self) -> None:
        fixture = Fixture(Path(self.temporary.name))
        first, second = self.noisy(fixture, 3), self.noisy(fixture, 5)
        fixture.export("run-a-alpha", "alpha_point", first)
        fixture.export("run-b-alpha", "alpha_point", second)
        rank_ic, _ = self.analyze(
            fixture, "--model", "A=run-a-alpha", "--model", "B=run-b-alpha", "--ensemble", "AB=A,B"
        )
        f32 = ANALYSIS["float32"]
        mean = [f32((f32(a) + f32(b)) / 2) for a, b in zip(first, second, strict=False)]
        self.assert_matches(rank_ic["test"]["AB"]["by_market"], reference_rank_ic(fixture, mean))

    def test_equal_market_score_follows_the_pre_registered_definition(self) -> None:
        fixture = Fixture(Path(self.temporary.name), markets=("TPEX", "TWSE", "US"))
        # Date 2 ranks perfectly; date 3 swaps the top two stocks (Spearman 0.8).
        values = []
        for index, (_s, date, _m, _a, _p) in enumerate(fixture.rows):
            source = index
            if date.startswith("2025-06-03") and index % 4 >= 2:
                source = index + 1 if index % 4 == 2 else index - 1
            values.extend(fixture.alpha[source * HORIZONS : (source + 1) * HORIZONS])
        fixture.export("run-e-alpha", "alpha_point", values)
        rank_ic, _ = self.analyze(fixture, "--model", "E=run-e-alpha")
        equal = rank_ic["validation"]["E"]["market_equal"]
        self.assertEqual(equal["markets"], ["US", "TWSE", "TPEX"])
        self.assertAlmostEqual(equal["rank_ic"], 0.9, places=12)
        self.assertAlmostEqual(equal["rank_ic_ir"], 0.9 / statistics.stdev([1.0, 0.8]), places=9)

    def test_exported_record_is_checked_and_mismatches_are_reported(self) -> None:
        fixture = Fixture(Path(self.temporary.name))
        values = self.noisy(fixture, 13)
        record = {"signal": "alpha_point", "by_market": reference_rank_ic(fixture, values)}
        fixture.export("run-r-alpha", "alpha_point", values, records=record)
        rank_ic, stdout = self.analyze(fixture, "--model", "R=run-r-alpha")
        check = rank_ic["validation"]["R"]["recorded_check"]
        self.assertTrue(check["matches"], check)
        self.assertEqual((check["compared"], check["missing"]), (2 * HORIZONS * 3, 0))
        self.assertNotIn("WARNING", stdout)
        record["by_market"]["US"]["3d"]["rank_ic_mean"] += 1e-6
        fixture.export("run-r-alpha", "alpha_point", values, records=record)
        rank_ic, stdout = self.analyze(fixture, "--model", "R=run-r-alpha")
        self.assertFalse(rank_ic["test"]["R"]["recorded_check"]["matches"])
        self.assertIn("WARNING: R test Rank IC differs", stdout)

    def test_single_date_rules_match_the_project_metric(self) -> None:
        date_rank_ic = ANALYSIS["date_rank_ic"]
        self.assertIsNone(date_rank_ic([1.0, 2.0], [1.0, 2.0]))
        self.assertIsNone(date_rank_ic([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]))
        self.assertIsNone(date_rank_ic([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(date_rank_ic([1.0, 2.0, math.nan, 4.0], [3.0, 1.0, 2.0, math.inf]))
        self.assertAlmostEqual(
            date_rank_ic([1.0, 2.0, 3.0, math.nan], [1.0, 3.0, 2.0, 5.0]),
            statistics.correlation([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], method="ranked"),
        )
        rng = random.Random(7)
        for _ in range(50):
            size = rng.randint(3, 40)
            ys = [rng.choice((-1.0, 0.0, 0.5, 2.0)) + rng.random() // 0.5 for _ in range(size)]
            xs = [round(rng.gauss(0.0, 1.0), 1) for _ in range(size)]
            if len(set(xs)) > 1 and len(set(ys)) > 1:
                self.assertAlmostEqual(
                    date_rank_ic(ys, xs),
                    statistics.correlation(ys, xs, method="ranked"),
                    places=12,
                )
        statistics_of = ANALYSIS["rank_ic_statistics"]
        self.assertEqual(statistics_of([])["rank_ic_status"], "unavailable")
        single = statistics_of([0.2])
        self.assertEqual((single["rank_ic_ir"], single["rank_ic_ir_status"]), (0.0, "unavailable"))


class FakeReader:
    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents
        self.wrapper, self.volume_id = Path("/nonexistent"), "volume"

    def read_json(self, key, limit=None):
        payload = self.documents[key]
        raw = json.dumps(payload).encode()
        return payload, hashlib.sha256(raw).hexdigest()

    def status_keys(self, prefix):
        yield from (
            key for key in self.documents if key.startswith(prefix) and key.endswith("/status.json")
        )


class PredictionDownloadTests(unittest.TestCase):
    def documents(
        self, run_id: str, export: str, created: str, state="complete", name="checkpoint-003680"
    ):
        prefix = f"diagnostics/signal-predictions/{run_id}/{name}/{export}/"
        checkpoint = f"/runpod-volume/savedModel/{run_id}/{name}"
        return {
            prefix + "status.json": {"state": state, "checkpoint": checkpoint},
            prefix + "export.json": {
                "schema_version": "1.0",
                "kind": "signal-prediction-export",
                "run_id": run_id,
                "checkpoint": {"path": checkpoint},
                "created_at": created,
                "splits": {
                    "validation": {"predictions_sha256": "a" * 64},
                    "test": {"predictions_sha256": "b" * 64},
                },
            },
        }

    def test_selects_the_newest_completed_export_and_skips_running_ones(self) -> None:
        documents = {
            **self.documents(
                "run-x-alpha", "export-20261017T000000Z-00000001", "2026-10-17T00:00:00+00:00"
            ),
            **self.documents(
                "run-x-alpha", "export-20261018T000000Z-00000002", "2026-10-18T00:00:00+00:00"
            ),
            **self.documents(
                "run-x-alpha",
                "export-20261019T000000Z-00000003",
                "2026-10-19T00:00:00+00:00",
                state="running",
            ),
        }
        result = DOWNLOAD["select_export"](FakeReader(documents), "run-x-alpha")
        self.assertEqual(result["relative"].name, "export-20261018T000000Z-00000002")
        self.assertEqual(result["splits"], {"validation": "a" * 64, "test": "b" * 64})

    def test_named_checkpoints_are_selected_and_bare_ids_must_be_unambiguous(self) -> None:
        documents = {
            **self.documents(
                "run-x-alpha", "export-20261017T000000Z-00000001", "2026-10-17T00:00:00+00:00"
            ),
            **self.documents(
                "run-x-alpha",
                "export-20261018T000000Z-00000002",
                "2026-10-18T00:00:00+00:00",
                name="checkpoint-022079",
            ),
        }
        reader = FakeReader(documents)
        with self.assertRaisesRegex(ValueError, "checkpoint-003680, checkpoint-022079"):
            DOWNLOAD["select_export"](reader, "run-x-alpha")
        for name, export in (
            ("checkpoint-003680", "export-20261017T000000Z-00000001"),
            ("checkpoint-022079", "export-20261018T000000Z-00000002"),
        ):
            result = DOWNLOAD["select_export"](reader, f"run-x-alpha/{name}")
            self.assertEqual(result["relative"], Path("run-x-alpha", name, export))
        with self.assertRaises(FileNotFoundError):
            DOWNLOAD["select_export"](reader, "run-x-alpha/checkpoint-011040")
        self.assertEqual(DOWNLOAD["safe_selector"]("run-x-alpha"), ("run-x-alpha", None))
        for value in ("run-x-alpha/", "run-x-alpha/checkpoint-1", "run-x-alpha/a/b", "/x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DOWNLOAD["safe_selector"](value)

    def test_forward_exports_are_selected_only_in_their_own_layout(self) -> None:
        run_id, name = "run-x-alpha", "checkpoint-018399"
        export = "export-20261021T000000Z-00000004"
        prefix = f"diagnostics/forward-predictions/2026-10-20/{run_id}/{name}/{export}/"
        checkpoint = f"/runpod-volume/savedModel/{run_id}/{name}"
        documents = {
            prefix + "status.json": {"state": "complete", "checkpoint": checkpoint},
            prefix + "export.json": {
                "schema_version": "1.0",
                "kind": "signal-prediction-export",
                "run_id": run_id,
                "checkpoint": {"path": checkpoint},
                "created_at": "2026-10-21T00:00:00+00:00",
                "forward_package": {"end_exclusive": "2026-10-20", "package_sha256": "c" * 64},
                "splits": {
                    "overlap": {"predictions_sha256": "a" * 64},
                    "forward": {"predictions_sha256": "b" * 64},
                },
            },
        }
        reader = FakeReader(documents)
        layout = DOWNLOAD["export_layout"]("2026-10-20")
        result = DOWNLOAD["select_export"](reader, f"{run_id}/{name}", layout)
        self.assertEqual(result["relative"], Path(run_id, name, export))
        self.assertEqual(result["splits"], {"overlap": "a" * 64, "forward": "b" * 64})
        self.assertEqual(
            layout["local"], Path("artifacts/diagnostics/forward-predictions/2026-10-20")
        )
        # Regular selection never sees forward exports, and the end date must match.
        with self.assertRaises(FileNotFoundError):
            DOWNLOAD["select_export"](reader, f"{run_id}/{name}")
        documents[prefix + "export.json"]["forward_package"]["end_exclusive"] = "2026-10-19"
        with self.assertRaisesRegex(ValueError, "identity"):
            DOWNLOAD["select_export"](reader, f"{run_id}/{name}", layout)
        for value in ("2026-10-2", "20261020", "2026-13-01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DOWNLOAD["export_layout"](value)

    def test_baseline_inputs_match_the_analysis_and_verify_digests(self) -> None:
        files = DOWNLOAD["baseline_files"]()
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(len(files), 15)
        for baseline in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"]):
            self.assertTrue(any(name.startswith(baseline + "/") for name in files))
        payload = b"fixture-array"
        artifacts = {
            name.partition("/")[2]: {"sha256": hashlib.sha256(payload).hexdigest()}
            for name in files
        }
        reader = FakeReader(
            {
                f"baselines/{baseline}/complete.json": {"state": "complete", "artifacts": artifacts}
                for baseline in (ANALYSIS["ALPHA_ID"], ANALYSIS["DIRECTION_ID"])
            }
        )
        calls = []

        def fake_download(_reader, key, destination, _timeout):
            calls.append(key)
            destination.write_bytes(payload if "corrupt" not in str(destination) else b"x")

        download = DOWNLOAD["download_baseline_arrays"]
        with (
            tempfile.TemporaryDirectory() as temporary,
            DOWNLOAD["ThreadPoolExecutor"](max_workers=2) as executor,
        ):
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

    def evaluation_reader(self, *, shard="shards/0001/part-0000.parquet"):
        """A fake volume whose dataset ID follows the baseline's evaluation request."""
        content = {
            "shard": b"shard-bytes",
            "index": b"symbol-index",
            "ranges": b"ranges",
            "universe-ranges": b"universe-ranges",
        }
        digest = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
        manifest = {
            "symbol_index": {
                "relative_path": "symbol-index.parquet",
                "sha256": digest["index"],
                "size_bytes": len(content["index"]),
            },
            "cutoff_ranges": {
                "relative_path": "cutoff-ranges.parquet",
                "sha256": digest["ranges"],
                "size_bytes": len(content["ranges"]),
            },
            "shards": [
                {
                    "relative_path": shard,
                    "sha256": digest["shard"],
                    "size_bytes": len(content["shard"]),
                }
            ],
        }
        manifest_sha256 = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        identity = {"source_manifest_sha256": manifest_sha256, "window_size": 128}
        universe = DOWNLOAD["canonical_json_sha256"](identity)
        request = EVALUATION_REQUEST
        good = DOWNLOAD["evaluation_dataset_id"](request)
        documents = {
            # Real baseline records exceed the 4 MiB probe limit.
            f"baselines/{ANALYSIS['ALPHA_ID']}/complete.json": {
                "padding": "x" * (5 * 1024**2),
                "identity": {"contract": {"evaluation_request": request}},
                "data_identity": {
                    "split_sources": {
                        split: {"manifest_sha256": manifest_sha256, "sample_universe": universe}
                        for split in ("validation", "test")
                    }
                },
            },
            f"datasets/{good}/prepared/bar-store/bar-store.json": manifest,
            f"datasets/{good}/prepared/sample-universes/{universe}/universe.json": {
                "state": "ready",
                "identity": identity,
                "ranges_sha256": digest["universe-ranges"],
            },
        }

        class Reader(FakeReader):
            def read_json(self, key, limit=4 * 1024**2):
                # Enforce the bounded read like the real wrapper does.
                payload, digest = super().read_json(key)
                if len(json.dumps(payload).encode()) > limit:
                    raise ValueError("Remote diagnostic metadata exceeds the bounded read limit")
                return payload, digest

        files = {
            f"datasets/{good}/prepared/bar-store/bar-store.json": json.dumps(manifest).encode(),
            f"datasets/{good}/prepared/bar-store/_SUCCESS.json": b"{}",
            f"datasets/{good}/prepared/bar-store/symbol-index.parquet": content["index"],
            f"datasets/{good}/prepared/bar-store/cutoff-ranges.parquet": content["ranges"],
            f"datasets/{good}/prepared/bar-store/{shard}": content["shard"],
            f"datasets/{good}/prepared/sample-universes/{universe}/universe.json": b"{}",
            f"datasets/{good}/prepared/sample-universes/{universe}/cutoff-ranges.parquet": content[
                "universe-ranges"
            ],
        }
        return Reader(documents), files, good

    def test_evaluation_data_plan_finds_the_baseline_source_and_verifies_files(self) -> None:
        reader, files, dataset_id = self.evaluation_reader()
        plan = DOWNLOAD["evaluation_plan"](reader)
        self.assertEqual(plan["dataset_id"], dataset_id)
        self.assertEqual({key for key, _, _ in plan["files"]}, set(files))
        calls = []

        def fetch(_reader, key, destination, _timeout):
            calls.append(key)
            destination.write_bytes(files[key])

        download = DOWNLOAD["download_evaluation_data"]
        with (
            tempfile.TemporaryDirectory() as temporary,
            DOWNLOAD["ThreadPoolExecutor"](max_workers=2) as executor,
        ):
            project = Path(temporary).resolve()
            root, downloaded, verified = download(
                reader, project, executor, 2, 60, plan, fetch=fetch
            )
            self.assertEqual((downloaded, verified), (len(files), 0))
            record = json.loads((root / "evaluation-data.json").read_text())
            self.assertEqual(record["dataset_id"], dataset_id)
            # Files with a digest are kept on a rerun; the two small records are refetched.
            _root, downloaded, verified = download(
                reader, project, executor, 2, 60, plan, fetch=fetch
            )
            self.assertEqual((downloaded, verified), (2, len(files) - 2))
            shard_key = next(key for key in files if "/shards/" in key)
            files[shard_key] = b"tampered!!!"
            (root / shard_key.split("/", 2)[2]).write_bytes(b"stale")
            with self.assertRaises(ValueError):
                download(reader, project, executor, 2, 60, plan, fetch=fetch)

    def test_evaluation_dataset_id_matches_the_data_workflow(self) -> None:
        policy = runpy.run_path(str(ROOT / "src/stock_forecasting/data_policy.py"))
        request = EVALUATION_REQUEST
        selection = {"dataset_request": request}
        workflow = policy["evaluation_dataset_id"](
            selection, {"evaluation_source_start": "2016-01-01"}
        )
        self.assertEqual(DOWNLOAD["evaluation_dataset_id"](request), workflow)

    def test_evaluation_data_rejects_unsafe_shards_and_identity_mismatch(self) -> None:
        reader, _files, _dataset_id = self.evaluation_reader(shard="../escape.parquet")
        with self.assertRaises(ValueError):
            DOWNLOAD["evaluation_plan"](reader)
        reader, _files, _dataset_id = self.evaluation_reader()
        key = f"baselines/{ANALYSIS['ALPHA_ID']}/complete.json"
        reader.documents[key]["data_identity"]["split_sources"]["test"]["sample_universe"] = (
            "c" * 64
        )
        with self.assertRaises(ValueError):
            DOWNLOAD["evaluation_plan"](reader)
        reader, _files, dataset_id = self.evaluation_reader()
        reader.documents[f"datasets/{dataset_id}/prepared/bar-store/bar-store.json"]["x"] = 1
        with self.assertRaisesRegex(ValueError, "differs from the manifest"):
            DOWNLOAD["evaluation_plan"](reader)

    def test_identity_mismatch_and_unsafe_ids_are_rejected(self) -> None:
        documents = self.documents(
            "run-x-alpha", "export-20261017T000000Z-00000001", "2026-10-17T00:00:00+00:00"
        )
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
            'printf \'%s\\0\' "$@" > "${HARNESS_ROOT}/worker-args"\n',
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
        # The terminal signal must match status.json in this session's directory.
        self.assertIn("--session stock-forecasting-export-predictions", runner)
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
            "--checkpoint",
            "run-a-alpha",
            "--checkpoint=run-a-alpha/checkpoint-022079",
            "--splits",
            "validation",
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
        self.assertEqual(source.count('diagnostics" / "signal-predictions"'), 1)

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
