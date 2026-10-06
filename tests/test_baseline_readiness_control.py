"""Dependency-free admission regressions for current-rule baseline populations."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "tests/test_baseline_independence.py"))
GATE = BASE["GATE"]
POLICY = runpy.run_path(str(ROOT / "src/stock_forecasting/data_policy.py"))
encode = BASE["encode"]


class Reader(GATE["BaselineArtifactReader"]):
    def __init__(self, objects):
        super().__init__(ROOT, bucket="fixture-volume")
        self.objects, self.calls = objects, []

    def s3(self, *args):
        self.calls.append(args)
        if args[:2] == ("s3api", "list-objects-v2"):
            key = args[args.index("--prefix") + 1]
            return encode(
                {
                    "Contents": [{"Key": key}] if key in self.objects else [],
                    "IsTruncated": False,
                }
            )
        return BASE["MemoryReader"].s3(self, *args)


def clean_fixture(selected, prepared, *, counts=None, policy=None, source_hash=None):
    prefix = f"datasets/{selected['dataset_request_sha256']}/prepared/"
    bar_bytes = prepared[prefix + "bar-store/bar-store.json"]
    identity = GATE["cleaning_identity"](
        ROOT,
        json.loads(bar_bytes),
        source_hash or hashlib.sha256(bar_bytes).hexdigest(),
        policy or POLICY["load_data_policy"](ROOT),
    )
    root = prefix + "sample-universes/" + GATE["digest"](identity) + "/"
    ranges = b"fixture-cleaned-ranges"
    counts = counts or {"train": 180, "validation": 60, "test": 70}
    payload = {
        "state": "ready",
        "identity": identity,
        "split_counts": counts,
        "ranges_sha256": hashlib.sha256(ranges).hexdigest(),
        "audit": {split: {"accepted": count} for split, count in counts.items()},
    }
    return {root + "universe.json": encode(payload), root + "cutoff-ranges.parquet": ranges}


class BaselineReadinessTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = BASE["selection"](), BASE["selection"]("2016-01-01")
        self.objects = {**BASE["fixture"](self.a), **BASE["fixture"](self.b)}

    def verify(self, selected=None, objects=None, workers=2):
        reader = Reader(self.objects if objects is None else objects)
        result = GATE["verify_baseline_data"](ROOT, selected or self.a, reader, workers=workers)
        self.assertNotIn("hf-models", repr(reader.calls))
        self.assertNotIn("lifecycle/stage1/dataset.json", repr(reader.calls))
        self.assertNotIn("sample_counts", result)
        return result, reader

    def test_prepared_counts_are_never_reported_as_eligible_counts(self):
        result, _ = self.verify()
        self.assertEqual(result["state"], "ready_for_baseline_build")
        self.assertEqual(result["cleaning_state"], "pending_build")
        self.assertEqual(
            result["eligible_sample_counts"], dict.fromkeys(("train", "validation", "test"))
        )
        self.assertIn("not_completed_baseline_results", result["scope"])
        for source in result["sources"].values():
            self.assertEqual(source["prepared"]["prepared_candidate_counts"]["train"], 300)
            self.assertNotIn("sample_counts", source["prepared"])
            self.assertEqual(source["prepared"]["scope"], "prepared_bar_store")
        self.assertIn(
            "CPU prepare and market-data downloads are not required", result["next_action"]
        )

    def test_full_clean_counts_use_a_train_and_only_b_evaluation(self):
        a_counts = {"train": 180, "validation": 11, "test": 13}
        b_counts = {"train": 220, "validation": 60, "test": 70}
        self.objects.update(clean_fixture(self.a, self.objects, counts=a_counts))
        self.objects.update(clean_fixture(self.b, self.objects, counts=b_counts))
        for selected, train_count in ((self.a, 180), (self.b, 220), (self.a, 180)):
            result, reader = self.verify(selected)
            self.assertEqual(result["cleaning_state"], "ready")
            self.assertEqual(
                result["eligible_sample_counts"],
                {"train": train_count, "validation": 60, "test": 70},
            )
            self.assertEqual(
                result["split_sources"]["validation"], self.b["dataset_request_sha256"]
            )
            self.assertEqual(result["split_sources"]["test"], self.b["dataset_request_sha256"])
            self.assertEqual(len(result["sources"]), 1 if selected is self.b else 2)
            listings = [call for call in reader.calls if call[:2] == ("s3api", "list-objects-v2")]
            self.assertEqual(len(listings), len(result["sources"]))

    def test_old_policy_or_snapshot_cache_cannot_satisfy_current_rules(self):
        changed = POLICY["load_data_policy"](ROOT)
        changed["minimum_median_daily_turnover"]["USD"] *= 2
        for kwargs in ({"policy": changed}, {"source_hash": "0" * 64}):
            objects = {**self.objects, **clean_fixture(self.a, self.objects, **kwargs)}
            objects.update(clean_fixture(self.b, self.objects))
            with self.subTest(kwargs=kwargs):
                result, _ = self.verify(objects=objects)
                self.assertEqual(result["cleaning_state"], "pending_build")
                self.assertIsNone(result["eligible_sample_counts"]["train"])
                self.assertEqual(result["eligible_sample_counts"]["test"], 70)

    def test_missing_shared_b_source_fails_without_fallback(self):
        with self.assertRaises((ValueError, KeyError)):
            self.verify(objects=BASE["fixture"](self.a))

    def test_inconsistent_current_marker_or_empty_used_population_fails(self):
        current = clean_fixture(self.a, self.objects)
        marker = next(key for key in current if key.endswith("universe.json"))
        original = json.loads(current[marker])
        variants = []
        for field, value in (("state", "building"), ("ranges_sha256", "wrong")):
            payload = copy.deepcopy(original)
            payload[field] = value
            variants.append(payload)
        for changes in (
            {"split_counts": {"train": 0, "validation": 1, "test": 1}},
            {"split_counts": {"train": True, "validation": 1, "test": 1}},
            {"audit": {"train": {"accepted": 179}}},
            {"identity": {**original["identity"], "window_size": 64}},
        ):
            variants.append({**original, **changes})
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.verify(objects={**self.objects, **current, marker: encode(payload)})

    def test_missing_or_empty_ranges_is_an_error_not_a_pending_rebuild(self):
        current = clean_fixture(self.a, self.objects)
        ranges = next(key for key in current if key.endswith(".parquet"))
        for value in (None, b""):
            objects = {**self.objects, **current}
            if value is None:
                objects.pop(ranges)
            else:
                objects[ranges] = value
            with self.subTest(value=value), self.assertRaises((ValueError, KeyError)):
                self.verify(objects=objects)

    def test_s3_failure_cannot_masquerade_as_absent_index(self):
        reader = GATE["BaselineArtifactReader"](ROOT, bucket="fixture-volume")
        for failure in (ValueError("Access denied"), RuntimeError("Network unavailable")):
            with patch.object(reader, "s3", side_effect=failure), self.assertRaises(type(failure)):
                reader.optional_read("datasets/fixture/prepared/sample-universes/key/universe.json")

    def test_invalid_or_incomplete_listing_cannot_report_pending(self):
        reader = GATE["BaselineArtifactReader"](ROOT, bucket="fixture-volume")
        for listing in (
            [],
            {},
            {"Contents": [], "IsTruncated": True},
            {"Contents": [{}], "IsTruncated": False},
        ):
            with (
                self.subTest(listing=listing),
                patch.object(reader, "s3", return_value=encode(listing)),
                self.assertRaisesRegex(ValueError, "Invalid cleaned-index metadata listing"),
            ):
                reader.optional_read("datasets/fixture/prepared/sample-universes/key/universe.json")

    def test_preparation_extreme_return_filter_does_not_cap_clean_population(self):
        # Runtime cleaning can restore genuine extreme returns omitted by old ranges.
        counts = {"train": 500, "validation": 120, "test": 130}
        self.objects.update(clean_fixture(self.b, self.objects, counts=counts))
        result, _ = self.verify(self.b, workers=1)
        self.assertEqual(result["eligible_sample_counts"], counts)

    def test_mounted_same_size_range_corruption_fails(self):
        objects = {
            **self.objects,
            **clean_fixture(self.a, self.objects),
            **clean_fixture(self.b, self.objects),
        }
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            for key, content in objects.items():
                path = volume / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            reader = GATE["BaselineArtifactReader"](ROOT, volume=volume)
            result = GATE["verify_baseline_data"](ROOT, self.a, reader, workers=2)
            self.assertEqual(result["eligible_sample_counts"]["train"], 180)
            ranges = next(
                volume.glob("datasets/*/prepared/sample-universes/*/cutoff-ranges.parquet")
            )
            ranges.write_bytes(b"x" * ranges.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                GATE["verify_baseline_data"](ROOT, self.a, reader, workers=2)

    def test_preflight_identity_equals_actual_runtime_identity_expression(self):
        source = ROOT / "src/stock_forecasting/data/sample_universe.py"
        tree = ast.parse(source.read_text())
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name == "_split_spec")
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ALGORITHM"
                    for target in node.targets
                )
            )
        ]
        runtime = next(
            node for node in tree.body if getattr(node, "name", "") == "ensure_sample_universe"
        )
        identity_node = next(
            node
            for node in runtime.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "identity" for target in node.targets
            )
        )
        for selected in (self.a, self.b):
            key = f"datasets/{selected['dataset_request_sha256']}/prepared/bar-store/bar-store.json"
            bar = json.loads(self.objects[key])
            checksum = hashlib.sha256(self.objects[key]).hexdigest()
            policy = POLICY["load_data_policy"](ROOT)
            scope = {
                "manifest": bar,
                "policy": policy,
                "root": Path("unused"),
                "sha256_file": lambda path, value=checksum: value,
            }
            exec(
                compile(
                    ast.Module(body=[*nodes, identity_node], type_ignores=[]), str(source), "exec"
                ),
                scope,
            )
            self.assertEqual(
                GATE["cleaning_identity"](ROOT, bar, checksum, policy), scope["identity"]
            )


if __name__ == "__main__":
    unittest.main()
