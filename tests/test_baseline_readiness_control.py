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
        if args[:2] == ("s3api", "list-objects-v2"):
            self.calls.append(args)
            key = args[args.index("--prefix") + 1]
            return encode(
                {
                    "Contents": [{"Key": key}] if key in self.objects else [],
                    "IsTruncated": False,
                }
            )
        return BASE["MemoryReader"].s3(self, *args)


class CacheReader(Reader):
    def s3(self, *args):
        if (args[:2] == ("s3api", "list-objects-v2")
                and args[args.index("--max-keys") + 1] == "1000"):
            self.calls.append(args)
            prefix = args[args.index("--prefix") + 1]
            return encode({"IsTruncated": False, "Contents": [
                {"Key": key, "Size": len(value), "ETag": hashlib.sha256(value).hexdigest(),
                 "LastModified": "2026-10-06T00:00:00Z"}
                for key, value in sorted(self.objects.items()) if key.startswith(prefix)
            ]})
        return super().s3(*args)


class ReadinessReuseTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = BASE["selection"](), BASE["selection"]("2016-01-01")
        self.objects = {**BASE["fixture"](self.a), **BASE["fixture"](self.b)}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_root = Path(self.temp.name)

    def verify(self, *, selected=None, objects=None, now=1000, release="a" * 64, reader=None):
        reader = reader or CacheReader(self.objects if objects is None else objects)
        result, reused = GATE["verify_with_receipt"](
            ROOT, selected or self.a, reader, workers=2, release_digest=release,
            cache_root=self.cache_root, now=now,
        )
        return result, reused, reader

    @staticmethod
    def artifact_heads(reader):
        return [call for call in reader.calls if call[:2] == ("s3api", "head-object")
                and call[call.index("--key") + 1].endswith(".parquet")]

    def test_launch_reuses_success_without_repeating_artifact_head_requests(self):
        first, reused, first_reader = self.verify()
        second, reused_second, second_reader = self.verify(now=1100)
        self.assertFalse(reused)
        self.assertTrue(reused_second)
        self.assertEqual(second, first)
        self.assertEqual(len(self.artifact_heads(first_reader)), 6)
        self.assertEqual(self.artifact_heads(second_reader), [])
        self.assertEqual(len([call for call in second_reader.calls
                              if call[:2] == ("s3api", "list-objects-v2")]), 4)
        self.assertNotIn("hf-models", repr(second_reader.calls))

    def test_full_size_shard_inventory_avoids_260_repeated_heads(self):
        for selected in (self.a, self.b):
            prefix = f"datasets/{selected['dataset_request_sha256']}/"
            bar_key = prefix + "prepared/bar-store/bar-store.json"
            bar = json.loads(self.objects[bar_key])
            template = bar["shards"][0]
            payload = self.objects[prefix + "prepared/bar-store/" + template["relative_path"]]
            bar["shards"] = []
            for index in range(128):
                relative = f"shards/bucket-{index:04d}/shard.parquet"
                bar["shards"].append({**template, "relative_path": relative})
                self.objects[prefix + "prepared/bar-store/" + relative] = payload
            bar["row_count"] *= 128
            bar["symbol_count"] *= 128
            self.objects[bar_key] = encode(bar)
            bar_hash = hashlib.sha256(self.objects[bar_key]).hexdigest()
            dataset_key = prefix + "dataset-manifest.json"
            dataset = json.loads(self.objects[dataset_key])
            dataset["artifacts"]["bar_store_manifest"].update(
                sha256=bar_hash, size_bytes=len(self.objects[bar_key]),
            )
            self.objects[dataset_key] = encode(dataset)
            success_key = prefix + "prepared/bar-store/_SUCCESS.json"
            success = json.loads(self.objects[success_key])
            success["bar_store_manifest_sha256"] = bar_hash
            self.objects[success_key] = encode(success)
        first, _, first_reader = self.verify()
        second, reused, second_reader = self.verify()
        self.assertTrue(reused)
        self.assertEqual(first, second)
        self.assertEqual(len(self.artifact_heads(first_reader)), 260)
        self.assertEqual(self.artifact_heads(second_reader), [])

    def test_cache_does_not_extend_its_own_lifetime(self):
        self.verify()
        self.assertTrue(self.verify(now=1500)[1])
        self.assertFalse(self.verify(now=1600)[1])
        self.assertFalse(self.verify(now=1599)[1])  # Clock rollback also invalidates reuse.

    def test_ab_source_switch_and_code_change_cannot_reuse_wrong_success(self):
        self.verify()
        self.assertFalse(self.verify(selected=self.b)[1])
        self.assertTrue(self.verify(selected=self.a)[1])
        self.assertFalse(self.verify(release="b" * 64)[1])

    def test_volume_switch_cannot_reuse_success(self):
        self.verify()
        reader = CacheReader(self.objects)
        reader.bucket = "different-volume"
        self.assertFalse(self.verify(reader=reader)[1])

    def test_policy_change_rechecks_even_with_same_release_argument(self):
        self.verify()
        original = GATE["baseline_sources"]

        def changed(*args):
            policy, sources, splits = original(*args)
            policy["minimum_median_daily_turnover"]["USD"] *= 2
            return policy, sources, splits

        with patch.dict(GATE["verify_with_receipt"].__globals__, {"baseline_sources": changed}):
            self.assertFalse(self.verify()[1])

    def test_deleted_shard_fails_instead_of_using_cached_success(self):
        self.verify()
        objects = dict(self.objects)
        objects.pop(next(key for key in objects if key.endswith("/shard.parquet")))
        with self.assertRaises((ValueError, KeyError)):
            self.verify(objects=objects)

    def test_same_size_replaced_shard_invalidates_reuse(self):
        self.verify()
        objects = dict(self.objects)
        key = next(key for key in objects if key.endswith("/shard.parquet"))
        objects[key] = b"x" * len(objects[key])
        self.assertFalse(self.verify(objects=objects)[1])

    def test_newly_completed_clean_index_rechecks_and_refreshes_counts(self):
        self.verify()
        self.objects.update(clean_fixture(self.a, self.objects))
        self.objects.update(clean_fixture(self.b, self.objects))
        result, reused, _ = self.verify()
        self.assertFalse(reused)
        self.assertEqual(result["cleaning_state"], "ready")
        self.assertEqual(result["eligible_sample_counts"]["test"], 70)

    def test_changed_manifest_failure_does_not_overwrite_success(self):
        self.verify()
        path = next(self.cache_root.glob("*.json"))
        original = path.read_bytes()
        objects = dict(self.objects)
        key = next(key for key in objects if key.endswith("/dataset-manifest.json"))
        payload = json.loads(objects[key])
        payload["state"] = "building"
        objects[key] = encode(payload)
        with self.assertRaises(ValueError):
            self.verify(objects=objects)
        self.assertEqual(path.read_bytes(), original)

    def test_remote_failure_never_falls_back_to_success(self):
        self.verify()
        reader = CacheReader(self.objects)
        with (patch.object(reader, "s3", side_effect=ValueError("network failure")),
              self.assertRaisesRegex(ValueError, "network failure")):
            self.verify(reader=reader)

    def test_missing_or_corrupt_receipt_performs_full_verification(self):
        self.verify()
        path = next(self.cache_root.glob("*.json"))
        for raw in (b"broken-json", b"[]", b"{}"):
            path.write_bytes(raw)
            self.assertFalse(self.verify()[1])

    def test_unverified_code_or_mounted_path_never_uses_receipt(self):
        self.verify()
        self.assertFalse(self.verify(release="")[1])
        reader = CacheReader(self.objects)
        reader.volume = self.cache_root
        with patch.dict(GATE["verify_with_receipt"].__globals__, {
            "verify_baseline_data": lambda *args, **kwargs: {"mounted": True},
        }):
            result, reused, _ = self.verify(reader=reader)
        self.assertEqual(result, {"mounted": True})
        self.assertFalse(reused)

    def test_paginated_inventory_is_complete_and_sorted(self):
        reader = CacheReader({})
        def entry(key):
            return {"Key": key, "Size": 1, "ETag": "fixture", "LastModified": "date"}
        pages = [encode({"IsTruncated": True, "NextContinuationToken": "page2",
                         "Contents": [entry("prepared/z")]}),
                 encode({"IsTruncated": False, "Contents": [entry("prepared/a")]})]
        with patch.object(reader, "s3", side_effect=pages) as s3:
            result = reader.inventory("prepared/")
        self.assertEqual([item["Key"] for item in result], ["prepared/a", "prepared/z"])
        self.assertIn("page2", s3.call_args_list[1].args)

    def test_inventory_without_revision_or_complete_pages_fails_closed(self):
        reader = CacheReader({})
        for page in ({}, {"IsTruncated": True, "Contents": []},
                     {"IsTruncated": False, "Contents": [{"Key": "prepared/a", "Size": 1}]}):
            with (patch.object(reader, "s3", return_value=encode(page)),
                  self.assertRaises(ValueError)):
                reader.inventory("prepared/")


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
