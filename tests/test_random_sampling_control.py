"""Standard-library sampling checks; no local ML environment is imported."""

from __future__ import annotations

import ast
import importlib.util
import math
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "random_sampling_control", ROOT / "src/stock_forecasting/random_sampling.py"
)
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


class RandomSamplingControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.scratch = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def draw(self, strata=None, seed=42, epoch=0, stable=False):
        return list(self.stream(strata, seed, epoch, stable))

    def stream(self, strata=None, seed=42, epoch=0, stable=False):
        return sampling.shuffled_positions(
            strata or [{"start": 0, "count": 4099, "samples": 4099}],
            seed=seed,
            epoch=epoch,
            stable_subset=stable,
            scratch_root=self.scratch,
            max_scratch_bytes=16 * 1024**2,
        )

    def test_full_coverage_without_duplicates_and_no_adjacent_blocks(self):
        values = self.draw()
        self.assertEqual(sorted(values), list(range(4099)))
        adjacent = sum(abs(a - b) == 1 for a, b in zip(values, values[1:]))
        self.assertLess(adjacent, 25)

    def test_exact_reproducibility_and_epoch_change(self):
        self.assertEqual(self.draw(), self.draw())
        self.assertNotEqual(self.draw(), self.draw(seed=43))
        self.assertNotEqual(self.draw(), self.draw(epoch=1))

    def test_resume_regeneration_preserves_exact_suffix(self):
        expected = self.draw(seed=71, epoch=3)
        regenerated = self.stream(seed=71, epoch=3)
        for _ in range(13 * 32):
            next(regenerated)
        self.assertEqual(list(regenerated), expected[13 * 32:])

    def test_subset_membership_can_be_fixed_while_order_changes(self):
        strata = [{"start": 0, "count": 12007, "samples": 1001}]
        first = self.draw(strata, stable=True)
        second = self.draw(strata, epoch=1, stable=True)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)
        self.assertEqual(len(set(first)), 1001)
        self.assertNotEqual(set(first), set(self.draw(strata, epoch=1)))

    def test_annual_quotas_preserve_weighting_and_exact_budget(self):
        strata = [
            {"year": 2023, "start": 0, "count": 1000},
            {"year": 2024, "start": 1000, "count": 1000},
            {"year": 2025, "start": 2000, "count": 1000},
        ]
        quotas = sampling.annual_quotas(strata, 3000, 0.8)
        self.assertEqual([row["samples"] for row in quotas], [787, 984, 1229])
        values = self.draw(quotas)
        self.assertEqual(Counter(v // 1000 for v in values), {0: 787, 1: 984, 2: 1229})
        self.assertEqual(len(set(v for v in values if v < 1000)), 787)
        newest = Counter(v for v in values if v >= 2000)
        self.assertEqual(set(newest), set(range(2000, 3000)))
        self.assertEqual(set(newest.values()), {1, 2})

    def test_oversampling_covers_every_sample_before_additional_cycle(self):
        counts = Counter(self.draw([{"start": 0, "count": 17, "samples": 53}]))
        self.assertEqual(set(counts), set(range(17)))
        self.assertEqual(Counter(counts.values()), {3: 15, 4: 2})

    def test_independent_iterators_have_unique_scratch_and_identical_order(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.draw(), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_early_close_and_budget_errors_release_storage(self):
        stream = self.stream()
        next(stream)
        self.assertEqual(len(list(self.scratch.iterdir())), 1)
        stream.close()
        self.assertEqual(list(self.scratch.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "budget"):
            list(sampling.shuffled_positions(
                [{"start": 0, "count": 100, "samples": 100}], seed=1, epoch=0,
                scratch_root=self.scratch, max_scratch_bytes=1599,
            ))

    def test_invalid_strata_fail_closed(self):
        cases = (
            [{"start": 1, "count": 100, "samples": 1}],
            [{"start": 0, "count": 100, "samples": 0}],
            [{"start": 0, "count": 100, "samples": -1}],
        )
        for strata in cases:
            with self.subTest(strata=strata), self.assertRaises(ValueError):
                self.draw(strata)

    def test_first_position_has_no_fixed_affine_or_stratum_pattern(self):
        def first(seed):
            stream = self.stream([{"start": 0, "count": 8, "samples": 8}], seed=seed)
            try:
                return next(stream)
            finally:
                stream.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            counts = Counter(pool.map(first, range(256)))
        chi_squared = sum((counts[i] - 32) ** 2 / 32 for i in range(8))
        self.assertLess(chi_squared, 30)

    def test_actual_legacy_blocks_vs_sample_random_optimizer_diversity(self):
        tree = ast.parse((ROOT / "src/stock_forecasting/data/dataset.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "BlockwisePermutationSampler")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), node], type_ignores=[])
        namespace = {"Sampler": list, "math": math, "gcd": math.gcd}
        exec(compile(ast.fix_missing_locations(module), "legacy-sampler", "exec"), namespace)
        legacy = namespace["BlockwisePermutationSampler"](65536, block_size=128, seed=42)
        legacy_values = list(legacy)
        random_values = self.draw([{"start": 0, "count": 65536, "samples": 65536}])
        old_groups = [len({v // 128 for v in legacy_values[i:i + 256]})
                      for i in range(0, 4096, 256)]
        new_groups = [len({v // 128 for v in random_values[i:i + 256]})
                      for i in range(0, 4096, 256)]
        self.assertEqual(set(old_groups), {2})
        self.assertGreater(min(new_groups), 180)

    def test_audit_follows_consumed_microbatches_and_bounds_storage(self):
        audit = sampling.SamplingDiversityAudit(max_updates=1)
        for _ in range(4):
            audit.observe(["2025-01-01T16:00:00"] * 2, ["US", "TWSE"])
        audit.observe(["2025-01-02"], ["US"])
        audit.optimizer_step(71)
        row = audit.as_dict()["updates"][0]
        self.assertEqual(row["step"], 71)
        self.assertEqual(row["samples"], 9)
        self.assertEqual(row["unique_dates"], 2)
        self.assertEqual(row["unique_date_markets"], 3)
        self.assertAlmostEqual(row["largest_date_market_share"], 4 / 9)
        self.assertEqual(audit.as_dict()["market_counts"], {"US": 5, "TWSE": 4})
        audit.observe(["2025-02-01"], ["US"])
        audit.optimizer_step(72)
        self.assertEqual(len(audit.records), 1)
        self.assertFalse(audit.pending)


if __name__ == "__main__":
    unittest.main()
