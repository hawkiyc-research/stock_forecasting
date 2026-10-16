"""Dependency-free controls for date-mixed training batches and their paired presets."""

from __future__ import annotations

import argparse
import ast
import random
import re
import runpy
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLER_SOURCE = ROOT / "src/stock_forecasting/date_market_sampler.py"
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
PAIRED = runpy.run_path(str(ROOT / "scripts/runpod_paired.py"))
PAIRED_CONTROL = runpy.run_path(str(ROOT / "scripts/runpod_paired_control.py"))
BASELINE = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
DATEMIX_PAIRS = ("a-signal64-datemix", "b-signal64-datemix")
BUFFER = 16384


def sampler_helpers() -> dict:
    """Execute only the stdlib helpers, never the numpy/torch sampler module."""
    tree = ast.parse(SAMPLER_SOURCE.read_text())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in ("bounded_local_shuffle", "batch_composition_summary")
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MARKET_CODE_RADIX"
                for target in node.targets
            )
        )
    ]
    namespace = {"statistics": statistics}
    code = compile(ast.Module(body=selected, type_ignores=[]), str(SAMPLER_SOURCE), "exec")
    exec(code, namespace)
    return namespace


def same_date_blocks(groups: int, block: int = 128) -> list[int]:
    """Mimic historical blocks: each consecutive block is one date/market group."""
    radix = sampler_helpers()["MARKET_CODE_RADIX"]
    return [
        (20_000 + group // 3) * radix + group % 3
        for group in range(groups)
        for _ in range(block)
    ]


class BoundedLocalShuffleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = sampler_helpers()
        cls.shuffle = staticmethod(helpers["bounded_local_shuffle"])
        cls.summary = staticmethod(helpers["batch_composition_summary"])
        cls.radix = helpers["MARKET_CODE_RADIX"]

    def test_preserves_the_exact_multiset_for_every_stream_length(self):
        for length in (0, 1, 7, 64, 65, 1000):
            items = [value % 13 for value in range(length)]
            emitted = list(self.shuffle(items, 64, random.Random(5)))
            with self.subTest(length=length):
                self.assertEqual(sorted(emitted), sorted(items))

    def test_lookahead_is_bounded_by_capacity(self):
        capacity, length = 32, 5000
        emitted = list(self.shuffle(range(length), capacity, random.Random(11)))
        # Before the final flush, output k can only come from inputs < k + capacity.
        for position, value in enumerate(emitted[: length - capacity]):
            self.assertLess(value, position + capacity)

    def test_equal_length_streams_receive_the_same_permutation(self):
        ordinals = list(range(3000))
        labels = [f"group-{value}" for value in ordinals]
        first = list(self.shuffle(ordinals, 256, random.Random("seed:0")))
        second = list(self.shuffle(labels, 256, random.Random("seed:0")))
        self.assertEqual([f"group-{value}" for value in first], second)
        self.assertNotEqual(first, ordinals)
        self.assertNotEqual(first, list(self.shuffle(ordinals, 256, random.Random("seed:1"))))

    def test_rejects_unbounded_or_degenerate_capacity(self):
        for capacity in (None, True, 1, 0, -4, 2.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                list(self.shuffle(range(10), capacity, random.Random(0)))

    def test_composition_counts_dates_and_markets_per_complete_batch(self):
        historical = same_date_blocks(600)
        before = self.summary(historical, 256)
        self.assertEqual(before["batches"], len(historical) // 256)
        self.assertEqual(before["distinct_date_markets"]["max"], 2)
        mixed = list(self.shuffle(historical, BUFFER, random.Random(42)))
        after = self.summary(mixed, 256)
        self.assertEqual(after["batches"], before["batches"])
        self.assertGreater(after["distinct_date_markets"]["median"], 100)
        self.assertLessEqual(
            after["distinct_dates"]["max"], after["distinct_date_markets"]["max"]
        )
        self.assertEqual(self.summary(historical[:10], 256), {"batch_size": 256, "batches": 0})
        with self.assertRaises(ValueError):
            self.summary(historical, 0)


class DateMixingPresetTests(unittest.TestCase):
    def test_paired_registries_agree_on_every_entry_point(self):
        expected = set(PAIRED["PAIR_EXPERIMENTS"])
        self.assertTrue(set(DATEMIX_PAIRS) <= expected)
        self.assertEqual(set(PAIRED_CONTROL["PAIRED_EXPERIMENTS"]), expected)
        concurrency = (ROOT / "scripts/runpod_concurrency.py").read_text()
        literal = re.search(r"paired_experiments = (\{[^}]+\})", concurrency)
        self.assertIsNotNone(literal)
        self.assertEqual(ast.literal_eval(literal.group(1)), expected)
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text()
        pattern = "|".join(PAIRED["PAIR_EXPERIMENTS"]) + ")"
        self.assertEqual(workflow.count(pattern), 2)

    def test_every_paired_child_has_a_registered_task_preset(self):
        for experiment in PAIRED["PAIR_EXPERIMENTS"]:
            for task in PAIRED["TASKS"]:
                child = PAIRED["child_experiment"](experiment, task)
                with self.subTest(experiment=experiment, task=task):
                    self.assertEqual(child[0], experiment[0])
                    self.assertIn(child, SELECTION["EXPERIMENT_CONFIGS"])
                    self.assertEqual(
                        SELECTION["SIGNAL_TASK_VARIANTS"][child.partition("-")[2]], task
                    )
                    path = ROOT / SELECTION["EXPERIMENT_CONFIGS"][child]["config_path"]
                    self.assertTrue(path.is_file())
        self.assertEqual(
            PAIRED["child_experiment"]("a-signal64", "alpha_point"), "a-alpha-point64"
        )
        self.assertEqual(
            PAIRED["child_experiment"]("b-signal64-datemix", "absolute_direction"),
            "b-absolute-direction64-datemix",
        )
        with self.assertRaises(ValueError):
            PAIRED["child_experiment"]("a-signal64-unknown", "alpha_point")

    def test_datemix_yaml_differs_from_its_base_only_by_names_and_mixing(self):
        for group in ("a", "b"):
            for variant in ("alpha_point64", "absolute_direction64"):
                base = (ROOT / f"configs/experiments/{group}_{variant}.yaml").read_text()
                mixed = (ROOT / f"configs/experiments/{group}_{variant}_datemix.yaml").read_text()
                name = f"{group}-{variant.replace('_', '-')}"
                expected = base
                for old, new in (
                    (f"experiment_name: {name}\n", f"experiment_name: {name}-datemix\n"),
                    (f"  name: {name}\n", f"  name: {name}-datemix\n"),
                    (
                        "  yearly_sampling_decay: 0.8\n",
                        f"  yearly_sampling_decay: 0.8\n  date_mixing_buffer_samples: {BUFFER}\n",
                    ),
                    (
                        "  tags: [quant-only, kronos-base, lora, stage2]\n",
                        "  tags: [quant-only, kronos-base, lora, stage2, date-mixing]\n",
                    ),
                ):
                    self.assertEqual(expected.count(old), 1)
                    expected = expected.replace(old, new)
                description = re.compile(r"(?m)^description: (.+)$")
                expected = description.sub(
                    lambda match: f"description: {match.group(1)[:-1]} and date-mixed "
                    "training batches.",
                    expected,
                    count=1,
                )
                with self.subTest(group=group, variant=variant):
                    self.assertEqual(mixed, expected)
                    self.assertIn("  ranking_loss_weight: 0.0\n", mixed)

    def test_historical_training_order_remains_the_default(self):
        tree = ast.parse((ROOT / "src/stock_forecasting/config.py").read_text())
        training = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig"
        )
        field = next(
            node
            for node in training.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "date_mixing_buffer_samples"
        )
        default = next(k.value for k in field.value.keywords if k.arg == "default")
        self.assertIsNone(ast.literal_eval(default))
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
            with self.subTest(path=path.name):
                mixed = "date_mixing_buffer_samples" in path.read_text()
                self.assertEqual(mixed, path.stem.endswith("_datemix"))

    def test_sampler_is_a_training_identity_but_not_a_baseline_identity(self):
        run_contract = ast.parse((ROOT / "src/stock_forecasting/run_contract.py").read_text())
        paths = next(
            ast.literal_eval(node.value)
            for node in run_contract.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TRAINING_IMPLEMENTATION_PATHS"
                for target in node.targets
            )
        )
        self.assertIn("date_market_sampler.py", paths)
        for source in ("date_market_sampler.py", "config.py"):
            self.assertNotIn(f"src/stock_forecasting/{source}", BASELINE["BASELINE_SOURCES"])

    def test_datemix_children_reuse_the_existing_task_baselines(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            shutil.copytree(ROOT / "configs", project / "configs")
            arguments = argparse.Namespace(
                stage="stage2",
                data_profile="us_tw_eodhd",
                dataset_revision="v1",
                start="2016-01-01",
                end="2026-06-01",
                h_start=1,
                universe="all",
                stocks=[],
                etfs=[],
                symbol_limit=None,
                feature_mode="combined",
            )
            original = SELECTION["_build_selection"](arguments, project)
            for experiment in DATEMIX_PAIRS:
                for task in PAIRED["TASKS"]:
                    mixed = PAIRED["child_experiment"](experiment, task)
                    base = PAIRED["child_experiment"](experiment.removesuffix("-datemix"), task)
                    selected = {
                        name: SELECTION["_with_experiment"](original, name, project)
                        for name in (base, mixed)
                    }
                    with self.subTest(experiment=experiment, task=task):
                        self.assertEqual(
                            selected[base]["baseline_task"], selected[mixed]["baseline_task"]
                        )
                        self.assertEqual(
                            selected[base]["dataset_request_sha256"],
                            selected[mixed]["dataset_request_sha256"],
                        )
                        self.assertEqual(
                            BASELINE["baseline_contract"](ROOT, selected[base])["baseline_id"],
                            BASELINE["baseline_contract"](ROOT, selected[mixed])["baseline_id"],
                        )
                        self.assertNotEqual(
                            selected[base]["stage"]["config_sha256"],
                            selected[mixed]["stage"]["config_sha256"],
                        )


if __name__ == "__main__":
    unittest.main()
