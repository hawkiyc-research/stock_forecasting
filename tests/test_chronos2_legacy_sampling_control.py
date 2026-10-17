"""Standard-library contracts for Chronos-2 on the original lazy sampling pipeline."""

from __future__ import annotations

import ast
import math
import runpy
import sys
import unittest
from itertools import islice
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Missing production definitions")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def section(text, name, following):
    return text.split("\n" + name + ":\n", 1)[1].split("\n" + following + ":\n", 1)[0]


class GenericSampler:
    def __class_getitem__(cls, _item):
        return cls


class ReadOnlyOrdinalIndex(dict):
    def __init__(self, count):
        super().__init__(ordinal=range(count))
        self.closed = False
        self._mmap = SimpleNamespace(close=self.close)

    def close(self):
        self.closed = True


class Chronos2LegacySamplingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
        cls.contract = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
        arguments = cls.selection["build_parser"]().parse_args([
            "create", "--project-root", str(ROOT), "--stage", "stage2",
            "--data-profile", "us_tw_eodhd", "--start", "2021-01-01",
            "--end", "2026-06-01", "--universe", "all",
        ])
        cls.base = cls.selection["_build_selection"](arguments, ROOT)

    def test_original_selector_uses_date_market_sampler_for_both_signal_tasks(self):
        class BlockRecorder:
            def __init__(self, source, **kwargs):
                self.source, self.kwargs = source, kwargs

        class DateRecorder(BlockRecorder):
            pass

        class SizeOnlySource:
            def __len__(self):
                return 4710117

            def __getitem__(self, _index):
                raise AssertionError("Sampler selection must not materialize windows")

        training_path = ROOT / "src/stock_forecasting/training.py"
        tree = ast.parse(training_path.read_text())
        constant = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == "DATALOADER_SELECTION_BLOCK_SIZE"
                                for target in node.targets))
        namespace = definitions(
            training_path, {"_training_sampler"},
            {"BlockwisePermutationSampler": BlockRecorder,
             "DATALOADER_SELECTION_BLOCK_SIZE": ast.literal_eval(constant)},
        )
        package = ModuleType("stock_forecasting")
        package.__path__ = []
        module = ModuleType("stock_forecasting.date_market_sampler")
        module.DateMarketSampler = DateRecorder
        source = SizeOnlySource()
        with patch.dict(sys.modules, {"stock_forecasting": package,
                                     "stock_forecasting.date_market_sampler": module}):
            for task in ("alpha_point", "absolute_direction"):
                config = SimpleNamespace(
                    model=SimpleNamespace(prediction_task=task, time_series_backend="chronos2",
                                          ranking_loss_weight=0),
                    training=SimpleNamespace(yearly_sampling_decay=0.8, seed=42),
                    data=SimpleNamespace(train_fraction=1.0, max_samples=None),
                )
                worker_plan = SimpleNamespace(effective_workers=2)
                sampler = namespace["_training_sampler"](config, source, worker_plan)
                self.assertIs(type(sampler), DateRecorder)
                self.assertIs(sampler.source, source)
                self.assertEqual(sampler.kwargs["block_size"], 128)
                self.assertEqual(sampler.kwargs["yearly_decay"], 0.8)
                probe = namespace["_training_sampler"](config, source, worker_plan, False)
                self.assertIs(type(probe), BlockRecorder)

    def test_real_annual_sampler_still_emits_contiguous_128_sample_blocks(self):
        namespace = definitions(
            ROOT / "src/stock_forecasting/data/dataset.py", {"BlockwisePermutationSampler"},
            {"Sampler": GenericSampler, "math": math, "gcd": math.gcd},
        )
        index = ReadOnlyOrdinalIndex(65536)
        namespace["np"] = SimpleNamespace(load=lambda *a, **kw: index)
        definitions(ROOT / "src/stock_forecasting/date_market_sampler.py",
                    {"DateMarketSampler"}, namespace)
        sampler_type = namespace["DateMarketSampler"]
        sampler = sampler_type.__new__(sampler_type)
        namespace["BlockwisePermutationSampler"].__init__(sampler, 65536, seed=42, block_size=128)
        sampler.index_path, sampler.yearly_decay = "fixture.npy", 0.8
        sampler.year_quotas = lambda _: [
            {"year": 2024, "start": 0, "count": 32768, "samples": 32768},
            {"year": 2025, "start": 32768, "count": 32768, "samples": 32768},
        ]
        iterator = iter(sampler)
        try:
            values = list(islice(iterator, 256))
        finally:
            iterator.close()
        self.assertTrue(index.closed)
        self.assertEqual(len({value // 128 for value in values}), 2)
        for offset in (0, 128):
            chunk = values[offset:offset + 128]
            self.assertEqual(len({b - a for a, b in zip(chunk, chunk[1:])}), 1)
            self.assertIn(chunk[1] - chunk[0], (-1, 1))

    def test_four_presets_preserve_data_optimizer_labels_and_evaluation(self):
        for group in ("a", "b"):
            for task in ("alpha_point", "absolute_direction"):
                original = (ROOT / f"configs/experiments/{group}_{task}64.yaml").read_text()
                chronos = (ROOT / f"configs/experiments/{group}_chronos2_{task}64.yaml").read_text()
                for name, following in (("data", "model"), ("training", "wandb"),
                                        ("validation", "runtime")):
                    self.assertEqual(section(original, name, following),
                                     section(chronos, name, following))
                def head_contract(text):
                    return [line for line in section(text, "model", "training").splitlines()
                            if not line.startswith(("  time_series_", "  kronos_",
                                                    "    target_modules:"))]
                self.assertEqual(head_contract(original), head_contract(chronos))
                self.assertIn("  yearly_sampling_decay: 0.8", chronos)
                self.assertNotIn("sampling_strategy", chronos)
                self.assertNotIn("sample-random", chronos)

    def test_new_experiments_preserve_preparation_and_baseline_identities(self):
        for group in ("a", "b"):
            for task in ("alpha-point", "absolute-direction"):
                old = self.selection["_with_experiment"](self.base, f"{group}-{task}64", ROOT)
                name = f"{group}-chronos2-{task}64"
                new = self.selection["_with_experiment"](self.base, name, ROOT)
                self.assertEqual(new["baseline_task"], old["baseline_task"])
                self.assertEqual(new["dataset_request_sha256"], old["dataset_request_sha256"])
                self.assertNotEqual(new["stage"]["config_sha256"], old["stage"]["config_sha256"])
                self.assertEqual(new["stage"]["config_path"],
                                 "configs/experiments/" + name.replace("-", "_") + ".yaml")
                self.assertEqual(self.contract["baseline_contract"](ROOT, new),
                                 self.contract["baseline_contract"](ROOT, old))

    def test_seed_override_preserves_data_identity(self):
        original = self.selection["_with_experiment"](
            self.base, "a-chronos2-alpha-point64", ROOT
        )
        seeded = self.selection["_with_training_seed"](original, 347, ROOT)
        self.assertEqual(seeded["stage"]["training_seed"], 347)
        self.assertEqual(seeded["dataset_request_sha256"], original["dataset_request_sha256"])
        self.assertNotEqual(seeded["selection_sha256"], original["selection_sha256"])

    def test_random_sampling_is_absent_from_live_configuration_and_registry(self):
        tree = ast.parse((ROOT / "src/stock_forecasting/config.py").read_text())
        config = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "TrainingConfig")
        fields = {node.target.id for node in config.body if isinstance(node, ast.AnnAssign)}
        self.assertFalse({"sampling_strategy", "sampling_scratch_max_gib",
                          "sampling_audit_updates"} & fields)
        self.assertFalse(any("-random-" in name for name in self.selection["EXPERIMENT_CONFIGS"]))
        self.assertFalse((ROOT / "src/stock_forecasting/random_training_sampler.py").exists())


if __name__ == "__main__":
    unittest.main()
