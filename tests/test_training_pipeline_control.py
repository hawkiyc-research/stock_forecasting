"""Dependency-free execution/control regressions; no local ML environment required."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import runpy
import time
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "src/stock_forecasting/training.py"


def definitions(path, names, namespace):
    nodes = ast.parse(path.read_text()).body
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("Missing production definitions")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class GenericSampler:
    def __class_getitem__(cls, _item):
        return cls


def samplers():
    scope = {"Sampler": GenericSampler, "math": math, "gcd": math.gcd, "Iterator": Iterator}
    definitions(ROOT / "src/stock_forecasting/data/dataset.py", {
        "BlockwisePermutationSampler", "FixedSizeBatchSampler",
    }, scope)
    scope["islice"] = islice
    return definitions(TRAINING, {"_ProbeBatchSampler"}, scope)


class TrainingPipelineControlTests(unittest.TestCase):
    def test_hotpath_migration_covers_only_the_verified_model_execution_patch(self):
        registry = runpy.run_path(str(
            ROOT / "src/stock_forecasting/checkpoint_resume_migrations.py"
        ))["CHECKPOINT_RETENTION_MIGRATIONS"]
        migration = next(row for row in registry if row["id"] == "asynchronous-model-hotpath-v1")
        self.assertEqual(set(migration["from_files"]), {
            "training.py", "models/forecast.py", "models/quant.py", "models/ranking.py",
            "models/scale_features.py",
        })
        self.assertEqual(migration["from_files"].keys(), migration["to_files"].keys())
        stop_migration = next(
            row for row in registry if row["id"] == "saved-boundary-runtime-stop-v2"
        )
        for name, digest in migration["to_files"].items():
            if name == "training.py":
                # The historical execution patch precedes the saved-boundary stop patch.
                self.assertEqual(digest, stop_migration["from_files"][name])
                digest = stop_migration["to_files"][name]
            self.assertEqual(digest, hashlib.sha256(
                (ROOT / "src/stock_forecasting" / name).read_bytes()
            ).hexdigest(), name)

    def test_cpu_pressure_accounting_supports_cgroup_v1_v2_and_missing_stats(self):
        mapping = {}

        class StatPath:
            def __init__(self, name):
                self.name = name

            def read_text(self):
                if self.name not in mapping:
                    raise FileNotFoundError(self.name)
                return mapping[self.name]

        scope = {"Path": StatPath}
        definitions(TRAINING, {"_cpu_pressure_snapshot"}, scope)
        snapshot = scope["_cpu_pressure_snapshot"]
        self.assertIsNone(snapshot())
        mapping["/sys/fs/cgroup/cpu.stat"] = "usage_usec 3500000\nnr_periods 30\nnr_throttled 1\n"
        self.assertEqual(snapshot(), (3.5, 30, 1))
        mapping.clear()
        mapping["/sys/fs/cgroup/cpu/cpu.stat"] = "nr_periods 30\nnr_throttled 1\n"
        mapping["/sys/fs/cgroup/cpuacct/cpuacct.usage"] = "3500000000"
        self.assertEqual(snapshot(), (3.5, 30, 1))

    def test_io_worker_ceiling_stays_within_memory_cpu_affinity_and_requested_limits(self):
        scope = {
            "os": SimpleNamespace(sched_getaffinity=lambda _pid: range(64)),
            "DATALOADER_MEMORY_BYTES_PER_WORKER": 512 * 1024**2,
            "DATALOADER_MAX_IO_WORKER_MULTIPLIER": 2,
        }
        definitions(TRAINING, {"_io_worker_ceiling", "_probe_cpu_pressure"}, scope)
        workers = SimpleNamespace(
            requested_workers=128, effective_workers=11, visible_cpu_count=13,
            active_persistent_pools=2, worker_memory_budget_bytes=32 * 1024**3,
        )
        self.assertEqual(scope["_io_worker_ceiling"](workers), 22)
        workers.worker_memory_budget_bytes = 12 * 1024**3
        self.assertEqual(scope["_io_worker_ceiling"](workers), 12)
        workers.requested_workers = 8
        self.assertEqual(scope["_io_worker_ceiling"](workers), 8)
        pressure = scope["_probe_cpu_pressure"]((10, 20, 1), (75, 120, 3), 10, 13)
        self.assertEqual(pressure, {"cpu_fraction": .5, "throttled_period_fraction": .02})
        self.assertIsNone(scope["_probe_cpu_pressure"](None, (75, 120, 3), 10, 13))
        self.assertIsNone(scope["_probe_cpu_pressure"]((10, 20, 1), (75, 20, 3), 10, 13))

    def test_resume_accepts_only_exact_execution_patch_and_unchanged_numerical_contract(self):
        registry = runpy.run_path(str(
            ROOT / "src/stock_forecasting/checkpoint_resume_migrations.py"
        ))["CHECKPOINT_RETENTION_MIGRATIONS"]
        migration = next(row for row in registry if row["id"] == "saved-boundary-runtime-stop-v2")
        self.assertEqual(set(migration["from_files"]), {"training.py"})
        self.assertEqual(migration["from_files"].keys(), migration["to_files"].keys())
        self.assertEqual(migration["to_files"]["training.py"],
                         hashlib.sha256(TRAINING.read_bytes()).hexdigest())
        scope = {"json": json, "hashlib": hashlib, "CHECKPOINT_RETENTION_MIGRATIONS": registry}
        contract_path = ROOT / "src/stock_forecasting/run_contract.py"
        tree = ast.parse(contract_path.read_text())
        scope["TRAINING_IMPLEMENTATION_PATHS"] = ast.literal_eval(next(
            node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "TRAINING_IMPLEMENTATION_PATHS"
                    for target in node.targets)
        ))
        definitions(contract_path, {
            "_canonical_payload_digest", "_validated_implementation_files",
            "_checkpoint_retention_migration_matches", "compatible_training_resume_contract_digest",
        }, scope)
        digest = scope["_canonical_payload_digest"]
        files = {name: hashlib.sha256(
            (ROOT / "src/stock_forecasting" / name).read_bytes()
        ).hexdigest() for name in scope["TRAINING_IMPLEMENTATION_PATHS"]}
        current = {"model": "unchanged", "data": "unchanged", "training_implementation": {
            "files": files, "sha256": digest(files),
        }}
        stored = copy.deepcopy(current)
        stored["training_implementation"]["files"].update(migration["from_files"])
        stored["training_implementation"]["sha256"] = digest(
            stored["training_implementation"]["files"]
        )
        matches = scope["_checkpoint_retention_migration_matches"]
        self.assertTrue(matches(stored, current))
        self.assertFalse(matches(stored, {**current, "data": "different"}))
        self.assertFalse(matches(stored, {**current, "model": "different"}))
        changed = copy.deepcopy(current)
        changed["training_implementation"]["files"]["models/quant.py"] = "0" * 64
        changed["training_implementation"]["sha256"] = digest(
            changed["training_implementation"]["files"]
        )
        self.assertFalse(matches(stored, changed))
        scope["training_resume_contract"] = lambda _config: current
        compatible = scope["compatible_training_resume_contract_digest"]
        manifest = {"training_resume_contract": stored,
                    "training_resume_contract_sha256": digest(stored)}
        self.assertEqual(compatible(None, manifest), digest(stored))
        prior_stop = next(row for row in registry
                          if row["id"] == "completion-before-runtime-stop-v1")
        self.assertEqual(set(prior_stop["from_files"]), {"training.py"})
        self.assertEqual(prior_stop["to_files"], migration["to_files"])
        prior_stored = copy.deepcopy(stored)
        prior_stored["training_implementation"]["files"].update(prior_stop["from_files"])
        prior_stored["training_implementation"]["sha256"] = digest(
            prior_stored["training_implementation"]["files"]
        )
        self.assertEqual(compatible(None, {
            "training_resume_contract": prior_stored,
            "training_resume_contract_sha256": digest(prior_stored),
        }), digest(prior_stored))
        for key in ("data", "model", "optimizer", "training"):
            incompatible = copy.deepcopy(stored)
            incompatible[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                compatible(None, {"training_resume_contract": incompatible,
                                  "training_resume_contract_sha256": digest(incompatible)})
        self.assertFalse(matches(current, stored))
        arbitrary = copy.deepcopy(stored)
        arbitrary["training_implementation"]["files"]["training.py"] = "0" * 64
        arbitrary["training_implementation"]["sha256"] = digest(
            arbitrary["training_implementation"]["files"]
        )
        self.assertFalse(matches(arbitrary, current))
        with self.assertRaises(ValueError):
            compatible(None, {**manifest, "training_resume_contract_sha256": "0" * 64})
        corrupt = copy.deepcopy(stored)
        corrupt["training_implementation"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            matches(corrupt, current)

    def test_old_two_iteration_plan_is_reprobed_without_resetting_training(self):
        hardware = SimpleNamespace(
            capacity_identity=lambda: "unchanged", device_type="cuda",
            device_name="fixture", device_total_memory_bytes=1000,
        )
        workers = SimpleNamespace(
            requested_workers=8, effective_workers=8,
            available_memory_bytes=10000, estimated_peak_prefetch_memory_bytes=100,
        )
        batch = SimpleNamespace(
            source="cuda_empirical", device_name="fixture", device_total_memory_bytes=1000,
        )
        stored = SimpleNamespace(hardware=hardware, worker_plan=workers, batch_plan=batch)
        scope = {"DATALOADER_PREFETCH_MEMORY_FRACTION": .10,
                 "CUDA_PIPELINE_PROBE_SOURCE": "cuda_pipeline_v3"}
        definitions(TRAINING, {"runtime_resource_plan_reuse_reason"}, scope)
        reason = scope["runtime_resource_plan_reuse_reason"]
        self.assertEqual(reason(stored, current_hardware=hardware, current_worker_plan=workers),
                         "pipeline_probe_version_changed")
        batch.source = "cuda_pipeline_v2"
        self.assertEqual(reason(stored, current_hardware=hardware, current_worker_plan=workers),
                         "pipeline_probe_version_changed")
        batch.source = "cuda_pipeline_v3"
        self.assertEqual(reason(stored, current_hardware=hardware, current_worker_plan=workers),
                         "checkpoint_hardware_match")

    def test_batch_search_passes_real_dataset_and_stops_after_oom(self):
        dataset = range(1024)
        modes = [SimpleNamespace(training=True), SimpleNamespace(training=False)]
        calls = []

        def measure(**kwargs):
            calls.append(kwargs["batch_size"])
            self.assertIs(kwargs["dataset"], dataset)
            self.assertIs(kwargs["worker_plan"], worker_plan)
            return SimpleNamespace(
                accepted=kwargs["batch_size"] < 16, batch_size=kwargs["batch_size"],
                samples_per_second=100.0,
                outcome="accepted" if kwargs["batch_size"] < 16 else "cuda_out_of_memory",
            )

        worker_plan = object()
        rng_events = []
        scope = {
            "torch": SimpleNamespace(
                get_rng_state=lambda: "cpu", set_rng_state=lambda value: rng_events.append(value),
                cuda=SimpleNamespace(
                    get_rng_state_all=lambda: "cuda",
                    set_rng_state_all=lambda value: rng_events.append(value),
                ),
            ),
            "_measure_cuda_batch": measure, "cast": cast,
            "_training_sampler": lambda *_args: dataset,
            "AUTO_BATCH_THROUGHPUT_TOLERANCE": .03,
        }
        definitions(TRAINING, {"_probe_cuda_candidates", "_select_batch_measurement"}, scope)
        best, results = scope["_probe_cuda_candidates"](
            bundle=SimpleNamespace(model=SimpleNamespace(modules=lambda: modes)),
            sample=None, candidates=(4, 8, 16, 32), config=None, device=None,
            training=True, optimizer_state_reserve_bytes=0, device_memory_limit_bytes=1000,
            dataset=dataset, worker_plan=worker_plan,
        )
        self.assertEqual(calls, [4, 8, 16])
        self.assertEqual(best.batch_size, 4)
        self.assertEqual(len(results), 3)
        self.assertEqual([module.training for module in modes], [True, False])
        self.assertEqual(rng_events, ["cpu", "cuda"])

    def test_ready_batch_is_yielded_before_requesting_another_batch(self):
        events = []

        class Stream:
            def wait_stream(self, _stream):
                events.append("wait")

        cuda = SimpleNamespace(
            Stream=lambda **_kw: Stream(), stream=lambda _stream: contextlib.nullcontext(),
            current_stream=lambda _device: Stream(),
        )
        scope = {
            "torch": SimpleNamespace(cuda=cuda), "time": time,
            "_move_batch_to_device": lambda batch, _device: batch,
            "_record_batch_stream": lambda batch, _stream: events.append(("record", batch)),
        }
        definitions(TRAINING, {"iter_device_batches"}, scope)

        def loader():
            events.append("fetch-0")
            yield 0
            self.assertIn("compute-0", events)
            events.append("fetch-1")
            yield 1

        timings = {}
        batches = scope["iter_device_batches"](loader(), SimpleNamespace(type="cuda"), timings)
        self.assertEqual(next(batches), 0)
        self.assertNotIn("fetch-1", events)
        self.assertIn(("record", 0), events)
        events.append("compute-0")
        self.assertEqual(list(batches), [1])
        self.assertGreaterEqual(timings["host_fetch_enqueue_seconds"], 0)

    def test_probe_is_bounded_and_does_not_consume_production_sampler(self):
        scope = samplers()
        dynamic = scope["BlockwisePermutationSampler"](100_000, seed=73, block_size=128)
        sampler = scope["_ProbeBatchSampler"](dynamic, 16, 9)
        first = list(sampler)
        self.assertEqual(len(first), 9)
        self.assertEqual(len({i for batch in first for i in batch}), 144)
        self.assertEqual(first, list(sampler))
        dynamic.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(len(dynamic), 100_000)
        tiny = scope["_ProbeBatchSampler"](
            scope["BlockwisePermutationSampler"](16, seed=73), 4, 8
        )
        rows = list(tiny)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[:4], rows[4:])

    def test_worker_search_honors_cpu_memory_bound_and_manual_worker_selection(self):
        @dataclass
        class Workers:
            effective_workers: int = 8
            prefetch_factor: int = 4
            prefetched_batches_per_pool: int = 32
            estimated_peak_prefetch_memory_bytes: int = 3200
            prefetch_memory_budget_bytes: int = 3200
            source: str = "test"

        @dataclass
        class Batch:
            training_batch_size: int = 32
            device_total_memory_bytes: int = 10000
            optimizer_state_reserve_bytes: int = 10
            seconds_per_training_batch: float = 0.2
            training_probe: tuple = ()

        visited = []

        def measure(**kwargs):
            workers = kwargs["worker_plan"]
            self.assertLessEqual(workers.estimated_peak_prefetch_memory_bytes, 3200)
            self.assertLessEqual(workers.effective_workers, 8)
            self.assertEqual(kwargs["probe_prefetch_factor"], workers.prefetch_factor)
            visited.append((workers.effective_workers, workers.prefetch_factor))
            return SimpleNamespace(
                accepted=True, samples_per_second=100.0, seconds_per_batch=0.32, batch_size=32
            )

        config = SimpleNamespace(training=SimpleNamespace(auto_batch_memory_fraction=.72))
        scope = {
            "math": math, "os": os, "json": json, "replace": replace,
            "CUDA_PIPELINE_PROBE_SOURCE": "cuda_pipeline_v3",
            "CUDA_FREE_MEMORY_FRACTION": .9, "AUTO_BATCH_THROUGHPUT_TOLERANCE": .03,
            "_measure_cuda_batch": measure,
            "_requested_dataloader_workers": lambda _config: (128, "training_config_auto"),
            "_cpu_pressure_snapshot": lambda: None,
            "_confirm_training_batch_plan": lambda _config, **kwargs: (
                kwargs["workers"], kwargs["batch_plan"]
            ),
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                mem_get_info=lambda _device: (9000, 10000), memory_allocated=lambda _device: 1000,
            )),
        }
        definitions(TRAINING, {"refine_training_pipeline_plan"}, scope)
        kwargs = dict(bundle=None, dataset=None, sample=None, device=SimpleNamespace(type="cuda"),
                      worker_plan=Workers(), batch_plan=Batch())
        with contextlib.redirect_stdout(io.StringIO()):
            workers, _batch = scope["refine_training_pipeline_plan"](config, **kwargs)
        self.assertEqual(visited, [(4, 2), (4, 4), (8, 2), (8, 4)])
        self.assertEqual((workers.effective_workers, workers.prefetch_factor), (4, 2))
        visited.clear()
        scope["_requested_dataloader_workers"] = lambda _config: (8, "environment")
        with contextlib.redirect_stdout(io.StringIO()):
            scope["refine_training_pipeline_plan"](config, **kwargs)
        self.assertEqual(visited, [(8, 2), (8, 4)])

    def test_batch_confirmation_rechecks_order_and_preserves_effective_batch_and_memory(self):
        @dataclass
        class Measurement:
            batch_size: int
            samples_per_second: float
            seconds_per_batch: float
            peak_allocated_bytes: int = 100
            projected_peak_bytes: int = 100
            accepted: bool = True

            def as_dict(self):
                return self.__dict__

        @dataclass
        class Plan:
            training_batch_size: int = 32
            evaluation_batch_size: int = 128
            gradient_accumulation_steps: int = 8
            effective_batch_size: int = 256
            optimizer_state_reserve_bytes: int = 10
            seconds_per_training_batch: float = .32
            training_probe: tuple = (
                Measurement(32, 100, .32), Measurement(64, 99, 64 / 99),
                Measurement(128, 70, 128 / 70),
            )

        @dataclass
        class Workers:
            effective_workers: int = 4
            prefetch_factor: int = 2
            prefetched_batches_per_pool: int = 8
            estimated_peak_prefetch_memory_bytes: int = 800

        visited = []
        fail_size = None

        def measure(**kwargs):
            size = kwargs["batch_size"]
            visited.append(size)
            speed = {32: 100, 64: 101, 128: 70}[size]
            return Measurement(size, speed, size / speed, accepted=size != fail_size)

        config = SimpleNamespace(training=SimpleNamespace(
            batch_size="auto", gradient_accumulation_steps="auto", target_effective_batch_size=256,
        ))
        scope = {
            "replace": replace, "json": json, "CUDA_PIPELINE_BATCH_CONFIRMATIONS": 2,
            "CUDA_PIPELINE_BATCH_FINALISTS": 3, "AUTO_BATCH_THROUGHPUT_TOLERANCE": .03,
            "_measure_cuda_batch": measure,
            "ModelBatchCollator": SimpleNamespace(for_model=lambda *_args, **_kw: SimpleNamespace(
                estimated_batch_bytes=lambda _sample, size: size * 100,
            )),
            "plan_runtime_prefetch": lambda workers, **kwargs: replace(
                workers, prefetch_factor=4, prefetched_batches_per_pool=16,
                estimated_peak_prefetch_memory_bytes=1600,
            ),
        }
        definitions(TRAINING, {
            "_confirm_training_batch_plan", "_automatic_gradient_accumulation_steps",
        }, scope)
        confirm = scope["_confirm_training_batch_plan"]
        kwargs = dict(bundle=None, dataset=None, sample=None, device=None,
                      workers=Workers(), batch_plan=Plan(), memory_limit=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            workers, plan = confirm(config, **kwargs)
        self.assertEqual(visited, [32, 64, 128, 128, 64, 32])
        self.assertEqual((plan.training_batch_size, plan.gradient_accumulation_steps), (64, 4))
        self.assertEqual(plan.effective_batch_size, 256)
        self.assertEqual(workers.prefetch_factor, 2)
        self.assertEqual(workers.estimated_peak_prefetch_memory_bytes, 800)
        visited.clear()
        fail_size = 64
        with contextlib.redirect_stdout(io.StringIO()):
            _, plan = confirm(config, **kwargs)
        self.assertEqual(visited.count(64), 1)
        self.assertEqual(plan.training_batch_size, 32)
        visited.clear()
        config.training.batch_size = 32
        self.assertEqual(confirm(config, **kwargs), (kwargs["workers"], kwargs["batch_plan"]))
        self.assertEqual(visited, [])

    def test_production_and_probe_share_dynamic_selection_but_evaluation_remains_full(self):
        tree = ast.parse(TRAINING.read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        loader = ast.unparse(functions["build_dataloaders"])
        probe = ast.unparse(functions["_measure_cuda_batch"])
        self.assertIn("_training_sampler(", loader)
        self.assertIn("_training_sampler(", probe)
        self.assertIn("evaluation_sampler(len(validation_dataset)", loader)
        self.assertIn("evaluation_sampler(len(test_dataset)", loader)
        self.assertEqual(loader.count("drop_last=False"), 2)
        self.assertIn("_preserve_probe_state(bundle.model)", probe)
        self.assertIn("optimizer.step()", probe)
        self.assertIn("loss.backward()", probe)
        self.assertNotIn("[sample] * batch_size", probe)
        self.assertIn("_shutdown_workers()", probe)
        transfer = ast.unparse(functions["_move_batch_to_device"])
        self.assertIn("non_blocking=non_blocking", transfer)
        self.assertIn("'ranking_group_ids'", transfer)


if __name__ == "__main__":
    unittest.main()
