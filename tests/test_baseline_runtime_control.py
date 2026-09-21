"""Dependency-free quota and batching contracts runnable on the control host."""

from __future__ import annotations

import ast
import json
import math
import runpy
import tempfile
import unittest
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = runpy.run_path(str(ROOT / "src/stock_forecasting/runtime_resources.py"))


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RuntimeTests(unittest.TestCase):
    def test_loader_shutdown_drains_only_bounded_pending_work_and_propagates_errors(self):
        close = definitions(
            "src/stock_forecasting/baseline_input.py", {"close_neural_loader"}, {}
        )["close_neural_loader"]

        class Iterator:
            _send_idx, _rcvd_idx, _shutdown = 7, 3, False
            error = None

            def __next__(self):
                if self.error is not None:
                    raise self.error
                if next(self._sampler_iter, None) is not None:
                    raise AssertionError("Cleanup submitted new samples")
                self._rcvd_idx += 1
                return object()

            def _shutdown_workers(self):
                self._shutdown = True

        for error in (None, ValueError("worker failed")):
            iterator = Iterator()
            iterator.error = error
            loader = SimpleNamespace(_iterator=iterator, num_workers=2, prefetch_factor=2)
            if error is None:
                close(loader)
                self.assertEqual(iterator._rcvd_idx, 7)
            else:
                with self.assertRaisesRegex(ValueError, "worker failed"):
                    close(loader)
            self.assertIsNone(loader._iterator)
            self.assertTrue(iterator._shutdown)

    def test_fractional_container_quota_is_not_host_cpu_count(self):
        select = RESOURCES["select_visible_cpu_count"]
        self.assertEqual(
            select(reported_cpu_count=48, affinity_cpu_count=48, quota_cpu_count=13.6), 13
        )
        self.assertEqual(select(reported_cpu_count=48, quota_cpu_count=0.5), 1)
        with self.assertRaises(ValueError):
            select(reported_cpu_count=48, quota_cpu_count=float("nan"))

    def test_cgroup_v1_v2_ancestors_and_unlimited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "membership"
            membership.write_text("0::/slice/job\n")
            (root / "slice/job").mkdir(parents=True)
            (root / "cpu.max").write_text("max 100000")
            (root / "slice/cpu.max").write_text("1360000 100000")
            (root / "slice/job/cpu.max").write_text("2000000 100000")
            self.assertEqual(RESOURCES["detect_cpu_quota"](root, membership), 13.6)
            (root / "slice/cpu.max").write_text("max 100000")
            (root / "slice/job/cpu.max").write_text("max 100000")
            self.assertIsNone(RESOURCES["detect_cpu_quota"](root, membership))
            (root / "cpu").mkdir()
            (root / "cpu/cpu.cfs_quota_us").write_text("1360000")
            (root / "cpu/cpu.cfs_period_us").write_text("100000")
            membership.write_text("3:cpu,cpuacct:/\n")
            self.assertEqual(RESOURCES["detect_cpu_quota"](root, membership), 13.6)

    def test_resume_with_another_batch_preserves_dynamic_epoch_order(self):
        from collections.abc import Iterator
        from math import gcd

        namespace = {
            "math": math,
            "gcd": gcd,
            "Iterator": Iterator,
            "Sampler": list,
            "islice": islice,
        }
        definitions(
            "src/stock_forecasting/data/dataset.py", {"BlockwisePermutationSampler"}, namespace
        )
        definitions(
            "src/stock_forecasting/baseline_runtime.py", {"SampleCursorBatchSampler"}, namespace
        )
        sampler_type = namespace["SampleCursorBatchSampler"]
        first = sampler_type(1003, 64, 42, 5)
        batches = list(first)
        original = [index for batch in batches for index in batch]
        cursor = sum(map(len, batches[:7]))
        resumed = sampler_type(1003, 91, 42, 5)
        resumed.set_epoch(0, cursor)
        rest = [index for batch in resumed for index in batch]
        self.assertEqual(original[:cursor] + rest, original)
        self.assertEqual(sorted(original), list(range(1003)))
        first.set_epoch(1)
        self.assertNotEqual([i for batch in first for i in batch], original)
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(1003)))
        self.assertEqual(len(resumed), len(list(resumed)))

    def test_prefetch_budget_accounts_for_one_active_pool_and_copies(self):
        namespace = definitions(
            "src/stock_forecasting/baseline_runtime.py", {"bounded_prefetch"}, {}
        )
        plan = namespace["bounded_prefetch"]
        self.assertEqual(
            plan(
                workers=2,
                batch_bytes=1024**2,
                host_bytes=1024**3,
                shared_bytes=64 * 1024**2,
                desired=8,
                maximum=4,
            ),
            4,
        )
        with self.assertRaises(MemoryError):
            plan(
                workers=2,
                batch_bytes=1024**2,
                host_bytes=1024**3,
                shared_bytes=1024,
                desired=2,
                maximum=4,
            )

    def test_full_size_mixed_admission_and_live_cpu_reallocation(self):
        memory = 52.4 * 1024**3
        namespace = {
            "detect_visible_cpu_count": lambda: 13,
            "detect_available_memory": lambda: SimpleNamespace(available_bytes=int(memory)),
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(mem_get_info=lambda: (24 * 1024**3, 24 * 1024**3))
            ),
            "shutil": SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=16 * 1024**3)),
        }
        definitions(
            "src/stock_forecasting/baseline_build.py",
            {"resource_plan", "live_cpu_allocation"},
            namespace,
        )
        parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        plan = namespace["resource_plan"](parameters, 30_224_227, list(range(1, 15)))
        self.assertEqual((plan["gpu_slots"], plan["loader_workers"]), (2, 2))
        self.assertEqual(plan["cpu_threads"], 5)
        allocation = namespace["live_cpu_allocation"]
        pending = [("gbdt", 42), ("gru", 42), ("dlinear", 42), ("patchtst", 42)]
        self.assertEqual(allocation(plan, [], pending)["gbdt_threads"], 5)
        running = [(None, 0, False, 1, "gbdt")]
        self.assertEqual(allocation(plan, running, [("patchtst", 42)])["gbdt_threads"], 8)
        tail = allocation(plan, running, [])
        self.assertEqual((tail["phase"], tail["gbdt_threads"]), ("cpu_tail", 11))
        memory = 43 * 1024**3
        with self.assertRaisesRegex(MemoryError, "overlap"):
            namespace["resource_plan"](parameters, 30_224_227, list(range(1, 15)))


if __name__ == "__main__":
    unittest.main()
