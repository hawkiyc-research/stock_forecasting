"""Dependency-free regressions for finite CUDA probe admission and cleanup."""

from __future__ import annotations

import contextlib
import io
import json
import math
import runpy
import time
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CONTROL = runpy.run_path(str(ROOT / "tests/test_training_pipeline_control.py"))


class CudaProbeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "math": math, "json": json, "cast": cast, "time": time,
            "pairwise": pairwise,
            "_is_cuda_out_of_memory": lambda error: isinstance(error, MemoryError),
            "AUTO_BATCH_THROUGHPUT_TOLERANCE": 0.03,
        }
        self.scope["dataclass"] = CONTROL["dataclass"]
        CONTROL["definitions"](CONTROL["TRAINING"], {
            "BatchProbeMeasurement", "_finish_probe_loader", "_project_next_probe_peak",
            "_probe_cuda_candidates", "_select_batch_measurement",
        }, self.scope)

    def row(self, size, peak, speed=100.0):
        return self.scope["BatchProbeMeasurement"](
            size, size / speed, speed, peak, peak, True, "accepted",
        )

    def test_cpu_and_worker_memory_errors_are_not_cuda_capacity_rejections(self):
        scope = {"torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError))}
        CONTROL["definitions"](CONTROL["TRAINING"], {"_is_cuda_out_of_memory"}, scope)
        check = scope["_is_cuda_out_of_memory"]
        self.assertTrue(check(MemoryError("injected OOM")))
        self.assertTrue(check(RuntimeError("CUDA out of memory")))
        self.assertFalse(check(RuntimeError("DefaultCPUAllocator: out of memory")))
        self.assertFalse(check(RuntimeError("DataLoader worker killed by signal: Aborted")))
        self.assertFalse(check(None))

    def loader(self, count, fail=False):
        events = []

        class Loader:
            def __init__(self):
                self.remaining = count
                self.exhausted = False

            def __next__(self):
                events.append("fetch")
                if self.remaining == 0:
                    self.exhausted = True
                    raise StopIteration
                self.remaining -= 1
                return object()

            def _shutdown_workers(self):
                events.append("shutdown")
                if fail or not self.exhausted:
                    raise RuntimeError("DataLoader worker is killed by signal: Aborted")

        return Loader(), events

    def test_oom_drains_host_queue_before_worker_shutdown(self):
        loader, events = self.loader(22)
        device = SimpleNamespace(close=lambda: events.append("close_cuda_generator"))
        self.scope["_finish_probe_loader"](
            device, loader, max_batches=80, primary_error=MemoryError("CUDA OOM"),
        )
        self.assertEqual(events[0], "close_cuda_generator")
        self.assertEqual(events.count("fetch"), 23)
        self.assertEqual(events[-1], "shutdown")

    def test_success_also_consumes_stop_iteration(self):
        loader, events = self.loader(0)
        self.scope["_finish_probe_loader"](None, loader, max_batches=80, primary_error=None)
        self.assertEqual(events, ["fetch", "shutdown"])

    def test_genuine_worker_failure_is_not_swallowed_or_drained(self):
        loader, events = self.loader(22, fail=True)
        original = ValueError("invalid input data")
        with self.assertRaises(BaseExceptionGroup) as caught:
            self.scope["_finish_probe_loader"](
                None, loader, max_batches=80, primary_error=original, drain=False,
            )
        self.assertIs(caught.exception.exceptions[0], original)
        self.assertRegex(str(caught.exception.exceptions[1]), "Aborted")
        self.assertEqual(events, ["shutdown"])

    def test_cleanup_failure_retains_original_oom(self):
        loader, _events = self.loader(0, fail=True)
        original = MemoryError("CUDA allocation failed")
        with self.assertRaises(BaseExceptionGroup) as caught:
            self.scope["_finish_probe_loader"](
                None, loader, max_batches=80, primary_error=original,
            )
        self.assertIs(caught.exception.exceptions[0], original)
        self.assertEqual(len(caught.exception.exceptions), 2)

    def test_cleanup_without_original_error_still_fails(self):
        loader, _events = self.loader(0, fail=True)
        with self.assertRaises(BaseExceptionGroup):
            self.scope["_finish_probe_loader"](
                None, loader, max_batches=80, primary_error=None,
            )

    def test_unbounded_source_is_rejected_and_shutdown_attempted(self):
        loader, events = self.loader(100)
        with self.assertRaises(BaseExceptionGroup) as caught:
            self.scope["_finish_probe_loader"](
                None, loader, max_batches=2, primary_error=None,
            )
        self.assertIn("finite batch budget", str(caught.exception.exceptions[0]))
        self.assertEqual(events, ["fetch"] * 3 + ["shutdown"])

    def test_no_iterator_and_close_failure_cleanup(self):
        self.scope["_finish_probe_loader"](None, None, max_batches=0, primary_error=None)
        loader, events = self.loader(0, fail=True)

        def fail():
            raise RuntimeError("transfer close failure")

        with self.assertRaises(BaseExceptionGroup) as caught:
            self.scope["_finish_probe_loader"](
                SimpleNamespace(close=fail), loader, max_batches=0, primary_error=None,
            )
        self.assertEqual(len(caught.exception.exceptions), 2)
        self.assertEqual(events, ["shutdown"])

    def test_slow_drain_has_a_total_deadline_and_restores_fetch_timeout(self):
        loader, events = self.loader(100)
        loader._timeout = 300
        ticks = iter([0.0, 1.0, 61.0])
        self.scope["time"] = SimpleNamespace(monotonic=lambda: next(ticks))
        with self.assertRaises(BaseExceptionGroup) as caught:
            self.scope["_finish_probe_loader"](
                None, loader, max_batches=100, primary_error=None, timeout_seconds=60,
            )
        self.assertIsInstance(caught.exception.exceptions[0], TimeoutError)
        self.assertEqual(events, ["fetch", "shutdown"])
        self.assertEqual(loader._timeout, 300)

    def test_actual_5090_peak_predicts_256_above_budget(self):
        measured = [self.row(64, 10965615616), self.row(128, 21104672256)]
        peak = self.scope["_project_next_probe_peak"](measured, 256)
        self.assertGreater(peak, 24242946048)
        self.assertIsNone(self.scope["_project_next_probe_peak"](measured[:1], 128))
        self.assertIsNone(self.scope["_project_next_probe_peak"](measured, 64))

    def test_growing_marginal_cost_is_not_ignored(self):
        rows = [self.row(4, 100), self.row(8, 200), self.row(16, 800)]
        self.assertEqual(self.scope["_project_next_probe_peak"](rows, 32), 2000)

    def test_memory_admission_skips_allocation_but_keeps_best_measured_candidate(self):
        calls = []
        self.scope["_training_sampler"] = lambda *_args: range(4096)
        self.scope["torch"] = SimpleNamespace(
            get_rng_state=lambda: None, set_rng_state=lambda _v: None,
            cuda=SimpleNamespace(get_rng_state_all=lambda: [], set_rng_state_all=lambda _v: None),
        )

        def measure(**kwargs):
            size = kwargs["batch_size"]
            calls.append(size)
            if size == 256:
                self.fail("Unsafe batch 256 must not be allocated")
            return self.row(size, size * 160_000_000, 500 if size == 64 else 400)

        self.scope["_measure_cuda_batch"] = measure
        with contextlib.redirect_stdout(io.StringIO()):
            best, rows = self.scope["_probe_cuda_candidates"](
                bundle=SimpleNamespace(model=SimpleNamespace(modules=lambda: [])),
                sample=None, candidates=(32, 64, 128, 256), config=None, device=None,
                training=True, optimizer_state_reserve_bytes=0,
                device_memory_limit_bytes=24242946048,
                dataset=range(4096), worker_plan=None,
            )
        self.assertEqual(calls, [32, 64, 128])
        self.assertEqual(best.batch_size, 64)
        self.assertEqual(rows[-1].outcome, "projected_memory_guard")
        self.assertFalse(rows[-1].accepted)


if __name__ == "__main__":
    unittest.main()
