"""Bounded per-experiment CUDA tuning and sample-exact baseline resumption."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from itertools import islice

import torch

from stock_forecasting.baseline_input import close_neural_loader, neural_loader
from stock_forecasting.baselines import _normalized_pinball_torch
from stock_forecasting.data.dataset import BlockwisePermutationSampler
from stock_forecasting.evaluation_store import Moments


class SampleCursorBatchSampler:
    """Keep random epoch order and validation boundaries independent of batch size."""

    def __init__(self, count, batch_size, seed, evaluations):
        if min(count, batch_size, evaluations) < 1:
            raise ValueError("Sample count, batch size and evaluations must be positive")
        self.sampler = BlockwisePermutationSampler(count, seed=seed, block_size=128)
        self.batch_size = batch_size
        self.boundaries = sorted(
            {math.ceil(count * i / evaluations) for i in range(1, evaluations + 1)}
        )
        self.count, self.cursor = count, 0

    def set_epoch(self, epoch, cursor=0):
        if not 0 <= cursor <= self.count:
            raise ValueError("Baseline resume cursor is outside the training population")
        self.sampler.set_epoch(epoch)
        self.cursor = cursor

    def __iter__(self):
        position = self.cursor
        iterator = islice(iter(self.sampler), position, None)
        for boundary in self.boundaries:
            while position < boundary:
                size = min(self.batch_size, boundary - position)
                batch = list(islice(iterator, size))
                if len(batch) != size:
                    raise RuntimeError("Baseline sampler exhausted before its exact boundary")
                yield batch
                position += size

    def __len__(self):
        previous, batches = self.cursor, 0
        for boundary in self.boundaries:
            if boundary > previous:
                batches += math.ceil((boundary - previous) / self.batch_size)
                previous = boundary
        return batches


def bounded_prefetch(*, workers, batch_bytes, host_bytes, shared_bytes, desired, maximum):
    """Account for one active pool, raw scratch, IPC and pinned copies."""
    per_depth = workers * batch_bytes * 4
    capacity = min(int(host_bytes * 0.5), int(shared_bytes * 0.5)) // max(1, per_depth)
    if capacity < 1:
        raise MemoryError("Baseline batch cannot fit one safely bounded prefetch per worker")
    return max(1, min(maximum, desired, capacity))


def _cuda_probe(model, context, horizons, *, training, maximum, budget, repetitions):
    rows = []
    size = min(32, maximum)
    optimizer_reserve = sum(p.numel() * p.element_size() * 2 for p in model.parameters())
    original_mode = model.training
    # Tuning does not update weights, optimizer state or training RNG streams.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        try:
            model.train(training)
            while size <= maximum:
                if (
                    rows
                    and rows[-1].get("projected_peak_bytes", 0) * size / rows[-1]["batch_size"]
                    > budget
                ):
                    break
                x = prediction = loss = None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    x = torch.randn(size, 2, context, 5, device="cuda")
                    timings = []
                    for iteration in range(repetitions + 1):
                        model.zero_grad(set_to_none=True)
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        with torch.set_grad_enabled(training):
                            prediction = model(x)
                            if training:
                                loss = prediction.square().mean()
                                loss.backward()
                        end.record()
                        end.synchronize()
                        if iteration:
                            timings.append(begin.elapsed_time(end) / 1000)
                        prediction = loss = None
                    peak = torch.cuda.max_memory_allocated() + optimizer_reserve
                    seconds = sum(timings) / len(timings)
                    rows.append(
                        {
                            "batch_size": size,
                            "seconds_per_batch": seconds,
                            "samples_per_second": size / max(seconds, 1e-9),
                            "projected_peak_bytes": peak,
                            "accepted": peak <= budget,
                        }
                    )
                    if peak > budget:
                        break
                except torch.cuda.OutOfMemoryError:
                    rows.append({"batch_size": size, "accepted": False, "outcome": "oom"})
                    break
                finally:
                    x = prediction = loss = None
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                if size == maximum:
                    break
                size = min(size * 2, maximum)
        finally:
            model.train(original_mode)
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        raise MemoryError("No baseline batch fits the assigned concurrent GPU memory budget")
    best = max(row["samples_per_second"] for row in accepted)
    selected = next(row for row in accepted if row["samples_per_second"] >= best * 0.95)
    return selected, rows


def _pipeline_probe(model, source, parameters, scales, *, training, batch_size, workers, prefetch):
    """Measure the actual input/transfer/loss/optimizer path on an isolated model."""
    settings = parameters["resources"]
    warmup = math.ceil(settings.get("runtime_probe_warmup_samples", 16384) / batch_size)
    measured = max(
        settings.get("loader_probe_batches", 8),
        math.ceil(settings.get("runtime_probe_samples", 32768) / batch_size),
    )
    sampler = SampleCursorBatchSampler(len(source), batch_size, 59, 1)
    batches = list(islice(iter(sampler), warmup + measured))
    warmup = min(warmup, max(0, len(batches) - 1))
    loader = neural_loader(
        source,
        workers=workers,
        batch_sampler=batches,
        prefetch=prefetch,
        metadata=not training,
    )
    iterator = None
    original_mode = model.training
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        probe = deepcopy(model)
        # Deepcopy invalidates cuDNN's contiguous RNN weight storage. Restore
        # the same fast path as the production model before measuring it.
        for module in probe.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        probe.train(training)
        optimizer = (
            torch.optim.AdamW(
                probe.parameters(),
                lr=parameters["learning_rate"],
                weight_decay=parameters["weight_decay"],
            )
            if training
            else None
        )
        scale_tensor = torch.tensor(scales, device="cuda")
        moments = Moments(list(source.horizons), scales) if not training else None
        timings = []
        rows = 0
        wait_seconds = 0.0
        startup = time.monotonic()
        started = None
        try:
            iterator = iter(loader)
            for i in range(len(batches)):
                if i == warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                before = time.monotonic()
                batch = next(iterator)
                if i >= warmup:
                    wait_seconds += time.monotonic() - before
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                with torch.set_grad_enabled(training):
                    prediction = probe(batch["sequences"].cuda(non_blocking=True))
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss = _normalized_pinball_torch(
                            prediction, batch["target_alpha"].cuda(non_blocking=True), scale_tensor
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                        optimizer.step()
                end.record()
                if moments is not None:
                    moments.add(batch["target_alpha"].numpy(), prediction.float().cpu().numpy())
                if i >= warmup:
                    rows += len(prediction)
                    timings.append((begin, end))
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            return {
                "batch_size": batch_size,
                "workers": workers,
                "prefetch_factor": prefetch,
                "rows": rows,
                "seconds": seconds,
                "samples_per_second": rows / max(seconds, 1e-9),
                "input_wait_seconds": wait_seconds,
                "cuda_seconds": sum(a.elapsed_time(b) / 1000 for a, b in timings),
                "startup_warmup_seconds": started - startup,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "accepted": True,
            }
        finally:
            close_neural_loader(loader)
            del iterator, optimizer, probe, scale_tensor, timings
            model.train(original_mode)
            torch.cuda.empty_cache()


def tune_baseline_runtime(model, source, parameters, plan, *, scales=None, evaluation_source=None):
    settings = parameters["resources"]
    maximum_workers = plan["loader_workers"]
    manual = parameters["batch_size"]
    result = {
        "loader_workers": maximum_workers,
        "evaluation_workers": maximum_workers,
        "training_batch_size": manual,
        "evaluation_batch_size": manual,
        "prefetch_factor": plan["prefetch_factor"],
        "evaluation_prefetch_factor": plan["prefetch_factor"],
        "auto_batch": settings.get("auto_batch", True),
    }
    if not result["auto_batch"]:
        return result
    slots = plan.get("gpu_slots", 1)
    shared = plan.get("shared_memory_bytes")
    if shared is None:
        import shutil

        shared = shutil.disk_usage("/dev/shm").free
    shared //= slots
    per_sample = 2 * source.window_size * 5 * (4 + 8) + len(source.horizons) * 4

    def host_limits(workers):
        host = plan.get("gpu_job_host_bytes", 4 * 1024**3) - (
            workers * settings["worker_bytes"] + 1024**3
        )
        maximum = max(0, min(int(host * 0.5), int(shared * 0.5)) // (4 * workers * per_sample))
        if maximum < 1:
            raise MemoryError("No host/shared-memory budget remains for baseline auto batching")
        return host, maximum

    gpu_budget = int(
        torch.cuda.get_device_properties(0).total_memory * plan["gpu_fraction_per_job"] * 0.8
    )
    scales = scales if scales is not None else [1.0] * len(source.horizons)
    result["source"] = "end_to_end_real_windows"
    result["gpu_budget_bytes"] = gpu_budget
    result["shared_memory_budget_bytes"] = shared
    for mode, training, limit in (
        ("training", True, "max_training_batch_size"),
        ("evaluation", False, "max_evaluation_batch_size"),
    ):
        maximum = min(settings.get(limit, 2048 if training else 4096), host_limits(1)[1])
        chosen, probes = _cuda_probe(
            model,
            source.window_size,
            source.horizons,
            training=training,
            maximum=min(maximum, len(source)) if training else maximum,
            budget=gpu_budget,
            repetitions=settings.get("batch_probe_repetitions", 3),
        )
        maximum = max(row["batch_size"] for row in probes if row["accepted"])
        result[f"{mode}_gpu_admission_probe"] = probes
        sizes = sorted({max(1, chosen["batch_size"] // 2), chosen["batch_size"], maximum})
        trials, visited = [], set()

        def measure(
            size, workers, desired_prefetch, *, training=training, trials=trials, visited=visited
        ):
            host, maximum_host_batch = host_limits(workers)
            size = min(size, maximum_host_batch)
            prefetch = bounded_prefetch(
                workers=workers,
                batch_bytes=size * per_sample,
                host_bytes=host,
                shared_bytes=shared,
                desired=desired_prefetch,
                maximum=settings["max_prefetch_factor"],
            )
            key = (size, workers, prefetch)
            if key in visited:
                return
            visited.add(key)
            try:
                trial = _pipeline_probe(
                    model,
                    source if training or evaluation_source is None else evaluation_source,
                    parameters,
                    scales,
                    training=training,
                    batch_size=size,
                    workers=workers,
                    prefetch=prefetch,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                trials.append(
                    {
                        "batch_size": size,
                        "workers": workers,
                        "prefetch_factor": prefetch,
                        "accepted": False,
                        "outcome": "oom",
                    }
                )
                return
            trials.append(trial)

        for workers in sorted({1, (maximum_workers + 1) // 2, maximum_workers}):
            for size in sizes:
                measure(size, workers, plan["prefetch_factor"])
        accepted = [row for row in trials if row["accepted"]]
        if not accepted:
            raise MemoryError("No end-to-end baseline candidate fits the resource allocation")
        best = max(accepted, key=lambda row: row["samples_per_second"])
        for prefetch in sorted({1, settings["max_prefetch_factor"]}):
            measure(best["batch_size"], best["workers"], prefetch)
        accepted = [row for row in trials if row["accepted"]]
        best = max(row["samples_per_second"] for row in accepted)
        winner = min(
            (row for row in accepted if row["samples_per_second"] >= best * 0.95),
            key=lambda row: (row["workers"], row["prefetch_factor"], row["batch_size"]),
        )
        result[f"{mode}_batch_size"] = winner["batch_size"]
        result["loader_workers" if training else "evaluation_workers"] = winner["workers"]
        result["prefetch_factor" if training else "evaluation_prefetch_factor"] = winner[
            "prefetch_factor"
        ]
        result[f"{mode}_pipeline_probe"] = trials
        result[f"{mode}_samples_per_second"] = winner["samples_per_second"]
        print(
            f"Baseline {mode} end-to-end tuning: batch={winner['batch_size']} "
            f"workers={winner['workers']} prefetch={winner['prefetch_factor']} "
            f"windows/s={winner['samples_per_second']:.1f}",
            flush=True,
        )
    return result
