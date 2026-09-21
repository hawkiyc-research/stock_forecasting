#!/usr/bin/env python3
"""Bounded real-data before/after, full-size native-fit and checkpoint-copy checks."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import shutil
import subprocess
import threading
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from stock_forecasting.baseline_build import _close_loader, _sequence, resource_plan
from stock_forecasting.baseline_contract import execution_identity, runtime_contract
from stock_forecasting.baseline_input import NeuralBatchDataset, neural_loader
from stock_forecasting.baseline_runtime import SampleCursorBatchSampler, tune_baseline_runtime
from stock_forecasting.baseline_storage import (
    lazy_dataset,
    loader_options,
    open_tabular,
    worker_init,
)
from stock_forecasting.baselines import (
    CausalGRUBaseline,
    DLinearBaseline,
    PatchTSTBaseline,
    _normalized_pinball_torch,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import FinancialBatchCollator
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.runtime_resources import detect_cpu_quota, detect_visible_cpu_count
from stock_forecasting.training_paths import resolve_bar_store_path


def neural_probe(config_payload, root, parameters, plan, name, output, barrier, rows, legacy):
    worker_init(0)
    config = ExperimentConfig.model_validate(config_payload)
    torch.cuda.set_per_process_memory_fraction(plan["gpu_fraction_per_job"])
    source = lazy_dataset(config, "train", relative=True, validated_root=Path(root))
    validation = lazy_dataset(config, "validation", relative=True, validated_root=Path(root))
    torch.manual_seed(42)
    model = {
        "gru": lambda: CausalGRUBaseline(horizons=source.horizons),
        "dlinear": lambda: DLinearBaseline(source.window_size, horizons=source.horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=source.horizons),
    }[name]().cuda()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    scales = [0.03] * len(source.horizons)
    runtime = (
        {
            "training_batch_size": {"gru": 512, "dlinear": 2048, "patchtst": 2048}[name],
            "loader_workers": 1,
            "prefetch_factor": 2,
            "source": "observed_v0.2.1_runtime",
        }
        if legacy
        else tune_baseline_runtime(
            model, source, parameters, plan, scales=scales, evaluation_source=validation
        )
    )
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    del weights
    atomic_write_json(Path(output).with_suffix(".runtime.json"), runtime)
    print(f"{name} legacy={legacy} runtime ready", flush=True)
    sampler = SampleCursorBatchSampler(len(source), runtime["training_batch_size"], 42, 5)
    indices = list(islice(iter(sampler.sampler), rows + 16384))
    size = runtime["training_batch_size"]
    bounded = [
        part[start : start + size]
        for part in (indices[:16384], indices[16384:])
        for start in range(0, len(part), size)
    ]
    if legacy:
        loader = DataLoader(
            source,
            batch_sampler=bounded,
            collate_fn=FinancialBatchCollator(),
            pin_memory=True,
            **loader_options(1, runtime["prefetch_factor"], persistent=True),
        )
    else:
        loader = neural_loader(
            source,
            workers=runtime["loader_workers"],
            batch_sampler=bounded,
            prefetch=runtime["prefetch_factor"],
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"]
    )
    scale_tensor = torch.tensor(scales, device="cuda")
    iterator = iter(loader)
    started, count, warmup_rows, wait_seconds = None, 0, 0, 0.0
    events = []
    model.train()
    try:
        for _ in range(len(bounded)):
            if started is None and warmup_rows >= 16384:
                torch.cuda.synchronize()
                barrier.wait(timeout=900)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            begin = time.monotonic()
            batch = next(iterator)
            if started is not None:
                wait_seconds += time.monotonic() - begin
            if not legacy:
                assert batch["sequences"].is_pinned() and batch["sequences"].is_contiguous()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_sequence(batch, "cuda"))
            loss = _normalized_pinball_torch(
                prediction, batch["target_alpha"].cuda(non_blocking=True), scale_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            end.record()
            if started is not None:
                count += len(prediction)
                events.append((begin, end))
            else:
                warmup_rows += len(prediction)
        torch.cuda.synchronize()
        ended = time.monotonic()
        atomic_write_json(
            Path(output),
            {
                "name": name,
                "legacy": legacy,
                "rows": count,
                "warmup_rows": warmup_rows,
                "started_monotonic": started,
                "ended_monotonic": ended,
                "seconds": ended - started,
                "windows_per_second": count / (ended - started),
                "input_wait_seconds": wait_seconds,
                "cuda_seconds": sum(a.elapsed_time(b) / 1000 for a, b in events),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "tuning_preserved_weights_and_rng": True,
                "runtime": runtime,
            },
        )
    finally:
        _close_loader(loader)


def native_fit(cache_root, output, plan, barrier):
    """One full-population tree/quantile; no production artifacts are changed."""
    worker_init(0)
    arrays = open_tabular(Path(cache_root) / "inputs", "train")
    barrier.wait(timeout=900)
    started = time.monotonic()
    with threadpool_limits(limits=plan["cpu_threads"]):
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            max_iter=1,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(arrays["features"], arrays["targets"][:, 0])
    assert np.isfinite(model.predict(arrays["features"][:256])).all()
    atomic_write_json(
        Path(output),
        {
            "rows": len(arrays["features"]),
            "iterations": 1,
            "quantiles": 1,
            "horizons": 1,
            "cpu_threads": plan["cpu_threads"],
            "seconds": time.monotonic() - started,
            "started_monotonic": started,
            "ended_monotonic": time.monotonic(),
            "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )


def sample_gpu(stop, records, errors):
    try:
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            utilization, memory = (float(value) for value in result.stdout.strip().split(","))
            records.append(
                {"time": time.monotonic(), "utilization": utilization, "memory_mib": memory}
            )
            stop.wait(1)
    except Exception as error:
        errors.append(str(error))


def run_wave(config, root, cache_root, parameters, plan, output, rows, legacy):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2 if legacy else 3)
    jobs = [
        context.Process(
            target=neural_probe,
            args=(
                config.model_dump(mode="json"),
                str(root),
                parameters,
                plan,
                name,
                str(output / f"{name}.json"),
                barrier,
                rows,
                legacy,
            ),
        )
        for name in ("gru", "dlinear")
    ]
    if not legacy:
        jobs.append(
            context.Process(
                target=native_fit,
                args=(str(cache_root), str(output / "full-size-gbdt.json"), plan, barrier),
            )
        )
    records, errors, stop = [], [], threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, records, errors), daemon=True)
    monitor.start()
    try:
        for job in jobs:
            job.start()
        deadline = time.monotonic() + 900
        while any(job.is_alive() for job in jobs):
            if time.monotonic() >= deadline or any(job.exitcode not in (None, 0) for job in jobs):
                raise RuntimeError(
                    f"Bounded concurrency probe failed: {[job.exitcode for job in jobs]}"
                )
            time.sleep(1)
        for job in jobs:
            job.join()
            assert job.exitcode == 0
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
            if job.pid:
                job.join(timeout=15)
                if job.is_alive():
                    job.kill()
                    job.join(timeout=10)
        stop.set()
        monitor.join(timeout=10)
        atomic_write_json(output / "gpu-samples.json", records)
    if errors:
        raise RuntimeError(f"GPU monitoring failed: {errors}")


def resume_copies(config, cache_root, output, parameters, plan):
    import stock_forecasting.baseline_build as build

    results = {}
    for name in ("gru", "dlinear"):
        original = cache_root / "jobs" / f"{name}-42"
        hashes = {file: sha256_file(original / file) for file in ("model.pt", "resume.pt")}
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        for file in hashes:
            shutil.copyfile(original / file, target / file)
        before = torch.load(target / "resume.pt", map_location="cpu", weights_only=False)
        settings = json.loads(json.dumps(parameters))
        settings["resources"].update(auto_batch=False, checkpoint_seconds=0)
        settings["batch_size"] = 1024
        saved = build._save_torch

        def stop_after_commit(path, payload, *, before=before, saved=saved):
            saved(path, payload)
            if path.name == "resume.pt" and payload["sample_cursor"] > before["sample_cursor"]:
                raise InterruptedError("Expected bounded checkpoint-copy verification")

        build._save_torch = stop_after_commit
        try:
            artifact = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
            try:
                build._train_neural(config, target, name, 42, settings, artifact["scales"], plan)
            except InterruptedError:
                pass
            else:
                raise AssertionError("The checkpoint-copy probe must stop after one commit")
        finally:
            build._save_torch = saved
        after = torch.load(target / "resume.pt", map_location="cpu", weights_only=False)
        assert before["epoch"] == after["epoch"]
        assert 0 < after["sample_cursor"] - before["sample_cursor"] <= 1024
        assert after["scheduler"]["last_epoch"] > before["scheduler"]["last_epoch"]
        assert all(sha256_file(original / file) == value for file, value in hashes.items())
        results[name] = {
            "before_cursor": before["sample_cursor"],
            "after_cursor": after["sample_cursor"],
            "original_hashes_unchanged": hashes,
            "epoch": after["epoch"],
        }
    atomic_write_json(output / "summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=262144)
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID") or not 16384 <= args.rows <= 1048576:
        raise ValueError("Only bounded authorized cloud probes are supported")
    worker_init(0)
    args.output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.from_yaml(
        os.environ.get("RUNPOD_CONFIG", "configs/stage2_kronos_base_lora.yaml")
    )
    root = resolve_bar_store_path(config.data.bar_store_path)
    before = sha256_file(root / "bar-store.json")
    project, identity = runtime_contract()
    cache_root = (
        Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
        / "baselines"
        / identity["baseline_id"]
    )
    cache_before = sha256_file(cache_root / "inputs/complete.json")
    source = lazy_dataset(config, "train", relative=True, validated_root=root)
    parameters = json.loads(Path("configs/baseline.json").read_text())
    plan = resource_plan(parameters, len(source), list(source.horizons), source.window_size)
    plan["validated_bar_store_root"] = str(root)
    if plan["gpu_slots"] < 2:
        raise MemoryError("The concurrency probe requires admission for two GPU experiments")
    atomic_write_json(args.output / "resource-plan.json", plan)
    atomic_write_json(args.output / "execution-contract.json", execution_identity(project))
    for split in ("train", "validation", "test"):
        data = lazy_dataset(config, split, relative=True, validated_root=root)
        indices = next(iter(SampleCursorBatchSampler(len(data), 128, 42, 1)))
        old = FinancialBatchCollator()(data.__getitems__(indices))
        new = NeuralBatchDataset(data, metadata=True).__getitems__(indices)
        torch.testing.assert_close(new["sequences"], _sequence(old, "cpu"), rtol=0, atol=0)
        torch.testing.assert_close(new["target_alpha"], old["target_alpha"], rtol=0, atol=0)
        for key in ("symbols", "cutoff_at", "markets", "asset_types", "providers"):
            assert old[key] == new[key]
    for phase, legacy in (("before", True), ("after", False)):
        output = args.output / phase
        output.mkdir(exist_ok=True)
        print(f"Starting {phase} real-window wave", flush=True)
        run_wave(config, root, cache_root, parameters, plan, output, args.rows, legacy)
    context = multiprocessing.get_context("spawn")
    neural_probe(
        config.model_dump(mode="json"),
        str(root),
        parameters,
        plan,
        "patchtst",
        str(args.output / "patchtst.json"),
        context.Barrier(1),
        args.rows,
        False,
    )
    resumed = resume_copies(config, cache_root, args.output / "resume-copies", parameters, plan)
    assert sha256_file(root / "bar-store.json") == before
    assert sha256_file(cache_root / "inputs/complete.json") == cache_before
    comparisons = {}
    for name in ("gru", "dlinear"):
        old = json.loads((args.output / "before" / f"{name}.json").read_text())
        new = json.loads((args.output / "after" / f"{name}.json").read_text())
        assert old["rows"] == new["rows"] == args.rows
        comparisons[name] = {
            "before": old,
            "after": new,
            "speedup": new["windows_per_second"] / old["windows_per_second"],
        }
    summary = {
        "baseline_id": identity["baseline_id"],
        "quota_cpus": detect_cpu_quota(),
        "usable_cpus": detect_visible_cpu_count(),
        "numeric_equivalence": True,
        "prepared_data_unchanged": True,
        "manifest_sha256": before,
        "production_cache_unchanged": True,
        "comparisons": comparisons,
        "resume_copies": resumed,
        "full_size_gbdt": json.loads((args.output / "after/full-size-gbdt.json").read_text()),
        "patchtst": json.loads((args.output / "patchtst.json").read_text()),
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "baseline_id": summary["baseline_id"],
                "speedups": {key: row["speedup"] for key, row in comparisons.items()},
                "numeric_equivalence": True,
                "original_artifacts_unchanged": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
