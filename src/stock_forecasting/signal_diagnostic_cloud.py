"""Bounded fixed-train evaluation and raw-label audit for cloud-only inference."""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from dataclasses import replace

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset

from stock_forecasting.data.sample_universe import open_clean_dataset
from stock_forecasting.direction_data import with_direction_targets


def fixed_train_loader(config, bundle, device, count=16384, seed=59):
    """Use the existing resource planner and eval collator with fixed uniform cutoffs."""
    from stock_forecasting.training import (
        ModelBatchCollator,
        _loader_process_options,
        _requested_dataloader_workers,
        plan_dataloader_workers,
        plan_runtime_prefetch,
        resolve_runtime_batch_plan,
    )

    if not 16 <= count <= 50000:
        raise ValueError("Fixed train diagnostics require 16-50000 samples")
    requested, reason = _requested_dataloader_workers(config)
    workers = replace(plan_dataloader_workers(requested, source=reason), active_persistent_pools=1)
    if workers.effective_workers < 1:
        raise ValueError("Cloud diagnostics require at least one resource-budgeted loader process")
    source = with_direction_targets(
        open_clean_dataset(config, "train", symbol_cache_size=workers.symbol_cache_size_per_worker),
        config,
    )
    indices = sorted(random.Random(seed).sample(range(len(source)), min(count, len(source))))
    dataset = Subset(source, indices)
    sample = dataset[0]
    plan = resolve_runtime_batch_plan(
        config,
        bundle=bundle,
        sample=sample,
        device=device,
        dataset=dataset,
        worker_plan=workers,
        evaluation_only=True,
    )
    collator = ModelBatchCollator.for_model(config, training=False)
    workers = plan_runtime_prefetch(
        workers,
        config=config,
        batch_plan=plan,
        largest_host_batch_bytes=collator.estimated_batch_bytes(sample, plan.evaluation_batch_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=plan.evaluation_batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=collator,
        pin_memory=True,
        **_loader_process_options(workers, persistent=False),
    )
    return loader, indices


def audit_candidates(store, limit):
    """Select high-loss/high-drawdown cases plus ordinary same-date/market controls."""
    if not 0 <= limit <= 128:
        raise ValueError("Audit size must be 0-128 cases")
    if limit < 2:
        return []
    pairs = limit // 2
    per_market = max(1, math.ceil(pairs / 3))
    heaps = {}
    for start in range(0, store.count, 4096):
        stop = min(start + 4096, store.count)
        targets = (
            store.targets[start:stop]
            if store.absolute_targets is None
            else store.absolute_targets[start:stop]
        )
        losses = store._moments().add(targets, store.predictions[start:stop])
        for local in np.flatnonzero(store.drawdown[start:stop] >= 0.4):
            i = start + int(local)
            market = str(store.metadata[i]["market"])
            heap = heaps.setdefault(market, [])
            heapq.heappush(heap, (float(losses[local]), i))
            if len(heap) > per_market:
                heapq.heappop(heap)
    hard = sorted((entry for heap in heaps.values() for entry in heap), reverse=True)[:pairs]
    groups = {(str(store.metadata[i]["date"]), str(store.metadata[i]["market"])) for _, i in hard}
    controls = {}
    hard_indices = {i for _, i in hard}
    for i in range(store.count):
        row = store.metadata[i]
        group = str(row["date"]), str(row["market"])
        if group not in groups or i in hard_indices or not store.drawdown[i] < 0.2:
            continue
        # Stable hash selection does not inspect the control's future return or loss.
        key = hashlib.sha256((str(row["symbol"]) + "|" + group[0] + "|59").encode()).hexdigest()
        if group not in controls or key < controls[group][0]:
            controls[group] = key, i
    result = []
    for loss, i in hard:
        group = str(store.metadata[i]["date"]), str(store.metadata[i]["market"])
        result.append(
            {
                "row": i,
                "role": "high_loss_drawdown_ge_40pct",
                "loss": loss,
                "matched_control_row": controls.get(group, (None, None))[1],
            }
        )
    for i in sorted(
        {row["matched_control_row"] for row in result if row["matched_control_row"] is not None}
    ):
        result.append({"row": i, "role": "same_date_market_drawdown_lt_20pct_control"})
    for case in result:
        case["captured_main_alpha_target"] = store.targets[case["row"]].tolist()
        if store.absolute_targets is not None:
            case["captured_main_absolute_target"] = store.absolute_targets[case["row"]].tolist()
    return result


def _frame_rows(frame):
    columns = ["open", "high", "low", "close", "volume", "adjusted_close"]
    return {
        "dates": [str(day.isoformat()) for day in frame["timestamp"]],
        "columns": columns,
        "values": frame[columns].to_numpy(dtype=float).tolist(),
    }


class RawLabelAuditDataset(Dataset):
    """Only raw readers enter loader workers; the parent GPU model is never copied."""

    def __init__(self, source, cases):
        self.source, self.cases = source, cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        ordinal = case["source_ordinal"]
        symbol, info, benchmark_symbol, frame, benchmark, cutoff, holding_dates, bholding = (
            self.source._sample_frames(ordinal)
        )
        observed = frame.iloc[cutoff - self.source.window_size + 1 : cutoff + 1]
        full = frame.iloc[cutoff - self.source.window_size + 1 : cutoff + 15]
        aligned = self.source._aligned_rows(
            benchmark_symbol, benchmark, pd.DatetimeIndex(full["timestamp"])
        )
        aholding = frame.iloc[cutoff + 1 : cutoff + 15]
        entry = float(
            aholding["open"].iloc[0]
            * aholding["adjusted_close"].iloc[0]
            / aholding["close"].iloc[0]
        )
        benchmark_entry = float(
            bholding["open"].iloc[0]
            * bholding["adjusted_close"].iloc[0]
            / bholding["close"].iloc[0]
        )
        absolute = [
            math.log(float(aholding["adjusted_close"].iloc[h - 1]) / entry)
            for h in self.source.horizons
        ]
        benchmark_returns = [
            math.log(float(bholding["adjusted_close"].iloc[h - 1]) / benchmark_entry)
            for h in self.source.horizons
        ]
        alpha = [a - b for a, b in zip(absolute, benchmark_returns, strict=True)]
        arrays = self.source.array_batch(
            [ordinal], include_timestamps=False, include_metadata=False
        )
        maximum = max(abs(a - float(b)) for a, b in zip(alpha, arrays["targets"][0], strict=True))
        if maximum > 1e-6:
            raise ValueError("Independent next-open/horizon-close label reconstruction disagrees")
        maximum_export = max(
            abs(a - float(b))
            for a, b in zip(alpha, case.get("captured_main_alpha_target", alpha), strict=True)
        )
        maximum_absolute = max(
            abs(a - float(b))
            for a, b in zip(
                absolute, case.get("captured_main_absolute_target", absolute), strict=True
            )
        )
        if max(maximum_export, maximum_absolute) > 1e-6:
            raise ValueError("Raw reconstruction disagrees with main-model exported targets")
        return {
            **case,
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
            "market": info["market"],
            "cutoff": str(observed["timestamp"].iloc[-1].isoformat()),
            "label_end": str(holding_dates[-1].isoformat()),
            "horizons": list(self.source.horizons),
            "asset_vendor_raw_and_adjusted": _frame_rows(full),
            "benchmark_vendor_raw_and_adjusted": _frame_rows(aligned),
            "point_in_time_input_asset": arrays["asset"][0].tolist(),
            "point_in_time_input_benchmark": arrays["benchmark"][0].tolist(),
            "entry_adjusted_open": entry,
            "benchmark_entry_adjusted_open": benchmark_entry,
            "recomputed_absolute_log_returns": absolute,
            "recomputed_alpha_log_returns": alpha,
            "maximum_internal_alpha_disagreement": maximum,
            "maximum_export_alpha_disagreement": maximum_export,
            "maximum_export_absolute_disagreement": maximum_absolute,
            "limitation": (
                "Internal consistency only; "
                "vendor corporate actions are not independently verified."
            ),
        }


def _audit_collate(batch):
    return batch


def collect_audit(loader, cases, indices=None, requested_workers=2):
    from stock_forecasting.training import _loader_process_options, plan_dataloader_workers

    if not cases:
        return [], None
    source = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    source.symbol_cache_size = min(source.symbol_cache_size, 4)
    cases = [
        dict(case, source_ordinal=indices[case["row"]] if indices is not None else case["row"])
        for case in cases
    ]
    if not 1 <= requested_workers <= 8:
        raise ValueError("Audit workers must be 1-8")
    workers = replace(
        plan_dataloader_workers(requested_workers, source="bounded_label_audit"),
        active_persistent_pools=1,
    )
    if workers.effective_workers < 1:
        raise MemoryError("No memory headroom for a raw-label audit worker")
    if workers.effective_workers == 1:
        print("Raw-label audit uses one process after CPU/memory resource limits.", flush=True)
    audit = DataLoader(
        RawLabelAuditDataset(source, cases),
        batch_size=1,
        collate_fn=_audit_collate,
        **_loader_process_options(workers, persistent=False),
    )
    records = []
    iterator = iter(audit)
    try:
        for batch in iterator:
            records.extend(batch)
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    return records, workers.as_dict()
