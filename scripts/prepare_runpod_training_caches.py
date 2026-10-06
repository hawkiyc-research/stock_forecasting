#!/usr/bin/env python3
"""Serialize cold shared-cache construction; model training remains concurrent."""

from __future__ import annotations

import argparse
import fcntl
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("Training cache preparation must run on a cloud Pod")
    if not 1 <= args.lock_timeout <= 7200:
        raise ValueError("Cache lock timeout must be between 1 and 7200 seconds")
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.date_market_sampler import date_market_index
    from stock_forecasting.scale_calibration import resolve_scale_feature_statistics
    from stock_forecasting.training import (
        _loader_process_options,
        _requested_dataloader_workers,
        build_lazy_datasets,
        plan_dataloader_workers,
        resolve_runtime_robust_scales,
    )

    config = ExperimentConfig.from_yaml(args.config)
    # Only construction is serialized. Published caches remain shared and no
    # window tensors, bar-store or model weights are copied for another run.
    root = Path(os.environ["DATA_ROOT"]) / "training-cache"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "initialization.lock").open("a") as lock:
        deadline = time.monotonic() + args.lock_timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Shared cache initialization exceeded its bounded wait"
                    ) from error
                print("Waiting for another Pod to publish shared training caches", flush=True)
                time.sleep(min(10, max(0, deadline - time.monotonic())))
        requested, origin = _requested_dataloader_workers(config)
        plan = plan_dataloader_workers(requested, source=origin)
        source = build_lazy_datasets(config, worker_plan=plan)[0]
        resolve_runtime_robust_scales(
            source,
            sample_count=config.data.label_scale_calibration_samples,
            seed=config.data.calibration_seed
            if config.data.fixed_split
            else config.training.seed + 17,
            worker_plan=plan,
        )
        if config.model.feature_mode in ("scales", "combined"):
            resolve_scale_feature_statistics(
                source,
                sample_count=config.data.label_scale_calibration_samples,
                seed=config.data.calibration_seed,
                loader_options=_loader_process_options(plan, persistent=False),
                extended=config.model.explicit_output_scale,
            )
        if config.model.ranking_loss_weight or config.training.yearly_sampling_decay is not None:
            date_market_index(source, requested_workers=max(1, plan.effective_workers))
    print("Shared training caches ready; independent model training can proceed", flush=True)


if __name__ == "__main__":
    main()
