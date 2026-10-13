"""Full-population signal evaluation using the unchanged loader resource policy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import SequentialSampler

from stock_forecasting.signal_evaluation import SignalEvaluationStore


def _past_drawdown(close, mask=None):
    """Measure observed context drawdown without treating padding as prices."""
    if mask is None:
        return (1 - close / close.cummax(dim=1).values).amax(dim=1)
    valid = mask.bool()
    running_peak = close.masked_fill(~valid, 0).cummax(dim=1).values
    declines = 1 - close / running_peak.clamp_min(torch.finfo(close.dtype).tiny)
    return declines.masked_fill(~valid, 0).amax(dim=1)


@torch.inference_mode()
def evaluate_signal_loader(bundle, loader, config, device):
    from stock_forecasting.training import _autocast_context, forward_batch, iter_device_batches

    if (
        not isinstance(loader.sampler, SequentialSampler)
        or len(loader.sampler) != len(loader.dataset)
        or loader.drop_last
    ):
        raise ValueError("Signal evaluation requires sequential full coverage without drop_last")
    was_training = bundle.model.training
    bundle.model.eval()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scales = bundle.model.alpha_head.robust_scales.float().cpu().tolist()
    scratch = config.runtime.log_root / "evaluation-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="signal-full-", dir=scratch) as directory:
            store = SignalEvaluationStore(
                Path(directory),
                len(loader.dataset),
                config.data.alpha_horizons,
                scales,
                config.model.prediction_task,
                beta=config.model.point_loss_beta,
                neutral_threshold=config.model.neutral_return_threshold,
            )
            try:
                completed, last_log = 0, started
                for batch in iter_device_batches(loader, device):
                    with _autocast_context(config, device):
                        output = forward_batch(bundle, batch, config, device)
                    predictions = (
                        output.alpha_point
                        if config.model.prediction_task == "alpha_point"
                        else output.absolute_direction_logits
                    )
                    close = batch["asset_series"][..., 3].float()
                    drawdown = _past_drawdown(close, batch.get("asset_attention_mask"))
                    absolute = batch.get("target_absolute")
                    store.append(
                        batch["target_alpha"].detach().float().cpu().numpy(),
                        predictions.detach().float().cpu().numpy(),
                        target_absolute=None
                        if absolute is None
                        else absolute.detach().float().cpu().numpy(),
                        past_drawdown=drawdown.cpu().numpy(),
                        symbols=batch["symbols"],
                        dates=batch["cutoff_at"],
                        markets=batch["markets"],
                        asset_types=batch["asset_types"],
                        providers=batch["providers"],
                    )
                    completed += len(batch["symbols"])
                    now = time.perf_counter()
                    if now - last_log >= 30:
                        print(
                            json.dumps(
                                {
                                    "evaluation_progress": completed,
                                    "total": len(loader.dataset),
                                    "prediction_task": config.model.prediction_task,
                                    "samples_per_second": completed / (now - started),
                                }
                            ),
                            flush=True,
                        )
                        last_log = now
                inference_seconds = time.perf_counter() - started
                result = store.finish()
                result["execution"] = {
                    "full_population": True,
                    "expected_samples": len(loader.dataset),
                    "completed_samples": completed,
                    "batch_size": loader.batch_size,
                    "workers": loader.num_workers,
                    "prefetch_factor": loader.prefetch_factor,
                    "pin_memory": loader.pin_memory,
                    "drop_last": False,
                    "inference_seconds": inference_seconds,
                    "samples_per_second": completed / max(inference_seconds, 1e-9),
                    "total_seconds": time.perf_counter() - started,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                }
                return result
            finally:
                store.close()
    finally:
        bundle.model.train(was_training)
