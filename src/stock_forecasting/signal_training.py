"""Full-population signal evaluation using the unchanged loader resource policy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
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


def _apply_v2_cross_sectional_evaluation(
    head,
    store: SignalEvaluationStore,
    *,
    task: str,
    split_name: str | None,
    raw_primary: np.ndarray,
    raw_secondary: np.ndarray,
    market_ids: np.ndarray | None,
    device: torch.device,
) -> None:
    """Apply exact full-population (date, market) standardization/centering and calibration."""
    _, inverse, counts = np.unique(
        store.metadata[["date", "market"]],
        return_inverse=True,
        return_counts=True,
    )
    num_groups = len(counts)
    num_horizons = raw_primary.shape[1]
    group_sums = np.zeros((num_groups, num_horizons), dtype=np.float64)
    np.add.at(group_sums, inverse, raw_primary)
    group_means = group_sums / np.maximum(counts[:, None], 1)
    centered = raw_primary.astype(np.float64) - group_means[inverse]
    active_group = (counts >= 2)[inverse, None]

    if task == "alpha_point":
        group_sq_sums = np.zeros((num_groups, num_horizons), dtype=np.float64)
        np.add.at(group_sq_sums, inverse, centered * centered)
        group_vars = group_sq_sums / np.maximum(counts[:, None], 1)
        z_tilde_np = np.where(
            active_group,
            centered / np.sqrt(group_vars[inverse] + 0.01),
            raw_primary.astype(np.float64),
        ).astype(np.float32)
        z_tensor = torch.from_numpy(z_tilde_np).to(device=device)
        d_tensor = torch.from_numpy(raw_secondary).to(device=device)
        alpha_pred = head.compute_alpha_from_standardized(z_tensor, d_tensor)
        store.predictions[:] = alpha_pred.detach().float().cpu().numpy()
        return

    mu_centered_np = np.where(active_group, centered, raw_primary.astype(np.float64)).astype(
        np.float32
    )
    assert market_ids is not None
    total = len(mu_centered_np)
    cdf_logits_chunks = []
    chunk_size = 8192
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        mu_chunk = torch.from_numpy(mu_centered_np[start:stop]).to(device=device)
        sig_chunk = torch.from_numpy(raw_secondary[start:stop]).to(device=device)
        m_chunk = torch.from_numpy(market_ids[start:stop]).to(device=device)
        cdf_logits_chunks.append(
            head.compute_direction_cdf_logits(mu_chunk, sig_chunk, m_chunk).detach().cpu()
        )
    cdf_logits_all = torch.cat(cdf_logits_chunks, dim=0)
    markets_all = torch.from_numpy(market_ids)

    if split_name == "validation" and store.absolute_targets is not None:
        targets_all = torch.from_numpy(np.asarray(store.absolute_targets, dtype=np.float32))
        head.fit_validation_calibration(cdf_logits_all, targets_all, markets_all)

    calibrated_chunks = []
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        c_chunk = cdf_logits_all[start:stop].to(device=device)
        m_chunk = markets_all[start:stop].to(device=device)
        calibrated_chunks.append(
            head.apply_direction_calibration(c_chunk, m_chunk).detach().float().cpu().numpy()
        )
    store.predictions[:] = np.concatenate(calibrated_chunks, axis=0)


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
    head = bundle.model.alpha_head
    scales = head.robust_scales.float().cpu().tolist()
    is_v2_head = getattr(head, "signal_architecture", "legacy") == "v2"
    total_samples = len(loader.dataset)
    num_horizons = len(config.data.alpha_horizons)
    raw_primary = (
        np.empty((total_samples, num_horizons), dtype=np.float32) if is_v2_head else None
    )
    raw_secondary = (
        np.empty((total_samples, num_horizons), dtype=np.float32) if is_v2_head else None
    )
    market_id_store = (
        np.empty((total_samples,), dtype=np.int64)
        if is_v2_head and config.model.prediction_task == "absolute_direction"
        else None
    )
    scratch = config.runtime.log_root / "evaluation-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="signal-full-", dir=scratch) as directory:
            store = SignalEvaluationStore(
                Path(directory),
                total_samples,
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
                    batch_len = len(batch["symbols"])
                    section = slice(completed, completed + batch_len)
                    if is_v2_head and raw_primary is not None and raw_secondary is not None:
                        if config.model.prediction_task == "alpha_point":
                            assert head._last_eval_alpha is not None
                            u_batch, d_batch = head._last_eval_alpha
                            raw_primary[section] = u_batch.float().cpu().numpy()
                            raw_secondary[section] = d_batch.float().cpu().numpy()
                        else:
                            assert head._last_eval_direction is not None
                            assert market_id_store is not None
                            mu_batch, sig_batch, m_batch = head._last_eval_direction
                            raw_primary[section] = mu_batch.float().cpu().numpy()
                            raw_secondary[section] = sig_batch.float().cpu().numpy()
                            market_id_store[section] = m_batch.cpu().numpy()
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
                    completed += batch_len
                    now = time.perf_counter()
                    if now - last_log >= 30:
                        print(
                            json.dumps(
                                {
                                    "evaluation_progress": completed,
                                    "total": total_samples,
                                    "prediction_task": config.model.prediction_task,
                                    "samples_per_second": completed / (now - started),
                                }
                            ),
                            flush=True,
                        )
                        last_log = now
                if is_v2_head and raw_primary is not None and raw_secondary is not None:
                    _apply_v2_cross_sectional_evaluation(
                        head,
                        store,
                        task=config.model.prediction_task,
                        split_name=getattr(loader.dataset, "split", None),
                        raw_primary=raw_primary,
                        raw_secondary=raw_secondary,
                        market_ids=market_id_store,
                        device=device,
                    )
                inference_seconds = time.perf_counter() - started
                result = store.finish()
                result["execution"] = {
                    "full_population": True,
                    "expected_samples": total_samples,
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
