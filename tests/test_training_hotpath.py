"""Cloud-only numerical, RNG, input-validation and CUDA hot-path regressions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from stock_forecasting import training
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.forecast import MultiHorizonAlphaHead
from stock_forecasting.models.ranking import eligible_ranking_pairs, same_date_ranking_loss
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner"
))]


def _sample(index=0, length=128):
    time = torch.arange(length, dtype=torch.float32)
    close = (time.sin() * .02 + time * .001).exp() * (100 + index)
    series = close[:, None].expand(-1, 5).clone()
    series[:, 4] = 1000 + time
    return {
        "asset_series": series, "benchmark_series": series * 2,
        "asset_timestamp_features": torch.zeros(length, 5, dtype=torch.long),
        "benchmark_timestamp_features": torch.zeros(length, 5, dtype=torch.long),
        "target_alpha": torch.arange(14, dtype=torch.float32) * .002 + index * .02,
        "sample_id": str(index), "symbol": f"S{index}", "benchmark_symbol": "VTI.US",
        "asset_type": "stock", "market": "US", "provider": "fixture",
        "dataset_profile": "fixture", "cutoff_at": "2025-01-03", "diagnostics": {},
    }


def _legacy_ranking(predictions, targets, scales, groups, securities, max_pairs):
    # Frozen pre-optimization formula, including device RNG consumption/order.
    candidates = torch.arange(len(targets), device=targets.device)
    if len(candidates) > 512:
        candidates = candidates[torch.randperm(len(candidates), device=targets.device)[:512]]
    pairs = candidates[
        torch.triu_indices(len(candidates), len(candidates), offset=1, device=targets.device)
    ]
    valid = (groups[pairs[0]] == groups[pairs[1]]) & (securities[pairs[0]] != securities[pairs[1]])
    pairs = pairs[:, valid]
    if pairs.shape[1] == 0:
        return predictions.sum() * 0
    if pairs.shape[1] > max_pairs:
        pairs = pairs[:, torch.randperm(pairs.shape[1], device=pairs.device)[:max_pairs]]
    left, right = pairs
    delta = (targets[left].float() - targets[right].float()) / scales
    gap = (predictions[left, :, 1].float() - predictions[right, :, 1].float()) / scales
    valid = torch.isfinite(delta) & (delta.abs() > .01)
    return F.softplus(-delta.nan_to_num().sign() * gap).masked_fill(~valid, 0).sum() / (
        valid.sum().clamp_min(1)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size,max_pairs", [(1, 8), (8, 256), (32, 16), (128, 256), (513, 32)])
@pytest.mark.parametrize("empty", [False, True])
def test_prepared_ranking_preserves_exact_loss_gradients_and_rng(device, size, max_pairs, empty):
    groups = torch.arange(size) if empty else torch.arange(size) % 3
    securities = torch.arange(size) // 2
    pairs = eligible_ranking_pairs(groups, securities)
    if size > 512:
        assert pairs is None
    torch.manual_seed(27)
    prediction = torch.randn(size, 14, 3, device=device, requires_grad=True)
    target = torch.randn(size, 14, device=device)
    target[0, 0] = float("nan")
    scales = torch.linspace(.01, .1, 14, device=device)
    groups, securities = groups.to(device), securities.to(device)
    pairs = None if pairs is None else pairs.to(device)
    outputs, gradients, rngs = [], [], []
    for legacy in (True, False):
        torch.manual_seed(19)
        loss = (
            _legacy_ranking(prediction, target, scales, groups, securities, max_pairs)
            if legacy else same_date_ranking_loss(
                prediction, target, scales, groups, securities, max_pairs, eligible_pairs=pairs,
            )
        )
        outputs.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, prediction)[0])
        rngs.append(torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state())
    for left, right in ((outputs[0], outputs[1]), (gradients[0], gradients[1]), (rngs[0], rngs[1])):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("size", [1, 32, 128, 512, 513])
def test_pair_memory_budget_is_bounded_and_accounts_for_quadratic_metadata(size):
    collator = training.ModelBatchCollator()
    samples = [_sample(i) for i in range(size)]
    batch = collator(samples)
    assert collator.estimated_batch_bytes(samples[0], size) >= training._batch_tensor_bytes(batch)
    assert ("ranking_pairs" in batch) == (size <= 512)
    evaluation = training.ModelBatchCollator(prepare_ranking=False)(samples)
    assert "ranking_pairs" not in evaluation


@pytest.mark.parametrize("invalid", ["close", "volume", "length", "mask"])
def test_invalid_scale_inputs_fail_in_cpu_worker_before_h2d(invalid):
    sample = _sample(length=60 if invalid == "length" else 128)
    if invalid == "close":
        sample["asset_series"][0, 3] = float("nan")
    elif invalid == "volume":
        sample["benchmark_series"][0, 4] = -1
    elif invalid == "mask":
        sample["benchmark_series"] = sample["benchmark_series"][:-1]
        sample["benchmark_timestamp_features"] = sample["benchmark_timestamp_features"][:-1]
    with pytest.raises(ValueError):
        training.ModelBatchCollator(validate_scales=True, extended_scales=True)([sample])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("h_start", [1, 3])
def test_nonpersistent_constants_preserve_pinball_and_checkpoint_keys(device, h_start):
    head = MultiHorizonAlphaHead(32, horizons=tuple(range(h_start, 15))).to(device)
    assert not any(key.startswith(("_horizon", "_quantile")) for key in head.state_dict())
    # Integer buffers must retain exact levels when the entire model changes dtype.
    head.to(dtype=torch.bfloat16)
    for dtype in (torch.float32, torch.bfloat16):
        torch.testing.assert_close(
            head._horizon_values.to(dtype=dtype).sqrt(),
            torch.tensor(head.horizons, dtype=dtype, device=device).sqrt(), rtol=0, atol=0,
        )
    predictions = torch.randn(4, len(head.horizons), 3, device=device, requires_grad=True)
    target = torch.randn(4, len(head.horizons), device=device)
    levels = predictions.new_tensor((.1, .5, .9))
    errors = (target[..., None] - predictions) / head.robust_scales.float()[None, :, None]
    expected = (
        torch.maximum(levels * errors, (levels - 1) * errors).sum()
        / torch.ones_like(errors).sum()
    )
    actual = head.pinball_loss(predictions, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, predictions)[0],
                               torch.autograd.grad(expected, predictions)[0], rtol=0, atol=0)


def _model_case(device):
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.input_length = 128
    config.data.h_start = 1
    config.model.feature_mode = "combined"
    config.model.market_aware = True
    config.model.explicit_output_scale = True
    config.model.ranking_loss_weight = .05
    config.model.ranking_max_pairs = 16
    config.model.resampler_dropout = .1
    config.model.benchmark_conditioner_dropout = .1
    config.model.alpha_head_dropout = .1
    samples = [_sample(i) for i in range(32)]
    collator = training.ModelBatchCollator.for_model(config, training=True)
    batch = collator(samples)
    features = historical_scale_features(
        batch["asset_series"], batch["benchmark_series"], extended=True,
    )
    statistics = fit_scale_feature_statistics(features.numpy(), {"split": "train"})
    torch.manual_seed(57)
    bundle = build_model_bundle(config, torch.device(device), robust_scales=[.03] * 14,
                                scale_feature_statistics=statistics)
    return config, bundle, training._move_batch_to_device(batch, torch.device(device))


@pytest.mark.parametrize("device", DEVICES)
def test_full_model_fast_path_preserves_predictions_gradients_and_dropout_rng(device):
    config, bundle, batch = _model_case(device)
    outputs, gradients, rngs = [], [], []
    # Identical model/data/random state with and without prevalidated worker metadata.
    for fast in (False, True):
        torch.manual_seed(18)
        bundle.model.zero_grad(set_to_none=True)
        sample = dict(batch)
        if not fast:
            sample.pop("scale_inputs_validated")
            sample.pop("ranking_pairs")
        output = training.forward_batch(bundle, sample, config, torch.device(device))
        output.loss.backward()
        outputs.append((output.alpha_quantiles.detach(), output.loss.detach()))
        gradients.append({name: p.grad.clone() for name, p in bundle.model.named_parameters()
                          if p.grad is not None})
        rngs.append(torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state())
    for left, right in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
    torch.testing.assert_close(rngs[0], rngs[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
@pytest.mark.parametrize("training_mode", [False, True])
def test_cuda_forward_backward_has_no_host_synchronization(training_mode):
    config, bundle, batch = _model_case("cuda")
    bundle.model.train(training_mode)
    original = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = training.forward_batch(bundle, batch, config, torch.device("cuda"))
        if training_mode:
            output.loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode(original)
    torch.cuda.synchronize()
    assert torch.isfinite(output.alpha_quantiles).all()


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_and_dropout_resume_across_worker_hotpath(tmp_path, device):
    config, bundle, batch = _model_case(device)
    model = bundle.model
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    final = []
    for interrupted in (False, True):
        model.load_state_dict(initial, strict=True)
        model.train()
        torch.manual_seed(92)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
        for step in range(4):
            sample = dict(batch)
            if not interrupted or step < 2:
                sample.pop("scale_inputs_validated")
                sample.pop("ranking_pairs")
            optimizer.zero_grad(set_to_none=True)
            output = training.forward_batch(bundle, sample, config, torch.device(device))
            output.loss.backward()
            optimizer.step()
            if interrupted and step == 1:
                path = tmp_path / "execution-checkpoint.pt"
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                }, path)
                restored = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(restored["model"], strict=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
                optimizer.load_state_dict(restored["optimizer"])
                torch.set_rng_state(restored["cpu_rng"])
                if device == "cuda":
                    torch.cuda.set_rng_state(restored["cuda_rng"])
        final.append({name: value.clone() for name, value in model.state_dict().items()})
    for name in final[0]:
        torch.testing.assert_close(final[0][name], final[1][name], rtol=0, atol=0, msg=name)
