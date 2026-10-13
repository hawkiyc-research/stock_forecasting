"""Cloud numerical contracts for independently optimized signal tasks."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stock_forecasting.models.forecast import GatedBenchmarkConditioner
from stock_forecasting.models.lora import LoRALinear
from stock_forecasting.models.outputs import TimeSeriesBackboneOutput
from stock_forecasting.models.projector import CausalPerceiverResampler
from stock_forecasting.models.quant import QuantForecastModel
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)
from stock_forecasting.models.signal import (
    SignalForecastModel,
    SingleTaskHead,
    fixed_direction_labels,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA"),
    ),
]
HORIZONS = tuple(range(1, 15))


class TinyLoRABackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(5, 16), rank=4, alpha=8, dropout=0)
        self.calls = 0

    def forward(self, ohlcv, *, attention_mask=None, timestamps=None):
        del timestamps
        self.calls += 1
        if attention_mask is None:
            attention_mask = torch.ones(ohlcv.shape[:2], dtype=torch.bool, device=ohlcv.device)
        hidden = self.projection(ohlcv.log1p())
        return TimeSeriesBackboneOutput(hidden, attention_mask)


def bars(device="cpu"):
    time = torch.arange(128, device=device, dtype=torch.float32)[None, :, None]
    offset = torch.arange(4, device=device, dtype=torch.float32)[:, None, None]
    close = (4 + 0.003 * time + 0.04 * (time * 0.3 + offset).sin()).exp()
    asset = torch.cat(
        (close, close * 1.01, close * 0.99, close, torch.ones_like(close) * 1e6), dim=-1
    )
    benchmark_close = (4.3 + 0.001 * time + 0.015 * (time * 0.2 + offset).cos()).exp()
    benchmark = torch.cat(
        (
            benchmark_close,
            benchmark_close * 1.01,
            benchmark_close * 0.99,
            benchmark_close,
            torch.ones_like(close) * 2e6,
        ),
        dim=-1,
    )
    return asset, benchmark


def model(task="alpha_point", device="cpu"):
    torch.manual_seed(17)
    head = SingleTaskHead(
        16,
        prediction_task=task,
        horizons=HORIZONS,
        robust_scales=[0.02 * math.sqrt(h) for h in HORIZONS],
        feature_mode="combined",
        explicit_output_scale=True,
        market_aware=True,
    )
    instance = SignalForecastModel(
        TinyLoRABackbone(),
        CausalPerceiverResampler(16, 16, num_soft_tokens=4, num_heads=2, depth=1),
        GatedBenchmarkConditioner(16, num_heads=2),
        head,
    ).to(device)
    asset, benchmark = bars(device)
    values = historical_scale_features(asset, benchmark, extended=True).cpu().numpy()
    head.numeric_branch.set_statistics(fit_scale_feature_statistics(values, {"split": "train"}))
    return instance


def batch(device="cpu"):
    asset, benchmark = bars(device)
    target = torch.linspace(-0.04, 0.05, 56, device=device).reshape(4, 14)
    return dict(
        asset_ohlcv=asset,
        benchmark_ohlcv=benchmark,
        target_alpha=target,
        target_absolute=target + 0.003,
        market_ids=torch.tensor([0, 1, 2, 3], device=device),
    )


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
@pytest.mark.parametrize("device", DEVICES)
def test_single_task_output_gradients_and_paired_encoding(task, device, monkeypatch):
    instance = model(task, device)
    kwargs = batch(device)

    def legacy_forward_must_not_run(*args, **kw):
        raise AssertionError("Single-task model called the legacy multi-task forward")

    monkeypatch.setattr(QuantForecastModel, "forward", legacy_forward_must_not_run)
    output = instance(**kwargs)
    assert instance.backbone.calls == 1
    assert instance.prediction_task == task and instance.direction_head is None
    assert "alpha_quantiles" not in output and "pinball_loss" not in output
    assert "ranking_loss" not in output and "alpha_direction_logits" not in output
    assert output.scale_features.shape == (4, 20)
    assert torch.isfinite(output.loss)
    if task == "alpha_point":
        assert output.alpha_point.shape == (4, 14)
        assert output.absolute_direction_logits is None and output.classification_loss is None
        assert output.loss is output.point_loss
    else:
        assert output.absolute_direction_logits.shape == (4, 14, 3)
        assert output.alpha_point is None and output.point_loss is None
        assert output.loss is output.classification_loss
    output.loss.backward()
    for name in (
        "backbone.projection.lora_b.weight",
        "resampler.context_projection.weight",
        "alpha_head.prediction.weight",
        "alpha_head.numeric_branch.fusion.2.weight",
    ):
        grad = dict(instance.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert instance.backbone.projection.base_layer.weight.grad is None
    assert all(torch.isfinite(p.grad).all() for p in instance.parameters() if p.grad is not None)


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_no_cost_or_other_target_dependency_and_target_free_inference(task):
    instance = model(task).eval()
    kwargs = batch()
    with torch.inference_mode():
        original = instance(**kwargs)
        changed = dict(kwargs)
        ignored_target = "target_absolute" if task == "alpha_point" else "target_alpha"
        changed[ignored_target] = torch.full((4, 14), float("nan"))
        changed["direction_bands"] = torch.tensor([[float("-inf"), 99.0]]).expand(4, -1)
        alternate = instance(**changed)
        torch.testing.assert_close(original.loss, alternate.loss, rtol=0, atol=0)
        without_targets = dict(kwargs)
        without_targets.pop("target_alpha")
        without_targets.pop("target_absolute")
        inference = instance(**without_targets)
        assert inference.loss is None
        actual = (
            inference.alpha_point if task == "alpha_point" else inference.absolute_direction_logits
        )
        expected = (
            original.alpha_point if task == "alpha_point" else original.absolute_direction_logits
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_direction_boundaries_are_absolute_simple_returns_with_inclusive_flat():
    threshold = 0.01
    lower, upper = math.log1p(-threshold), math.log1p(threshold)
    target = torch.tensor([[lower - 1e-5, lower, 0, upper, upper + 1e-5]])
    assert fixed_direction_labels(target, threshold).tolist() == [[0, 1, 1, 1, 2]]
    stock = torch.tensor([[math.log(0.96)]])
    alpha = stock - math.log(0.92)
    assert fixed_direction_labels(stock).item() == 0
    assert fixed_direction_labels(alpha).item() == 2
    for invalid in (0, -0.01, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="neutral_return_threshold"):
            fixed_direction_labels(target, invalid)


def test_point_smooth_l1_exact_scaling_equal_horizons_and_bounded_residual_gradient():
    head = SingleTaskHead(
        16, prediction_task="alpha_point", horizons=HORIZONS, robust_scales=[0.1] * 14
    )
    point = torch.full((2, 14), 0.05, requires_grad=True)
    with torch.no_grad():
        point[0, 0] = 1000
        point[1, 0] = -1000
    target = torch.zeros_like(point)
    loss = head.point_loss(point, target)
    expected = F.smooth_l1_loss(point / 0.1, target, beta=1)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(point.grad[0, 0], torch.tensor(1 / (28 * 0.1)))
    torch.testing.assert_close(point.grad[1, 0], torch.tensor(-1 / (28 * 0.1)))
    torch.testing.assert_close(point.grad[0, 1], torch.tensor(0.5 / (28 * 0.1)))
    assert torch.isfinite(point.grad).all()
    sparse_target = target.detach().clone()
    sparse_target[1, 0] = float("nan")
    elementwise = F.smooth_l1_loss(point.detach() / 0.1, target, reduction="none")
    expected = (elementwise[0, 0] + elementwise[:, 1:].mean(0).sum()) / 14
    torch.testing.assert_close(head.point_loss(point, sparse_target), expected)


def test_fixed_training_scales_control_point_units_not_sample_volatility():
    head = SingleTaskHead(
        16, prediction_task="alpha_point", horizons=HORIZONS, robust_scales=[0.02] * 14
    )
    instance = model()
    head.numeric_branch.set_statistics(instance.alpha_head.numeric_branch.statistics)
    with torch.no_grad():
        head.prediction.weight.zero_()
        head.prediction.bias.fill_(2)
    tokens = torch.randn(4, 4, 16)
    features = historical_scale_features(*bars(), extended=True)
    kwargs = dict(benchmark_tokens=tokens, market_ids=torch.arange(4))
    first = head(tokens, scale_features=features, **kwargs)
    changed = features.clone()
    changed[:, 2] *= 100
    second = head(tokens, scale_features=changed, **kwargs)
    torch.testing.assert_close(first, torch.full((4, 14), 0.04))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_classifier_is_plain_cross_entropy_and_handles_missing_labels():
    head = SingleTaskHead(16, prediction_task="absolute_direction", horizons=HORIZONS)
    logits = torch.randn(4, 14, 3, requires_grad=True)
    target = batch()["target_absolute"]
    expected = F.cross_entropy(logits.reshape(-1, 3), head.direction_labels(target).reshape(-1))
    torch.testing.assert_close(head.classification_loss(logits, target), expected)
    missing = torch.full_like(target, float("nan"))
    empty = head.classification_loss(logits, missing)
    assert empty.item() == 0
    empty.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() == 0


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
@pytest.mark.parametrize("device", DEVICES)
def test_real_model_loss_matches_streaming_evaluation_under_autocast(task, device):
    from stock_forecasting.signal_evaluation import SignalMoments

    instance = model(task, device).eval()
    kwargs = batch(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        output = instance(**kwargs)
    prediction = output.alpha_point if task == "alpha_point" else output.absolute_direction_logits
    target = kwargs["target_alpha"] if task == "alpha_point" else kwargs["target_absolute"]
    moments = SignalMoments(HORIZONS, instance.alpha_head.robust_scales.cpu().tolist(), task)
    moments.add(target.detach().cpu().numpy(), prediction.detach().float().cpu().numpy())
    assert prediction.dtype == torch.float32
    assert output.loss.dtype == torch.float32
    assert moments.result()["aggregate"]["selection_score"] == pytest.approx(
        output.loss.detach().item(), rel=2e-6, abs=1e-7
    )
    output.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in instance.parameters() if p.grad is not None)


def test_confident_wrong_classification_stays_finite_without_clipping_logits():
    head = SingleTaskHead(16, prediction_task="absolute_direction", horizons=HORIZONS)
    logits = torch.tensor([-10000.0, 0.0, 10000.0]).expand(4, 14, 3).clone().requires_grad_()
    target = torch.full((4, 14), -0.1)
    loss = head.classification_loss(logits, target)
    assert loss.item() == pytest.approx(20000.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() <= 1 / 56 + 1e-8


def test_task_instances_have_disjoint_parameters_and_optimizer_updates():
    regression, classification = model(), model("absolute_direction")
    before = copy.deepcopy(classification.state_dict())
    assert {id(p) for p in regression.parameters()}.isdisjoint(
        id(p) for p in classification.parameters()
    )
    optimizer = torch.optim.AdamW(regression.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        regression(**batch()).loss.backward()
        optimizer.step()
    for name, value in classification.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(p.grad is None for p in classification.parameters())
    # The initially zero residual output layer must not permanently block features.
    for name in ("scale_projection.0.weight", "benchmark_projection.weight"):
        parameter = dict(regression.alpha_head.numeric_branch.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
def test_model_optimizer_save_restore_next_update_matches(task):
    instance = model(task)
    optimizer = torch.optim.AdamW(instance.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        instance(**batch()).loss.backward()
        optimizer.step()
    state = copy.deepcopy(instance.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = model(task)
    restored.load_state_dict(state, strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(optimizer_state)
    for network, optim in ((instance, optimizer), (restored, resumed_optimizer)):
        optim.zero_grad(set_to_none=True)
        network(**batch()).loss.backward()
        optim.step()
    for name, value in instance.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_mask_validation_and_separate_encoding_keep_the_pair_contract():
    instance = model().eval()
    kwargs = batch()
    with torch.inference_mode():
        fused = instance(**kwargs)
        separate = instance(**kwargs, asset_timestamps=torch.zeros(4, 128, 5))
    assert instance.backbone.calls == 3
    torch.testing.assert_close(fused.alpha_point, separate.alpha_point, rtol=1e-5, atol=1e-6)
    mask = torch.ones(4, 128, dtype=torch.bool)
    other = mask.clone()
    other[0, -1] = False
    with pytest.raises(ValueError, match="aligned"):
        instance(**kwargs, asset_attention_mask=mask, benchmark_attention_mask=other)
    with pytest.raises(ValueError, match="aligned"):
        instance(**kwargs, asset_attention_mask=mask)


@pytest.mark.parametrize(
    "overrides",
    (
        {"prediction_task": "quantiles"},
        {"point_loss_beta": 0},
        {"neutral_return_threshold": 1},
        {"robust_scales": [0.1]},
        {"robust_scales": [float("nan")] * 14},
        {"feature_mode": "baseline"},
    ),
)
def test_invalid_head_configuration_fails_early(overrides):
    arguments = dict(prediction_task="alpha_point", horizons=HORIZONS)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        SingleTaskHead(16, **arguments)
