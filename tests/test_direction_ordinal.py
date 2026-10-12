"""Cloud CPU/CUDA acceptance for ordered probabilities and state-guided reading."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stock_forecasting.config import DirectionConfig, ExperimentConfig
from stock_forecasting.direction_ordinal_runtime import (
    DirectionOrdinalMetrics,
    decision_audit,
    ordinal_reference,
)
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.direction import direction_labels
from stock_forecasting.models.direction_ordinal import (
    MARKET_STATE_INDICES,
    MarketOrdinalDualDirectionHead,
    conditional_log_probabilities,
)
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Cloud CUDA required"),
    ),
]


def config():
    return DirectionConfig(
        enabled=True, architecture="market_ordinal_v3", market_residual_hidden_dim=16
    )


def reference(horizons=3):
    return ordinal_reference(
        {
            "samples": 32,
            "references": [0.3, math.log(3), math.log(3)],
            "priors": torch.tensor([0.6, 0.1, 0.3]).expand(4, 2, horizons, 3).tolist(),
            "market_weights": [1, 1, 1, 1],
            "pinball_guard_reference": 0.3,
            "pinball_guard_basis": "first_full_validation",
        }
    )


@pytest.mark.parametrize("device", DEVICES)
def test_chain_is_normalized_monotone_and_stable_at_extremes(device):
    logits = torch.tensor(
        [[0, 0], [10000, -10000], [-10000, 10000], [10000, 10000]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    logp = conditional_log_probabilities(logits)
    assert torch.isfinite(logp).all()
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(4, device=device))
    assert (logp.exp()[:, 2] <= logp.exp()[:, 1:].sum(-1)).all()
    assert torch.isfinite(F.nll_loss(logp, torch.tensor([0, 1, 0, 2], device=device)))
    F.nll_loss(logp, torch.tensor([2, 2, 2, 0], device=device)).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
def test_nll_equals_masked_conditional_binary_likelihood(device, labels):
    logits = torch.randn(3, 2, device=device, requires_grad=True)
    y = torch.tensor(labels, device=device)
    actual = F.nll_loss(conditional_log_probabilities(logits), y)
    expected = (
        F.binary_cross_entropy_with_logits(logits[:, 0], (y > 0).float(), reduction="none")
        + (y > 0)
        * F.binary_cross_entropy_with_logits(logits[:, 1], (y > 1).float(), reduction="none")
    ).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    if all(v == 0 for v in labels):
        assert logits.grad[:, 1].abs().sum() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_zero_signal_matches_train_population_prior_not_two_half_probabilities(device):
    head = MarketOrdinalDualDirectionHead(32, [1, 5, 14], config(), 0).to(device)
    head.restore_references(reference())
    with torch.no_grad():
        for layer in (head.absolute, head.alpha):
            layer.weight.zero_()
            layer.bias.zero_()
    markets = torch.tensor([0, 1], device=device)
    logits = head.logits(torch.randn(2, 3, 32, device=device), markets)
    for task, logp in enumerate(logits):
        torch.testing.assert_close(logp.exp(), head.train_priors[markets, task])
        torch.testing.assert_close(logp.exp(), logp.softmax(-1))
    clone = MarketOrdinalDualDirectionHead(32, [1, 5, 14], config(), 0).to(device)
    clone.load_state_dict(head.state_dict())
    clone.restore_references(head.reference_state())
    assert clone.reference_state() == head.reference_state()
    with pytest.raises(ValueError, match="ordinal-v3"):
        clone.restore_references({**reference(), "schema_version": 2})


@pytest.mark.parametrize("device", DEVICES)
def test_gates_respond_only_to_market_state_and_are_bounded(device):
    head = MarketOrdinalDualDirectionHead(32, [1, 5, 14], config(), 0).to(device)
    x = torch.randn(4, 20, device=device, requires_grad=True)
    stream, feature = head.feature_gates(x)
    torch.testing.assert_close(stream.sum(-1), torch.full((4,), 3.0, device=device))
    assert ((feature > 0) & (feature < 2)).all()
    stock_changed = x.detach().clone()
    other = [i for i in range(20) if i not in MARKET_STATE_INDICES]
    stock_changed[:, other] += 10
    for a, b in zip((stream, feature), head.feature_gates(stock_changed), strict=True):
        torch.testing.assert_close(a, b)
    state_changed = x.detach().clone()
    state_changed[:, MARKET_STATE_INDICES] += 5
    assert not torch.allclose(feature, head.feature_gates(state_changed)[1])
    feature.sum().backward()
    assert x.grad[:, MARKET_STATE_INDICES].abs().sum() > 0
    assert x.grad[:, other].abs().sum() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_full_model_preserves_alpha_and_receives_direction_gradients(device):
    cfg = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    cfg.model.feature_mode = "combined"
    cfg.model.explicit_output_scale = True
    cfg.model.decoupled_output_scale = True
    cfg.model.market_aware = True
    cfg.model.direction = config()
    close = torch.exp(torch.randn(4, 128, 1, device=device).cumsum(1) * 0.01 + 4)
    bars = torch.cat([close, close * 1.01, close * 0.99, close, torch.ones_like(close) * 1e6], -1)
    benchmark = bars.clone()
    benchmark[..., :4] *= 1.02
    scales = historical_scale_features(bars, benchmark, extended=True)
    statistics = fit_scale_feature_statistics(scales.cpu().numpy(), {"split": "train"})
    bundle = build_model_bundle(cfg, torch.device(device), scale_feature_statistics=statistics)
    bundle.model.direction_head.restore_references(reference(len(cfg.data.alpha_horizons)))
    targets = torch.randn(4, len(cfg.data.alpha_horizons), device=device) * 0.02
    bands = torch.tensor(
        [[-float("inf"), 2], [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]], device=device
    )
    output = bundle.model(
        bars,
        benchmark,
        market_ids=torch.tensor([0, 1, 2, 0], device=device),
        direction_bands=bands,
        target_alpha=targets,
        target_absolute=targets + 0.01,
    )
    assert output.alpha_quantiles.shape[-1] == 3 and torch.isfinite(output.loss)
    assert direction_labels(targets[:1], bands[:1]).eq(1).all()
    output.direction_weighted_loss.backward(retain_graph=True)
    assert bundle.model.backbone.technical_adapter.projection.weight.grad.abs().sum() > 0
    assert bundle.model.direction_head.state_gate[-1].weight.grad.abs().sum() > 0
    bundle.model.zero_grad(set_to_none=True)
    output.loss.backward()
    assert torch.isfinite(bundle.model.direction_head.absolute.weight.grad).all()


def _metrics(probabilities, *, labels=None, markets=None):
    size = len(probabilities)
    labels = np.tile(np.arange(size) % 3, (2, 1)).T[:, :, None] if labels is None else labels
    metrics = DirectionOrdinalMetrics([5])
    metrics.append_probabilities(
        labels,
        probabilities,
        probabilities.argmax(-1),
        markets or ["US"] * size,
        ["2025-09-01"] * size,
        pinball=0.3,
    )
    head = SimpleNamespace(
        reference_state=lambda: reference(1),
        weights=lambda: torch.tensor([0.3 / math.log(3)] * 2),
        config=config(),
    )
    return metrics.finish(head, SimpleNamespace(model_dump=lambda: {"scenario": "test"}))


def test_selection_penalizes_overconfidence_with_same_argmax_and_balanced_accuracy():
    y = np.tile(np.arange(6) % 3, (2, 1)).T[:, :, None]
    predicted = y.copy()
    predicted[:3] = (predicted[:3] + 1) % 3

    def probabilities(confidence):
        p = np.full((6, 2, 1, 3), (1 - confidence) / 2)
        np.put_along_axis(p, predicted[..., None], confidence, -1)
        return p

    modest, excessive = _metrics(probabilities(0.6)), _metrics(probabilities(0.999))
    assert (
        modest["balanced_decision"]["market_macro"]
        == excessive["balanced_decision"]["market_macro"]
    )
    assert excessive["selection_loss"] > modest["selection_loss"]
    assert excessive["schema_version"] == 3
    assert (
        excessive["selection_definition"]["balanced_accuracy_role"]
        == "diagnostic_only_not_selection_objective"
    )


def test_audit_separates_opposite_errors_from_neutral_errors_and_reports_collapse():
    opposite = decision_audit(
        [[0, 0, 10], [0, 10, 0], [10, 0, 0]], support_min=0.05, prediction_min=0.01
    )
    adjacent = decision_audit(
        [[0, 10, 0], [0, 10, 0], [0, 10, 0]], support_min=0.05, prediction_min=0.01
    )
    assert opposite["balanced_opposite_error"] == 1
    assert adjacent["balanced_opposite_error"] == 0
    assert adjacent["collapsed_classes"] == ["down", "up"]
    empty = decision_audit(np.zeros((3, 3)), support_min=0.05, prediction_min=0.01)
    assert empty["collapse_deficit"] == 0


def test_market_selection_is_not_dominated_by_duplicate_us_rows():
    labels = np.asarray([0, 1, 2] * 2)[:, None, None].repeat(2, axis=1)
    p = np.full((6, 2, 1, 3), 0.1)
    np.put_along_axis(p, labels[..., None], 0.8, -1)
    p[3:] = 0.05
    p[3:, :, :, 0] = 0.9
    first = _metrics(p, labels=labels, markets=["US"] * 3 + ["TWSE"] * 3)
    indices = [0, 1, 2] * 50 + [3, 4, 5]
    repeated = _metrics(p[indices], labels=labels[indices], markets=["US"] * 150 + ["TWSE"] * 3)
    assert first["selection_loss"] == pytest.approx(repeated["selection_loss"])


def test_v3_config_presets_and_only_v3_selection_fields_affect_its_digest():
    for group in ("a", "b"):
        cfg = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{group}_direction64_v3.yaml")
        assert cfg.model.direction.architecture == "market_ordinal_v3"
        assert cfg.model.lora.rank == 64 and cfg.model.lora.alpha == 128
    v2 = ExperimentConfig.from_yaml(ROOT / "configs/experiments/a_direction64_v2.yaml")
    same = copy.deepcopy(v2)
    same.model.direction.selection_collapse_weight = 1
    assert v2.model_architecture_digest() == same.model_architecture_digest()
    old = cfg.model_architecture_digest()
    cfg.model.direction.selection_collapse_weight = 1
    assert cfg.model_architecture_digest() != old
