"""Cloud numerical acceptance for train-prior direction signal separation."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from stock_forecasting.config import DirectionConfig, ExperimentConfig
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.direction import (
    DualDirectionHead,
    MarketResidualDualDirectionHead,
    direction_labels,
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
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA"),
    ),
]


def v2_config():
    return DirectionConfig(
        enabled=True, architecture="market_residual_v2", class_balance_power=1.0,
        market_residual_hidden_dim=16,
    )


def reference_state(horizons=3):
    priors = torch.tensor([0.7, 0.2, 0.1]).expand(4, 2, horizons, 3).clone()
    priors[1] = torch.tensor([0.2, 0.3, 0.5])
    return {
        "schema_version": 2,
        "samples": 8192,
        "references": [0.3, math.log(3), math.log(3)],
        "priors": priors.tolist(),
        "market_weights": [0.6, 1.2, 2.0, 1.0],
        "basis": "train_null_quantiles_and_uniform_classes_v2",
    }


def observed_bars(batch, device):
    changes = torch.randn(batch, 128, device=device) * 0.01
    close = (changes.cumsum(1) + 4).exp()[..., None]
    volume = (torch.randn_like(close) * 0.1 + 12).exp()
    return torch.cat((close, close * 1.01, close * 0.99, close, volume), -1)


@pytest.mark.parametrize("device", DEVICES)
def test_population_logits_prior_and_balanced_decision_are_separate(device):
    head = MarketResidualDualDirectionHead(32, [1, 5, 14], v2_config(), 0).to(device)
    head.restore_references(reference_state())
    with torch.no_grad():
        head.absolute.weight.zero_()
        head.absolute.bias.zero_()
        head.alpha.weight.zero_()
        head.alpha.bias.zero_()
    hidden = torch.randn(2, 3, 32, device=device)
    markets = torch.tensor([0, 1], device=device)
    population = head.logits(hidden, markets)
    decisions = head.decision_logits(population, markets)
    for task in range(2):
        torch.testing.assert_close(
            population[task].softmax(-1), head.train_priors[markets, task]
        )
        torch.testing.assert_close(decisions[task], torch.zeros_like(decisions[task]))
    assert population[0][0, 0].argmax().item() == 0
    assert population[0][1, 0].argmax().item() == 2


@pytest.mark.parametrize("device", DEVICES)
def test_menon_ce_has_no_second_inverse_class_weight(device):
    head = MarketResidualDualDirectionHead(32, [1, 5, 14], v2_config(), 0).to(device)
    head.restore_references(reference_state())
    head.eval()
    hidden = torch.randn(2, 3, 32, device=device, requires_grad=True)
    markets = torch.tensor([0, 2], device=device)
    bands = torch.tensor([[-0.005, 0.005]], device=device).expand(2, -1)
    targets = torch.tensor([[-0.03, 0, 0.03], [0.03, -0.03, 0]], device=device)
    logits = head.logits(hidden, markets)
    losses = head.losses(hidden, logits, (targets, targets), bands, market_ids=markets)
    labels = direction_labels(targets, bands)
    for task in range(2):
        expected = F.cross_entropy(
            logits[task].flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(2, 3)
        expected = (expected * head.train_market_weights[markets, None]).mean()
        torch.testing.assert_close(losses[task], expected)
    assert losses[2] is None


@pytest.mark.parametrize("device", DEVICES)
def test_historical_scale_and_context_paths_receive_finite_gradients(device):
    head = MarketResidualDualDirectionHead(32, [1, 5, 14], v2_config(), 0).to(device)
    head.restore_references(reference_state())
    context = torch.randn(4, 8, 32, device=device, requires_grad=True)
    standardized = torch.randn(4, 20, device=device, requires_grad=True)
    scales = torch.ones(4, 20, device=device) * 0.02
    markets = torch.tensor([0, 1, 2, 0], device=device)
    bands = torch.tensor([[-0.005, 0.005]], device=device).expand(4, -1).clone()
    bands[0, 0] = -torch.inf
    hidden = head.read(
        context, context * 0.7, context * 1.2, markets, bands,
        standardized_scales=standardized, scale_features=scales,
    )
    logits = head.logits(hidden, markets)
    targets = torch.tensor([[-0.02, 0, 0.03]], device=device).expand(4, -1)
    losses = head.losses(hidden, logits, (targets, targets), bands, market_ids=markets)
    sum(loss for loss in losses if loss is not None).backward()
    for value in (*logits, hidden, context.grad, standardized.grad):
        assert torch.isfinite(value).all()
    assert context.grad.abs().sum() > 0 and standardized.grad.abs().sum() > 0
    assert losses[2] is not None
    head.eval()
    with torch.no_grad():
        original = head.read(
            context, context, context, markets, bands,
            standardized_scales=standardized, scale_features=scales,
        )
        changed = head.read(
            context * 0.0, context * 0.0, context * 0.0, markets, bands,
            standardized_scales=standardized, scale_features=scales,
        )
        assert not torch.allclose(original, changed)
        # Market IDs route priors/output experts, never the history reader query.
        rerouted = head.read(
            context, context, context, markets.roll(1), bands,
            standardized_scales=standardized, scale_features=scales,
        )
        torch.testing.assert_close(original, rerouted)


def test_v2_reference_roundtrip_and_guard_freeze_remain_train_bound():
    head = MarketResidualDualDirectionHead(32, [1, 5, 14], v2_config(), 0)
    head.restore_references(reference_state())
    head.set_pinball_guard_reference(0.27)
    state = head.reference_state()
    clone = MarketResidualDualDirectionHead(32, [1, 5, 14], v2_config(), 0)
    clone.restore_references(state)
    assert clone.reference_state() == state
    with pytest.raises(ValueError, match="already frozen"):
        clone.set_pinball_guard_reference(0.26)
    for change in ({"schema_version": 1}, {"market_weights": [1, 0, 1, 1]}):
        with pytest.raises(ValueError):
            clone.restore_references({**state, **change})
    malformed = copy.deepcopy(state)
    malformed["priors"][0][0][0] = [0.1, 0.1, 0.1]
    with pytest.raises(ValueError):
        clone.restore_references(malformed)
    assert "train_priors" not in clone.state_dict()
    assert "loss_references" not in clone.state_dict()


@pytest.mark.parametrize("device", DEVICES)
def test_quant_model_reuses_scale_statistics_and_preserves_legacy_outputs(device):
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.input_length = 128
    config.model.feature_mode = "combined"
    config.model.explicit_output_scale = True
    config.model.decoupled_output_scale = True
    config.model.market_aware = True
    config.model.ranking_loss_weight = 0
    config.model.direction = v2_config()
    asset, benchmark = observed_bars(4, device), observed_bars(4, device)
    raw = historical_scale_features(asset, benchmark, extended=True)
    stats = fit_scale_feature_statistics(raw.detach().cpu().numpy(), {"split": "train"})
    bundle = build_model_bundle(config, torch.device(device), scale_feature_statistics=stats)
    head = bundle.model.direction_head
    head.restore_references(reference_state(len(config.data.alpha_horizons)))
    markets = torch.tensor([0, 1, 2, 0], device=device)
    bands = torch.tensor([[-0.005, 0.005]], device=device).expand(4, -1)
    targets = torch.randn(4, len(config.data.alpha_horizons), device=device) * 0.02
    output = bundle.model(
        asset, benchmark, market_ids=markets, direction_bands=bands,
        target_alpha=targets, target_absolute=targets + 0.01,
    )
    assert output.alpha_quantiles.shape == (4, len(config.data.alpha_horizons), 3)
    assert output.absolute_direction_decision_logits.shape == output.absolute_direction_logits.shape
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert bundle.model.backbone.technical_adapter.projection.weight.grad.abs().sum() > 0
    assert bundle.model.alpha_head.numeric_branch.statistics == stats
    bundle.model.eval()
    with torch.inference_mode():
        prediction = bundle.model(asset, benchmark, market_ids=markets, direction_bands=bands)
    assert prediction.loss is None
    torch.testing.assert_close(
        prediction.absolute_direction_logits - prediction.absolute_direction_decision_logits,
        head.train_priors[markets, 0].log(),
    )
    legacy = DualDirectionHead(32, 3, DirectionConfig(enabled=True), 0)
    assert legacy.reference_state()["schema_version"] == 1
    assert set(legacy.state_dict()) == {
        "stream", "query.weight", "market.weight", "cost_projection.weight",
        "attention.in_proj_weight", "attention.in_proj_bias", "attention.out_proj.weight",
        "attention.out_proj.bias", "norm.weight", "norm.bias", "absolute.weight",
        "absolute.bias", "alpha.weight", "alpha.bias",
    }
