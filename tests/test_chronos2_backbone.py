"""Cloud-only numerical tests against the actual pinned Chronos-2 implementation."""

from __future__ import annotations

import pytest
import torch

from stock_forecasting.models.chronos2 import Chronos2Backbone, verify_chronos2_dependency

chronos = pytest.importorskip("chronos", reason="Enable the cloud-only chronos2 Poetry extra")
from chronos.chronos2.config import Chronos2CoreConfig


def tiny_backbone(device="cpu"):
    verify_chronos2_dependency()
    torch.manual_seed(59)
    config = Chronos2CoreConfig(
        d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, dropout_rate=0,
        chronos_config={
            "context_length": 128, "input_patch_size": 16, "input_patch_stride": 16,
            "output_patch_size": 16, "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": True, "use_arcsinh": True, "time_encoding_scale": 8192,
        },
    )
    return Chronos2Backbone(chronos.Chronos2Model(config), max_context=128).to(device)


def inputs(device="cpu"):
    generator = torch.Generator().manual_seed(17)
    values = 10 + torch.rand(3, 128, 5, generator=generator)
    values[..., 4] *= 100000
    return values.to(device)


def test_native_encoder_equivalence_no_future_targets_and_patch_order(monkeypatch):
    backbone = tiny_backbone()
    values = inputs()
    original = backbone.model.encode
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        assert "future_target" not in kwargs
        assert "future_covariates" not in kwargs
        return original(**kwargs)

    monkeypatch.setattr(backbone.model, "encode", spy)
    output = backbone(values)
    assert output.last_hidden_state.shape == (3, 40, 32)
    assert output.attention_mask.all()
    assert torch.isfinite(output.last_hidden_state).all()
    assert not output.last_hidden_state.requires_grad
    assert calls[0]["group_ids"].tolist() == [0] * 5 + [1] * 5 + [2] * 5
    with torch.no_grad():
        native, _, _, patches = original(**calls[0])
    expected = native.last_hidden_state[:, :patches].reshape(3, 5, 8, 32)
    expected = expected.permute(0, 2, 1, 3).reshape(3, 40, 32)
    torch.testing.assert_close(output.last_hidden_state, expected)


def test_other_windows_and_batch_order_cannot_change_a_prediction():
    backbone = tiny_backbone().eval()
    values = inputs()
    original = backbone(values).last_hidden_state
    changed = values.clone()
    changed[1:] = torch.sin(changed[1:]) * 10000
    torch.testing.assert_close(backbone(changed).last_hidden_state[0], original[0])
    torch.testing.assert_close(backbone(values[:1]).last_hidden_state[0], original[0])
    permutation = torch.tensor([2, 0, 1])
    reordered = backbone(values[permutation]).last_hidden_state
    torch.testing.assert_close(reordered, original[permutation])


def test_masked_padding_does_not_enter_native_normalization():
    backbone = tiny_backbone().eval()
    values = inputs()
    mask = torch.ones(3, 128, dtype=torch.bool)
    mask[:, 73:] = False
    changed = values.clone()
    changed[:, 73:] = 1e12
    first = backbone(values, mask)
    second = backbone(changed, mask)
    torch.testing.assert_close(first.last_hidden_state, second.last_hidden_state)
    assert not first.attention_mask.reshape(3, 8, 5)[:, :3].any()
    assert first.attention_mask.reshape(3, 8, 5)[:, 3:].all()
    assert first.last_hidden_state[~first.attention_mask].eq(0).all()
    cropped = backbone(values[:, :73])
    valid = first.last_hidden_state[first.attention_mask].reshape(3, 25, 32)
    torch.testing.assert_close(valid, cropped.last_hidden_state, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires cloud CUDA"))])
def test_lora_backward_freezes_base_and_reaches_trainable_encoder(device):
    backbone = tiny_backbone(device)
    modules = backbone.enable_lora(
        target_modules=["q", "k", "v", "o", "wi", "wo"], rank=4, alpha=8, dropout=0
    )
    assert modules and all(name.startswith("encoder.") for name in modules)
    backbone.train()
    values = inputs(device)
    output = backbone(values).last_hidden_state
    output.square().mean().backward()
    nonzero = []
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad:
            assert ".lora_a." in name or ".lora_b." in name
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            nonzero.append(bool(parameter.grad.abs().sum() > 0))
        else:
            assert parameter.grad is None
    assert any(nonzero)


def test_factory_and_both_signal_losses_use_chronos_encoder(monkeypatch):
    from pathlib import Path

    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.factory import build_model_bundle

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(Chronos2Backbone, "from_pretrained", lambda **kw: tiny_backbone())
    for task in ("alpha_point", "absolute_direction"):
        config = ExperimentConfig.from_yaml(root / f"configs/experiments/a_chronos2_{task}64.yaml")
        config.model.explicit_output_scale = False
        config.model.feature_mode = "baseline"
        bundle = build_model_bundle(config, torch.device("cpu"))
        values = inputs()
        targets = torch.zeros(3, len(config.data.alpha_horizons))
        output = bundle.model(
            asset_ohlcv=values, benchmark_ohlcv=values * 1.01,
            target_alpha=targets, target_absolute=targets,
            market_ids=torch.tensor([0, 1, 2]),
        )
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        assert bundle.lora_parameter_names
