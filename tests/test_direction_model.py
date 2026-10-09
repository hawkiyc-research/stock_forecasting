"""Cloud numerical acceptance for the opt-in dual-direction architecture."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from stock_forecasting.config import DirectionConfig, ExperimentConfig
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.direction import DualDirectionHead, direction_labels
from stock_forecasting.models.technical_channels import technical_channels

ROOT = Path(__file__).resolve().parents[1]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA"),
    ),
]


def mock_config():
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.model.direction = DirectionConfig(enabled=True, loss_reference_samples=32)
    config.data.input_length = 128
    config.data.h_start = 1
    return config


def bars(batch=4, device="cpu"):
    close = torch.exp(torch.randn(batch, 128, 1, device=device) * 0.01 + 4)
    return torch.cat([close, close * 1.01, close * 0.99, close, torch.ones_like(close) * 1e6], -1)


def test_new_presets_exactly_retain_lora64_base_and_old_presets():
    for group in ("a", "b"):
        old = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{group}_lora64.yaml")
        new = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{group}_direction64.yaml")
        assert new.model.direction.enabled and not old.model.direction.enabled
        for payload in (old, new):
            assert payload.model.lora.rank == 64 and payload.model.unfreeze_last_blocks == 0
            assert not payload.model.market_residual_hidden_dim
            assert not payload.model.ranking_numeric_features
            assert payload.training.market_loss_weights == [1] * 4
        before, after = old.as_dict(), new.as_dict()
        for data in (before, after):
            data.pop("experiment_name")
            data.pop("description")
            data["model"].pop("direction")
            data["training"].pop("checkpoint_monitor")
        assert before == after
        assert new.model.direction.costs.us_shares == 50
        assert new.model.direction.costs.taiwan_shares == 1000


@pytest.mark.parametrize("device", DEVICES)
def test_technical_channels_causal_scale_invariant_and_warmup(device):
    x = bars(device=device)
    original = technical_channels(x)
    assert original.shape == (4, 128, 5) and torch.isfinite(original).all()
    changed = x.clone()
    changed[:, 100:] *= 3
    torch.testing.assert_close(technical_channels(changed)[:, :100], original[:, :100])
    changed = x.clone()
    changed[..., :4] *= 7
    changed[..., 4] *= 3
    torch.testing.assert_close(technical_channels(changed), original, rtol=1e-4, atol=1e-4)
    assert (original[:, :19] == 0).all() and (original[:, :59, 2] == 0).all()
    x[..., 4] = 0
    assert (technical_channels(x)[..., 4] == 0).all()
    assert torch.isfinite(technical_channels(x[:, :1])).all()


def test_technical_definitions_on_deterministic_prices():
    close = torch.arange(1, 129, dtype=torch.float32)[None, :, None]
    x = torch.cat([close, close + 1, close - 0.5, close, close * 0 + 100], -1)
    values = technical_channels(x)[0, -1]
    expected = torch.tensor(
        [
            np.log(128 / 118.5),
            np.log(126 / 118.5),
            np.log(118.5 / 98.5),
            (256 - 128 - 107.5) / (128 - 107.5),
            1.0,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(values, expected)


def test_direction_threshold_edges_and_market_falling_example():
    band = torch.tensor([[-0.005, 0.005]])
    target = torch.tensor([[-0.006, -0.005, 0, 0.005, 0.006]])
    assert direction_labels(target, band).tolist() == [[0, 1, 1, 1, 2]]
    absolute = torch.tensor([[np.log(0.96)]], dtype=torch.float32)
    alpha = absolute - np.log(0.92)
    assert direction_labels(absolute, band).item() == 0
    assert direction_labels(alpha, band).item() == 2


@pytest.mark.parametrize("device", DEVICES)
def test_direction_heads_gradients_regularization_and_inference(device):
    config = mock_config()
    bundle = build_model_bundle(config, torch.device(device))
    bundle.model.train()
    x = bars(device=device)
    kwargs = dict(
        market_ids=torch.zeros(4, dtype=torch.long, device=device),
        direction_bands=torch.tensor([[-0.005, 0.005]], device=device).expand(4, -1),
    )
    target = torch.randn(4, 14, device=device) * 0.03
    output = bundle.model(x, x * 1.01, target_alpha=target, target_absolute=target - 0.02, **kwargs)
    assert output.absolute_direction_logits.shape == (4, 14, 3)
    assert output.alpha_direction_logits.shape == (4, 14, 3)
    assert output.representation_loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    for name in (
        "direction_head.absolute.weight",
        "direction_head.alpha.weight",
        "direction_head.query.weight",
        "backbone.technical_adapter.projection.weight",
    ):
        grad = dict(bundle.model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    bundle.model.eval()
    with torch.inference_mode():
        inference = bundle.model(x, x * 1.01, **kwargs)
    assert inference.loss is None and inference.representation_loss is None
    assert inference.alpha_quantiles.shape == (4, 14, 3)
    with pytest.raises(ValueError, match="both absolute"):
        bundle.model(x, x, target_alpha=target, **kwargs)


def test_loss_reference_state_is_strict_and_not_a_trainable_parameter():
    head = DualDirectionHead(64, 14, DirectionConfig(enabled=True), 0.05)
    state = {"schema_version": 1, "samples": 32, "references": [0.3, 1.1, 1.2]}
    head.restore_references(state)
    torch.testing.assert_close(head.weights(), torch.tensor([0.3 / 1.1, 0.3 / 1.2]))
    assert "loss_references" not in dict(head.named_parameters())
    with pytest.raises(ValueError):
        head.restore_references({**state, "references": [0, 1, 1]})
    assert head.reference_state()["samples"] == 32


def test_checkpoint_selection_retains_old_metric_and_accepts_only_exact_direction_metric():
    from stock_forecasting.checkpointing import (
        _validate_checkpoint_policy,
        _validate_validation_selection,
    )

    for metric in ("primary_5d/selection_score", "direction/selection_loss"):
        _validate_checkpoint_policy(monitor=metric, mode="min", save_top_k=5)
        selection = {"source": "validation", "metric": metric, "value": 0.5, "mode": "min"}
        assert _validate_validation_selection(selection, "fixture") == selection
    for metric in ("direction/training_loss", "test/selection_loss", "training/loss"):
        with pytest.raises(ValueError):
            _validate_checkpoint_policy(monitor=metric, mode="min", save_top_k=5)
        with pytest.raises(ValueError):
            _validate_validation_selection(
                {"source": "validation", "metric": metric, "value": 0.5, "mode": "min"},
                "fixture",
            )


def test_new_adapter_initialization_preserves_base_weights():
    config = mock_config()
    config.model.direction.enabled = False
    torch.manual_seed(42)
    old = build_model_bundle(config, torch.device("cpu"))
    config.model.direction.enabled = True
    torch.manual_seed(42)
    new = build_model_bundle(config, torch.device("cpu"))
    before, after = old.model.state_dict(), new.model.state_dict()
    for name, value in before.items():
        torch.testing.assert_close(after[name], value, rtol=0, atol=0, msg=name)


def test_lazy_direction_targets_reuse_membership_and_adjusted_execution(
    market_frame, tmp_path, monkeypatch,
):
    from collections import Counter

    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset
    from stock_forecasting.data.manifest import artifact_metadata
    from stock_forecasting.direction_data import DirectionDataset

    raw = tmp_path / "market.parquet"
    market_frame.to_parquet(raw, index=False)
    root = tmp_path / "bars"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=root,
        window_size=32,
        bucket_count=2,
        batch_rows=512,
        purge_bars=20,
        embargo_bars=14,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(market_frame))}
        },
    )
    for split in ("train", "validation", "test"):
        source = LazyFinancialWindowDataset(
            root, split=split, window_size=32, h_start=1, symbol_cache_size=2,
        )
        wrapper = pickle.loads(pickle.dumps(DirectionDataset(source)))
        assert len(wrapper) == len(source)
        indices = [len(source) - 1, 0, len(source) // 2, 0]
        misses = Counter()
        original_numeric = wrapper.source._numeric_symbol

        def count_misses(symbol, owner=wrapper.source, original=original_numeric, counts=misses):
            if symbol not in owner._numeric_cache:
                counts[symbol] += 1
            return original(symbol)

        monkeypatch.setattr(wrapper.source, "_numeric_symbol", count_misses)
        actual = wrapper.__getitems__(indices)
        assert all(misses[symbol] == 1 for symbol in {item["symbol"] for item in actual})
        assert wrapper.__getitems__([]) == []
        with pytest.raises(IndexError):
            wrapper[len(source)]
        original = source.__getitems__(indices)
        for index, item, previous in zip(indices, actual, original, strict=True):
            assert item["sample_id"] == previous["sample_id"]
            torch.testing.assert_close(
                item["target_alpha"], previous["target_alpha"], rtol=0, atol=0
            )
            record = source.record_at(index)
            expected = np.array(
                [record["label"]["asset_total_returns"][f"{h}d"] for h in source.horizons]
            )
            np.testing.assert_allclose(
                item["target_absolute"].numpy(), np.log1p(expected), rtol=1e-5, atol=1e-7
            )


def test_streaming_metrics_do_not_hide_neutral_collapse():
    from types import SimpleNamespace

    from stock_forecasting.direction_runtime import DirectionMetrics

    config = mock_config()
    head = DualDirectionHead(64, 14, config.model.direction, 0)
    head.restore_references({"schema_version": 1, "samples": 32, "references": [0.3, 1.1, 1.1]})
    targets = torch.zeros(4, 14)
    targets[0] = 0.1
    targets[1] = -0.1
    logits = torch.zeros(4, 14, 3)
    logits[..., 1] = 5
    output = SimpleNamespace(
        absolute_direction_logits=logits,
        alpha_direction_logits=logits,
        pinball_loss=torch.tensor(0.3),
    )
    batch = {
        "target_absolute": targets,
        "target_alpha": targets,
        "direction_bands": torch.tensor([[-0.005, 0.005]]).expand(4, -1),
        "markets": ["US", "TWSE", "US", "TWSE"],
        "cutoff_at": ["2025-12-01"] * 4,
    }
    metrics = DirectionMetrics(range(1, 15))
    metrics.append(output, batch)
    result = metrics.finish(head, config.model.direction.costs)
    summary = result["overall"]["absolute"]["aggregate"]
    assert summary["accuracy"] == summary["majority_accuracy"] == 0.5
    assert summary["balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["market"]["US"]["absolute"]["aggregate"]["balanced_accuracy"] == 0.5
    assert result["market"]["TWSE"]["absolute"]["aggregate"]["balanced_accuracy"] == 0.5
    assert summary["classes"]["up"]["predicted"] == 0
    assert result["selection_loss"] > 0.3
    json.dumps(result, allow_nan=False)


def test_inference_exports_both_direction_probabilities_without_changing_alpha(
    market_frame, tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    from stock_forecasting.cli import infer as inference

    config = mock_config()
    config.validation.output_root = tmp_path / "evaluations"
    checkpoint = tmp_path / "run-direction" / "checkpoint-000001"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        inference, "run_preflight",
        lambda *_args, **_kwargs: SimpleNamespace(require_success=lambda: None),
    )
    monkeypatch.setattr(inference, "resolve_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(inference, "provenance_summary", lambda *_args: {})
    observation = inference.prepare_causal_observation(
        market_frame, input_length=128, symbol="AAPL.US",
    )
    payload = inference.infer_observation(config, checkpoint, observation)
    direction = payload["direction"]
    assert direction["class_order"] == ["down", "neutral", "up"]
    assert direction["costs"]["shares"] == 50
    assert direction["costs"]["reference_price"] == float(observation.asset_series[0, -1, 3])
    for task in ("absolute", "alpha"):
        probabilities = direction[f"{task}_probabilities"]
        assert list(probabilities) == [str(h) for h in range(1, 15)]
        for row in probabilities.values():
            assert len(row) == 3 and sum(row) == pytest.approx(1, abs=1e-6)
    assert len(payload["forecast"]["by_horizon"]) == 14
    assert payload["model_output_schema_version"] == "5.0"
    json.dumps(payload, allow_nan=False)
