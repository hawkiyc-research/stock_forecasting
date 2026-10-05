"""Cloud numerical tests for the controlled adaptation experiments."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.date_market_sampler import INDEX_DTYPE, DateMarketSampler
from stock_forecasting.forecast_evaluation import ForecastEvaluationStore, calibrate_predictions
from stock_forecasting.models.backbones import KronosBackbone
from stock_forecasting.models.forecast import MultiHorizonAlphaHead
from stock_forecasting.models.ranking import same_date_ranking_loss

ROOT = Path(__file__).resolve().parents[1]


def test_six_experiments_are_valid_and_controlled():
    configs = [
        ExperimentConfig.from_yaml(path)
        for path in sorted((ROOT / "configs/experiments").glob("[ab]_*.yaml"))
    ]
    assert len(configs) == 6
    assert len({c.model_architecture_digest() for c in configs}) == 3
    for config in configs:
        assert config.training.stage == "stage2"
        assert config.data.train_fraction == 1 and config.data.max_samples is None
        assert config.training.evaluation_max_samples is None
        assert config.training.warmup_samples == 256000
        assert config.model.decoupled_output_scale and config.model.independent_ranking_head


def test_median_independent_of_scale_gate_and_ranking_parameters():
    head = MultiHorizonAlphaHead(
        64,
        horizons=range(1, 15),
        robust_scales=[0.1] * 14,
        feature_mode="combined",
        explicit_output_scale=True,
        decoupled_output_scale=True,
        independent_ranking_head=True,
        market_aware=True,
    )
    head.eval()
    tokens = torch.randn(4, 8, 64)
    features = torch.ones(4, 24)
    # Obtain the feature dimension from the branch rather than duplicating its contract.
    from stock_forecasting.models.scale_features import historical_scale_features

    bars = torch.ones(4, 128, 5)
    bars[..., :4] = torch.exp(torch.randn(4, 128, 1) * 0.01 + 4)
    features = historical_scale_features(bars, bars * 1.001, extended=True)
    from stock_forecasting.models.scale_features import fit_scale_feature_statistics

    head.numeric_branch.set_statistics(
        fit_scale_feature_statistics(
            features.numpy(),
            {"split": "train"},
        )
    )
    kwargs = dict(
        scale_features=features,
        benchmark_tokens=tokens,
        market_ids=torch.zeros(4, dtype=torch.long),
        return_ranking=True,
    )
    before, score = head(tokens, **kwargs)
    with torch.no_grad():
        head.scale_gate.bias.fill_(2)
        head.ranking_head.weight.add_(1)
    after, score = head(tokens, **kwargs)
    torch.testing.assert_close(before[..., 1], after[..., 1])
    assert (after[..., 0] < after[..., 1]).all() and (after[..., 1] < after[..., 2]).all()
    groups = torch.zeros(4, dtype=torch.long)
    loss = same_date_ranking_loss(
        score, torch.randn(4, 14), head.robust_scales, groups, torch.arange(4), 16
    )
    loss.backward()
    assert head.quantile_parameters.weight.grad is None
    assert head.ranking_head.weight.grad is not None


def test_annual_sampler_exact_quota_and_reproducible_resume(tmp_path):
    path = tmp_path / "index.npy"
    index = np.zeros(100, dtype=INDEX_DTYPE)
    index["ordinal"] = np.arange(100)
    index["group"][:60] = int(np.datetime64("2020-01-01", "D").astype(int)) * 256
    index["group"][60:] = int(np.datetime64("2022-01-01", "D").astype(int)) * 256
    np.save(path, index)
    with patch("stock_forecasting.date_market_sampler.date_market_index", return_value=path):
        sampler = DateMarketSampler(range(100), requested_workers=2, yearly_decay=0.8, block_size=8)
    first = list(sampler)
    # Per-window weights are 0.8**2 and 1, so strata have masses 38.4 and 40.
    assert len(first) == 100 and sum(i >= 60 for i in first) == 51
    assert first == list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second and sum(i >= 60 for i in second) == 51
    # Reconstructing and skipping exactly the saved position is deterministic.
    assert second[23:] == list(sampler)[23:]


def test_validation_calibration_preserves_location_and_raw_predictions(tmp_path):
    n, h = 256, 14
    store = ForecastEvaluationStore(tmp_path, n, list(range(1, h + 1)), [1.0] * h)
    y = np.linspace(-2, 2, n, dtype=np.float32)[:, None] * np.ones((1, h), dtype=np.float32)
    q = np.tile(np.array([-0.5, 0, 0.5], dtype=np.float32), (n, h, 1))
    store.append(
        y,
        q,
        symbols=[f"S{i}" for i in range(n)],
        dates=["2025-06-02"] * n,
        markets=["US"] * n,
        asset_types=["stock"] * n,
        providers=["fixture"] * n,
    )
    try:
        fitted = store.fit_interval_calibration()
        calibrated = calibrate_predictions(q, ["US"] * n, fitted)
        np.testing.assert_array_equal(calibrated[..., 1], q[..., 1])
        np.testing.assert_array_equal(store.predictions, q)
        assert abs(np.mean((y >= calibrated[..., 0]) & (y <= calibrated[..., 2])) - 0.8) < 0.02
        assert store.calibrated_metrics(fitted)["samples"] == n
    finally:
        store.close()


def test_partial_unfreezing_has_no_redundant_adapters():
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleList(
                [nn.ModuleDict({"q_proj": nn.Linear(8, 8)}) for _ in range(4)]
            )
            self.norm = nn.LayerNorm(8)

        def decode_s1(self, *args):
            raise NotImplementedError

    class Tokenizer(nn.Linear):
        def encode(self, *args):
            raise NotImplementedError

    backbone = KronosBackbone(Predictor(), Tokenizer(8, 8), hidden_size=8)
    names = backbone.enable_lora(
        target_modules=["q_proj"], rank=2, alpha=4, dropout=0, unfreeze_last_blocks=2
    )
    assert names == ("transformer.0.q_proj", "transformer.1.q_proj")
    assert all(p.requires_grad for p in backbone.model.transformer[-1].parameters())
    assert all(not p.requires_grad for p in backbone.tokenizer.parameters())
    assert all(
        not p.requires_grad for p in backbone.model.transformer[0]["q_proj"].base_layer.parameters()
    )
    assert backbone.unfrozen_parameter_names
