"""Cloud-only numerical tests for the integrated market-adaptation release."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stock_forecasting.config import ExperimentConfig, IntervalCalibrationConfig
from stock_forecasting.forecast_evaluation import (
    ForecastEvaluationStore,
    calibrate_predictions,
    validate_calibration,
)
from stock_forecasting.interval_calibration import prequential_predictions, volatility_ratio
from stock_forecasting.models.forecast import MultiHorizonAlphaHead
from stock_forecasting.models.scale_features import fit_scale_feature_statistics

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner"
))]


def test_integrated_configs_preserve_controlled_capacity_comparison():
    configs = {f"{g}{r}": ExperimentConfig.from_yaml(
        ROOT / f"configs/experiments/{g}_adaptive{r}.yaml"
    ) for g in ("a", "b") for r in (64, 128)}
    for config in configs.values():
        assert config.model.unfreeze_last_blocks == 0
        assert config.model.market_residual_hidden_dim == 64
        assert config.model.ranking_numeric_features
        assert config.training.market_loss_weights == [1, 2, 2, 1]
        assert config.training.learning_rate_schedule == "sample_plateau"
        assert config.training.evaluation_max_samples is None
        assert config.validation.interval_calibration.mode == "regime_adaptive"
    for group in ("a", "b"):
        first, second = (configs[f"{group}{r}"].as_dict() for r in (64, 128))
        for payload in (first, second):
            payload.pop("experiment_name")
            payload.pop("description")
            payload["wandb"].pop("name")
            payload["wandb"].pop("tags")
            payload["model"]["lora"].pop("rank")
            payload["model"]["lora"].pop("alpha")
        assert first == second
    assert configs["a64"].model_architecture_digest() == configs["b64"].model_architecture_digest()


@pytest.mark.parametrize("device", DEVICES)
def test_market_residual_routing_ranking_features_and_checkpoint(device):
    torch.manual_seed(42)
    head = MultiHorizonAlphaHead(
        32, hidden_dim=32, horizons=range(1, 15), feature_mode="combined", market_aware=True,
        explicit_output_scale=True, decoupled_output_scale=True, independent_ranking_head=True,
        market_residual_hidden_dim=16, ranking_numeric_features=True,
    ).to(device)
    features = torch.rand(8, 20, device=device) + .1
    head.numeric_branch.set_statistics(fit_scale_feature_statistics(
        features.cpu().numpy(), {"split": "train"}
    ))
    tokens = torch.randn(8, 4, 32, device=device)
    markets = torch.arange(8, device=device) % 4
    args = dict(scale_features=features, benchmark_tokens=tokens, market_ids=markets,
                return_ranking=True)
    before, _ = head(tokens, **args)
    with torch.no_grad():
        # Only TWSE median residual changes, with no output-crossing risk.
        head.market_residual[-1].bias[4] = .25
    after, score = head(tokens, **args)
    changed = (markets == 1)[:, None].expand(-1, 14)
    torch.testing.assert_close(after[..., 1] - before[..., 1], changed.float() * .25)
    assert (after[..., 0] < after[..., 1]).all() and (after[..., 1] < after[..., 2]).all()
    with torch.no_grad():
        head.ranking_feature_head[-1].weight.fill_(.3)
    baseline, score = head(tokens, **args)
    modified = dict(args, scale_features=features * 2, benchmark_tokens=tokens * 2)
    _, different = head(tokens, **modified)
    assert not torch.equal(score, different)
    head.zero_grad(set_to_none=True)
    score.sum().backward()
    assert head.quantile_parameters.weight.grad is None
    assert head.market_residual[-1].weight.grad is None
    assert head.numeric_branch.scale_projection[0].weight.grad is not None
    assert head.numeric_branch.benchmark_projection.weight.grad is not None
    state = copy.deepcopy(head.state_dict())
    head.load_state_dict(state, strict=True)
    torch.testing.assert_close(head(tokens, **args)[0], baseline)
    unweighted = head.pinball_loss(after, torch.zeros(8, 14, device=device))
    weighted = head.pinball_loss(after, torch.zeros(8, 14, device=device),
                                 torch.full((8,), 2.0, device=device))
    torch.testing.assert_close(weighted, unweighted * 2)


def _store(path, *, test=False, h_start=1):
    rng = np.random.default_rng(42)
    days = np.busday_offset("2025-12-01" if test else "2025-06-02", np.arange(60))
    # Scramble symbol/date order to ensure online evaluation explicitly sorts dates.
    dates = np.repeat(days.astype(str), 18)
    markets = np.tile(np.repeat(["US", "TWSE"], 9), len(days))
    n = len(dates)
    order = rng.permutation(n)
    features = rng.uniform(.4, 1.6, (n, 20)).astype("float32")
    horizons = list(range(h_start, 15))
    q = np.tile(np.array([-.5, 0, .5], dtype="float32"), (n, len(horizons), 1))
    y = rng.normal(0, .7, (n, len(horizons))).astype("float32")
    store = ForecastEvaluationStore(
        path, n, horizons, [1.] * len(horizons),
        calibration_policy=IntervalCalibrationConfig(mode="regime_adaptive").model_dump(),
    )
    store.append(y[order], q[order], scale_features=features[order],
                 symbols=[f"S{i % 18}" for i in order], dates=dates[order], markets=markets[order],
                 asset_types=["stock"] * n, providers=["fixture"] * n)
    return store


def test_regime_calibration_raw_static_and_conditional_metrics(tmp_path):
    store = _store(tmp_path)
    try:
        calibration = store.fit_interval_calibration()
        validate_calibration(calibration, store.horizons)
        assert calibration["schema_version"] == 2
        assert calibration["static_reference"]["schema_version"] == 1
        before = store.predictions.copy()
        q = calibrate_predictions(before, store.metadata["market"], calibration, store.volatility)
        np.testing.assert_array_equal(q[..., 1], before[..., 1])
        np.testing.assert_array_equal(store.predictions, before)
        result = store.calibrated_metrics(calibration)
        assert result["samples"] == store.count
        assert result["market_macro"]["market_count"] == 2
        assert result["by_month"] and result["by_market_month"]["TWSE"]
        with pytest.raises(ValueError, match="past-only"):
            calibrate_predictions(before, store.metadata["market"], calibration)
        malformed = copy.deepcopy(calibration)
        malformed["regimes"]["US"]["factors"][0]["upper"][2] = -1
        with pytest.raises(ValueError):
            validate_calibration(malformed, store.horizons)
    finally:
        store.close()


def test_delayed_feedback_cannot_see_future_labels_and_keeps_q50(tmp_path):
    validation, test = _store(tmp_path / "val"), _store(tmp_path / "test", test=True)
    try:
        calibration = validation.fit_interval_calibration()
        frozen = copy.deepcopy(calibration)
        sessions = np.busday_offset("2025-12-01", np.arange(100))
        calendars = {market: sessions for market in ("US", "TWSE")}
        output = np.lib.format.open_memmap(tmp_path / "online.npy", mode="w+", dtype="float32",
                                          shape=test.predictions.shape)
        try:
            prequential_predictions(test, calibration, output, calendars=calendars)
            original = output.copy()
            # Only labels for signals issued on/after session 20 are changed.
            future = test.metadata["date"] >= str(sessions[20])
            test.targets[future] += 100
            evidence = prequential_predictions(test, calibration, output, calendars=calendars)
            for h, horizon in enumerate(test.horizons):
                protected = test.metadata["date"] <= str(sessions[20 + horizon])
                np.testing.assert_array_equal(output[protected, h], original[protected, h])
            np.testing.assert_array_equal(output[..., 1], original[..., 1])
            assert not np.array_equal(output[..., 2], original[..., 2])
            assert calibration == frozen
            assert evidence["by_market"]["US"]["feedback_batches"] > 0
            frozen_prediction = calibrate_predictions(test.predictions, test.metadata["market"],
                                                       calibration, test.volatility)
            multiplier = (output[..., 2] - output[..., 1]) / (
                frozen_prediction[..., 2] - frozen_prediction[..., 1]
            )
            assert multiplier.min() >= .5 - 1e-6 and multiplier.max() <= 2 + 1e-6
        finally:
            output._mmap.close()
        with pytest.raises(ValueError, match="strictly after"):
            validation.adaptive_metrics(calibration, calendars=calendars)
        assert test.adaptive_metrics(calibration, calendars=calendars)["samples"] == test.count
    finally:
        validation.close()
        test.close()


def test_volatility_ratio_is_past_only_and_rejects_missing_values():
    features = np.ones((4, 20), dtype="float32")
    np.testing.assert_array_equal(volatility_ratio(features), np.ones(4))
    features[0, 2] = np.nan
    with pytest.raises(ValueError):
        volatility_ratio(features)


def test_online_calendar_holidays_and_nondefault_horizon_are_causal(tmp_path):
    from stock_forecasting.data.sample_universe import market_sessions

    validation = _store(tmp_path / "val", h_start=3)
    test = _store(tmp_path / "test", test=True, h_start=3)
    try:
        calibration = validation.fit_interval_calibration()
        calibration["fit_label_end_exclusive"] = "2025-12-01"
        sessions = market_sessions("XNYS", "2025-12-19", "2026-06-01")
        sessions = sessions.tz_localize(None).to_numpy().astype("datetime64[D]")
        assert np.datetime64("2025-12-25") not in sessions
        assert np.datetime64("2026-01-01") not in sessions
        original_dates = test.metadata["date"].copy()
        for old, day in zip(np.unique(original_dates), sessions, strict=False):
            test.metadata["date"][original_dates == old] = str(day)
        test.metadata["market"] = "US"
        output = np.lib.format.open_memmap(
            tmp_path / "calendar-online.npy", mode="w+", dtype="float32",
            shape=test.predictions.shape,
        )
        try:
            # No injected calendar: exercise the pinned offline production path.
            prequential_predictions(test, calibration, output)
            original = output.copy()
            changed = test.metadata["date"] >= str(sessions[2])
            test.targets[changed] += 100
            prequential_predictions(test, calibration, output)
            for index, horizon in enumerate(test.horizons):
                protected = test.metadata["date"] <= str(sessions[2 + horizon])
                np.testing.assert_array_equal(output[protected, index], original[protected, index])
            assert not np.array_equal(output[..., 2], original[..., 2])
            calibration["fit_label_end_exclusive"] = "2026-01-01"
            with pytest.raises(ValueError, match="label boundary"):
                prequential_predictions(test, calibration, output)
        finally:
            output._mmap.close()
    finally:
        validation.close()
        test.close()


def test_single_observation_inference_uses_frozen_regime_not_online_state(tmp_path, monkeypatch):
    import stock_forecasting.cli.infer as inference
    from stock_forecasting.data.manifest import sha256_file

    config = ExperimentConfig.from_yaml(ROOT / "configs/experiments/a_adaptive64.yaml")
    config.validation.output_root = tmp_path / "evaluations"
    checkpoint = tmp_path / "run-infer" / "checkpoint-000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"fixture-weight-identity")
    store = _store(tmp_path / "validation")
    try:
        calibration = store.fit_interval_calibration()
    finally:
        store.close()
    calibration["checkpoint_weights_sha256"] = sha256_file(checkpoint / "adapter.safetensors")
    directory = config.validation.output_root / "run-infer"
    directory.mkdir(parents=True)
    (directory / "interval-calibration.json").write_text(json.dumps(calibration))
    output = SimpleNamespace(
        alpha_quantiles=torch.tensor([[[-.5, 0., .5]] * 14]),
        scale_features=torch.ones(1, 20), ranking_scores=torch.arange(14)[None].float(),
        **{key: torch.zeros(1, 2, 4) for key in (
            "asset_last_hidden_state", "benchmark_last_hidden_state", "asset_latent_tokens",
            "benchmark_latent_tokens", "conditioned_latent_tokens",
        )},
    )

    class Model(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            return output

    monkeypatch.setattr(inference, "run_preflight", lambda *_a, **_kw: SimpleNamespace(
        require_success=lambda: None
    ))
    monkeypatch.setattr(inference, "build_model_bundle", lambda *_a: SimpleNamespace(model=Model()))
    monkeypatch.setattr(inference, "resolve_checkpoint", lambda *_a, **_kw: checkpoint)
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_a, **_kw: {})
    monkeypatch.setattr(inference, "provenance_summary", lambda *_a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observation = SimpleNamespace(
        asset_series=torch.ones(1, 128, 5), benchmark_series=torch.ones(1, 128, 5),
        asset_attention_mask=torch.ones(1, 128, dtype=torch.bool),
        benchmark_attention_mask=torch.ones(1, 128, dtype=torch.bool),
        asset_timestamp_features=torch.zeros(1, 128, 5, dtype=torch.long),
        benchmark_timestamp_features=torch.zeros(1, 128, 5, dtype=torch.long),
        metadata={"market": "US"}, benchmark_metadata={}, symbol="AAPL.US",
        benchmark_symbol="VTI.US", asset_type="stock", window_start_at="2025-12-01",
        cutoff_at="2026-06-10",
    )
    payload = inference.infer_observation(config, checkpoint, observation)
    expected = calibrate_predictions(output.alpha_quantiles.numpy(), ["US"], calibration,
                                      volatility_ratio(output.scale_features.numpy()))
    assert payload["interval_calibration_status"] == "validation_fitted_frozen_regime"
    for h in range(1, 15):
        actual = payload["calibrated_forecast"]["by_horizon"][f"{h}d"]["alpha_quantiles"]
        assert list(actual.values()) == pytest.approx(expected[0, h - 1])
        assert actual["q50"] == payload["forecast"]["by_horizon"][f"{h}d"]["alpha_quantiles"]["q50"]
    assert payload["ranking_scores"] == {f"{h}d": float(h - 1) for h in range(1, 15)}
