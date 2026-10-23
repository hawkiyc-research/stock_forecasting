"""Cross-stock transformer: model switches, loss, cache table and a tiny walk-forward run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from stock_forecasting.cross_stock_training import (
    EXPERIMENTS,
    build_group_index,
    decide_h_a,
    make_config,
    paired_comparison,
    plan_fold,
    run_cross_stock,
    table_from_caches,
    write_xs_manifest,
)
from stock_forecasting.models.cross_stock import (
    CrossStockConfig,
    CrossStockModel,
    cross_sectional_loss,
    cross_sectional_zscore,
    normalize_windows,
    parameter_count,
)
from stock_forecasting.walk_forward import quarterly_folds
from stock_forecasting.walk_forward_lgbm import MODEL_NAME as BASELINE

SCHEDULE = {
    "schema": "walk-forward-schedule-v1",
    "name": "fixture",
    "calendar": "XTAI",
    "train_start": "2023-01-02",
    "data_end": "2024-07-01",
    "first_quarter": "2024Q1",
    "last_quarter": "2024Q2",
    "embargo_sessions": 15,
    "validation_months": 3,
    "periods": {"P1": ["2024-01-01", "2024-04-01"], "P2": ["2024-04-01", "2024-07-01"]},
}
HORIZONS = [1, 5, 10]
WINDOW, FEATURES = 32, 12


def _sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", "2024-06-28")


def _cache(split: str, sessions: pd.DatetimeIndex, seed: int, symbols_per_market: int = 24):
    """A fake XsCache-like object whose first feature and the window trend carry signal."""
    rng = np.random.default_rng(seed)
    markets, suffixes = ["TPEX", "TWSE"], {"TPEX": "TWO", "TWSE": "TW"}
    count = len(sessions) * len(markets) * symbols_per_market
    windows = np.empty((count, WINDOW, 5), dtype=np.float32)
    features = np.empty((count, FEATURES), dtype=np.float32)
    ranks = np.empty((count, len(HORIZONS)), dtype=np.float32)
    alpha = np.empty((count, len(HORIZONS)), dtype=np.float32)
    cutoff = np.empty(count, dtype=np.int64)
    market_codes = np.empty(count, dtype=np.int8)
    symbol_codes = np.empty(count, dtype=np.int32)
    symbol_names = [f"{1000 + i}.{suffixes[m]}" for m in markets for i in range(symbols_per_market)]
    row = 0
    for day in sessions:
        for code, _market in enumerate(markets):
            signal = rng.standard_normal(symbols_per_market)
            noise = rng.standard_normal((symbols_per_market, len(HORIZONS)))
            a = 0.03 * signal[:, None] + 0.01 * noise
            r = pd.DataFrame(a).rank().to_numpy() / symbols_per_market
            for i in range(symbols_per_market):
                drift = 0.002 * signal[i]
                closes = 100.0 * np.exp(
                    np.cumsum(np.full(WINDOW, drift) + 0.01 * rng.standard_normal(WINDOW))
                )
                windows[row, :, 3] = closes
                windows[row, :, 0] = closes * (1 + 0.001 * rng.standard_normal(WINDOW))
                windows[row, :, 1] = closes * 1.01
                windows[row, :, 2] = closes * 0.99
                windows[row, :, 4] = rng.uniform(1e5, 1e6, WINDOW)
                features[row] = rng.standard_normal(FEATURES)
                features[row, 0] = signal[i]
                ranks[row] = r[i]
                alpha[row] = a[i]
                cutoff[row] = day.value
                market_codes[row] = code
                symbol_codes[row] = code * symbols_per_market + i
                row += 1
    return SimpleNamespace(
        split=split,
        windows=windows,
        features=features,
        ranks=ranks,
        alpha=alpha,
        cutoff_ns=cutoff,
        market_codes=market_codes,
        symbol_codes=symbol_codes,
        markets=markets,
        symbols=symbol_names,
        horizons=HORIZONS,
        feature_columns=[f"f{i}" for i in range(FEATURES)],
    )


def test_model_shapes_switches_and_parameter_budget() -> None:
    torch.manual_seed(0)
    windows = torch.rand(50, 128, 5) * 100 + 1
    features = torch.randn(50, 183)
    full = CrossStockModel(CrossStockConfig())
    assert full(windows, features).shape == (50, 14)
    assert 500_000 < parameter_count(full) < 3_000_000
    for name, overrides in EXPERIMENTS.items():
        if "encoder" in overrides:
            continue  # D1 experiments need cached embeddings; covered by test_encoders
        overrides = {k: v for k, v in overrides.items() if k != "lambda_corr"}
        config = CrossStockConfig(**overrides)
        model = CrossStockModel(config)
        assert model(windows, None if not config.feature_input else features).shape == (50, 14), (
            name
        )
    cached = CrossStockModel(CrossStockConfig(encoder_dim=64))
    assert cached(None, features, encoded=torch.randn(50, 64)).shape == (50, 14)
    normalized = normalize_windows(windows)
    assert torch.isfinite(normalized).all() and normalized.abs().max() <= 5
    assert torch.allclose(normalized[:, -1, 3], torch.zeros(50), atol=1e-6)
    z = cross_sectional_zscore(torch.tensor([[1.0, float("nan")], [3.0, 2.0], [5.0, 4.0]]))
    assert torch.allclose(z[:, 0].mean(), torch.zeros(()), atol=1e-6) and z[0, 1] == 0


def test_loss_rewards_cross_sectional_correlation_and_masks_nan() -> None:
    target = torch.tensor([[0.1, 0.5], [0.4, float("nan")], [0.9, 0.2], [0.3, 0.8]])
    aligned = target.clone().nan_to_num(0.0)
    loss_aligned, parts = cross_sectional_loss(aligned, target, 1.0)
    loss_reversed, _ = cross_sectional_loss(-aligned, target, 1.0)
    assert loss_aligned < loss_reversed
    assert parts["corr"] > 0.9
    loss_mse_only, _ = cross_sectional_loss(aligned, target, 0.0)
    assert loss_mse_only <= loss_aligned
    with pytest.raises(ValueError):
        cross_sectional_loss(torch.zeros(2, 1), torch.full((2, 1), float("nan")), 1.0)


def test_table_group_index_and_fold_plan() -> None:
    sessions = _sessions()
    table = table_from_caches({"train": _cache("train", sessions, 1, 8)})
    assert table.windows.dtype == torch.float32 and len(table) == len(sessions) * 16
    index = build_group_index(table)
    assert len(index.sizes) == len(sessions) * 2 and (index.sizes == 8).all()
    folds = quarterly_folds(sessions, SCHEDULE)
    plan = plan_fold(table, index, folds[0], min_group_size=5)
    assert len(plan.predict_groups) > 0 and len(plan.train_groups) > len(plan.valid_groups)
    assert plan.valid_alpha_ranks.shape == (len(plan.valid_rows), len(HORIZONS))
    config, lambda_corr = make_config("t-xs-a2-mse-only", table)
    assert (config.feature_dim, config.horizons, lambda_corr) == (FEATURES, len(HORIZONS), 0.0)


def test_walk_forward_run_writes_b0_compatible_outputs_and_pairs(tmp_path: Path) -> None:
    sessions = _sessions()
    table = table_from_caches({"train": _cache("train", sessions, 2)})
    folds = quarterly_folds(sessions, SCHEDULE)
    output = tmp_path / "xs"
    # A fake baseline daily series on the same dates lets the paired comparison run.
    dates = pd.DatetimeIndex(sorted(set(table.dates)))
    predicted = dates[(dates >= pd.Timestamp("2024-01-01")) & (dates < pd.Timestamp("2024-07-01"))]
    baseline = pd.DataFrame(
        [
            {"model": BASELINE, "market": market, "h": h, "date": day, "ic": 0.0, "n": 24}
            for day in predicted
            for market in ("TPEX", "TWSE")
            for h in HORIZONS
        ]
    )
    results = run_cross_stock(
        table,
        experiment="t-xs-scratch",
        folds=folds,
        schedule=SCHEDULE,
        output_root=output,
        seeds=(1, 2),
        min_group_size=5,
        device="cpu",
        params={"max_epochs": 3, "patience": 2},
        baseline_daily=baseline,
        max_groups_per_epoch=40,
    )
    predictions = pd.read_parquet(output / "predictions.parquet")
    targets = pd.read_parquet(output / "targets.parquet")
    assert set(predictions["model"]) == {"t-xs-scratch", "t-xs-scratch-seed1", "t-xs-scratch-seed2"}
    assert len(predictions) == 3 * len(targets)
    assert results["fits"]["count"] == len(folds) * 2
    assert results["device"]["device"] == "cpu"
    stored = json.loads((output / "results.json").read_text())
    assert stored["decision"]["hypothesis"] == "H-A"
    pairs = results["paired_vs_baseline"]
    assert {p["cell"] for p in pairs} == {"short", "holding"}
    assert any(p["period"] == "P1+P2" for p in pairs)
    history = json.loads((output / "training_history.json").read_text())
    assert len(history) == len(folds) * 2 and len(history[0]["history"]) <= 3


def test_paired_comparison_and_decision_logic() -> None:
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    rows = []
    for day in dates:
        for market in ("TPEX", "TWSE"):
            for h in (1, 5, 10):
                rows.append(
                    {"model": "m", "market": market, "h": h, "date": day, "ic": 0.05, "n": 30}
                )
                rows.append(
                    {"model": "b", "market": market, "h": h, "date": day, "ic": 0.01, "n": 30}
                )
    daily = pd.DataFrame(rows)
    pairs = paired_comparison(
        daily[daily["model"] == "m"],
        daily[daily["model"] == "b"],
        SCHEDULE,
        model="m",
        baseline="b",
    )
    short = next(p for p in pairs if p["cell"] == "short" and p["period"] == "P1+P2")
    assert short["difference"] == pytest.approx(0.04) and short["lower_bound_positive"]
    decision = decide_h_a(pairs)
    assert decision["passed_both_cells"] and decision["passed_any_cell"]
    assert not decide_h_a([])["passed_any_cell"]


def test_finished_fold_seeds_are_written_and_resumed(tmp_path: Path) -> None:
    sessions = _sessions()
    table = table_from_caches({"train": _cache("train", sessions, 5, 12)})
    folds = quarterly_folds(sessions, SCHEDULE)
    shared = dict(
        experiment="t-xs-a1-no-cross",
        folds=folds,
        schedule=SCHEDULE,
        seeds=(1,),
        min_group_size=5,
        device="cpu",
        max_groups_per_epoch=20,
    )
    first = tmp_path / "first"
    run_cross_stock(table, output_root=first, params={"max_epochs": 2, "patience": 2}, **shared)
    assert sorted(p.name for p in (first / "partial").glob("*.npz")) == [
        "fold-00-seed-1.npz",
        "fold-01-seed-1.npz",
    ]
    second = tmp_path / "second"
    results = run_cross_stock(
        table,
        output_root=second,
        params={"max_epochs": 1, "patience": 1},
        resume_root=first,
        **shared,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(second / "predictions.parquet"),
        pd.read_parquet(first / "predictions.parquet"),
    )
    assert results["fits"]["count"] == len(folds)
    # Partials of another experiment are ignored.
    other = run_cross_stock(
        table,
        output_root=tmp_path / "other",
        params={"max_epochs": 1, "patience": 1},
        resume_root=first,
        **{**shared, "experiment": "t-xs-a4-no-gating"},
    )
    assert other["fits"]["mean_epochs"] == 1.0


def _variant(cache, seed: int):
    """Fake label-variant arrays: the residual target is the alpha with a group shift removed."""
    rng = np.random.default_rng(seed)
    residual = cache.alpha - 0.5 * cache.alpha.mean(axis=1, keepdims=True)
    residual = residual + 0.002 * rng.standard_normal(residual.shape).astype(np.float32)
    frame = pd.DataFrame(residual)
    frame["group"] = cache.cutoff_ns.astype(np.int64) * 4 + cache.market_codes.astype(np.int64)
    ranks = frame.groupby("group").rank().to_numpy(dtype=np.float32)
    sizes = frame.groupby("group")[0].transform("size").to_numpy()[:, None]
    return SimpleNamespace(
        split=cache.split,
        variant="residual_test",
        ranks=(ranks - 0.5) / sizes,
        residual=residual.astype(np.float32),
    )


def test_label_variant_scores_the_residual_target_and_reports_seed_stability(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    cache = _cache("train", sessions, 7)
    with pytest.raises(ValueError, match="exactly the cached splits"):
        table_from_caches({"train": cache}, label_variants={})
    table = table_from_caches({"train": cache}, label_variants={"train": _variant(cache, 1)})
    assert table.residual is not None and table.residual.shape == (len(table), len(HORIZONS))
    assert not torch.equal(table.ranks, torch.from_numpy(cache.ranks))
    folds = quarterly_folds(sessions, SCHEDULE)
    with pytest.raises(ValueError, match="label variant"):
        run_cross_stock(
            table,
            experiment="t-xs-scratch",
            folds=folds,
            schedule=SCHEDULE,
            output_root=tmp_path / "unnamed",
            seeds=(1,),
            min_group_size=5,
            device="cpu",
        )
    dates = pd.DatetimeIndex(sorted(set(table.dates)))
    predicted = dates[(dates >= pd.Timestamp("2024-01-01")) & (dates < pd.Timestamp("2024-07-01"))]
    baseline_name = f"{BASELINE}-residual_test"
    baseline = pd.DataFrame(
        [
            {"model": baseline_name, "market": market, "h": h, "date": day, "ic": 0.01, "n": 24}
            for day in predicted
            for market in ("TPEX", "TWSE")
            for h in HORIZONS
        ]
    )
    output = tmp_path / "variant"
    results = run_cross_stock(
        table,
        experiment="t-xs-scratch",
        folds=folds,
        schedule=SCHEDULE,
        output_root=output,
        seeds=(1, 2),
        min_group_size=5,
        device="cpu",
        params={"max_epochs": 2, "patience": 2},
        baseline_daily=baseline,
        baseline_daily_residual=baseline.assign(ic=0.02),
        baseline_model=baseline_name,
        label_variant="residual_test",
        max_groups_per_epoch=40,
    )
    assert results["label_variant"] == "residual_test"
    assert results["baseline_model"] == baseline_name
    targets = pd.read_parquet(output / "targets.parquet")
    assert {f"residual_h{h:02d}" for h in HORIZONS} <= set(targets.columns)
    residual_daily = pd.read_parquet(output / "daily_rank_ic_residual.parquet")
    assert set(residual_daily["model"]) == {
        "t-xs-scratch",
        "t-xs-scratch-seed1",
        "t-xs-scratch-seed2",
    }
    # HORIZONS has no h >= 11, so the long cell is empty.
    assert {c["cell"] for c in results["cells_residual"]} == {"short", "holding"}
    assert all(p["baseline"] == baseline_name for p in results["paired_vs_baseline"])
    residual_pairs = results["paired_vs_baseline_residual"]
    assert {p["cell"] for p in residual_pairs} == {"short", "holding"}
    stability = {(s["cell"], s["period"]): s for s in results["seed_stability"]}
    row = stability[("short", "P1+P2")]
    assert row["seeds"] == 2 and len(row["seed_rank_ic"]) == 2
    assert row["seed_range"] == pytest.approx(max(row["seed_rank_ic"]) - min(row["seed_rank_ic"]))
    assert row["ensemble_gain"] == pytest.approx(row["ensemble_rank_ic"] - row["seed_mean"])
    assert isinstance(row["single_seed_adequate"], bool)
    manifest = write_xs_manifest(output, results=results, label_variant="residual_test")
    assert "daily_rank_ic_residual.parquet" in manifest["files"]
    # Partials of the standard labels are not reused by a variant run and vice versa.
    standard = run_cross_stock(
        table_from_caches({"train": cache}),
        experiment="t-xs-scratch",
        folds=folds,
        schedule=SCHEDULE,
        output_root=tmp_path / "standard",
        seeds=(1, 2),
        min_group_size=5,
        device="cpu",
        params={"max_epochs": 1, "patience": 1},
        resume_root=output,
        max_groups_per_epoch=20,
    )
    assert standard["label_variant"] is None
    assert standard["cells_residual"] == [] and standard["fits"]["mean_epochs"] == 1.0
    assert not (tmp_path / "standard" / "daily_rank_ic_residual.parquet").exists()
