"""Reversal-exposure diagnostic: orthogonalization, quantile labels, cohort spreads, end to end."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_tw_strategy import _fixture
from twbacktest import reversal_exposure as rx
from twbacktest.exposure_diagnostics import quantile_long_short
from twbacktest.exposure_features import orthogonalize
from twbacktest.walkforward_data import load_prices


def test_orthogonalize_removes_the_regressors():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    score = 0.8 * x[:, 0] - 0.5 * x[:, 2] + rng.normal(scale=0.1, size=500)
    residual = orthogonalize(score, x)
    for k in range(3):
        assert abs(np.corrcoef(residual, x[:, k])[0, 1]) < 1e-6
    assert residual.mean() == pytest.approx(0.0, abs=1e-9)
    score[3] = np.nan
    x[7, 1] = np.nan
    residual = orthogonalize(score, x)
    assert np.isnan(residual[3]) and np.isnan(residual[7]) and np.isfinite(residual).sum() == 498
    assert np.isnan(orthogonalize(score[:3], x[:3])).all()  # too few rows


def test_quantile_labels_are_balanced_and_monotone():
    values = np.arange(23, dtype=float)[::-1]
    labels = rx.quantile_labels(values, 5)
    counts = np.bincount(labels, minlength=5)
    assert counts.min() >= 4 and counts.max() <= 5 and counts.sum() == 23
    assert np.all(np.diff(labels[np.argsort(values)]) >= 0)


def test_momentum_neutral_spread_equals_plain_when_score_is_independent_of_momentum():
    # Four momentum classes of four names with identical score orderings and target spreads:
    # the plain top quartile (4 names) is the four class maxima, the same as one per class.
    score = np.tile(np.arange(4, dtype=float), 4)
    target = np.tile(np.arange(4, dtype=float) / 100, 4)
    momentum = np.repeat(np.arange(4), 4).astype(object)
    plain, _, _ = quantile_long_short(score, target, None)
    neutral, _, _ = quantile_long_short(score, target, momentum)
    assert plain == pytest.approx(neutral) == pytest.approx(0.03)
    # When momentum drives both the score and the target, neutralizing it kills the spread.
    score = np.repeat(np.arange(4, dtype=float), 4)
    target = np.repeat(np.arange(4, dtype=float) / 100, 4)
    plain, _, _ = quantile_long_short(score, target, None)
    neutral, _, _ = quantile_long_short(score, target, momentum)
    assert plain > 0 and neutral == pytest.approx(0.0, abs=1e-12)


def test_trailing_features_are_past_only(tmp_path):
    _predictions, prices_path, _ = _fixture(tmp_path)
    prices = load_prices(prices_path, liquidity_lookback=20)
    features = rx.trailing_features(prices_path)
    one = features[features["symbol"] == "1103.TW"].sort_values("date").reset_index(drop=True)
    stock = prices[prices["symbol"] == "1103.TW"].sort_values("date").reset_index(drop=True)
    i = 100
    expected = np.log(stock.loc[i, "adj_close"] / stock.loc[i - 20, "adj_close"])
    assert one.loc[i, "ret20"] == pytest.approx(expected)
    assert np.isnan(one.loc[10, "ret20"]) and np.isnan(one.loc[30, "beta"])
    assert np.isfinite(one.loc[i, "beta"]) and np.isfinite(one.loc[i, "turnover_spike"])


def test_end_to_end_writes_grid_daily_and_reading(tmp_path):
    predictions_path, prices_path, _ = _fixture(tmp_path)
    output = tmp_path / "out"
    rx.BOOTSTRAP["replicates"] = 50
    rx.MIN_NAMES = 10  # the fixture's TPEX universe has fewer than 20 liquid names
    report = rx.run([predictions_path], ["good"], None, prices_path, output)
    grid = pd.read_csv(output / "grid.csv")
    assert (output / "daily.csv").is_file() and (output / "summary.md").is_file()
    assert set(grid["variant"]) == {"v0", "v1"}
    assert {"plain", "momentum", "turnover_low", "turnover_high", "correlation"} <= set(
        grid["construction"])
    assert set(grid["market"]) == {"TWSE", "TPEX"}
    # v1 is orthogonal to the trailing returns and beta on average.
    corr = grid[(grid["construction"] == "correlation") & (grid["variant"] == "v1")]
    assert (corr["spread"].abs() < 0.05).all()
    # The fixture's h10 score knows the forward move, so the plain raw spread is positive.
    plain = grid[(grid["variant"] == "v0") & (grid["construction"] == "plain")
                 & (grid["target"] == "raw") & (grid["period"] == "P1+P2")]
    assert (plain["spread"] > 0).all()
    assert {"net_annual_institution", "net_annual_retail", "cost_drag_institution"} <= set(
        grid.columns)
    reading = report["reading"]["good"]["TWSE"]
    assert {"p1_retention", "exposure_is_main_cause", "model_is_reversal_only",
            "conditioning_helps"} <= set(reading)
    # Without targets the residual columns are all NaN but still present.
    assert grid[grid["target"] == "residual"]["spread"].isna().all()
