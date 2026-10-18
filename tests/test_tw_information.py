"""Cloud tests for the Taiwan information-test features, metrics and report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.research import tw_evaluation as evaluation
from stock_forecasting.research import tw_features as features
from stock_forecasting.research import tw_information as information

ROOT = Path(__file__).resolve().parents[1]


def bars(count=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, count)))
    opens = close * np.exp(rng.normal(0, 0.005, count))
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-03", periods=count, freq="B", tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, close) * 1.01,
        "low": np.minimum(opens, close) * 0.99,
        "close": close,
        "adjusted_close": close,
        "volume": rng.integers(1_000, 10_000, count).astype(float),
    })


def test_technical_features_never_read_future_sessions():
    original = features.adjusted_bars(bars())
    changed = original.copy()
    changed.loc[200:, ["open", "high", "low", "close", "volume"]] *= 3.0
    left = features.technical_frame(original, window=128, long_lookback=True).iloc[:200]
    right = features.technical_frame(changed, window=128, long_lookback=True).iloc[:200]
    pd.testing.assert_frame_equal(left, right)


def test_next_open_returns_enter_at_the_next_open_and_exit_h_closes_later():
    frame = pd.DataFrame({"open": [10.0, 11.0, 12.0, 13.0], "close": [10.5, 11.5, 12.5, 13.5]})
    returns = features.next_open_log_returns(frame, (1, 2))
    assert returns[0, 0] == pytest.approx(np.log(11.5 / 11.0))
    assert returns[0, 1] == pytest.approx(np.log(12.5 / 11.0))
    assert np.isnan(returns[3]).all() and np.isnan(returns[2, 1])


def test_flow_features_scale_by_volume_and_track_balances():
    frame = features.adjusted_bars(bars(30))
    flows = pd.DataFrame({
        "date": frame["date"],
        "foreign_net_shares": 100.0,
        "trust_net_shares": 0.0,
        "dealer_net_shares": 0.0,
        "institutional_net_shares": 100.0,
        "margin_balance_lots": np.arange(1, 31, dtype=float),
        "margin_limit_lots": 100.0,
        "short_balance_lots": 2.0,
        "margin_buy_lots": 1.0,
        "short_sell_lots": 0.0,
    })
    result = features.flow_features(frame, flows)
    assert result.loc[5, "foreign_ratio_1d"] == pytest.approx(100.0 / frame.loc[5, "volume"])
    assert result.loc[5, "margin_change_1d"] == pytest.approx(6 / 5 - 1)
    assert result.loc[5, "short_margin_ratio"] == pytest.approx(2 / 6)
    empty = features.flow_features(frame, None)
    assert empty[list(features.FLOW)].isna().all().all()


def test_us_alignment_respects_the_taiwan_signal_time():
    us = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
        "spy_us_r1": [0.01, 0.02, 0.05],
    })
    taiwan = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]))
    pre_open = features.align_us(taiwan, us, variant="pre_open")["us_pre_open_spy_us_r1"]
    evening = features.align_us(taiwan, us, variant="evening")["us_evening_spy_us_r1"]
    assert pre_open.tolist() == [0.02, 0.02, 0.05]
    assert evening.tolist() == [0.01, 0.02, 0.02]


def test_hybrid_band_labels_calibration_and_cost_floor():
    labels = evaluation.band_labels(np.log1p(np.array([0.01, 0.0101, -0.0101, np.nan])), 0.01)
    assert labels.tolist() == [1, 2, 0, -1]
    rng = np.random.default_rng(3)
    sigma = rng.uniform(0.01, 0.04, 20_000)
    returns = rng.normal(0, sigma)
    floors = np.full_like(sigma, 0.00585)
    calibration = evaluation.calibrate_k(returns, sigma, floors, 1, 1 / 3)
    assert calibration["neutral_share"] == pytest.approx(1 / 3, abs=0.01)
    bands = evaluation.hybrid_bands(sigma, floors, calibration["k"], 1)
    assert (bands >= floors).all()
    tiny = evaluation.calibrate_k(returns * 1e-3, sigma, floors, 1, 1 / 3)
    assert tiny["floor_binding"] and tiny["k"] == 0.0


def test_ce_parts_split_move_and_direction():
    probabilities = np.array([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [0.2, 0.5, 0.3]])
    parts = evaluation.ce_parts(probabilities, np.array([2, 1, 0], dtype=np.int8))
    assert parts["ce"][0] == pytest.approx(-np.log(0.3))
    assert parts["move"][1] == pytest.approx(-np.log(0.5))
    assert parts["move"][0] == pytest.approx(-np.log(0.5))
    assert parts["direction"][0] == pytest.approx(-np.log(0.6))
    assert parts["direction"][2] == pytest.approx(-np.log(0.4))
    assert np.isnan(parts["direction"][1])
    assert parts["ce"][0] == pytest.approx(parts["move"][0] + parts["direction"][0])


def test_bootstrap_rank_ic_and_date_means():
    constant = evaluation.block_bootstrap(
        np.full(50, 2.0), np.full(50, 4.0), replicates=200, block=5, seed=1
    )
    assert constant == {"estimate": 0.5, "low": 0.5, "high": 0.5}
    dates = np.repeat(np.arange(4), 5)
    markets = np.array(["TWSE"] * 20)
    target = np.tile(np.arange(5.0), 4)
    ic = evaluation.rank_ic(target * 2, target, dates, markets)
    assert ic["TWSE"]["mean"] == pytest.approx(1.0) and ic["TWSE"]["dates"] == 4
    assert evaluation.rank_ic(np.ones(20), target, dates, markets) == {}
    keys = pd.DataFrame({"date": [1, 1, 2], "market": ["A", "A", "A"]})
    means = evaluation.date_only_probabilities(
        np.array([[0.2, 0.2, 0.6], [0.4, 0.2, 0.4], [0.1, 0.1, 0.8]]), keys
    )
    assert means[0] == pytest.approx([0.3, 0.2, 0.5]) and means[2] == pytest.approx([0.1, 0.1, 0.8])


def test_fits_leave_out_columns_without_train_variation():
    rng = np.random.default_rng(5)
    rows = 3_000
    informative = rng.normal(size=rows)
    x = np.column_stack([informative, np.full(rows, np.nan), np.full(rows, 2.0)]).astype(np.float32)
    assert information.informative_columns(x).tolist() == [True, False, False]
    labels = np.where(informative > 0.5, 2, np.where(informative < -0.5, 0, 1)).astype(np.int8)
    dates = np.repeat(np.arange(100), rows // 100)
    model = {"learning_rate": 0.1, "max_iter": 10, "max_leaf_nodes": 7, "min_samples_leaf": 20,
             "l2_regularization": 0.0, "stage_step": 5, "holdout_fraction": 0.2,
             "holdout_embargo_sessions": 2, "seed": 0}
    estimator, stage, _loss = information.fit_classifier(x, labels, dates, model)
    assert information._left_out(["a", "b", "c"], estimator) == ["b", "c"]
    probabilities = information._at_stage(estimator, x[:10], stage)
    assert probabilities.shape == (10, 3)


def _sums(values):
    return pd.DataFrame({"sum": values, "count": np.full(len(values), 10.0)},
                        index=pd.RangeIndex(len(values)))


def test_criteria_require_relative_gain_and_intervals_above_zero():
    days = 40
    prior = _sums(np.full(days, 6.93))
    strong = _sums(np.full(days, 6.70))
    noisy = _sums(6.93 + np.where(np.arange(days) % 2, 0.5, -0.5))
    sums = {("F0", h): prior for h in (1, 2, 3)}
    sums.update({("flows", h): strong for h in (1, 2, 3)})
    sums.update({("market", h): noisy for h in (1, 2, 3)})
    outcome = evaluation.evaluate_criteria(
        sums, {h: prior for h in (1, 2, 3)}, groups=["F0", "flows", "market"],
        criteria={"decision_horizons": [1, 2, 3], "min_passing_horizons": 2,
                  "min_relative_improvement": 0.01},
        bootstrap={"replicates": 200, "block": 5, "seed": 1},
    )
    assert outcome["flows"]["passed"] and not outcome["market"]["passed"]
    assert outcome["information_found"]


def synthetic_panel(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-06-01", periods=260)
    split = np.where(
        dates < dates[180], "train", np.where(dates < dates[220], "validation", "test")
    )
    rows = []
    for market in ("TWSE", "TPEX"):
        for symbol in range(25):
            for index, date in enumerate(dates):
                rows.append((split[index], date, f"{symbol}.{market}", market))
    panel = pd.DataFrame(rows, columns=["split", "date", "symbol", "market"])
    count = len(panel)
    panel["asset_type"] = "stock"
    panel["close_raw"] = 100.0
    panel["volume_raw"] = 1_000.0
    numeric = [
        *features.TECHNICAL, *(f"bench_{name}" for name in features.TECHNICAL), *features.LONG,
        *features.FLOW, *features.RAW_COLUMNS[2:],
    ]
    for column in numeric:
        panel[column] = rng.normal(size=count).astype(np.float32)
    for column in (*features.FLOW, *features.RAW_COLUMNS[2:]):
        panel[column] = np.nan
    panel["sigma_20d"] = 0.02
    for horizon in (1, 2, 3, 5, 10, 14):
        signal = 0.01 * panel["return_1d"].to_numpy()
        panel[f"return_{horizon}"] = signal + rng.normal(0, 0.02 * np.sqrt(horizon), count)
        panel[f"alpha_{horizon}"] = panel[f"return_{horizon}"] * 0.5
    return panel


def test_end_to_end_report_from_a_synthetic_panel(tmp_path):
    config = json.loads((ROOT / "configs/research/tw_information_test.json").read_text())
    config["model"].update({"max_iter": 20, "stage_step": 5, "min_samples_leaf": 20})
    config["bootstrap"]["replicates"] = 50
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    auxiliary = tmp_path / "aux"
    auxiliary.mkdir()
    (auxiliary / "auxiliary-manifest.json").write_text(json.dumps({"sources": {
        "twse_t86": {"status": "failed"}, "taifex_futures": {"status": "failed"},
    }}))
    us = {"SPY.US": pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=400),
                                  "close": np.linspace(100, 120, 400)})}
    report = information.run(
        config_path, auxiliary, tmp_path / "out", workers=1, panel=synthetic_panel(), us=us
    )
    assert report["skipped_feature_sets"] == ["all_evening", "all_pre_open", "flows"]
    assert report["bands"]["1"]["neutral_share"] == pytest.approx(1 / 3, abs=0.02)
    validation = report["results"]["validation"]["F0"]["1"]["hybrid_classification"]["TW"]
    assert validation["samples"] > 0 and np.isfinite(validation["direction"])
    assert "F0" in report["results"]["test"] and "us_pre_open" in report["criteria"]
    # Flow aggregates are empty without flow sources and are left out, not fatal.
    market = next(fit for fit in report["fits"] if fit["set"] == "market")
    assert "agg_foreign_ratio_1d" in market["left_out_features"]
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# 台股資訊測試摘要")
    with pytest.raises(FileExistsError):
        information.run(config_path, auxiliary, tmp_path / "out", workers=1,
                        panel=synthetic_panel(), us=us)
