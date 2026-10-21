"""Feature library: fixed spec, exact operator arithmetic, adjustment invariance, store layout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _spec():
    from stock_forecasting.data.feature_library import DEFAULT_SPEC_PATH, load_spec

    return load_spec(ROOT / DEFAULT_SPEC_PATH)


def _bars(rng, length, *, factor=None):
    """A numeric symbol cache like the dataset's: raw OHLCV plus adjustment factors."""
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, length)))
    open_ = close * np.exp(rng.normal(0.0, 0.01, length))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, length)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, length)))
    volume = np.round(rng.lognormal(12.0, 0.5, length))
    raw = np.column_stack([open_, high, low, close, volume])
    factor = np.ones(length) if factor is None else factor
    timestamps = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=length, tz="UTC"))
    return {
        "raw": raw,
        "adjusted_close": close * factor,
        "timestamps": timestamps,
        "price_factor": factor,
        "volume_factor": 1.0 / factor,
    }


def test_spec_is_fixed_and_names_are_unique():
    from stock_forecasting.data.feature_library import KBAR_FEATURES, feature_names

    spec = _spec()
    names = feature_names(spec)
    assert len(names) == 183 and len(set(names)) == 183
    assert names[:9] == list(KBAR_FEATURES)
    assert names[9:14] == ["OPEN0", "OPEN1", "OPEN2", "OPEN3", "OPEN4"]
    assert "VOLUME1" in names and "VOLUME0" not in names
    assert names.index("ROC5") < names.index("ROC60") < names.index("MA5")
    assert names[-5:] == ["REL_RET_5", "REL_RET_20", "REL_RET_60", "DVOL_RANK_20", "MARKET_IS_TPEX"]


def test_spec_validation_rejects_unknown_operators(tmp_path):
    from stock_forecasting.data.feature_library import load_spec

    spec = json.loads((ROOT / "configs/features/alpha_tw_v1.json").read_text())
    spec["rolling"]["operators"].append("MAGIC")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unknown operators"):
        load_spec(path)


def _reference(bars, benchmark, size, t):
    """Loop reference for a subset of operators at one row t (window ends at t)."""
    raw, eps = bars["raw"], 1e-12
    o, h, lo, c = (raw[:, i] * bars["price_factor"] for i in range(4))
    v = raw[:, 4] * bars["volume_factor"]
    window = slice(t - size + 1, t + 1)
    y = c[window]
    x = np.arange(size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    total = ((y - y.mean()) ** 2).sum()
    diff = np.diff(c[t - size : t + 1])
    vdiff = np.diff(v[t - size : t + 1])
    ratio = np.abs(c[t - size : t + 1][1:] / c[t - size : t + 1][:-1] - 1.0) * v[window]
    bench = benchmark["adjusted_close"][benchmark["timestamps"].get_indexer(bars["timestamps"])]
    return {
        f"ROC{size}": c[t - size] / c[t],
        f"MA{size}": y.mean() / c[t],
        f"STD{size}": y.std(ddof=1) / c[t],
        f"BETA{size}": slope / c[t],
        f"RSQR{size}": ((fitted - y.mean()) ** 2).sum() / total,
        f"RESI{size}": (y[-1] - fitted[-1]) / c[t],
        f"MAX{size}": h[window].max() / c[t],
        f"MIN{size}": lo[window].min() / c[t],
        f"QTLU{size}": np.quantile(y, 0.8) / c[t],
        f"RANK{size}": (np.sum(y < y[-1]) + 0.5 * np.sum(y == y[-1]) + 0.5) / size,
        f"RSV{size}": (c[t] - lo[window].min()) / (h[window].max() - lo[window].min() + eps),
        f"IMAX{size}": (np.argmax(h[window]) + 1) / size,
        f"IMIN{size}": (np.argmin(lo[window]) + 1) / size,
        f"CORR{size}": np.corrcoef(y, np.log1p(v[window]))[0, 1],
        f"CNTP{size}": np.mean(diff > 0),
        f"SUMP{size}": np.maximum(diff, 0).sum() / (np.abs(diff).sum() + eps),
        f"VMA{size}": v[window].mean() / (v[t] + eps),
        f"WVMA{size}": ratio.std(ddof=1) / (ratio.mean() + eps),
        f"VSUMP{size}": np.maximum(vdiff, 0).sum() / (np.abs(vdiff).sum() + eps),
        "KMID": (c[t] - o[t]) / o[t],
        "KUP2": (h[t] - max(o[t], c[t])) / (h[t] - lo[t] + eps),
        "OPEN2": o[t - 2] / c[t],
        "VWAP0": (h[t] + lo[t] + c[t]) / 3.0 / c[t],
        "VOLUME3": v[t - 3] / (v[t] + eps),
        "REL_RET_5": np.log(c[t] / c[t - 5]) - np.log(bench[t] / bench[t - 5]),
    }


def test_symbol_features_match_loop_reference():
    from stock_forecasting.data.feature_library import feature_names, symbol_features

    spec = _spec()
    rng = np.random.default_rng(3)
    bars, benchmark = _bars(rng, 120), _bars(rng, 120)
    values = symbol_features(bars, benchmark, spec, is_indicator_market=True)
    names = feature_names(spec)
    assert values.shape == (120, len(names))
    for size in (5, 10, 20):
        for t in (70, 119):
            expected = _reference(bars, benchmark, size, t)
            for name, value in expected.items():
                np.testing.assert_allclose(
                    values[t, names.index(name)], value, rtol=1e-9, atol=1e-12, err_msg=name
                )
    assert values[:, names.index("MARKET_IS_TPEX")].tolist() == [1.0] * 120
    # Insufficient history is NaN, never a silently shortened window.
    roc60 = values[:, names.index("ROC60")]
    assert np.isnan(roc60[58]) and np.isfinite(roc60[60])
    assert np.isnan(values[3, names.index("MA5")]) and np.isfinite(values[4, names.index("MA5")])
    dollar = bars["raw"][:, 3] * bars["raw"][:, 4]
    np.testing.assert_allclose(values[30, names.index("DVOL_RANK_20")], dollar[11:31].mean())


def test_ratio_features_are_invariant_to_split_adjustment():
    from stock_forecasting.data.feature_library import feature_names, symbol_features

    spec = _spec()
    rng = np.random.default_rng(5)
    continuous = _bars(rng, 100)
    factor = np.where(np.arange(100) < 50, 0.5, 1.0)
    jumped = dict(continuous)
    # Raw prices before a 2:1 split are twice the adjusted ones; volumes are half.
    raw = continuous["raw"].copy()
    raw[:, :4] /= factor[:, None]
    raw[:, 4] *= factor
    jumped.update({"raw": raw, "price_factor": factor, "volume_factor": 1.0 / factor})
    benchmark = _bars(rng, 100)
    names = feature_names(spec)
    keep = [i for i, name in enumerate(names) if name not in ("DVOL_RANK_20", "MARKET_IS_TPEX")]
    a = symbol_features(continuous, benchmark, spec, is_indicator_market=False)[:, keep]
    b = symbol_features(jumped, benchmark, spec, is_indicator_market=False)[:, keep]
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_percentile_rank_handles_ties_and_missing_values():
    from stock_forecasting.data.feature_library import percentile_rank

    values = np.asarray([3.0, 1.0, 2.0, 2.0, np.nan, 9.0, 7.0])
    groups = np.asarray([0, 0, 0, 0, 0, 1, 1])
    ranks = percentile_rank(values, groups)
    np.testing.assert_allclose(ranks[:4], (np.asarray([4.0, 1.0, 2.5, 2.5]) - 0.5) / 4)
    assert np.isnan(ranks[4])
    np.testing.assert_allclose(ranks[5:], [0.75, 0.25])


@pytest.mark.parametrize("split", ["validation", "test"])
def test_store_layout_values_and_cross_sectional_rank(rank_config, split, tmp_path):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data.feature_library import (
        attach_features,
        ensure_features,
        feature_names,
        load_spec,
        read_split_features,
        symbol_features,
    )

    spec = json.loads((ROOT / "configs/features/alpha_tw_v1.json").read_text())
    spec["taiwan"]["market_indicator"] = "TWSE"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    spec = load_spec(path)
    names = feature_names(spec)
    source = lazy_dataset(rank_config, split)
    try:
        with pytest.raises(ValueError, match="missing"):
            attach_features(source, spec)
        root = ensure_features(source, spec, workers=1)
        assert root.parent.name == "features" and root.parent.parent == source.root.parent
        marker = (root / "features.json").stat().st_mtime_ns
        assert ensure_features(source, spec, workers=1) == root
        assert attach_features(source, spec) == root
        assert (root / "features.json").stat().st_mtime_ns == marker
        assert not list((root / "work").glob("*.npy"))
        payload = json.loads((root / "features.json").read_text())
        assert payload["feature_names"] == names
        assert payload["columns"]["MA5"]["finite_fraction"] == 1.0
        assert 0.0 < payload["columns"]["ROC60"]["finite_fraction"] <= 1.0
        index = pd.read_parquet(root / "feature-index.parquet")
        spans = source.symbol_spans()
        assert set(index["symbol"]) == set(spans)
        for row in index.itertuples():
            handle = pq.ParquetFile(root / row.shard_relative_path)
            assert handle.metadata.row_group(row.row_group).num_rows == row.row_count
            assert spans[row.symbol] == (row.ordinal_begin, row.ordinal_begin + row.row_count)
        frame = read_split_features(root)
        assert len(frame) == len(source) and frame["ordinal"].tolist() == list(range(len(source)))
        labels = source.target_batch(np.arange(len(source)))
        stamps = pd.DatetimeIndex(frame["timestamp"]).asi8
        np.testing.assert_array_equal(stamps, labels["cutoff_ns"])
        markets = pd.Series(labels["markets"])
        np.testing.assert_array_equal(
            frame["MARKET_IS_TWSE"].to_numpy(), (markets == "TWSE").to_numpy(dtype=np.float32)
        )
        # Five stocks per market and date: ranks are a permutation of (i - 0.5) / 5.
        grouped = frame.assign(market=markets.to_numpy()).groupby(["timestamp", "market"])
        for _, rows in grouped:
            np.testing.assert_allclose(
                np.sort(rows["DVOL_RANK_20"].to_numpy()), (np.arange(5) + 0.5) / 5, rtol=1e-6
            )
        # One symbol's rows equal the direct computation at its cutoffs.
        symbol, (begin, end) = next(iter(spans.items()))
        asset = source._numeric_symbol(symbol)
        benchmark = source._numeric_symbol(str(source._index[symbol]["benchmark_symbol"]))
        direct = symbol_features(
            asset, benchmark, spec,
            is_indicator_market=str(source._index[symbol]["market"]) == "TWSE",
        )
        cutoffs = asset["timestamps"].get_indexer(pd.DatetimeIndex(frame["timestamp"][begin:end]))
        stored = frame.iloc[begin:end][names].to_numpy(dtype=np.float32)
        keep = [i for i, name in enumerate(names) if name != "DVOL_RANK_20"]
        np.testing.assert_allclose(
            stored[:, keep], direct[cutoffs][:, keep].astype(np.float32), rtol=1e-6, equal_nan=True
        )
    finally:
        source.close()
