"""Cross-sectional gauss-rank labels: exact transform, Parquet layout and dataset alignment."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
US_STOCKS = ("AAAA.US", "BBBB.US", "CCCC.US", "DDDD.US", "EEEE.US")
TW_STOCKS = ("1101.TW", "1102.TW", "1103.TW", "1104.TW", "1216.TW")


def ranked_frame() -> pd.DataFrame:
    from stock_forecasting.data.sample_universe import market_sessions

    rng = np.random.default_rng(7)
    instruments = [
        ("VTI.US", "etf", "US", "eodhd", 220.0),
        ("TAIEX.TW", "index", "TWSE", "twse_official", 15000.0),
        *((symbol, "stock", "US", "eodhd", 50.0 + 10 * i) for i, symbol in enumerate(US_STOCKS)),
        *(
            (symbol, "stock", "TWSE", "twse_official", 30.0 + 5 * i)
            for i, symbol in enumerate(TW_STOCKS)
        ),
    ]
    frames = []
    for symbol, asset_type, market, provider, base in instruments:
        dates = market_sessions("XNYS" if market == "US" else "XTAI", "2022-01-03", "2023-12-31")[
            :360
        ]
        steps = rng.normal(0.0003, 0.015, len(dates))
        close = base * np.exp(np.cumsum(steps))
        # Distinct intraday moves give every cutoff a strict cross-sectional order.
        open_ = close * np.exp(rng.normal(0.0, 0.006, len(dates)))
        index = np.arange(len(dates), dtype=np.float64)
        factor = np.exp(0.00005 * index) if asset_type == "index" else np.ones_like(index)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.004,
                    "low": np.minimum(open_, close) * 0.996,
                    "close": close,
                    "volume": 1_000_000.0 + index * 100.0,
                    "adjusted_close": close * factor,
                    "split_adjusted_volume": 1_000_000.0 + index * 100.0,
                    "adjustment_source": (
                        "official_total_return_index" if asset_type == "index" else "fixture"
                    ),
                    "provider": provider,
                    "market": market,
                    "currency": "USD" if market == "US" else "TWD",
                    "source_symbol": symbol.split(".", maxsplit=1)[0],
                    "is_active": True,
                    "dataset_profile": "us_tw_eodhd",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def rank_config(tmp_path, monkeypatch):
    import stock_forecasting.data.rank_labels as rank_labels
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.manifest import artifact_metadata

    # Linux-only memory probes; a fixed budget keeps the local run deterministic.
    monkeypatch.setattr(
        rank_labels,
        "detect_available_memory",
        lambda: SimpleNamespace(available_bytes=16 * 1024**3),
    )
    frame = ranked_frame()
    raw, store = tmp_path / "rank.parquet", tmp_path / "prepared" / "bar-store"
    frame.to_parquet(raw, index=False)
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(frame))}
        },
        window_size=32,
        bucket_count=4,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.bar_store_path, config.data.input_length = store, 32
    config.data.h_start = 1
    config.model.direction.enabled = False
    return config


def expected_ranks(source, min_group_size):
    batch = source.target_batch(np.arange(len(source)))
    frame = pd.DataFrame({"cutoff": batch["cutoff_ns"], "market": batch["markets"]})
    result = np.empty(batch["targets"].shape, dtype=np.float64)
    groups = frame.groupby(["cutoff", "market"], sort=False)
    sizes = groups["cutoff"].transform("size").to_numpy()
    for i in range(batch["targets"].shape[1]):
        frame["value"] = batch["targets"][:, i]
        rank = frame.groupby(["cutoff", "market"], sort=False)["value"].rank(method="average")
        result[:, i] = ndtri((rank.to_numpy() - 0.5) / sizes)
    result[sizes < min_group_size] = np.nan
    return batch, result


def test_gauss_rank_average_ties_and_small_group_mask():
    from stock_forecasting.data.rank_labels import gauss_rank, group_ids

    values = np.asarray([3.0, 1.0, 2.0, 2.0, 5.0, 9.0, 7.0, 4.0])
    cutoff = np.asarray([1, 1, 1, 1, 1, 2, 2, 1], dtype=np.int64)
    markets = np.asarray([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    groups = group_ids(cutoff, markets)
    assert len(np.unique(groups)) == 3
    labels = gauss_rank(values, groups, 3)
    # Group of five: ranks 4, 1, 2.5, 2.5, 5.
    expected = ndtri((np.asarray([4.0, 1.0, 2.5, 2.5, 5.0]) - 0.5) / 5)
    np.testing.assert_allclose(labels[:5], expected, rtol=1e-6)
    assert labels[2] == labels[3]
    assert np.isnan(labels[5:]).all()
    # Without ties the labels are symmetric around zero.
    free = gauss_rank(np.asarray([0.4, -1.0, 2.0, 0.1]), np.zeros(4, dtype=np.int64), 2)
    assert abs(float(free.mean())) < 1e-6


def test_target_batch_matches_array_batch(rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset

    source = lazy_dataset(rank_config, "validation")
    try:
        indices = np.asarray([0, 5, len(source) // 2, len(source) - 1])
        full = source.array_batch(indices)
        labels = source.target_batch(indices)
        np.testing.assert_array_equal(labels["targets"], full["targets"])
        assert labels["cutoff_ns"].tolist() == [m[2].value for m in full["metadata"]]
        assert labels["markets"].tolist() == [str(m[3]["market"]) for m in full["metadata"]]
    finally:
        source.close()


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_store_layout_values_and_dataset_alignment(rank_config, split):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data.rank_labels import attach_rank_labels, ensure_rank_labels

    source = lazy_dataset(rank_config, split)
    try:
        with pytest.raises(ValueError, match="missing"):
            attach_rank_labels(source, 3)
        root = ensure_rank_labels(source, 3, workers=1)
        assert root.parent.name == "rank-labels"
        assert root.parent.parent == source.root.parent
        marker = (root / "rank-labels.json").stat().st_mtime_ns
        assert ensure_rank_labels(source, 3, workers=1) == root
        assert (root / "rank-labels.json").stat().st_mtime_ns == marker
        assert not list((root / "work").glob("*.npy"))
        index = pd.read_parquet(root / "label-index.parquet")
        spans = source.symbol_spans()
        assert set(index["symbol"]) == set(spans)
        for row in index.itertuples():
            handle = pq.ParquetFile(root / row.shard_relative_path)
            group = handle.metadata.row_group(row.row_group)
            assert group.num_rows == row.row_count == spans[row.symbol][1] - row.ordinal_begin
            assert handle.schema_arrow.names == ["timestamp"] + [
                f"rank_h{h:02d}" for h in source.horizons
            ]
        batch, expected = expected_ranks(source, 3)
        attach_rank_labels(source, 3)
        actual = source.array_batch(np.arange(len(source)))["rank_targets"]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
        assert np.isfinite(actual).mean() > 0.9
        # Labels are monotone in the raw return inside each (cutoff, market) group.
        frame = pd.DataFrame({"cutoff": batch["cutoff_ns"], "market": batch["markets"]})
        for _, rows in frame.groupby(["cutoff", "market"]).groups.items():
            rows = np.asarray(rows)
            if np.isfinite(actual[rows, 0]).all():
                order = np.argsort(batch["targets"][rows, 4], kind="stable")
                assert np.all(np.diff(actual[rows, 4][order]) >= 0)
        restored = pickle.loads(pickle.dumps(source))
        try:
            picked = np.asarray([len(source) - 1, 0, 3])
            np.testing.assert_array_equal(
                restored.array_batch(picked)["rank_targets"], actual[picked]
            )
        finally:
            restored.close()
    finally:
        source.close()


def test_attach_rejects_another_split_and_misaligned_cutoffs(rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data.rank_labels import ensure_rank_labels

    train = lazy_dataset(rank_config, "train")
    validation = lazy_dataset(rank_config, "validation")
    try:
        root = ensure_rank_labels(train, 3, workers=1)
        with pytest.raises(ValueError, match="different split population"):
            validation.attach_rank_labels(root)
        train.attach_rank_labels(root)
        symbol = next(iter(train.symbol_spans()))
        train._rank_index[symbol] = {**train._rank_index[symbol], "ordinal_begin": -1}
        with pytest.raises(RuntimeError, match="misaligned"):
            train.array_batch([0])
    finally:
        train.close()
        validation.close()


def test_process_pool_build_matches_reference(rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data.rank_labels import ensure_rank_labels

    source = lazy_dataset(rank_config, "train")
    try:
        root = ensure_rank_labels(source, 3, workers=2)
        _batch, expected = expected_ranks(source, 3)
        source.attach_rank_labels(root)
        actual = source.array_batch(np.arange(len(source)))["rank_targets"]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    finally:
        source.close()
