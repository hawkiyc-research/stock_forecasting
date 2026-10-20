"""Cross-sectional gauss-rank labels: exact transform, Parquet layout and dataset alignment."""

import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.special import ndtri


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
