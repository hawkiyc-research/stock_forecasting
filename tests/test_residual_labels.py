"""Residualized gauss-rank labels (protocol 13.16): design matrix, orthogonality, layout,
identity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest


def _spec(tmp_path, source, *, size_proxy="log_dollar_volume_median", fallback=None):
    """A spec whose reference table covers half of the fixture's symbols."""
    from stock_forecasting.data import residual_labels

    symbols = sorted(source.symbol_spans())
    rows = ["symbol,market,industry_code,issued_shares,par_value_twd,snapshot_date"]
    for i, symbol in enumerate(symbols):
        if i % 2:
            continue  # the odd symbols stay unknown
        code = "24" if i % 4 == 0 else "17"
        rows.append(f"{symbol},TWSE,{code},{1_000_000 * (i + 1)},10,2026-10-22")
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = tmp_path / "reference.csv"
    reference.write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = {
        "schema": residual_labels.RESIDUAL_LABEL_SCHEMA,
        "name": "residual_test",
        "size_proxy": size_proxy,
        "dollar_volume_window": 5,
        "industry": {
            "reference": str(reference),
            "unknown_class": "UNK",
            "min_names_per_class": 2,
            "min_group_for_industry": 4,
        },
    }
    if fallback:
        spec["fallback_size_proxy"] = fallback
    path = tmp_path / "residual_test.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return residual_labels.load_label_spec(path)


def test_design_matrix_merges_small_classes_and_skips_industry_for_small_groups():
    from stock_forecasting.data.residual_labels import OTHER_CLASS, design_matrix

    size = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    industry = np.array(["A", "A", "A", "B", "B", "C"], dtype=object)
    design, used = design_matrix(size, industry, min_group_for_industry=4, min_names_per_class=2)
    assert used and design.shape == (6, 4)  # intercept, size, two non-baseline classes (B, other)
    assert np.allclose(design[:, 0], 1.0) and abs(design[:, 1].mean()) < 1e-12
    assert design[5, 2:].sum() == 1.0  # "C" was merged into the other class dummy
    small, used_small = design_matrix(size[:3], industry[:3], min_group_for_industry=4,
                                      min_names_per_class=2)
    assert not used_small and small.shape == (3, 2)
    assert OTHER_CLASS not in industry


def test_residualize_removes_the_fitted_part():
    from stock_forecasting.data.residual_labels import residualize

    rng = np.random.default_rng(1)
    design = np.column_stack([np.ones(50), rng.normal(size=50)])
    targets = design @ np.array([[0.5, -1.0], [2.0, 0.3]]) + rng.normal(scale=0.1, size=(50, 2))
    residual = residualize(targets, design)
    assert np.allclose(design.T @ residual, 0.0, atol=1e-9)


def test_spec_validation(tmp_path, rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data import residual_labels

    source = lazy_dataset(rank_config, "validation")
    try:
        spec = _spec(tmp_path, source)
        assert spec["name"] == "residual_test" and spec["_reference_path"].endswith("reference.csv")
        bad = dict(spec)
        bad.pop("_reference_path")
        bad["size_proxy"] = "log_price"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="size_proxy"):
            residual_labels.load_label_spec(path)
    finally:
        source.close()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_store_layout_identity_and_orthogonality(tmp_path, rank_config, split):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data import residual_labels
    from stock_forecasting.data.rank_labels import label_root

    source = lazy_dataset(rank_config, split)
    try:
        spec = _spec(tmp_path, source)
        with pytest.raises(ValueError, match="missing"):
            residual_labels.attach_residual_labels(source, 3, spec)
        root = residual_labels.ensure_residual_labels(source, 3, spec, workers=1)
        assert root.parent.name == "rank-labels"
        standard_root, _ = label_root(source, 3)
        assert root != standard_root  # a distinct identity, the standard store is untouched
        marker = (root / "rank-labels.json").stat().st_mtime_ns
        assert residual_labels.ensure_residual_labels(source, 3, spec, workers=1) == root
        assert (root / "rank-labels.json").stat().st_mtime_ns == marker
        assert residual_labels.attach_residual_labels(source, 3, spec) == root
        assert not list((root / "work").glob("*.npy"))
        payload = json.loads((root / "rank-labels.json").read_text())
        assert payload["schema"] == residual_labels.RESIDUAL_LABEL_SCHEMA
        assert payload["identity"]["residualization"]["name"] == "residual_test"
        assert payload["residualization"]["rows"] == len(source)
        assert 0 < payload["residualization"]["rows_with_known_industry"] < len(source)
        index = pd.read_parquet(root / "label-index.parquet")
        spans = source.symbol_spans()
        assert set(index["symbol"]) == set(spans)
        horizons = list(source.horizons)
        for row in index.itertuples():
            handle = pq.ParquetFile(root / row.shard_relative_path)
            assert handle.schema_arrow.names == residual_labels.expected_columns(horizons)
            assert handle.metadata.row_group(row.row_group).num_rows == row.row_count
        # Residuals are orthogonal to the size proxy within every (cutoff, market) group and
        # the ranks are finite wherever the group is large enough.
        batch = source.target_batch(np.arange(len(source)))
        frames = []
        for row in index.sort_values("ordinal_begin").itertuples():
            table = pq.ParquetFile(root / row.shard_relative_path).read_row_group(
                int(row.row_group)).to_pandas()
            table["ordinal"] = np.arange(row.ordinal_begin, row.ordinal_begin + len(table))
            frames.append(table)
        store = pd.concat(frames).sort_values("ordinal").reset_index(drop=True)
        store["cutoff"] = batch["cutoff_ns"]
        store["market"] = batch["markets"]
        column = residual_labels.residual_column(horizons[0])
        group_means = store.groupby(["cutoff", "market"])[column].mean()
        assert float(group_means.abs().max()) < 1e-4
        ranks = store[[f"rank_h{h:02d}" for h in horizons]].to_numpy()
        sizes = store.groupby(["cutoff", "market"])[column].transform("size").to_numpy()
        assert np.array_equal(np.isfinite(ranks[:, 0]), sizes >= 3)
        assert np.isfinite(ranks).mean() > 0.9
    finally:
        source.close()


def test_market_cap_variant_falls_back_for_missing_shares(tmp_path, rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data import residual_labels

    source = lazy_dataset(rank_config, "validation")
    try:
        spec = _spec(tmp_path, source, size_proxy="log_market_cap_snapshot",
                     fallback="log_dollar_volume_median")
        root = residual_labels.ensure_residual_labels(source, 3, spec, workers=1)
        payload = json.loads((root / "rank-labels.json").read_text())
        assert payload["residualization"]["size_proxy"] == "log_market_cap_snapshot"
        assert 0 < payload["residualization"]["fallback_rows"] < len(source)
        strict = _spec(tmp_path / "strict", source, size_proxy="log_market_cap_snapshot")
        with pytest.raises(ValueError, match="no fallback"):
            residual_labels.ensure_residual_labels(source, 3, strict, workers=1)
    finally:
        source.close()
