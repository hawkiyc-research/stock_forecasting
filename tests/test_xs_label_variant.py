"""Label-variant arrays beside the cross-stock cache (protocol 13.16): layout, identity,
alignment checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from test_residual_labels import _spec


class _FakeCache:
    """The parts of an XsCache the variant builder reads."""

    def __init__(self, root, split, horizons, cutoff_ns):
        self.root, self.split, self.horizons, self.cutoff_ns = root, split, horizons, cutoff_ns

    def __len__(self) -> int:
        return len(self.cutoff_ns)


def test_variant_arrays_are_built_once_and_aligned_to_the_cache(tmp_path, rank_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data import residual_labels, xs_label_variant

    source = lazy_dataset(rank_config, "train")
    try:
        spec = _spec(tmp_path / "spec", source)
        horizons = list(source.horizons)
        batch = source.target_batch(np.arange(len(source)))
        cutoffs = np.asarray(batch["cutoff_ns"], dtype=np.int64)
        cache_root = tmp_path / "xs-cache" / "deadbeef"
        cache_root.mkdir(parents=True)
        cache = _FakeCache(cache_root, "train", horizons, cutoffs)
        # The residual store is an explicit earlier step.
        with pytest.raises(ValueError, match="missing"):
            xs_label_variant.ensure_label_variant(cache, source, 3, spec)
        label_root = residual_labels.ensure_residual_labels(source, 3, spec, workers=1)
        root = xs_label_variant.ensure_label_variant(cache, source, 3, spec)
        assert root.parent == cache_root / "label-variants"
        marker = json.loads((root / xs_label_variant.MARKER_NAME).read_text())
        assert marker["state"] == "ready" and marker["variant"] == "residual_test"
        assert marker["identity"]["cache"] == "deadbeef"
        assert marker["label_root"] == str(label_root)
        assert set(marker["files"]) == {"ranks.npy", "residual.npy"}
        stamp = (root / xs_label_variant.MARKER_NAME).stat().st_mtime_ns
        assert xs_label_variant.ensure_label_variant(cache, source, 3, spec) == root
        assert (root / xs_label_variant.MARKER_NAME).stat().st_mtime_ns == stamp
        arrays = xs_label_variant.open_label_variant(root)
        assert len(arrays) == len(source) and arrays.variant == "residual_test"
        assert arrays.ranks.shape == arrays.residual.shape == (len(source), len(horizons))
        assert np.isfinite(arrays.residual).all()
        finite = np.isfinite(arrays.ranks[:, 0])
        assert 0.9 < finite.mean() <= 1.0
        # Gauss-rank labels: symmetric around zero within every group.
        assert abs(float(np.nanmean(arrays.ranks))) < 0.05
        # Another cache digest gets its own arrays; a misaligned cache is refused.
        other_root = tmp_path / "xs-cache" / "cafef00d"
        other_root.mkdir()
        other = xs_label_variant.ensure_label_variant(
            _FakeCache(other_root, "train", horizons, cutoffs), source, 3, spec
        )
        assert other != root and other.parent == other_root / "label-variants"
        shifted = _FakeCache(tmp_path / "xs-cache" / "0bad", "train", horizons, cutoffs + 1)
        shifted.root.mkdir()
        with pytest.raises(ValueError, match="cutoffs differ"):
            xs_label_variant.ensure_label_variant(shifted, source, 3, spec)
        short = _FakeCache(tmp_path / "xs-cache" / "1bad", "train", horizons, cutoffs[:-1])
        short.root.mkdir()
        with pytest.raises(ValueError, match="rows differ"):
            xs_label_variant.ensure_label_variant(short, source, 3, spec)
    finally:
        source.close()
