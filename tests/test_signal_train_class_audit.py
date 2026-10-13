"""Opt-in full training-only class audit over existing cleaned data.

Run this in the authorized cloud runtime with SIGNAL_QA_AUDIT_DATA_ROOT and
SIGNAL_QA_AUDIT_INDEX. Optionally provide SIGNAL_QA_AUDIT_BASELINE_INPUTS to use
already completed absolute-return arrays, after matching the data snapshot and
cleaned train sample universe. Neither path rebuilds caches or materializes
windows; no validation/test data are consulted to choose a threshold.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from stock_forecasting import training
from stock_forecasting.baseline_signal import append_label_audit
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import sha256_file
from stock_forecasting.direction_data import DirectionDataset

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not os.environ.get("SIGNAL_QA_AUDIT_DATA_ROOT"),
    reason="Requires explicit existing training-only cloud audit paths",
)


def _validated_existing_cache(source):
    value = os.environ.get("SIGNAL_QA_AUDIT_BASELINE_INPUTS")
    if not value:
        return None
    root = Path(value).resolve(strict=True)
    complete = json.loads((root / "train/complete.json").read_text())
    identity = complete["identity"]
    expected_source = {
        "manifest_sha256": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "samples": len(source),
    }
    assert identity["split_sources"]["train"] == expected_source, (
        "Cached labels do not belong to this exact cleaned training population"
    )
    assert identity["horizons"] == list(source.horizons)
    assert complete["count"] == len(source)
    path = root / "train/absolute_targets.npy"
    values = np.load(path, mmap_mode="r")
    try:
        assert values.shape == (len(source), len(source.horizons))
        assert values.dtype == np.dtype("float32")
    finally:
        values._mmap.close()
    return path


class AbsoluteTrainAuditDataset(Dataset):
    """Read only cumulative absolute labels, never overlapping OHLCV windows."""

    def __init__(self, source, cache_path=None):
        if source.split != "train":
            raise ValueError("Threshold audits must not use validation or test data")
        self.source, self.cache_path, self._absolute = source, cache_path, None
        self.range_markets = np.asarray(
            [str(source._index[symbol]["market"]) for symbol in source._range_symbols]
        )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        ordinals = np.asarray(indices, dtype=np.int64)
        if (
            ordinals.ndim != 1
            or not len(ordinals)
            or (ordinals < 0).any()
            or (ordinals >= len(self)).any()
        ):
            raise ValueError("Audit indexes must be a nonempty bounded vector")
        ranges = np.searchsorted(self.source._range_ends, ordinals, side="right")
        if self.cache_path is not None:
            if self._absolute is None:
                self._absolute = np.load(self.cache_path, mmap_mode="r")
            targets = np.asarray(self._absolute[ordinals], dtype=np.float32)
        else:
            cutoffs = (
                self.source._range_starts[ranges] + ordinals - self.source._range_begins[ranges]
            )
            symbols = self.source._range_symbols[ranges]
            targets = np.empty((len(ordinals), len(self.source.horizons)), dtype=np.float32)
            for symbol in np.unique(symbols):
                selected = np.flatnonzero(symbols == symbol)
                bars = self.source._numeric_symbol(symbol)
                entry_index = cutoffs[selected] + 1
                entry = bars["raw"][entry_index, 0] * bars["price_factor"][entry_index]
                exits = cutoffs[selected, None] + np.asarray(self.source.horizons)[None]
                gross = bars["adjusted_close"][exits] / entry[:, None]
                if not np.isfinite(gross).all() or (gross <= 0).any():
                    raise ValueError("Absolute execution ratios must be finite and positive")
                targets[selected] = np.log(gross).astype(np.float32)
        if not np.isfinite(targets).all():
            raise ValueError("Audit absolute targets must be finite")
        return {
            "targets": targets,
            "markets": self.range_markets[ranges],
            "first": int(ordinals[0]),
            "last": int(ordinals[-1]),
        }

    def close(self):
        if self._absolute is not None:
            self._absolute._mmap.close()
            self._absolute = None
        self.source.close()


def _identity_batch(batch):
    return batch


def test_full_existing_training_class_histograms_without_costs(tmp_path):
    root = Path(os.environ["SIGNAL_QA_AUDIT_DATA_ROOT"]).resolve(strict=True)
    index = Path(os.environ["SIGNAL_QA_AUDIT_INDEX"]).resolve(strict=True)
    config = ExperimentConfig.from_yaml(ROOT / "configs/experiments/a_absolute_direction64.yaml")
    source = LazyFinancialWindowDataset(
        root / "prepared/bar-store",
        split="train",
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        sample_index_root=index,
        symbol_cache_size=4,
    )
    assert source.sample_universe_identity is not None
    cache_path = _validated_existing_cache(source)
    dataset = AbsoluteTrainAuditDataset(source, cache_path)
    iterator = None
    protected = (
        source.root / "bar-store.json",
        index / "universe.json",
        index / "cutoff-ranges.parquet",
    )
    before = {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in protected}
    try:
        # Check the label-only fast path against the actual production dataset.
        positions = [i * (len(source) - 1) // 15 for i in range(16)]
        actual = dataset.__getitems__(positions)["targets"]
        expected = np.stack(
            [
                row["target_absolute"].numpy()
                for row in DirectionDataset(source).__getitems__(positions)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        source.close()
        # Closing parent caches before spawn avoids serialized decoded bar arrays.
        if dataset._absolute is not None:
            dataset._absolute._mmap.close()
            dataset._absolute = None
        requested = int(os.environ.get("SIGNAL_QA_AUDIT_WORKERS", "2"))
        batch_size = int(os.environ.get("SIGNAL_QA_AUDIT_BATCH_SIZE", "8192"))
        if requested < 1 or not 1 <= batch_size <= 65536:
            raise ValueError("Audit requires positive workers and a bounded batch size")
        workers = training.plan_dataloader_workers(requested)
        assert workers.effective_workers > 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_identity_batch,
            **training._loader_process_options(workers, persistent=False),
        )
        audit, seen = {}, 0
        started = last_log = time.monotonic()
        iterator = iter(loader)
        for batch in iterator:
            size = len(batch["targets"])
            assert batch["first"] == seen and batch["last"] == seen + size - 1
            append_label_audit(
                audit,
                batch["targets"],
                batch["markets"],
                config.model.neutral_return_threshold,
                source.horizons,
            )
            seen += size
            now = time.monotonic()
            if now - last_log >= 30:
                print(
                    json.dumps(
                        {
                            "audit_split": "train",
                            "samples": seen,
                            "total": len(source),
                            "samples_per_second": seen / (now - started),
                        }
                    ),
                    flush=True,
                )
                last_log = now
        assert seen == len(source)
        assert all(sum(counts) == seen for counts in audit["overall"].values())
        for horizon in map(str, source.horizons):
            by_market = np.sum(
                [counts[horizon] for market, counts in audit.items() if market != "overall"], axis=0
            )
            np.testing.assert_array_equal(by_market, audit["overall"][horizon])
        assert before == {
            str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in protected
        }
        record = {
            "prediction_task": "absolute_direction",
            "split": "train",
            "threshold_simple_return": config.model.neutral_return_threshold,
            "class_order": ["down", "neutral", "up"],
            "samples": seen,
            "horizons": list(source.horizons),
            "class_counts": audit,
            "class_shares": {
                market: {
                    horizon: (np.asarray(values) / sum(values)).tolist()
                    for horizon, values in counts.items()
                }
                for market, counts in audit.items()
            },
            "source_manifest_sha256": sha256_file(source.root / "bar-store.json"),
            "sample_universe": source.sample_universe_identity,
            "source_root": str(source.root),
            "cached_absolute_targets": str(cache_path) if cache_path else None,
            "threshold_selected_using_test": False,
            "workers": workers.effective_workers,
            "batch_size": batch_size,
            "seconds": time.monotonic() - started,
        }
        (tmp_path / "full-train-class-audit.json").write_text(
            json.dumps(record, indent=2, allow_nan=False)
        )
        print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        if iterator is not None:
            iterator._shutdown_workers()
        dataset.close()
