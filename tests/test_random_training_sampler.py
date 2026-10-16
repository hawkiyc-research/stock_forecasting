"""Cloud integration contracts for lazy training, fixed batches and exact resume."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import FixedSizeBatchSampler
from stock_forecasting.random_training_sampler import RandomTrainingSampler
from stock_forecasting.training import ResumableFixedSizeBatchSampler, _training_sampler

ROOT = Path(__file__).resolve().parents[1]


class OrdinalsOnly:
    def __len__(self):
        return 3000


def test_annual_index_maps_random_positions_back_to_original_ordinals(tmp_path, monkeypatch):
    from stock_forecasting.date_market_sampler import INDEX_DTYPE

    index = np.empty(3000, dtype=INDEX_DTYPE)
    origin = date(1970, 1, 1)
    for i, year in enumerate((2023, 2024, 2025)):
        index["group"][i * 1000:(i + 1) * 1000] = (date(year, 1, 1) - origin).days * 256
    index["ordinal"] = np.arange(2999, -1, -1)
    path = tmp_path / "date-market.npy"
    np.save(path, index)
    monkeypatch.setattr("stock_forecasting.random_training_sampler.date_market_index",
                        lambda *a, **kw: path)
    sampler = RandomTrainingSampler(
        OrdinalsOnly(), requested_workers=2, scratch_root=tmp_path / "scratch",
        max_scratch_bytes=1024**2, yearly_decay=0.8, seed=42,
    )
    values = list(sampler)
    assert len(values) == 3000
    assert sum(v >= 2000 for v in values) == 787
    assert sum(1000 <= v < 2000 for v in values) == 984
    assert sum(v < 1000 for v in values) == 1229
    batches = FixedSizeBatchSampler(sampler, batch_size=32)
    resumable = ResumableFixedSizeBatchSampler(batches)
    resumable.set_epoch(2)
    expected = list(resumable)
    resumable.set_epoch(2, start_batch_index=13)
    assert list(resumable) == expected[13:]
    assert batches.padded_sample_count == 8
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("group", ("a", "b"))
@pytest.mark.parametrize("task", ("alpha_point", "absolute_direction"))
@pytest.mark.parametrize("backend", ("random", "chronos2"))
def test_all_new_presets_select_true_random_sampling(group, task, backend, tmp_path, monkeypatch):
    path = ROOT / f"configs/experiments/{group}_{backend}_{task}64.yaml"
    config = ExperimentConfig.from_yaml(path)
    config.training.yearly_sampling_decay = None
    config.runtime.log_root = tmp_path
    sampler = _training_sampler(config, OrdinalsOnly(), SimpleNamespace(effective_workers=2))
    assert isinstance(sampler, RandomTrainingSampler)
    assert sampler.sampling_contract["ordering"] == "global_sample_fisher_yates"
    values = list(sampler)
    assert len(values) == len(sampler)
    assert len(set(values)) == len(values)
    payload = config.as_dict()
    payload["model"]["ranking_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="ranking"):
        ExperimentConfig.model_validate(payload)


def test_real_bar_store_stays_lazy_with_spawn_workers_and_resume(tmp_path, market_frame):
    import json

    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.dataset import FinancialBatchCollator, LazyFinancialWindowDataset
    from stock_forecasting.data.manifest import artifact_metadata

    raw, store = tmp_path / "market.parquet", tmp_path / "bar-store"
    market_frame.to_parquet(raw, index=False)
    build_symbol_bar_store(
        raw_path=raw, output_root=store, window_size=32, bucket_count=2,
        batch_rows=512, purge_bars=20, embargo_bars=14, workers=2,
        download_manifest={"artifacts": {"raw": artifact_metadata(
            raw, root=tmp_path, row_count=len(market_frame)
        )}},
    )
    manifest_before = (store / "bar-store.json").read_bytes()
    manifest = json.loads(manifest_before)
    assert manifest["execution"]["window_materialized"] is False
    assert manifest["execution"]["labels_materialized"] is False
    source = LazyFinancialWindowDataset(store, split="train", window_size=32, h_start=1,
                                        symbol_cache_size=2)
    sampler = RandomTrainingSampler(
        source, requested_workers=2, scratch_root=tmp_path / "scratch",
        max_scratch_bytes=1024**2, yearly_decay=0.8, seed=42,
    )
    batches = ResumableFixedSizeBatchSampler(FixedSizeBatchSampler(sampler, batch_size=8))
    batches.set_epoch(1)
    expected_ids = [
        [sample["sample_id"] for sample in source.__getitems__(indices)] for indices in batches
    ]
    loader = DataLoader(
        source, batch_sampler=batches, collate_fn=FinancialBatchCollator(), num_workers=2,
        multiprocessing_context="spawn", prefetch_factor=2, persistent_workers=False,
    )
    batches.set_epoch(1, start_batch_index=3)
    actual_ids = []
    for batch in loader:
        assert batch["asset_series"].shape[:2] == (8, 32)
        actual_ids.append(batch["sample_ids"])
    assert actual_ids == expected_ids[3:]
    assert (store / "bar-store.json").read_bytes() == manifest_before
    assert not list((tmp_path / "scratch").iterdir())
