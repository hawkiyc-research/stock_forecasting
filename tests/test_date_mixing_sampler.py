"""Cloud numerical tests for date-mixed training order and its controlled presets."""

import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import BlockwisePermutationSampler, FixedSizeBatchSampler
from stock_forecasting.date_market_sampler import (
    INDEX_DTYPE,
    MARKET_CODE_RADIX,
    DateMarketSampler,
    batch_composition_summary,
)
from stock_forecasting.training import ResumableFixedSizeBatchSampler, _training_sampler

ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    (group, variant)
    for group in ("a", "b")
    for variant in ("alpha_point64", "absolute_direction64")
]


def _index(tmp_path):
    """Three years x 40 dates x 2 markets x 32 stocks, sorted like the runtime index.

    Each date/market spans exactly two 16-sample blocks, as production US/TW
    groups span several 128-sample blocks.
    """
    groups = []
    for year in (2021, 2022, 2023):
        start = int(np.datetime64(f"{year}-01-04", "D").astype(int))
        for day in range(40):
            for market in (0, 1):
                groups.extend([(start + day) * MARKET_CODE_RADIX + market] * 32)
    index = np.zeros(len(groups), dtype=INDEX_DTYPE)
    index["group"] = groups
    # Real ordinals are symbol-major, so date order never coincides with ordinal order.
    index["ordinal"] = np.random.default_rng(3).permutation(len(groups))
    index.sort(order=["group", "ordinal"])
    path = tmp_path / "index.npy"
    np.save(path, index)
    return path, index


def _sampler(path, size, **kwargs):
    with patch("stock_forecasting.date_market_sampler.date_market_index", return_value=path):
        return DateMarketSampler(
            range(size), requested_workers=1, yearly_decay=0.8, seed=7, block_size=16, **kwargs
        )


def _historical_order(index, sampler):
    """Frozen copy of the pre-mixing annual algorithm, used only as a reference."""
    strata = sampler.year_quotas(index)
    rng = random.Random(sampler.seed + sampler.epoch * 1_000_003)
    remaining = [row["samples"] for row in strata]

    def stream(row):
        cycle = 0
        while True:
            blocks = BlockwisePermutationSampler(
                row["count"],
                seed=sampler.seed + row["year"] * 19 + cycle * 99991,
                block_size=sampler.block_size,
            )
            blocks.set_epoch(sampler.epoch)
            for position in blocks:
                yield int(index["ordinal"][row["start"] + position])
            cycle += 1

    streams = [stream(row) for row in strata]
    order = []
    while sum(remaining):
        draw = rng.randrange(sum(remaining))
        chosen = 0
        while draw >= remaining[chosen]:
            draw -= remaining[chosen]
            chosen += 1
        take = min(sampler.block_size, remaining[chosen])
        order.extend(next(streams[chosen]) for _ in range(take))
        remaining[chosen] -= take
    return order


def test_unmixed_order_is_identical_to_the_historical_algorithm(tmp_path):
    path, index = _index(tmp_path)
    sampler = _sampler(path, len(index))
    assert sampler.shuffle_buffer is None
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        assert list(sampler) == _historical_order(index, sampler)


def test_mixing_preserves_each_epoch_sample_multiset_and_is_reproducible(tmp_path):
    path, index = _index(tmp_path)
    historical = _sampler(path, len(index))
    mixed = _sampler(path, len(index), shuffle_buffer=512)
    orders = []
    for epoch in (0, 1):
        historical.set_epoch(epoch)
        mixed.set_epoch(epoch)
        expected, actual = list(historical), list(mixed)
        # Identical samples, hence identical annual quotas, in a different order.
        assert sorted(actual) == sorted(expected) and actual != expected
        rebuilt = _sampler(path, len(index), shuffle_buffer=512)
        rebuilt.set_epoch(epoch)
        assert list(rebuilt) == actual
        orders.append(actual)
    assert orders[0] != orders[1]


def test_mixed_resume_skips_to_exactly_the_same_batches(tmp_path):
    path, index = _index(tmp_path)
    batches = ResumableFixedSizeBatchSampler(
        FixedSizeBatchSampler(_sampler(path, len(index), shuffle_buffer=512), batch_size=32)
    )
    batches.set_epoch(1)
    complete = list(batches)
    restarted = ResumableFixedSizeBatchSampler(
        FixedSizeBatchSampler(_sampler(path, len(index), shuffle_buffer=512), batch_size=32)
    )
    restarted.set_epoch(1, start_batch_index=5)
    assert list(restarted) == complete[5:]


def test_batch_composition_describes_the_emitted_training_order(tmp_path):
    path, index = _index(tmp_path)
    group_of = {int(ordinal): int(group) for group, ordinal in index.tolist()}
    mixed = _sampler(path, len(index), shuffle_buffer=512)
    composition = mixed.batch_composition(16, 2, max_optimizer_steps=100)
    emitted = [group_of[ordinal] for ordinal in mixed][: composition["evaluated_samples"]]
    assert composition["evaluated_samples"] == 3200
    assert composition["configured_order"] == {
        "micro_batch": batch_composition_summary(emitted, 16),
        "optimizer_step": batch_composition_summary(emitted, 32),
    }
    reference = composition["unmixed_reference_order"]["optimizer_step"]
    configured = composition["configured_order"]["optimizer_step"]
    # Historical steps hold two aligned same-date blocks; mixed steps span many dates.
    assert reference["distinct_date_markets"]["max"] <= 2
    assert configured["distinct_date_markets"]["median"] > 10
    unmixed = _sampler(path, len(index)).batch_composition(16, 2, max_optimizer_steps=100)
    assert "unmixed_reference_order" not in unmixed
    assert unmixed["configured_order"] == composition["unmixed_reference_order"]


@pytest.mark.parametrize("value", [1, 0, True, 2.5, "512"])
def test_invalid_shuffle_buffer_is_rejected(tmp_path, value):
    path, index = _index(tmp_path)
    with pytest.raises(ValueError, match="shuffle_buffer"):
        _sampler(path, len(index), shuffle_buffer=value)


@pytest.mark.parametrize(("group", "variant"), PRESETS)
def test_training_sampler_uses_only_the_configured_mixing_buffer(tmp_path, group, variant):
    path, index = _index(tmp_path)
    for suffix, expected in (("", None), ("_datemix", 16384)):
        config = ExperimentConfig.from_yaml(
            ROOT / f"configs/experiments/{group}_{variant}{suffix}.yaml"
        )
        with patch("stock_forecasting.date_market_sampler.date_market_index", return_value=path):
            sampler = _training_sampler(
                config, range(len(index)), SimpleNamespace(effective_workers=1)
            )
        assert isinstance(sampler, DateMarketSampler)
        assert sampler.shuffle_buffer == expected


@pytest.mark.parametrize(("group", "variant"), PRESETS)
def test_datemix_presets_change_only_the_training_order(group, variant):
    base = ExperimentConfig.from_yaml(ROOT / f"configs/experiments/{group}_{variant}.yaml")
    mixed = ExperimentConfig.from_yaml(
        ROOT / f"configs/experiments/{group}_{variant}_datemix.yaml"
    )
    assert base.training.date_mixing_buffer_samples is None
    assert mixed.training.date_mixing_buffer_samples == 16384
    assert base.model_architecture_digest() == mixed.model_architecture_digest()
    payloads = []
    for config in (base, mixed):
        payload = config.as_dict()
        for key in ("experiment_name", "description"):
            payload.pop(key)
        payload["wandb"].pop("name")
        payload["wandb"].pop("tags")
        payload["training"].pop("date_mixing_buffer_samples")
        payloads.append(payload)
    assert payloads[0] == payloads[1]


def test_mixing_requires_the_annual_sampler_and_no_same_date_ranking(tmp_path):
    source = (ROOT / "configs/experiments/a_alpha_point64_datemix.yaml").read_text()
    without_annual = tmp_path / "without_annual.yaml"
    without_annual.write_text(source.replace("  yearly_sampling_decay: 0.8\n", ""))
    with pytest.raises(ValueError, match="requires yearly_sampling_decay"):
        ExperimentConfig.from_yaml(without_annual)
    ranking = (ROOT / "configs/experiments/a_lora64.yaml").read_text()
    assert "  ranking_loss_weight: 0.05\n" in ranking
    ranked = tmp_path / "ranked.yaml"
    ranked.write_text(
        ranking.replace(
            "  yearly_sampling_decay: 0.8\n",
            "  yearly_sampling_decay: 0.8\n  date_mixing_buffer_samples: 16384\n",
        )
    )
    with pytest.raises(ValueError, match="same-date pairs"):
        ExperimentConfig.from_yaml(ranked)
