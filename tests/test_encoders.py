"""Frozen-encoder contract: registry, embedding cache round trip and the cross-stock hook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from stock_forecasting.cross_stock_training import EXPERIMENTS, make_config, table_from_caches
from stock_forecasting.encoders import (
    ENCODERS,
    embeddings_root,
    ensure_embeddings,
    open_embeddings,
)


class _MeanEncoder(nn.Module):
    """A stand-in adapter honouring the (B, L, F) -> (B, D) contract."""

    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.calls = 0

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        x = windows.float()
        prices = torch.log(x[:, :, :4].clamp_min(1e-8) / x[:, -1:, 3:4].clamp_min(1e-8))
        return torch.cat([prices.mean(dim=1), prices.std(dim=1)[:, :2]], dim=1)[:, : self.dim]


class _Cache:
    """A minimal XsCache stand-in: attributes plus ``__len__``."""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)

    def __len__(self) -> int:
        return len(self.cutoff_ns)


def _cache(tmp_path: Path, samples: int = 70, window: int = 16) -> _Cache:
    rng = np.random.default_rng(0)
    root = tmp_path / "xs-cache"
    root.mkdir()
    windows = (rng.uniform(50, 150, (samples, window, 5))).astype(np.float32)
    windows[:, :, 4] = rng.uniform(1e5, 1e9, (samples, window))
    return _Cache(
        root=root,
        split="train",
        windows=windows,
        features=rng.standard_normal((samples, 3)).astype(np.float32),
        ranks=rng.uniform(0, 1, (samples, 2)).astype(np.float32),
        alpha=rng.standard_normal((samples, 2)).astype(np.float32),
        cutoff_ns=np.arange(samples, dtype=np.int64) * 86_400_000_000_000,
        market_codes=np.zeros(samples, dtype=np.int8),
        symbol_codes=(np.arange(samples) % 7).astype(np.int32),
        markets=["TWSE"],
        symbols=[f"{1101 + i}.TW" for i in range(7)],
        horizons=[1, 5],
        feature_columns=["f0", "f1", "f2"],
    )


def test_registry_names_match_experiments_and_contract_fields() -> None:
    for experiment, overrides in EXPERIMENTS.items():
        encoder = overrides.get("encoder")
        if encoder is not None:
            assert encoder in ENCODERS, experiment
    for spec in ENCODERS.values():
        assert spec.family in {"kronos", "chronos2"} and spec.repositories and spec.pooling


def test_embedding_cache_round_trip_and_reuse(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    encoder = _MeanEncoder()
    root = ensure_embeddings(
        cache,
        "kronos-small",
        encoder=encoder,
        device=torch.device("cpu"),
        batch_size=32,
        revisions={"NeoQuasar/Kronos-small": "abc"},
    )
    assert root == embeddings_root(cache.root, "kronos-small")
    assert encoder.calls == 3  # 70 samples in batches of 32
    embeddings = open_embeddings(root)
    assert embeddings.shape == (70, 6) and np.isfinite(embeddings).all()
    payload = json.loads((root / "embeddings.json").read_text())
    assert payload["revisions"] == {"NeoQuasar/Kronos-small": "abc"} and payload["dim"] == 6
    index = pd.read_parquet(root / "embeddings-index.parquet")
    assert list(index["ordinal"]) == list(range(70)) and index["symbol"].iloc[8] == "1102.TW"
    # A ready store is reused without touching the encoder again.
    ensure_embeddings(cache, "kronos-small", encoder=encoder, device=torch.device("cpu"))
    assert encoder.calls == 3
    with pytest.raises(ValueError):
        ensure_embeddings(
            cache,
            "chronos2-small",
            encoder=lambda windows: torch.full((len(windows), 2), float("nan")),
            device=torch.device("cpu"),
        )


def test_cross_stock_config_requires_matching_embeddings(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    table = table_from_caches({"train": cache})
    with pytest.raises(ValueError, match="needs cached"):
        make_config("t-xs-kronos-small", table)
    table.encoded = torch.zeros(len(table), 6)
    config, _ = make_config("t-xs-kronos-small", table)
    assert config.encoder_dim == 6
    with pytest.raises(ValueError, match="trains its own encoder"):
        make_config("t-xs-scratch", table)
