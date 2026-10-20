"""Keep unit tests independent from the active RunPod Pod environment."""

import os

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.data.splits import chronological_split
from stock_forecasting.data.windows import build_causal_windows

_PERSISTENT_ENVIRONMENT_KEYS = frozenset(
    {
        "NETWORK_VOLUME_ROOT",
        "PROJECT_ROOT",
        "DATA_ROOT",
        "CACHE_ROOT",
        "MODEL_ROOT",
        "LOG_ROOT",
        "LIFECYCLE_ROOT",
        "SAVED_MODEL_ROOT",
        "CHECKPOINT_ROOT",
        "WANDB_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "POETRY_CACHE_DIR",
        "PIP_CACHE_DIR",
        "KRONOS_ROOT",
        "MODEL_CACHE_MANIFEST",
        "FIN_TS_DATALOADER_WORKERS",
        "FIN_TS_H_START",
        "FIN_TS_FEATURE_MODE",
    }
)
_ISOLATED_ENVIRONMENT_PREFIXES = ("RUNPOD_", "WANDB_", "HF_", "AWS_")


@pytest.fixture(autouse=True)
def isolate_runpod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Pod paths and credentials from leaking into subprocess fixtures."""

    for name in tuple(os.environ):
        if name in _PERSISTENT_ENVIRONMENT_KEYS or name.startswith(_ISOLATED_ENVIRONMENT_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def market_frame() -> pd.DataFrame:
    """Return deterministic assets plus exact US and Taiwan benchmarks."""

    from stock_forecasting.data.sample_universe import market_sessions

    frames: list[pd.DataFrame] = []
    instruments = (
        ("AAPL.US", "stock", "US", "eodhd", 150.0, 0.00045),
        ("VTI.US", "etf", "US", "eodhd", 220.0, 0.00025),
        ("0050.TW", "etf", "TWSE", "twse_official", 120.0, 0.00040),
        ("TAIEX.TW", "index", "TWSE", "twse_official", 15000.0, 0.00022),
    )
    for symbol, asset_type, market, provider, base_price, drift in instruments:
        dates = market_sessions("XNYS" if market == "US" else "XTAI", "2022-01-03", "2023-12-31")[
            :360
        ]
        index = np.arange(len(dates), dtype=np.float64)
        close = base_price * np.exp(drift * index + 0.01 * np.sin(index / 13.0))
        total_return_factor = (
            np.exp(0.00005 * index) if asset_type == "index" else np.ones_like(index)
        )
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "open": close * 0.998,
                    "high": close * 1.006,
                    "low": close * 0.994,
                    "close": close,
                    "volume": 1_000_000.0 + index * 100.0,
                    "adjusted_close": close * total_return_factor,
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
def window_records(market_frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return processed schema 4.0 records with non-empty purged splits."""

    windows = build_causal_windows(
        market_frame,
        window_size=32,
        stride=2,
    )
    return chronological_split(
        windows,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_bars=20,
        embargo_bars=14,
    )


RANK_US_STOCKS = ("AAAA.US", "BBBB.US", "CCCC.US", "DDDD.US", "EEEE.US")
RANK_TW_STOCKS = ("1101.TW", "1102.TW", "1103.TW", "1104.TW", "1216.TW")


def ranked_market_frame() -> pd.DataFrame:
    """Five noisy stocks per market so every cutoff has a strict cross-sectional order."""

    from stock_forecasting.data.sample_universe import market_sessions

    rng = np.random.default_rng(7)
    instruments = [
        ("VTI.US", "etf", "US", "eodhd", 220.0),
        ("TAIEX.TW", "index", "TWSE", "twse_official", 15000.0),
        *((s, "stock", "US", "eodhd", 50.0 + 10 * i) for i, s in enumerate(RANK_US_STOCKS)),
        *(
            (s, "stock", "TWSE", "twse_official", 30.0 + 5 * i)
            for i, s in enumerate(RANK_TW_STOCKS)
        ),
    ]
    frames = []
    for symbol, asset_type, market, provider, base in instruments:
        dates = market_sessions("XNYS" if market == "US" else "XTAI", "2022-01-03", "2023-12-31")[
            :360
        ]
        close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
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
    """A lazily windowed bar store with five ranked stocks per market and date."""

    from pathlib import Path
    from types import SimpleNamespace

    import stock_forecasting.data.rank_labels as rank_labels
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.manifest import artifact_metadata

    # Linux-only memory probes; a fixed budget keeps worker planning deterministic.
    monkeypatch.setattr(
        rank_labels,
        "detect_available_memory",
        lambda: SimpleNamespace(available_bytes=16 * 1024**3),
    )
    frame = ranked_market_frame()
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
    project = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_yaml(project / "configs/local_mock.yaml")
    config.data.bar_store_path, config.data.input_length = store, 32
    config.data.h_start = 1
    config.model.direction.enabled = False
    return config
