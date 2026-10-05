"""Cloud regressions for complete market-session windows and causal liquidity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import artifact_metadata, sha256_file
from stock_forecasting.data.sample_universe import (
    eligible_cutoffs,
    ensure_sample_universe,
    market_sessions,
    split_bar_store,
)
from stock_forecasting.data_policy import evaluation_dataset_id, load_data_policy


@pytest.fixture
def pair():
    sessions = market_sessions("XNYS", "2024-01-02", "2026-05-31")
    price = 100.0 * np.exp(np.arange(len(sessions)) * 0.0001)
    asset = pd.DataFrame(
        {
            "timestamp": sessions,
            "symbol": "AAPL.US",
            "asset_type": "stock",
            "market": "US",
            "currency": "USD",
            "provider": "fixture",
            "dataset_profile": "us_only_eodhd",
            "source_symbol": "AAPL",
            "adjustment_source": "fixture",
            "is_active": True,
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "adjusted_close": price,
            "volume": 100000.0,
            "split_adjusted_volume": 100000.0,
        }
    )
    benchmark = asset.copy()
    benchmark["symbol"], benchmark["source_symbol"], benchmark["asset_type"] = (
        "VTI.US",
        "VTI",
        "etf",
    )
    return asset, benchmark, sessions


def clean(asset, benchmark, sessions, **kwargs):
    return eligible_cutoffs(
        asset,
        benchmark,
        sessions,
        window_size=128,
        max_horizon=14,
        policy=load_data_policy(),
        currency="USD",
        **kwargs,
    )


def at_cutoff(asset, result, timestamp):
    candidates, reasons, _ = result
    return reasons[
        np.flatnonzero(pd.DatetimeIndex(asset.timestamp.iloc[candidates]) == timestamp)[0]
    ]


@pytest.mark.parametrize("offset", [-127, -60, 0, 1, 2, 14])
def test_missing_session_cannot_shift_input_entry_or_exit(pair, offset):
    asset, benchmark, sessions = pair
    cutoff = 200
    if offset == 0:
        # A nonexistent cutoff must not be emitted as a signal date.
        asset = asset.drop(index=cutoff).reset_index(drop=True)
        candidates, _, _ = clean(asset, benchmark, sessions)
        assert sessions[cutoff] not in set(asset.timestamp.iloc[candidates])
    else:
        asset = asset.drop(index=cutoff + offset).reset_index(drop=True)
        reason = at_cutoff(asset, clean(asset, benchmark, sessions), sessions[cutoff])
        assert reason == "asset_market_session_gap"


@pytest.mark.parametrize("offset", [-127, 0, 1, 7, 14])
def test_zero_volume_is_not_an_executable_continuous_window(pair, offset):
    asset, benchmark, sessions = pair
    asset.loc[200 + offset, "volume"] = 0
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions), sessions[200])
        == "invalid_or_zero_volume_asset"
    )


def test_missing_day_in_both_streams_is_not_mistaken_for_market_holiday(pair):
    asset, benchmark, sessions = pair
    asset = asset.drop(index=201).reset_index(drop=True)
    benchmark = benchmark.drop(index=201).reset_index(drop=True)
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions), sessions[200])
        == "asset_market_session_gap"
    )


def test_benchmark_gap_and_index_volume_semantics(pair):
    asset, benchmark, sessions = pair
    missing = benchmark.drop(index=205).reset_index(drop=True)
    assert (
        at_cutoff(asset, clean(asset, missing, sessions), sessions[200])
        == "benchmark_market_session_gap"
    )
    benchmark["volume"] = 0
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions), sessions[200])
        == "invalid_benchmark_bar"
    )
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions, benchmark_is_index=True), sessions[200])
        == ""
    )


def test_large_real_returns_remain_and_liquidity_does_not_read_future(pair):
    asset, benchmark, sessions = pair
    for column in ("open", "high", "low", "close", "adjusted_close"):
        asset.loc[202:, column] *= 2
    candidates, reasons, extremes = clean(asset, benchmark, sessions)
    position = int(np.flatnonzero(candidates == 200)[0])
    assert reasons[position] == "" and extremes[position]
    # Future turnover must not rescue a low-liquidity decision-date history.
    asset.loc[:200, "volume"] = 100.0
    asset.loc[201:, "volume"] = 1e12
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions), sessions[200])
        == "insufficient_trailing_liquidity"
    )
    asset.loc[201:, "volume"] = 1.0
    assert (
        at_cutoff(asset, clean(asset, benchmark, sessions), sessions[200])
        == "insufficient_trailing_liquidity"
    )


def test_off_calendar_rows_do_not_count_as_market_sessions(pair):
    asset, benchmark, sessions = pair
    asset.loc[200, "timestamp"] = sessions[200] + pd.Timedelta(hours=1)
    candidates, reasons, _ = clean(asset, benchmark, sessions)
    assert reasons[np.flatnonzero(candidates == 200)[0]] == "off_calendar_asset_bar"
    assert pd.Timestamp("2025-01-09", tz="UTC") not in sessions
    assert pd.Timestamp("2025-01-01", tz="UTC") not in sessions


def test_sparse_multiyear_history_cannot_satisfy_a_128_session_context(pair):
    asset, benchmark, sessions = pair
    asset = asset.iloc[::3].reset_index(drop=True)
    _, reasons, _ = clean(asset, benchmark, sessions)
    assert len(reasons) > 0 and np.all(reasons != "")


def test_taiwan_official_exceptional_sessions_are_not_asset_gaps():
    sessions = market_sessions("XTAI", "2016-01-01", "2026-06-01")
    changes = load_data_policy()["calendar_overrides"]["XTAI"]
    assert all(pd.Timestamp(day, tz="UTC") in sessions for day in changes["open"])
    assert all(pd.Timestamp(day, tz="UTC") not in sessions for day in changes["closed"])
    assert pd.Timestamp("2025-02-08", tz="UTC") not in sessions


def test_runtime_index_reuses_bars_and_keeps_full_deterministic_splits(tmp_path, pair):
    from stock_forecasting.data.bar_store import build_symbol_bar_store

    asset, benchmark, sessions = pair
    # Add a large economic jump which the prepared legacy candidate mask rejects.
    for column in ("open", "high", "low", "close", "adjusted_close"):
        asset.loc[210:, column] *= 2
    asset.loc[400, "volume"] = 0
    raw = tmp_path / "market.parquet"
    frame = pd.concat([asset, benchmark], ignore_index=True)
    frame.to_parquet(raw, index=False)
    root = tmp_path / "bar-store"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=root,
        window_size=128,
        bucket_count=2,
        batch_rows=1024,
        fixed_split={
            "train_end": "2025-06-01",
            "validation_end": "2025-12-01",
            "test_end": "2026-06-01",
        },
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(frame))}
        },
    )
    immutable = {str(path): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    universe = ensure_sample_universe(root, workers=2)
    assert ensure_sample_universe(root, workers=2) == universe
    payload = json.loads((universe / "universe.json").read_text())
    assert payload["audit"]["train"]["retained_extreme_windows"] > 0
    assert not list(universe.rglob("*targets*")) and not list(universe.rglob("*windows*"))
    for name, digest in immutable.items():
        assert sha256_file(Path(name)) == digest
    for split, start, end in (
        ("train", "2024-01-01", "2025-06-01"),
        ("validation", "2025-06-01", "2025-12-01"),
        ("test", "2025-12-01", "2026-06-01"),
    ):
        dataset = LazyFinancialWindowDataset(
            root, split=split, window_size=128, h_start=1, sample_index_root=universe
        )
        ids = []
        for ordinal in range(len(dataset)):
            record = dataset.record_at(ordinal)
            cutoff = pd.Timestamp(record["cutoff_at"])
            position = sessions.get_loc(cutoff)
            assert pd.DatetimeIndex(record["context"]["timestamp"]).equals(
                sessions[position - 127 : position + 1]
            )
            assert pd.Timestamp(record["label"]["entry_at"]) == sessions[position + 1]
            assert pd.Timestamp(record["label"]["end_at"]["14d"]) == sessions[position + 14]
            assert pd.Timestamp(start, tz="UTC") <= cutoff < pd.Timestamp(end, tz="UTC")
            assert sessions[position + 14] < pd.Timestamp(end, tz="UTC")
            ids.append(record["sample_id"])
        assert len(ids) == len(set(ids)) == payload["split_counts"][split]
        assert ids == [dataset.record_at(i)["sample_id"] for i in range(len(dataset))]


def test_a_b_use_identical_evaluation_source_not_an_intersection(tmp_path, monkeypatch):
    import argparse
    import runpy
    from types import SimpleNamespace

    project = Path(__file__).resolve().parents[1]
    control = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    selected_roots = []
    monkeypatch.setenv("NETWORK_VOLUME_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "stock_forecasting.training_paths.resolve_bar_store_path", lambda path: Path(path)
    )
    for start in ("2021-01-01", "2016-01-01"):
        args = argparse.Namespace(
            stage="stage2",
            data_profile="us_tw_eodhd",
            dataset_revision="v1",
            start=start,
            end="2026-06-01",
            h_start=1,
            universe="all",
            stocks=[],
            etfs=[],
            symbol_limit=None,
            feature_mode="combined",
        )
        selection = control["_build_selection"](args, project)
        common = (
            tmp_path
            / "datasets"
            / evaluation_dataset_id(selection, load_data_policy())
            / "prepared/bar-store"
        )
        common.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "selection.json"
        path.write_text(json.dumps(selection))
        monkeypatch.setenv("RUNPOD_SELECTION_FILE", str(path))
        train = tmp_path / "datasets" / selection["dataset_request_sha256"] / "prepared/bar-store"
        config = SimpleNamespace(data=SimpleNamespace(bar_store_path=train))
        assert split_bar_store(config, "train") == train
        assert split_bar_store(config, "validation") == split_bar_store(config, "test") == common
        selected_roots.append(common)
    assert selected_roots[0] == selected_roots[1]


def test_production_evaluation_never_silently_uses_train_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("RUNPOD_SELECTION_FILE", raising=False)
    monkeypatch.delenv("RUNPOD_REMOTE_SELECTION_PATH", raising=False)
    config = SimpleNamespace(
        data=SimpleNamespace(bar_store_path=tmp_path),
        validation=SimpleNamespace(require_prebuilt_baselines=True),
    )
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="shared source"):
            split_bar_store(config, split)
