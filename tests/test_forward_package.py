"""The forward package must reproduce validation/test inputs from independently fetched bars."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import artifact_metadata
from stock_forecasting.data.sample_universe import ensure_sample_universe, market_sessions
from stock_forecasting.forward.compare import compare_inputs
from stock_forecasting.forward.labels import population_labels, write_population_labels
from stock_forecasting.forward.package import (
    PACKAGE_FILE,
    build_package,
    load_package,
    open_population,
)

END = "2026-08-01"
# A dividend inside the overlap windows, and one after 2026-06-01 that only the
# forward fetch sees: it rescales every earlier adjusted price.
EX_DATES = {"2026-01-15": 0.97, "2026-07-15": 0.98}
# An unscheduled closure that the pinned calendar and the project's cleaning policy both
# still list as a session (2026-07-10 is now a closure in configs/data_cleaning.json, so it
# no longer serves as the "unknown closure" of this fixture).
CLOSED = "2026-07-17"
ADDITIONS = {"XTAI": {"open": [], "closed": [CLOSED]}}


def _bars(sessions, symbol, asset_type, market, *, scale, phase, last_day):
    sessions = sessions[sessions < pd.Timestamp(last_day, tz="UTC")]
    sessions = sessions[sessions != pd.Timestamp(CLOSED, tz="UTC")]
    steps = np.arange(len(sessions), dtype=np.float64)
    close = scale * np.exp(0.0005 * steps + 0.02 * np.sin(steps / 7 + phase))
    opening = close * (1 + 0.002 * np.cos(steps + phase))
    factor = np.ones(len(sessions))
    for day, ratio in EX_DATES.items():
        if day < last_day:
            factor[sessions < pd.Timestamp(day, tz="UTC")] *= ratio
    volume = 1e5 + 1000 * (steps % 10) if asset_type != "index" else 1e9 + steps
    return pd.DataFrame(
        {
            "timestamp": sessions,
            "symbol": symbol,
            "asset_type": asset_type,
            "market": market,
            "currency": "TWD",
            "provider": "twse_official" if market == "TWSE" else "tpex_official",
            "dataset_profile": "tw_only",
            "source_symbol": symbol.split(".")[0],
            "adjustment_source": "fixture",
            "is_active": True,
            "open": opening,
            "high": np.maximum(opening, close) * 1.01,
            "low": np.minimum(opening, close) * 0.99,
            "close": close,
            "adjusted_close": close * factor,
            "volume": volume,
            "split_adjusted_volume": volume,
        }
    )


def _raw(path, last_day):
    sessions = market_sessions("XTAI", "2024-06-01", "2026-08-31")
    frame = pd.concat(
        [
            _bars(sessions, "2330.TW", "stock", "TWSE", scale=600, phase=0.0, last_day=last_day),
            _bars(sessions, "6488.TWO", "stock", "TPEX", scale=400, phase=1.0, last_day=last_day),
            _bars(sessions, "TAIEX.TW", "index", "TWSE", scale=20000, phase=2.0, last_day=last_day),
            _bars(sessions, "TPEX.TWO", "index", "TPEX", scale=250, phase=3.0, last_day=last_day),
        ],
        ignore_index=True,
    )
    frame.to_parquet(path, index=False)
    return path, sessions


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("forward")
    reference_raw, _ = _raw(root / "reference.parquet", "2026-06-01")
    reference = root / "reference" / "bar-store"
    try:
        build_symbol_bar_store(
            raw_path=reference_raw,
            output_root=reference,
            download_manifest={
                "artifacts": {
                    "raw": artifact_metadata(
                        reference_raw,
                        root=root,
                        row_count=len(pd.read_parquet(reference_raw)),
                    )
                }
            },
            window_size=128,
            fixed_split={
                "train_end": "2025-06-01",
                "validation_end": "2025-12-01",
                "test_end": "2026-06-01",
            },
        )
        universe = ensure_sample_universe(reference, workers=1)
        forward_raw, sessions = _raw(root / "forward.parquet", END)
        payload = build_package(
            forward_raw, root / "package", end_exclusive=END, calendar_additions=ADDITIONS
        )
        unadjusted = build_package(forward_raw, root / "unadjusted", end_exclusive=END)
    except PermissionError as error:
        pytest.skip(f"process pools are not permitted here: {error}")
    test = LazyFinancialWindowDataset(
        reference, split="test", window_size=128, h_start=1, sample_index_root=universe
    )
    return {"test": test, "package": root / "package", "payload": payload,
            "unadjusted": unadjusted, "unadjusted_root": root / "unadjusted",
            "sessions": sessions}


def test_forward_builder_copy_is_pinned_to_the_frozen_builder():
    import ast
    import hashlib
    import inspect

    from stock_forecasting.data import sample_universe
    from stock_forecasting.forward import universe

    # The forward copy must be reviewed whenever the frozen builder changes.
    source = inspect.getsource(sample_universe)
    node = next(n for n in ast.parse(source).body
                if getattr(n, "name", "") == "ensure_sample_universe")
    digest = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    assert digest == universe.BASE_BUILDER_SHA256


def test_forward_builder_without_additions_is_the_frozen_builder(stores):
    # Same identity, sessions and ranges: the frozen builder reuses the index as is.
    store = stores["unadjusted_root"] / "bar-store"
    built = stores["unadjusted_root"] / stores["unadjusted"]["sample_universe"]["relative_path"]
    assert ensure_sample_universe(store, workers=1) == built


def test_overlap_inputs_and_labels_match_the_validation_test_pipeline(stores):
    overlap = open_population(stores["package"], "overlap", h_start=1)
    result = compare_inputs(stores["test"], overlap, tolerance=1e-12)
    assert result["reference_rows"] == result["candidate_rows"] == result["matched_rows"] > 0
    assert result["coverage"] == 1.0
    assert result["within_tolerance_fraction"] == 1.0
    assert result["timestamp_mismatch_rows"] == 0
    # The later dividend rescales adjusted history; as-of inputs only change by rounding.
    assert result["max_relative_difference"] < 1e-12
    expected, actual = population_labels(stores["test"]), population_labels(overlap)
    assert np.array_equal(expected["metadata"], actual["metadata"])
    for name in ("targets", "absolute_targets"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)


def test_forward_population_follows_the_registered_definition(stores):
    forward = open_population(stores["package"], "forward", h_start=1)
    sessions = stores["sessions"]
    sessions = sessions[(sessions < pd.Timestamp(END, tz="UTC"))
                        & (sessions != pd.Timestamp(CLOSED, tz="UTC"))]
    expected = [
        day for position, day in enumerate(sessions[:-14])
        if day >= pd.Timestamp("2026-06-01", tz="UTC")
        and sessions[position + 14] < pd.Timestamp(END, tz="UTC")
    ]
    labels = population_labels(forward)
    days = pd.DatetimeIndex(pd.to_datetime(labels["metadata"]["date"], utc=True))
    recorded = stores["payload"]["populations"]["forward"]["samples"]
    assert len(forward) == 2 * len(expected) == recorded
    for symbol in ("2330.TW", "6488.TWO"):
        assert days[labels["metadata"]["symbol"] == symbol].equals(pd.DatetimeIndex(expected))
    assert set(labels["metadata"]["market"]) == {"TWSE", "TPEX"}
    assert stores["payload"]["calendar_additions"] == ADDITIONS
    # Without the closure in the calendar, every window that spans it is rejected as a
    # missing market session; with the addition nothing is rejected for that reason.
    unadjusted = stores["unadjusted"]["populations"]
    gap = "asset_market_session_gap"
    assert unadjusted["forward"]["samples"] < recorded
    assert stores["payload"]["populations"]["forward"]["audit"].get(gap, 0) == 0
    assert unadjusted["forward"]["audit"][gap] == recorded - unadjusted["forward"]["samples"]
    assert unadjusted["overlap"]["samples"] == stores["payload"]["populations"]["overlap"][
        "samples"
    ]


def test_calendar_additions_come_only_from_forward_dates(tmp_path):
    from stock_forecasting.forward.package import calendar_additions_from_fetch

    def fetch(closed, opened=()):
        return {"calendar": {"calendar_sessions_without_trading": list(closed),
                             "trading_days_missing_from_calendar": list(opened)}}

    assert calendar_additions_from_fetch(fetch([])) == {}
    assert calendar_additions_from_fetch(fetch([CLOSED])) == ADDITIONS
    with pytest.raises(ValueError, match="2026-05-29"):
        calendar_additions_from_fetch(fetch([CLOSED, "2026-05-29"]))
    with pytest.raises(ValueError, match="only from 2026-06-01"):
        build_package(
            tmp_path / "none.parquet", tmp_path / "p", end_exclusive=END,
            calendar_additions={"XTAI": {"open": [], "closed": ["2026-05-29"]}},
        )


def test_package_is_sealed_and_labels_are_written_in_baseline_format(stores, tmp_path):
    payload = load_package(stores["package"])
    assert payload["populations"]["overlap"]["slot"] == "validation"
    assert payload["populations"]["forward"]["slot"] == "test"
    forward = open_population(stores["package"], "forward", h_start=1)
    summary = write_population_labels(forward, tmp_path / "labels", population="forward")
    assert summary["samples"] == len(forward)
    assert np.load(tmp_path / "labels" / "targets.npy").shape == (len(forward), 14)
    assert np.load(tmp_path / "labels" / "metadata.npy").dtype.names == (
        "symbol", "date", "market", "asset_type", "provider"
    )
    with pytest.raises(FileExistsError):
        write_population_labels(forward, tmp_path / "labels", population="forward")


def test_package_rejects_tampering_and_bad_end_dates(stores, tmp_path):
    copy = tmp_path / "copy"
    import shutil

    shutil.copytree(stores["package"], copy)
    shard = next((copy / "bar-store" / "shards").rglob("*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed after the package was sealed"):
        load_package(copy)
    payload = json.loads((copy / PACKAGE_FILE).read_text())
    payload["populations"]["forward"]["slot"] = "validation"
    (copy / PACKAGE_FILE).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="slot mapping"):
        load_package(copy, verify_files=False)
    with pytest.raises(ValueError, match="after 2026-06-01"):
        build_package(tmp_path / "none.parquet", tmp_path / "p", end_exclusive="2026-06-01")
