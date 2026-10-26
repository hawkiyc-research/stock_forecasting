"""Chip feature library (protocol 13.18 item 2): spec, point-in-time rule, arithmetic, store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
TAIPEI = "Asia/Taipei"


def _spec(name="chip_tw_v1"):
    from stock_forecasting.data.feature_library import load_spec

    return load_spec(ROOT / "configs" / "features" / f"{name}.json")


def _write(root: Path, dataset: str, frame: pd.DataFrame, date_column: str) -> None:
    frame = frame.copy()
    # Like the real curated layer: naive dates in milliseconds, available_at in microseconds.
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            unit = "us" if getattr(frame[column].dt, "tz", None) is not None else "ms"
            frame[column] = frame[column].dt.as_unit(unit)
    for year, part in frame.groupby(pd.to_datetime(frame[date_column]).dt.year):
        folder = root / "curated" / dataset / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False),
                       folder / "part-0.parquet")
        (folder / "_SUCCESS.json").write_text(json.dumps({"rows": len(part), "sha256": "x"}))
    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests" / f"{dataset}.json").write_text(json.dumps({"dataset": dataset}))


def _stamp(dates, hour=23, minute=59, second=59, offset_days=0):
    base = pd.DatetimeIndex(pd.to_datetime(dates)) + pd.Timedelta(days=offset_days)
    return (base + pd.Timedelta(hours=hour, minutes=minute, seconds=second)).tz_localize(TAIPEI)


def _synthetic_root(tmp_path: Path, days: pd.DatetimeIndex, symbols: dict[str, str],
                    *, volume_per_day: float = 1000.0) -> Path:
    """A curated layer for ``symbols`` ({code: market}) over the trading ``days``.

    Flows and balances follow simple deterministic patterns so the tests can state exact
    values: the foreign investor buys 100 shares a day for the first ten sessions and sells
    100 a day afterwards; the margin balance grows 1% a session; day-trade volume equals 10
    times the session index; one attention notice, one disposition and one revenue row.
    """
    root = tmp_path / "chip"
    n = len(days)
    index = np.arange(n)
    inst_rows, margin_rows, sbl_rows, dt_rows, fh_rows = [], [], [], [], []
    for code, market in symbols.items():
        foreign = np.where(index < 10, 100.0, -100.0)
        for investor, net in (("foreign", foreign), ("foreign_dealer", np.zeros(n)),
                              ("trust", np.full(n, 50.0)), ("dealer", np.zeros(n)),
                              ("dealer_self", np.zeros(n)), ("dealer_hedge", np.zeros(n)),
                              ("total", foreign + 50.0)):
            inst_rows.append(pd.DataFrame({
                "date": days, "symbol": code, "market": market, "investor_type": investor,
                "buy_shares": np.maximum(net, 0), "sell_shares": np.maximum(-net, 0),
                "net_shares": net.astype(np.int64), "coverage": "full",
                "available_at": _stamp(days)}))
        balance = np.round(1000.0 * 1.01 ** index).astype(np.int64)
        margin_rows.append(pd.DataFrame({
            "date": days, "symbol": code, "market": market, "margin_balance": balance,
            "margin_limit": 2500, "short_balance": 100 + index, "offset_shares": 50,
            "available_at": _stamp(days)}))
        sbl_rows.append(pd.DataFrame({
            "date": days, "symbol": code, "market": market, "sbl_balance": 200 + 2 * index,
            "sbl_sell": 20, "available_at": _stamp(days)}))
        dt_rows.append(pd.DataFrame({
            "date": days, "symbol": code, "market": market, "daytrade_volume": 10 * index,
            "available_at": _stamp(days, offset_days=2)}))
        if market == "TWSE":
            fh_rows.append(pd.DataFrame({
                "date": days, "symbol": code, "market": market,
                "foreign_held_pct": 20.0 + 0.1 * index, "foreign_cap_pct": 100.0,
                "available_at": _stamp(days)}))
    _write(root, "institutional", pd.concat(inst_rows, ignore_index=True), "date")
    _write(root, "margin", pd.concat(margin_rows, ignore_index=True), "date")
    _write(root, "sbl", pd.concat(sbl_rows, ignore_index=True), "date")
    _write(root, "daytrade", pd.concat(dt_rows, ignore_index=True), "date")
    _write(root, "foreign_holding", pd.concat(fh_rows, ignore_index=True), "date")
    first_code, first_market = next(iter(symbols.items()))
    at = [days[min(i, n - 1)] for i in (5, 10, 11, 20)]  # notice, disposition announce, start, end
    _write(root, "attention", pd.DataFrame({
        "announce_date": [at[0]], "symbol": [first_code], "market": [first_market],
        "available_at": _stamp([at[0]])}), "announce_date")
    _write(root, "disposition", pd.DataFrame({
        "announce_date": [at[1]], "symbol": [first_code], "market": [first_market],
        "period_start": [at[2]], "period_end": [at[3]],
        "available_at": _stamp([at[1]])}), "announce_date")
    month = days[0].to_period("M").to_timestamp()
    months = pd.date_range(month - pd.DateOffset(months=5), periods=6, freq="MS")
    yoy = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 40.0])
    _write(root, "monthly_revenue", pd.DataFrame({
        "year_month": months, "symbol": first_code, "market": first_market, "yoy_pct": yoy,
        "mom_pct": 5.0, "basis": ["parent"] * 5 + ["consolidated"],
        "available_at": _stamp(months + pd.DateOffset(months=1) + pd.Timedelta(days=9))}),
        "year_month")
    return root


def test_spec_names_and_validation(tmp_path):
    from stock_forecasting.data.chip_features import chip_feature_names, load_chip_spec
    from stock_forecasting.data.feature_library import feature_names, spec_identity

    spec = _spec()
    names = chip_feature_names(spec)
    assert len(names) == 67 and len(set(names)) == 67
    assert names[:4] == ["FRGN_NET_V1", "FRGN_NET_V5", "FRGN_NET_V10", "FRGN_NET_V20"]
    assert names[4:7] == ["FRGN_STREAK", "FRGN_NET20_FLOAT", "FRGN_Z60"]
    assert names[-9:] == ["ATTN_CNT_30D", "DISP_ACTIVE", "DISP_DAYS_SINCE_ANN",
                          "DISP_DAYS_SINCE_END", "REV_DAYS_SINCE", "REV_YOY", "REV_MOM",
                          "REV_YOY_SURPRISE", "REV_BASIS_SWITCH"]
    assert feature_names(spec) == names
    combined = _spec("alpha_chip_tw_v1")
    assert feature_names(combined)[:183] == feature_names(
        _spec_alpha()) and feature_names(combined)[183:] == names
    identity = spec_identity(combined)
    assert identity["name"] == "alpha_chip_tw_v1" and identity["alpha"]["name"] == "alpha_tw_v1"
    assert "_alpha_spec" not in json.dumps(identity)
    broken = json.loads((ROOT / "configs/features/chip_tw_v1.json").read_text())
    broken["chip_data"]["datasets"].remove("sbl")
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="datasets"):
        load_chip_spec(path)


def _spec_alpha():
    from stock_forecasting.data.feature_library import load_spec

    return load_spec(ROOT / "configs/features/alpha_tw_v1.json")


def test_partitions_are_found_under_percent_encoded_folders(tmp_path):
    """The RunPod S3 gateway stores ``year=2024`` as ``year%3D2024`` on the volume."""
    from stock_forecasting.data.chip_features import partition_files, read_curated

    root = tmp_path / "chip"
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["1101"],
                          "market": ["TWSE"], "value": [1.0]})
    for folder_name in ("year%3D2024", "year=2025"):
        folder = root / "curated" / "margin" / folder_name
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), folder / "part-0.parquet")
        (folder / "_SUCCESS.json").write_text(json.dumps({"sha256": "x"}))
    unfinished = root / "curated" / "margin" / "year%3D2026"
    unfinished.mkdir()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), unfinished / "part-0.parquet")
    files = partition_files(root, "margin")
    assert [Path(f).parent.name for f in files] == ["year%3D2024", "year=2025"]
    assert [Path(f).parent.name for f in partition_files(root, "margin", [2024])] == ["year%3D2024"]
    assert len(read_curated(root, "margin", ["date", "symbol", "value"])) == 2


def test_usable_positions_follow_the_next_open_rule():
    from stock_forecasting.data.chip_features import usable_positions

    spec = _spec()
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    stamps = pd.Series(pd.DatetimeIndex([
        "2024-01-03 23:59:59", "2024-01-03 08:00:00", "2024-01-01 15:00:00",
        "2024-01-04 23:59:59", "2024-01-04 08:59:59",
    ]).tz_localize(TAIPEI))
    positions = usable_positions(stamps, days, spec)
    # A row published after the close of 01-03 serves the signal date 01-03 (orders execute at
    # the 01-04 open); one published before the 01-03 open serves 01-02; one published before
    # the first open serves nothing; the last row serves 01-04, the last day of the calendar.
    assert positions.tolist() == [1, 0, -1, 2, 1]


def test_chip_arithmetic_and_point_in_time(tmp_path, monkeypatch):
    from stock_forecasting.data.chip_features import (
        build_chip_tables,
        chip_feature_names,
        chip_root,
        symbol_chip_features,
    )

    spec = _spec()
    days = pd.bdate_range("2024-01-02", periods=80)
    root = _synthetic_root(tmp_path, days, {"1101": "TWSE", "6274": "TPEX"})
    volume = pd.concat([
        pd.DataFrame({"symbol": symbol, "date": days, "volume": 1000.0})
        for symbol in ("1101.TW", "6274.TWO")
    ], ignore_index=True)
    tables = build_chip_tables(root, spec, days, volume)
    names = chip_feature_names(spec)
    col = {name: i for i, name in enumerate(names)}
    cutoffs = pd.DatetimeIndex(days).tz_localize("UTC")
    tw = symbol_chip_features(tables, "1101.TW", cutoffs, spec)
    assert tw.shape == (80, 67)
    # Institutional flows: +100 a day for ten sessions, then -100, on 1,000 shares of volume.
    assert tw[4, col["FRGN_NET_V5"]] == pytest.approx(0.1)
    assert np.isnan(tw[3, col["FRGN_NET_V5"]])  # window not yet full
    assert tw[9, col["FRGN_STREAK"]] == 10 and tw[10, col["FRGN_STREAK"]] == -1
    assert tw[14, col["FRGN_NET_V10"]] == pytest.approx(0.0)  # five buys, five sells
    assert tw[19, col["FRGN_NET20_FLOAT"]] == pytest.approx(0.0)  # float = 2,500 x 4
    assert tw[9, col["FRGN_NET20_FLOAT"]] is not None and np.isnan(tw[9, col["FRGN_NET20_FLOAT"]])
    assert tw[0, col["TRST_NET_V1"]] == pytest.approx(0.05)
    assert tw[4, col["INST_NET_V5"]] == pytest.approx(0.15)
    assert tw[0, col["DSLF_STREAK"]] == 0.0
    # Margin: 1% growth a session, limit 2,500, short balance 100 + i, offsets 50.
    assert tw[1, col["MARGIN_BAL_D1"]] == pytest.approx(0.01, abs=1e-3)
    assert tw[0, col["MARGIN_UTIL"]] == pytest.approx(0.4)
    assert tw[0, col["SHORT_MARGIN_RATIO"]] == pytest.approx(0.1)
    assert tw[0, col["OFFSET_V"]] == pytest.approx(0.05)
    assert np.isnan(tw[0, col["MARGIN_BAL_D1"]])
    # Securities lending: balance 200 + 2i on a float of 10,000; 20 sold a day.
    assert tw[0, col["SBL_BAL_FLOAT"]] == pytest.approx(0.02)
    assert tw[4, col["SBL_SELL_V5"]] == pytest.approx(0.02)
    # Day trade: final figures two business days later, so the cutoff i sees session i - 2.
    assert np.isnan(tw[1, col["DT_RATIO"]])
    assert tw[2, col["DT_RATIO"]] == pytest.approx(0.0)
    assert tw[12, col["DT_RATIO"]] == pytest.approx(0.1)  # 10 x 10 / 1,000 from session 10
    assert tw[12, col["DT_RATIO_JUMP"]] == pytest.approx(0.01)
    # Foreign holding (TWSE only): +0.1 point a session, cap 100%.
    assert tw[5, col["FH_PCT_D5"]] == pytest.approx(0.5)
    assert tw[0, col["FH_ROOM"]] == pytest.approx(80.0)
    tpex = symbol_chip_features(tables, "6274.TWO", cutoffs, spec)
    assert np.isnan(tpex[:, col["FH_ROOM"]]).all()
    assert tpex[0, col["MARGIN_UTIL"]] == pytest.approx(0.4)
    # Events: one notice on session 5, one disposition announced on 10 for sessions 11-20.
    assert tw[4, col["ATTN_CNT_30D"]] == 0 and tw[5, col["ATTN_CNT_30D"]] == 1
    # 2024-01-09 + 30 calendar days = 2024-02-08 (22 sessions later): the window is left-open.
    assert tw[5 + 21, col["ATTN_CNT_30D"]] == 1 and tw[5 + 22, col["ATTN_CNT_30D"]] == 0
    assert tw[10, col["DISP_ACTIVE"]] == 0 and tw[11, col["DISP_ACTIVE"]] == 1
    assert tw[20, col["DISP_ACTIVE"]] == 1 and tw[21, col["DISP_ACTIVE"]] == 0
    assert np.isnan(tw[9, col["DISP_DAYS_SINCE_ANN"]]) and tw[10, col["DISP_DAYS_SINCE_ANN"]] == 0
    assert tw[15, col["DISP_DAYS_SINCE_ANN"]] == 5
    assert np.isnan(tw[20, col["DISP_DAYS_SINCE_END"]]) and tw[21, col["DISP_DAYS_SINCE_END"]] == 1
    assert np.isnan(tpex[:, col["DISP_ACTIVE"]]).sum() == 0
    assert tpex[:, col["DISP_ACTIVE"]].sum() == 0
    # Revenue: the latest month (yoy 40 against a 10, 10, 10 trend) is usable from the 10th of
    # the following month, which is inside the sample; the row before it is five months old.
    usable = np.flatnonzero(tw[:, col["REV_YOY"]] == 40.0)
    assert usable.size > 0
    first = usable[0]
    assert tw[first, col["REV_DAYS_SINCE"]] == 0
    assert tw[first, col["REV_YOY_SURPRISE"]] == pytest.approx(30.0)
    assert tw[first, col["REV_BASIS_SWITCH"]] == 1.0 and tw[first, col["REV_MOM"]] == 5.0
    # The staleness limit blanks a symbol whose rows stop.
    short_days = days[:20]
    root2 = _synthetic_root(tmp_path / "second", short_days, {"1101": "TWSE"})
    tables2 = build_chip_tables(root2, spec, days, volume[volume.symbol == "1101.TW"])
    tw2 = symbol_chip_features(tables2, "1101.TW", cutoffs, spec)
    assert np.isfinite(tw2[19 + 10, col["MARGIN_UTIL"]])
    assert np.isnan(tw2[19 + 11, col["MARGIN_UTIL"]])
    # The environment lookup names the variable when it is missing (the Pod job exports it).
    monkeypatch.delenv("TW_CHIP_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="TW_CHIP_DATA_ROOT"):
        chip_root(spec)


def _fixture_days():
    from stock_forecasting.data.sample_universe import market_sessions

    return market_sessions("XTAI", "2021-01-04", "2024-12-31")[:360].tz_localize(None)


@pytest.mark.parametrize("name", ["chip_tw_v1", "alpha_chip_tw_v1"])
def test_store_holds_chip_columns_after_the_alpha_block(rank_config, tmp_path, monkeypatch, name):
    pytest.importorskip("exchange_calendars")
    from conftest import RANK_TW_STOCKS  # the tests directory is on sys.path under pytest

    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.data.feature_library import (
        ensure_features,
        feature_names,
        read_split_features,
    )

    days = _fixture_days()
    root = _synthetic_root(tmp_path / "chipdata", days,
                           {symbol.split(".")[0]: "TWSE" for symbol in RANK_TW_STOCKS})
    monkeypatch.setenv("TW_CHIP_DATA_ROOT", str(root))
    spec = _spec(name)
    names = feature_names(spec)
    source = lazy_dataset(rank_config, "train")
    try:
        store = ensure_features(source, spec, workers=1)
        payload = json.loads((store / "features.json").read_text())
        assert payload["feature_names"] == names
        assert payload["identity"]["chip_data"]["manifests"]["margin"]
        assert payload["chip_coverage"]["symbols_with_margin"] == len(RANK_TW_STOCKS)
        frame = read_split_features(store)
        assert len(frame) == len(source)
        markets = pd.Series(source.target_batch(np.arange(len(source)))["markets"])
        tw_rows = frame[(markets == "TWSE").to_numpy()]
        us_rows = frame[(markets != "TWSE").to_numpy()]
        # The synthetic margin balance grows 1% a session on a fixed limit of 2,500, so the
        # utilization is 0.4 x 1.01^i for the i-th session: finite and positive on every TW row.
        assert np.isfinite(tw_rows["MARGIN_UTIL"]).all() and (tw_rows["MARGIN_UTIL"] > 0).all()
        # The cutoff dates are sessions, so each row's utilization matches the fixture formula.
        stamps = pd.DatetimeIndex(tw_rows["timestamp"]).tz_convert("UTC").tz_localize(None)
        session_index = np.searchsorted(days.asi8, stamps.normalize().asi8)
        np.testing.assert_allclose(tw_rows["MARGIN_UTIL"].to_numpy(dtype=np.float64),
                                   0.4 * 1.01 ** session_index, rtol=2e-3)
        assert np.isin(tw_rows["DISP_ACTIVE"].to_numpy(), [0.0, 1.0]).all()
        assert us_rows["MARGIN_UTIL"].isna().all()
        if name == "alpha_chip_tw_v1":
            alpha_store = ensure_features(source, _spec_alpha(), workers=1)
            alpha = read_split_features(alpha_store)
            alpha_names = feature_names(_spec_alpha())
            np.testing.assert_allclose(
                frame[alpha_names].to_numpy(dtype=np.float32),
                alpha[alpha_names].to_numpy(dtype=np.float32), equal_nan=True)
    finally:
        source.close()
