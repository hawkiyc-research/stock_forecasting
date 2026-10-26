"""Chip (籌碼) and event features from the ``tw_chip_data`` curated layer (protocol 13.18 item 2).

The sibling project publishes Parquet partitions ``curated/<dataset>/year=YYYY/part-0.parquet``
with a ``_SUCCESS.json`` marker per partition and one manifest per dataset under ``manifests/``
(its ``docs/INTEGRATION.md`` is the contract). This module turns those tables into the fixed
column list of ``configs/features/chip_tw_v1.json`` and aligns them with the bar store's
sample rows. The feature library (``feature_library.py``) owns the store layout; this module
owns the chip arithmetic and the point-in-time rule.

Point in time: every source row carries ``available_at`` (Asia/Taipei). A row is usable for
the signal date t when ``available_at`` precedes the open of the trading day after t, which is
when the model's orders execute; daily tables published after the close of T therefore serve
the signal date T, the day-trade statistics (final two business days later) the signal date
T + 2, and the monthly revenue (10th of the next month) the signal dates from that 10th on.

Why the chip block is computed in the parent process: the arithmetic is a handful of
vectorized pandas group operations over a few million rows and finishes in seconds, well
below the cost of shipping the tables to a process pool; the per-symbol alignment loop is a
``searchsorted`` per table. The price / volume block of a combined spec keeps the bounded
process pool of ``feature_library``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHIP_SCHEMA = "chip-tw-feature-library-v1"
SUFFIX = {"TWSE": ".TW", "TPEX": ".TWO"}
DAILY_DATASETS = ("institutional", "margin", "sbl", "daytrade", "foreign_holding")
EVENT_DATASETS = ("attention", "disposition", "monthly_revenue")
REQUIRED_DATASETS = DAILY_DATASETS + EVENT_DATASETS
INSTITUTIONAL_SUFFIXES = ("NET_V{window}", "STREAK", "NET{float_window}_FLOAT", "Z{zscore_window}")
ALPHA_PRIVATE_KEY = "_alpha_spec"


# --- spec -----------------------------------------------------------------------------------


def is_chip_spec(spec: dict) -> bool:
    return spec.get("schema") == CHIP_SCHEMA


def _windows(values, minimum: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) > 0
        and all(type(v) is int and v >= minimum for v in values)
        and values == sorted(set(values))
    )


def load_chip_spec(path: str | Path, *, project_root: Path | None = None) -> dict:
    """Read and validate a chip spec; the included alpha spec is loaded under a private key."""
    from stock_forecasting.data.feature_library import load_spec as load_alpha_spec

    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "name", "include_alpha", "chip_data", "point_in_time", "float_proxy",
        "institutional", "margin", "sbl", "daytrade", "foreign_holding", "events",
    }
    if not required <= set(spec) or spec["schema"] != CHIP_SCHEMA:
        raise ValueError("Chip feature spec is incomplete or uses an unsupported schema")
    if not isinstance(spec["name"], str) or not spec["name"].isidentifier():
        raise ValueError("Chip feature spec name must be an identifier")
    chip = spec["chip_data"]
    if not isinstance(chip.get("root_env"), str) or not chip["root_env"].isidentifier():
        raise ValueError("chip_data.root_env must name an environment variable")
    if set(chip.get("datasets", [])) != set(REQUIRED_DATASETS):
        raise ValueError(f"chip_data.datasets must list exactly {sorted(REQUIRED_DATASETS)}")
    pit = spec["point_in_time"]
    stale = pit.get("max_stale_sessions", {})
    if (
        pit.get("decision") != "next_open"
        or not isinstance(pit.get("open_time"), str)
        or pit.get("timezone") != "Asia/Taipei"
        or not all(type(stale.get(k)) is int and stale[k] > 0
                   for k in ("daily", "daytrade", "monthly"))
    ):
        raise ValueError("point_in_time must be the next_open rule with positive staleness")
    proxy = spec["float_proxy"]
    if proxy.get("source") != "margin.margin_limit" or type(proxy.get("multiplier")) is not int:
        raise ValueError("float_proxy must be margin.margin_limit with an integer multiplier")
    inst = spec["institutional"]
    types = inst.get("investor_types")
    if (
        not isinstance(types, dict) or not types
        or not all(k.isidentifier() and k.isupper() and isinstance(v, str)
                   for k, v in types.items())
        or not _windows(inst.get("flow_windows"), 1)
        or type(inst.get("float_window")) is not int
        or type(inst.get("zscore_window")) is not int
        or inst["zscore_window"] < 5
        or not _windows(inst.get("total_windows"), 1)
    ):
        raise ValueError("institutional block is invalid")
    if not _windows(spec["margin"].get("balance_windows"), 1):
        raise ValueError("margin.balance_windows is invalid")
    if (not _windows(spec["sbl"].get("balance_windows"), 1)
            or type(spec["sbl"].get("sell_window")) is not int):
        raise ValueError("sbl block is invalid")
    if type(spec["daytrade"].get("window")) is not int or spec["daytrade"]["window"] < 2:
        raise ValueError("daytrade.window is invalid")
    if not _windows(spec["foreign_holding"].get("change_windows"), 1):
        raise ValueError("foreign_holding.change_windows is invalid")
    events = spec["events"]
    if not all(type(events.get(k)) is int and events[k] > 0 for k in
               ("attention_window_days", "max_event_sessions", "revenue_trend_months")):
        raise ValueError("events block is invalid")
    include = spec["include_alpha"]
    if include is not None:
        if not isinstance(include, str):
            raise ValueError("include_alpha must be null or a path to an alpha spec")
        root = project_root if project_root is not None else _project_root(path)
        alpha_path = Path(include) if Path(include).is_absolute() else root / include
        spec[ALPHA_PRIVATE_KEY] = load_alpha_spec(alpha_path)
    return spec


def _project_root(spec_path: Path) -> Path:
    """``configs/features/<spec>.json`` lives two levels under the project root."""
    resolved = spec_path.resolve()
    if resolved.parent.name == "features" and resolved.parent.parent.name == "configs":
        return resolved.parent.parent.parent
    return Path.cwd()


def alpha_part(spec: dict) -> dict | None:
    return spec.get(ALPHA_PRIVATE_KEY)


def public_spec(spec: dict) -> dict:
    """The spec as written on disk (private keys removed), for hashing."""
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def chip_feature_names(spec: dict) -> list[str]:
    """The chip block's column order: institutional, margin, sbl, day trade, foreign, events."""
    inst = spec["institutional"]
    names: list[str] = []
    for key in inst["investor_types"]:
        names += [f"{key}_NET_V{w}" for w in inst["flow_windows"]]
        names.append(f"{key}_STREAK")
        names.append(f"{key}_NET{inst['float_window']}_FLOAT")
        names.append(f"{key}_Z{inst['zscore_window']}")
    names += [f"INST_NET_V{w}" for w in inst["total_windows"]]
    names += [f"MARGIN_BAL_D{w}" for w in spec["margin"]["balance_windows"]]
    names.append("MARGIN_UTIL")
    names += [f"SHORT_BAL_D{w}" for w in spec["margin"]["balance_windows"]]
    names += ["SHORT_MARGIN_RATIO", "OFFSET_V"]
    names += [f"SBL_BAL_D{w}" for w in spec["sbl"]["balance_windows"]]
    names += ["SBL_BAL_FLOAT", f"SBL_SELL_V{spec['sbl']['sell_window']}"]
    window = spec["daytrade"]["window"]
    names += ["DT_RATIO", f"DT_RATIO_MA{window}", f"DT_RATIO_Z{window}", "DT_RATIO_JUMP"]
    names += [f"FH_PCT_D{w}" for w in spec["foreign_holding"]["change_windows"]]
    names.append("FH_ROOM")
    names += [
        f"ATTN_CNT_{spec['events']['attention_window_days']}D", "DISP_ACTIVE",
        "DISP_DAYS_SINCE_ANN", "DISP_DAYS_SINCE_END", "REV_DAYS_SINCE", "REV_YOY", "REV_MOM",
        "REV_YOY_SURPRISE", "REV_BASIS_SWITCH",
    ]
    if len(set(names)) != len(names):
        raise ValueError("Chip feature spec produces duplicate column names")
    return names


# --- curated layer --------------------------------------------------------------------------


def chip_root(spec: dict) -> Path:
    name = spec["chip_data"]["root_env"]
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"Chip feature spec {spec['name']!r} needs the curated layer: set {name} to "
            "<tw_chip_data>/data (or pass --chip-data-root)"
        )
    root = Path(value)
    if not (root / "curated").is_dir():
        raise ValueError(f"{name}={root} has no curated/ directory")
    return root


def partition_files(root: Path, dataset: str, years: list[int] | None = None) -> list[str]:
    """Partitions carrying the _SUCCESS marker, optionally restricted to some years."""
    files = []
    for path in sorted(glob.glob(str(root / "curated" / dataset / "year=*" / "part-0.parquet"))):
        folder = os.path.dirname(path)
        year = int(os.path.basename(folder).split("=", 1)[1])
        if years is not None and year not in years:
            continue
        if os.path.exists(os.path.join(folder, "_SUCCESS.json")):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No finished partitions for {dataset!r} under {root}")
    return files


def read_curated(
    root: Path, dataset: str, columns: list[str], years: list[int] | None = None,
    filter=None,
) -> pd.DataFrame:
    import pyarrow.dataset as ds

    table = ds.dataset(partition_files(root, dataset, years), format="parquet")
    frame = table.to_table(columns=columns, filter=filter).to_pandas()
    if "symbol" in frame:
        frame["symbol"] = frame["symbol"].astype(str).str.strip()
    return frame


def data_identity(root: Path, spec: dict) -> dict:
    """What the store was built from: the dataset manifests' digests (or the partition
    markers' digests when a manifest is absent), so a refreshed curated layer is a new store."""
    digests = {}
    for dataset in sorted(spec["chip_data"]["datasets"]):
        manifest = root / "manifests" / f"{dataset}.json"
        if manifest.is_file():
            digests[dataset] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            continue
        parts = hashlib.sha256()
        for path in partition_files(root, dataset):
            marker = json.loads(Path(os.path.dirname(path), "_SUCCESS.json").read_text("utf-8"))
            parts.update(f"{os.path.relpath(path, root)}:{marker.get('sha256', '')}\n".encode())
        digests[dataset] = parts.hexdigest()
    return {"root_env": spec["chip_data"]["root_env"], "manifests": digests}


def prediction_symbol(frame: pd.DataFrame) -> pd.Series:
    """``2330`` on TWSE -> ``2330.TW``; ``6274`` on TPEX -> ``6274.TWO``."""
    return frame["symbol"] + frame["market"].map(SUFFIX)


# --- point in time --------------------------------------------------------------------------


def usable_positions(
    available_at: pd.Series, trading_days: pd.DatetimeIndex, spec: dict
) -> np.ndarray:
    """Position in ``trading_days`` of the last signal date each row may serve: the trading
    day before the first open that is at or after ``available_at`` (-1: none)."""
    pit = spec["point_in_time"]
    hours, minutes = (int(x) for x in pit["open_time"].split(":"))
    opens = (
        pd.DatetimeIndex(trading_days).tz_localize(pit["timezone"])
        + pd.Timedelta(hours=hours, minutes=minutes)
    ).tz_convert("UTC").asi8
    stamps = pd.DatetimeIndex(available_at).tz_convert("UTC").asi8
    return np.searchsorted(opens, stamps, side="left").astype(np.int64) - 1


def _date_positions(dates: pd.Series | np.ndarray, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Position of the last trading day <= each date (-1: before the first)."""
    values = pd.DatetimeIndex(dates).asi8
    return np.searchsorted(trading_days.asi8, values, side="right").astype(np.int64) - 1


# --- derived tables -------------------------------------------------------------------------


@dataclass
class SymbolTable:
    """One dataset's derived columns for one symbol, by usable trading-day position."""

    positions: np.ndarray  # int64, strictly increasing
    values: np.ndarray  # float64 [rows, columns]


@dataclass
class ChipTables:
    """Everything ``symbol_chip_features`` needs, grouped by prediction symbol."""

    daily: dict[str, dict[str, SymbolTable]]  # dataset -> symbol -> table
    daily_columns: dict[str, list[str]]
    attention: dict[str, np.ndarray]  # symbol -> announce dates (ns) at their usable positions
    disposition: dict[str, np.ndarray]  # symbol -> [usable pos, start ns, end ns, end pos]
    revenue: dict[str, SymbolTable]  # columns: yoy, mom, surprise, basis_switch
    trading_days: pd.DatetimeIndex
    coverage: dict[str, Any]


def _grouped_rolling(frame: pd.DataFrame, column: str, window: int, how: str) -> pd.Series:
    rolling = frame.groupby("symbol", sort=False)[column].rolling(window, min_periods=window)
    return getattr(rolling, how)().reset_index(level=0, drop=True)


def _grouped_shift(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].shift(periods)


def _ratio(numerator, denominator) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out) & (den > 0), out, np.nan)


def _streak(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Consecutive same-sign days (positive for buying, negative for selling, 0 on a flat day)."""
    sign = np.sign(frame[column].to_numpy(dtype=np.float64))
    sign = np.where(np.isfinite(sign), sign, 0.0)
    symbol = frame["symbol"].to_numpy()
    new_run = np.ones(len(frame), dtype=bool)
    new_run[1:] = (symbol[1:] != symbol[:-1]) | (sign[1:] != sign[:-1])
    index = np.arange(len(frame))
    position = index - np.maximum.accumulate(np.where(new_run, index, 0))
    return np.where(sign == 0.0, 0.0, sign * (position + 1))


def _to_symbol_tables(
    frame: pd.DataFrame, columns: list[str], positions: np.ndarray
) -> dict[str, SymbolTable]:
    """Keep the last row per (symbol, usable position) and split the frame by symbol."""
    keep = positions >= 0
    frame = frame.loc[keep, ["symbol", *columns]].copy()
    frame["pos"] = positions[keep]
    frame = frame.sort_values(["symbol", "pos"], kind="stable").drop_duplicates(
        ["symbol", "pos"], keep="last")
    out = {}
    for symbol, part in frame.groupby("symbol", sort=False):
        out[str(symbol)] = SymbolTable(
            positions=part["pos"].to_numpy(dtype=np.int64),
            values=part[columns].to_numpy(dtype=np.float64),
        )
    return out


def _prepare_daily(frame: pd.DataFrame, volume: pd.DataFrame | None) -> pd.DataFrame:
    frame = frame.copy()
    frame["symbol"] = prediction_symbol(frame)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    if volume is not None:
        frame = frame.merge(volume, on=["symbol", "date"], how="left")
    return frame


def build_chip_tables(
    root: Path, spec: dict, trading_days: pd.DatetimeIndex, volume: pd.DataFrame
) -> ChipTables:
    """Derive every chip column from the curated layer.

    ``trading_days`` are the bar store's session dates (naive, sorted); ``volume`` has one row
    per (symbol, date) with the raw traded shares of the bar store, used as the denominator of
    the flow ratios. Source years from one year before the first session are read so the
    longest window (60 sessions) is warm at the first sample.
    """
    first_year, last_year = int(trading_days[0].year) - 1, int(trading_days[-1].year)
    years = list(range(first_year, last_year + 1))
    stale = spec["point_in_time"]["max_stale_sessions"]
    coverage: dict[str, Any] = {}
    daily: dict[str, dict[str, SymbolTable]] = {}
    daily_columns: dict[str, list[str]] = {}

    # Margin first: it also supplies the float proxy (margin limit x multiplier).
    margin = _prepare_daily(read_curated(
        root, "margin",
        ["date", "symbol", "market", "margin_balance", "margin_limit", "short_balance",
         "offset_shares", "available_at"], years), volume)
    multiplier = float(spec["float_proxy"]["multiplier"])
    limit = margin["margin_limit"].to_numpy(dtype=np.float64)
    margin["float_shares"] = np.where(limit > 0, limit * multiplier, np.nan)
    float_table = margin[["symbol", "date", "float_shares"]]
    columns = []
    for w in spec["margin"]["balance_windows"]:
        margin[f"MARGIN_BAL_D{w}"] = _ratio(
            margin["margin_balance"], _grouped_shift(margin, "margin_balance", w)) - 1.0
        columns.append(f"MARGIN_BAL_D{w}")
    margin["MARGIN_UTIL"] = _ratio(margin["margin_balance"], margin["margin_limit"])
    columns.append("MARGIN_UTIL")
    for w in spec["margin"]["balance_windows"]:
        margin[f"SHORT_BAL_D{w}"] = _ratio(
            margin["short_balance"], _grouped_shift(margin, "short_balance", w)) - 1.0
        columns.append(f"SHORT_BAL_D{w}")
    margin["SHORT_MARGIN_RATIO"] = _ratio(margin["short_balance"], margin["margin_balance"])
    margin["OFFSET_V"] = _ratio(margin["offset_shares"], margin["volume"])
    columns += ["SHORT_MARGIN_RATIO", "OFFSET_V"]
    positions = usable_positions(margin["available_at"], trading_days, spec)
    daily["margin"] = _to_symbol_tables(margin, columns, positions)
    daily_columns["margin"] = columns
    coverage["margin_rows"] = len(margin)

    # Institutional flows: one filtered read per investor type onto the total rows.
    import pyarrow.compute as pc

    inst_spec = spec["institutional"]
    base_columns = ["date", "symbol", "market", "net_shares", "available_at"]
    total = read_curated(root, "institutional", base_columns, years,
                         filter=(pc.field("investor_type") == "total")
                         & (pc.field("coverage") == "full"))
    total = _prepare_daily(total.rename(columns={"net_shares": "net_total"}), volume)
    total = total.merge(float_table, on=["symbol", "date"], how="left")
    for key, investor in inst_spec["investor_types"].items():
        part = read_curated(root, "institutional", ["date", "symbol", "market", "net_shares"],
                            years, filter=(pc.field("investor_type") == investor)
                            & (pc.field("coverage") == "full"))
        part["symbol"] = prediction_symbol(part)
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        total = total.merge(part[["symbol", "date", "net_shares"]].rename(
            columns={"net_shares": f"net_{key}"}), on=["symbol", "date"], how="left")
    columns = []
    for key in inst_spec["investor_types"]:
        net = f"net_{key}"
        total[f"r1_{key}"] = _ratio(total[net], total["volume"])
        for w in inst_spec["flow_windows"]:
            total[f"{key}_NET_V{w}"] = _ratio(_grouped_rolling(total, net, w, "sum"),
                                              _grouped_rolling(total, "volume", w, "sum"))
            columns.append(f"{key}_NET_V{w}")
        total[f"{key}_STREAK"] = _streak(total, net)
        columns.append(f"{key}_STREAK")
        fw = inst_spec["float_window"]
        total[f"{key}_NET{fw}_FLOAT"] = _ratio(_grouped_rolling(total, net, fw, "sum"),
                                               total["float_shares"])
        columns.append(f"{key}_NET{fw}_FLOAT")
        zw = inst_spec["zscore_window"]
        mean = _grouped_rolling(total, f"r1_{key}", zw, "mean")
        std = _grouped_rolling(total, f"r1_{key}", zw, "std")
        total[f"{key}_Z{zw}"] = _ratio(total[f"r1_{key}"] - mean, std)
        columns.append(f"{key}_Z{zw}")
    for w in inst_spec["total_windows"]:
        total[f"INST_NET_V{w}"] = _ratio(_grouped_rolling(total, "net_total", w, "sum"),
                                         _grouped_rolling(total, "volume", w, "sum"))
        columns.append(f"INST_NET_V{w}")
    positions = usable_positions(total["available_at"], trading_days, spec)
    daily["institutional"] = _to_symbol_tables(total, columns, positions)
    daily_columns["institutional"] = columns
    coverage["institutional_rows"] = len(total)
    del total

    # Securities lending.
    sbl = _prepare_daily(read_curated(
        root, "sbl", ["date", "symbol", "market", "sbl_balance", "sbl_sell", "available_at"],
        years), volume)
    sbl = sbl.merge(float_table, on=["symbol", "date"], how="left")
    columns = []
    for w in spec["sbl"]["balance_windows"]:
        sbl[f"SBL_BAL_D{w}"] = _ratio(sbl["sbl_balance"],
                                      _grouped_shift(sbl, "sbl_balance", w)) - 1.0
        columns.append(f"SBL_BAL_D{w}")
    sbl["SBL_BAL_FLOAT"] = _ratio(sbl["sbl_balance"], sbl["float_shares"])
    sw = spec["sbl"]["sell_window"]
    sbl[f"SBL_SELL_V{sw}"] = _ratio(_grouped_rolling(sbl, "sbl_sell", sw, "sum"),
                                    _grouped_rolling(sbl, "volume", sw, "sum"))
    columns += ["SBL_BAL_FLOAT", f"SBL_SELL_V{sw}"]
    positions = usable_positions(sbl["available_at"], trading_days, spec)
    daily["sbl"] = _to_symbol_tables(sbl, columns, positions)
    daily_columns["sbl"] = columns
    del sbl

    # Day trade (final figures two business days later: the available_at carries the lag).
    daytrade = _prepare_daily(read_curated(
        root, "daytrade", ["date", "symbol", "market", "daytrade_volume", "available_at"],
        years), volume)
    window = spec["daytrade"]["window"]
    daytrade["DT_RATIO"] = _ratio(daytrade["daytrade_volume"], daytrade["volume"])
    daytrade[f"DT_RATIO_MA{window}"] = _grouped_rolling(daytrade, "DT_RATIO", window, "mean")
    std = _grouped_rolling(daytrade, "DT_RATIO", window, "std")
    daytrade[f"DT_RATIO_Z{window}"] = _ratio(
        daytrade["DT_RATIO"] - daytrade[f"DT_RATIO_MA{window}"], std)
    daytrade["DT_RATIO_JUMP"] = (daytrade["DT_RATIO"]
                                 - _grouped_shift(daytrade, "DT_RATIO", 1)).to_numpy()
    columns = ["DT_RATIO", f"DT_RATIO_MA{window}", f"DT_RATIO_Z{window}", "DT_RATIO_JUMP"]
    positions = usable_positions(daytrade["available_at"], trading_days, spec)
    daily["daytrade"] = _to_symbol_tables(daytrade, columns, positions)
    daily_columns["daytrade"] = columns
    del daytrade

    # Foreign holding (TWSE only; TPEX names stay NaN).
    foreign = _prepare_daily(read_curated(
        root, "foreign_holding",
        ["date", "symbol", "market", "foreign_held_pct", "foreign_cap_pct", "available_at"],
        years), None)
    columns = []
    for w in spec["foreign_holding"]["change_windows"]:
        foreign[f"FH_PCT_D{w}"] = (foreign["foreign_held_pct"]
                                   - _grouped_shift(foreign, "foreign_held_pct", w)).to_numpy()
        columns.append(f"FH_PCT_D{w}")
    foreign["FH_ROOM"] = (foreign["foreign_cap_pct"] - foreign["foreign_held_pct"]).to_numpy()
    columns.append("FH_ROOM")
    positions = usable_positions(foreign["available_at"], trading_days, spec)
    daily["foreign_holding"] = _to_symbol_tables(foreign, columns, positions)
    daily_columns["foreign_holding"] = columns
    del foreign

    # Events.
    attention = read_curated(root, "attention",
                             ["announce_date", "symbol", "market", "available_at"], years)
    attention["symbol"] = prediction_symbol(attention)
    attention["pos"] = usable_positions(attention["available_at"], trading_days, spec)
    attention = attention[attention["pos"] >= 0]
    attention_dates = {
        str(symbol): np.sort(trading_days.asi8[part["pos"].to_numpy()])
        for symbol, part in attention.groupby("symbol", sort=False)
    }
    disposition = read_curated(
        root, "disposition",
        ["announce_date", "symbol", "market", "period_start", "period_end", "available_at"], years)
    disposition["symbol"] = prediction_symbol(disposition)
    disposition["pos"] = usable_positions(disposition["available_at"], trading_days, spec)
    disposition = disposition[disposition["pos"] >= 0]
    end_pos = _date_positions(pd.to_datetime(disposition["period_end"]), trading_days)
    disposition = disposition.assign(
        start_ns=pd.DatetimeIndex(pd.to_datetime(disposition["period_start"])).asi8,
        end_ns=pd.DatetimeIndex(pd.to_datetime(disposition["period_end"])).asi8, end_pos=end_pos)
    disposition_rows = {
        str(symbol): part.sort_values("pos")[["pos", "start_ns", "end_ns", "end_pos"]].to_numpy(
            dtype=np.int64)
        for symbol, part in disposition.groupby("symbol", sort=False)
    }
    revenue = read_curated(
        root, "monthly_revenue",
        ["year_month", "symbol", "market", "yoy_pct", "mom_pct", "basis", "available_at"],
        list(range(first_year - 1, last_year + 1)))
    revenue["symbol"] = prediction_symbol(revenue)
    revenue["year_month"] = pd.to_datetime(revenue["year_month"])
    revenue = revenue.sort_values(["symbol", "year_month"], kind="stable").reset_index(drop=True)
    trend = spec["events"]["revenue_trend_months"]
    previous_mean = (revenue.groupby("symbol", sort=False)["yoy_pct"]
                     .transform(lambda s: s.shift(1).rolling(trend, min_periods=trend).mean()))
    revenue["surprise"] = (revenue["yoy_pct"] - previous_mean).to_numpy()
    previous_basis = _grouped_shift(revenue, "basis", 1)
    revenue["basis_switch"] = np.where(
        previous_basis.isna(), 0.0, (revenue["basis"] != previous_basis).astype(float))
    positions = usable_positions(revenue["available_at"], trading_days, spec)
    revenue_tables = _to_symbol_tables(
        revenue.rename(columns={"yoy_pct": "yoy", "mom_pct": "mom"}),
        ["yoy", "mom", "surprise", "basis_switch"], positions)
    coverage["symbols_with_margin"] = len(daily["margin"])
    coverage["symbols_with_institutional"] = len(daily["institutional"])
    coverage["stale_sessions"] = dict(stale)
    return ChipTables(
        daily=daily, daily_columns=daily_columns, attention=attention_dates,
        disposition=disposition_rows, revenue=revenue_tables, trading_days=trading_days,
        coverage=coverage,
    )


# --- alignment ------------------------------------------------------------------------------


def _lookup(table: SymbolTable | None, cutoff_pos: np.ndarray, max_stale: int,
            width: int) -> np.ndarray:
    """Latest row usable at each cutoff position, NaN when none or older than ``max_stale``."""
    out = np.full((cutoff_pos.size, width), np.nan)
    if table is None or table.positions.size == 0:
        return out
    idx = np.searchsorted(table.positions, cutoff_pos, side="right") - 1
    ok = idx >= 0
    age = cutoff_pos[ok] - table.positions[idx[ok]]
    fresh = age <= max_stale
    rows = np.flatnonzero(ok)[fresh]
    out[rows] = table.values[idx[rows]]
    return out


def symbol_chip_features(tables: ChipTables, symbol: str, cutoffs: pd.DatetimeIndex,
                         spec: dict) -> np.ndarray:
    """All chip columns for one symbol at its cutoff dates, in ``chip_feature_names`` order."""
    days = tables.trading_days
    cutoff_dates = pd.DatetimeIndex(cutoffs).tz_convert("UTC").tz_localize(None).normalize() \
        if pd.DatetimeIndex(cutoffs).tz is not None else pd.DatetimeIndex(cutoffs).normalize()
    cutoff_pos = _date_positions(cutoff_dates, days)
    stale = spec["point_in_time"]["max_stale_sessions"]
    blocks = []
    for dataset in DAILY_DATASETS:
        limit = stale["daytrade"] if dataset == "daytrade" else stale["daily"]
        blocks.append(_lookup(tables.daily[dataset].get(symbol), cutoff_pos, limit,
                              len(tables.daily_columns[dataset])))
    events = spec["events"]
    n = cutoff_pos.size
    cutoff_ns = cutoff_dates.asi8
    window_ns = np.int64(events["attention_window_days"]) * np.int64(86_400_000_000_000)
    announcements = tables.attention.get(symbol)
    if announcements is None:
        attn = np.zeros(n)
    else:
        attn = (np.searchsorted(announcements, cutoff_ns, side="right")
                - np.searchsorted(announcements, cutoff_ns - window_ns, side="right")).astype(float)
    active = np.zeros(n)
    since_ann = np.full(n, np.nan)
    since_end = np.full(n, np.nan)
    rows = tables.disposition.get(symbol)
    if rows is not None and rows.size:
        max_sessions = events["max_event_sessions"]
        announced = rows[:, 0][None, :] <= cutoff_pos[:, None]  # [cutoffs, rows]
        inside = announced & (rows[:, 1][None, :] <= cutoff_ns[:, None]) \
            & (cutoff_ns[:, None] <= rows[:, 2][None, :])
        active = inside.any(axis=1).astype(float)
        latest_ann = np.where(announced, rows[:, 0][None, :], -1).max(axis=1)
        age = cutoff_pos - latest_ann
        since_ann = np.where((latest_ann >= 0) & (age <= max_sessions), age, np.nan).astype(float)
        ended = announced & (rows[:, 2][None, :] < cutoff_ns[:, None])
        latest_end = np.where(ended, rows[:, 3][None, :], -1).max(axis=1)
        age_end = cutoff_pos - latest_end
        since_end = np.where((latest_end >= 0) & (age_end <= max_sessions) & (active == 0.0),
                             age_end, np.nan).astype(float)
    revenue = tables.revenue.get(symbol)
    rev = np.full((n, 5), np.nan)
    if revenue is not None and revenue.positions.size:
        idx = np.searchsorted(revenue.positions, cutoff_pos, side="right") - 1
        ok = idx >= 0
        age = cutoff_pos[ok] - revenue.positions[idx[ok]]
        fresh = age <= stale["monthly"]
        sel = np.flatnonzero(ok)[fresh]
        rev[sel, 0] = age[fresh]
        rev[sel, 1:] = revenue.values[idx[sel]]
    blocks.append(np.column_stack([attn, active, since_ann, since_end, rev]))
    return np.column_stack(blocks)


__all__ = [
    "CHIP_SCHEMA",
    "ChipTables",
    "alpha_part",
    "build_chip_tables",
    "chip_feature_names",
    "chip_root",
    "data_identity",
    "is_chip_spec",
    "load_chip_spec",
    "public_spec",
    "read_curated",
    "symbol_chip_features",
    "usable_positions",
]
