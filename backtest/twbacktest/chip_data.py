"""Readers for the tw_chip_data curated layer (protocol 13.19 supplement 4).

The sibling project publishes Parquet partitions ``curated/<dataset>/year=YYYY/part-0.parquet``
with a ``_SUCCESS.json`` marker per partition (its ``docs/INTEGRATION.md`` is the contract). Only
pyarrow is needed to read them. Every row carries ``available_at`` (Asia/Taipei); for the daily
lists and the securities-lending table it is the data date's 23:59:59, so the row of signal date
t is usable at the open of t + 1, which is exactly when the band books trade. Symbols there are the
bare exchange codes (``2330``); the walk-forward predictions use ``2330.TW`` / ``2330.TWO``, so the
join strips the suffix and keeps the market.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

SUFFIX = {"TWSE": ".TW", "TPEX": ".TWO"}


def partition_files(root: Path, dataset: str, years: list[int] | None = None) -> list[str]:
    """Partitions of a dataset that carry the _SUCCESS marker (optionally only some years)."""
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


def partition_digests(root: Path, dataset: str, years: list[int] | None = None) -> dict[str, str]:
    """sha256 of every partition read, from its _SUCCESS.json, for the downstream manifest."""
    out = {}
    for path in partition_files(root, dataset, years):
        marker = json.loads(Path(os.path.dirname(path), "_SUCCESS.json").read_text("utf-8"))
        out[os.path.relpath(path, root)] = str(marker.get("sha256", ""))
    return out


def load_curated(
    root: Path, dataset: str, columns: list[str] | None = None, years: list[int] | None = None
) -> pd.DataFrame:
    table = ds.dataset(partition_files(root, dataset, years), format="parquet")
    frame = table.to_table(columns=columns).to_pandas()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def with_prediction_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the ``2330.TW`` style symbol the walk-forward predictions use."""
    frame = frame.copy()
    frame["prediction_symbol"] = frame["symbol"] + frame["market"].map(SUFFIX)
    return frame


def short_eligibility(root: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Per (prediction_symbol, date): the short-sale flags of the daily eligibility list."""
    frame = load_curated(
        root, "eligibility",
        columns=["date", "symbol", "market", "short_suspended", "sbl_suspended",
                 "limit_down_block", "sbl_eligible"],
        years=years,
    )
    frame = with_prediction_symbol(frame)
    frame["margin_short_allowed"] = ~(frame["short_suspended"] | frame["limit_down_block"])
    return frame[["prediction_symbol", "date", "sbl_eligible", "margin_short_allowed",
                  "short_suspended", "sbl_suspended", "limit_down_block"]]


def disposition_periods(root: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Per announcement: the disposition window of one stock (protocol 13.23 supplement 3).

    Partitions are by announcement year; a window announced late in a year can run into the
    next, so the caller passes the years it needs plus the one before.
    """
    table = ds.dataset(partition_files(root, "disposition", years), format="parquet")
    frame = table.to_table(
        columns=["announce_date", "symbol", "market", "period_start", "period_end"]
    ).to_pandas()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    for column in ("announce_date", "period_start", "period_end"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    frame = with_prediction_symbol(frame)
    return frame[["prediction_symbol", "announce_date", "period_start", "period_end"]]


def borrow_quota(root: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Per (prediction_symbol, date): next-session securities-lending sell quota (shares) and the
    margin short room (short_limit - short_balance, shares)."""
    frame = load_curated(
        root, "sbl",
        columns=["date", "symbol", "market", "sbl_next_quota", "short_limit", "short_balance"],
        years=years,
    )
    frame = with_prediction_symbol(frame)
    quota = frame["sbl_next_quota"].astype("float64")
    room = (frame["short_limit"].astype("float64") - frame["short_balance"].astype("float64"))
    frame["sbl_quota_shares"] = quota.to_numpy()
    frame["margin_short_room_shares"] = np.maximum(room.to_numpy(), 0.0)
    return frame[["prediction_symbol", "date", "sbl_quota_shares", "margin_short_room_shares"]]


def align_rows(
    predictions: pd.DataFrame, table: pd.DataFrame, columns: list[str]
) -> dict[str, np.ndarray]:
    """``columns`` of ``table`` aligned with the prediction rows by (symbol, signal date).

    Rows without a match get NaN (numeric) or False (boolean); callers decide what a missing
    list entry means (for the short book: not borrowable).
    """
    merged = predictions[["symbol", "signal_date"]].merge(
        table, left_on=["symbol", "signal_date"], right_on=["prediction_symbol", "date"],
        how="left", validate="many_to_one",
    )
    out = {}
    for column in columns:
        values = merged[column]
        # After a left merge a bool column becomes object with NaN, so test the source dtype.
        if table[column].dtype == bool or str(table[column].dtype) == "boolean":
            out[column] = values.astype("boolean").fillna(False).to_numpy(dtype=bool)
        else:
            out[column] = values.to_numpy(dtype=np.float64)
    out["matched"] = merged["date"].notna().to_numpy(dtype=bool)
    return out
