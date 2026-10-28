"""Walk-forward predictions and raw Taiwan prices as backtest inputs (protocol 13.13 item 3).

The B0/D0 jobs export one row per (symbol, signal date) with a score per horizon. Prices come
from the dataset's raw ``market.parquet``; open-to-open moves are built per symbol from its own
session sequence, the way the evaluation arrays did for the earlier stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from twbacktest.data import MARKETS, SplitIndex
from twbacktest.positions import Links, Moves, links

SCORE_COLUMNS = [f"h{h:02d}" for h in range(1, 15)]
INDEX_SYMBOLS = {"TWSE": "TAIEX.TW", "TPEX": "TPEX.TWO"}


def load_predictions(path: Path, model: str) -> pd.DataFrame:
    """Rows of one model (seed or ensemble) sorted by signal date, market and symbol."""
    frame = pd.read_parquet(path)
    if "model" in frame.columns:
        frame = frame[frame["model"] == model]
    if frame.empty:
        raise ValueError(f"{path} has no rows for model {model!r}")
    columns = ["symbol", "market", "signal_date", *[c for c in SCORE_COLUMNS if c in frame]]
    frame = frame[columns].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    frame = frame[frame["market"].isin(MARKETS)]
    frame = frame.drop_duplicates(["symbol", "signal_date"], keep="first")
    return frame.sort_values(["signal_date", "market", "symbol"], kind="stable").reset_index(
        drop=True
    )


LIQUIDITY_LOOKBACK = 20


def load_prices(path: Path, *, liquidity_lookback: int = LIQUIDITY_LOOKBACK) -> pd.DataFrame:
    """Adjusted open/close per (symbol, date), plus asset type, market and liquidity.

    ``raw_close`` and ``raw_open_1`` (the unadjusted close and next-session open) serve the
    price filter and board-lot feasibility; ``dollar_volume`` is the rolling median of
    close x volume over the last ``liquidity_lookback`` sessions up to the row's date (NaN
    when the file has no volume column).
    """
    columns = ["timestamp", "symbol", "asset_type", "open", "close", "adjusted_close", "market"]
    has_volume = "volume" in pq.ParquetFile(path).schema_arrow.names
    frame = pd.read_parquet(path, columns=columns + (["volume"] if has_volume else []))
    frame = frame[frame["market"].isin(MARKETS)].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    factor = frame["adjusted_close"] / frame["close"]
    frame["adj_open"] = frame["open"] * factor
    frame["adj_close"] = frame["adjusted_close"]
    frame["raw_close"] = frame["close"]
    frame["raw_open"] = frame["open"]
    frame["traded_value"] = frame["close"] * frame["volume"] if has_volume else np.nan
    frame = frame[
        [
            "symbol", "date", "asset_type", "market", "adj_open", "adj_close", "raw_close",
            "raw_open", "traded_value",
        ]
    ]
    frame = frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False)
    frame["raw_open_1"] = grouped["raw_open"].shift(-1)
    frame["dollar_volume"] = grouped["traded_value"].transform(
        lambda s: s.rolling(liquidity_lookback, min_periods=max(1, liquidity_lookback // 2))
        .median()
    )
    frame = frame.drop(columns=["raw_open", "traded_value"])
    # Per-symbol session sequence: the next two opens and the close two sessions ahead.
    frame["open_1"] = grouped["adj_open"].shift(-1)
    frame["open_2"] = grouped["adj_open"].shift(-2)
    frame["close_2"] = grouped["adj_close"].shift(-2)
    # Holding-aligned exits: the close h sessions after the signal date, h = 1..14.
    for h in range(1, 15):
        frame[f"close_{h}"] = grouped["adj_close"].shift(-h)
    return frame


@dataclass(frozen=True)
class WalkForwardInputs:
    index: SplitIndex
    scores: np.ndarray  # [rows, horizons]
    horizons: list[int]
    link: Links
    move: Moves
    dates: pd.DatetimeIndex
    raw_close: np.ndarray | None = None  # unadjusted close on the signal date
    raw_open_1: np.ndarray | None = None  # unadjusted open at the execution that follows
    dollar_volume: np.ndarray | None = None  # rolling median close x volume up to the date


def build_inputs(predictions: pd.DataFrame, prices: pd.DataFrame) -> WalkForwardInputs:
    """Join predictions with the price moves that follow each signal date."""
    stocks = prices[prices["asset_type"] != "index"]
    merged = predictions.merge(
        stocks[
            [
                "symbol", "date", "asset_type", "adj_open", "adj_close", "open_1", "open_2",
                "close_2", "raw_close", "raw_open_1", "dollar_volume",
            ]
        ],
        left_on=["symbol", "signal_date"],
        right_on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    indexes = prices[prices["asset_type"] == "index"]
    bench_frames = []
    for market, symbol in INDEX_SYMBOLS.items():
        rows = indexes[indexes["symbol"] == symbol][["date", "open_1", "open_2"]].copy()
        rows["market"] = market
        bench_frames.append(
            rows.rename(columns={"open_1": "bench_open_1", "open_2": "bench_open_2"})
        )
    bench = pd.concat(bench_frames, ignore_index=True)
    merged = merged.merge(
        bench,
        left_on=["market", "signal_date"],
        right_on=["market", "date"],
        how="left",
        suffixes=("", "_bench"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.log(merged["open_2"].to_numpy() / merged["open_1"].to_numpy())
        fallback = np.log(merged["close_2"].to_numpy() / merged["open_1"].to_numpy())
        gap = np.log(merged["open_1"].to_numpy() / merged["adj_close"].to_numpy())
        index_growth = np.log(merged["bench_open_2"].to_numpy() / merged["bench_open_1"].to_numpy())
    dates = pd.DatetimeIndex(sorted(merged["signal_date"].unique()))
    date_codes = dates.get_indexer(merged["signal_date"]).astype(np.int32)
    symbols, symbol_codes = np.unique(merged["symbol"].to_numpy(), return_inverse=True)
    market_codes = np.where(merged["market"].to_numpy() == MARKETS[0], 0, 1).astype(np.int8)
    is_etf = merged["asset_type"].fillna("stock").to_numpy() == "etf"
    index = SplitIndex(
        split="walk_forward",
        rows=np.arange(len(merged), dtype=np.int64),
        date_codes=date_codes,
        dates=tuple(day.strftime("%Y-%m-%d") for day in dates),
        market_codes=market_codes,
        is_etf=is_etf,
        symbol_codes=symbol_codes.astype(np.int64),
        symbol_names=tuple(str(s) for s in symbols),
    )
    horizons = [int(c[1:]) for c in SCORE_COLUMNS if c in merged.columns]
    scores = merged[[f"h{h:02d}" for h in horizons]].to_numpy(dtype=np.float64)
    move = Moves(
        growth=growth.astype(np.float64),
        fallback=fallback.astype(np.float64),
        gap=gap.astype(np.float64),
        index_growth=index_growth.astype(np.float64),
    )
    return WalkForwardInputs(
        index=index,
        scores=scores,
        horizons=horizons,
        link=links(index),
        move=move,
        dates=dates,
        raw_close=merged["raw_close"].to_numpy(dtype=np.float64),
        raw_open_1=merged["raw_open_1"].to_numpy(dtype=np.float64),
        dollar_volume=merged["dollar_volume"].to_numpy(dtype=np.float64),
    )


HOLDING_HORIZONS = list(range(1, 15))


def build_holding_returns(
    predictions: pd.DataFrame, prices: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Log return from the next open to the close h sessions later, and its log alpha.

    Row order follows ``predictions``; column h-1 holds horizon h. These are the quantities the
    walk-forward labels are built from, so a top-K batch bought at the next open and sold at the
    close h sessions later (the stage 1 engine) is evaluated on exactly what the model ranked.
    """
    stock_columns = ["symbol", "date", "open_1", *[f"close_{h}" for h in HOLDING_HORIZONS]]
    stocks = prices[prices["asset_type"] != "index"][stock_columns]
    merged = predictions[["symbol", "market", "signal_date"]].merge(
        stocks,
        left_on=["symbol", "signal_date"],
        right_on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    index_frames = []
    for market, symbol in INDEX_SYMBOLS.items():
        rows = prices[(prices["asset_type"] == "index") & (prices["symbol"] == symbol)]
        rows = rows[["date", "open_1", *[f"close_{h}" for h in HOLDING_HORIZONS]]].copy()
        rows["market"] = market
        index_frames.append(rows)
    bench = pd.concat(index_frames, ignore_index=True)
    merged = merged.merge(
        bench,
        left_on=["market", "signal_date"],
        right_on=["market", "date"],
        how="left",
        suffixes=("", "_bench"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entry = merged["open_1"].to_numpy(dtype=np.float64)
        bench_entry = merged["open_1_bench"].to_numpy(dtype=np.float64)
        log_return = np.column_stack(
            [
                np.log(merged[f"close_{h}"].to_numpy(dtype=np.float64) / entry)
                for h in HOLDING_HORIZONS
            ]
        )
        bench_return = np.column_stack(
            [
                np.log(merged[f"close_{h}_bench"].to_numpy(dtype=np.float64) / bench_entry)
                for h in HOLDING_HORIZONS
            ]
        )
    return log_return, log_return - bench_return


# --------------------------------------------------------------------------- ETF benchmark


MAX_ETF_MOVE = 0.5  # a larger open-to-open move means an unadjusted corporate action


def etf_growth(
    prices: pd.DataFrame, dates: pd.DatetimeIndex, symbol: str,
    splits: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Open-to-open total return of the ETF per interval j (open of dates[j+1] to open of
    dates[j+2]).

    The dataset's adjusted prices carry dividends but not stock splits (0050 split 1:4 on
    2025-06-18 and traded nothing on the five sessions before), so ``splits`` lists the
    ratios to apply to the opens before each split date; a session on which the ETF did not
    trade carries the previous open forward (its return lands on the next traded session).
    """
    rows = prices[prices["symbol"] == symbol].drop_duplicates("date").set_index("date")
    adj_open = rows["adj_open"].sort_index().astype(np.float64)
    for item in splits or []:
        day = pd.Timestamp(item["date"])
        adj_open.loc[adj_open.index < day] /= float(item["ratio"])
    opens = adj_open.reindex(dates, method="ffill")
    values = opens.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        growth = values[2:] / values[1:-1] - 1.0
    growth = np.append(growth, 0.0)  # the last interval has no following open in the sample
    missing = int((~np.isfinite(growth)).sum())
    if missing > 0.05 * growth.size:
        raise ValueError(f"{symbol} is missing on {missing} of {growth.size} intervals")
    growth = np.where(np.isfinite(growth), growth, 0.0)
    if np.any(np.abs(growth) > MAX_ETF_MOVE):
        when = dates[1:][np.abs(growth) > MAX_ETF_MOVE][0].date()
        raise ValueError(f"{symbol} moves more than {MAX_ETF_MOVE:.0%} around {when}: an "
                         "unadjusted split or data gap; list it under core.splits")
    return growth
