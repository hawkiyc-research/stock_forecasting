"""Trailing-return, beta and turnover features, and score purification helpers.

Shared by the reversal-exposure diagnostic (protocol 13.19 supplement 1) and the purified band
construction of the scenario suite (supplement 2). Every feature uses data up to the signal date
only: trailing 20- and 60-session log returns of the adjusted close, a 60-session rolling beta of
the stock's daily log return on its market index, and the turnover spike (5-session mean dollar
volume over the 60-session median).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BETA_WINDOW = 60
MIN_BETA_OBSERVATIONS = 40
TURNOVER_SHORT, TURNOVER_LONG = 5, 60
INDEX_SYMBOLS = {"TAIEX.TW": "TWSE", "TPEX.TWO": "TPEX"}
FEATURES = ("ret20", "ret60", "beta")


def trailing_features(prices_path: Path) -> pd.DataFrame:
    """Per (symbol, date): trailing 20/60-session log returns of the adjusted close, the 60-session
    beta to the market index and the turnover spike, all from data up to that date.

    Reads the raw parquet itself because ``load_prices`` drops the volume column.
    """
    frame = pd.read_parquet(
        prices_path,
        columns=["timestamp", "symbol", "asset_type", "market", "close", "adjusted_close",
                 "volume"],
    )
    frame = frame[frame["market"].isin(set(INDEX_SYMBOLS.values()))].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    index_rows = frame[(frame["asset_type"] == "index") & frame["symbol"].isin(INDEX_SYMBOLS)]
    index_rows = index_rows[["symbol", "date", "adjusted_close"]].copy()
    index_rows["market"] = index_rows["symbol"].map(INDEX_SYMBOLS)
    index_rows = index_rows.drop_duplicates(["market", "date"]).sort_values(["market", "date"])
    index_rows["index_ret"] = np.log(
        index_rows["adjusted_close"] / index_rows.groupby("market")["adjusted_close"].shift(1)
    )
    stocks = frame[frame["asset_type"] != "index"].drop_duplicates(["symbol", "date"])
    stocks = stocks.merge(index_rows[["market", "date", "index_ret"]], on=["market", "date"],
                          how="left")
    stocks = stocks.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = stocks.groupby("symbol", sort=False)
    close = stocks["adjusted_close"]
    stocks["ret1"] = np.log(close / grouped["adjusted_close"].shift(1))
    stocks["ret20"] = np.log(close / grouped["adjusted_close"].shift(20))
    stocks["ret60"] = np.log(close / grouped["adjusted_close"].shift(60))
    # Rolling beta from rolling moments, per symbol.
    stocks["rm"] = stocks["ret1"] * stocks["index_ret"]
    stocks["mm"] = stocks["index_ret"] ** 2

    def rolling(column: str, window: int, min_periods: int, how: str) -> pd.Series:
        by_symbol = stocks.groupby("symbol", sort=False)[column]
        roller = by_symbol.rolling(window, min_periods=min_periods)
        return getattr(roller, how)().reset_index(level=0, drop=True)

    e_r, e_m, e_rm, e_mm = (
        rolling(c, BETA_WINDOW, MIN_BETA_OBSERVATIONS, "mean")
        for c in ("ret1", "index_ret", "rm", "mm")
    )
    var_m = e_mm - e_m**2
    with np.errstate(divide="ignore", invalid="ignore"):
        stocks["beta"] = np.where(var_m > 0, (e_rm - e_r * e_m) / var_m, np.nan)
    stocks["traded"] = stocks["close"] * stocks["volume"]
    short = rolling("traded", TURNOVER_SHORT, TURNOVER_SHORT, "mean")
    long = rolling("traded", TURNOVER_LONG, TURNOVER_LONG // 2, "median")
    with np.errstate(divide="ignore", invalid="ignore"):
        stocks["turnover_spike"] = np.log(short / long)
    return stocks[["symbol", "date", "ret20", "ret60", "beta", "turnover_spike"]]


def row_features(predictions: pd.DataFrame, features: pd.DataFrame) -> dict[str, np.ndarray]:
    """The feature table aligned with the prediction rows (symbol, signal date)."""
    merged = predictions[["symbol", "signal_date"]].merge(
        features, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left",
        validate="one_to_one",
    )
    return {name: merged[name].to_numpy(dtype=np.float64)
            for name in ("ret20", "ret60", "beta", "turnover_spike")}


def orthogonalize(score: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """OLS residual of ``score`` on an intercept and ``regressors`` [n, k]; NaN where any input is
    missing. With too few complete rows the residual is NaN everywhere."""
    out = np.full(score.shape[0], np.nan)
    ok = np.isfinite(score) & np.all(np.isfinite(regressors), axis=1)
    if ok.sum() <= regressors.shape[1] + 1:
        return out
    design = np.column_stack([np.ones(int(ok.sum())), regressors[ok]])
    coefficients, *_ = np.linalg.lstsq(design, score[ok], rcond=None)
    out[ok] = score[ok] - design @ coefficients
    return out


def orthogonalize_by_group(
    score: np.ndarray, regressors: np.ndarray, groups: tuple[np.ndarray, ...]
) -> np.ndarray:
    """``orthogonalize`` applied inside every date group (rows outside the groups stay NaN)."""
    out = np.full(score.shape[0], np.nan)
    for rows in groups:
        if rows.size:
            out[rows] = orthogonalize(score[rows], regressors[rows])
    return out


def quantile_labels(values: np.ndarray, bins: int) -> np.ndarray:
    """Balanced quantile bins (0..bins-1) of a finite vector; equal values share a bin."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=np.int64)
    labels[order] = (np.arange(values.size) * bins) // max(values.size, 1)
    return labels
