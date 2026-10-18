"""Taiwan research panel: past-only features, next-open labels and auxiliary joins.

Rows are exactly the cleaned training/evaluation samples of the experiment data
group, restricted to Taiwan markets. Features use bars up to and including the
cutoff session; labels enter at the next session's adjusted open and exit at the
adjusted close h sessions later, as in ``execution_total_return``.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.adjustments import ADJUSTED_CLOSE_FIELD

TECHNICAL = (
    "return_1d", "return_5d", "return_20d", "return_window", "volatility_20d", "atr_14d",
    "volume_zscore_20d", "close_vs_sma20", "close_vs_sma50", "rsi_14d",
)
SHORT_EXCLUDED = ("return_window", "close_vs_sma50")
LONG = (
    "return_60d", "return_120d", "return_250d", "close_vs_sma100", "close_vs_sma200",
    "from_high_250d", "from_low_250d",
)
FLOW = (
    "foreign_ratio_1d", "foreign_ratio_5d", "foreign_ratio_20d", "trust_ratio_1d",
    "trust_ratio_5d", "trust_ratio_20d", "dealer_ratio_1d", "dealer_ratio_5d",
    "institutional_ratio_1d", "institutional_ratio_5d", "margin_change_1d", "margin_change_5d",
    "margin_utilization", "short_change_1d", "short_change_5d", "short_margin_ratio",
    "margin_buy_ratio", "short_sell_ratio",
)
MARKET = (
    "agg_foreign_ratio_1d", "agg_foreign_ratio_5d", "agg_trust_ratio_1d", "agg_trust_ratio_5d",
    "agg_institutional_ratio_1d", "agg_institutional_ratio_5d", "agg_margin_change_1d",
    "agg_short_margin_ratio", "breadth_up_1d", "breadth_above_sma20", "mean_return_1d",
    "dispersion_1d",
)
TAIFEX = (
    "futures_foreign_net_oi", "futures_foreign_net_oi_change_1d",
    "futures_foreign_net_oi_change_5d", "futures_trust_net_oi", "futures_dealer_net_oi",
    "futures_foreign_net_trade", "put_call_volume_ratio", "put_call_oi_ratio",
    "put_call_oi_ratio_change_5d",
)
RAW_COLUMNS = ("close_raw", "volume_raw", "foreign_net_shares", "trust_net_shares",
               "dealer_net_shares", "institutional_net_shares")


def adjusted_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Total-return-adjusted OHLC with raw volume, one row per session."""
    bars = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    factor = bars[ADJUSTED_CLOSE_FIELD].astype(float) / bars["close"].astype(float).clip(1e-12)
    return pd.DataFrame({
        "date": pd.to_datetime(bars["timestamp"], utc=True).dt.tz_localize(None).dt.normalize(),
        "open": bars["open"].astype(float) * factor,
        "high": bars["high"].astype(float) * factor,
        "low": bars["low"].astype(float) * factor,
        "close": bars[ADJUSTED_CLOSE_FIELD].astype(float),
        "close_raw": bars["close"].astype(float),
        "volume": bars["volume"].astype(float),
    })


def technical_frame(bars: pd.DataFrame, *, window: int, long_lookback: bool) -> pd.DataFrame:
    """Past-only indicators; every value at row t uses rows up to t only."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    log_close = np.log(close)
    daily = log_close.diff()
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    log_volume = np.log1p(bars["volume"])
    gains, losses = daily.clip(lower=0), (-daily).clip(lower=0)
    average_gain, average_loss = gains.rolling(14).mean(), losses.rolling(14).mean()
    output = pd.DataFrame({
        "return_1d": daily,
        "return_5d": log_close.diff(5),
        "return_20d": log_close.diff(20),
        "return_window": log_close.diff(window - 1),
        "volatility_20d": daily.rolling(20).std() * np.sqrt(252.0),
        "atr_14d": true_range.rolling(14).mean() / close,
        "volume_zscore_20d": (log_volume - log_volume.rolling(20).mean())
        / log_volume.rolling(20).std(),
        "close_vs_sma20": close / close.rolling(20).mean() - 1.0,
        "close_vs_sma50": close / close.rolling(50).mean() - 1.0,
        "rsi_14d": 100.0 - 100.0 / (1.0 + average_gain / average_loss.replace(0.0, np.nan)),
        "sigma_20d": daily.rolling(20).std(),
    })
    if long_lookback:
        output["return_60d"] = log_close.diff(60)
        output["return_120d"] = log_close.diff(120)
        output["return_250d"] = log_close.diff(250)
        output["close_vs_sma100"] = close / close.rolling(100).mean() - 1.0
        output["close_vs_sma200"] = close / close.rolling(200).mean() - 1.0
        output["from_high_250d"] = close / close.rolling(250).max() - 1.0
        output["from_low_250d"] = close / close.rolling(250).min() - 1.0
    return output.replace([np.inf, -np.inf], np.nan)


def next_open_log_returns(bars: pd.DataFrame, horizons: tuple[int, ...]) -> np.ndarray:
    """Log return from the open of session t+1 to the close of session t+h."""
    opens, closes = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    count = len(bars)
    result = np.full((count, len(horizons)), np.nan)
    for column, horizon in enumerate(horizons):
        last = count - horizon
        if last <= 0:
            continue
        result[:last, column] = np.log(closes[horizon:horizon + last] / opens[1:1 + last])
    return result


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def flow_features(bars: pd.DataFrame, flows: pd.DataFrame | None) -> pd.DataFrame:
    """Stock-level institutional and margin features aligned to the symbol's sessions."""
    columns = {name: np.nan for name in (*FLOW, *RAW_COLUMNS[2:])}
    if flows is None or flows.empty:
        return pd.DataFrame(columns, index=bars.index)
    aligned = flows.set_index("date").reindex(bars["date"]).reset_index(drop=True)
    volume = bars["volume"].reset_index(drop=True)
    output = pd.DataFrame(index=bars.index)
    for prefix, column in (
        ("foreign", "foreign_net_shares"),
        ("trust", "trust_net_shares"),
        ("dealer", "dealer_net_shares"),
        ("institutional", "institutional_net_shares"),
    ):
        net = aligned.get(column, pd.Series(np.nan, index=aligned.index))
        output[column] = net.to_numpy()
        output[f"{prefix}_ratio_1d"] = _ratio(net, volume).to_numpy()
        output[f"{prefix}_ratio_5d"] = _ratio(
            net.rolling(5).sum(), volume.rolling(5).sum()
        ).to_numpy()
        if prefix in {"foreign", "trust"}:
            output[f"{prefix}_ratio_20d"] = _ratio(
                net.rolling(20).sum(), volume.rolling(20).sum()
            ).to_numpy()
    margin = aligned.get("margin_balance_lots", pd.Series(np.nan, index=aligned.index))
    short = aligned.get("short_balance_lots", pd.Series(np.nan, index=aligned.index))
    limit = aligned.get("margin_limit_lots", pd.Series(np.nan, index=aligned.index))
    output["margin_change_1d"] = (_ratio(margin, margin.shift(1)) - 1.0).to_numpy()
    output["margin_change_5d"] = (_ratio(margin, margin.shift(5)) - 1.0).to_numpy()
    output["margin_utilization"] = _ratio(margin, limit).to_numpy()
    output["short_change_1d"] = (_ratio(short, short.shift(1)) - 1.0).to_numpy()
    output["short_change_5d"] = (_ratio(short, short.shift(5)) - 1.0).to_numpy()
    output["short_margin_ratio"] = _ratio(short, margin).to_numpy()
    output["margin_buy_ratio"] = _ratio(
        aligned.get("margin_buy_lots", pd.Series(np.nan, index=aligned.index)) * 1000.0, volume
    ).to_numpy()
    output["short_sell_ratio"] = _ratio(
        aligned.get("short_sell_lots", pd.Series(np.nan, index=aligned.index)) * 1000.0, volume
    ).to_numpy()
    return output.replace([np.inf, -np.inf], np.nan)


def symbol_rows(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    cutoffs: np.ndarray,
    *,
    horizons: tuple[int, ...],
    window: int,
    flows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Features and labels of one symbol at the given cutoff row positions."""
    bars, bench_bars = adjusted_bars(frame), adjusted_bars(benchmark)
    technical = technical_frame(bars, window=window, long_lookback=True)
    bench = technical_frame(bench_bars, window=window, long_lookback=False)[list(TECHNICAL)]
    bench.index = bench_bars["date"]
    bench = bench.reindex(bars["date"]).reset_index(drop=True).add_prefix("bench_")
    asset_returns = next_open_log_returns(bars, horizons)
    aligned_bench = bench_bars.set_index("date").reindex(bars["date"]).reset_index(drop=True)
    bench_returns = next_open_log_returns(aligned_bench, horizons)
    output = pd.concat(
        [bars[["date", "close_raw"]], technical, bench, flow_features(bars, flows)], axis=1
    )
    output["volume_raw"] = bars["volume"]
    for column, horizon in enumerate(horizons):
        output[f"return_{horizon}"] = asset_returns[:, column]
        output[f"alpha_{horizon}"] = asset_returns[:, column] - bench_returns[:, column]
    selected = output.iloc[np.asarray(cutoffs, dtype=np.int64)].reset_index(drop=True)
    numeric = selected.columns.difference(["date"])
    selected[numeric] = selected[numeric].astype(np.float32)
    return selected


_WORKER: dict[str, Any] = {}


def _initialize_worker(experiment_config: str, roots: dict[str, str]) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    from stock_forecasting.config import ExperimentConfig

    _WORKER.update(config=ExperimentConfig.from_yaml(experiment_config), roots=roots, datasets={})


def _dataset(split: str):
    from stock_forecasting.data.sample_universe import open_clean_dataset

    datasets = _WORKER["datasets"]
    if split not in datasets:
        datasets[split] = open_clean_dataset(
            _WORKER["config"], split, validated_root=_WORKER["roots"][split], symbol_cache_size=4
        )
    return datasets[split]


def _symbol_task(task: tuple[Any, ...]) -> pd.DataFrame:
    split, symbol, cutoffs, flows, horizons, window = task
    dataset = _dataset(split)
    row = dataset._index[symbol]
    frame = dataset._load_symbol(symbol)
    benchmark = dataset._load_symbol(str(row["benchmark_symbol"]))
    result = symbol_rows(frame, benchmark, cutoffs, horizons=horizons, window=window, flows=flows)
    result.insert(0, "split", split)
    result.insert(2, "symbol", symbol)
    result.insert(3, "market", str(row["market"]))
    result.insert(4, "asset_type", str(row["asset_type"]))
    return result


def enumerate_cutoffs(experiment_config: str, markets: tuple[str, ...]) -> tuple[dict, dict]:
    """Return {split: {symbol: cutoff positions}} for Taiwan samples and the split roots."""
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.sample_universe import open_clean_dataset, validated_split_roots

    config = ExperimentConfig.from_yaml(experiment_config)
    roots = {split: str(root) for split, root in validated_split_roots(config).items()}
    cutoffs: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        dataset = open_clean_dataset(config, split, validated_root=roots[split])
        try:
            per_symbol: dict[str, list[np.ndarray]] = {}
            for row in dataset.ranges.itertuples(index=False):
                symbol = str(row.symbol)
                if str(dataset._index[symbol]["market"]) not in markets:
                    continue
                per_symbol.setdefault(symbol, []).append(
                    np.arange(int(row.start_index), int(row.stop_index), dtype=np.int64)
                )
            cutoffs[split] = {s: np.concatenate(parts) for s, parts in per_symbol.items()}
        finally:
            dataset.close()
    return cutoffs, roots


def build_panel(
    experiment_config: str,
    *,
    markets: tuple[str, ...],
    horizons: tuple[int, ...],
    window: int,
    flows_by_symbol: dict[str, pd.DataFrame],
    workers: int,
) -> pd.DataFrame:
    """Compute every Taiwan sample with a bounded spawn pool (one symbol per task)."""
    cutoffs, roots = enumerate_cutoffs(experiment_config, markets)
    tasks = (
        (split, symbol, positions, flows_by_symbol.get(symbol), horizons, window)
        for split, symbols in cutoffs.items()
        for symbol, positions in symbols.items()
    )
    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(str(Path(experiment_config).resolve()), roots),
    ) as pool:
        pending = set()
        for task in tasks:
            pending.add(pool.submit(_symbol_task, task))
            # Backpressure: never hold more than two tasks per worker in flight.
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                parts.extend(future.result() for future in done)
        parts.extend(future.result() for future in pending)
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["split", "date", "symbol"], kind="stable").reset_index(drop=True)


def market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Date x market aggregates from same-session values only."""
    value = panel["volume_raw"].astype(float) * panel["close_raw"].astype(float)
    frame = panel[["date", "market"]].copy()
    for prefix, column in (
        ("foreign", "foreign_net_shares"),
        ("trust", "trust_net_shares"),
        ("institutional", "institutional_net_shares"),
    ):
        frame[f"{prefix}_value"] = panel[column].astype(float) * panel["close_raw"].astype(float)
    frame["traded_value"] = value
    frame["up"] = (panel["return_1d"] > 0).astype(float)
    frame["above_sma20"] = (panel["close_vs_sma20"] > 0).astype(float)
    frame["return_1d"] = panel["return_1d"].astype(float)
    frame["margin_change_1d"] = panel["margin_change_1d"].astype(float)
    frame["short_margin_ratio"] = panel["short_margin_ratio"].astype(float)
    # One row per stock and session even if the same session appears in two splits.
    frame = frame.assign(symbol=panel["symbol"]).drop_duplicates(["date", "market", "symbol"])
    grouped = frame.groupby(["market", "date"], sort=True)
    sums = grouped[["foreign_value", "trust_value", "institutional_value", "traded_value"]].sum(
        min_count=1
    )
    result = pd.DataFrame(index=sums.index)
    for prefix in ("foreign", "trust", "institutional"):
        result[f"agg_{prefix}_ratio_1d"] = sums[f"{prefix}_value"] / sums["traded_value"]
        result[f"agg_{prefix}_ratio_5d"] = (
            result[f"agg_{prefix}_ratio_1d"].groupby(level="market").transform(
                lambda series: series.rolling(5).mean()
            )
        )
    result["agg_margin_change_1d"] = grouped["margin_change_1d"].median()
    result["agg_short_margin_ratio"] = grouped["short_margin_ratio"].median()
    result["breadth_up_1d"] = grouped["up"].mean()
    result["breadth_above_sma20"] = grouped["above_sma20"].mean()
    result["mean_return_1d"] = grouped["return_1d"].mean()
    result["dispersion_1d"] = grouped["return_1d"].std()
    return result.reset_index()


def taifex_features(futures: pd.DataFrame | None, put_call: pd.DataFrame | None) -> pd.DataFrame:
    """Date-level TAIFEX positioning; columns are absent when a source is missing."""
    frames = []
    if futures is not None and not futures.empty:
        pivot = futures.pivot_table(
            index="timestamp",
            columns="investor",
            values=["net_open_interest", "net_trade_contracts"],
            aggfunc="sum",
        ).sort_index()

        def column(value: str, investor: str) -> pd.Series:
            return pivot.get((value, investor), pd.Series(np.nan, index=pivot.index))

        table = pd.DataFrame(index=pivot.index)
        foreign = column("net_open_interest", "foreign")
        table["futures_foreign_net_oi"] = foreign
        table["futures_foreign_net_oi_change_1d"] = foreign.diff(1)
        table["futures_foreign_net_oi_change_5d"] = foreign.diff(5)
        table["futures_trust_net_oi"] = column("net_open_interest", "trust")
        table["futures_dealer_net_oi"] = column("net_open_interest", "dealer")
        table["futures_foreign_net_trade"] = column("net_trade_contracts", "foreign")
        frames.append(table)
    if put_call is not None and not put_call.empty:
        ratio = put_call.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        table = pd.DataFrame(index=ratio.index)
        table["put_call_volume_ratio"] = ratio["volume_ratio_percent"] / 100.0
        table["put_call_oi_ratio"] = ratio["open_interest_ratio_percent"] / 100.0
        table["put_call_oi_ratio_change_5d"] = table["put_call_oi_ratio"].diff(5)
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=["date"])
    combined = pd.concat(frames, axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index).normalize()
    return combined.rename_axis("date").reset_index()


def us_daily(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-day and five-day log returns per US session date and symbol."""
    columns = []
    for symbol, frame in series.items():
        ordered = frame.sort_values("date")
        log_close = np.log(ordered.set_index("date")["close"].astype(float))
        name = symbol.replace(".", "_").lower()
        columns.append(log_close.diff(1).rename(f"{name}_r1"))
        columns.append(log_close.diff(5).rename(f"{name}_r5"))
    if not columns:
        return pd.DataFrame(columns=["date"])
    return pd.concat(columns, axis=1).sort_index().rename_axis("date").reset_index()


def align_us(dates: pd.Series, us: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    """US values known at the Taiwan signal time for each Taiwan session date.

    pre_open (next morning): the US session dated t has closed, so dates <= t.
    evening (same night): only US sessions dated before t have closed.
    """
    if variant not in {"pre_open", "evening"}:
        raise ValueError("US alignment variant must be pre_open or evening")
    left = pd.DataFrame({"date": pd.to_datetime(dates).values}).reset_index()
    if us.empty or len(us.columns) == 1:
        return pd.DataFrame(index=range(len(left)))
    right = us.sort_values("date").assign(date=lambda frame: pd.to_datetime(frame["date"]))
    merged = pd.merge_asof(
        left.sort_values("date"), right, on="date", direction="backward",
        allow_exact_matches=variant == "pre_open",
    ).sort_values("index")
    values = merged.drop(columns=["index", "date"]).reset_index(drop=True)
    return values.add_prefix(f"us_{variant}_")
