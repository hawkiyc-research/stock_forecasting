"""Hand-written Alpha158-style feature library stored as Parquet beside each split's bar store.

The feature list is fixed by a JSON spec (``configs/features/alpha_tw_v1.json``). Each split keeps
its own store under ``<prepared>/features/<identity>/`` with the layout of the rank-label store:
shards reuse the bar store's buckets and hold one row group per symbol in the split's canonical
cutoff order, so a reader locates a row by (symbol, ordinal) and checks the cutoff timestamp.

Every feature uses only bars up to and including the signal date. Values are raw (no
normalization); folds normalize them with statistics of their own training window.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data.rank_labels import _bounded_map, _open_source, _worker_count, group_ids

FEATURE_SCHEMA = "alpha-tw-feature-library-v1"
IMPLEMENTATION_FILES = ("data/feature_library.py",)
DEFAULT_SPEC_PATH = Path("configs/features/alpha_tw_v1.json")
ROLLING_OPERATORS = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
    "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
    "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
)
PRICE_FIELDS = ("OPEN", "HIGH", "LOW", "VWAP")
KBAR_FEATURES = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
DOLLAR_VOLUME_RANK = "DVOL_RANK_{window}"
MARKET_INDICATOR = "MARKET_IS_{market}"
# Columns computed from one symbol's history; the rest are cross-sectional or categorical.
STATISTICS_CHUNK_ROWS = 262_144


def implementation_sha256() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode() + b"\0" + (package / relative).read_bytes())
    return digest.hexdigest()


def load_spec(path: str | Path = DEFAULT_SPEC_PATH) -> dict:
    """Read and validate the fixed feature list; unknown operators are refused."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "name", "epsilon", "kbar", "price", "volume", "rolling", "taiwan"}
    if not required <= set(spec) or spec["schema"] != FEATURE_SCHEMA:
        raise ValueError("Feature spec is incomplete or uses an unsupported schema")
    if not isinstance(spec["name"], str) or not spec["name"].isidentifier():
        raise ValueError("Feature spec name must be an identifier")
    if not (isinstance(spec["epsilon"], float) and 0 < spec["epsilon"] < 1e-6):
        raise ValueError("Feature spec epsilon must be a small positive float")
    if list(spec["kbar"]) != list(KBAR_FEATURES):
        raise ValueError("Feature spec kbar block must list the nine candlestick features")
    price, volume, rolling, taiwan = spec["price"], spec["volume"], spec["rolling"], spec["taiwan"]
    if not set(price["fields"]) <= set(PRICE_FIELDS) or not _lags(price["lags"], 0):
        raise ValueError("Feature spec price block has unknown fields or invalid lags")
    if not _lags(volume["lags"], 1):
        raise ValueError("Feature spec volume lags must be distinct positive integers")
    if not set(rolling["operators"]) <= set(ROLLING_OPERATORS) or not _lags(rolling["windows"], 2):
        raise ValueError("Feature spec rolling block has unknown operators or invalid windows")
    if (
        not _lags(taiwan["relative_return_windows"], 1)
        or type(taiwan["dollar_volume_rank_window"]) is not int
        or taiwan["dollar_volume_rank_window"] < 2
        or not isinstance(taiwan["market_indicator"], str)
    ):
        raise ValueError("Feature spec taiwan block is invalid")
    return spec


def _lags(values, minimum: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) > 0
        and all(type(v) is int and v >= minimum for v in values)
        and values == sorted(set(values))
    )


def feature_names(spec: dict) -> list[str]:
    """Column order of the store: kbar, price lags, volume lags, rolling, Taiwan."""
    names = list(spec["kbar"])
    names += [f"{field}{lag}" for field in spec["price"]["fields"] for lag in spec["price"]["lags"]]
    names += [f"VOLUME{lag}" for lag in spec["volume"]["lags"]]
    names += [
        f"{operator}{window}"
        for operator in spec["rolling"]["operators"]
        for window in spec["rolling"]["windows"]
    ]
    taiwan = spec["taiwan"]
    names += [f"REL_RET_{window}" for window in taiwan["relative_return_windows"]]
    names.append(DOLLAR_VOLUME_RANK.format(window=taiwan["dollar_volume_rank_window"]))
    names.append(MARKET_INDICATOR.format(market=taiwan["market_indicator"]))
    if len(set(names)) != len(names):
        raise ValueError("Feature spec produces duplicate column names")
    return names


def spec_identity(spec: dict) -> dict:
    return {"name": spec["name"], "sha256": canonical_json_sha256(spec)}


def feature_identity(source, spec: dict) -> dict:
    return {
        "schema": FEATURE_SCHEMA,
        "source_manifest_sha256": sha256_file(source.root / "bar-store.json"),
        "sample_universe": source.sample_universe_identity,
        "split": source.split,
        "samples": len(source),
        "spec": spec_identity(spec),
        "implementation_sha256": implementation_sha256(),
    }


def feature_root(source, spec: dict) -> tuple[Path, dict]:
    if source.sample_universe_identity is None:
        raise ValueError("Feature stores require a cleaned sample universe")
    identity = feature_identity(source, spec)
    return source.root.parent / "features" / canonical_json_sha256(identity), identity


def _ready(root: Path, identity: dict) -> bool:
    marker = root / "features.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    if payload.get("identity") != identity or payload.get("state") != "ready":
        raise ValueError("Feature store has an inconsistent identity")
    if sha256_file(root / "feature-index.parquet") != payload["feature_index_sha256"]:
        raise ValueError("Feature index is incomplete or corrupted")
    return True


def attach_features(source, spec: dict) -> Path:
    """Locate a ready store for this split; building is an explicit earlier step."""
    root, identity = feature_root(source, spec)
    if not _ready(root, identity):
        raise ValueError(
            f"Features for split={source.split} are missing; build them with ensure_features"
        )
    return root


# --- per-symbol arithmetic -------------------------------------------------------------


def _windows(values: np.ndarray, size: int) -> np.ndarray:
    """Trailing windows of one series as a (T - size + 1, size) view."""
    return np.lib.stride_tricks.sliding_window_view(values, size)


def _pad(values: np.ndarray, length: int, size: int) -> np.ndarray:
    out = np.full(length, np.nan)
    out[size - 1 :] = values
    return out


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if lag < len(values):
        out[lag:] = values[: len(values) - lag]
    return out


def _regression(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling least squares of the window on 0..size-1: slope, R-squared, last residual."""
    length = len(values)
    if length < size:
        empty = np.full(length, np.nan)
        return empty, empty.copy(), empty.copy()
    y = _windows(values, size)
    x = np.arange(size, dtype=np.float64)
    x_centered = x - x.mean()
    y_mean = y.mean(axis=1)
    slope = (y - y_mean[:, None]) @ x_centered / float(x_centered @ x_centered)
    fitted = y_mean[:, None] + slope[:, None] * x_centered[None, :]
    total = ((y - y_mean[:, None]) ** 2).sum(axis=1)
    explained = ((fitted - y_mean[:, None]) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = np.where(total > 0, explained / total, np.nan)
    residual = y[:, -1] - fitted[:, -1]
    return _pad(slope, length, size), _pad(r_squared, length, size), _pad(residual, length, size)


def _rolling_argpos(values: np.ndarray, size: int, *, largest: bool) -> np.ndarray:
    """One-based position of the window's extreme value, counted from the window start."""
    length = len(values)
    if length < size:
        return np.full(length, np.nan)
    windows = _windows(values, size)
    position = (np.argmax if largest else np.argmin)(windows, axis=1) + 1.0
    position[~np.isfinite(windows).all(axis=1)] = np.nan
    return _pad(position, length, size)


def rolling_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    operator: str,
    size: int,
    epsilon: float,
) -> np.ndarray:
    """One Qlib Alpha158 rolling expression on adjusted bars."""
    s_close, s_high, s_low, s_volume = (pd.Series(x) for x in (close, high, low, volume))
    previous_close, previous_volume = _shift(close, 1), _shift(volume, 1)
    if operator == "ROC":
        return _shift(close, size) / close
    if operator == "MA":
        return s_close.rolling(size).mean().to_numpy() / close
    if operator == "STD":
        return s_close.rolling(size).std().to_numpy() / close
    if operator in ("BETA", "RSQR", "RESI"):
        slope, r_squared, residual = _regression(close, size)
        return {"BETA": slope / close, "RSQR": r_squared, "RESI": residual / close}[operator]
    if operator == "MAX":
        return s_high.rolling(size).max().to_numpy() / close
    if operator == "MIN":
        return s_low.rolling(size).min().to_numpy() / close
    if operator == "QTLU":
        return s_close.rolling(size).quantile(0.8).to_numpy() / close
    if operator == "QTLD":
        return s_close.rolling(size).quantile(0.2).to_numpy() / close
    if operator == "RANK":
        return s_close.rolling(size).rank(pct=True).to_numpy()
    if operator == "RSV":
        lowest = s_low.rolling(size).min().to_numpy()
        return (close - lowest) / (s_high.rolling(size).max().to_numpy() - lowest + epsilon)
    if operator == "IMAX":
        return _rolling_argpos(high, size, largest=True) / size
    if operator == "IMIN":
        return _rolling_argpos(low, size, largest=False) / size
    if operator == "IMXD":
        return (
            _rolling_argpos(high, size, largest=True) - _rolling_argpos(low, size, largest=False)
        ) / size
    if operator == "CORR":
        return s_close.rolling(size).corr(pd.Series(np.log1p(volume))).to_numpy()
    if operator == "CORD":
        with np.errstate(invalid="ignore", divide="ignore"):
            price_ratio = close / previous_close
            volume_ratio = np.log1p(volume / previous_volume)
        return pd.Series(price_ratio).rolling(size).corr(pd.Series(volume_ratio)).to_numpy()
    if operator in ("CNTP", "CNTN", "CNTD"):
        up = pd.Series((close > previous_close).astype(np.float64)).rolling(size).mean().to_numpy()
        down = pd.Series((close < previous_close).astype(np.float64)).rolling(size).mean()
        down = down.to_numpy()
        return {"CNTP": up, "CNTN": down, "CNTD": up - down}[operator]
    if operator in ("SUMP", "SUMN", "SUMD"):
        change = close - previous_close
        total = pd.Series(np.abs(change)).rolling(size).sum().to_numpy() + epsilon
        up = pd.Series(np.maximum(change, 0.0)).rolling(size).sum().to_numpy() / total
        down = pd.Series(np.maximum(-change, 0.0)).rolling(size).sum().to_numpy() / total
        return {"SUMP": up, "SUMN": down, "SUMD": up - down}[operator]
    if operator == "VMA":
        return s_volume.rolling(size).mean().to_numpy() / (volume + epsilon)
    if operator == "VSTD":
        return s_volume.rolling(size).std().to_numpy() / (volume + epsilon)
    if operator == "WVMA":
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted = pd.Series(np.abs(close / previous_close - 1.0) * volume)
        return weighted.rolling(size).std().to_numpy() / (
            weighted.rolling(size).mean().to_numpy() + epsilon
        )
    if operator in ("VSUMP", "VSUMN", "VSUMD"):
        change = volume - previous_volume
        total = pd.Series(np.abs(change)).rolling(size).sum().to_numpy() + epsilon
        up = pd.Series(np.maximum(change, 0.0)).rolling(size).sum().to_numpy() / total
        down = pd.Series(np.maximum(-change, 0.0)).rolling(size).sum().to_numpy() / total
        return {"VSUMP": up, "VSUMN": down, "VSUMD": up - down}[operator]
    raise ValueError(f"Unknown rolling operator: {operator}")


def symbol_features(
    bars: dict, benchmark: dict, spec: dict, *, is_indicator_market: bool
) -> np.ndarray:
    """All feature columns for one symbol's full history, in ``feature_names`` order.

    ``bars`` and ``benchmark`` are the dataset's numeric symbol caches (raw OHLCV, adjustment
    factors and timestamps). The dollar-volume column holds the raw rolling mean; the store
    builder replaces it by its within-date, within-market percentile rank.
    """
    raw, epsilon = bars["raw"], float(spec["epsilon"])
    length = len(raw)
    open_, high, low, close = (raw[:, i] * bars["price_factor"] for i in range(4))
    volume = raw[:, 4] * bars["volume_factor"]
    vwap = (high + low + close) / 3.0
    columns: list[np.ndarray] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        span = high - low + epsilon
        body_top, body_bottom = np.maximum(open_, close), np.minimum(open_, close)
        kbar = {
            "KMID": (close - open_) / open_,
            "KLEN": (high - low) / open_,
            "KMID2": (close - open_) / span,
            "KUP": (high - body_top) / open_,
            "KUP2": (high - body_top) / span,
            "KLOW": (body_bottom - low) / open_,
            "KLOW2": (body_bottom - low) / span,
            "KSFT": (2.0 * close - high - low) / open_,
            "KSFT2": (2.0 * close - high - low) / span,
        }
        columns += [kbar[name] for name in spec["kbar"]]
        prices = {"OPEN": open_, "HIGH": high, "LOW": low, "VWAP": vwap}
        for field in spec["price"]["fields"]:
            for lag in spec["price"]["lags"]:
                columns.append(_shift(prices[field], lag) / close)
        for lag in spec["volume"]["lags"]:
            columns.append(_shift(volume, lag) / (volume + epsilon))
        for operator in spec["rolling"]["operators"]:
            for size in spec["rolling"]["windows"]:
                columns.append(
                    rolling_features(open_, high, low, close, volume, operator, size, epsilon)
                )
        positions = benchmark["timestamps"].get_indexer(bars["timestamps"])
        benchmark_close = np.where(
            positions >= 0, benchmark["adjusted_close"][np.maximum(positions, 0)], np.nan
        )
        log_close, log_benchmark = np.log(close), np.log(benchmark_close)
        for size in spec["taiwan"]["relative_return_windows"]:
            columns.append(
                (log_close - _shift(log_close, size))
                - (log_benchmark - _shift(log_benchmark, size))
            )
        dollar_volume = pd.Series(raw[:, 3] * raw[:, 4])
        columns.append(
            dollar_volume.rolling(spec["taiwan"]["dollar_volume_rank_window"]).mean().to_numpy()
        )
    columns.append(np.full(length, 1.0 if is_indicator_market else 0.0))
    return np.column_stack(columns)


def percentile_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """(average rank - 0.5) / n within each group; non-finite values stay NaN."""
    frame = pd.DataFrame({"value": values, "group": groups})
    finite = np.isfinite(values)
    ranks = frame.loc[finite].groupby("group")["value"].rank(method="average")
    sizes = frame.loc[finite].groupby("group")["value"].transform("size")
    out = np.full(len(values), np.nan)
    out[finite] = ((ranks - 0.5) / sizes).to_numpy()
    return out


# --- bounded build ------------------------------------------------------------------------


def _bucket_features(task):
    """Write one bucket's rows straight into the shared scratch arrays."""
    bucket, spans, work, spec, columns, indicator_market = task
    from stock_forecasting.data.rank_labels import _SOURCE as source

    features = np.load(Path(work) / "features.npy", mmap_mode="r+")
    cutoff = np.load(Path(work) / "cutoff_ns.npy", mmap_mode="r+")
    covered = []
    for symbol, begin, end in spans:
        row = source._index[symbol]
        asset = source._numeric_symbol(symbol)
        benchmark = source._numeric_symbol(str(row["benchmark_symbol"]))
        ordinals = np.arange(begin, end)
        ranges = np.searchsorted(source._range_ends, ordinals, side="right")
        cutoffs = source._range_starts[ranges] + (ordinals - source._range_begins[ranges])
        values = symbol_features(
            asset, benchmark, spec, is_indicator_market=str(row["market"]) == indicator_market
        )
        if values.shape[1] != columns:
            raise RuntimeError("Feature spec and computed columns disagree")
        features[begin:end] = values[cutoffs].astype(np.float32)
        cutoff[begin:end] = asset["timestamps"].asi8[cutoffs]
        covered.append((begin, end))
    features.flush()
    cutoff.flush()
    source.close()
    return bucket, covered


def _write_bucket(task):
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, spans, work, output, names = task
    features = np.load(Path(work) / "features.npy", mmap_mode="r")
    cutoff = np.load(Path(work) / "cutoff_ns.npy", mmap_mode="r")
    relative = f"shards/bucket-{bucket:04d}/shard.parquet"
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    schema = pa.schema(
        [("timestamp", pa.timestamp("ns", tz="UTC"))] + [(name, pa.float32()) for name in names]
    )
    index = []
    with pq.ParquetWriter(pending, schema) as writer:
        for group, (symbol, begin, end) in enumerate(spans):
            stamps = pa.array(np.asarray(cutoff[begin:end]), type=pa.int64())
            block = np.ascontiguousarray(features[begin:end])
            arrays = [stamps.cast(schema[0].type)]
            arrays += [pa.array(block[:, i]) for i in range(len(names))]
            writer.write_table(
                pa.Table.from_arrays(arrays, schema=schema), row_group_size=max(1, end - begin)
            )
            index.append(
                {
                    "symbol": symbol,
                    "shard_relative_path": relative,
                    "row_group": group,
                    "row_count": end - begin,
                    "ordinal_begin": begin,
                }
            )
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    pending.replace(path)
    return relative, {"bytes": path.stat().st_size, "sha256": sha256_file(path)}, index


def _column_statistics(features: np.ndarray, names: list[str]) -> dict:
    """Finite share and mean per column, streamed in chunks over the scratch array."""
    count = features.shape[0]
    finite = np.zeros(len(names), dtype=np.int64)
    total = np.zeros(len(names), dtype=np.float64)
    for start in range(0, count, STATISTICS_CHUNK_ROWS):
        block = np.asarray(features[start : start + STATISTICS_CHUNK_ROWS], dtype=np.float64)
        mask = np.isfinite(block)
        finite += mask.sum(axis=0)
        total += np.where(mask, block, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite > 0, total / finite, np.nan)
    return {
        name: {
            "finite_fraction": float(finite[i] / count) if count else 0.0,
            "mean": None if not np.isfinite(mean[i]) else float(mean[i]),
        }
        for i, name in enumerate(names)
    }


def ensure_features(source, spec: dict, *, workers: int | None = None) -> Path:
    """Build one split's feature store with bounded process pools; reuse it when ready."""
    from stock_forecasting.data.sample_universe import _exclusive_build

    root, identity = feature_root(source, spec)
    if _ready(root, identity):
        return root
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_build(root):
        if _ready(root, identity):
            return root
        started = time.monotonic()
        names = feature_names(spec)
        count = len(source)
        spans = source.symbol_spans()
        markets = sorted({str(source._index[symbol]["market"]) for symbol in spans})
        buckets: dict[int, list] = {}
        for symbol, (begin, end) in sorted(spans.items(), key=lambda item: item[1][0]):
            buckets.setdefault(int(source._index[symbol]["bucket"]), []).append(
                (symbol, begin, end)
            )
        work = root / "work"
        work.mkdir(exist_ok=True)
        features = np.lib.format.open_memmap(
            work / "features.npy", mode="w+", dtype="float32", shape=(count, len(names))
        )
        cutoff = np.lib.format.open_memmap(
            work / "cutoff_ns.npy", mode="w+", dtype="int64", shape=(count,)
        )
        features.flush()
        cutoff.flush()
        del features, cutoff
        # Each worker holds one symbol's full history of float64 columns plus its windows.
        longest = max(int(source._index[symbol]["row_count"]) for symbol in spans)
        compute_workers, compute_plan = _worker_count(
            workers, len(buckets), 768 * 1024**2 + longest * len(names) * 8 * 4
        )
        print(
            f"Features {source.split}: {len(names)} columns for {count} samples in "
            f"{len(buckets)} buckets, workers={compute_workers}",
            flush=True,
        )
        covered = 0
        indicator_market = str(spec["taiwan"]["market_indicator"])
        for _bucket, rows in _bounded_map(
            _bucket_features,
            [
                (bucket, rows, str(work), spec, len(names), indicator_market)
                for bucket, rows in sorted(buckets.items())
            ],
            compute_workers,
            _open_source,
            (source.__getstate__(),),
        ):
            covered += sum(end - begin for begin, end in rows)
        if covered != count:
            raise RuntimeError("Feature rows did not cover every sample exactly once")
        features = np.load(work / "features.npy", mmap_mode="r+")
        cutoff = np.load(work / "cutoff_ns.npy", mmap_mode="r")
        market_codes = np.empty(count, dtype=np.int64)
        for symbol, (begin, end) in spans.items():
            market_codes[begin:end] = markets.index(str(source._index[symbol]["market"]))
        groups = group_ids(np.asarray(cutoff), market_codes)
        rank_column = names.index(
            DOLLAR_VOLUME_RANK.format(window=spec["taiwan"]["dollar_volume_rank_window"])
        )
        features[:, rank_column] = percentile_rank(
            np.asarray(features[:, rank_column], dtype=np.float64), groups
        ).astype(np.float32)
        features.flush()
        statistics = _column_statistics(features, names)
        del features, cutoff, groups
        write_workers, write_plan = _worker_count(workers, len(buckets), 512 * 1024**2)
        files, index = {}, []
        for relative, metadata, rows in _bounded_map(
            _write_bucket,
            [
                (bucket, rows, str(work), str(root), names)
                for bucket, rows in sorted(buckets.items())
            ],
            write_workers,
        ):
            files[relative] = metadata
            index.extend(rows)
        frame = pd.DataFrame(index).sort_values("ordinal_begin", kind="stable")
        pending = root / "feature-index.pending.parquet"
        frame.to_parquet(pending, index=False)
        pending.replace(root / "feature-index.parquet")
        atomic_write_json(
            root / "features.json",
            {
                "state": "ready",
                "identity": identity,
                "feature_names": names,
                "feature_index_sha256": sha256_file(root / "feature-index.parquet"),
                "files": files,
                "markets": markets,
                "columns": statistics,
                "execution": {
                    "compute_workers": compute_workers,
                    "compute_plan": compute_plan,
                    "write_workers": write_workers,
                    "write_plan": write_plan,
                    "seconds": time.monotonic() - started,
                },
            },
        )
        for name in ("features.npy", "cutoff_ns.npy"):
            (work / name).unlink(missing_ok=True)
        print(f"Features {source.split}: ready at {root}", flush=True)
    return root


def read_split_features(root: Path) -> pd.DataFrame:
    """Load one split's store as a frame in ordinal order (for CPU models such as GBDT)."""
    import pyarrow.parquet as pq

    payload = json.loads((root / "features.json").read_text())
    if payload.get("state") != "ready":
        raise ValueError("Feature store is not ready")
    index = pd.read_parquet(root / "feature-index.parquet").sort_values("ordinal_begin")
    frames = []
    for relative, rows in index.groupby("shard_relative_path", sort=False):
        handle = pq.ParquetFile(root / relative)
        for row in rows.itertuples():
            table = handle.read_row_group(int(row.row_group)).to_pandas()
            table.insert(0, "symbol", row.symbol)
            begin = int(row.ordinal_begin)
            table.insert(0, "ordinal", np.arange(begin, begin + len(table)))
            frames.append(table)
    frame = pd.concat(frames, ignore_index=True).sort_values("ordinal", kind="stable")
    return frame.reset_index(drop=True)
