"""Cross-stock sample cache (protocol 13.13): one flat store per split for GPU training.

The cache holds, per sample, the 128-session adjusted OHLCV window the Kronos path saw
(``LazyFinancialWindowDataset._numeric_context``), the feature-library row, the gauss-rank labels
and the raw alpha targets, as memory-mapped ``.npy`` arrays beside the split's bar store. It is
built once on bounded spawn workers and reused by every fold, seed and ablation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_forecasting.data.feature_library import (
    attach_features,
    feature_identity,
    feature_names,
    read_split_features,
)
from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data.rank_labels import (
    _bounded_map,
    _open_source,
    _worker_count,
    attach_rank_labels,
    label_identity,
)

CACHE_SCHEMA = "cross-stock-sample-cache-v1"
ARRAYS = ("windows", "features", "ranks", "alpha", "cutoff_ns", "market_codes", "symbol_codes")


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cache_identity(source, spec: dict, min_group_size: int) -> dict:
    return {
        "schema": CACHE_SCHEMA,
        "window": int(source.window_size),
        "features": feature_identity(source, spec),
        "labels": label_identity(source, min_group_size),
        "implementation_sha256": implementation_sha256(),
    }


def cache_root(source, spec: dict, min_group_size: int) -> tuple[Path, dict]:
    identity = cache_identity(source, spec, min_group_size)
    return source.root.parent / "xs-cache" / canonical_json_sha256(identity), identity


def _ready(root: Path, identity: dict) -> bool:
    marker = root / "xs-cache.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    if payload.get("identity") != identity or payload.get("state") != "ready":
        raise ValueError("Cross-stock cache has an inconsistent identity")
    for name in ARRAYS:
        if not (root / f"{name}.npy").is_file():
            raise ValueError(f"Cross-stock cache is missing {name}.npy")
    return True


def _bucket_windows(task):
    """Write one bucket's adjusted windows straight into the shared memmap."""
    bucket, spans, work = task
    from stock_forecasting.data.rank_labels import _SOURCE as source

    windows = np.load(Path(work) / "windows.npy", mmap_mode="r+")
    covered = []
    for symbol, begin, end in spans:
        asset = source._numeric_symbol(symbol)
        ordinals = np.arange(begin, end)
        ranges = np.searchsorted(source._range_ends, ordinals, side="right")
        cutoffs = source._range_starts[ranges] + (ordinals - source._range_begins[ranges])
        context = cutoffs[:, None] + np.arange(1 - source.window_size, 1)
        if (context < 0).any():
            raise RuntimeError(f"Prepared cutoff index is outside the symbol history: {symbol}")
        windows[begin:end] = source._numeric_context(asset, context).astype(np.float32)
        covered.append((begin, end))
    windows.flush()
    source.close()
    return bucket, covered


def ensure_xs_cache(source, spec: dict, min_group_size: int, *, workers: int | None = None) -> Path:
    """Build the split's cache from its ready feature and label stores; reuse it when ready."""
    from stock_forecasting.data.sample_universe import _exclusive_build
    from stock_forecasting.walk_forward_lgbm import read_split_rank_labels, split_targets

    root, identity = cache_root(source, spec, min_group_size)
    if _ready(root, identity):
        return root
    feature_store = attach_features(source, spec)
    label_store = attach_rank_labels(source, min_group_size)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_build(root):
        if _ready(root, identity):
            return root
        started = time.monotonic()
        count, horizons, window = len(source), list(source.horizons), int(source.window_size)
        spans = source.symbol_spans()
        buckets: dict[int, list] = {}
        for symbol, (begin, end) in sorted(spans.items(), key=lambda item: item[1][0]):
            buckets.setdefault(int(source._index[symbol]["bucket"]), []).append(
                (symbol, begin, end)
            )
        pending = root / "pending"
        pending.mkdir(exist_ok=True)
        windows = np.lib.format.open_memmap(
            pending / "windows.npy", mode="w+", dtype="float32", shape=(count, window, 5)
        )
        del windows
        largest = max(sum(end - begin for _, begin, end in rows) for rows in buckets.values())
        window_workers, plan = _worker_count(
            workers, len(buckets), 768 * 1024**2 + largest * window * 5 * 8 * 2
        )
        print(
            f"Cross-stock cache {source.split}: windows for {count} samples in {len(buckets)} "
            f"buckets, workers={window_workers}",
            flush=True,
        )
        covered = 0
        for _bucket, rows in _bounded_map(
            _bucket_windows,
            [(bucket, rows, str(pending)) for bucket, rows in sorted(buckets.items())],
            window_workers,
            _open_source,
            (source.__getstate__(),),
        ):
            covered += sum(end - begin for begin, end in rows)
        if covered != count:
            raise RuntimeError("Cross-stock windows did not cover every sample exactly once")
        alpha, cutoff, markets = split_targets(source, workers=window_workers)
        features = read_split_features(feature_store)
        labels = read_split_rank_labels(label_store)
        names = feature_names(spec)
        if len(features) != count or len(labels) != count:
            raise ValueError(f"{source.split}: store row counts differ from the sample universe")
        stamps = np.asarray(labels["timestamp"].astype("int64"))
        if (stamps != cutoff).any():
            raise ValueError(f"{source.split}: rank-label cutoffs differ from the sample universe")
        market_names = sorted(set(markets.tolist()))
        symbols = features["symbol"].astype(str).to_numpy()
        symbol_names = sorted(set(symbols.tolist()))
        symbol_lookup = {name: code for code, name in enumerate(symbol_names)}
        np.save(pending / "features.npy", features[names].to_numpy(dtype=np.float32))
        np.save(
            pending / "ranks.npy",
            labels[[f"rank_h{h:02d}" for h in horizons]].to_numpy(dtype=np.float32),
        )
        np.save(pending / "alpha.npy", alpha.astype(np.float32))
        np.save(pending / "cutoff_ns.npy", cutoff.astype(np.int64))
        np.save(
            pending / "market_codes.npy",
            np.array([market_names.index(m) for m in markets], dtype=np.int8),
        )
        np.save(
            pending / "symbol_codes.npy",
            np.array([symbol_lookup[s] for s in symbols], dtype=np.int32),
        )
        files = {}
        for name in ARRAYS:
            (pending / f"{name}.npy").replace(root / f"{name}.npy")
            files[f"{name}.npy"] = {
                "bytes": (root / f"{name}.npy").stat().st_size,
                "sha256": sha256_file(root / f"{name}.npy"),
            }
        pending.rmdir()
        atomic_write_json(
            root / "xs-cache.json",
            {
                "state": "ready",
                "identity": identity,
                "split": source.split,
                "samples": count,
                "window": window,
                "channels": ["open", "high", "low", "close", "volume"],
                "horizons": horizons,
                "feature_columns": names,
                "markets": market_names,
                "symbols": symbol_names,
                "files": files,
                "worker_plan": plan,
                "seconds": time.monotonic() - started,
            },
        )
    return root


@dataclass(frozen=True)
class XsCache:
    """Memory-mapped view of one split's cache."""

    root: Path
    split: str
    windows: np.ndarray
    features: np.ndarray
    ranks: np.ndarray
    alpha: np.ndarray
    cutoff_ns: np.ndarray
    market_codes: np.ndarray
    symbol_codes: np.ndarray
    markets: list[str]
    symbols: list[str]
    horizons: list[int]
    feature_columns: list[str]

    def __len__(self) -> int:
        return len(self.cutoff_ns)


def open_xs_cache(root: Path) -> XsCache:
    payload = json.loads((root / "xs-cache.json").read_text())
    if payload.get("state") != "ready":
        raise ValueError("Cross-stock cache is not ready")
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    return XsCache(
        root=root,
        split=payload["split"],
        markets=list(payload["markets"]),
        symbols=list(payload["symbols"]),
        horizons=[int(h) for h in payload["horizons"]],
        feature_columns=list(payload["feature_columns"]),
        **arrays,
    )
