"""Residualized cross-sectional gauss-rank labels (experiment protocol 13.16).

Within each (signal date, market) group, every horizon's raw alpha target (already relative to
the market index) is regressed on an intercept, a size proxy and industry dummies, and the
residual is gauss-ranked exactly as in :mod:`stock_forecasting.data.rank_labels`. The store has
the rank-label layout (``shards/bucket-*/shard.parquet``, ``label-index.parquet``,
``rank-labels.json``) plus one ``residual_hNN`` column per horizon, so the walk-forward readers
can consume it, and its identity includes the residualization spec, the company reference
table and this module, so the existing rank-label stores are untouched.

Size proxies: ``log_dollar_volume_median`` (log of the median close x volume over the last
``dollar_volume_window`` sessions up to the signal date; point in time) or
``log_market_cap_snapshot`` (log of the raw close times the reference snapshot's issued shares,
with a configurable fallback for names missing from the snapshot). Industry codes come from the
reference CSV; missing codes form the ``unknown_class``; in groups with at least
``min_group_for_industry`` names, classes with at least ``min_names_per_class`` names get a
dummy (smaller classes are merged into one), otherwise only the size proxy is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data.rank_labels import (
    _bounded_map,
    _group_statistics,
    _worker_count,
    gauss_rank,
    group_ids,
    label_identity,
)

RESIDUAL_LABEL_SCHEMA = "cross-sectional-gauss-rank-residual-v1"
IMPLEMENTATION_FILES = ("data/residual_labels.py", "data/rank_labels.py", "data/dataset.py")
SIZE_PROXIES = ("log_dollar_volume_median", "log_market_cap_snapshot")
DEFAULT_SPEC_DIR = Path("configs/labels")
MANIFEST_NAME = "rank-labels.json"
OTHER_CLASS = "__other__"

_SOURCE = None


def implementation_sha256() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode() + b"\0" + (package / relative).read_bytes())
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------------------
# Spec and identity
# --------------------------------------------------------------------------------------


def load_label_spec(path: str | Path) -> dict:
    """Read and validate a residualization spec; the reference path is resolved to the repo."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "name", "size_proxy", "dollar_volume_window", "industry"}
    if not required <= set(spec) or spec["schema"] != RESIDUAL_LABEL_SCHEMA:
        raise ValueError(f"{path} is not a {RESIDUAL_LABEL_SCHEMA} spec")
    if not isinstance(spec["name"], str) or not spec["name"].isidentifier():
        raise ValueError("Label spec name must be an identifier")
    if spec["size_proxy"] not in SIZE_PROXIES:
        raise ValueError(f"size_proxy must be one of {SIZE_PROXIES}")
    fallback = spec.get("fallback_size_proxy")
    if fallback is not None and fallback not in SIZE_PROXIES:
        raise ValueError(f"fallback_size_proxy must be one of {SIZE_PROXIES}")
    window = spec["dollar_volume_window"]
    if type(window) is not int or window < 2:
        raise ValueError("dollar_volume_window must be an integer of at least two")
    industry = spec["industry"]
    for key in ("reference", "unknown_class", "min_names_per_class", "min_group_for_industry"):
        if key not in industry:
            raise ValueError(f"industry.{key} is missing from the label spec")
    if type(industry["min_names_per_class"]) is not int or industry["min_names_per_class"] < 2:
        raise ValueError("industry.min_names_per_class must be an integer of at least two")
    minimum_group = industry["min_group_for_industry"]
    if type(minimum_group) is not int or minimum_group < 2:
        raise ValueError("industry.min_group_for_industry must be an integer of at least two")
    reference = Path(industry["reference"])
    if not reference.is_absolute():
        reference = project_root() / reference
    if not reference.is_file():
        raise FileNotFoundError(f"Company reference table is missing: {reference}")
    return {**spec, "_reference_path": str(reference)}


def public_spec(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if not key.startswith("_")}


def spec_identity(spec: dict) -> dict:
    return {
        "name": spec["name"],
        "sha256": canonical_json_sha256(public_spec(spec)),
        "reference_sha256": sha256_file(Path(spec["_reference_path"])),
    }


def residual_label_identity(source, min_group_size: int, spec: dict) -> dict:
    base = label_identity(source, min_group_size)
    return {
        **base,
        "schema": RESIDUAL_LABEL_SCHEMA,
        "transform": (
            "inverse_normal((average_rank - 0.5) / group_size) of the cross-sectional OLS "
            "residual on intercept, size proxy and industry dummies"
        ),
        "residualization": spec_identity(spec),
        "implementation_sha256": implementation_sha256(),
    }


def residual_label_root(source, min_group_size: int, spec: dict) -> tuple[Path, dict]:
    if source.sample_universe_identity is None:
        raise ValueError("Residual labels require a cleaned sample universe")
    identity = residual_label_identity(source, min_group_size, spec)
    return source.root.parent / "rank-labels" / canonical_json_sha256(identity), identity


def _ready(root: Path, identity: dict) -> bool:
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text())
    if payload.get("identity") != identity or payload.get("state") != "ready":
        raise ValueError("Residual-label store has an inconsistent identity")
    return True


def attach_residual_labels(source, min_group_size: int, spec: dict) -> Path:
    """Path of the ready store for this source and spec; raises when it is missing."""
    root, identity = residual_label_root(source, min_group_size, spec)
    if not _ready(root, identity):
        raise ValueError(f"Residual-label store is missing for {source.split}: {root}")
    return root


# --------------------------------------------------------------------------------------
# Reference table and covariates
# --------------------------------------------------------------------------------------


def load_reference(path: str | Path) -> dict[str, tuple[str, float]]:
    """symbol -> (industry code, issued shares) from the company reference CSV."""
    table = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                shares = float(row["issued_shares"]) if row["issued_shares"] else float("nan")
            except ValueError:
                shares = float("nan")
            table[row["symbol"]] = (row["industry_code"].strip(), shares)
    if not table:
        raise ValueError(f"Company reference table is empty: {path}")
    return table


def _open_source(state: dict) -> None:
    global _SOURCE
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset

    _SOURCE = LazyFinancialWindowDataset.__new__(LazyFinancialWindowDataset)
    _SOURCE.__setstate__(state)


def _bucket_rows(task):
    """Targets, cutoffs, raw close and log median dollar volume of one bucket's symbols."""
    bucket, spans, window = task
    rows = []
    for symbol, begin, end in spans:
        batch = _SOURCE.target_batch(np.arange(begin, end))
        asset = _SOURCE._numeric_symbol(symbol)
        stamps = asset["timestamps"].asi8
        positions = np.searchsorted(stamps, batch["cutoff_ns"])
        if (positions >= len(stamps)).any() or (stamps[positions] != batch["cutoff_ns"]).any():
            raise RuntimeError(f"Sample cutoffs are not bars of {symbol}")
        raw = asset["raw"]
        close = raw[positions, 3]
        dollar = raw[:, 3] * raw[:, 4]
        median = (
            pd.Series(dollar)
            .rolling(window, min_periods=max(1, window // 2))
            .median()
            .to_numpy()[positions]
        )
        log_dollar = np.log(np.maximum(np.nan_to_num(median, nan=0.0), 1.0))
        rows.append((begin, end, batch["targets"], batch["cutoff_ns"], close, log_dollar))
    _SOURCE.close()
    return bucket, rows


def design_matrix(
    size: np.ndarray,
    industry: np.ndarray,
    *,
    min_group_for_industry: int,
    min_names_per_class: int,
) -> tuple[np.ndarray, bool]:
    """Intercept, standardized size and (for large groups) industry dummies.

    Classes below ``min_names_per_class`` are merged into one class; the largest class is the
    baseline absorbed by the intercept. Returns the matrix and whether industries were used.
    """
    n = size.shape[0]
    centered = size - size.mean()
    scale = float(centered.std())
    columns = [np.ones(n), centered / scale if scale > 1e-12 else np.zeros(n)]
    used = False
    if n >= min_group_for_industry:
        codes, counts = np.unique(industry, return_counts=True)
        merged = np.where(np.isin(industry, codes[counts >= min_names_per_class]), industry,
                          OTHER_CLASS)
        classes, counts = np.unique(merged, return_counts=True)
        if classes.size >= 2:
            baseline = classes[np.argmax(counts)]
            for name in classes:
                if name != baseline:
                    columns.append((merged == name).astype(np.float64))
            used = True
    return np.column_stack(columns), used


def residualize(targets: np.ndarray, design: np.ndarray) -> np.ndarray:
    """OLS residuals of every target column on the design matrix (least squares, rank-safe)."""
    beta, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return targets - design @ beta


def _write_bucket(task):
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, spans, work, output, horizons = task
    ranks = np.load(Path(work) / "ranks.npy", mmap_mode="r")
    residuals = np.load(Path(work) / "residuals.npy", mmap_mode="r")
    cutoff = np.load(Path(work) / "cutoff_ns.npy", mmap_mode="r")
    relative = f"shards/bucket-{bucket:04d}/shard.parquet"
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    schema = pa.schema(
        [("timestamp", pa.timestamp("ns", tz="UTC"))]
        + [(f"rank_h{h:02d}", pa.float32()) for h in horizons]
        + [(f"residual_h{h:02d}", pa.float32()) for h in horizons]
    )
    index = []
    with pq.ParquetWriter(pending, schema) as writer:
        for group, (symbol, begin, end) in enumerate(spans):
            stamps = pa.array(np.asarray(cutoff[begin:end]), type=pa.int64())
            columns = [stamps.cast(schema[0].type)]
            columns += [pa.array(np.asarray(ranks[begin:end, i])) for i in range(len(horizons))]
            columns += [
                pa.array(np.asarray(residuals[begin:end, i], dtype=np.float32))
                for i in range(len(horizons))
            ]
            writer.write_table(pa.Table.from_arrays(columns, schema=schema),
                               row_group_size=max(1, end - begin))
            index.append(
                {
                    "symbol": symbol,
                    "shard_relative_path": relative,
                    "row_group": group,
                    "row_count": end - begin,
                    "ordinal_begin": begin,
                }
            )
    import os

    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    pending.replace(path)
    return relative, {"bytes": path.stat().st_size, "sha256": sha256_file(path)}, index


# --------------------------------------------------------------------------------------
# Store builder
# --------------------------------------------------------------------------------------


def ensure_residual_labels(
    source, min_group_size: int, spec: dict, *, workers: int | None = None
) -> Path:
    """Build one split's residual-label store (idempotent per identity)."""
    from stock_forecasting.data.sample_universe import _exclusive_build

    root, identity = residual_label_root(source, min_group_size, spec)
    if _ready(root, identity):
        return root
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_build(root):
        if _ready(root, identity):
            return root
        started = time.monotonic()
        count, horizons = len(source), list(source.horizons)
        window = int(spec["dollar_volume_window"])
        industry_spec = spec["industry"]
        reference = load_reference(spec["_reference_path"])
        spans = source.symbol_spans()
        markets = sorted({str(source._index[symbol]["market"]) for symbol in spans})
        buckets: dict[int, list] = {}
        for symbol, (begin, end) in sorted(spans.items(), key=lambda item: item[1][0]):
            buckets.setdefault(int(source._index[symbol]["bucket"]), []).append(
                (symbol, begin, end)
            )
        work = root / "work"
        work.mkdir(exist_ok=True)
        targets = np.lib.format.open_memmap(
            work / "targets.npy", mode="w+", dtype="float64", shape=(count, len(horizons))
        )
        cutoff = np.lib.format.open_memmap(
            work / "cutoff_ns.npy", mode="w+", dtype="int64", shape=(count,)
        )
        close = np.empty(count, dtype=np.float64)
        log_dollar = np.empty(count, dtype=np.float64)
        market_codes = np.empty(count, dtype=np.int64)
        industry = np.empty(count, dtype=object)
        shares = np.full(count, np.nan)
        known_industry = 0
        for symbol, (begin, end) in spans.items():
            market_codes[begin:end] = markets.index(str(source._index[symbol]["market"]))
            code, issued = reference.get(symbol, ("", float("nan")))
            industry[begin:end] = code or industry_spec["unknown_class"]
            shares[begin:end] = issued
            known_industry += (end - begin) if code else 0
        largest = max(sum(end - begin for _, begin, end in rows) for rows in buckets.values())
        return_workers, return_plan = _worker_count(
            workers, len(buckets), 768 * 1024**2 + largest * (len(horizons) + 3) * 8 * 3
        )
        print(
            f"Residual labels {source.split} ({spec['name']}): returns and covariates for "
            f"{count} samples in {len(buckets)} buckets, workers={return_workers}",
            flush=True,
        )
        covered = 0
        for _bucket, rows in _bounded_map(
            _bucket_rows,
            [(bucket, rows, window) for bucket, rows in sorted(buckets.items())],
            return_workers,
            _open_source,
            (source.__getstate__(),),
        ):
            for begin, end, values, stamps, closes, dollars in rows:
                targets[begin:end], cutoff[begin:end] = values, stamps
                close[begin:end], log_dollar[begin:end] = closes, dollars
                covered += end - begin
        if covered != count:
            raise RuntimeError("Residual-label returns did not cover every sample exactly once")
        targets.flush()
        cutoff.flush()
        # Size proxy per row (variant B falls back to the dollar-volume proxy when shares miss).
        if spec["size_proxy"] == "log_market_cap_snapshot":
            cap = close * shares
            valid = np.isfinite(cap) & (cap > 0)
            size = np.where(valid, np.log(np.where(valid, cap, 1.0)), np.nan)
            fallback_rows = int((~valid).sum())
            if spec.get("fallback_size_proxy") == "log_dollar_volume_median":
                size = np.where(valid, size, log_dollar)
            elif fallback_rows:
                raise ValueError("Market-cap size proxy is missing for some rows and no fallback")
        else:
            size, fallback_rows = log_dollar.copy(), 0
        groups = group_ids(np.asarray(cutoff), market_codes)
        order = np.argsort(groups, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(groups[order]) != 0, True])
        residuals = np.lib.format.open_memmap(
            work / "residuals.npy", mode="w+", dtype="float64", shape=(count, len(horizons))
        )
        industry_groups = 0
        for start, stop in pairwise(bounds):
            rows = order[start:stop]
            design, used = design_matrix(
                size[rows],
                industry[rows],
                min_group_for_industry=int(industry_spec["min_group_for_industry"]),
                min_names_per_class=int(industry_spec["min_names_per_class"]),
            )
            industry_groups += int(used)
            residuals[rows] = residualize(np.asarray(targets[rows]), design)
        residuals.flush()
        ranks = np.lib.format.open_memmap(
            work / "ranks.npy", mode="w+", dtype="float32", shape=(count, len(horizons))
        )
        rank_threads, rank_plan = _worker_count(workers, len(horizons), 256 * 1024**2 + count * 72)

        def rank_column(index, residuals=residuals, ranks=ranks, groups=groups):
            column = np.ascontiguousarray(residuals[:, index])
            ranks[:, index] = gauss_rank(column, groups, min_group_size)
            return index

        with ThreadPoolExecutor(max_workers=rank_threads) as pool:
            list(pool.map(rank_column, range(len(horizons))))
        ranks.flush()
        labelled = np.isfinite(ranks[:, 0])
        sizes = np.bincount(groups)[groups]
        if not np.array_equal(labelled, sizes >= min_group_size):
            raise ValueError("Residual labels are missing outside the small-group mask")
        statistics = _group_statistics(groups, market_codes, markets, min_group_size)
        total_groups = int(bounds.size - 1)
        del targets, cutoff, groups, sizes, labelled
        write_workers, write_plan = _worker_count(workers, len(buckets), 512 * 1024**2)
        files, index = {}, []
        for relative, metadata, rows in _bounded_map(
            _write_bucket,
            [
                (bucket, rows, str(work), str(root), horizons)
                for bucket, rows in sorted(buckets.items())
            ],
            write_workers,
        ):
            files[relative] = metadata
            index.extend(rows)
        del ranks, residuals
        frame = pd.DataFrame(index).sort_values("ordinal_begin", kind="stable")
        pending = root / "label-index.pending.parquet"
        frame.to_parquet(pending, index=False)
        pending.replace(root / "label-index.parquet")
        for name in ("targets.npy", "cutoff_ns.npy", "ranks.npy", "residuals.npy"):
            (work / name).unlink(missing_ok=True)
        atomic_write_json(
            root / MANIFEST_NAME,
            {
                "state": "ready",
                "identity": identity,
                "schema": RESIDUAL_LABEL_SCHEMA,
                "spec": public_spec(spec),
                "label_index_sha256": sha256_file(root / "label-index.parquet"),
                "files": files,
                "groups_by_market": statistics,
                "residualization": {
                    "size_proxy": spec["size_proxy"],
                    "fallback_rows": fallback_rows,
                    "rows_with_known_industry": known_industry,
                    "rows": count,
                    "groups": total_groups,
                    "groups_with_industry_dummies": industry_groups,
                },
                "execution": {
                    "return_workers": return_workers,
                    "return_plan": return_plan,
                    "rank_threads": rank_threads,
                    "rank_plan": rank_plan,
                    "write_workers": write_workers,
                    "write_plan": write_plan,
                    "seconds": time.monotonic() - started,
                },
            },
        )
    return root


def residual_column(h: int) -> str:
    return f"residual_h{h:02d}"


def expected_columns(horizons: list[int]) -> list[str]:
    return (
        ["timestamp"]
        + [f"rank_h{h:02d}" for h in horizons]
        + [residual_column(h) for h in horizons]
    )


def describe_store(root: Path) -> dict[str, Any]:
    payload = json.loads((root / MANIFEST_NAME).read_text())
    return {
        "root": str(root),
        "schema": payload.get("schema"),
        "spec": payload.get("spec", {}).get("name"),
        "residualization": payload.get("residualization"),
        "seconds": math.floor(payload.get("execution", {}).get("seconds", 0.0)),
    }


__all__ = [
    "RESIDUAL_LABEL_SCHEMA",
    "attach_residual_labels",
    "describe_store",
    "design_matrix",
    "ensure_residual_labels",
    "expected_columns",
    "load_label_spec",
    "load_reference",
    "residual_column",
    "residual_label_identity",
    "residual_label_root",
    "residualize",
    "spec_identity",
]
