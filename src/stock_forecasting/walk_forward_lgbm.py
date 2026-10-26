"""B0 ``t-lgbm-alpha``: LightGBM walk-forward baseline on the feature library (protocol 13.13).

One regressor per fold, horizon and seed on the gauss-rank label; early stopping on the fold
validation's horizon-equal daily Rank IC. Per-sample predictions of every fold's quarter are
stored as Parquet next to the raw alpha targets, daily Rank IC series and the cell summary with
circular moving-block bootstrap intervals. Training runs in a bounded spawn process pool; each
worker holds one fold's feature matrix through a memory map and trains a chunk of horizons.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.feature_library import (
    feature_identity,
    feature_names,
    feature_root,
    read_split_features,
)
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.data.rank_labels import (
    _bounded_map,
    _bucket_returns,
    _open_source,
    label_identity,
    label_root,
)
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count
from stock_forecasting.walk_forward import (
    Fold,
    assign_folds,
    fold_frame,
    period_of,
    quarterly_folds,
    taiwan_sessions,
)

SCHEMA_VERSION = "walk-forward-lgbm-v1"
MODEL_NAME = "t-lgbm-alpha"
SPLITS = ("train", "validation", "test")
DEFAULT_SEEDS = (42, 43, 44)
WORKERS_ENV = "FIN_TS_LGBM_WORKERS"
WORKER_MEMORY_BYTES = 2 * 1024**3
MEMORY_FRACTION = 0.4
MAX_WORKERS = 16
TASK_TIMEOUT_SECONDS = 6 * 3600
# Fixed before training (protocol 13.13 item 1).
LGBM_PARAMS = {
    "objective": "regression",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "verbosity": -1,
}
DEFAULT_MAX_ROUNDS = 1000
DEFAULT_EARLY_STOPPING = 50
CELLS = {"short": (1,), "holding": (5, 6, 7, 8, 9, 10), "long": (11, 12, 13, 14)}
BLOCK_LENGTHS = {"short": 5, "holding": 20, "long": 28}
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 11


def rank_column(h: int) -> str:
    return f"rank_h{h:02d}"


def alpha_column(h: int) -> str:
    return f"alpha_h{h:02d}"


def score_column(h: int) -> str:
    return f"h{h:02d}"


def residual_column(h: int) -> str:
    return f"residual_h{h:02d}"


# --------------------------------------------------------------------------------------
# Store readers
# --------------------------------------------------------------------------------------


def read_split_rank_labels(root: Path) -> pd.DataFrame:
    """One split's gauss-rank store in ordinal order: ordinal, symbol, timestamp, rank_hNN."""
    import pyarrow.parquet as pq

    payload = json.loads((root / "rank-labels.json").read_text())
    if payload.get("state") != "ready":
        raise ValueError("Rank-label store is not ready")
    index = pd.read_parquet(root / "label-index.parquet").sort_values("ordinal_begin")
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


def split_targets(source, *, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw alpha log returns, cutoffs and markets of every sample, via bounded workers."""
    count, horizons = len(source), list(source.horizons)
    spans = source.symbol_spans()
    buckets: dict[int, list] = {}
    for symbol, (begin, end) in sorted(spans.items(), key=lambda item: item[1][0]):
        buckets.setdefault(int(source._index[symbol]["bucket"]), []).append((symbol, begin, end))
    targets = np.full((count, len(horizons)), np.nan, dtype=np.float64)
    cutoff = np.zeros(count, dtype=np.int64)
    markets = np.empty(count, dtype=object)
    for symbol, (begin, end) in spans.items():
        markets[begin:end] = str(source._index[symbol]["market"])
    covered = 0
    for _bucket, rows in _bounded_map(
        _bucket_returns,
        sorted(buckets.items()),
        max(1, min(workers, len(buckets))),
        _open_source,
        (source.__getstate__(),),
    ):
        for begin, end, values, stamps in rows:
            targets[begin:end], cutoff[begin:end] = values, stamps
            covered += end - begin
    if covered != count:
        raise RuntimeError("Alpha targets did not cover every sample exactly once")
    return targets, cutoff, markets


@dataclass(frozen=True)
class SplitStores:
    split: str
    source_root: str
    samples: int
    feature_root: Path
    feature_identity: dict
    label_root: Path
    label_identity: dict


def locate_stores(
    config, spec: dict, split: str, root, *, label_variant: dict | None = None
) -> SplitStores:
    """Resolve the split's ready feature and rank-label stores without building anything."""
    from stock_forecasting.data.sample_universe import open_clean_dataset

    source = open_clean_dataset(config, split, validated_root=root)
    try:
        features, feature_id = feature_root(source, spec)
        if label_variant is None:
            labels, label_id = label_root(source, config.model.rank_min_group_size)
        else:
            from stock_forecasting.data.residual_labels import residual_label_root

            labels, label_id = residual_label_root(
                source, config.model.rank_min_group_size, label_variant
            )
        for store, name in ((features, "features.json"), (labels, "rank-labels.json")):
            manifest = store / name
            if not manifest.is_file() or json.loads(manifest.read_text()).get("state") != "ready":
                raise FileNotFoundError(
                    f"{split}: the {name.split('.')[0]} store is not built; run cpu-features first"
                )
        return SplitStores(
            split=split,
            source_root=str(source.root),
            samples=len(source),
            feature_root=features,
            feature_identity=feature_id,
            label_root=labels,
            label_identity=label_id,
        )
    finally:
        source.close()


def load_split_table(
    config, spec: dict, split: str, root, *, workers: int, label_variant: dict | None = None
) -> pd.DataFrame:
    """Features, gauss-rank labels and raw alpha targets of one split, one row per sample.

    With ``label_variant`` the residualized store supplies the ranks and, as extra columns, the
    residual targets the models are scored against in protocol 13.16.
    """
    from stock_forecasting.data.sample_universe import open_clean_dataset

    stores = locate_stores(config, spec, split, root, label_variant=label_variant)
    source = open_clean_dataset(config, split, validated_root=root)
    try:
        horizons = list(source.horizons)
        targets, cutoff, markets = split_targets(source, workers=workers)
    finally:
        source.close()
    features = read_split_features(stores.feature_root)
    labels = read_split_rank_labels(stores.label_root)
    if len(features) != stores.samples or len(labels) != stores.samples:
        raise ValueError(f"{split}: store row counts differ from the sample universe")
    if (features["ordinal"].to_numpy() != np.arange(stores.samples)).any():
        raise ValueError(f"{split}: feature store ordinals are not contiguous")
    if (labels["ordinal"].to_numpy() != np.arange(stores.samples)).any():
        raise ValueError(f"{split}: rank-label store ordinals are not contiguous")
    stamps = pd.DatetimeIndex(labels["timestamp"]).tz_convert("UTC").asi8
    if (stamps != cutoff).any():
        raise ValueError(f"{split}: rank-label cutoffs differ from the sample universe")
    table = features
    table.insert(2, "split", split)
    table.insert(3, "market", markets)
    dates = pd.DatetimeIndex(labels["timestamp"]).tz_convert("UTC").tz_localize(None).normalize()
    table.insert(4, "signal_date", dates)
    for position, h in enumerate(horizons):
        table[rank_column(h)] = labels[rank_column(h)].to_numpy(dtype=np.float32)
        table[alpha_column(h)] = targets[:, position].astype(np.float32)
        if residual_column(h) in labels.columns:
            table[residual_column(h)] = labels[residual_column(h)].to_numpy(dtype=np.float32)
    return table


def assemble_table(
    config, spec: dict, roots: dict, *, workers: int, label_variant: dict | None = None
) -> pd.DataFrame:
    frames = [
        load_split_table(
            config, spec, split, roots[split], workers=workers, label_variant=label_variant
        )
        for split in SPLITS
    ]
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["signal_date", "market", "symbol"], kind="stable").reset_index(
        drop=True
    )


# --------------------------------------------------------------------------------------
# Rank IC helpers
# --------------------------------------------------------------------------------------


def group_codes(dates: np.ndarray, markets: np.ndarray) -> np.ndarray:
    """Dense (date, market) group ids in sorted order."""
    keys = pd.MultiIndex.from_arrays([dates, markets])
    return pd.factorize(keys, sort=True)[0].astype(np.int64)


def fast_grouped_spearman(
    predictions: np.ndarray, target_ranks: np.ndarray, groups: np.ndarray, sizes: np.ndarray
) -> float:
    """Mean per-group Spearman with ordinal prediction ranks (early-stopping metric only)."""
    predictions = np.asarray(predictions, dtype=np.float64)
    order = np.lexsort((predictions, groups))
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    position = np.empty(len(predictions), dtype=np.float64)
    position[order] = np.arange(len(predictions)) - np.repeat(starts, sizes)
    valid = sizes > 2
    mean = np.repeat((sizes - 1) / 2.0, sizes)
    dp, dt = position - mean, target_ranks - mean
    cov = np.bincount(groups, weights=dp * dt, minlength=len(sizes))
    var_p = np.bincount(groups, weights=dp * dp, minlength=len(sizes))
    var_t = np.bincount(groups, weights=dt * dt, minlength=len(sizes))
    denominator = np.sqrt(var_p * var_t)
    # Ordinal ranks hide constant predictions; a group without spread has no correlation.
    group_mean = np.bincount(groups, weights=predictions, minlength=len(sizes)) / np.maximum(
        sizes, 1
    )
    spread = np.bincount(
        groups, weights=(predictions - group_mean[groups]) ** 2, minlength=len(sizes)
    )
    valid &= (denominator > 0) & (spread > 0)
    if not valid.any():
        return 0.0
    return float(np.mean(cov[valid] / denominator[valid]))


def average_ranks(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average ranks within each group (0-based), ties averaged."""
    frame = pd.DataFrame({"g": groups, "v": values})
    return frame.groupby("g", sort=False)["v"].rank(method="average").to_numpy() - 1.0


def daily_rank_ic(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons: list[int],
    minimum: int,
    target: Callable[[int], str] = alpha_column,
) -> pd.DataFrame:
    """Spearman Rank IC per (model, market, h, date) between scores and the target columns.

    ``target`` names the realized column per horizon: the raw alpha return by default, or the
    residual target of protocol 13.16.
    """
    keys = ["signal_date", "market"]
    records = []
    realized = targets[["symbol", *keys, *[target(h) for h in horizons]]]
    for model, group in scores.groupby("model", sort=True):
        merged = group.merge(realized, on=["symbol", *keys], how="inner", validate="one_to_one")
        for h in horizons:
            frame = merged[[*keys, score_column(h), target(h)]].dropna()
            counts = frame.groupby(keys, sort=True).size()
            ranked = frame.groupby(keys, sort=True)[[score_column(h), target(h)]].rank()
            ranked[keys] = frame[keys]

            def correlation(block, score=score_column(h), target=target(h)):
                return block[score].corr(block[target])

            ic = ranked.groupby(keys, sort=True).apply(correlation, include_groups=False)
            for (date, market), value in ic.items():
                n = int(counts.loc[(date, market)])
                if n < minimum or not np.isfinite(value):
                    continue
                records.append(
                    {
                        "model": model,
                        "market": market,
                        "h": h,
                        "date": date,
                        "ic": float(value),
                        "n": n,
                    }
                )
    return pd.DataFrame(records)


def cell_series(daily: pd.DataFrame, model: str, horizons: tuple[int, ...]) -> pd.Series:
    """Per-date cell value: markets weighted equally, then horizons averaged."""
    rows = daily[(daily["model"] == model) & daily["h"].isin(horizons)]
    per_market = rows.groupby(["date", "market", "h"])["ic"].mean()
    per_date = per_market.groupby(["date", "h"]).mean().groupby("date").mean()
    return per_date.sort_index()


def block_bootstrap_mean(values: np.ndarray, block: int, *, replicates: int, seed: int):
    """Circular moving-block bootstrap of the mean; returns (low, high) at 95%."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(replicates, blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(replicates, -1)[:, :n] % n
    means = values[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_cells(daily: pd.DataFrame, schedule: dict) -> list[dict[str, Any]]:
    rows = []
    if daily.empty:
        return rows
    periods = period_of(pd.DatetimeIndex(sorted(daily["date"].unique())), schedule)
    for model in sorted(daily["model"].unique()):
        for cell, horizons in CELLS.items():
            series = cell_series(daily, model, horizons)
            if series.empty:
                continue
            names = periods.reindex(series.index).fillna("")
            scopes = {"P1+P2": names.isin(["P1", "P2"]).to_numpy()}
            for name in sorted(set(names) - {""}):
                scopes[name] = (names == name).to_numpy()
            for scope, mask in scopes.items():
                values = series.to_numpy()[mask]
                if len(values) == 0:
                    continue
                std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                low, high = block_bootstrap_mean(
                    values,
                    BLOCK_LENGTHS[cell],
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED,
                )
                rows.append(
                    {
                        "model": model,
                        "cell": cell,
                        "period": scope,
                        "days": len(values),
                        "rank_ic": float(values.mean()),
                        "rank_icir": float(values.mean() / std) if std and std > 0 else None,
                        "positive_share": float((values > 0).mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "block_length": BLOCK_LENGTHS[cell],
                    }
                )
    return rows


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def backend_name() -> str:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return "sklearn_hist_gradient_boosting"
    return "lightgbm"


def worker_plan(requested: int | None, tasks: int) -> tuple[int, dict[str, Any]]:
    """Bounded by the override, visible CPUs (4 threads each), memory and task count."""
    override = os.environ.get(WORKERS_ENV)
    limit = requested if requested is not None else (int(override) if override else None)
    if limit is not None and not 1 <= limit <= MAX_WORKERS:
        raise ValueError(f"{WORKERS_ENV} must be 1-{MAX_WORKERS}")
    cpus = detect_visible_cpu_count()
    memory = int(detect_available_memory().available_bytes * MEMORY_FRACTION)
    count = max(
        1,
        min(
            limit or MAX_WORKERS,
            max(1, cpus // 4),
            max(1, memory // WORKER_MEMORY_BYTES),
            max(1, tasks),
        ),
    )
    threads = max(1, cpus // count)
    return count, {
        "requested": limit,
        "count": count,
        "threads_per_worker": threads,
        "visible_cpus": cpus,
        "memory_budget_bytes": memory,
        "per_worker_bytes": WORKER_MEMORY_BYTES,
        "tasks": tasks,
    }


def _fit_lightgbm(
    task: dict,
    X_train,
    y_train,
    X_valid,
    y_valid,
    valid_groups,
    valid_sizes,
    valid_target_ranks,
    X_pred,
    seed: int,
):
    import lightgbm as lgb

    params = {**LGBM_PARAMS, "seed": seed, "num_threads": task["threads"]}
    train_set = task["_train_set"]
    train_set.set_label(y_train)
    valid_set = task["_valid_set"]
    valid_set.set_label(y_valid)

    def rank_ic(preds, _data):
        return (
            "rank_ic",
            fast_grouped_spearman(preds, valid_target_ranks, valid_groups, valid_sizes),
            True,
        )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=task["max_rounds"],
        valid_sets=[valid_set],
        valid_names=["validation"],
        feval=rank_ic,
        callbacks=[
            lgb.early_stopping(task["early_stopping"], first_metric_only=True, verbose=False)
        ],
    )
    best = int(booster.best_iteration or booster.current_iteration())
    score = float(booster.best_score["validation"]["rank_ic"])
    return booster.predict(X_pred, num_iteration=best).astype(np.float32), best, score


def _fit_sklearn(
    task: dict,
    X_train,
    y_train,
    X_valid,
    y_valid,
    valid_groups,
    valid_sizes,
    valid_target_ranks,
    X_pred,
    seed: int,
):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=task["max_rounds"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        max_leaf_nodes=LGBM_PARAMS["num_leaves"],
        min_samples_leaf=LGBM_PARAMS["min_data_in_leaf"],
        early_stopping=True,
        n_iter_no_change=task["early_stopping"],
        validation_fraction=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    score = fast_grouped_spearman(valid_pred, valid_target_ranks, valid_groups, valid_sizes)
    return model.predict(X_pred).astype(np.float32), int(model.n_iter_), float(score)


def run_fold_task(task: dict) -> dict:
    """Train every seed of the task's horizons on one fold; the matrices are memory-mapped."""
    work = Path(task["work"])
    X_train = np.load(work / "X_train.npy", mmap_mode="r")
    X_valid = np.load(work / "X_valid.npy", mmap_mode="r")
    X_pred = np.load(work / "X_pred.npy", mmap_mode="r")
    y_train_all = np.load(work / "y_train.npy")
    y_valid_all = np.load(work / "y_valid.npy")
    alpha_valid_all = np.load(work / "alpha_valid.npy")
    valid_groups = np.load(work / "valid_groups.npy")
    valid_sizes = np.bincount(valid_groups)
    backend = task["backend"]
    if backend == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(
            np.asarray(X_train),
            label=y_train_all[:, 0],
            feature_name=task["feature_names"],
            # The same binned dataset serves every horizon and seed; pre-filtering is
            # disabled so the training parameters never conflict with the bins.
            params={
                "num_threads": task["threads"],
                "verbosity": -1,
                "min_data_in_leaf": LGBM_PARAMS["min_data_in_leaf"],
                "feature_pre_filter": False,
            },
            free_raw_data=True,
        )
        train_set.construct()
        valid_set = lgb.Dataset(
            np.asarray(X_valid), label=y_valid_all[:, 0], reference=train_set, free_raw_data=True
        )
        valid_set.construct()
        task = {**task, "_train_set": train_set, "_valid_set": valid_set}
        fit = _fit_lightgbm
    else:
        fit = _fit_sklearn
    records, predictions = [], {}
    for position, h in zip(task["positions"], task["horizons"], strict=True):
        y_train = y_train_all[:, position]
        y_valid = y_valid_all[:, position]
        valid_target_ranks = average_ranks(alpha_valid_all[:, position], valid_groups)
        for seed in task["seeds"]:
            started = time.monotonic()
            scores, best, valid_ic = fit(
                task,
                X_train,
                y_train,
                X_valid,
                y_valid,
                valid_groups,
                valid_sizes,
                valid_target_ranks,
                X_pred,
                seed,
            )
            predictions[(h, seed)] = scores
            records.append(
                {
                    "fold": task["fold"],
                    "h": h,
                    "seed": seed,
                    "best_iteration": best,
                    "validation_rank_ic": valid_ic,
                    "train_rows": len(y_train),
                    "validation_rows": len(y_valid),
                    "predict_rows": len(X_pred),
                    "seconds": time.monotonic() - started,
                }
            )
    return {"fold": task["fold"], "records": records, "predictions": predictions}


def fold_masks(dates: pd.DatetimeIndex, fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = dates.to_numpy()
    train = (values >= np.datetime64(fold.train_start)) & (
        values < np.datetime64(fold.validation_start)
    )
    valid = (values >= np.datetime64(fold.validation_start)) & (
        values < np.datetime64(fold.train_end)
    )
    predict = (values >= np.datetime64(fold.predict_start)) & (
        values < np.datetime64(fold.predict_end)
    )
    return train, valid, predict


def _write_fold_work(
    table: pd.DataFrame, columns: list[str], horizons: list[int], fold: Fold, work: Path
) -> dict:
    """Memory-mapped matrices of one fold; rows with an unlabelled group are dropped."""
    train, valid, predict = fold_masks(pd.DatetimeIndex(table["signal_date"]), fold)
    labelled = ~table[[rank_column(h) for h in horizons]].isna().any(axis=1).to_numpy()
    train &= labelled
    valid &= labelled
    if not train.any() or not valid.any() or not predict.any():
        raise ValueError(f"Fold {fold.name} has an empty train, validation or prediction slice")
    work.mkdir(parents=True, exist_ok=True)
    features = table[columns].to_numpy(dtype=np.float32)
    np.save(work / "X_train.npy", features[train])
    np.save(work / "X_valid.npy", features[valid])
    np.save(work / "X_pred.npy", features[predict])
    ranks = table[[rank_column(h) for h in horizons]].to_numpy(dtype=np.float32)
    np.save(work / "y_train.npy", ranks[train])
    np.save(work / "y_valid.npy", ranks[valid])
    alpha = table[[alpha_column(h) for h in horizons]].to_numpy(dtype=np.float64)
    np.save(work / "alpha_valid.npy", alpha[valid])
    valid_dates = table["signal_date"].to_numpy()[valid]
    valid_markets = table["market"].to_numpy()[valid]
    np.save(work / "valid_groups.npy", group_codes(valid_dates, valid_markets))
    return {
        "train_rows": int(train.sum()),
        "validation_rows": int(valid.sum()),
        "predict_rows": int(predict.sum()),
        "predict_index": np.flatnonzero(predict),
    }


def _cleanup_work(work: Path) -> None:
    if not work.is_dir():
        return
    for path in work.glob("*.npy"):
        path.unlink(missing_ok=True)
    with os.scandir(work) as entries:
        if not any(True for _ in entries):
            work.rmdir()


def _partial_name(fold_index: int) -> str:
    return f"fold-{fold_index:02d}.npz"


def feature_tag(feature_columns: list[str]) -> str:
    """Short identity of the feature list a fold was trained on (count and name digest)."""
    import hashlib

    digest = hashlib.sha256("\n".join(feature_columns).encode()).hexdigest()[:16]
    return f"{len(feature_columns)}:{digest}"


def _save_partial(
    partial_root: Path,
    fold_index: int,
    fold_scores: dict[tuple[int, int], np.ndarray],
    fold_records: list[dict[str, Any]],
    label_variant: str | None = None,
    features: str | None = None,
) -> None:
    """Persist one finished fold (every horizon and seed) so a restart can reuse it.

    The label variant and the feature tag are stored so a resume never mixes folds trained
    on different labels or on another feature spec (protocol 13.18 runs F1 / F2 beside F0).
    """
    arrays = {f"h{h:02d}_seed{seed}": values for (h, seed), values in fold_scores.items()}
    pending = partial_root / (_partial_name(fold_index) + ".pending")
    with pending.open("wb") as stream:
        np.savez(
            stream,
            records=json.dumps(fold_records),
            label_variant=np.array(label_variant or ""),
            features=np.array(features or ""),
            **arrays,
        )
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(partial_root / _partial_name(fold_index))


def _load_partials(
    resume_root: Path | None,
    partial_root: Path,
    folds: list[Fold],
    horizons: list[int],
    seeds: tuple[int, ...],
    label_variant: str | None = None,
    features: str | None = None,
) -> dict[int, tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]]:
    """Finished folds of an earlier run that cover every horizon and seed asked for now.

    Folds saved under another label variant (or by the standard labels when a variant is
    asked for) or under another feature tag are ignored; folds saved before the tag existed
    are ignored whenever a tag is asked for.
    """
    if resume_root is None:
        return {}
    source = Path(resume_root) / "partial"
    resumed = {}
    for fold in folds:
        path = source / _partial_name(fold.index)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            keys = {f"h{h:02d}_seed{seed}" for h in horizons for seed in seeds}
            if not keys <= set(payload.files):
                continue
            stored = str(payload["label_variant"]) if "label_variant" in payload.files else ""
            if stored != (label_variant or ""):
                continue
            stored_features = str(payload["features"]) if "features" in payload.files else ""
            if stored_features != (features or ""):
                continue
            fold_scores = {
                (h, seed): np.asarray(payload[f"h{h:02d}_seed{seed}"], dtype=np.float32)
                for h in horizons
                for seed in seeds
            }
            fold_records = json.loads(str(payload["records"]))
        resumed[fold.index] = (fold_scores, fold_records)
        if path.resolve() != (partial_root / path.name).resolve():
            (partial_root / path.name).write_bytes(path.read_bytes())
    return resumed


def _resumed_fold_info(
    table: pd.DataFrame, horizons: list[int], fold: Fold, fold_records: list[dict[str, Any]]
) -> dict:
    train, valid, predict = fold_masks(pd.DatetimeIndex(table["signal_date"]), fold)
    labelled = ~table[[rank_column(h) for h in horizons]].isna().any(axis=1).to_numpy()
    predict_index = np.flatnonzero(predict)
    expected = {int(r["predict_rows"]) for r in fold_records}
    if expected and expected != {len(predict_index)}:
        raise ValueError(f"Fold {fold.name}: resumed predictions do not match this table")
    return {
        "train_rows": int((train & labelled).sum()),
        "validation_rows": int((valid & labelled).sum()),
        "predict_rows": len(predict_index),
        "predict_index": predict_index,
        "resumed": True,
    }


def run_walk_forward(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    horizons: list[int],
    folds: list[Fold],
    schedule: dict,
    output_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    workers: int | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    early_stopping: int = DEFAULT_EARLY_STOPPING,
    min_group_size: int,
    scratch: Path | None = None,
    resume_root: Path | None = None,
    model_name: str = MODEL_NAME,
    label_variant: str | None = None,
) -> dict[str, Any]:
    """Train every fold, write predictions, targets, daily Rank IC and the cell summary.

    Each finished fold is written to ``output_root/partial/`` at once; ``resume_root`` names an
    earlier output root whose partial folds are reused (a Pod restart costs at most one fold).
    """
    started = time.monotonic()
    backend = backend_name()
    count, plan = worker_plan(workers, len(folds) * len(horizons))
    chunks = max(1, min(count, len(horizons)))
    horizon_chunks = [list(chunk) for chunk in np.array_split(np.arange(len(horizons)), chunks)]
    work_root = (scratch or output_root) / "work"
    partial_root = output_root / "partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    records, scores = [], {}
    features = feature_tag(feature_columns)
    resumed = _load_partials(resume_root, partial_root, folds, horizons, seeds, label_variant,
                             features)
    for fold_index, (fold_scores, fold_records) in resumed.items():
        scores[fold_index] = fold_scores
        records.extend(fold_records)
    tasks, fold_info = [], {}
    pending_chunks: dict[int, int] = {}
    for fold in folds:
        if fold.index in resumed:
            fold_info[fold.index] = _resumed_fold_info(
                table, horizons, fold, resumed[fold.index][1]
            )
            print(json.dumps({"fold": fold.index, "resumed": True}), flush=True)
            continue
        work = work_root / f"fold-{fold.index:02d}"
        fold_info[fold.index] = _write_fold_work(table, feature_columns, horizons, fold, work)
        pending_chunks[fold.index] = len(horizon_chunks)
        for positions in horizon_chunks:
            tasks.append(
                {
                    "fold": fold.index,
                    "work": str(work),
                    "positions": [int(p) for p in positions],
                    "horizons": [horizons[int(p)] for p in positions],
                    "seeds": list(seeds),
                    "threads": plan["threads_per_worker"],
                    "max_rounds": max_rounds,
                    "early_stopping": early_stopping,
                    "backend": backend,
                    "feature_names": feature_columns,
                }
            )
    fold_records: dict[int, list[dict[str, Any]]] = {}
    for result in _bounded_map(run_fold_task, tasks, count) if tasks else ():
        records.extend(result["records"])
        fold_records.setdefault(result["fold"], []).extend(result["records"])
        scores.setdefault(result["fold"], {}).update(result["predictions"])
        pending_chunks[result["fold"]] -= 1
        if pending_chunks[result["fold"]] == 0:
            _save_partial(
                partial_root,
                result["fold"],
                scores[result["fold"]],
                fold_records[result["fold"]],
                label_variant,
                features,
            )
            _cleanup_work(work_root / f"fold-{result['fold']:02d}")
        print(
            json.dumps(
                {
                    "fold": result["fold"],
                    "completed_fits": len(result["records"]),
                    "fold_done": pending_chunks[result["fold"]] == 0,
                    "seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
    for fold in folds:
        if fold.index not in resumed:
            _cleanup_work(work_root / f"fold-{fold.index:02d}")
    # Predictions: one row per (sample, model) where model is each seed and the mean ensemble.
    keys = ["symbol", "market", "signal_date", "split", "ordinal"]
    prediction_frames, target_frames = [], []
    for fold in folds:
        index = fold_info[fold.index]["predict_index"]
        base = table.iloc[index][keys].reset_index(drop=True)
        base["fold"] = fold.index
        targets = base.copy()
        for h in horizons:
            targets[alpha_column(h)] = table.iloc[index][alpha_column(h)].to_numpy()
            targets[rank_column(h)] = table.iloc[index][rank_column(h)].to_numpy()
            if residual_column(h) in table.columns:
                targets[residual_column(h)] = table.iloc[index][residual_column(h)].to_numpy()
        target_frames.append(targets)
        per_seed = []
        for seed in seeds:
            frame = base.copy()
            frame["model"] = f"{model_name}-seed{seed}"
            for h in horizons:
                frame[score_column(h)] = scores[fold.index][(h, seed)]
            per_seed.append(frame)
        ensemble = base.copy()
        ensemble["model"] = model_name
        for h in horizons:
            ensemble[score_column(h)] = np.mean(
                [scores[fold.index][(h, seed)] for seed in seeds], axis=0
            ).astype(np.float32)
        prediction_frames.extend([*per_seed, ensemble])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    dates = pd.DatetimeIndex(targets["signal_date"])
    targets["period"] = period_of(dates, schedule).to_numpy()
    targets["fold"] = assign_folds(dates, folds).to_numpy()
    daily = daily_rank_ic(predictions, targets, horizons=horizons, minimum=min_group_size)
    cells = summarize_cells(daily, schedule)
    has_residual = all(residual_column(h) in targets.columns for h in horizons)
    daily_residual = (
        daily_rank_ic(
            predictions, targets, horizons=horizons, minimum=min_group_size, target=residual_column
        )
        if has_residual
        else pd.DataFrame()
    )
    cells_residual = summarize_cells(daily_residual, schedule) if has_residual else []
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_root / "predictions.parquet", index=False)
    targets.to_parquet(output_root / "targets.parquet", index=False)
    daily.to_parquet(output_root / "daily_rank_ic.parquet", index=False)
    if has_residual:
        daily_residual.to_parquet(output_root / "daily_rank_ic_residual.parquet", index=False)
    fits = pd.DataFrame(records)
    fits.to_parquet(output_root / "fits.parquet", index=False)
    per_horizon = (
        daily.groupby(["model", "market", "h"])["ic"].agg(["mean", "std", "count"]).reset_index()
        if not daily.empty
        else pd.DataFrame()
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "label_variant": label_variant,
        "backend": backend,
        "generated_at": datetime.now(UTC).isoformat(),
        "params": {**LGBM_PARAMS, "max_rounds": max_rounds, "early_stopping": early_stopping},
        "seeds": list(seeds),
        "horizons": horizons,
        "folds": fold_frame(folds).to_dict(orient="records"),
        "fold_rows": {
            str(index): {k: v for k, v in info.items() if k != "predict_index"}
            for index, info in fold_info.items()
        },
        "worker_plan": plan,
        "cells": cells,
        "per_market_horizon": per_horizon.to_dict(orient="records"),
        "cells_residual": cells_residual,
        "per_market_horizon_residual": (
            daily_residual.groupby(["model", "market", "h"])["ic"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .to_dict(orient="records")
            if has_residual and not daily_residual.empty
            else []
        ),
        "fits": {
            "count": len(fits),
            "mean_best_iteration": float(fits["best_iteration"].mean()) if len(fits) else None,
            "mean_validation_rank_ic": float(fits["validation_rank_ic"].mean())
            if len(fits)
            else None,
            "seconds": float(fits["seconds"].sum()) if len(fits) else 0.0,
        },
        "seconds": time.monotonic() - started,
    }
    atomic_write_json(output_root / "results.json", results)
    return results


def build_folds(schedule: dict) -> list[Fold]:
    return quarterly_folds(taiwan_sessions(schedule), schedule)


def write_manifest(
    output_root: Path,
    *,
    config_path: Path,
    spec: dict,
    schedule_path: Path,
    stores: list[SplitStores],
    results: dict[str, Any],
    label_variant: dict | None = None,
) -> dict[str, Any]:
    names = [
        "results.json",
        "predictions.parquet",
        "targets.parquet",
        "daily_rank_ic.parquet",
        "fits.parquet",
    ]
    if (output_root / "daily_rank_ic_residual.parquet").is_file():
        names.append("daily_rank_ic_residual.parquet")  # label variants (protocol 13.16)
    files = {
        name: {
            "sha256": sha256_file(output_root / name),
            "bytes": (output_root / name).stat().st_size,
        }
        for name in names
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "walk-forward-lgbm",
        "model": results["model"],
        "label_variant": label_variant,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_config": str(config_path),
        "schedule": str(schedule_path),
        "feature_spec_columns": len(feature_names(spec)),
        "splits": {
            store.split: {
                "source": store.source_root,
                "samples": store.samples,
                "features": {"root": str(store.feature_root), "identity": store.feature_identity},
                "rank_labels": {"root": str(store.label_root), "identity": store.label_identity},
            }
            for store in stores
        },
        "backend": results["backend"],
        "files": files,
    }
    atomic_write_json(output_root / "walk-forward-manifest.json", payload)
    return payload


__all__ = [
    "MODEL_NAME",
    "assemble_table",
    "block_bootstrap_mean",
    "build_folds",
    "daily_rank_ic",
    "fast_grouped_spearman",
    "feature_identity",
    "label_identity",
    "load_split_table",
    "locate_stores",
    "read_split_rank_labels",
    "run_walk_forward",
    "summarize_cells",
    "worker_plan",
    "write_manifest",
]
