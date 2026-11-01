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
from multiprocessing import get_context
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
from stock_forecasting.model_lineage import (
    LINEAGE_NAME,
    MODELS_DIR,
    copy_fold_models,
    fold_models_dir,
    launch_models,
    lineage_payload,
    link_file,
    model_fold_indexes,
    owning_launch,
    prediction_key,
    prune_models,
    read_model_manifest,
    registry_key,
    stamp_superseded,
    training_key,
    write_lineage,
    write_model_manifest,
)
from stock_forecasting.resume_sources import (
    resume_candidates,
    resume_summary,
    stamp_resume_source,
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
TASK_TIMEOUT_ENV = "FIN_TS_LGBM_TASK_TIMEOUT_SECONDS"
# Wall-time budget of one training task (one fold, a chunk of horizons, every seed) once it
# is the oldest pending task; a liveness guard, not a performance target.
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


def task_timeout_seconds() -> int:
    """Per-task wall-time budget: ``TASK_TIMEOUT_ENV`` (positive integer) or the default."""
    override = os.environ.get(TASK_TIMEOUT_ENV)
    if not override:
        return TASK_TIMEOUT_SECONDS
    try:
        value = int(override)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{TASK_TIMEOUT_ENV} must be a positive integer of seconds")
    return value


def _bounded_training_map(function, tasks, count: int, timeout: float):
    """Yield results in submission order with at most 2 * count pending training tasks.

    Same bounded spawn pool as ``rank_labels._bounded_map`` but with the training task
    budget: that helper's 30-minute limit fits label buckets, not fold training. It is
    kept here because ``data/rank_labels.py`` is byte-hashed into every label store
    identity, so changing it would orphan the built stores.
    """
    if count == 1:
        for task in tasks:
            yield function(task)
        return
    with get_context("spawn").Pool(processes=count) as pool:
        iterator, pending, exhausted = iter(tasks), [], False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * count:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    pending.append(pool.apply_async(function, (task,)))
            if pending:
                # Pool termination cancels and joins all children on timeout or error.
                yield pending.pop(0).get(timeout=timeout)


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
        "task_timeout_seconds": task_timeout_seconds(),
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
    model_path: Path | None = None,
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
    scores = booster.predict(X_pred, num_iteration=best).astype(np.float32)
    if model_path is not None:
        # 2026-11-02 rule: the model is part of the result. Saved at the best iteration and
        # read back before the fold counts as done: a file that does not reproduce the
        # predictions is a failure, not a warning.
        booster.save_model(str(model_path), num_iteration=best)
        reloaded = lgb.Booster(model_file=str(model_path)).predict(X_pred).astype(np.float32)
        if not np.allclose(reloaded, scores, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                f"Saved LightGBM model does not reproduce its predictions: {model_path}"
            )
    return scores, best, score


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
    model_path: Path | None = None,
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
    scores = model.predict(X_pred).astype(np.float32)
    if model_path is not None:
        import joblib

        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path).predict(X_pred).astype(np.float32)
        if not np.allclose(reloaded, scores, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Saved model does not reproduce its predictions: {model_path}")
    return scores, int(model.n_iter_), float(score)


def score_table_with_launch(
    table: pd.DataFrame,
    launch: Path,
    *,
    feature_columns: list[str],
    start: str,
    end: str,
    fold: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scores of every sample with a signal date in [start, end) from a launch's saved models
    (one row per sample and model: each seed and the mean ensemble; the ensemble rows carry
    per-(date, market) ranks, 1 = highest score). Each date is scored by the model fold that
    ``select_model_fold`` picks unless ``fold`` names one. Returns the frame and a record of
    which fold scored which dates."""
    from stock_forecasting.model_lineage import read_lineage, select_model_fold

    launch = Path(launch)
    results = json.loads((launch / "results.json").read_text(encoding="utf-8"))
    horizons = [int(h) for h in results["horizons"]]
    seeds = tuple(int(seed) for seed in results["seeds"])
    model_folds = [int(index) for index in results.get("model_folds") or []]
    if not model_folds:
        raise ValueError(f"{launch} kept no models (it predates the 2026-11-02 rule)")
    model_name = str(results["model"])
    dates = pd.DatetimeIndex(table["signal_date"]).normalize()
    inside = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    if not inside.any():
        raise ValueError(f"No samples with a signal date in [{start}, {end})")
    chosen: dict[str, int] = {}
    for day in sorted({d.strftime("%Y-%m-%d") for d in dates[inside]}):
        chosen[day] = (
            fold if fold is not None else select_model_fold(results["folds"], model_folds, day)
        )
    if fold is not None and fold not in model_folds:
        raise ValueError(f"Fold {fold} has no saved models; model folds: {model_folds}")
    frames = []
    keys = ["symbol", "market", "signal_date", "split", "ordinal"]
    for index in sorted(set(chosen.values())):
        days = pd.DatetimeIndex([pd.Timestamp(d) for d, i in chosen.items() if i == index])
        mask = inside & dates.isin(days)
        rows = table.loc[mask]
        manifest, scores = score_with_fold_models(
            fold_models_dir(launch, index),
            rows[feature_columns].to_numpy(dtype=np.float32),
            horizons,
            seeds,
        )
        if list(manifest.get("feature_columns") or []) != list(feature_columns):
            raise ValueError(f"Saved models of fold {index} use other feature columns")
        base = rows[keys].reset_index(drop=True)
        base["fold"] = index
        for seed in seeds:
            frame = base.copy()
            frame["model"] = f"{model_name}-seed{seed}"
            for h in horizons:
                frame[score_column(h)] = scores[(h, seed)]
            frames.append(frame)
        ensemble = base.copy()
        ensemble["model"] = model_name
        for h in horizons:
            ensemble[score_column(h)] = np.mean(
                [scores[(h, seed)] for seed in seeds], axis=0
            ).astype(np.float32)
            ensemble[f"rank_h{h:02d}"] = (
                ensemble.groupby(["signal_date", "market"])[score_column(h)]
                .rank(ascending=False, method="first")
                .astype(np.int64)
            )
        frames.append(ensemble)
    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["signal_date", "market", "model", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    record = {
        "launch": launch.name,
        "model": model_name,
        "lineage": read_lineage(launch),
        "start": start,
        "end": end,
        "folds_by_date": chosen,
        "rows": len(out),
    }
    return out, record


def model_file_name(h: int, seed: int, backend: str) -> str:
    """One model file per (horizon, seed) in a fold's ``models/fold-XX/`` directory."""
    suffix = "txt" if backend == "lightgbm" else "joblib"
    return f"h{h:02d}-seed{seed}.{suffix}"


def load_fold_models(fold_dir: Path) -> tuple[dict[str, Any], dict[tuple[int, int], Callable]]:
    """The verified model manifest of a fold and one predictor per (horizon, seed)."""
    manifest = read_model_manifest(fold_dir)
    if manifest is None:
        raise ValueError(f"No verified model manifest in {fold_dir}")
    backend = str(manifest.get("backend"))
    predictors: dict[tuple[int, int], Callable] = {}
    for name, details in manifest["files"].items():
        path = Path(fold_dir) / name
        if backend == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(path))
            predictors[(int(details["h"]), int(details["seed"]))] = booster.predict
        else:
            import joblib

            predictors[(int(details["h"]), int(details["seed"]))] = joblib.load(path).predict
    return manifest, predictors


def score_with_fold_models(
    fold_dir: Path, X, horizons: list[int], seeds: tuple[int, ...]
) -> tuple[dict[str, Any], dict[tuple[int, int], np.ndarray]]:
    """Scores of ``X`` (feature columns in the manifest's order) from a fold's saved models."""
    manifest, predictors = load_fold_models(fold_dir)
    missing = [(h, seed) for h in horizons for seed in seeds if (h, seed) not in predictors]
    if missing:
        raise ValueError(f"{fold_dir} has no model for {missing}")
    scores = {
        (h, seed): np.asarray(predictors[(h, seed)](np.asarray(X)), dtype=np.float32)
        for h in horizons
        for seed in seeds
    }
    return manifest, scores


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
    model_dir = Path(task["model_dir"]) if task.get("model_dir") else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    for position, h in zip(task["positions"], task["horizons"], strict=True):
        y_train = y_train_all[:, position]
        y_valid = y_valid_all[:, position]
        valid_target_ranks = average_ranks(alpha_valid_all[:, position], valid_groups)
        for seed in task["seeds"]:
            started = time.monotonic()
            model_file = None if model_dir is None else model_file_name(h, seed, backend)
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
                None if model_dir is None else model_dir / model_file,
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
                    "model_file": model_file,
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
    context: str | None = None,
) -> None:
    """Persist one finished fold (every horizon and seed) so a restart can reuse it.

    The label variant, the feature tag and the context (schedule name and dataset identity)
    are stored so a resume never mixes folds trained on different labels, on another feature
    spec (protocol 13.18 runs F1 / F2 beside F0) or on another dataset or schedule (protocol
    13.20 extends the walk-forward on a re-prepared dataset).
    """
    arrays = {f"h{h:02d}_seed{seed}": values for (h, seed), values in fold_scores.items()}
    pending = partial_root / (_partial_name(fold_index) + ".pending")
    with pending.open("wb") as stream:
        np.savez(
            stream,
            records=json.dumps(fold_records),
            label_variant=np.array(label_variant or ""),
            features=np.array(features or ""),
            context=np.array(context or ""),
            **arrays,
        )
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(partial_root / _partial_name(fold_index))


def _load_partials(
    resume_root: Path | list[Path] | None,
    partial_root: Path,
    folds: list[Fold],
    horizons: list[int],
    seeds: tuple[int, ...],
    label_variant: str | None = None,
    features: str | None = None,
    context: str | None = None,
) -> tuple[
    dict[int, tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]],
    dict[Path, list[int]],
]:
    """Finished folds of earlier runs (one or several candidates, newest first) that cover
    every horizon and seed asked for now; returns them and, per source used, its fold indexes.

    Folds saved under another label variant (or by the standard labels when a variant is
    asked for), under another feature tag or under another context (schedule and dataset)
    are ignored; folds saved before a tag existed are ignored whenever that tag is asked for.
    """
    resumed: dict[int, tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]] = {}
    contributions: dict[Path, list[int]] = {}
    for source_root in resume_candidates(resume_root):
        for fold in folds:
            if fold.index in resumed:
                continue
            _load_partial(
                source_root,
                partial_root,
                fold,
                horizons,
                seeds,
                label_variant,
                features,
                context,
                resumed,
                contributions,
            )
    return resumed, contributions


def _load_partial(
    source_root: Path,
    partial_root: Path,
    fold: Fold,
    horizons: list[int],
    seeds: tuple[int, ...],
    label_variant: str | None,
    features: str | None,
    context: str | None,
    resumed: dict,
    contributions: dict[Path, list[int]],
) -> None:
    source = source_root / "partial"
    path = source / _partial_name(fold.index)
    if not path.is_file():
        return
    with np.load(path, allow_pickle=False) as payload:
        keys = {f"h{h:02d}_seed{seed}" for h in horizons for seed in seeds}
        if not keys <= set(payload.files):
            return
        stored = str(payload["label_variant"]) if "label_variant" in payload.files else ""
        if stored != (label_variant or ""):
            return
        stored_features = str(payload["features"]) if "features" in payload.files else ""
        if stored_features != (features or ""):
            return
        stored_context = str(payload["context"]) if "context" in payload.files else ""
        if stored_context != (context or ""):
            return
        fold_scores = {
            (h, seed): np.asarray(payload[f"h{h:02d}_seed{seed}"], dtype=np.float32)
            for h in horizons
            for seed in seeds
        }
        fold_records = json.loads(str(payload["records"]))
    resumed[fold.index] = (fold_scores, fold_records)
    contributions.setdefault(source_root, []).append(fold.index)
    if path.resolve() != (partial_root / path.name).resolve():
        (partial_root / path.name).write_bytes(path.read_bytes())
    # The unfinished launch saved this fold's models (a P3 fold): carry them.
    source_models = fold_models_dir(source_root, fold.index)
    target_models = fold_models_dir(partial_root.parent, fold.index)
    if (
        source_models.resolve() != target_models.resolve()
        and read_model_manifest(source_models) is not None
    ):
        copy_fold_models(source_models, target_models)


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


def _context_dataset(context: str | None) -> str:
    """The dataset part of a resume context (everything after the schedule name)."""
    return (context or "").split("|", 1)[1] if context and "|" in context else ""


def _read_partial(
    path: Path,
    horizons: list[int],
    seeds: tuple[int, ...],
    label_variant: str | None,
    features: str | None,
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]], str] | None:
    """A fold's partial when it covers every horizon and seed under these labels and features;
    returns (scores, records, stored context)."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        keys = {f"h{h:02d}_seed{seed}" for h in horizons for seed in seeds}
        if not keys <= set(payload.files):
            return None
        stored = str(payload["label_variant"]) if "label_variant" in payload.files else ""
        if stored != (label_variant or ""):
            return None
        stored_features = str(payload["features"]) if "features" in payload.files else ""
        if stored_features != (features or ""):
            return None
        context = str(payload["context"]) if "context" in payload.files else ""
        fold_scores = {
            (h, seed): np.asarray(payload[f"h{h:02d}_seed{seed}"], dtype=np.float32)
            for h in horizons
            for seed in seeds
        }
        fold_records = json.loads(str(payload["records"]))
    return fold_scores, fold_records, context


def _load_extension(
    extend_from: Path | None,
    output_root: Path,
    table: pd.DataFrame,
    folds: list[Fold],
    horizons: list[int],
    seeds: tuple[int, ...],
    *,
    model_name: str,
    label_variant: str | None,
    features: str | None,
    context: str | None,
    model_folds: set[int],
) -> tuple[
    dict[int, tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]],
    dict[int, dict[str, Any]],
    dict[int, Path],
]:
    """Folds of a FINISHED launch of the same model this run carries on (2026-11-02 rule).

    A fold matches by its training window (``TRAINING_DEFINITION``), not by index or schedule
    name, on the same labels, feature tag and dataset. Then:

    * same prediction window and no model needed (or the source holds verified models for
      it): the source's partial is **linked** into this launch (no copy) and, for a P3 fold,
      its models are copied; the fold is not retrained;
    * same training window but a different prediction window, with verified source models:
      the saved models **predict** this run's rows (no retraining; returned as ``to_predict``);
    * a P3 fold whose source has no models, or any other fold: **trained** here.

    Returns the linked folds (scores and records), per-fold provenance for the lineage, and
    the folds to predict with saved models.
    """
    linked: dict[int, tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]] = {}
    sources: dict[int, dict[str, Any]] = {}
    to_predict: dict[int, Path] = {}
    if extend_from is None:
        return linked, sources, to_predict
    source_root = Path(extend_from)
    results_path = source_root / "results.json"
    if not results_path.is_file():
        raise ValueError(f"--extend-from needs a finished launch (results.json): {source_root}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("model") != model_name:
        raise ValueError(
            f"--extend-from launch trained {results.get('model')!r}, this run is {model_name!r}"
        )
    if (results.get("label_variant") or None) != (label_variant or None):
        raise ValueError(
            f"--extend-from launch used label variant {results.get('label_variant')!r}, "
            f"this run uses {label_variant!r}"
        )
    by_training = {training_key(entry): entry for entry in results.get("folds", [])}
    source_models = launch_models(source_root)
    partial_root = output_root / "partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(table["signal_date"])
    for fold in folds:
        entry = by_training.get(training_key(fold))
        if entry is None:
            continue
        source_index = int(entry["index"])
        source_partial = source_root / "partial" / _partial_name(source_index)
        loaded = _read_partial(source_partial, horizons, seeds, label_variant, features)
        if loaded is None:
            continue
        fold_scores, fold_records, stored_context = loaded
        if _context_dataset(stored_context) != _context_dataset(context):
            continue  # another dataset: its folds are not this run's folds
        has_models = source_index in source_models
        needs_model = fold.index in model_folds
        same_window = prediction_key(entry) == prediction_key(fold)
        if same_window:
            if needs_model and not has_models:
                sources[fold.index] = {
                    "how": "trained",
                    "reason": "P3 fold: the source launch kept no models",
                    "source_fold": source_index,
                }
                continue
            _, _, predict = fold_masks(dates, fold)
            if int(predict.sum()) != len(next(iter(fold_scores.values()))):
                continue  # the rows are not this table's rows of the fold
            mode = link_file(partial_root / _partial_name(fold.index), source_partial)
            for record in fold_records:
                record["fold"] = fold.index
                record["extended_from"] = {"launch": source_root.name, "fold": source_index}
            linked[fold.index] = (fold_scores, fold_records)
            sources[fold.index] = {
                "how": "linked" if mode == "link" else "copied",
                "predictions_launch": owning_launch(source_partial),
                "source_fold": source_index,
            }
            if needs_model:
                copy_fold_models(
                    fold_models_dir(source_root, source_index),
                    fold_models_dir(output_root, fold.index),
                )
                sources[fold.index]["models_from"] = source_root.name
            continue
        if has_models:
            to_predict[fold.index] = fold_models_dir(source_root, source_index)
            sources[fold.index] = {
                "how": "predicted",
                "models_from": source_root.name,
                "source_fold": source_index,
            }
    return linked, sources, to_predict


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
    resume_root: Path | list[Path] | None = None,
    model_name: str = MODEL_NAME,
    label_variant: str | None = None,
    resume_context: str | None = None,
    extend_from: Path | None = None,
    feature_spec: str | None = None,
) -> dict[str, Any]:
    """Train every fold, write predictions, targets, daily Rank IC and the cell summary.

    Each finished fold is written to ``output_root/partial/`` at once; ``resume_root`` names one
    or several earlier output roots whose partial folds are reused (a Pod restart costs at most
    one fold); every source used is stamped with ``resumed_into.json`` afterwards.

    2026-11-02 rule: the folds the schedule's P3 needs keep their models under
    ``models/fold-XX/`` (one file per horizon and seed, verified by reloading), every launch
    writes ``lineage.json``, and ``extend_from`` names a finished launch of the same model whose
    folds are carried on instead of retrained (see ``_load_extension``); that launch is then
    stamped ``superseded_by.json`` and loses its model files.
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
    model_folds = model_fold_indexes(folds, schedule)
    fold_sources: dict[int, dict[str, Any]] = {}
    resumed, fold_sources, to_predict = _load_extension(
        extend_from,
        output_root,
        table,
        folds,
        horizons,
        seeds,
        model_name=model_name,
        label_variant=label_variant,
        features=features,
        context=resume_context,
        model_folds=model_folds,
    )
    more, contributions = _load_partials(
        resume_root,
        partial_root,
        [fold for fold in folds if fold.index not in resumed and fold.index not in to_predict],
        horizons,
        seeds,
        label_variant,
        features,
        resume_context,
    )
    resumed.update(more)
    for fold_index in more:
        fold_sources[fold_index] = {"how": "resumed"}
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
        if fold.index in to_predict:
            # Same training window as a fold of the extended launch: its saved models score
            # this run's rows (the prediction window changed, e.g. a quarter that was cut at
            # the old data end); nothing is retrained.
            predicted_at = time.monotonic()
            manifest, fold_scores = score_with_fold_models(
                to_predict[fold.index], np.load(work / "X_pred.npy", mmap_mode="r"), horizons, seeds
            )
            if list(manifest.get("feature_columns") or []) != list(feature_columns):
                raise ValueError(f"Saved models of {to_predict[fold.index]} use other features")
            fold_records = []
            for (h, seed), values in fold_scores.items():
                details = manifest["files"][model_file_name(h, seed, str(manifest["backend"]))]
                fold_records.append(
                    {
                        "fold": fold.index,
                        "h": h,
                        "seed": seed,
                        "best_iteration": details.get("best_iteration"),
                        "validation_rank_ic": details.get("validation_rank_ic"),
                        "train_rows": fold_info[fold.index]["train_rows"],
                        "validation_rows": fold_info[fold.index]["validation_rows"],
                        "predict_rows": len(values),
                        "seconds": time.monotonic() - predicted_at,
                        "predicted_with": {
                            "launch": to_predict[fold.index].parent.parent.name,
                            "fold": int(manifest["fold"]["index"]),
                        },
                    }
                )
            scores[fold.index] = fold_scores
            records.extend(fold_records)
            _save_partial(
                partial_root,
                fold.index,
                fold_scores,
                fold_records,
                label_variant,
                features,
                resume_context,
            )
            if fold.index in model_folds:
                copy_fold_models(to_predict[fold.index], fold_models_dir(output_root, fold.index))
            _cleanup_work(work)
            print(json.dumps({"fold": fold.index, "predicted_with_saved_models": True}), flush=True)
            continue
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
                    "model_dir": (
                        str(fold_models_dir(output_root, fold.index))
                        if fold.index in model_folds
                        else None
                    ),
                }
            )
    fold_records: dict[int, list[dict[str, Any]]] = {}
    results = (
        _bounded_training_map(run_fold_task, tasks, count, plan["task_timeout_seconds"])
        if tasks
        else ()
    )
    for result in results:
        records.extend(result["records"])
        fold_records.setdefault(result["fold"], []).extend(result["records"])
        scores.setdefault(result["fold"], {}).update(result["predictions"])
        pending_chunks[result["fold"]] -= 1
        if pending_chunks[result["fold"]] == 0:
            if result["fold"] in model_folds:
                fold = next(f for f in folds if f.index == result["fold"])
                write_model_manifest(
                    fold_models_dir(output_root, fold.index),
                    kind="lgbm",
                    fold=fold_frame([fold]).to_dict(orient="records")[0],
                    files={
                        r["model_file"]: {
                            "h": r["h"],
                            "seed": r["seed"],
                            "best_iteration": r["best_iteration"],
                            "validation_rank_ic": r["validation_rank_ic"],
                        }
                        for r in fold_records[fold.index]
                    },
                    backend=backend,
                    model=model_name,
                    label_variant=label_variant,
                    features=features,
                    feature_columns=list(feature_columns),
                    horizons=list(horizons),
                    seeds=list(seeds),
                    context=resume_context,
                )
            _save_partial(
                partial_root,
                result["fold"],
                scores[result["fold"]],
                fold_records[result["fold"]],
                label_variant,
                features,
                resume_context,
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
        # Provenance of a resumed run: which launch trained which folds.
        "resume": resume_summary(contributions),
        "extended_from": None if extend_from is None else Path(extend_from).name,
        "model_folds": sorted(model_folds),
    }
    # Every P3 fold's models must be on disk and verified before the launch is a result.
    for index in sorted(model_folds):
        if read_model_manifest(fold_models_dir(output_root, index)) is None:
            raise RuntimeError(f"Fold {index} has no verified models under {MODELS_DIR}/")
    atomic_write_json(output_root / "results.json", results)
    key = registry_key("lgbm", model_name, label_variant, feature_spec)
    write_lineage(
        output_root,
        lineage_payload(
            launch=output_root.name,
            kind="lgbm",
            key=key,
            model=model_name,
            label_variant=label_variant,
            feature_spec=feature_spec,
            schedule=schedule,
            folds=folds,
            fold_sources=fold_sources,
            model_folds=model_folds,
            extended_from=None if extend_from is None else Path(extend_from).name,
            backend=backend,
            features=features,
            context=resume_context,
        ),
    )
    results["lineage"] = {"key": key, "file": LINEAGE_NAME}
    if extend_from is not None:
        # The extended launch is history now: its folds live on here (linked or re-predicted),
        # the registry points at this launch, and its model files go (this launch holds every
        # model P3 needs, verified above) — the user's 2026-11-02 rule, applied before the Pod
        # ends.
        carried = sorted(i for i, src in fold_sources.items() if src.get("how") != "trained")
        stamp_superseded(
            extend_from,
            superseded_by=output_root,
            folds=carried,
            model=model_name,
            label_variant=label_variant,
            pruned_models=prune_models(extend_from),
        )
    for source, keys in contributions.items():
        # The source keeps its partial/ and never gets a results.json of its own; the stamp
        # keeps later jobs from offering it again and the volume reset from deleting it.
        stamp_resume_source(
            source,
            resumed_into=output_root,
            keys=keys,
            model=model_name,
            label_variant=label_variant,
            features=features,
            context=resume_context,
        )
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
    if (output_root / LINEAGE_NAME).is_file():
        names.append(LINEAGE_NAME)
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
        # Saved models (2026-11-02 rule): downloaded and verified like the result files.
        "models": model_files(output_root),
    }
    atomic_write_json(output_root / "walk-forward-manifest.json", payload)
    return payload


def model_files(output_root: Path) -> dict[str, dict[str, Any]]:
    """Every file under ``models/`` with its SHA-256, keyed by launch-relative path."""
    models_root = output_root / MODELS_DIR
    found: dict[str, dict[str, Any]] = {}
    if not models_root.is_dir():
        return found
    for file in sorted(models_root.rglob("*")):
        if file.is_file() and not file.is_symlink():
            found[str(file.relative_to(output_root))] = {
                "sha256": sha256_file(file),
                "bytes": file.stat().st_size,
            }
    return found


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
