"""Walk-forward training of the cross-stock transformer (protocol 13.13 D0, A1-A4, D1).

Every (signal date, market) group is one batch. Per fold and seed the model is trained on the
fold's training dates, early-stopped on the validation dates' horizon-equal daily Rank IC and
then scores the fold's quarter. Outputs use the same files and columns as the B0 LightGBM
baseline, so the scorer, the backtest and the paired comparison read both alike.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.models.cross_stock import (
    MODEL_SCHEMA,
    CrossStockConfig,
    CrossStockModel,
    cross_sectional_loss,
    parameter_count,
)
from stock_forecasting.walk_forward import Fold, assign_folds, fold_frame, period_of
from stock_forecasting.walk_forward_lgbm import (
    BLOCK_LENGTHS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELLS,
    alpha_column,
    average_ranks,
    block_bootstrap_mean,
    cell_series,
    daily_rank_ic,
    fast_grouped_spearman,
    fold_masks,
    group_codes,
    rank_column,
    score_column,
    summarize_cells,
)

SCHEMA_VERSION = "cross-stock-walk-forward-v1"
DEFAULT_SEEDS = (42, 43, 44)
# Fixed before training (protocol 13.13 item 1); not tuned on any evaluation period.
TRAINING_PARAMS = {
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "max_epochs": 20,
    "patience": 4,
    "gradient_clip": 1.0,
    "lambda_corr": 1.0,
}
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "t-xs-scratch": {},
    "t-xs-a1-no-cross": {"inter_stock_attention": False},
    "t-xs-a2-mse-only": {"lambda_corr": 0.0},
    "t-xs-a3-no-features": {"feature_input": False},
    "t-xs-a4-no-gating": {"market_gating": False},
    # D1: frozen pluggable encoders replace the scratch patch transformer (cached embeddings).
    "t-xs-kronos-small": {"encoder": "kronos-small"},
    "t-xs-chronos2-small": {"encoder": "chronos2-small"},
}
BASELINE_MODEL = "t-lgbm-alpha"
DECISION_CELLS = ("short", "holding")
DECISION_PERIOD = "P1+P2"


@dataclass
class SampleTable:
    """All splits of one dataset in memory.

    Windows stay float32: raw share volumes exceed the float16 range, and the encoder
    contract hands raw OHLCV to every adapter.
    """

    windows: torch.Tensor  # (N, L, 5)
    features: torch.Tensor  # (N, F) float32
    ranks: torch.Tensor  # (N, H) float32, NaN = unlabelled
    alpha: np.ndarray  # (N, H) float32 raw alpha log returns
    dates: np.ndarray  # datetime64[ns], tz-naive
    markets: np.ndarray  # object
    symbols: np.ndarray  # object
    split: np.ndarray  # object
    ordinal: np.ndarray  # int64
    horizons: list[int]
    feature_columns: list[str]
    encoded: torch.Tensor | None = None  # (N, D_enc) cached frozen-encoder embeddings

    def __len__(self) -> int:
        return len(self.dates)


def table_from_caches(caches: dict[str, Any], *, windows_dtype=torch.float32) -> SampleTable:
    """Concatenate the split caches (train, validation, test) into one table."""
    order = ("train", "validation", "test")
    parts = [caches[split] for split in order if split in caches]
    if not parts:
        raise ValueError("At least one split cache is required")
    horizons, columns = parts[0].horizons, parts[0].feature_columns
    for cache in parts[1:]:
        if cache.horizons != horizons or cache.feature_columns != columns:
            raise ValueError("Split caches disagree on horizons or feature columns")
    windows = torch.cat(
        [torch.from_numpy(np.ascontiguousarray(c.windows)).to(windows_dtype) for c in parts]
    )
    features = torch.cat([torch.from_numpy(np.ascontiguousarray(c.features)) for c in parts])
    ranks = torch.cat([torch.from_numpy(np.ascontiguousarray(c.ranks)) for c in parts])
    alpha = np.concatenate([np.asarray(c.alpha, dtype=np.float32) for c in parts])
    dates = np.concatenate(
        [np.asarray(c.cutoff_ns, dtype="int64").astype("datetime64[ns]") for c in parts]
    )
    markets = np.concatenate(
        [np.asarray(c.markets, dtype=object)[np.asarray(c.market_codes)] for c in parts]
    )
    symbols = np.concatenate(
        [np.asarray(c.symbols, dtype=object)[np.asarray(c.symbol_codes)] for c in parts]
    )
    split = np.concatenate([np.full(len(c.cutoff_ns), c.split, dtype=object) for c in parts])
    ordinal = np.concatenate([np.arange(len(c.cutoff_ns), dtype=np.int64) for c in parts])
    dates = pd.DatetimeIndex(dates).normalize().to_numpy()
    return SampleTable(
        windows=windows,
        features=features,
        ranks=ranks,
        alpha=alpha,
        dates=dates,
        markets=markets,
        symbols=symbols,
        split=split,
        ordinal=ordinal,
        horizons=horizons,
        feature_columns=columns,
    )


@dataclass
class GroupIndex:
    """Rows of every (date, market) group as one contiguous slice of ``order``."""

    order: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    dates: np.ndarray  # per group, datetime64[ns]
    markets: np.ndarray  # per group
    labelled: np.ndarray  # per group, bool (every row carries finite labels)

    def rows(self, group: int) -> np.ndarray:
        start = int(self.starts[group])
        return self.order[start : start + int(self.sizes[group])]


def build_group_index(table: SampleTable) -> GroupIndex:
    codes = group_codes(table.dates, table.markets)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    first = order[starts]
    labelled_rows = torch.isfinite(table.ranks).all(dim=1).numpy()
    labelled = np.array(
        [labelled_rows[order[s : s + n]].all() for s, n in zip(starts, sizes, strict=True)]
    )
    return GroupIndex(
        order=order,
        starts=starts,
        sizes=sizes,
        dates=table.dates[first],
        markets=table.markets[first],
        labelled=labelled,
    )


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TensorStore:
    """Hold the table's tensors on the GPU when they fit, else gather from pinned host memory."""

    def __init__(self, table: SampleTable, device: torch.device, *, headroom_bytes: int) -> None:
        tensors = {"windows": table.windows, "features": table.features, "ranks": table.ranks}
        if table.encoded is not None:
            tensors["encoded"] = table.encoded
        needed = sum(t.numel() * t.element_size() for t in tensors.values())
        self.device = device
        self.resident = False
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
            self.resident = free > needed + headroom_bytes
        if self.resident:
            self.tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
        else:
            self.tensors = {
                k: (v.pin_memory() if device.type == "cuda" else v) for k, v in tensors.items()
            }
        self.plan = {
            "resident_on_device": self.resident,
            "device": str(device),
            "tensor_bytes": int(needed),
        }

    def batch(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(rows, dtype=torch.long)
        if self.resident:
            index = index.to(self.device)
            return {k: v.index_select(0, index) for k, v in self.tensors.items()}
        return {
            k: v.index_select(0, index).to(self.device, non_blocking=True)
            for k, v in self.tensors.items()
        }


def make_config(experiment: str, table: SampleTable) -> tuple[CrossStockConfig, float]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    overrides = dict(EXPERIMENTS[experiment])
    lambda_corr = float(overrides.pop("lambda_corr", TRAINING_PARAMS["lambda_corr"]))
    encoder = overrides.pop("encoder", None)
    if encoder is not None and table.encoded is None:
        raise ValueError(f"{experiment} needs cached {encoder} embeddings on the table")
    if encoder is None and table.encoded is not None:
        raise ValueError(f"{experiment} trains its own encoder; drop the cached embeddings")
    config = CrossStockConfig(
        window=int(table.windows.shape[1]),
        channels=int(table.windows.shape[2]),
        feature_dim=len(table.feature_columns),
        horizons=len(table.horizons),
        encoder_dim=None if table.encoded is None else int(table.encoded.shape[1]),
        **overrides,
    )
    return config, lambda_corr


def _forward(model: CrossStockModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch.get("windows") if "encoded" not in batch else None,
        batch["features"],
        encoded=batch.get("encoded"),
    )


def _predict(
    model: CrossStockModel,
    store: TensorStore,
    index: GroupIndex,
    groups: np.ndarray,
    *,
    autocast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Scores for every row of the groups; returns (rows, scores)."""
    model.eval()
    rows_out, scores_out = [], []
    with torch.no_grad():
        for group in groups:
            rows = index.rows(int(group))
            batch = store.batch(rows)
            with torch.autocast(store.device.type, dtype=torch.bfloat16, enabled=autocast):
                scores = _forward(model, batch)
            rows_out.append(rows)
            scores_out.append(scores.float().cpu().numpy())
    return np.concatenate(rows_out), np.concatenate(scores_out)


def _validation_ic(
    scores: np.ndarray, alpha_ranks: np.ndarray, groups: np.ndarray, sizes: np.ndarray
) -> float:
    values = [
        fast_grouped_spearman(scores[:, h], alpha_ranks[:, h], groups, sizes)
        for h in range(scores.shape[1])
    ]
    return float(np.mean(values))


@dataclass
class FoldPlan:
    fold: Fold
    train_groups: np.ndarray
    valid_groups: np.ndarray
    predict_groups: np.ndarray
    valid_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    valid_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    valid_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    valid_alpha_ranks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def plan_fold(
    table: SampleTable, index: GroupIndex, fold: Fold, *, min_group_size: int
) -> FoldPlan:
    dates = pd.DatetimeIndex(index.dates)
    train, valid, predict = fold_masks(dates, fold)
    usable = index.labelled & (index.sizes >= min_group_size)
    train_groups = np.flatnonzero(train & usable)
    valid_groups = np.flatnonzero(valid & usable)
    predict_groups = np.flatnonzero(predict)
    if not len(train_groups) or not len(valid_groups) or not len(predict_groups):
        raise ValueError(f"Fold {fold.name} has an empty train, validation or prediction slice")
    valid_rows = np.concatenate([index.rows(int(g)) for g in valid_groups])
    valid_codes = np.repeat(np.arange(len(valid_groups)), index.sizes[valid_groups])
    valid_sizes = index.sizes[valid_groups]
    alpha = table.alpha[valid_rows]
    alpha_ranks = np.column_stack(
        [average_ranks(alpha[:, h], valid_codes) for h in range(alpha.shape[1])]
    )
    return FoldPlan(
        fold=fold,
        train_groups=train_groups,
        valid_groups=valid_groups,
        predict_groups=predict_groups,
        valid_rows=valid_rows,
        valid_codes=valid_codes,
        valid_sizes=valid_sizes,
        valid_alpha_ranks=alpha_ranks,
    )


def train_fold(
    table: SampleTable,
    index: GroupIndex,
    store: TensorStore,
    plan: FoldPlan,
    *,
    config: CrossStockConfig,
    lambda_corr: float,
    seed: int,
    params: dict[str, Any],
    max_groups_per_epoch: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train one seed on one fold; returns (predict rows, scores, record)."""
    started = time.monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossStockModel(config).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    autocast = store.device.type == "cuda"
    best_ic, best_epoch, best_state, bad_epochs, history = -np.inf, -1, None, 0, []
    for epoch in range(int(params["max_epochs"])):
        model.train()
        groups = rng.permutation(plan.train_groups)
        if max_groups_per_epoch is not None:
            groups = groups[:max_groups_per_epoch]
        losses = []
        for group in groups:
            batch = store.batch(index.rows(int(group)))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(store.device.type, dtype=torch.bfloat16, enabled=autocast):
                scores = _forward(model, batch)
            loss, _parts = cross_sectional_loss(scores, batch["ranks"], lambda_corr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        rows, scores = _predict(model, store, index, plan.valid_groups, autocast=autocast)
        if not np.array_equal(rows, plan.valid_rows):
            raise RuntimeError("Validation rows changed order between planning and prediction")
        ic = _validation_ic(scores, plan.valid_alpha_ranks, plan.valid_codes, plan.valid_sizes)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_rank_ic": ic}
        )
        if ic > best_ic:
            best_ic, best_epoch, bad_epochs = ic, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(params["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    rows, scores = _predict(model, store, index, plan.predict_groups, autocast=autocast)
    record = {
        "fold": plan.fold.index,
        "seed": seed,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "validation_rank_ic": best_ic,
        "train_groups": len(plan.train_groups),
        "validation_rows": len(plan.valid_rows),
        "predict_rows": len(rows),
        "parameters": parameter_count(model),
        "seconds": time.monotonic() - started,
        "history": history,
    }
    return rows, scores.astype(np.float32), record


def paired_comparison(
    daily_model: pd.DataFrame,
    daily_baseline: pd.DataFrame,
    schedule: dict,
    *,
    model: str,
    baseline: str,
) -> list[dict[str, Any]]:
    """Model - baseline Rank IC on common dates per cell and period, with block-bootstrap CIs."""
    rows = []
    if daily_model.empty or daily_baseline.empty:
        return rows
    for cell, horizons in CELLS.items():
        series_model = cell_series(daily_model, model, horizons)
        series_base = cell_series(daily_baseline, baseline, horizons)
        common = series_model.index.intersection(series_base.index)
        if len(common) == 0:
            continue
        diff = (series_model.loc[common] - series_base.loc[common]).sort_index()
        names = period_of(pd.DatetimeIndex(diff.index), schedule).fillna("")
        scopes = {DECISION_PERIOD: names.isin(["P1", "P2"]).to_numpy()}
        for name in sorted(set(names) - {""}):
            scopes[name] = (names == name).to_numpy()
        for scope, mask in scopes.items():
            values = diff.to_numpy()[mask]
            if len(values) == 0:
                continue
            low, high = block_bootstrap_mean(
                values, BLOCK_LENGTHS[cell], replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
            )
            rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "cell": cell,
                    "period": scope,
                    "days": len(values),
                    "difference": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "block_length": BLOCK_LENGTHS[cell],
                    "lower_bound_positive": bool(low > 0),
                }
            )
    return rows


def decide_h_a(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """H-A: the paired difference's lower bound is above 0 in both short and holding cells."""
    by_cell = {row["cell"]: row for row in pairs if row["period"] == DECISION_PERIOD}
    verdicts = {cell: by_cell.get(cell, {}).get("lower_bound_positive") for cell in DECISION_CELLS}
    passed = all(verdicts.get(cell) is True for cell in DECISION_CELLS)
    any_cell = any(verdicts.get(cell) is True for cell in DECISION_CELLS)
    return {
        "hypothesis": "H-A",
        "period": DECISION_PERIOD,
        "cells": verdicts,
        "passed_both_cells": passed,
        "passed_any_cell": any_cell,
        "disposition": (
            "D0 is the candidate model; proceed to D1 and H-B"
            if any_cell
            else "B0 stays the strategy baseline; DL stays research"
        ),
    }


def run_cross_stock(
    table: SampleTable,
    *,
    experiment: str,
    folds: list[Fold],
    schedule: dict,
    output_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    min_group_size: int,
    device: str | None = None,
    params: dict[str, Any] | None = None,
    baseline_daily: pd.DataFrame | None = None,
    max_groups_per_epoch: int | None = None,
    headroom_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    started = time.monotonic()
    params = {**TRAINING_PARAMS, **(params or {})}
    config, lambda_corr = make_config(experiment, table)
    index = build_group_index(table)
    torch_device = resolve_device(device)
    store = TensorStore(table, torch_device, headroom_bytes=headroom_bytes)
    plans = [plan_fold(table, index, fold, min_group_size=min_group_size) for fold in folds]
    records, scores = [], {}
    for plan in plans:
        for seed in seeds:
            rows, values, record = train_fold(
                table,
                index,
                store,
                plan,
                config=config,
                lambda_corr=lambda_corr,
                seed=seed,
                params=params,
                max_groups_per_epoch=max_groups_per_epoch,
            )
            scores[(plan.fold.index, seed)] = (rows, values)
            records.append(record)
            print(
                json.dumps(
                    {
                        "fold": plan.fold.index,
                        "seed": seed,
                        "epochs": record["epochs"],
                        "validation_rank_ic": round(record["validation_rank_ic"], 4),
                        "seconds": round(record["seconds"], 1),
                    }
                ),
                flush=True,
            )
    horizons = table.horizons
    prediction_frames, target_frames = [], []
    for plan in plans:
        rows = scores[(plan.fold.index, seeds[0])][0]
        base = pd.DataFrame(
            {
                "symbol": table.symbols[rows],
                "market": table.markets[rows],
                "signal_date": table.dates[rows],
                "split": table.split[rows],
                "ordinal": table.ordinal[rows],
                "fold": plan.fold.index,
            }
        )
        targets = base.copy()
        ranks = table.ranks[rows].numpy()
        for position, h in enumerate(horizons):
            targets[alpha_column(h)] = table.alpha[rows, position]
            targets[rank_column(h)] = ranks[:, position]
        target_frames.append(targets)
        stacked = []
        for seed in seeds:
            seed_rows, values = scores[(plan.fold.index, seed)]
            if not np.array_equal(seed_rows, rows):
                raise RuntimeError("Seeds predicted different rows of the same fold")
            frame = base.copy()
            frame["model"] = f"{experiment}-seed{seed}"
            for position, h in enumerate(horizons):
                frame[score_column(h)] = values[:, position]
            prediction_frames.append(frame)
            stacked.append(values)
        ensemble = base.copy()
        ensemble["model"] = experiment
        mean = np.mean(stacked, axis=0).astype(np.float32)
        for position, h in enumerate(horizons):
            ensemble[score_column(h)] = mean[:, position]
        prediction_frames.append(ensemble)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    dates = pd.DatetimeIndex(targets["signal_date"])
    targets["period"] = period_of(dates, schedule).to_numpy()
    targets["fold"] = assign_folds(dates, folds).to_numpy()
    daily = daily_rank_ic(predictions, targets, horizons=horizons, minimum=min_group_size)
    cells = summarize_cells(daily, schedule)
    pairs = (
        paired_comparison(
            daily, baseline_daily, schedule, model=experiment, baseline=BASELINE_MODEL
        )
        if baseline_daily is not None
        else []
    )
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_root / "predictions.parquet", index=False)
    targets.to_parquet(output_root / "targets.parquet", index=False)
    daily.to_parquet(output_root / "daily_rank_ic.parquet", index=False)
    fits = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in records])
    fits.to_parquet(output_root / "fits.parquet", index=False)
    (output_root / "training_history.json").write_text(
        json.dumps([{k: r[k] for k in ("fold", "seed", "history")} for r in records]) + "\n"
    )
    per_horizon = (
        daily.groupby(["model", "market", "h"])["ic"].agg(["mean", "std", "count"]).reset_index()
        if not daily.empty
        else pd.DataFrame()
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "model_schema": MODEL_SCHEMA,
        "model": experiment,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "lambda_corr": lambda_corr,
        "training_params": params,
        "parameters": int(fits["parameters"].iloc[0]) if len(fits) else None,
        "seeds": list(seeds),
        "horizons": horizons,
        "folds": fold_frame(folds).to_dict(orient="records"),
        "device": store.plan,
        "cells": cells,
        "paired_vs_baseline": pairs,
        "decision": decide_h_a(pairs) if pairs else None,
        "per_market_horizon": per_horizon.to_dict(orient="records"),
        "fits": {
            "count": len(fits),
            "mean_epochs": float(fits["epochs"].mean()) if len(fits) else None,
            "mean_validation_rank_ic": (
                float(fits["validation_rank_ic"].mean()) if len(fits) else None
            ),
            "seconds": float(fits["seconds"].sum()) if len(fits) else 0.0,
        },
        "seconds": time.monotonic() - started,
    }
    atomic_write_json(output_root / "results.json", results)
    return results


def write_xs_manifest(
    output_root: Path, *, results: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    names = (
        "results.json",
        "predictions.parquet",
        "targets.parquet",
        "daily_rank_ic.parquet",
        "fits.parquet",
        "training_history.json",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cross-stock-walk-forward",
        "model": results["model"],
        "generated_at": datetime.now(UTC).isoformat(),
        "files": {
            name: {
                "sha256": sha256_file(output_root / name),
                "bytes": (output_root / name).stat().st_size,
            }
            for name in names
        },
        **extra,
    }
    atomic_write_json(output_root / "walk-forward-manifest.json", payload)
    return payload
