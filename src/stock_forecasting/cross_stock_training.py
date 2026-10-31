"""Walk-forward training of the cross-stock transformer (protocol 13.13 D0, A1-A4, D1).

Every (signal date, market) group is one batch. Per fold and seed the model is trained on the
fold's training dates, early-stopped on the validation dates' horizon-equal daily Rank IC and
then scores the fold's quarter. Outputs use the same files and columns as the B0 LightGBM
baseline, so the scorer, the backtest and the paired comparison read both alike.
"""

from __future__ import annotations

import json
import os
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
    residual_column,
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
    # (N, H) float32 residual targets of a label variant (protocol 13.16); None = standard labels
    residual: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.dates)


def table_from_caches(
    caches: dict[str, Any],
    *,
    windows_dtype=torch.float32,
    label_variants: dict[str, Any] | None = None,
) -> SampleTable:
    """Concatenate the split caches (train, validation, test) into one table.

    With ``label_variants`` (one ``LabelVariantArrays`` per split present in ``caches``) the
    variant's ranks replace the cache's standard ranks and its residual targets are attached.
    """
    order = ("train", "validation", "test")
    splits = [split for split in order if split in caches]
    parts = [caches[split] for split in splits]
    if not parts:
        raise ValueError("At least one split cache is required")
    horizons, columns = parts[0].horizons, parts[0].feature_columns
    for cache in parts[1:]:
        if cache.horizons != horizons or cache.feature_columns != columns:
            raise ValueError("Split caches disagree on horizons or feature columns")
    residual = None
    if label_variants is not None:
        if set(label_variants) != set(splits):
            raise ValueError("Label-variant arrays must cover exactly the cached splits")
        variants = [label_variants[split] for split in splits]
        for cache, variant in zip(parts, variants, strict=True):
            if len(variant.ranks) != len(cache.cutoff_ns):
                raise ValueError(f"{cache.split}: label-variant rows differ from the cache")
        ranks = torch.cat([torch.from_numpy(np.ascontiguousarray(v.ranks)) for v in variants])
        residual = np.concatenate([np.asarray(v.residual, dtype=np.float32) for v in variants])
    else:
        ranks = torch.cat([torch.from_numpy(np.ascontiguousarray(c.ranks)) for c in parts])
    windows = torch.cat(
        [torch.from_numpy(np.ascontiguousarray(c.windows)).to(windows_dtype) for c in parts]
    )
    features = torch.cat([torch.from_numpy(np.ascontiguousarray(c.features)) for c in parts])
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
        residual=residual,
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


def pad_multiple(device: torch.device) -> int:
    """Batch-size bucket for one (date, market) group: ``FIN_TS_XS_PAD_MULTIPLE`` (default 64 on
    CUDA, 0 = no padding elsewhere). Padding every group up to a multiple keeps the number of
    distinct shapes small, which is what lets ``torch.compile`` replay CUDA graphs instead of
    launching the model's few hundred tiny kernels from Python at every step."""
    raw = os.environ.get("FIN_TS_XS_PAD_MULTIPLE")
    if raw is None or raw.strip() == "":
        return 64 if device.type == "cuda" else 0
    value = int(raw)
    if value < 0:
        raise ValueError("FIN_TS_XS_PAD_MULTIPLE must be a non-negative integer")
    return value


def compile_mode(device: torch.device) -> str:
    """``FIN_TS_XS_COMPILE``: ``1`` (default on CUDA) compiles the model with CUDA-graph replay
    (``reduce-overhead``); ``0`` keeps eager execution."""
    if device.type != "cuda":
        return "eager"
    if os.environ.get("FIN_TS_XS_COMPILE", "1").strip() == "0":
        return "eager"
    return "reduce-overhead"


EXECUTION_ENV = "FIN_TS_XS_EXECUTION"
EXECUTION_MODES = ("step", "model", "eager")


def execution_mode(device: torch.device) -> str:
    """How a training step runs on CUDA (``FIN_TS_XS_EXECUTION``): ``step`` (default) replays
    forward + loss, backward, and gradient clipping + the fused AdamW update as CUDA graphs and
    gathers every group into static per-bucket buffers, so a step leaves only the gather,
    ``zero_grad`` and the loss accumulation to Python; ``model`` compiles the model alone (the
    2026-10-30 path, which left the loss, clipping and optimizer as ~50 eager launches per step
    and held an RTX PRO 4500 at ~50% utilization); ``eager`` compiles nothing.
    ``FIN_TS_XS_COMPILE=0`` still means eager; off CUDA the mode is always eager."""
    if device.type != "cuda" or os.environ.get("FIN_TS_XS_COMPILE", "1").strip() == "0":
        return "eager"
    raw = os.environ.get(EXECUTION_ENV, "step").strip().lower() or "step"
    if raw not in EXECUTION_MODES:
        raise ValueError(f"{EXECUTION_ENV} must be one of {', '.join(EXECUTION_MODES)}")
    return raw


def maybe_compile(model: torch.nn.Module, device: torch.device) -> tuple[Any, str]:
    """Return (callable module, mode); eager when compilation is off or unavailable."""
    mode = compile_mode(device)
    if mode == "eager":
        return model, "eager"
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
        return torch.compile(model, mode=mode, dynamic=False), mode
    except Exception as error:  # pragma: no cover - depends on the Pod's torch build
        return model, f"eager (compile unavailable: {error})"


class TensorStore:
    """Hold the table's tensors on the GPU when they fit, else gather from pinned host memory."""

    def __init__(self, table: SampleTable, device: torch.device, *, headroom_bytes: int) -> None:
        self.pad_multiple = pad_multiple(device)
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
            "pad_multiple": self.pad_multiple,
        }
        self.order: torch.Tensor | None = None
        # Static per-bucket buffers (step execution): a group batch is written in place, so
        # the compiled step sees the same addresses every time and replays without copies.
        self._buckets: dict[int, dict[str, torch.Tensor]] = {}

    def attach_order(self, order: np.ndarray) -> None:
        """Keep the group row order where the gather runs, so a group batch is one slice and
        one index_select with no per-step host-to-device copy or host sync."""
        tensor = torch.as_tensor(np.ascontiguousarray(order), dtype=torch.long)
        self.order = tensor.to(self.device) if self.resident else tensor

    def gather(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.resident:
            index = index.to(self.device)
            return {k: v.index_select(0, index) for k, v in self.tensors.items()}
        return {
            k: v.index_select(0, index).to(self.device, non_blocking=True)
            for k, v in self.tensors.items()
        }

    def batch(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        return self.gather(torch.as_tensor(rows, dtype=torch.long))

    def group_batch(self, index: GroupIndex, group: int) -> dict[str, torch.Tensor]:
        """Batch of one (date, market) group through the attached order, padded to the bucket.

        With the order on the same device as the tensors and padding on, the rows are gathered
        straight into that bucket's static buffers (one ``index_select`` per tensor plus the
        padding fills) and the same dict is returned every time: callers must consume it
        before requesting the next group.
        """
        size = int(index.sizes[group])
        if self.order is None:
            return self.pad(self.batch(index.rows(group)), size)
        start = int(index.starts[group])
        rows = self.order[start : start + size]
        if self.pad_multiple <= 0 or rows.device != self.device:
            return self.pad(self.gather(rows), size)
        padded = -(-size // self.pad_multiple) * self.pad_multiple
        bucket = self._bucket(padded)
        for key, value in self.tensors.items():
            torch.index_select(value, 0, rows, out=bucket[key][:size])
            if padded > size:
                bucket[key][size:].fill_(float("nan") if key in ("features", "ranks") else 0.0)
        valid = bucket["valid"]
        valid[:size].fill_(True)
        if padded > size:
            valid[size:].fill_(False)
        return bucket

    def _bucket(self, padded: int) -> dict[str, torch.Tensor]:
        bucket = self._buckets.get(padded)
        if bucket is None:
            bucket = {
                key: torch.empty((padded, *value.shape[1:]), dtype=value.dtype, device=self.device)
                for key, value in self.tensors.items()
            }
            bucket["valid"] = torch.empty(padded, dtype=torch.bool, device=self.device)
            if self.device.type == "cuda":
                # Tell CUDA-graph trees the addresses never change: no per-replay input copy.
                from torch._dynamo import mark_static_address

                for tensor in bucket.values():
                    mark_static_address(tensor)
            self._buckets[padded] = bucket
        return bucket

    def pad(self, batch: dict[str, torch.Tensor], size: int) -> dict[str, torch.Tensor]:
        """Append masked rows up to the next multiple of ``pad_multiple``; ``valid`` marks the
        real rows. Padded features and ranks are NaN (left out of the z-score and the loss),
        padded windows / embeddings are zeros (never attended to)."""
        if self.pad_multiple <= 0:
            return batch
        padded = -(-size // self.pad_multiple) * self.pad_multiple
        extra = padded - size
        valid = torch.ones(padded, dtype=torch.bool, device=self.device)
        if extra:
            valid[size:] = False
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if not extra:
                out[key] = value
                continue
            fill_value = float("nan") if key in ("features", "ranks") else 0.0
            fill = torch.full(
                (extra, *value.shape[1:]), fill_value, dtype=value.dtype, device=value.device
            )
            out[key] = torch.cat([value, fill], dim=0)
        out["valid"] = valid
        return out


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


def _forward(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch.get("windows") if "encoded" not in batch else None,
        batch["features"],
        encoded=batch.get("encoded"),
        valid=batch.get("valid"),
    )


def _step_loss(model: Any, batch: dict[str, torch.Tensor], lambda_corr: float) -> torch.Tensor:
    scores = _forward(model, batch)
    loss, _parts = cross_sectional_loss(scores, batch["ranks"], lambda_corr, check_labels=False)
    return loss


class StepRunner:
    """One training step or one prediction pass under an execution mode (``execution_mode``).

    ``step``: forward + loss, backward, and clipping + fused AdamW are compiled with
    ``reduce-overhead`` and replayed as CUDA graphs. ``model``: only the model is compiled.
    ``eager``: nothing is. The model and optimizer objects are shared, so a runner can be
    replaced by one in a cheaper mode mid-fold without losing training state.
    """

    def __init__(
        self,
        model: CrossStockModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        lambda_corr: float,
        gradient_clip: float,
        mode: str,
    ) -> None:
        if mode not in EXECUTION_MODES:
            raise ValueError(f"Unknown execution mode: {mode}")
        self.model, self.optimizer, self.device, self.mode = model, optimizer, device, mode
        self.autocast = device.type == "cuda"
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._clip = float(gradient_clip)
        self._mark_step: Any = None

        def loss_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return _step_loss(model, batch, lambda_corr)

        self.loss_fn: Any = loss_fn
        self.update_fn: Any = self._update
        self.predict_fn: Any = model
        if mode == "eager":
            return
        from torch._dynamo import config as dynamo_config

        dynamo_config.cache_size_limit = max(int(dynamo_config.cache_size_limit), 64)
        compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        self.predict_fn = compiled_model
        if mode == "model":

            def model_loss_fn(batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return _step_loss(compiled_model, batch, lambda_corr)

            self.loss_fn = model_loss_fn
            return
        self.loss_fn = torch.compile(loss_fn, mode="reduce-overhead", dynamic=False)
        self.update_fn = torch.compile(self._update, mode="reduce-overhead", dynamic=False)
        self._mark_step = getattr(
            getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
        )

    def _update(self) -> None:
        torch.nn.utils.clip_grad_norm_(self._params, self._clip)
        self.optimizer.step()

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward, loss, backward, clip, update; returns the detached loss (no host sync)."""
        if self._mark_step is not None:
            self._mark_step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            loss = self.loss_fn(batch)
        loss.backward()
        self.update_fn()
        return loss.detach()

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            return _forward(self.predict_fn, batch)


def _predict(
    runner: StepRunner, store: TensorStore, index: GroupIndex, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scores for every row of the groups; returns (rows, scores)."""
    runner.model.eval()
    rows_out, scores_out = [], []
    with torch.no_grad():
        for group in groups:
            batch = store.group_batch(index, int(group))
            scores = runner.predict(batch)
            rows_out.append(index.rows(int(group)))
            # Drop the padded rows; clone because a CUDA-graph replay reuses the output buffer.
            scores_out.append(scores[: int(index.sizes[int(group)])].float().clone())
    # A single device-to-host copy per call; per-group .cpu() calls serialized the GPU.
    return np.concatenate(rows_out), torch.cat(scores_out).cpu().numpy()


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
    # One fused kernel per step instead of per-parameter updates: the per-group steps are
    # launch-bound (one (date, market) group per step), so kernel count is the wall clock.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        fused=store.device.type == "cuda",
    )

    def make_runner(mode: str) -> StepRunner:
        return StepRunner(
            model,
            optimizer,
            store.device,
            lambda_corr=lambda_corr,
            gradient_clip=params["gradient_clip"],
            mode=mode,
        )

    # Padded groups have a handful of static shapes, so the compiled regions replay CUDA graphs
    # instead of launching every small kernel from Python: the steps are launch-bound, the
    # model's compute per group being well under a millisecond on a data-centre GPU.
    requested = execution_mode(store.device)
    try:
        runner = make_runner(requested)
        execution_label = requested
    except Exception as error:  # pragma: no cover - depends on the Pod's torch build
        print(f"cross-stock: {requested} execution unavailable, eager: {error}", flush=True)
        runner, execution_label = make_runner("eager"), f"eager (compile unavailable: {error})"
    best_ic, best_epoch, best_state, bad_epochs, history = -np.inf, -1, None, 0, []
    for epoch in range(int(params["max_epochs"])):
        model.train()
        groups = rng.permutation(plan.train_groups)
        if max_groups_per_epoch is not None:
            groups = groups[:max_groups_per_epoch]
        # The epoch loss stays on the device: a float(loss) per step forced a host sync that
        # kept the GPU idle while Python queued the next step.
        loss_sum, steps = torch.zeros((), device=store.device), 0
        for group in groups:
            batch = store.group_batch(index, int(group))
            try:
                loss = runner.train_step(batch)
            except Exception as error:  # pragma: no cover - compiler backend failures
                if runner.mode == "eager":
                    raise
                # A failed step changed nothing the next attempt does not reset (grads are
                # zeroed first; the fused update is one kernel), so redo it one mode down.
                fallback = "model" if runner.mode == "step" else "eager"
                print(
                    f"cross-stock: {runner.mode} execution failed, back to {fallback}: {error}",
                    flush=True,
                )
                execution_label = (
                    f"{fallback} (fallback from {runner.mode}: {type(error).__name__})"
                )
                runner = make_runner(fallback)
                loss = runner.train_step(batch)
            loss_sum += loss
            steps += 1
        losses = [float(loss_sum) / max(steps, 1)]
        rows, scores = _predict(runner, store, index, plan.valid_groups)
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
    rows, scores = _predict(runner, store, index, plan.predict_groups)
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
        "execution": {"mode": execution_label, "pad_multiple": store.pad_multiple},
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


def seed_stability(
    cells: list[dict[str, Any]], experiment: str, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Per cell and period: dispersion of the single-seed Rank ICs and the ensemble's gain.

    ``single_seed_adequate`` holds when both the seed range and the ensemble gain (ensemble
    minus mean single seed) stay below half the ensemble's 95% interval half-width, so a
    single seed would land inside the ensemble's own sampling noise.
    """
    rows = []
    by_key = {(row["model"], row["cell"], row["period"]): row for row in cells}
    scopes = sorted({(row["cell"], row["period"]) for row in cells if row["model"] == experiment})
    for cell, period in scopes:
        ensemble = by_key[(experiment, cell, period)]
        values = [
            by_key[(f"{experiment}-seed{seed}", cell, period)]["rank_ic"]
            for seed in seeds
            if (f"{experiment}-seed{seed}", cell, period) in by_key
        ]
        if len(values) < 2:
            continue
        array = np.asarray(values, dtype=np.float64)
        half_width = (ensemble["ci95_high"] - ensemble["ci95_low"]) / 2.0
        seed_range = float(array.max() - array.min())
        gain = float(ensemble["rank_ic"] - array.mean())
        rows.append(
            {
                "model": experiment,
                "cell": cell,
                "period": period,
                "seeds": len(values),
                "seed_rank_ic": [float(v) for v in values],
                "seed_mean": float(array.mean()),
                "seed_std": float(array.std(ddof=1)),
                "seed_range": seed_range,
                "ensemble_rank_ic": float(ensemble["rank_ic"]),
                "ensemble_gain": gain,
                "ensemble_ci95_half_width": float(half_width),
                "single_seed_adequate": bool(
                    np.isfinite(half_width)
                    and seed_range < 0.5 * half_width
                    and abs(gain) < 0.5 * half_width
                ),
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
    resume_root: Path | None = None,
    baseline_model: str = BASELINE_MODEL,
    baseline_daily_residual: pd.DataFrame | None = None,
    label_variant: str | None = None,
) -> dict[str, Any]:
    """Train every fold and seed; each finished (fold, seed) is written to ``output_root/partial``
    immediately and reused from ``resume_root`` after a restart.

    With a label variant (``table.residual`` set, ``label_variant`` named) the scores are also
    evaluated against the residual targets (protocol 13.16) and, when
    ``baseline_daily_residual`` is given, paired against the baseline on that target too.
    """
    started = time.monotonic()
    if (table.residual is None) != (label_variant is None):
        raise ValueError("A label variant needs both the residual targets and its name")
    partial_root = output_root / "partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    resumed = _load_xs_partials(
        resume_root, partial_root, experiment, folds, seeds, label_variant=label_variant
    )
    params = {**TRAINING_PARAMS, **(params or {})}
    config, lambda_corr = make_config(experiment, table)
    index = build_group_index(table)
    torch_device = resolve_device(device)
    store = TensorStore(table, torch_device, headroom_bytes=headroom_bytes)
    store.attach_order(index.order)
    plans = [plan_fold(table, index, fold, min_group_size=min_group_size) for fold in folds]
    records, scores = [], {}
    for plan in plans:
        for seed in seeds:
            if (plan.fold.index, seed) in resumed:
                rows, values, record = resumed[(plan.fold.index, seed)]
                scores[(plan.fold.index, seed)] = (rows, values)
                records.append(record)
                print(
                    json.dumps({"fold": plan.fold.index, "seed": seed, "resumed": True}), flush=True
                )
                continue
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
            _save_xs_partial(
                partial_root,
                experiment,
                plan.fold.index,
                seed,
                rows,
                values,
                record,
                label_variant=label_variant,
            )
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
            if table.residual is not None:
                targets[residual_column(h)] = table.residual[rows, position]
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
            daily, baseline_daily, schedule, model=experiment, baseline=baseline_model
        )
        if baseline_daily is not None
        else []
    )
    # Protocol 13.16: the residual target is what the variant model predicts; the raw alpha
    # target above is what a long portfolio earns.
    daily_residual, cells_residual, pairs_residual = None, [], []
    if table.residual is not None:
        daily_residual = daily_rank_ic(
            predictions, targets, horizons=horizons, minimum=min_group_size, target=residual_column
        )
        cells_residual = summarize_cells(daily_residual, schedule)
        if baseline_daily_residual is not None:
            pairs_residual = paired_comparison(
                daily_residual,
                baseline_daily_residual,
                schedule,
                model=experiment,
                baseline=baseline_model,
            )
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_root / "predictions.parquet", index=False)
    targets.to_parquet(output_root / "targets.parquet", index=False)
    daily.to_parquet(output_root / "daily_rank_ic.parquet", index=False)
    if daily_residual is not None:
        daily_residual.to_parquet(output_root / "daily_rank_ic_residual.parquet", index=False)
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
        "label_variant": label_variant,
        "baseline_model": baseline_model if baseline_daily is not None else None,
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
        "cells_residual": cells_residual,
        "seed_stability": seed_stability(cells, experiment, seeds),
        "paired_vs_baseline": pairs,
        "paired_vs_baseline_residual": pairs_residual,
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


def _xs_partial_name(fold_index: int, seed: int) -> str:
    return f"fold-{fold_index:02d}-seed-{seed}.npz"


def _save_xs_partial(
    partial_root: Path,
    experiment: str,
    fold_index: int,
    seed: int,
    rows: np.ndarray,
    values: np.ndarray,
    record: dict[str, Any],
    *,
    label_variant: str | None = None,
) -> None:
    pending = partial_root / (_xs_partial_name(fold_index, seed) + ".pending")
    with pending.open("wb") as stream:
        np.savez(
            stream,
            rows=rows,
            values=values,
            record=json.dumps(record),
            experiment=experiment,
            label_variant=label_variant or "",
        )
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(partial_root / _xs_partial_name(fold_index, seed))


def _load_xs_partials(
    resume_root: Path | None,
    partial_root: Path,
    experiment: str,
    folds: list[Fold],
    seeds: tuple[int, ...],
    label_variant: str | None = None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Finished (fold, seed) pairs of an earlier run of the same experiment and label variant.

    Partials saved under another label variant (or by the standard labels when a variant is
    asked for) are ignored.
    """
    if resume_root is None:
        return {}
    source = Path(resume_root) / "partial"
    resumed = {}
    for fold in folds:
        for seed in seeds:
            path = source / _xs_partial_name(fold.index, seed)
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as payload:
                if str(payload["experiment"]) != experiment:
                    continue
                stored = str(payload["label_variant"]) if "label_variant" in payload.files else ""
                if stored != (label_variant or ""):
                    continue
                resumed[(fold.index, seed)] = (
                    np.asarray(payload["rows"], dtype=np.int64),
                    np.asarray(payload["values"], dtype=np.float32),
                    json.loads(str(payload["record"])),
                )
            if path.resolve() != (partial_root / path.name).resolve():
                (partial_root / path.name).write_bytes(path.read_bytes())
    return resumed


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
    if (output_root / "daily_rank_ic_residual.parquet").is_file():
        names = (*names, "daily_rank_ic_residual.parquet")
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
