"""Walk-forward LightGBM baseline: metrics, bootstrap, worker plan and an end-to-end fold run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stock_forecasting.walk_forward_lgbm as module
from stock_forecasting.walk_forward import quarterly_folds
from stock_forecasting.walk_forward_lgbm import (
    MODEL_NAME,
    WORKERS_ENV,
    alpha_column,
    block_bootstrap_mean,
    daily_rank_ic,
    fast_grouped_spearman,
    fold_masks,
    rank_column,
    run_walk_forward,
    score_column,
    summarize_cells,
    worker_plan,
)

SCHEDULE = {
    "schema": "walk-forward-schedule-v1",
    "name": "fixture",
    "calendar": "XTAI",
    "train_start": "2023-01-02",
    "data_end": "2024-07-01",
    "first_quarter": "2024Q1",
    "last_quarter": "2024Q2",
    "embargo_sessions": 15,
    "validation_months": 3,
    "periods": {"P1": ["2024-01-01", "2024-04-01"], "P2": ["2024-04-01", "2024-07-01"]},
}
HORIZONS = [1, 5, 10]


def _sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", "2024-06-28")


def _table(seed: int = 0, symbols_per_market: int = 40) -> pd.DataFrame:
    """Synthetic samples whose first feature carries real cross-sectional signal."""
    rng = np.random.default_rng(seed)
    sessions = _sessions()
    rows = []
    features = [f"f{i}" for i in range(6)]
    for market, suffix in (("TWSE", "TW"), ("TPEX", "TWO")):
        for day in sessions:
            values = rng.standard_normal((symbols_per_market, len(features))).astype(np.float32)
            noise = rng.standard_normal((symbols_per_market, len(HORIZONS)))
            alpha = 0.03 * values[:, :1] + 0.01 * noise
            ranks = pd.DataFrame(alpha).rank().to_numpy() / symbols_per_market
            for i in range(symbols_per_market):
                record = {
                    "ordinal": 0,
                    "symbol": f"{1000 + i}.{suffix}",
                    "split": "train",
                    "market": market,
                    "signal_date": day,
                }
                record.update(dict(zip(features, values[i], strict=True)))
                for position, h in enumerate(HORIZONS):
                    record[rank_column(h)] = np.float32(ranks[i, position])
                    record[alpha_column(h)] = np.float32(alpha[i, position])
                rows.append(record)
    table = pd.DataFrame(rows)
    table["ordinal"] = np.arange(len(table))
    return table


def test_fast_grouped_spearman_matches_pandas() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(5), 30)
    predictions = rng.standard_normal(150)
    targets = 0.5 * predictions + rng.standard_normal(150)
    sizes = np.bincount(groups)
    target_ranks = module.average_ranks(targets, groups)
    fast = fast_grouped_spearman(predictions, target_ranks, groups, sizes)
    frame = pd.DataFrame({"g": groups, "p": predictions, "t": targets})
    expected = (
        frame.groupby("g")
        .apply(lambda block: block["p"].corr(block["t"], method="spearman"), include_groups=False)
        .mean()
    )
    assert fast == pytest.approx(expected, abs=1e-12)
    assert fast_grouped_spearman(np.zeros(4), np.arange(4.0), np.zeros(4, int), np.array([4])) == 0


def test_block_bootstrap_mean_is_deterministic_and_brackets_the_mean() -> None:
    values = np.linspace(-0.1, 0.3, 200)
    low, high = block_bootstrap_mean(values, 20, replicates=500, seed=11)
    again = block_bootstrap_mean(values, 20, replicates=500, seed=11)
    assert (low, high) == again
    assert low < values.mean() < high
    assert all(np.isnan(v) for v in block_bootstrap_mean(np.array([]), 5, replicates=10, seed=1))


def test_daily_rank_ic_and_cells_follow_the_signal() -> None:
    table = _table(symbols_per_market=30)
    keys = ["symbol", "market", "signal_date", "split", "ordinal"]
    targets = table[keys + [alpha_column(h) for h in HORIZONS]].copy()
    scores = table[keys].copy()
    scores["model"] = MODEL_NAME
    for h in HORIZONS:
        scores[score_column(h)] = table["f0"].to_numpy()
    daily = daily_rank_ic(scores, targets, horizons=HORIZONS, minimum=10)
    assert set(daily["market"]) == {"TWSE", "TPEX"}
    assert daily["ic"].mean() > 0.5
    cells = summarize_cells(daily, SCHEDULE)
    short = [c for c in cells if c["cell"] == "short" and c["period"] == "P1+P2"]
    assert len(short) == 1 and short[0]["ci95_low"] > 0 and short[0]["positive_share"] > 0.9
    assert {c["period"] for c in cells} == {"P1+P2", "P1", "P2"}
    assert {c["cell"] for c in cells} == {"short", "holding"}  # no h>=11 in this fixture


def test_worker_plan_is_bounded_by_cpus_memory_tasks_and_override(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.delenv(WORKERS_ENV, raising=False)
    monkeypatch.setattr(module, "detect_visible_cpu_count", lambda: 16)
    monkeypatch.setattr(
        module, "detect_available_memory", lambda: SimpleNamespace(available_bytes=64 * gib)
    )
    count, plan = worker_plan(None, 196)
    assert (count, plan["threads_per_worker"]) == (4, 4)
    monkeypatch.setattr(
        module, "detect_available_memory", lambda: SimpleNamespace(available_bytes=10 * gib)
    )
    assert worker_plan(None, 196)[0] == 2  # 4 GiB budget at 2 GiB per worker
    assert worker_plan(None, 1)[0] == 1
    monkeypatch.setenv(WORKERS_ENV, "3")
    assert worker_plan(None, 196)[0] == 2
    monkeypatch.setattr(
        module, "detect_available_memory", lambda: SimpleNamespace(available_bytes=64 * gib)
    )
    assert worker_plan(None, 196)[0] == 3
    assert worker_plan(1, 196)[0] == 1
    with pytest.raises(ValueError):
        worker_plan(0, 196)


def test_fold_masks_follow_the_schedule() -> None:
    folds = quarterly_folds(_sessions(), SCHEDULE)
    assert [fold.name for fold in folds] == ["2024Q1", "2024Q2"]
    dates = pd.DatetimeIndex(["2023-06-01", "2023-11-15", "2023-12-28", "2024-02-01", "2024-05-02"])
    train, valid, predict = fold_masks(dates, folds[0])
    assert train.tolist() == [True, False, False, False, False]
    assert valid.tolist() == [False, True, False, False, False]
    assert predict.tolist() == [False, False, False, True, False]
    _, _, predict_q2 = fold_masks(dates, folds[1])
    assert predict_q2.tolist() == [False, False, False, False, True]


def _run(tmp_path: Path, monkeypatch, backend: str, workers: int) -> dict:
    monkeypatch.setattr(module, "backend_name", lambda: backend)
    monkeypatch.setenv(WORKERS_ENV, str(workers))
    table = _table()
    folds = quarterly_folds(_sessions(), SCHEDULE)
    output = tmp_path / backend
    results = run_walk_forward(
        table,
        feature_columns=[f"f{i}" for i in range(6)],
        horizons=HORIZONS,
        folds=folds,
        schedule=SCHEDULE,
        output_root=output,
        seeds=(1, 2),
        workers=workers,
        max_rounds=40,
        early_stopping=5,
        min_group_size=10,
    )
    predictions = pd.read_parquet(output / "predictions.parquet")
    targets = pd.read_parquet(output / "targets.parquet")
    predicted_rows = sum(results["fold_rows"][str(f.index)]["predict_rows"] for f in folds)
    assert len(targets) == predicted_rows
    assert len(predictions) == predicted_rows * 3  # two seeds and the ensemble
    assert set(predictions["model"]) == {MODEL_NAME, f"{MODEL_NAME}-seed1", f"{MODEL_NAME}-seed2"}
    assert results["backend"] == backend
    assert results["fits"]["count"] == len(folds) * len(HORIZONS) * 2
    assert not list((output / "work").glob("*/*.npy"))
    stored = json.loads((output / "results.json").read_text())
    assert stored["cells"] == results["cells"]
    ensemble_short = next(
        c
        for c in results["cells"]
        if c["model"] == MODEL_NAME and c["cell"] == "short" and c["period"] == "P1+P2"
    )
    assert ensemble_short["rank_ic"] > 0.2
    assert ensemble_short["days"] == targets["signal_date"].nunique()
    return results


def test_walk_forward_end_to_end_with_sklearn_inline(tmp_path: Path, monkeypatch) -> None:
    _run(tmp_path, monkeypatch, "sklearn_hist_gradient_boosting", workers=1)


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_walk_forward_end_to_end_with_lightgbm(tmp_path: Path, monkeypatch) -> None:
    results = _run(tmp_path, monkeypatch, "lightgbm", workers=1)
    fits = pd.read_parquet(tmp_path / "lightgbm" / "fits.parquet")
    assert (fits["best_iteration"] >= 1).all()
    assert results["worker_plan"]["count"] == 1


def test_partial_folds_are_written_and_resumed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "backend_name", lambda: "sklearn_hist_gradient_boosting")
    monkeypatch.setenv(WORKERS_ENV, "1")
    table = _table(symbols_per_market=20)
    folds = quarterly_folds(_sessions(), SCHEDULE)
    shared = dict(
        feature_columns=[f"f{i}" for i in range(6)],
        horizons=HORIZONS,
        folds=folds,
        schedule=SCHEDULE,
        seeds=(1,),
        workers=1,
        early_stopping=3,
        min_group_size=10,
    )
    first = tmp_path / "first"
    run_walk_forward(table, output_root=first, max_rounds=20, **shared)
    assert sorted(p.name for p in (first / "partial").glob("*.npz")) == [
        "fold-00.npz",
        "fold-01.npz",
    ]
    second = tmp_path / "second"
    # Fewer rounds would change a fresh fit; identical predictions prove the folds were reused.
    results = run_walk_forward(table, output_root=second, max_rounds=2, resume_root=first, **shared)
    pd.testing.assert_frame_equal(
        pd.read_parquet(second / "predictions.parquet"),
        pd.read_parquet(first / "predictions.parquet"),
    )
    assert results["fits"]["count"] == len(folds) * len(HORIZONS)
    assert all(info.get("resumed") for info in results["fold_rows"].values())
    assert sorted(p.name for p in (second / "partial").glob("*.npz")) == [
        "fold-00.npz",
        "fold-01.npz",
    ]


def test_resume_ignores_folds_of_another_store_build(tmp_path: Path, monkeypatch) -> None:
    """Folds trained on another build of the stores (or saved before the store tag existed)
    are refitted: the same column names do not make a rebuilt store the same store."""
    monkeypatch.setattr(module, "backend_name", lambda: "sklearn_hist_gradient_boosting")
    monkeypatch.setenv(WORKERS_ENV, "1")
    table = _table(symbols_per_market=20)
    folds = quarterly_folds(_sessions(), SCHEDULE)
    shared = dict(
        feature_columns=[f"f{i}" for i in range(6)],
        horizons=HORIZONS,
        folds=folds,
        schedule=SCHEDULE,
        seeds=(1,),
        workers=1,
        early_stopping=3,
        min_group_size=10,
    )
    first = tmp_path / "first"
    run_walk_forward(table, output_root=first, max_rounds=20, stores="store-a", **shared)
    rebuilt = tmp_path / "rebuilt"
    results = run_walk_forward(
        table, output_root=rebuilt, max_rounds=2, resume_root=first, stores="store-b", **shared
    )
    assert not any(info.get("resumed") for info in results["fold_rows"].values())
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            pd.read_parquet(rebuilt / "predictions.parquet"),
            pd.read_parquet(first / "predictions.parquet"),
        )
    untagged = tmp_path / "untagged"
    run_walk_forward(table, output_root=untagged, max_rounds=20, **shared)
    results = run_walk_forward(
        table, output_root=tmp_path / "tagged", max_rounds=2, resume_root=untagged,
        stores="store-a", **shared,
    )
    assert not any(info.get("resumed") for info in results["fold_rows"].values())
    same = run_walk_forward(
        table, output_root=tmp_path / "same", max_rounds=2, resume_root=first, stores="store-a",
        **shared,
    )
    assert all(info.get("resumed") for info in same["fold_rows"].values())


def test_store_tag_follows_the_store_identities(tmp_path: Path) -> None:
    def stores(feature_digest: str) -> list[module.SplitStores]:
        return [
            module.SplitStores(
                split=split,
                source_root=str(tmp_path / split),
                samples=10,
                feature_root=tmp_path / "features" / feature_digest,
                feature_identity={},
                label_root=tmp_path / "labels" / "label-digest",
                label_identity={},
            )
            for split in ("train", "validation", "test")
        ]

    tag = module.store_tag(stores("build-a"))
    assert tag == module.store_tag(list(reversed(stores("build-a"))))
    assert tag != module.store_tag(stores("build-b"))
