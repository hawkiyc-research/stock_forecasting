"""Phase 1 information test: can free Taiwan data predict up versus down beyond chance?

GBDT models per feature set and horizon are fitted on the Taiwan train split (the
last train fraction, after an embargo, only selects the boosting iteration) and
evaluated on validation; test is recorded only. Decisions follow the criteria
pre-registered in the JSON config.
"""

# Bilingual report text intentionally preserves Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.research import tw_evaluation as evaluation
from stock_forecasting.research import tw_features as features

SCHEMA_VERSION = "1.0"
REQUIRED_KEYS = (
    "experiment_config", "markets", "horizons", "comparison_horizons", "fixed_band",
    "cost_floor", "neutral_share_target", "us_symbols", "feature_sets", "fixed_band_sets",
    "model", "bootstrap", "criteria", "backtest",
)
FLOW_SOURCES = ("twse_t86", "tpex_institutional", "twse_margin", "tpex_margin")
USABLE = {"complete", "partial"}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Information-test config is incomplete or unsupported: {missing}")
    if "F0" not in config["feature_sets"]:
        raise ValueError("The F0 technical baseline set is required for every comparison")
    return config


def _read_table(root: Path, manifest: dict[str, Any], name: str) -> pd.DataFrame | None:
    source = manifest["sources"].get(name, {})
    table = source.get("table")
    if source.get("status") not in USABLE or not table:
        return None
    return pd.read_parquet(root / table["path"])


def flows_by_symbol(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], list]:
    """Merge institutional and margin rows into one date-indexed frame per symbol."""
    tables = {name: _read_table(root, manifest, name) for name in FLOW_SOURCES}
    used = [name for name, table in tables.items() if table is not None]
    institutional = [
        tables[name] for name in ("twse_t86", "tpex_institutional") if tables[name] is not None
    ]
    margin = [tables[name] for name in ("twse_margin", "tpex_margin") if tables[name] is not None]
    frames = []
    for parts in (institutional, margin):
        if parts:
            frame = pd.concat(parts, ignore_index=True).drop(columns=["market"])
            frame["date"] = pd.to_datetime(frame.pop("timestamp")).dt.normalize()
            frames.append(frame.drop_duplicates(["symbol", "date"], keep="last"))
    if not frames:
        return {}, used
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["symbol", "date"], how="outer")
    return {symbol: rows.drop(columns="symbol") for symbol, rows in merged.groupby("symbol")}, used


def us_series(config: dict[str, Any], root: Path, manifest: dict[str, Any]) -> dict:
    """Adjusted US closes from the shared evaluation bar store plus EODHD series."""
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.sample_universe import open_clean_dataset

    series: dict[str, pd.DataFrame] = {}
    experiment = ExperimentConfig.from_yaml(config["experiment_config"])
    dataset = open_clean_dataset(experiment, "validation")
    try:
        for symbol in config["us_symbols"]:
            if symbol in dataset._index:
                bars = features.adjusted_bars(dataset._load_symbol(symbol))
                series[symbol] = bars[["date", "close"]]
    finally:
        dataset.close()
    extra = _read_table(root, manifest, "eodhd_series")
    if extra is not None:
        for symbol, rows in extra.groupby("symbol"):
            series[str(symbol)] = pd.DataFrame({
                "date": pd.to_datetime(rows["timestamp"]).dt.normalize(),
                "close": rows["adjusted_close"].astype(float),
            })
    return series


def assemble(
    panel: pd.DataFrame,
    root: Path,
    manifest: dict[str, Any],
    us: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Attach market, TAIFEX and US features; return the column list of each block."""
    market = features.market_features(panel)
    panel = panel.merge(market, on=["date", "market"], how="left")
    taifex = features.taifex_features(
        _read_table(root, manifest, "taifex_futures"),
        _read_table(root, manifest, "taifex_put_call"),
    )
    taifex_columns = [column for column in taifex.columns if column != "date"]
    if taifex_columns:
        panel = panel.merge(taifex, on="date", how="left")
    daily = features.us_daily(us)
    evening = features.align_us(panel["date"], daily, variant="evening")
    pre_open = features.align_us(panel["date"], daily, variant="pre_open")
    panel = pd.concat([panel.reset_index(drop=True), evening, pre_open], axis=1)
    short = [name for name in features.TECHNICAL if name not in features.SHORT_EXCLUDED]
    blocks = {
        "technical": [*features.TECHNICAL, *(f"bench_{name}" for name in features.TECHNICAL)],
        "technical_short": [*short, *(f"bench_{name}" for name in short)],
        "long_lookback": list(features.LONG),
        "flows": list(features.FLOW),
        "market": [*features.MARKET, *taifex_columns],
        "us_evening": list(evening.columns),
        "us_pre_open": list(pre_open.columns),
    }
    return panel, blocks


def informative_columns(x: np.ndarray) -> np.ndarray:
    """Columns with at least two distinct finite values (sklearn cannot bin empty ones)."""
    keep = np.zeros(x.shape[1], dtype=bool)
    for column in range(x.shape[1]):
        values = x[:, column]
        values = values[np.isfinite(values)]
        keep[column] = values.size > 0 and values.min() < values.max()
    return keep


def _staged_fit(estimator, x_core, y_core, x_holdout, y_holdout, score, step):
    """Fit on the core rows and pick the boosting stage with the best holdout score.

    Columns without variation in the core rows (for example features of a failed
    auxiliary source) are left out; ``research_columns_`` records the kept mask.
    """
    estimator.research_columns_ = informative_columns(x_core)
    if not estimator.research_columns_.any():
        raise ValueError("No feature varies in the training rows")
    estimator.fit(x_core[:, estimator.research_columns_], y_core)
    x_holdout = x_holdout[:, estimator.research_columns_]
    staged = (
        estimator.staged_predict_proba(x_holdout)
        if hasattr(estimator, "predict_proba")
        else estimator.staged_predict(x_holdout)
    )
    best_stage, best_value = 1, np.inf
    for stage, prediction in enumerate(staged, start=1):
        if stage % step and stage != estimator.n_iter_:
            continue
        value = score(y_holdout, prediction)
        if value < best_value:
            best_stage, best_value = stage, value
    return best_stage, float(best_value)


def _at_stage(estimator, x: np.ndarray, stage: int) -> np.ndarray:
    x = x[:, estimator.research_columns_]
    staged = (
        estimator.staged_predict_proba(x)
        if hasattr(estimator, "predict_proba")
        else estimator.staged_predict(x)
    )
    for index, prediction in enumerate(staged, start=1):
        if index == stage:
            return prediction
    raise ValueError("Requested boosting stage exceeds the fitted model")


def _left_out(columns: list[str], estimator) -> list[str]:
    return [name for name, kept in zip(columns, estimator.research_columns_, strict=True)
            if not kept]


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    rows = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probabilities[rows, labels], 1e-12, 1.0))))


def _train_partition(dates: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(dates)
    holdout_start = int(len(unique) * (1 - model["holdout_fraction"]))
    core_end = max(1, holdout_start - model["holdout_embargo_sessions"])
    core = dates < unique[core_end]
    holdout = dates >= unique[holdout_start]
    return core, holdout


def _three_class(estimator, probabilities: np.ndarray) -> np.ndarray:
    full = np.zeros((len(probabilities), 3))
    full[:, estimator.classes_.astype(int)] = probabilities
    return full


def fit_classifier(x_train, labels, dates, model) -> tuple[Any, int, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    core, holdout = _train_partition(dates, model)
    estimator = HistGradientBoostingClassifier(
        learning_rate=model["learning_rate"],
        max_iter=model["max_iter"],
        max_leaf_nodes=model["max_leaf_nodes"],
        min_samples_leaf=model["min_samples_leaf"],
        l2_regularization=model["l2_regularization"],
        early_stopping=False,
        random_state=model["seed"],
    )

    def score(y, probabilities):
        return _log_loss(y, _three_class(estimator, probabilities))

    stage, value = _staged_fit(
        estimator, x_train[core], labels[core], x_train[holdout], labels[holdout], score,
        model["stage_step"],
    )
    return estimator, stage, value


def fit_regressor(x_train, target, dates, model) -> tuple[Any, int, float]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    core, holdout = _train_partition(dates, model)
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=model["learning_rate"],
        max_iter=model["max_iter"],
        max_leaf_nodes=model["max_leaf_nodes"],
        min_samples_leaf=model["min_samples_leaf"],
        l2_regularization=model["l2_regularization"],
        early_stopping=False,
        random_state=model["seed"],
    )

    def score(y, prediction):
        return float(np.mean((y - prediction) ** 2))

    stage, value = _staged_fit(
        estimator, x_train[core], target[core], x_train[holdout], target[holdout], score,
        model["stage_step"],
    )
    return estimator, stage, value


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: pd.DataFrame,
    priors: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Pooled and per-market metrics plus per-date sums used by the bootstrap."""
    prior = np.vstack([priors[market] for market in rows["market"]])
    keys = rows[["date", "market"]].reset_index(drop=True)
    date_only = evaluation.date_only_probabilities(probabilities, keys)
    model, base, timing = (
        evaluation.ce_parts(probabilities, labels),
        evaluation.ce_parts(prior, labels),
        evaluation.ce_parts(date_only, labels),
    )
    dates = rows["date"].to_numpy()
    result: dict[str, Any] = {}
    for scope in ("TW", *sorted(rows["market"].unique())):
        mask = np.ones(len(rows), bool) if scope == "TW" else (rows["market"] == scope).to_numpy()
        mask &= labels != evaluation.INVALID

        def mean(values, selection=mask):
            selected = values[selection]
            return float(np.nanmean(selected)) if np.isfinite(selected).any() else float("nan")

        result[scope] = {
            "samples": int(mask.sum()),
            "ce": mean(model["ce"]),
            "prior_ce": mean(base["ce"]),
            "move": mean(model["move"]),
            "prior_move": mean(base["move"]),
            "direction": mean(model["direction"]),
            "prior_direction": mean(base["direction"]),
            "date_only_ce": mean(timing["ce"]),
            "timing_skill": mean(base["ce"]) - mean(timing["ce"]),
            "balanced_accuracy": evaluation.balanced_accuracy(
                labels[mask], probabilities[mask].argmax(axis=1)
            ),
            "class_shares": np.bincount(labels[mask], minlength=3).tolist(),
        }
    sums = {
        "direction": evaluation.per_date_sums(model["direction"], dates),
        "prior_direction": evaluation.per_date_sums(base["direction"], dates),
        "ce": evaluation.per_date_sums(model["ce"], dates),
        "prior_ce": evaluation.per_date_sums(base["ce"], dates),
    }
    return result, sums


def _summary_markdown(report: dict[str, Any]) -> str:
    criteria = report["criteria"]
    lines = [
        "# 台股資訊測試摘要 / Taiwan Information Test Summary",
        "",
        "## 中文",
        "",
        f"- 是否找到超越擲硬幣的方向資訊：**{'是' if criteria['information_found'] else '否'}**",
        "- 判準：至少兩個決策天期中，「漲還是跌」CE 比 train 先驗改善至少 1%，"
        "且相對先驗與 F0 的日期區塊 bootstrap 95% 區間都高於 0。",
        "",
        "| 特徵組 | 通過天期數 | 通過 |",
        "| --- | ---: | --- |",
    ]
    rows = [
        (group, value["passed_horizons"], value["passed"])
        for group, value in criteria.items()
        if group != "information_found"
    ]
    lines += [f"| {group} | {count} | {'是' if passed else '否'} |" for group, count, passed in rows]
    lines += [
        "",
        "## English",
        "",
        "- Direction information beyond a coin flip found: "
        f"**{'yes' if criteria['information_found'] else 'no'}**",
        "- Rule: on at least two decision horizons, up-versus-down CE beats the train prior "
        "by at least 1%, and the date-block bootstrap 95% intervals against the prior and F0 "
        "are both above zero.",
        "",
        "| Feature set | Passing horizons | Passed |",
        "| --- | ---: | --- |",
    ]
    lines += [
        f"| {group} | {count} | {'yes' if passed else 'no'} |" for group, count, passed in rows
    ]
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(
    config_path: Path,
    auxiliary_root: Path,
    output_root: Path,
    *,
    workers: int,
    panel: pd.DataFrame | None = None,
    us: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build (or accept) the panel, fit every set and horizon, and write the report."""
    config = load_config(config_path)
    manifest = json.loads((auxiliary_root / "auxiliary-manifest.json").read_text())
    horizons_all = tuple(sorted({*config["horizons"], *config["comparison_horizons"]}))
    flow_tables, flow_sources = flows_by_symbol(auxiliary_root, manifest)
    if panel is None:
        panel = features.build_panel(
            config["experiment_config"],
            markets=tuple(config["markets"]),
            horizons=horizons_all,
            window=int(config.get("window", 128)),
            flows_by_symbol=flow_tables,
            workers=workers,
        )
    if us is None:
        us = us_series(config, auxiliary_root, manifest)
    panel, blocks = assemble(panel, auxiliary_root, manifest, us)
    available = {
        "flows": bool(flow_sources),
        "market": True,
        "us_evening": bool(blocks["us_evening"]),
        "us_pre_open": bool(blocks["us_pre_open"]),
    }
    groups = {
        name: [column for block in parts for column in blocks[block]]
        for name, parts in config["feature_sets"].items()
        if all(available.get(block, True) for block in parts)
    }
    floors = np.where(
        panel["asset_type"].to_numpy() == "etf",
        config["cost_floor"]["etf"],
        config["cost_floor"]["stock"],
    )
    split = panel["split"].to_numpy()
    train, sigma = split == "train", panel["sigma_20d"].to_numpy(float)
    bands, labels_hybrid, labels_fixed = {}, {}, {}
    for horizon in horizons_all:
        returns = panel[f"return_{horizon}"].to_numpy(float)
        calibration = evaluation.calibrate_k(
            returns[train], sigma[train], floors[train], horizon, config["neutral_share_target"]
        )
        bands[horizon] = calibration
        labels_hybrid[horizon] = evaluation.band_labels(
            returns, evaluation.hybrid_bands(sigma, floors, calibration["k"], horizon)
        )
        labels_fixed[horizon] = evaluation.band_labels(returns, config["fixed_band"])
    markets = panel["market"].to_numpy()
    model_settings = config["model"]
    results: dict[str, Any] = {"validation": {}, "test": {}}
    direction_sums: dict[tuple[str, int], pd.DataFrame] = {}
    prior_sums: dict[int, pd.DataFrame] = {}
    backtests: dict[str, Any] = {}
    alpha_scales: dict[int, float] = {}
    fits = []
    for group, columns in groups.items():
        x_all = panel[columns].to_numpy(np.float32)
        # Ratios with a zero denominator (and float32 overflow) are missing, not infinite.
        x_all[~np.isfinite(x_all)] = np.nan
        variants = [("hybrid", config["horizons"])]
        if group in config["fixed_band_sets"]:
            variants += [("fixed", horizons_all), ("hybrid", config["comparison_horizons"])]
        for band, horizons in variants:
            for horizon in horizons:
                labels = (labels_hybrid if band == "hybrid" else labels_fixed)[horizon]
                usable = train & (labels != evaluation.INVALID)
                estimator, stage, holdout_loss = fit_classifier(
                    x_all[usable], labels[usable], panel["date"].to_numpy()[usable], model_settings
                )
                priors = evaluation.prior_probabilities(labels[usable], markets[usable])
                fits.append({"set": group, "band": band, "horizon": horizon, "stage": stage,
                             "holdout_log_loss": holdout_loss,
                             "left_out_features": _left_out(columns, estimator)})
                for split_name in ("validation", "test"):
                    rows = (split == split_name) & (labels != evaluation.INVALID)
                    probabilities = _three_class(
                        estimator, _at_stage(estimator, x_all[rows], stage)
                    )
                    metrics, sums = classification_metrics(
                        probabilities, labels[rows], panel.loc[rows], priors
                    )
                    key = f"{band}_classification"
                    results[split_name].setdefault(group, {}).setdefault(str(horizon), {})[
                        key
                    ] = metrics
                    if split_name == "validation" and band == "hybrid":
                        direction_sums[(group, horizon)] = sums["direction"]
                        prior_sums[horizon] = sums["prior_direction"]
                    if band == "hybrid" and group in {"F0", "all_evening", "all_pre_open"}:
                        backtests.setdefault(split_name, {}).setdefault(group, {})[
                            str(horizon)
                        ] = evaluation.backtest(
                            probabilities[:, evaluation.UP] - probabilities[:, evaluation.DOWN],
                            panel.loc[rows, f"return_{horizon}"].to_numpy(float),
                            floors[rows],
                            panel.loc[rows, "date"].to_numpy(),
                            top_fraction=config["backtest"]["top_fraction"],
                        )
        for horizon in config["horizons"]:
            alpha = panel[f"alpha_{horizon}"].to_numpy(float)
            usable = train & np.isfinite(alpha)
            scale = alpha_scales.setdefault(horizon, evaluation.robust_scale(alpha[usable]))
            estimator, stage, holdout_loss = fit_regressor(
                x_all[usable], alpha[usable] / scale, panel["date"].to_numpy()[usable],
                model_settings,
            )
            fits.append({"set": group, "band": "alpha", "horizon": horizon, "stage": stage,
                         "holdout_mse": holdout_loss,
                         "left_out_features": _left_out(columns, estimator)})
            for split_name in ("validation", "test"):
                rows = (split == split_name) & np.isfinite(alpha)
                prediction = _at_stage(estimator, x_all[rows], stage) * scale
                target = alpha[rows]
                results[split_name].setdefault(group, {}).setdefault(str(horizon), {})["alpha"] = {
                    "normalized_smooth_l1": float(np.mean(evaluation.smooth_l1(
                        (prediction - target) / scale
                    ))),
                    "zero_normalized_smooth_l1": float(np.mean(evaluation.smooth_l1(
                        target / scale
                    ))),
                    "rank_ic": evaluation.rank_ic(
                        prediction, target, panel.loc[rows, "date"].to_numpy(), markets[rows]
                    ),
                }
    criteria = evaluation.evaluate_criteria(
        direction_sums,
        prior_sums,
        groups=list(groups),
        criteria=config["criteria"],
        bootstrap=config["bootstrap"],
    )
    window = {}
    for group in ("F0_long", "short_only"):
        for horizon in config["criteria"]["decision_horizons"]:
            if (group, horizon) in direction_sums and ("F0", horizon) in direction_sums:
                base, other = direction_sums[("F0", horizon)], direction_sums[(group, horizon)]
                window[f"{group}_{horizon}"] = evaluation.block_bootstrap(
                    base["sum"].to_numpy() - other["sum"].to_numpy(),
                    other["count"].to_numpy(),
                    **config["bootstrap"],
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tw-information-test",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": _sha256(config_path),
        "auxiliary_manifest_sha256": _sha256(auxiliary_root / "auxiliary-manifest.json"),
        "auxiliary_sources": {
            name: source["status"] for name, source in manifest["sources"].items()
        },
        "flow_sources_used": flow_sources,
        "feature_sets": {name: len(columns) for name, columns in groups.items()},
        "skipped_feature_sets": sorted(set(config["feature_sets"]) - set(groups)),
        "samples": {name: int((split == name).sum()) for name in ("train", "validation", "test")},
        "bands": {str(horizon): value for horizon, value in bands.items()},
        "alpha_scales": {str(horizon): value for horizon, value in alpha_scales.items()},
        "fits": fits,
        "results": results,
        "criteria": criteria,
        "window_question_direction_gap_vs_F0": window,
        "backtest": backtests,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (output_root / "summary.md").write_text(_summary_markdown(report))
    return report
