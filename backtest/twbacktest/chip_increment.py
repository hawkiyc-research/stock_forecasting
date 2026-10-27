"""Protocol 13.18 items 4-5: does the chip feature library add ranking skill? (H-H and gate G1)

Three B0 LightGBM walk-forwards share one schedule, one label variant and one set of signal
dates and differ only in their feature spec: F0 = ``alpha_tw_v1`` (price / volume), F1 =
``chip_tw_v1`` (chip and event data only), F2 = both. This module scores the *paired* differences
F1 - F0 and F2 - F0 date by date and applies the two pre-registered decisions:

* **H-H** (chip data adds skill): (i) the F2 - F0 paired Rank IC difference against the residual
  target in the holding cell (h = 5..10, markets weighted equally, every ranked name per market,
  the protocol 13.13 cell) on P1 + P2 has a block-bootstrap 95% lower bound above 0, and (ii) F2's
  top-minus-bottom quartile residual spread per 10-session cohort inside the protocol 13.14
  liquidity universe on P1 + P2 is at least 1.5 x F0's. Both must hold.
* **G1** (first gate for buying broker-branch data): inside (a) the liquidity universe or (b) the
  small / mid caps (liquidity universe minus the protocol 13.15 liquid stock-futures universe),
  the same paired holding-cell residual Rank IC difference has a lower bound above 0 and F2's
  quartile residual spread (h = 10) is not below F0's. Either universe suffices.

The G1 sub-analysis (descriptive) repeats every metric by period and by segment: *winners* =
the top fifth of the universe group by the trailing 20-session return on the signal date, *rest*
= everyone else, and adds the bottom-quartile excess return (mean raw forward return of the
names the model ranks in the bottom quartile minus the universe group's mean) -- whether chip
features stop the model from shorting the winners that keep rising.

Metrics per (model, universe, segment, signal date, ranking group): Spearman Rank IC of the
model's h-score against the residual target and against the raw alpha target of the same run
(``targets.parquet``, h = 1 and 5..10), top-minus-bottom quartile spreads of both targets
(h = 1, 5, 10), the bottom-quartile excess, and (segment ``all``, h = 10) the industry-neutral,
dollar-neutral quartile long-short gross return of protocol 13.17 step 0. Ranking groups are
the markets (TWSE, TPEX) for ``all``, ``liquidity`` and ``small_mid`` and both markets together
for ``futures_liquid``; a date's value averages its groups with equal weight, as the protocol
13.13 cells do. Intervals: circular block bootstrap of the per-date series (1,000 replicates,
seed 11; blocks 5 for h = 1, 20 for the holding cell and h = 10 spreads, 2h otherwise); paired
differences are taken on common dates first and bootstrapped as one series.

Every per-group statistic is a vectorized pandas group operation over a few million rows; only
the industry-neutral long-short loops over a few thousand (date, group) pairs. The whole run is
a few minutes on one core, below the cost of a process pool, so it runs serially on purpose.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats
from twbacktest.exposure_diagnostics import (
    DEFAULT_ARTIFACTS,
    DEFAULT_REFERENCE,
    DEFAULT_TAIFEX,
    LIQUIDITY,
    MIN_CLOSE,
    load_reference,
    quantile_long_short,
)
from twbacktest.futures_universe import PERIODS, eligibility_frame, segment_masks
from twbacktest.walkforward_data import build_inputs, load_predictions, load_prices

DECISION_PERIOD = "P1+P2"
IC_HORIZONS = (1, 5, 6, 7, 8, 9, 10)
SPREAD_HORIZONS = (1, 5, 10)
CELLS = {"short": (1,), "holding": (5, 6, 7, 8, 9, 10)}
CELL_BLOCKS = {"short": 5, "holding": 20}
UNIVERSES = ("all", "liquidity", "small_mid", "futures_liquid")
SEGMENTS = ("all", "winners", "rest")
QUANTILE = 0.25
WINNER_SHARE = 0.2
LOOKBACK_SESSIONS = 20
MIN_NAMES = 20
BOOTSTRAP = {"replicates": 1000, "seed": 11}
LS_HORIZON = 10
SPREAD_MULTIPLE = 1.5
H_H_SPREAD_UNIVERSE = "liquidity"
G1_UNIVERSES = ("liquidity", "small_mid")
TARGET_KINDS = {"residual": "residual_h{h:02d}", "raw": "alpha_h{h:02d}"}


def spread_block(h: int) -> int:
    return max(5, 2 * h)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def load_targets(path: Path, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = ["symbol", "signal_date"]
    for h in horizons:
        columns += [TARGET_KINDS["residual"].format(h=h), TARGET_KINDS["raw"].format(h=h)]
    frame = pd.read_parquet(path, columns=columns)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.drop_duplicates(["symbol", "signal_date"], keep="first")


def trailing_returns(prices: pd.DataFrame, sessions: int) -> pd.DataFrame:
    """(symbol, date, prior) with prior = adjusted close over the close ``sessions`` earlier."""
    stocks = prices[prices["asset_type"] != "index"].sort_values(["symbol", "date"], kind="stable")
    prior = stocks.groupby("symbol", sort=False)["adj_close"].pct_change(sessions)
    return pd.DataFrame(
        {
            "symbol": stocks["symbol"].to_numpy(),
            "signal_date": stocks["date"].to_numpy(),
            "prior": prior.to_numpy(dtype=np.float64),
        }
    )


def universe_masks(
    frame: pd.DataFrame, inputs, eligibility: pd.DataFrame
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Row mask and ranking-group key per universe."""
    market = frame["market"].to_numpy().astype(object)
    is_etf = inputs.index.is_etf
    dollar_volume = inputs.dollar_volume
    threshold = np.array([LIQUIDITY[m] for m in frame["market"].to_numpy()])
    liquidity = (
        ~is_etf
        & np.isfinite(dollar_volume)
        & (dollar_volume >= threshold)
        & (inputs.raw_close >= MIN_CLOSE)
    )
    futures = segment_masks(frame, eligibility)["futures_liquid"] & ~is_etf
    tw = np.full(len(frame), "TW", dtype=object)
    return {
        "all": (np.ones(len(frame), dtype=bool), market),
        "liquidity": (liquidity, market),
        "small_mid": (liquidity & ~futures, market),
        "futures_liquid": (futures, tw),
    }


# --------------------------------------------------------------------------------------
# Vectorized per-group statistics
# --------------------------------------------------------------------------------------

KEYS = ["signal_date", "group"]


def grouped_spearman(frame: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Spearman correlation of ``a`` and ``b`` inside every (date, group); NaN below MIN_NAMES."""
    sub = frame[[*KEYS, a, b]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    grouped = sub.groupby(KEYS, sort=True)
    ranks = pd.DataFrame({"ra": grouped[a].rank(), "rb": grouped[b].rank()})
    ranks["ab"] = ranks["ra"] * ranks["rb"]
    ranks["aa"] = ranks["ra"] ** 2
    ranks["bb"] = ranks["rb"] ** 2
    ranks[KEYS] = sub[KEYS]
    moments = ranks.groupby(KEYS, sort=True).agg(
        ra=("ra", "mean"),
        rb=("rb", "mean"),
        ab=("ab", "mean"),
        aa=("aa", "mean"),
        bb=("bb", "mean"),
        n=("ra", "size"),
    )
    cov = moments["ab"] - moments["ra"] * moments["rb"]
    var_a = moments["aa"] - moments["ra"] ** 2
    var_b = moments["bb"] - moments["rb"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = cov / np.sqrt(var_a * var_b)
    ic[(moments["n"] < MIN_NAMES) | ~np.isfinite(ic)] = np.nan
    return ic


def quartile_statistics(
    frame: pd.DataFrame, score: str, columns: dict[str, str], universe_mean: pd.Series
) -> pd.DataFrame:
    """Per (date, group): top-minus-bottom quartile mean of every target column and the
    bottom quartile's excess raw return over the universe group's mean raw return."""
    sub = frame[[*KEYS, score, *columns.values()]].dropna(subset=[score])
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby(KEYS, sort=True)
    n = grouped[score].transform("size")
    k = np.ceil(n * QUANTILE).clip(lower=1)
    desc = grouped[score].rank(method="first", ascending=False)
    asc = grouped[score].rank(method="first", ascending=True)
    top = sub[desc <= k]
    bottom = sub[asc <= k]
    out = pd.DataFrame({"names": grouped.size()})
    for name, column in columns.items():
        top_mean = top.groupby(KEYS, sort=True)[column].mean()
        bottom_mean = bottom.groupby(KEYS, sort=True)[column].mean()
        out[f"spread_{name}"] = (top_mean - bottom_mean).reindex(out.index)
        if name == "raw":
            out["bottom_excess"] = (bottom_mean - universe_mean.reindex(bottom_mean.index)).reindex(
                out.index
            )
    out.loc[out["names"] < MIN_NAMES, out.columns.difference(["names"])] = np.nan
    return out


def industry_long_short(frame: pd.DataFrame, score: str, ret: str) -> pd.Series:
    """Industry-neutral quartile long-short spread per (date, group) (protocol 13.17 step 0)."""
    sub = frame[[*KEYS, score, ret, "industry"]].dropna(subset=[score, ret])
    values = {}
    if sub.empty:
        return pd.Series(dtype=float)
    sub = sub.sort_values(KEYS, kind="stable")
    dates = sub["signal_date"].to_numpy()
    groups = sub["group"].to_numpy()
    change = np.r_[True, (dates[1:] != dates[:-1]) | (groups[1:] != groups[:-1]), True]
    s = sub[score].to_numpy(dtype=np.float64)
    r = sub[ret].to_numpy(dtype=np.float64)
    industry = sub["industry"].to_numpy()
    for start, stop in pairwise(np.flatnonzero(change)):
        if stop - start < MIN_NAMES:
            continue
        spread, _used, _weights = quantile_long_short(
            s[start:stop], r[start:stop], industry[start:stop]
        )
        values[(pd.Timestamp(dates[start]), str(groups[start]))] = spread
    series = pd.Series(values, dtype=float)
    series.index = pd.MultiIndex.from_tuples(series.index, names=KEYS)
    return series


def winner_flags(frame: pd.DataFrame) -> np.ndarray:
    """Top WINNER_SHARE of every (date, group) by trailing return; NaN trailing = not a winner."""
    prior = frame["prior"]
    finite = prior.notna()
    sub = frame[finite]
    grouped = sub.groupby(KEYS, sort=False)["prior"]
    k = np.ceil(grouped.transform("size") * WINNER_SHARE).clip(lower=1)
    rank = grouped.rank(method="first", ascending=False)
    flags = np.zeros(len(frame), dtype=bool)
    flags[np.flatnonzero(finite.to_numpy())] = (rank <= k).to_numpy()
    return flags


def model_daily(frame: pd.DataFrame, label: str, universe: str) -> pd.DataFrame:
    """Long table of per-date metrics (groups averaged) for one model and universe."""
    universe_mean = {
        h: frame.groupby(KEYS, sort=True)[TARGET_KINDS["raw"].format(h=h)].mean()
        for h in SPREAD_HORIZONS
    }
    winners = winner_flags(frame)
    records = []
    for segment in SEGMENTS:
        if segment == "all":
            part = frame
        elif segment == "winners":
            part = frame[winners]
        else:
            part = frame[~winners]
        if part.empty:
            continue
        tables = []
        for h in IC_HORIZONS:
            for kind, pattern in TARGET_KINDS.items():
                ic = grouped_spearman(part, f"h{h:02d}", pattern.format(h=h))
                if not ic.empty:
                    tables.append(ic.rename(f"ic_{kind}").to_frame().assign(h=h))
        for h in SPREAD_HORIZONS:
            columns = {kind: pattern.format(h=h) for kind, pattern in TARGET_KINDS.items()}
            quartiles = quartile_statistics(part, f"h{h:02d}", columns, universe_mean[h])
            if not quartiles.empty:
                tables.append(quartiles.assign(h=h))
        if segment == "all":
            ls = industry_long_short(
                part, f"h{LS_HORIZON:02d}", TARGET_KINDS["raw"].format(h=LS_HORIZON)
            )
            if not ls.empty:
                tables.append(ls.rename("ls_industry").to_frame().assign(h=LS_HORIZON))
        if not tables:
            continue
        wide = pd.concat(tables).reset_index()
        wide = wide.groupby([*KEYS, "h"], sort=True).first().reset_index()
        # Markets weigh equally on each date: average the groups, then melt to one row per metric.
        metrics = [c for c in wide.columns if c not in (*KEYS, "h")]
        per_date = wide.groupby(["signal_date", "h"], sort=True)[metrics].mean().reset_index()
        long = per_date.melt(id_vars=["signal_date", "h"], var_name="metric", value_name="value")
        long = long.dropna(subset=["value"])
        long["model"], long["universe"], long["segment"] = label, universe, segment
        records.append(long)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def cell_series(daily: pd.DataFrame, metric: str, horizons: tuple[int, ...]) -> pd.Series:
    """Per-date cell value: horizons averaged (groups were already averaged)."""
    rows = daily[(daily["metric"] == metric) & daily["h"].isin(horizons)]
    return rows.groupby("signal_date")["value"].mean().sort_index()


# --------------------------------------------------------------------------------------
# Aggregation: levels, paired differences, decisions
# --------------------------------------------------------------------------------------


def period_mask(dates: pd.DatetimeIndex, period: str) -> np.ndarray:
    if period == DECISION_PERIOD:
        return np.any([period_mask(dates, name) for name in PERIODS], axis=0)
    start, end = PERIODS[period]
    dates = pd.DatetimeIndex(dates)
    return np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)))


def series_catalog(daily: pd.DataFrame) -> dict[tuple[str, str, int | str], pd.Series]:
    """Every per-date series of one (model, universe, segment): cells and per-horizon metrics."""
    out: dict[tuple[str, str, int | str], pd.Series] = {}
    for cell, horizons in CELLS.items():
        for kind in TARGET_KINDS:
            series = cell_series(daily, f"ic_{kind}", horizons)
            if not series.empty:
                out[(f"ic_{kind}", cell, CELL_BLOCKS[cell])] = series
    for metric in ("spread_residual", "spread_raw", "bottom_excess", "ls_industry"):
        for h in SPREAD_HORIZONS:
            series = cell_series(daily, metric, (h,))
            if not series.empty:
                out[(metric, h, spread_block(h))] = series
    return out


def _summary(values: np.ndarray, block: int) -> dict[str, float]:
    mean, low, high = stats.bootstrap_mean(values, block=block, **BOOTSTRAP)
    return {
        "days": len(values),
        "mean": float(mean),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "block_length": int(block),
    }


def level_rows(daily: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (model, universe, segment), part in daily.groupby(["model", "universe", "segment"]):
        for (metric, scope, block), series in series_catalog(part).items():
            for period in (*PERIODS, DECISION_PERIOD):
                values = series.to_numpy()[period_mask(series.index, period)]
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "universe": universe,
                        "segment": segment,
                        "metric": metric,
                        "scope": str(scope),
                        "period": period,
                        **_summary(values, block),
                    }
                )
    return rows


def paired_rows(daily: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Model minus baseline on common dates, per universe, segment, metric, scope and period."""
    rows = []
    catalogs = {
        key: series_catalog(part) for key, part in daily.groupby(["model", "universe", "segment"])
    }
    for model, baseline in pairs:
        for (m, universe, segment), catalog in catalogs.items():
            if m != model or (baseline, universe, segment) not in catalogs:
                continue
            base = catalogs[(baseline, universe, segment)]
            for (metric, scope, block), series in catalog.items():
                if (metric, scope, block) not in base:
                    continue
                other = base[(metric, scope, block)]
                common = series.index.intersection(other.index)
                if len(common) == 0:
                    continue
                diff = (series.loc[common] - other.loc[common]).sort_index()
                for period in (*PERIODS, DECISION_PERIOD):
                    values = diff.to_numpy()[period_mask(diff.index, period)]
                    if len(values) == 0:
                        continue
                    summary = _summary(values, block)
                    rows.append(
                        {
                            "model": model,
                            "baseline": baseline,
                            "universe": universe,
                            "segment": segment,
                            "metric": metric,
                            "scope": str(scope),
                            "period": period,
                            **summary,
                            "lower_bound_positive": bool(summary["ci95_low"] > 0),
                        }
                    )
    return rows


def _lookup(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in where.items()):
            return row
    return None


def decide(
    levels: list[dict[str, Any]], paired: list[dict[str, Any]], *, model: str, baseline: str
) -> dict[str, Any]:
    """H-H and G1 from the decision-period rows (segment ``all``)."""

    def ic_diff(universe: str):
        return _lookup(
            paired,
            model=model,
            baseline=baseline,
            universe=universe,
            segment="all",
            metric="ic_residual",
            scope="holding",
            period=DECISION_PERIOD,
        )

    def spread(label: str, universe: str):
        return _lookup(
            levels,
            model=label,
            universe=universe,
            segment="all",
            metric="spread_residual",
            scope="10",
            period=DECISION_PERIOD,
        )

    hh_ic = ic_diff("all")
    hh_f2, hh_f0 = spread(model, H_H_SPREAD_UNIVERSE), spread(baseline, H_H_SPREAD_UNIVERSE)
    hh_i = bool(hh_ic and hh_ic["lower_bound_positive"])
    hh_ii = bool(
        hh_f2 and hh_f0 and hh_f0["mean"] > 0 and hh_f2["mean"] >= SPREAD_MULTIPLE * hh_f0["mean"]
    )
    g1 = {}
    for universe in G1_UNIVERSES:
        diff, f2, f0 = ic_diff(universe), spread(model, universe), spread(baseline, universe)
        g1[universe] = {
            "ic_difference": None if diff is None else diff["mean"],
            "ic_ci95_low": None if diff is None else diff["ci95_low"],
            "ic_lower_bound_positive": bool(diff and diff["lower_bound_positive"]),
            "spread_model": None if f2 is None else f2["mean"],
            "spread_baseline": None if f0 is None else f0["mean"],
            "spread_not_below": bool(f2 and f0 and f2["mean"] >= f0["mean"]),
        }
        g1[universe]["pass"] = (
            g1[universe]["ic_lower_bound_positive"] and g1[universe]["spread_not_below"]
        )
    return {
        "model": model,
        "baseline": baseline,
        "period": DECISION_PERIOD,
        "H-H": {
            "ic_universe": "all",
            "ic_difference": None if hh_ic is None else hh_ic["mean"],
            "ic_ci95": None if hh_ic is None else [hh_ic["ci95_low"], hh_ic["ci95_high"]],
            "condition_i": hh_i,
            "spread_universe": H_H_SPREAD_UNIVERSE,
            "spread_model": None if hh_f2 is None else hh_f2["mean"],
            "spread_baseline": None if hh_f0 is None else hh_f0["mean"],
            "spread_multiple_required": SPREAD_MULTIPLE,
            "condition_ii": hh_ii,
            "pass": hh_i and hh_ii,
        },
        "G1": {**g1, "pass": any(g1[u]["pass"] for u in g1)},
    }


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


def run(
    predictions: list[Path],
    targets: list[Path],
    models: list[str],
    labels: list[str],
    prices: Path,
    taifex_root: Path,
    reference: Path,
    output_root: Path,
    *,
    baseline: str,
    min_volume: float,
    lookback: int,
) -> dict[str, Any]:
    if not (len(predictions) == len(targets) == len(models) == len(labels)) or not labels:
        raise ValueError("Give one --predictions, --targets, --model and --label per run")
    if baseline not in labels or len(set(labels)) != len(labels):
        raise ValueError("--baseline must be one of the distinct --label values")
    price_frame = load_prices(prices)
    prior = trailing_returns(price_frame, LOOKBACK_SESSIONS)
    eligibility, audit = eligibility_frame(taifex_root, min_volume=min_volume, lookback=lookback)
    industry_by_symbol = load_reference(reference)
    horizons = tuple(sorted(set(IC_HORIZONS) | set(SPREAD_HORIZONS)))
    dailies = []
    for path, target_path, model, label in zip(predictions, targets, models, labels, strict=True):
        frame = load_predictions(path, model)
        inputs = build_inputs(frame, price_frame)
        frame = frame.merge(
            load_targets(target_path, horizons),
            on=["symbol", "signal_date"],
            how="left",
            validate="one_to_one",
        )
        frame = frame.merge(prior, on=["symbol", "signal_date"], how="left", validate="one_to_one")
        frame["industry"] = [industry_by_symbol.get(s, "UNK") for s in frame["symbol"]]
        for universe, (mask, group) in universe_masks(frame, inputs, eligibility).items():
            part = frame[mask].assign(group=group[mask])
            daily = model_daily(part, label, universe)
            if not daily.empty:
                dailies.append(daily)
    daily = pd.concat(dailies, ignore_index=True)
    pairs = [(label, baseline) for label in labels if label != baseline]
    levels = level_rows(daily)
    paired = paired_rows(daily, pairs)
    verdicts = [decide(levels, paired, model=label, baseline=baseline) for label, _ in pairs]
    output_root.mkdir(parents=True, exist_ok=False)
    daily.to_csv(output_root / "daily.csv", index=False)
    pd.DataFrame(levels).to_csv(output_root / "grid.csv", index=False)
    pd.DataFrame(paired).to_csv(output_root / "paired.csv", index=False)
    report = {
        "kind": "chip-increment",
        "protocol": "13.18 items 4-5 (H-H, G1, G1 sub-analysis)",
        "runs": {
            label: {"predictions": str(p), "targets": str(t), "model": m}
            for p, t, m, label in zip(predictions, targets, models, labels, strict=True)
        },
        "baseline": baseline,
        "prices": str(prices),
        "taifex_root": str(taifex_root),
        "reference": str(reference),
        "min_volume": min_volume,
        "lookback": lookback,
        "quantile": QUANTILE,
        "winner_share": WINNER_SHARE,
        "trailing_sessions": LOOKBACK_SESSIONS,
        "bootstrap": BOOTSTRAP,
        "universes": list(UNIVERSES),
        "segments": list(SEGMENTS),
        "verdicts": verdicts,
        "eligibility_audit": audit,
        "rows": {"levels": len(levels), "paired": len(paired), "daily": len(daily)},
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(
        summary_markdown(pd.DataFrame(levels), pd.DataFrame(paired), verdicts), encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def _fmt(metric: str, value: float) -> str:
    return runner.number(value) if metric.startswith("ic_") else runner.percent(value)


def _cell(sel: pd.DataFrame, metric: str, scope: str, *, interval=False, annual=False) -> str:
    row = sel[(sel["metric"] == metric) & (sel["scope"] == scope)]
    if row.empty:
        return "—"
    row = row.iloc[0]
    if annual:
        return runner.percent(0.5 * row["mean"] * 252.0 / LS_HORIZON)
    text = _fmt(metric, row["mean"])
    if interval:
        text += f" [{_fmt(metric, row['ci95_low'])}, {_fmt(metric, row['ci95_high'])}]"
    return text


def _days(sel: pd.DataFrame) -> str:
    days = sel[(sel["metric"] == "ic_residual") & (sel["scope"] == "holding")]
    return str(int(days.iloc[0]["days"])) if not days.empty else "—"


def _paired_table(paired: pd.DataFrame, segment: str) -> list[str]:
    columns = (
        "對照 / Pair",
        "宇宙 / Universe",
        "期間 / Period",
        "ΔIC 殘差 持有 [95%]",
        "ΔIC 原始 持有",
        "ΔIC 殘差 短",
        "Δ四分位價差 殘差 h10 [95%]",
        "Δ四分位價差 原始 h10",
        "Δ後四分位超額 h10",
        "天數 / Days",
    )
    header = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    if paired.empty:
        return header
    rows = []
    part = paired[paired["segment"] == segment]
    keys = part[["model", "baseline", "universe", "period"]].drop_duplicates()
    for _, key in keys.iterrows():
        sel = part[
            (part["model"] == key.model)
            & (part["baseline"] == key.baseline)
            & (part["universe"] == key.universe)
            & (part["period"] == key.period)
        ]
        cells = [
            f"{key.model} - {key.baseline}",
            key.universe,
            key.period,
            _cell(sel, "ic_residual", "holding", interval=True),
            _cell(sel, "ic_raw", "holding"),
            _cell(sel, "ic_residual", "short"),
            _cell(sel, "spread_residual", "10", interval=True),
            _cell(sel, "spread_raw", "10"),
            _cell(sel, "bottom_excess", "10"),
            _days(sel),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return header + rows


def _level_table(levels: pd.DataFrame, segment: str) -> list[str]:
    columns = (
        "模型 / Model",
        "宇宙 / Universe",
        "期間 / Period",
        "IC 殘差 持有 [95%]",
        "IC 原始 持有",
        "四分位價差 殘差 h10 [95%]",
        "四分位價差 原始 h10",
        "後四分位超額 h10",
        "產業中性多空毛年化 h10",
        "天數 / Days",
    )
    header = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    if levels.empty:
        return header
    rows = []
    part = levels[levels["segment"] == segment]
    keys = part[["model", "universe", "period"]].drop_duplicates()
    for _, key in keys.iterrows():
        sel = part[
            (part["model"] == key.model)
            & (part["universe"] == key.universe)
            & (part["period"] == key.period)
        ]
        cells = [
            key.model,
            key.universe,
            key.period,
            _cell(sel, "ic_residual", "holding", interval=True),
            _cell(sel, "ic_raw", "holding"),
            _cell(sel, "spread_residual", "10", interval=True),
            _cell(sel, "spread_raw", "10"),
            _cell(sel, "bottom_excess", "10"),
            _cell(sel, "ls_industry", "10", annual=True),
            _days(sel),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return header + rows


def _held(flag: bool) -> str:
    return "成立" if flag else "不成立"


def _pass(flag: bool) -> str:
    return "通過 / pass" if flag else "未通過 / fail"


def _verdict_lines(verdicts: list[dict[str, Any]]) -> list[str]:
    lines = []
    for v in verdicts:
        hh, g1 = v["H-H"], v["G1"]
        ic = (
            "—"
            if hh["ic_ci95"] is None
            else (
                f"{runner.number(hh['ic_difference'])} [{runner.number(hh['ic_ci95'][0])}, "
                f"{runner.number(hh['ic_ci95'][1])}]"
            )
        )
        held = _held(hh["condition_i"])
        spread = f"{runner.percent(hh['spread_model'])} 對 {runner.percent(hh['spread_baseline'])}"
        lines.append(
            f"- H-H {v['model']} - {v['baseline']}：(i) 持有格殘差 ΔIC {ic} → {held}；"
            f"(ii) {hh['spread_universe']} h10 殘差四分位價差 {spread}"
            f"（需 ≥ {SPREAD_MULTIPLE} 倍）→ {_held(hh['condition_ii'])}；**{_pass(hh['pass'])}**"
        )
        for universe in G1_UNIVERSES:
            g = g1[universe]
            spread = (
                f"{runner.percent(g['spread_model'])} 對 {runner.percent(g['spread_baseline'])}"
            )
            lines.append(
                f"- G1 {universe}：ΔIC {runner.number(g['ic_difference'])}（下界 "
                f"{runner.number(g['ic_ci95_low'])}）→ {_held(g['ic_lower_bound_positive'])}；"
                f"價差 {spread} → {'不低於' if g['spread_not_below'] else '低於'}；"
                f"**{_pass(g['pass'])}**"
            )
        lines.append(f"- G1 整體 / overall: **{_pass(g1['pass'])}**")
    return lines


ZH_NOTES = (
    "§13.18 第 4、5 項。宇宙：all = 每市場全部有分數的股票（§13.13 的格）；",
    "liquidity = §13.14 市場內流動性宇宙；small_mid = liquidity 扣除 §13.15 股票期貨流動宇宙；",
    "futures_liquid = 股票期貨流動宇宙（兩市場合併）。IC 為 Spearman Rank IC，",
    "持有格 = h 5-10 平均、市場等權；殘差 = residual_v1a 目標，原始 = 同一 run 的 alpha 目標。",
    "四分位價差 = 前四分位減後四分位的目標平均（每批）；",
    "後四分位超額 = 模型排在後四分位者的原始報酬減宇宙平均。",
    "配對差在共同日期逐日相減後做 circular block bootstrap",
    "（1,000 次、seed 11；短 5、持有與 h10 價差 20）。",
    "強勢股 = 訊號日前 20 個交易日報酬在宇宙組內前五分之一；其餘 = 其他（含缺前 20 日報酬者）。",
)
EN_NOTES = (
    "Protocol 13.18 items 4-5. Universes: all = every scored stock per market (the 13.13 cell);",
    "liquidity = the 13.14 per-market liquidity universe; small_mid = liquidity minus the 13.15",
    "liquid stock-futures universe; futures_liquid = that universe, both markets together.",
    "IC = Spearman Rank IC, holding cell = mean over h 5-10 with equal market weights; residual =",
    "the residual_v1a target, raw = the same run's alpha target. Quartile spread = top minus",
    "bottom quartile mean of the target per cohort; bottom-quartile excess = raw return of the",
    "names ranked in the bottom quartile minus the universe mean. Paired differences subtract on",
    "common dates, then a circular block bootstrap (1,000 replicates, seed 11; blocks 5 short,",
    "20 holding and h = 10 spreads). Winners = top fifth by trailing 20-session return inside",
    "the universe group; rest = everyone else (names without a trailing return included).",
)


def summary_markdown(levels: pd.DataFrame, paired: pd.DataFrame, verdicts: list[dict]) -> str:
    sections = [
        "# §13.18 籌碼增量 G1/H-H / Chip increment G1 / H-H",
        "",
        "## 中文",
        "",
        " ".join(ZH_NOTES),
        "",
        "### 判定 / Verdicts",
        *_verdict_lines(verdicts),
        "",
    ]
    for segment, title in (
        ("all", "全部 / all"),
        ("winners", "強勢股 / winners"),
        ("rest", "其餘 / rest"),
    ):
        sections += [
            f"### 配對差 / Paired differences — {title}",
            *_paired_table(paired, segment),
            "",
            f"### 水準 / Levels — {title}",
            *_level_table(levels, segment),
            "",
        ]
    sections += [
        "## English",
        "",
        " ".join(EN_NOTES),
        "",
        "The tables above are shared by both languages (headers carry both).",
        "",
    ]
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        action="append",
        required=True,
        help="targets.parquet of the same run (alpha_hNN and residual_hNN)",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="model column value inside that predictions file",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="short name of the run in the outputs (F0, F1, F2)",
    )
    parser.add_argument("--baseline", default="F0", help="label every other run is paired against")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--taifex-root", type=Path, help=f"default <artifacts>/{DEFAULT_TAIFEX}")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--min-volume", type=float, default=100.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    taifex_root = args.taifex_root or args.artifacts_root / DEFAULT_TAIFEX
    output_root = args.output_root or runner.default_output_root(
        args.artifacts_root, "chip-increment-" + "-".join(label.lower() for label in args.label)
    )
    report = run(
        args.predictions,
        args.targets,
        args.model,
        args.label,
        args.prices,
        taifex_root,
        args.reference,
        output_root,
        baseline=args.baseline,
        min_volume=args.min_volume,
        lookback=args.lookback,
    )
    for verdict in report["verdicts"]:
        hh, g1 = verdict["H-H"], verdict["G1"]
        g1_cells = ", ".join(f"{u}={g1[u]['pass']}" for u in G1_UNIVERSES)
        print(
            f"{verdict['model']} - {verdict['baseline']}: H-H {'pass' if hh['pass'] else 'fail'}"
            f" (i={hh['condition_i']}, ii={hh['condition_ii']});"
            f" G1 {'pass' if g1['pass'] else 'fail'} ({g1_cells})"
        )
    print(f"Wrote {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
