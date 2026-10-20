"""Exploratory capturability analysis of the Kronos ranking (experiment protocol section 13.9).

Part A, before costs: where the ranking information lies on validation. For every signal,
inside each market (TWSE, TPEX) and each segment of it:

- segments: Rank IC, top/bottom decile excess and their spread, with stocks ranked and
  averaged inside the segment (with or without stock futures, liquid futures, terciles of
  the traded value over the 20 sessions up to the signal date);
- decile curves: the excess of all ten deciles;
- decay: Rank IC for h = 1..13 and the session/overnight split of both extreme deciles.

Excess is open-to-open over the same-date equal weight of the compared stocks (section
13.6). Writes segments.csv, deciles.csv, decay.csv, report.json and a bilingual summary.md
to a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from twbacktest import data, holding, positions, taifex
from twbacktest.runner import (
    MAX_WORKERS,
    clean,
    default_output_root,
    default_workers,
    prepare_inputs,
    run_tasks,
)

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "capturability.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
MARKETS = ("TWSE", "TPEX")
SEGMENTS = ("all", "futures", "no_futures", "liquid_futures",
            "liquidity_low", "liquidity_mid", "liquidity_high")
DECILES = 10

_STATE: dict[str, Any] = {}


def traded_value(bar_store: Path, symbols: set[str], start: str, end: str,
                 lookback: int) -> pd.DataFrame:
    """Mean close x volume over the ``lookback`` sessions up to each date, inclusive."""
    timestamp = pa.timestamp("ns", tz="UTC")
    table = ds.dataset(bar_store / "shards", format="parquet").to_table(
        columns=["timestamp", "symbol", "close", "volume"],
        filter=(ds.field("market").isin(list(MARKETS))
                & (ds.field("timestamp") >= pa.scalar(pd.Timestamp(start, tz="UTC"), timestamp))
                & (ds.field("timestamp") <= pa.scalar(pd.Timestamp(end, tz="UTC"), timestamp))),
    )
    frame = table.to_pandas()
    frame = frame[frame["symbol"].isin(symbols)].copy()
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["value"] = frame["close"] * frame["volume"]
    frame = frame.sort_values(["symbol", "date"])
    frame["mean_value"] = frame.groupby("symbol")["value"].transform(
        lambda values: values.rolling(lookback, min_periods=lookback).mean())
    return frame[["date", "symbol", "mean_value"]].reset_index(drop=True)


def load_segment_inputs(config: dict[str, Any], artifacts: Path, index: data.SplitIndex
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Daily futures eligibility and trailing traded value for the split's stocks."""
    segments = config["segments"]
    futures, listing = taifex.load(artifacts / config["inputs"]["taifex"])
    daily, audit = taifex.eligibility(
        futures, listing, min_volume=float(segments["liquid_futures_min_volume"]),
        lookback=int(segments["liquid_futures_lookback_sessions"]),
    )
    missing = sorted(set(index.dates) - set(daily["date"]))
    if missing:
        raise ValueError(f"TAIFEX data lacks {len(missing)} signal dates, first {missing[0]}")
    first = (pd.Timestamp(index.dates[0]) - timedelta(days=60)).date().isoformat()
    liquidity = traded_value(artifacts / config["inputs"]["bar_store"], set(index.symbol_names),
                             first, index.dates[-1], int(segments["liquidity_lookback_sessions"]))
    return daily, liquidity, audit


def segment_masks(index: data.SplitIndex, futures: pd.DataFrame, liquidity: pd.DataFrame,
                  groups: int) -> dict[str, np.ndarray]:
    """Row masks of every segment; liquidity groups are formed within each date and market."""
    keys = pd.MultiIndex.from_arrays([
        np.asarray(index.dates)[index.date_codes],
        np.asarray(index.symbol_names)[index.symbol_codes],
    ])
    table = futures.set_index(["date", "symbol"])
    listed = keys.isin(table.index)
    liquid = table["liquid"].astype(bool).reindex(keys, fill_value=False).to_numpy(dtype=bool)
    liquid &= listed
    value = (liquidity.set_index(["date", "symbol"])["mean_value"].reindex(keys)
             .to_numpy(dtype=float))
    if groups != 3:
        raise ValueError("The registered liquidity segments are terciles")
    frame = pd.DataFrame({"date": index.date_codes, "market": index.market_codes,
                          "value": value})
    # Percentile rank in (0, 1] within the date and market -> group 0 .. groups - 1.
    share = frame.groupby(["date", "market"])["value"].rank(method="first", pct=True)
    level = np.ceil(share.to_numpy() * groups) - 1
    return {
        "all": np.ones(index.count, dtype=bool),
        "futures": listed,
        "no_futures": ~listed,
        "liquid_futures": liquid,
        "liquidity_low": level == 0,
        "liquidity_mid": level == 1,
        "liquidity_high": level == groups - 1,
    }


def decile_curve(scores: np.ndarray, simple: np.ndarray, groups, count: int) -> np.ndarray:
    """[dates, count] excess of each score bucket (bucket 0 = highest scores)."""
    output = np.full((len(groups), count), np.nan)
    for j, rows in enumerate(groups):
        valid = rows[np.isfinite(scores[rows]) & np.isfinite(simple[rows])]
        if valid.size < count:
            continue
        order = valid[np.argsort(-scores[valid], kind="stable")]
        mean = simple[valid].mean()
        for bucket, chosen in enumerate(np.array_split(order, count)):
            output[j, bucket] = simple[chosen].mean() - mean
    return output


def _initialize(config: dict[str, Any], inputs: dict[str, Any], index: data.SplitIndex,
                total: int, masks: dict[str, np.ndarray]) -> None:
    _STATE.clear()
    _STATE.update(config=config, inputs=inputs, index=index, total=total, masks=masks)


def _shared() -> dict[str, Any]:
    """Labels and groups of the split, built once per process."""
    if "labels" not in _STATE:
        config, index = _STATE["config"], _STATE["index"]
        absolute, alpha = data.load_returns(
            _STATE["inputs"]["alpha_baseline"], _STATE["inputs"]["direction_baseline"], index,
            _STATE["total"],
        )
        table = positions.symbol_date_table(index)
        needed = sorted({*config["horizons"]["segments"], *config["horizons"]["deciles"],
                         *config["horizons"]["decay"]})
        _STATE["labels"] = {h: holding.open_to_open(absolute, alpha, index, table, h)
                            for h in needed}
        _STATE["groups"] = {
            market: positions.date_groups(index, (data.MARKETS.index(market),))
            for market in MARKETS
        }
    return _STATE


def _block(h: int) -> int:
    bootstrap = _STATE["config"]["bootstrap"]
    return max(int(bootstrap["min_block"]), int(bootstrap["block_per_holding_day"]) * h)


def _segment_rows(signal: str, scores: np.ndarray) -> list[dict[str, Any]]:
    shared, config = _shared(), _STATE["config"]
    rows = []
    for market in MARKETS:
        for segment in SEGMENTS:
            mask = _STATE["masks"][segment]
            groups = tuple(rows_[mask[rows_]] for rows_ in shared["groups"][market])
            for h in config["horizons"]["segments"]:
                labels, score = shared["labels"][h], scores[:, h - 1]
                ranks = holding.deciles(score, np.expm1(labels.stock), groups,
                                        config["decile_fraction"])
                ic = holding.rank_summary(holding.daily_rank_ic(score, labels.alpha, groups))
                row: dict[str, Any] = {"signal": signal, "market": market, "segment": segment,
                                       "h": h, "ic_mean": ic["mean"], "ic_ir": ic["ir"],
                                       "ic_dates": ic["dates"]}
                for name, series in (("top", ranks.top), ("bottom", ranks.bottom),
                                     ("spread", ranks.top - ranks.bottom)):
                    summary = holding.interval(series, block=_block(h),
                                               bootstrap=config["bootstrap"])
                    row.update({f"{name}_{key}": value for key, value in summary.items()})
                sizes = [int(np.isfinite(labels.stock[g]).sum()) for g in groups]
                row["stocks_per_date"] = float(np.mean(sizes)) if sizes else 0.0
                rows.append(row)
    return rows


def _decile_rows(signal: str, scores: np.ndarray) -> list[dict[str, Any]]:
    shared, config = _shared(), _STATE["config"]
    rows = []
    for market in MARKETS:
        for segment in ("all", "futures"):
            mask = _STATE["masks"][segment]
            groups = tuple(rows_[mask[rows_]] for rows_ in shared["groups"][market])
            for h in config["horizons"]["deciles"]:
                labels = shared["labels"][h]
                curve = decile_curve(scores[:, h - 1], np.expm1(labels.stock), groups, DECILES)
                for bucket in range(DECILES):
                    summary = holding.interval(curve[:, bucket], block=_block(h),
                                               bootstrap=config["bootstrap"])
                    rows.append({"signal": signal, "market": market, "segment": segment, "h": h,
                                 "decile": bucket + 1,
                                 **{f"excess_{key}": value for key, value in summary.items()}})
    return rows


def _decay_rows(signal: str, scores: np.ndarray) -> list[dict[str, Any]]:
    shared, config = _shared(), _STATE["config"]
    fraction = config["decile_fraction"]
    rows = []
    for market in MARKETS:
        groups = shared["groups"][market]
        for h in config["horizons"]["decay"]:
            labels, score = shared["labels"][h], scores[:, h - 1]
            ic = holding.rank_summary(holding.daily_rank_ic(score, labels.alpha, groups))
            ranks = holding.deciles(score, np.expm1(labels.stock), groups, fraction)
            top_session, top_overnight = holding.session_split(score, labels, groups, fraction)
            # The bottom decile is the top decile of the negated scores.
            bottom_session, bottom_overnight = holding.session_split(-score, labels, groups,
                                                                     fraction)
            rows.append({
                "signal": signal, "market": market, "h": h, "ic_mean": ic["mean"],
                "ic_ir": ic["ir"], "top_mean": float(np.nanmean(ranks.top)),
                "bottom_mean": float(np.nanmean(ranks.bottom)),
                "top_session": float(np.nanmean(top_session)),
                "top_overnight": float(np.nanmean(top_overnight)),
                "bottom_session": float(np.nanmean(bottom_session)),
                "bottom_overnight": float(np.nanmean(bottom_overnight)),
            })
    return rows


def run_task(signal: str) -> list[dict[str, Any]]:
    """All part A rows of one signal, tagged by table."""
    spec = _STATE["config"]["signals"][signal]
    scores = data.load_signal(spec, inputs=_STATE["inputs"], index=_STATE["index"],
                              total_rows=_STATE["total"])
    return ([{"table": "segments", **row} for row in _segment_rows(signal, scores)]
            + [{"table": "deciles", **row} for row in _decile_rows(signal, scores)]
            + [{"table": "decay", **row} for row in _decay_rows(signal, scores)])


def _bps(value: Any) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 1e4:+.1f}"


def summary_markdown(report: dict[str, Any], segments: pd.DataFrame, deciles: pd.DataFrame,
                     decay: pd.DataFrame) -> str:
    """Compact bilingual tables at h = 10: segments, decile curves and decay."""
    h = max(report["horizons"]["segments"])
    names = {"zh": {"all": "全部", "futures": "有個股期貨", "no_futures": "無個股期貨",
                    "liquid_futures": "期貨流動性足", "liquidity_low": "成交值低",
                    "liquidity_mid": "成交值中", "liquidity_high": "成交值高"},
             "en": {"all": "All", "futures": "With stock futures",
                    "no_futures": "Without stock futures", "liquid_futures": "Liquid futures",
                    "liquidity_low": "Low traded value", "liquidity_mid": "Mid traded value",
                    "liquidity_high": "High traded value"}}
    text = {"zh": ("可捕捉性分析 A：排序資訊在哪裡（validation，探索性）",
                   f"{h} 日，Rank IC 與前後十分位超額（bps，同群等權為基準）",
                   "訊號", "市場", "群組", "每日檔數", "前十分位", "後十分位", "前後差",
                   f"{h} 日十分位曲線（全部標的，bps；1 = 分數最高）",
                   "Rank IC 衰減（上市，開盤到開盤）"),
            "en": ("Capturability analysis A: where the ranking information lies "
                   "(validation, exploratory)",
                   f"{h} sessions: Rank IC and top/bottom decile excess (bps, over the "
                   "segment's equal weight)",
                   "Signal", "Market", "Segment", "Stocks per date", "Top decile",
                   "Bottom decile", "Spread",
                   f"{h}-session decile curve (all stocks, bps; 1 = highest scores)",
                   "Rank IC decay (TWSE, open to open)")}
    lines = [f"# {text['zh'][0]} / {text['en'][0]}", ""]
    for language, heading in (("zh", "## 中文"), ("en", "## English")):
        t = text[language]
        lines += [heading, "", f"### {t[1]}", "",
                  f"| {t[2]} | {t[3]} | {t[4]} | {t[5]} | Rank IC (IR) | {t[6]} | {t[7]} | "
                  f"{t[8]} |", "| --- | --- | --- | ---: | --- | --- | ---: | --- |"]
        part = segments[segments["h"] == h]
        for _, row in part.iterrows():
            lines.append(
                f"| {row['signal']} | {row['market']} | {names[language][row['segment']]} | "
                f"{row['stocks_per_date']:.0f} | {row['ic_mean']:.3f} ({row['ic_ir']:.2f}) | "
                f"{_bps(row['top_mean'])} [{_bps(row['top_low'])}, {_bps(row['top_high'])}] | "
                f"{_bps(row['bottom_mean'])} | {_bps(row['spread_mean'])} "
                f"[{_bps(row['spread_low'])}, {_bps(row['spread_high'])}] |")
        lines += ["", f"### {t[9]}", "",
                  f"| {t[2]} | {t[3]} | " + " | ".join(str(i) for i in range(1, DECILES + 1))
                  + " |", "| --- | --- | " + " | ".join("---:" for _ in range(DECILES)) + " |"]
        curve = deciles[(deciles["h"] == h) & (deciles["segment"] == "all")]
        for (signal, market), group in curve.groupby(["signal", "market"], sort=False):
            values = group.sort_values("decile")["excess_mean"]
            lines.append(f"| {signal} | {market} | " + " | ".join(_bps(v) for v in values)
                         + " |")
        horizons = sorted(decay["h"].unique())
        lines += ["", f"### {t[10]}", "",
                  f"| {t[2]} | " + " | ".join(f"h={v}" for v in horizons) + " |",
                  "| --- | " + " | ".join("---:" for _ in horizons) + " |"]
        part = decay[decay["market"] == "TWSE"]
        for signal, group in part.groupby("signal", sort=False):
            values = group.sort_values("h")["ic_mean"]
            lines.append(f"| {signal} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


def run(config_path: Path, artifacts: Path, output_root: Path, workers: int | None,
        segment_inputs=None) -> dict[str, Any]:
    """Part A on the configured split; ``segment_inputs`` replaces the loaders in tests."""
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    config = json.loads(config_path.read_text())
    if config["splits"] != ["validation"]:
        raise ValueError("Protocol section 13.9 selects on validation only")
    inputs, indexes, totals = prepare_inputs(config, artifacts)
    index, total = indexes["validation"], totals["validation"]
    loader = segment_inputs or load_segment_inputs
    futures, liquidity, audit = loader(config, artifacts, index)
    masks = segment_masks(index, futures, liquidity,
                          int(config["segments"]["liquidity_groups"]))
    signals = list(config["signals"])
    count = workers or default_workers(len(signals))
    rows = run_tasks(signals, run_task, initializer=_initialize,
                     initargs=(config, inputs, index, total, masks), workers=count)
    frames = {name: pd.DataFrame([{k: v for k, v in row.items() if k != "table"}
                                  for row in rows if row["table"] == name])
              for name in ("segments", "deciles", "decay")}
    order = {"signal": signals, "market": list(MARKETS), "segment": list(SEGMENTS)}
    for name, frame in frames.items():
        for column, values in order.items():
            if column in frame:
                frame[column] = pd.Categorical(frame[column], categories=values, ordered=True)
        keys = [c for c in ("signal", "market", "segment", "h", "decile") if c in frame]
        frames[name] = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        for column in order:
            if column in frames[name]:
                frames[name][column] = frames[name][column].astype(str)
    coverage = {market: {segment: float(np.mean([
        int(mask[g].sum()) for g in positions.date_groups(index, (data.MARKETS.index(market),))
    ])) for segment, mask in masks.items()} for market in MARKETS}
    report = {
        "schema_version": "1.0",
        "kind": "capturability-part-a",
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split": "validation",
        "dates": [index.dates[0], index.dates[-1], len(index.dates)],
        "signals": signals,
        "horizons": config["horizons"],
        "taifex": audit,
        "stocks_per_date": coverage,
        "workers": count,
    }
    output_root.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / "report.json").write_text(json.dumps(clean(report), ensure_ascii=False,
                                                        indent=1) + "\n")
    (output_root / "summary.md").write_text(
        summary_markdown(report, frames["segments"], frames["deciles"], frames["decay"]))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, help=f"Processes, 1-{MAX_WORKERS} (default: auto)")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    output = args.output_root or default_output_root(args.artifacts_root, "capturability")
    report = run(args.config, args.artifacts_root, output, args.workers)
    print(json.dumps({"output": str(output), "stocks_per_date": report["stocks_per_date"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
