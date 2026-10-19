"""Run the pre-registered holding-aligned evaluation (experiment protocol section 13.6).

Writes report.json (identity checks, watch list), ranking.csv (Rank IC, deciles
and session/overnight split per split, signal, market and horizon),
portfolio.csv (the fixed K, b holding portfolio) and a bilingual summary.md to a
new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import data, engine, holding, positions
from twbacktest.costs import primary_scenario, side_costs
from twbacktest.runner import (
    MAX_WORKERS,
    clean,
    default_output_root,
    default_workers,
    number,
    percent,
    prepare_inputs,
    run_tasks,
)
from twbacktest.stage1b import identity_checks

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "holding_eval.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
RANK_KEYS = ["split", "market", "signal", "kind", "h"]

_STATE: dict[str, Any] = {}


def horizons(config: dict[str, Any]) -> list[int]:
    return sorted({*config["horizons"]["primary"], *config["horizons"]["auxiliary"]})


def _initialize(config: dict[str, Any], inputs: dict[str, Any], indexes: dict, totals: dict):
    _STATE.update(config=config, inputs=inputs, indexes=indexes, totals=totals, splits={})


def _split(split: str) -> dict[str, Any]:
    """Labels, moves, cost rates and market groups of one split, loaded once per process."""
    if split not in _STATE["splits"]:
        config, index = _STATE["config"], _STATE["indexes"][split]
        absolute, alpha = data.load_returns(
            _STATE["inputs"]["alpha_baseline"], _STATE["inputs"]["direction_baseline"], index,
            _STATE["totals"][split],
        )
        table = positions.symbol_date_table(index)
        link = positions.links(index)
        move = positions.moves(absolute, alpha, link)
        groups = {
            market: positions.date_groups(
                index, tuple(data.MARKETS.index(name) for name in names)
            )
            for market, names in config["markets"].items()
        }
        _STATE["splits"][split] = {
            "current": alpha,
            "labels": {h: holding.open_to_open(absolute, alpha, index, table, h)
                       for h in horizons(config)},
            "link": link,
            "move": move,
            "groups": groups,
            "benchmark": {market: positions.equal_weight(rows, move)
                          for market, rows in groups.items()},
            "rates": side_costs(config["costs"], primary_scenario(config["costs"]),
                                index.market_codes, index.is_etf, same_day=False),
        }
    return _STATE["splits"][split]


def _ranking_rows(split: str, signal: str, kind: str, scores: np.ndarray) -> list[dict]:
    config, shared = _STATE["config"], _split(split)
    bootstrap = config["bootstrap"]
    rows = []
    for market in config["rank_markets"]:
        groups = shared["groups"][market]
        for h in horizons(config):
            labels = shared["labels"][h]
            score = scores[:, h - 1]
            block = max(int(bootstrap["min_block"]), int(bootstrap["block_per_holding_day"]) * h)
            ranks = holding.deciles(score, np.expm1(labels.stock), groups,
                                    config["decile_fraction"])
            session, overnight = holding.session_split(score, labels, groups,
                                                       config["decile_fraction"])
            row: dict[str, Any] = {"split": split, "market": market, "signal": signal,
                                   "kind": kind, "h": h}
            for name, target in (("oo", labels.alpha), ("current", shared["current"][:, h - 1])):
                summary = holding.rank_summary(holding.daily_rank_ic(score, target, groups))
                row.update({f"ic_{name}_{key}": value for key, value in summary.items()})
            for name, series in (("top", ranks.top), ("bottom", ranks.bottom),
                                 ("spread", ranks.top - ranks.bottom),
                                 ("session", session), ("overnight", overnight)):
                summary = holding.interval(series, block=block, bootstrap=bootstrap)
                row.update({f"{name}_{key}": value for key, value in summary.items()})
            member = np.concatenate(groups)
            row["label_coverage"] = float(np.isfinite(labels.stock[member]).mean())
            rows.append(row)
    return rows


def _portfolio_rows(split: str, signal: str, kind: str, scores: np.ndarray) -> list[dict]:
    config, shared = _STATE["config"], _split(split)
    index = _STATE["indexes"][split]
    portfolio = config["portfolio"]
    buy_rate, sell_rate = shared["rates"]
    free = np.zeros_like(buy_rate)
    rows = []
    for market, groups in shared["groups"].items():
        for h in horizons(config):
            ranked = engine.rankings(scores[:, h - 1], groups)
            path = positions.simulate(
                ranked, k=portfolio["k"], buffer=portfolio["buffer"], symbols=index.symbol_codes,
                link=shared["link"], move=shared["move"], limit=config["price_limit"]["open_gap"],
            )
            statistics = positions.path_statistics(path)
            metrics = positions.summarize(
                positions.replay(path, buy_rate, sell_rate).daily,
                positions.replay(path, free, free).daily, shared["benchmark"][market],
                turnover=statistics["turnover"], annualization_days=config["annualization_days"],
                bootstrap={**config["bootstrap"], "block": config["bootstrap"]["daily_block"]},
            )
            rows.append({"split": split, "market": market, "signal": signal, "kind": kind,
                         "h": h, **metrics, **statistics})
    return rows


def run_task(task: tuple[str, str]) -> list[dict[str, Any]]:
    """Ranking and portfolio rows of one split and signal, tagged by table."""
    split, signal = task
    config = _STATE["config"]
    spec = config["signals"][signal]
    scores = data.load_signal(spec, inputs=_STATE["inputs"], index=_STATE["indexes"][split],
                              total_rows=_STATE["totals"][split])
    return ([{"table": "ranking", **row}
             for row in _ranking_rows(split, signal, spec["kind"], scores)]
            + [{"table": "portfolio", **row}
               for row in _portfolio_rows(split, signal, spec["kind"], scores)])


def watch_list(ranking: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive only: primary-horizon validation top deciles with a lower bound above 0."""
    rule = config["watch_rule"]
    rows = ranking[(ranking["split"] == rule["split"])
                   & ranking["h"].isin(config["horizons"][rule["horizons"]])
                   & (ranking["top_low"] > rule["lower_bound_above"])]
    output = []
    for item in rows.itertuples(index=False):
        test = ranking[(ranking["split"] == "test") & (ranking["market"] == item.market)
                       & (ranking["signal"] == item.signal) & (ranking["h"] == item.h)]
        output.append({
            "market": item.market, "signal": item.signal, "h": int(item.h),
            "validation_top": item.top_mean, "validation_top_low": item.top_low,
            "test_top": float(test["top_mean"].iloc[0]) if len(test) else None,
            "test_top_low": float(test["top_low"].iloc[0]) if len(test) else None,
        })
    return output


def _bps(value: Any) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{1e4 * float(value):+.1f}"


def summary_markdown(report: dict[str, Any], ranking: pd.DataFrame,
                     portfolio: pd.DataFrame) -> str:
    primary = report["config"]["horizons"]["primary"]

    def rows() -> list[str]:
        output = []
        for market in report["config"]["rank_markets"]:
            for h in primary:
                part = ranking[(ranking["market"] == market) & (ranking["h"] == h)]
                book = portfolio[(portfolio["market"] == market) & (portfolio["h"] == h)]
                for signal in report["config"]["signals"]:
                    val = part[(part["split"] == "validation") & (part["signal"] == signal)]
                    test = part[(part["split"] == "test") & (part["signal"] == signal)]
                    val_book = book[(book["split"] == "validation") & (book["signal"] == signal)]
                    test_book = book[(book["split"] == "test") & (book["signal"] == signal)]
                    if val.empty or test.empty:
                        continue
                    v, t = val.iloc[0], test.iloc[0]
                    output.append(
                        f"| {market} | {h} | {signal} | {number(v['ic_oo_mean'])} | "
                        f"{number(t['ic_oo_mean'])} | {_bps(v['top_mean'])} "
                        f"[{_bps(v['top_low'])}, {_bps(v['top_high'])}] | {_bps(t['top_mean'])} | "
                        f"{_bps(v['session_mean'])} / {_bps(v['overnight_mean'])} | "
                        f"{percent(val_book['net_excess_annual'].iloc[0])} | "
                        f"{percent(test_book['net_excess_annual'].iloc[0])} |"
                    )
        return output

    header_zh = ("| 市場 | h | 訊號 | validation Rank IC | test Rank IC | validation 前十分位超額 "
                 "bps [95% 區間] | test 前十分位超額 bps | validation 交易時段 / 隔夜 bps "
                 "| validation 組合淨超額 | test 組合淨超額 |")
    header_en = ("| Market | h | Signal | Validation Rank IC | Test Rank IC | Validation "
                 "top-decile excess bps [95% CI] | Test top-decile excess bps | Validation "
                 "session / overnight bps | Validation portfolio net excess | Test portfolio "
                 "net excess |")
    divider = "| --- | ---: | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |"
    table = rows()
    watch = [f"- {item['market']} h={item['h']} {item['signal']}"
             for item in report["watch_list"]]
    watch_zh, watch_en = watch or ["- 無"], watch or ["- None"]
    portfolio_rule = report["config"]["portfolio"]
    lines = [
        "# 與持有方式對齊的評估摘要 / Holding-Aligned Evaluation Summary", "", "## 中文", "",
        "- 標籤：次一交易日開盤到 h 日後開盤的報酬（開盤到開盤）；Rank IC 對開盤到開盤 alpha。",
        "- 前十分位超額：每日前 10% 減同日同市場等權，單位 bps（每個 h 日窗口）。",
        f"- 組合：K = {portfolio_rule['k']}、b = {portfolio_rule['buffer']}、主要成本情境、"
        "只做多，年化淨超額相對等權。",
        "- 只作描述，不設門檻；test 只作探索性判讀。", "",
        header_zh, divider, *table, "",
        "主要天期中 validation 前十分位超額區間下界大於 0 的項目（可留意，不據此做決策）：", "",
        *watch_zh, "", "## English", "",
        "- Labels: returns from the next session's open to the open h sessions later "
        "(open-to-open); Rank IC is against open-to-open alpha.",
        "- Top-decile excess: the daily top 10% minus the same-date, same-market equal weight, "
        "in bps per h-session window.",
        f"- Portfolio: K = {portfolio_rule['k']}, b = {portfolio_rule['buffer']}, primary costs, "
        "long-only, annual net excess over equal weight.",
        "- Descriptive only, with no gate; test is exploratory.", "",
        header_en, divider, *table, "",
        "Primary-horizon validation top deciles with a lower bound above 0 (worth watching; no "
        "decision rests on them):", "", *watch_en,
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path, artifacts: Path, output_root: Path, workers: int | None) -> dict:
    config = json.loads(config_path.read_text())
    if max(horizons(config)) > holding.MAX_HORIZON:
        raise ValueError(f"Horizons above {holding.MAX_HORIZON} need prices beyond the targets")
    inputs, indexes, totals = prepare_inputs(config, artifacts)
    checks = identity_checks(config, inputs, indexes, totals)
    tasks = [(split, signal) for split in config["splits"] for signal in config["signals"]]
    used_workers = workers or default_workers(len(tasks))
    rows = run_tasks(tasks, run_task, initializer=_initialize,
                     initargs=(config, inputs, indexes, totals), workers=used_workers)
    frame = pd.DataFrame(rows)
    ranking = (frame[frame["table"] == "ranking"].dropna(axis=1, how="all")
               .drop(columns="table").sort_values(RANK_KEYS, kind="stable")
               .reset_index(drop=True))
    portfolio = (frame[frame["table"] == "portfolio"].dropna(axis=1, how="all")
                 .drop(columns="table").sort_values(RANK_KEYS, kind="stable")
                 .reset_index(drop=True))
    report = clean({
        "schema_version": "1.0",
        "kind": "holding-aligned-evaluation",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "workers": used_workers,
        "data": {
            split: {"dates": len(index.dates), "first_date": index.dates[0],
                    "last_date": index.dates[-1], "identity": checks[split]}
            for split, index in indexes.items()
        },
        "ranking_rows": len(ranking),
        "portfolio_rows": len(portfolio),
        "watch_list": watch_list(ranking, config),
    })
    output_root.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(output_root / "ranking.csv", index=False)
    portfolio.to_csv(output_root / "portfolio.csv", index=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report, ranking, portfolio))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path, help="New directory for this run.")
    parser.add_argument("--workers", type=int, help=f"Process workers, 1-{MAX_WORKERS}.")
    args = parser.parse_args(argv)
    if args.workers is not None and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1-{MAX_WORKERS}")
    output_root = args.output_root or default_output_root(args.artifacts_root, "holding-eval")
    report = run(args.config, args.artifacts_root, output_root, args.workers)
    print(json.dumps({"output_root": str(output_root),
                      "watch_list": len(report["watch_list"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
