"""Run the pre-registered stage 1b hold-buffer backtest (experiment protocol section 13.5).

Writes report.json (identity checks, selection, gate, sensitivities), grid.csv
(every configuration), daily.csv (daily series of the primary configurations)
and a bilingual summary.md to a new output directory.
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

from twbacktest import data, engine, positions
from twbacktest.costs import primary_scenario, scenarios, side_costs
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

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKTEST_ROOT / "config" / "stage1b.json"
DEFAULT_ARTIFACTS = BACKTEST_ROOT.parent / "artifacts"
KEY_COLUMNS = ["split", "market", "signal", "kind", "s", "k", "b", "version", "discount",
               "slippage_multiplier"]
CHOICE_COLUMNS = ["signal", "s", "k", "b"]

_STATE: dict[str, Any] = {}


def _initialize(config: dict[str, Any], inputs: dict[str, Any], indexes: dict, totals: dict):
    _STATE.update(config=config, inputs=inputs, indexes=indexes, totals=totals, splits={},
                  markets={})


def _split(split: str) -> dict[str, Any]:
    """Links, per-row moves and cost rates of one split, loaded once per process."""
    if split not in _STATE["splits"]:
        config, index = _STATE["config"], _STATE["indexes"][split]
        absolute, alpha = data.load_returns(
            _STATE["inputs"]["alpha_baseline"], _STATE["inputs"]["direction_baseline"], index,
            _STATE["totals"][split],
        )
        link = positions.links(index)
        _STATE["splits"][split] = {
            "link": link,
            "move": positions.moves(absolute, alpha, link),
            "rates": {
                scenario: side_costs(config["costs"], scenario, index.market_codes, index.is_etf,
                                     same_day=False)
                for scenario in scenarios(config["costs"])
            },
        }
    return _STATE["splits"][split]


def _market(split: str, market: str) -> dict[str, Any]:
    key = (split, market)
    if key not in _STATE["markets"]:
        codes = tuple(data.MARKETS.index(name) for name in _STATE["config"]["markets"][market])
        groups = positions.date_groups(_STATE["indexes"][split], codes)
        _STATE["markets"][key] = {
            "groups": groups,
            "benchmark": positions.equal_weight(groups, _split(split)["move"]),
        }
    return _STATE["markets"][key]


def run_task(task: tuple[str, str, str]) -> list[dict[str, Any]]:
    """Every score horizon, K, buffer, cost scenario and version of one split, signal, market."""
    split, signal, market = task
    config = _STATE["config"]
    index = _STATE["indexes"][split]
    spec = config["signals"][signal]
    scores = data.load_signal(spec, inputs=_STATE["inputs"], index=index,
                              total_rows=_STATE["totals"][split])
    shared, universe = _split(split), _market(split, market)
    rows = []
    for s in config["score_horizons"]:
        ranked = engine.rankings(scores[:, s - 1], universe["groups"])
        for k in config["top_k"]:
            for b in config["buffer_multiples"]:
                path = positions.simulate(
                    ranked, k=k, buffer=b, symbols=index.symbol_codes, link=shared["link"],
                    move=shared["move"], limit=config["price_limit"]["open_gap"],
                )
                for row in positions.evaluate(
                    path, shared["move"], universe["benchmark"], rates=shared["rates"],
                    versions=config["versions"], annualization_days=config["annualization_days"],
                    bootstrap=config["bootstrap"],
                ):
                    rows.append({"split": split, "market": market, "signal": signal,
                                 "kind": spec["kind"], "s": s, "k": k, "b": b, **row})
    return rows


def identity_checks(config: dict[str, Any], inputs: dict[str, Any], indexes: dict,
                    totals: dict) -> dict[str, Any]:
    """Re-check the open-to-open identity for stocks and the index; stop when it fails."""
    rule = config["identity_check"]
    output = {}
    for split, index in indexes.items():
        absolute, alpha = data.load_returns(inputs["alpha_baseline"], inputs["direction_baseline"],
                                            index, totals[split])
        link = positions.links(index)
        pairs, stock = positions.identity_fraction(absolute, link, rule["tolerance"])
        _pairs, bench = positions.identity_fraction(absolute - alpha, link, rule["tolerance"])
        output[split] = {"pairs": pairs, "stock_fraction": stock, "index_fraction": bench,
                         "missing_next_rows": int((link.next_row < 0).sum()
                                                  - (index.date_codes == len(index.dates) - 1)
                                                  .sum())}
        if not (stock >= rule["min_fraction"] and bench >= rule["min_fraction"]):
            raise ValueError(f"{split} open-to-open identity holds for only {stock:.4%} of stock "
                             f"pairs and {bench:.4%} of index pairs")
    return output


def run_grid(config: dict[str, Any], artifacts: Path, workers: int | None) -> tuple:
    inputs, indexes, totals = prepare_inputs(config, artifacts)
    checks = identity_checks(config, inputs, indexes, totals)
    tasks = [(split, signal, market) for split in config["splits"]
             for signal in config["signals"] for market in config["markets"]]
    workers = workers or default_workers(len(tasks))
    rows = run_tasks(tasks, run_task, initializer=_initialize,
                     initargs=(config, inputs, indexes, totals), workers=workers)
    grid = pd.DataFrame(rows).sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    return grid, (inputs, indexes, totals), checks, workers


def _primary_mask(grid: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    primary = primary_scenario(config["costs"])
    return ((grid["discount"] == primary.discount)
            & (grid["slippage_multiplier"] == primary.slippage_multiplier))


def _match(grid: pd.DataFrame, row: pd.Series, split: str) -> pd.Series | None:
    mask = grid["split"] == split
    for column in KEY_COLUMNS[1:]:
        mask &= grid[column] == row[column]
    matches = grid[mask]
    return matches.iloc[0] if len(matches) else None


def select_and_gate(grid: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Validation-only selection of one primary configuration per market, then the gate."""
    selection, gate = config["selection"], config["gate"]
    candidates = grid[(grid["split"] == selection["split"])
                      & grid["kind"].isin(selection["signal_kinds"])
                      & (grid["version"] == selection["version"])
                      & _primary_mask(grid, config)]
    metric = selection["metric"]
    result = {}
    for market in config["markets"]:
        pool = candidates[(candidates["market"] == market) & np.isfinite(candidates[metric])]
        if pool.empty:
            result[market] = {"selected": None}
            continue
        best = pool.sort_values([metric, *CHOICE_COLUMNS],
                                ascending=[False, True, True, True, True]).iloc[0]
        test = _match(grid, best, "test")
        passed = bool(
            best["net_excess_low"] > gate["validation_lower_bound_above"]
            and test is not None and test["net_excess_mean"] > gate["test_mean_above"]
        )
        result[market] = {
            "selected": {key: best[key] for key in CHOICE_COLUMNS},
            "candidates": len(pool),
            "validation": best.to_dict(),
            "test": None if test is None else test.to_dict(),
            "passed": passed,
        }
    return result


def sensitivities(grid: pd.DataFrame, selection: dict[str, Any]) -> dict[str, Any]:
    """Every cost scenario and version of each primary configuration, both splits."""
    output = {}
    for market, item in selection.items():
        chosen = item.get("selected")
        if not chosen:
            continue
        mask = grid["market"] == market
        for column in CHOICE_COLUMNS:
            mask &= grid[column] == chosen[column]
        output[market] = grid[mask][[*KEY_COLUMNS, "net_excess_annual", "net_excess_annual_low",
                                     "net_excess_annual_high", "net_excess_ir", "sharpe",
                                     "max_drawdown", "turnover", "holding_mean"]].to_dict("records")
    return output


def best_per_signal(grid: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive: each signal's validation-best (s, K, b) per market, with its test values."""
    base = grid[(grid["version"] == "long_only") & _primary_mask(grid, config)]
    output = []
    for (market, signal), rows in base[base["split"] == "validation"].groupby(
        ["market", "signal"], sort=True
    ):
        rows = rows[np.isfinite(rows["net_excess_ir"])]
        if rows.empty:
            continue
        best = rows.sort_values(["net_excess_ir", "s", "k", "b"],
                                ascending=[False, True, True, True]).iloc[0]
        test = _match(grid, best, "test")
        output.append({
            "market": market, "signal": signal, "kind": best["kind"], "s": int(best["s"]),
            "k": int(best["k"]), "b": int(best["b"]),
            "validation_net_excess_annual": best["net_excess_annual"],
            "validation_net_excess_ir": best["net_excess_ir"],
            "test_net_excess_annual": None if test is None else test["net_excess_annual"],
            "test_net_excess_ir": None if test is None else test["net_excess_ir"],
            "test_turnover": None if test is None else test["turnover"],
            "test_holding_mean": None if test is None else test["holding_mean"],
        })
    return output


def daily_series(config: dict[str, Any], prepared: tuple, selection: dict[str, Any]
                 ) -> pd.DataFrame:
    """Daily series of each primary configuration under the primary costs, both splits."""
    inputs, indexes, totals = prepared
    _initialize(config, inputs, indexes, totals)
    primary = primary_scenario(config["costs"])
    frames = []
    for market, item in selection.items():
        chosen = item.get("selected")
        if not chosen:
            continue
        for split, index in indexes.items():
            spec = config["signals"][chosen["signal"]]
            scores = data.load_signal(spec, inputs=inputs, index=index, total_rows=totals[split])
            shared, universe = _split(split), _market(split, market)
            ranked = engine.rankings(scores[:, int(chosen["s"]) - 1], universe["groups"])
            path = positions.simulate(
                ranked, k=int(chosen["k"]), buffer=int(chosen["b"]), symbols=index.symbol_codes,
                link=shared["link"], move=shared["move"],
                limit=config["price_limit"]["open_gap"],
            )
            buy_rate, sell_rate = shared["rates"][primary]
            net = positions.replay(path, buy_rate, sell_rate)
            free = np.zeros_like(buy_rate)
            gross = positions.replay(path, free, free)
            frames.append(pd.DataFrame({
                "split": split, "market": market,
                # Interval j starts at the open of signal date j + 1.
                "open_date": list(index.dates[1:]),
                "net_return": net.daily,
                "gross_return": gross.daily,
                "equal_weight_return": universe["benchmark"],
                "net_excess": net.daily - universe["benchmark"],
                "hedged_net_return": net.daily - positions.hedge_returns(path, net,
                                                                         shared["move"]),
                "bought": (path.buy >= 0).sum(axis=1),
                "held": (path.held >= 0).sum(axis=1),
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary_markdown(report: dict[str, Any]) -> str:
    primary = report["config"]["costs"]["primary"]
    header_zh = ("| 市場 | 訊號 | s | K | b | validation 淨超額 [95% 區間] "
                 "| test 淨超額 [95% 區間] | test IR | test 最大回撤 | test 每日換手 "
                 "| test 平均持有天數 | 通過 |")
    header_en = ("| Market | Signal | s | K | b | Validation net excess [95% CI] | Test net excess "
                 "[95% CI] | Test IR | Test max drawdown | Test daily turnover | Test mean "
                 "holding (sessions) | Passed |")
    divider = "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |"
    rows_zh, rows_en = [], []
    for market, item in report["selection"].items():
        chosen = item.get("selected")
        if not chosen:
            continue
        val, test = item["validation"], item["test"] or {}
        holding = test.get("holding_mean")
        cells = (
            f"| {market} | {chosen['signal']} | {chosen['s']} | {chosen['k']} | {chosen['b']} | "
            f"{percent(val['net_excess_annual'])} [{percent(val['net_excess_annual_low'])}, "
            f"{percent(val['net_excess_annual_high'])}] | "
            f"{percent(test.get('net_excess_annual'))} "
            f"[{percent(test.get('net_excess_annual_low'))}, "
            f"{percent(test.get('net_excess_annual_high'))}] | "
            f"{number(test.get('net_excess_ir'))} | {percent(test.get('max_drawdown'))} | "
            f"{percent(test.get('turnover'))} | "
            f"{'—' if holding is None else f'{holding:.1f}'} | "
        )
        rows_zh.append(cells + ("是 |" if item["passed"] else "否 |"))
        rows_en.append(cells + ("yes |" if item["passed"] else "no |"))
    lines = [
        "# 台股回測第一階段 1b 摘要 / Taiwan Backtest Stage 1b Summary", "", "## 中文", "",
        f"- 主要成本情境：手續費折扣 {primary['discount']:g}、滑價 "
        f"{primary['slippage_multiplier']:g} 倍；開盤集合競價買賣，逐部位記帳。",
        "- 淨超額報酬：扣除成本後的每日開盤到開盤報酬，相對同日同市場全部股票等權"
        "（每日再平衡、不計成本）的年化值。",
        "- 主要設定只用 validation 選出；test 已在第一階段事後分析中看過，結果只作探索性判讀。",
        "", header_zh, divider, *rows_zh, "", "## English", "",
        f"- Primary costs: commission discount {primary['discount']:g}, slippage "
        f"x{primary['slippage_multiplier']:g}; trades in the opening auction, per-position "
        "accounting.",
        "- Net excess: annualized daily open-to-open return after costs over the same-date, "
        "same-market, daily-rebalanced, cost-free equal weight of all stocks.",
        "- Primary configurations are chosen on validation only; test was already seen in the "
        "stage 1 post-hoc analysis, so its results are exploratory.",
        "", header_en, divider, *rows_en,
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path, artifacts: Path, output_root: Path, workers: int | None) -> dict:
    config = json.loads(config_path.read_text())
    grid, prepared, checks, used_workers = run_grid(config, artifacts, workers)
    selection = select_and_gate(grid, config)
    indexes = prepared[1]
    report = {
        "schema_version": "1.0",
        "kind": "stage1b-tw-hold-buffer-backtest",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "workers": used_workers,
        "data": {
            split: {
                "dates": len(index.dates), "first_date": index.dates[0],
                "last_date": index.dates[-1],
                "rows": {market: int((index.market_codes == code).sum())
                         for code, market in enumerate(data.MARKETS)},
                "identity": checks[split],
            }
            for split, index in indexes.items()
        },
        "configurations": len(grid),
        "selection": selection,
        "gate_passed_markets": [market for market, item in selection.items()
                                if item.get("passed")],
        "sensitivity": sensitivities(grid, selection),
        "best_per_signal": best_per_signal(grid, config),
    }
    report = clean(report)
    daily = daily_series(config, prepared, selection)
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    daily.to_csv(output_root / "daily.csv", index=False)
    (output_root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (output_root / "summary.md").write_text(summary_markdown(report))
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
    output_root = args.output_root or default_output_root(args.artifacts_root, "stage1b")
    report = run(args.config, args.artifacts_root, output_root, args.workers)
    print(json.dumps({"output_root": str(output_root),
                      "gate_passed_markets": report["gate_passed_markets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
