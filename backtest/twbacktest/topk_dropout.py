"""Top-k dropout backtest of walk-forward ranking scores (protocol 13.13 item 3).

Every session, by the previous close's ranking, the ``drop`` lowest-ranked holdings are sold
and the highest-ranked non-holdings are bought at the next open; capital sits in K slots with
per-position open-to-open accounting (the stage 1b engine). The score horizon is the one with
the highest TWSE net excess IR on P1 under the primary costs; it is then fixed for P2 and the
decision period, and H-B is judged on the TWSE annualized net excess over P1+P2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import engine, positions, runner
from twbacktest.costs import Scenario, primary_scenario, scenarios, side_costs
from twbacktest.data import MARKETS
from twbacktest.walkforward_data import (
    WalkForwardInputs,
    build_inputs,
    load_predictions,
    load_prices,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "topk_dropout.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DECISION_PERIOD = "P1+P2"


def simulate_dropout(
    ranked: list[np.ndarray],
    *,
    k: int,
    drop: int,
    symbols: np.ndarray,
    link: positions.Links,
    move: positions.Moves,
    limit: float,
) -> positions.Path:
    """Run the top-k dropout rule; ``ranked`` holds each signal date's rows best to worst."""
    intervals = len(ranked) - 1
    if intervals < 1:
        raise ValueError("At least two signal dates are needed")
    if not 1 <= drop <= k:
        raise ValueError("drop must be between 1 and k")
    rank = np.full(symbols.size, np.inf)
    for rows in ranked:
        rank[rows] = np.arange(1, rows.size + 1)
    up, down = math.log1p(limit), math.log1p(-limit)
    shape = (intervals, k)
    buy, sell_open, sell_close, held = (np.full(shape, -1, dtype=np.int64) for _ in range(4))
    growth = np.zeros(shape)
    slot = np.full(k, -1, dtype=np.int64)
    entry = np.zeros(k, dtype=np.int64)
    waiting = np.zeros(k, dtype=bool)
    periods: list[int] = []
    blocked_buys = blocked_sells = missing_exits = 0
    for j in range(intervals):
        sold = set()
        # Sell the `drop` worst-ranked holdings (unranked holdings count as worst).
        held_slots = [s for s in range(k) if slot[s] >= 0]
        held_slots.sort(key=lambda s: (-rank[slot[s]], s))
        for s in held_slots[:drop]:
            row = slot[s]
            if move.gap[row] <= down:
                blocked_sells += 1
                continue
            sell_open[j, s] = row
            sold.add(int(symbols[row]))
            periods.append(j - int(entry[s]))
            slot[s] = -1
        taken = {int(symbols[row]) for row in slot if row >= 0} | sold
        candidates = (row for row in ranked[j] if int(symbols[row]) not in taken)
        for s in range(k):
            if slot[s] >= 0 or waiting[s]:
                continue
            row = next(candidates, None)
            if row is None:
                break
            if move.gap[row] >= up:
                blocked_buys += 1
                continue
            buy[j, s] = row
            slot[s] = row
            entry[s] = j
        waiting[:] = False
        for s in range(k):
            row = slot[s]
            if row < 0:
                continue
            held[j, s] = row
            following = link.next_row[row]
            if following >= 0 and np.isfinite(move.growth[row]):
                growth[j, s] = move.growth[row]
                slot[s] = following
            else:
                growth[j, s] = move.fallback[row] if np.isfinite(move.fallback[row]) else 0.0
                sell_close[j, s] = row
                missing_exits += 1
                periods.append(j - int(entry[s]) + 1)
                slot[s] = -1
                waiting[s] = True
    periods.extend(intervals - int(entry[s]) for s in range(k) if slot[s] >= 0)
    return positions.Path(
        buy=buy,
        sell_open=sell_open,
        sell_close=sell_close,
        held=held,
        growth=growth,
        final=slot.copy(),
        holding_periods=np.array(periods, dtype=np.int64),
        blocked_buys=blocked_buys,
        blocked_sells=blocked_sells,
        missing_exits=missing_exits,
    )


def period_masks(
    interval_dates: pd.DatetimeIndex, periods: dict[str, list[str]],
    decision: list[str] | None = None,
) -> dict:
    """Boolean masks per named period, plus the decision union ``P1+P2`` over ``decision``
    (every period when None, so a forward period such as P3 stays out of the union only when
    the configuration names the decision periods)."""
    masks = {}
    union = np.zeros(len(interval_dates), dtype=bool)
    names = list(periods) if decision is None else list(decision)
    unknown = [name for name in names if name not in periods]
    if unknown:
        raise ValueError(f"decision periods are not defined: {unknown}")
    for name, (start, end) in periods.items():
        mask = (interval_dates >= pd.Timestamp(start)) & (interval_dates < pd.Timestamp(end))
        masks[name] = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
        if name in names:
            union |= masks[name]
    masks[DECISION_PERIOD] = union
    return masks


def _turnover(path: positions.Path, mask: np.ndarray) -> float:
    buys = (path.buy >= 0).sum(axis=1)[mask]
    k = path.buy.shape[1]
    return float(buys[1:].mean() / k) if buys.size > 1 else float("nan")


def evaluate_path(
    path: positions.Path,
    inputs: WalkForwardInputs,
    benchmark: np.ndarray,
    *,
    rates: dict[Scenario, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    free = np.zeros_like(next(iter(rates.values()))[0])
    gross = positions.replay(path, free, free)
    gross_hedge = gross.daily - positions.hedge_returns(path, gross, inputs.move)
    rows = []
    for scenario, (buy_rate, sell_rate) in rates.items():
        result = positions.replay(path, buy_rate, sell_rate)
        series = {
            "long_only": (result.daily, gross.daily, benchmark),
            "hedged": (
                result.daily - positions.hedge_returns(path, result, inputs.move),
                gross_hedge,
                np.zeros_like(benchmark),
            ),
        }
        for version in config["versions"]:
            net, gross_series, base = series[version]
            for period, mask in masks.items():
                if not mask.any():
                    continue
                rows.append(
                    {
                        "period": period,
                        "version": version,
                        "discount": scenario.discount,
                        "slippage_multiplier": scenario.slippage_multiplier,
                        **positions.summarize(
                            net[mask],
                            gross_series[mask],
                            base[mask],
                            turnover=_turnover(path, mask),
                            annualization_days=config["annualization_days"],
                            bootstrap=config["bootstrap"],
                        ),
                        "turnover": _turnover(path, mask),
                        "holding_mean": float(path.holding_periods.mean())
                        if path.holding_periods.size
                        else float("nan"),
                        "blocked_buys": path.blocked_buys,
                        "blocked_sells": path.blocked_sells,
                        "missing_exits": path.missing_exits,
                    }
                )
    return rows


def run_market(
    inputs: WalkForwardInputs, market: str, config: dict[str, Any], *, label: str
) -> tuple[list[dict[str, Any]], dict[int, tuple[positions.Path, np.ndarray]]]:
    spec = config["markets"][market]
    codes = tuple(MARKETS.index(c) for c in spec["codes"])
    groups = positions.date_groups(inputs.index, codes)
    benchmark = positions.equal_weight(groups, inputs.move)
    interval_dates = inputs.dates[1:]
    masks = period_masks(interval_dates, config["periods"])
    rates = {
        scenario: side_costs(
            config["costs"],
            scenario,
            inputs.index.market_codes,
            inputs.index.is_etf,
            same_day=False,
        )
        for scenario in scenarios(config["costs"])
    }
    rows, paths = [], {}
    for h in config["score_horizons"]:
        column = inputs.horizons.index(h)
        ranked = engine.rankings(inputs.scores[:, column], groups)
        path = simulate_dropout(
            ranked,
            k=int(spec["k"]),
            drop=int(spec["drop"]),
            symbols=inputs.index.symbol_codes,
            link=inputs.link,
            move=inputs.move,
            limit=config["price_limit"]["open_gap"],
        )
        paths[h] = (path, benchmark)
        for row in evaluate_path(path, inputs, benchmark, rates=rates, config=config, masks=masks):
            rows.append(
                {
                    "model": label,
                    "market": market,
                    "h": h,
                    "k": spec["k"],
                    "drop": spec["drop"],
                    **row,
                }
            )
    return rows, paths


def _primary_mask(grid: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    primary = primary_scenario(config["costs"])
    return (grid["discount"] == primary.discount) & (
        grid["slippage_multiplier"] == primary.slippage_multiplier
    )


def select_horizon(grid: pd.DataFrame, config: dict[str, Any], label: str) -> dict[str, Any]:
    rule = config["selection"]
    rows = grid[
        (grid["model"] == label)
        & (grid["market"] == rule["market"])
        & (grid["period"] == rule["period"])
        & (grid["version"] == rule["version"])
        & _primary_mask(grid, config)
    ]
    rows = rows[np.isfinite(rows[rule["metric"]])]
    if rows.empty:
        return {"h": None, "reason": "no finite selection metric"}
    best = rows.sort_values([rule["metric"], "h"], ascending=[False, True]).iloc[0]
    return {
        "h": int(best["h"]),
        "metric": rule["metric"],
        "value": float(best[rule["metric"]]),
        "market": rule["market"],
        "period": rule["period"],
    }


def decide(
    grid: pd.DataFrame,
    config: dict[str, Any],
    label: str,
    h: int | None,
    baseline_label: str | None,
) -> dict[str, Any]:
    rule = config["decision"]
    if h is None:
        return {"hypothesis": rule["hypothesis"], "passed": None, "reason": "no horizon"}

    def pick(model: str) -> pd.Series | None:
        rows = grid[
            (grid["model"] == model)
            & (grid["market"] == rule["market"])
            & (grid["period"] == rule["period"])
            & (grid["version"] == rule["version"])
            & (grid["h"] == h)
            & _primary_mask(grid, config)
        ]
        return None if rows.empty else rows.iloc[0]

    chosen = pick(label)
    if chosen is None:
        return {"hypothesis": rule["hypothesis"], "passed": None, "reason": "no decision row"}
    lower = float(chosen["net_excess_annual_low"])
    passed = bool(np.isfinite(lower) and lower > rule["lower_bound_above"])
    verdict = {
        "hypothesis": rule["hypothesis"],
        "market": rule["market"],
        "period": rule["period"],
        "h": h,
        "net_excess_annual": float(chosen["net_excess_annual"]),
        "net_excess_annual_low": lower,
        "net_excess_annual_high": float(chosen["net_excess_annual_high"]),
        "net_excess_ir": float(chosen["net_excess_ir"]),
        "lower_bound_positive": passed,
    }
    if baseline_label is not None:
        base = pick(baseline_label)
        base_ir = None if base is None else float(base["net_excess_ir"])
        verdict["baseline_net_excess_ir"] = base_ir
        verdict["ir_not_below_baseline"] = (
            None
            if base_ir is None or not np.isfinite(base_ir)
            else bool(verdict["net_excess_ir"] >= base_ir)
        )
        passed = passed and bool(verdict["ir_not_below_baseline"])
    verdict["passed"] = passed
    return verdict


def _table_row(row) -> str:
    low, high = (
        runner.percent(row.net_excess_annual_low),
        runner.percent(row.net_excess_annual_high),
    )
    interval = f"[{low}, {high}]"
    cells = [
        row.model,
        row.market,
        row.period,
        runner.percent(row.net_excess_annual),
        interval,
        runner.number(row.net_excess_ir),
        runner.percent(row.turnover),
        runner.percent(row.max_drawdown),
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame) -> str:
    selection, decision = report["selection"], report["decision"]
    chosen_h = selection.get("h")
    primary = report["primary_scenario"]
    base = grid[(grid["version"] == "long_only") & (grid["h"] == chosen_h)]
    base = base[
        (base["discount"] == primary["discount"])
        & (base["slippage_multiplier"] == primary["slippage_multiplier"])
    ]
    rows = [_table_row(row) for row in base.sort_values(["model", "market", "period"]).itertuples()]
    verdict = json.dumps(runner.clean(decision), ensure_ascii=False)
    where = f"{selection.get('market')} {selection.get('period')}"
    lines = ["# Top-k dropout 回測摘要 / Top-k dropout backtest summary", ""]
    lines += [
        "## 中文",
        "",
        f"選定 horizon：h = {chosen_h}（{where} 的 net excess IR 最高）。",
        "",
        "| 模型 | 市場 | 期間 | 年化淨超額 | 95% 區間 | IR | 換手 | 最大回撤 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"H-B：{'通過' if decision.get('passed') else '未通過'}（{verdict}）",
        "",
        "## English",
        "",
        f"Selected horizon: h = {chosen_h} (highest net excess IR on {where}).",
        "",
        "| Model | Market | Period | Net excess (annual) | 95% CI | IR | Turnover | Max drawdown |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"H-B: {'passed' if decision.get('passed') else 'not passed'} ({verdict})",
        "",
    ]
    return "\n".join(lines)


def run(
    config_path: Path,
    predictions: Path,
    model: str,
    prices: Path,
    output_root: Path,
    *,
    compare_predictions: Path | None = None,
    compare_model: str | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    price_frame = load_prices(prices)
    targets = [(predictions, model)]
    if compare_predictions is not None:
        targets.append((compare_predictions, compare_model or "baseline"))
    grid_rows, series_frames = [], []
    for path, label in targets:
        inputs = build_inputs(load_predictions(path, label), price_frame)
        for market in config["markets"]:
            rows, paths = run_market(inputs, market, config, label=label)
            grid_rows.extend(rows)
            primary = primary_scenario(config["costs"])
            buy_rate, sell_rate = side_costs(
                config["costs"],
                primary,
                inputs.index.market_codes,
                inputs.index.is_etf,
                same_day=False,
            )
            for h, (sim_path, benchmark) in paths.items():
                net = positions.replay(sim_path, buy_rate, sell_rate)
                series_frames.append(
                    pd.DataFrame(
                        {
                            "model": label,
                            "market": market,
                            "h": h,
                            "open_date": list(inputs.dates[1:].strftime("%Y-%m-%d")),
                            "net_return": net.daily,
                            "equal_weight_return": benchmark,
                        }
                    )
                )
    grid = pd.DataFrame(grid_rows)
    selection = select_horizon(grid, config, model)
    decision = decide(
        grid,
        config,
        model,
        selection.get("h"),
        None if compare_predictions is None else targets[1][1],
    )
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(series_frames, ignore_index=True).to_csv(
        output_root / "daily_series.csv", index=False
    )
    primary = primary_scenario(config["costs"])
    report = {
        "config": str(config_path),
        "predictions": str(predictions),
        "model": model,
        "compare_predictions": None if compare_predictions is None else str(compare_predictions),
        "compare_model": None if compare_predictions is None else targets[1][1],
        "primary_scenario": {
            "discount": primary.discount,
            "slippage_multiplier": primary.slippage_multiplier,
        },
        "selection": selection,
        "decision": decision,
        "rows": len(grid),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(summary_markdown(report, grid), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name inside the predictions file.")
    parser.add_argument("--prices", type=Path, required=True, help="Raw market.parquet.")
    parser.add_argument("--compare-predictions", type=Path)
    parser.add_argument("--compare-model")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if (args.compare_predictions is None) != (args.compare_model is None):
        raise SystemExit("--compare-predictions and --compare-model go together")
    output_root = args.output_root or runner.default_output_root(DEFAULT_ARTIFACTS, "topk-dropout")
    report = run(
        args.config,
        args.predictions,
        args.model,
        args.prices,
        output_root,
        compare_predictions=args.compare_predictions,
        compare_model=args.compare_model,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "selection": report["selection"],
                "decision": report["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
