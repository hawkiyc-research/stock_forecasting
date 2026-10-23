"""Stock-futures short-side day trade on official TAIFEX prices (protocol 13.15).

After the close of signal date t the h = 1 scores rank the names whose near-month stock
futures are liquid; the bottom q are shorted at the futures open of session t+1 and covered
at that session's daily settlement price. Versions: short-only (decides H-E), long-short
(plus the top q long) and hedged (plus an equal-notional TX long). Capital scenarios turn the
notional into whole contracts (volume participation cap, margin usage); costs are the
futures transaction tax, a per-contract commission and tick slippage. The whole grid is a few
thousand small per-session arrays, so it runs serially on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import runner, stats, taifex
from twbacktest.walkforward_data import load_predictions

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "futures_daytrade.json"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FRACTIONAL = "fractional"
DECISION_PERIOD = "P1+P2"
VERSIONS = ("short_only", "long_short", "hedged")


# --------------------------------------------------------------------------- data


def tick_size(price: float, bands: list[list[Any]]) -> float:
    """Tick of the underlying's price band: [[upper_exclusive, tick], ..., [null, tick]]."""
    for upper, tick in bands:
        if upper is None or price < float(upper):
            return float(tick)
    return float(bands[-1][1])


@dataclass(frozen=True)
class Session:
    """Everything the trade on one session needs, aligned over the eligible names."""

    signal_date: pd.Timestamp
    trade_date: str
    scores: np.ndarray  # h = 1 score of each eligible name (finite)
    open: np.ndarray  # futures open on the trade date
    settlement: np.ndarray
    volume: np.ndarray  # contracts traded on the trade date
    shares: np.ndarray  # shares per contract
    tick: np.ndarray  # tick size of each name's price band
    tx_open: float
    tx_settlement: float
    skipped_no_price: int  # eligible names without a tradable futures open
    small_contract: int  # names priced on the small contract


def build_sessions(
    predictions: pd.DataFrame,
    futures_table: pd.DataFrame,
    eligibility: pd.DataFrame,
    tx_front: pd.DataFrame,
    *,
    horizon_column: str,
    tick_bands: list[list[Any]],
    require_open: bool,
) -> list[Session]:
    """Join scores (signal date t) with the next session's futures prices and the universe.

    Eligibility (listed, liquid) is read on the signal date; prices on the next session of the
    futures calendar. Names whose contract has no open that session are skipped and counted.
    """
    dates = sorted(futures_table["date"].unique())
    following = dict(pairwise(dates))
    liquid = eligibility[eligibility["liquid"]][["date", "symbol"]]
    liquid_keys = set(zip(liquid["date"], liquid["symbol"], strict=True))
    table = futures_table.set_index(["date", "symbol"])
    tx = tx_front.set_index("date")
    frame = predictions[["symbol", "signal_date", horizon_column]].copy()
    frame["date"] = frame["signal_date"].dt.strftime("%Y-%m-%d")
    sessions = []
    for signal_date, group in frame.groupby("signal_date", sort=True):
        day = group["date"].iloc[0]
        trade_date = following.get(day)
        if trade_date is None or trade_date not in tx.index:
            continue
        eligible = group[[(day, s) in liquid_keys for s in group["symbol"]]]
        eligible = eligible[np.isfinite(eligible[horizon_column])]
        if eligible.empty:
            continue
        rows, skipped, small = [], 0, 0
        for symbol, score in zip(eligible["symbol"], eligible[horizon_column], strict=True):
            key = (trade_date, symbol)
            if key not in table.index:
                skipped += 1
                continue
            row = table.loc[key]
            open_, settle = float(row["open"]), float(row["settlement"])
            if (require_open and not (np.isfinite(open_) and open_ > 0)) or not (
                np.isfinite(settle) and settle > 0
            ):
                skipped += 1
                continue
            if not bool(row["is_standard"]):
                small += 1
            rows.append((score, open_, settle, float(row["volume"]), float(row["shares"]),
                         tick_size(open_, tick_bands)))
        if not rows:
            continue
        arrays = np.array(rows, dtype=np.float64)
        sessions.append(Session(
            signal_date=pd.Timestamp(signal_date), trade_date=trade_date,
            scores=arrays[:, 0], open=arrays[:, 1], settlement=arrays[:, 2],
            volume=arrays[:, 3], shares=arrays[:, 4], tick=arrays[:, 5],
            tx_open=float(tx.at[trade_date, "open"]),
            tx_settlement=float(tx.at[trade_date, "settlement"]),
            skipped_no_price=skipped, small_contract=small,
        ))
    return sessions


# --------------------------------------------------------------------------- one session


@dataclass(frozen=True)
class CostScenario:
    commission: float  # NT$ per contract per side
    ticks: float  # slippage per side in ticks


@dataclass(frozen=True)
class DayResult:
    gross: float  # P&L before costs / capital
    net: float
    names: int  # names actually traded
    contracts: float  # contracts traded (short leg)
    skipped_size: int  # names skipped for lack of a whole contract
    skipped_participation: int  # names cut to zero by the participation cap
    margin_usage: float  # initial margin of the open positions / capital
    deployed: float  # notional actually traded / (leverage x capital); 1 = fully deployed


def trade_session(
    session: Session,
    *,
    fraction: float,
    version: str,
    capital: float | None,
    leverage: float,
    cost: CostScenario,
    tax_rate: float,
    participation_cap: float,
    margin_rate: float,
    tx: dict[str, float],
) -> DayResult:
    """P&L of one session for one version, capital scenario and cost scenario."""
    n = session.scores.size
    count = max(1, math.ceil(n * fraction))
    order = np.argsort(session.scores, kind="stable")
    legs = [(order[:count], -1.0)]  # short the lowest scores
    if version == "long_short":
        legs.append((order[::-1][:count], 1.0))
    names_total = sum(len(rows) for rows, _ in legs)
    gross = net = 0.0
    contracts_total = 0.0
    skipped_size = skipped_part = 0
    margin = 0.0
    notional_short = 0.0
    deployed = 0.0
    for rows, sign in legs:
        open_, settle = session.open[rows], session.settlement[rows]
        shares, tick, volume = session.shares[rows], session.tick[rows], session.volume[rows]
        ret = sign * (settle - open_) / open_
        # Per-side cost as a fraction of notional: tax + commission + tick slippage.
        side_cost = tax_rate + cost.commission / (open_ * shares) + cost.ticks * tick / open_
        if capital is None:
            weight = leverage / names_total
            gross += float(np.sum(weight * ret))
            net += float(np.sum(weight * (ret - 2.0 * side_cost)))
            margin += weight * margin_rate * len(rows)
            deployed += weight * len(rows) / leverage
            if sign < 0:
                notional_short += weight * len(rows)
            continue
        per_name = leverage * capital / names_total
        contracts = np.floor(per_name / (open_ * shares))
        skipped_size += int(np.sum(contracts < 1))
        allowed = np.floor(participation_cap * volume)
        capped = np.minimum(contracts, allowed)
        skipped_part += int(np.sum((contracts >= 1) & (capped < 1)))
        contracts = np.where(capped >= 1, capped, 0.0)
        notional = contracts * shares * open_
        pnl_gross = sign * contracts * shares * (settle - open_)
        pnl_net = pnl_gross - 2.0 * notional * side_cost
        gross += float(pnl_gross.sum()) / capital
        net += float(pnl_net.sum()) / capital
        contracts_total += float(contracts.sum())
        margin += float(notional.sum()) * margin_rate / capital
        deployed += float(notional.sum()) / (leverage * capital)
        if sign < 0:
            notional_short += float(notional.sum()) / capital
    if version == "hedged":
        # Long TX with the short leg's notional, open to settlement, with TX costs per side.
        tx_ret = (session.tx_settlement - session.tx_open) / session.tx_open
        tx_notional_per_contract = session.tx_open * tx["multiplier"]
        tx_side = tx["tax_rate"] + (
            tx["commission"] + tx["slippage_points"] * tx["multiplier"]
        ) / tx_notional_per_contract
        if capital is None:
            exposure = notional_short
        else:
            contracts = math.floor(notional_short * capital / tx_notional_per_contract)
            exposure = contracts * tx_notional_per_contract / capital
        gross += exposure * tx_ret
        net += exposure * (tx_ret - 2.0 * tx_side)
        margin += exposure * margin_rate
    traded = names_total - skipped_size - skipped_part if capital is not None else names_total
    return DayResult(gross=gross, net=net, names=int(max(traded, 0)),
                     contracts=contracts_total, skipped_size=skipped_size,
                     skipped_participation=skipped_part, margin_usage=margin,
                     deployed=deployed)


# --------------------------------------------------------------------------- evaluation


def period_mask(dates: pd.Series, period: str, periods: dict[str, list[str]]) -> np.ndarray:
    if period == DECISION_PERIOD:
        return np.any([period_mask(dates, name, periods) for name in periods], axis=0)
    start, end = periods[period]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()


def summarize(daily: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    net, gross = daily["net"].to_numpy(), daily["gross"].to_numpy()
    days = int(config["annualization_days"])
    boot = config["bootstrap"]
    mean, low, high = stats.bootstrap_mean(
        net, block=int(boot["block"]), replicates=int(boot["replicates"]), seed=int(boot["seed"])
    )
    candidates = max(daily["candidates"].sum(), 1)
    eligible = max(daily["eligible"].sum(), 1)
    return {
        "days": len(daily),
        "gross_return_annual": float(np.nanmean(gross)) * days,
        "net_return_annual": mean * days,
        "net_return_annual_low": low * days,
        "net_return_annual_high": high * days,
        "net_cagr": stats.compound_annual(net, days),
        "sharpe": stats.ratio(net, days),
        "max_drawdown": stats.max_drawdown(net),
        "hit_rate": float(np.mean(net > 0)) if net.size else float("nan"),
        "cost_drag_annual": float(np.nanmean(gross - net)) * days,
        "names_mean": float(daily["names"].mean()),
        "contracts_mean": float(daily["contracts"].mean()),
        "skipped_size_share": float(daily["skipped_size"].sum() / candidates),
        "skipped_participation_share": float(daily["skipped_participation"].sum() / candidates),
        "skipped_no_price_share": float(daily["skipped_no_price"].sum() / eligible),
        "small_contract_share": float(daily["small_contract"].sum() / eligible),
        "margin_usage_mean": float(daily["margin_usage"].mean()),
        "deployed_share_mean": float(daily["deployed"].mean()),
    }


def cost_scenarios(config: dict[str, Any]) -> list[CostScenario]:
    costs = config["costs"]
    return [CostScenario(float(c), float(t))
            for c in costs["commission_per_contract_twd"] for t in costs["slippage_ticks"]]


def primary_cost(config: dict[str, Any]) -> CostScenario:
    primary = config["costs"]["primary"]
    return CostScenario(float(primary["commission_per_contract_twd"]),
                        float(primary["slippage_ticks"]))


def capital_scenarios(config: dict[str, Any]) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = [(FRACTIONAL, None)]
    out.extend((str(item["name"]), float(item["twd"])) for item in config["capital"]["scenarios"])
    return out


def evaluate_model(
    sessions: list[Session], config: dict[str, Any], *, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grid rows and the daily series (primary costs) for one model."""
    tx_cfg = config["tx"]
    tx = {"multiplier": float(tx_cfg["multiplier"]), "tax_rate": float(tx_cfg["tax_rate"]),
          "commission": float(tx_cfg["commission_per_contract_twd"]),
          "slippage_points": float(tx_cfg["slippage_points"])}
    primary = primary_cost(config)
    rows, dailies = [], []
    signal_dates = pd.Series([s.signal_date for s in sessions])
    for fraction in config["grid"]["bottom_fractions"]:
        for capital_name, capital in capital_scenarios(config):
            for cost in cost_scenarios(config):
                for version in config["versions"]:
                    results = [
                        trade_session(
                            s, fraction=float(fraction), version=version, capital=capital,
                            leverage=float(config["leverage"]), cost=cost,
                            tax_rate=float(config["costs"]["tax_rate"]),
                            participation_cap=float(config["capital"]["participation_cap"]),
                            margin_rate=float(config["margin"]["initial_rate"]), tx=tx,
                        )
                        for s in sessions
                    ]
                    daily = pd.DataFrame({
                        "signal_date": signal_dates,
                        "trade_date": [s.trade_date for s in sessions],
                        "gross": [r.gross for r in results], "net": [r.net for r in results],
                        "names": [r.names for r in results],
                        "contracts": [r.contracts for r in results],
                        "skipped_size": [r.skipped_size for r in results],
                        "skipped_participation": [r.skipped_participation for r in results],
                        "margin_usage": [r.margin_usage for r in results],
                        "deployed": [r.deployed for r in results],
                        "candidates": [
                            max(1, math.ceil(s.scores.size * float(fraction)))
                            * (2 if version == "long_short" else 1)
                            for s in sessions
                        ],
                        "eligible": [s.scores.size + s.skipped_no_price for s in sessions],
                        "skipped_no_price": [s.skipped_no_price for s in sessions],
                        "small_contract": [s.small_contract for s in sessions],
                    })
                    shared = {"model": label, "fraction": float(fraction),
                              "capital": capital_name,
                              "capital_twd": capital if capital is not None else float("nan"),
                              "commission": cost.commission, "ticks": cost.ticks,
                              "version": version}
                    for period in (*config["periods"], DECISION_PERIOD):
                        part = daily[period_mask(daily["signal_date"], period, config["periods"])]
                        if part.empty:
                            continue
                        rows.append({**shared, "period": period, **summarize(part, config)})
                    if cost == primary:
                        dailies.append(daily.assign(**shared))
    return pd.DataFrame(rows), pd.concat(dailies, ignore_index=True)


def select_fraction(grid: pd.DataFrame, config: dict[str, Any], label: str) -> dict[str, Any]:
    rule, primary = config["selection"], primary_cost(config)
    rows = grid[(grid["model"] == label) & (grid["period"] == rule["period"])
                & (grid["version"] == rule["version"]) & (grid["capital"] == FRACTIONAL)
                & (grid["commission"] == primary.commission) & (grid["ticks"] == primary.ticks)]
    rows = rows[np.isfinite(rows[rule["metric"]])]
    if rows.empty:
        return {"fraction": None, "reason": "no finite selection metric"}
    best = rows.sort_values([rule["metric"], "fraction"], ascending=[False, True]).iloc[0]
    return {"fraction": float(best["fraction"]), "metric": rule["metric"],
            "value": float(best[rule["metric"]]), "period": rule["period"]}


def decide(grid: pd.DataFrame, config: dict[str, Any], label: str, chosen: dict[str, Any]):
    rule, primary = config["decision"], primary_cost(config)
    if chosen.get("fraction") is None:
        return {"hypothesis": rule["hypothesis"], "passed": None, "reason": "no fraction"}

    def pick(period: str) -> pd.Series | None:
        rows = grid[(grid["model"] == label) & (grid["period"] == period)
                    & (grid["version"] == rule["version"]) & (grid["capital"] == FRACTIONAL)
                    & (grid["fraction"] == chosen["fraction"])
                    & (grid["commission"] == primary.commission) & (grid["ticks"] == primary.ticks)]
        return None if rows.empty else rows.iloc[0]

    row, confirm = pick(rule["period"]), pick(rule["require_positive_period"])
    if row is None:
        return {"hypothesis": rule["hypothesis"], "passed": None, "reason": "no decision row"}
    lower = float(row["net_return_annual_low"])
    lower_ok = bool(np.isfinite(lower) and lower > float(rule["lower_bound_above"]))
    confirm_ok = bool(confirm is not None and float(confirm["net_return_annual"]) > 0)
    return {
        "hypothesis": rule["hypothesis"], "period": rule["period"], "fraction": chosen["fraction"],
        "gross_return_annual": float(row["gross_return_annual"]),
        "net_return_annual": float(row["net_return_annual"]),
        "net_return_annual_low": lower,
        "net_return_annual_high": float(row["net_return_annual_high"]),
        "net_cagr": float(row["net_cagr"]), "sharpe": float(row["sharpe"]),
        "max_drawdown": float(row["max_drawdown"]), "hit_rate": float(row["hit_rate"]),
        "cost_drag_annual": float(row["cost_drag_annual"]),
        "names_mean": float(row["names_mean"]),
        "lower_bound_positive": lower_ok,
        "confirmation_period": rule["require_positive_period"],
        "confirmation_net_return_annual": (
            float(confirm["net_return_annual"]) if confirm is not None else None),
        "confirmation_positive": confirm_ok,
        "passed": lower_ok and confirm_ok,
    }


def forward_candidate(grid: pd.DataFrame, config: dict[str, Any], decisions: dict) -> str | None:
    primary = primary_cost(config)
    best, best_sharpe = None, -np.inf
    for label, verdict in decisions.items():
        if not verdict.get("passed"):
            continue
        rows = grid[(grid["model"] == label) & (grid["period"] == DECISION_PERIOD)
                    & (grid["version"] == config["decision"]["version"])
                    & (grid["capital"] == FRACTIONAL) & (grid["fraction"] == verdict["fraction"])
                    & (grid["commission"] == primary.commission) & (grid["ticks"] == primary.ticks)]
        if not rows.empty and float(rows.iloc[0]["sharpe"]) > best_sharpe:
            best, best_sharpe = label, float(rows.iloc[0]["sharpe"])
    return best


# --------------------------------------------------------------------------- timing noise


def timing_noise(sessions: list[Session], futures_table: pd.DataFrame, prices: pd.DataFrame):
    """Std of log differences on traded sessions: futures open vs stock open, settlement vs
    stock close."""
    stock = prices[prices["asset_type"] != "index"][["symbol", "date", "raw_open", "raw_close"]]
    stock = stock.assign(date=stock["date"].dt.strftime("%Y-%m-%d"))
    merged = futures_table.merge(stock, on=["date", "symbol"], how="inner")
    merged = merged[(merged["open"] > 0) & (merged["raw_open"] > 0) & (merged["settlement"] > 0)
                    & (merged["raw_close"] > 0)]
    trade_dates = {s.trade_date for s in sessions}
    merged = merged[merged["date"].isin(trade_dates)]
    if merged.empty:
        return {"rows": 0}
    open_diff = np.log(merged["open"] / merged["raw_open"])
    close_diff = np.log(merged["settlement"] / merged["raw_close"])
    return {"rows": len(merged),
            "open_vs_stock_open_std": float(open_diff.std()),
            "open_vs_stock_open_mean": float(open_diff.mean()),
            "settlement_vs_stock_close_std": float(close_diff.std()),
            "settlement_vs_stock_close_mean": float(close_diff.mean())}


# --------------------------------------------------------------------------- run


def run_frames(
    predictions: dict[str, pd.DataFrame],
    futures_table: pd.DataFrame,
    eligibility: pd.DataFrame,
    tx_front: pd.DataFrame,
    config: dict[str, Any],
    output_root: Path,
    *,
    prices: pd.DataFrame | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    column = f"h{int(config['horizon']):02d}"
    grids, dailies, sessions_by_model = [], [], {}
    for label, frame in predictions.items():
        sessions = build_sessions(
            frame, futures_table, eligibility, tx_front, horizon_column=column,
            tick_bands=config["costs"]["tick_bands"],
            require_open=bool(config["universe"]["require_open"]),
        )
        if not sessions:
            raise ValueError(f"No tradable sessions for {label}")
        sessions_by_model[label] = sessions
        grid, daily = evaluate_model(sessions, config, label=label)
        grids.append(grid)
        dailies.append(daily)
    grid = pd.concat(grids, ignore_index=True)
    selection = {label: select_fraction(grid, config, label) for label in predictions}
    decisions = {label: decide(grid, config, label, selection[label]) for label in predictions}
    candidate = forward_candidate(grid, config, decisions)
    infeasible = {}
    threshold = float(config["capital"]["infeasible_skip_share"])
    for label in predictions:
        chosen = selection[label].get("fraction")
        if chosen is None:
            continue
        rows = grid[(grid["model"] == label) & (grid["period"] == DECISION_PERIOD)
                    & (grid["version"] == "short_only") & (grid["fraction"] == chosen)
                    & (grid["commission"] == primary_cost(config).commission)
                    & (grid["ticks"] == primary_cost(config).ticks)]
        infeasible[label] = sorted(
            rows[rows["skipped_size_share"] + rows["skipped_participation_share"] > threshold]
            ["capital"].tolist())
    output_root.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output_root / "grid.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_csv(output_root / "daily.csv", index=False)
    report = {
        "kind": "stock-futures-short-daytrade",
        "config": str(config.get("_path", "")),
        "selection": selection,
        "decisions": decisions,
        "forward_candidate": candidate,
        "infeasible_capital": infeasible,
        "sessions": {label: len(s) for label, s in sessions_by_model.items()},
        "timing_noise": (
            timing_noise(next(iter(sessions_by_model.values())), futures_table, prices)
            if prices is not None else None),
        "rows": len(grid),
        **(extra or {}),
    }
    (output_root / "report.json").write_text(
        json.dumps(runner.clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(report, grid, config),
                                            encoding="utf-8")
    return report


def _row(row) -> str:
    return (
        f"| {row.model} | {row.version} | {row.capital} | {row.period} | "
        f"{runner.percent(row.gross_return_annual)} | {runner.percent(row.net_return_annual)} | "
        f"[{runner.percent(row.net_return_annual_low)}, "
        f"{runner.percent(row.net_return_annual_high)}] | "
        f"{runner.number(row.sharpe)} | {runner.percent(row.max_drawdown)} | "
        f"{runner.percent(row.hit_rate)} | {row.names_mean:.1f} | {row.contracts_mean:.0f} | "
        f"{runner.percent(row.skipped_size_share + row.skipped_participation_share)} | "
        f"{runner.percent(row.deployed_share_mean)} | {runner.percent(row.margin_usage_mean)} |"
    )


def summary_markdown(report: dict[str, Any], grid: pd.DataFrame, config: dict[str, Any]) -> str:
    primary = primary_cost(config)
    body = []
    for label, chosen in report["selection"].items():
        if chosen.get("fraction") is None:
            continue
        rows = grid[(grid["model"] == label) & (grid["fraction"] == chosen["fraction"])
                    & (grid["commission"] == primary.commission) & (grid["ticks"] == primary.ticks)]
        ordered = rows.sort_values(["version", "capital", "period"])
        body.extend(_row(r) for r in ordered.itertuples())
    header_zh = (
        "| 模型 | 版本 | 資本情境 | 期間 | 年化毛報酬 | 年化淨報酬 | 95% 區間 | Sharpe | 最大回撤 "
        "| 勝率 | 平均檔數 | 平均口數 | 被擋比例 | 名目部署率 | 保證金使用率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    )
    header_en = (
        "| Model | Version | Capital | Period | Gross return (annual) | Net return (annual) "
        "| 95% CI | Sharpe | Max drawdown | Hit rate | Mean names | Mean contracts | Blocked "
        "| Deployed | Margin usage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    )
    sel_zh = "；".join(f"{m}：q = {c.get('fraction')}" for m, c in report["selection"].items())
    dec_zh = "；".join(
        f"{m} {v.get('hypothesis')}：{'通過' if v.get('passed') else '未通過'}"
        for m, v in report["decisions"].items())
    dec_en = "; ".join(
        f"{m} {v.get('hypothesis')}: {'passed' if v.get('passed') else 'not passed'}"
        for m, v in report["decisions"].items())
    return "\n".join([
        "# 股票期貨空方當沖回測摘要 / Stock-futures short day-trade summary", "",
        "## 中文", "",
        f"選定後 q 分位（P1 空方版淨 Sharpe 最高）：{sel_zh}。", "",
        f"判定：{dec_zh}。前瞻候選模型：{report['forward_candidate'] or '無'}。", "",
        *header_zh, *body, "",
        "## English", "",
        f"Selected bottom fraction (highest P1 short-only net Sharpe): {sel_zh}.", "",
        f"Decisions: {dec_en}. Forward candidate: {report['forward_candidate'] or 'none'}.", "",
        *header_en, *body, "",
    ])


def run(
    config_path: Path,
    predictions: list[Path],
    models: list[str],
    prices_path: Path | None,
    output_root: Path,
    *,
    artifacts: Path,
) -> dict[str, Any]:
    if len(predictions) != len(models) or not models:
        raise ValueError("Give one --model per --predictions file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_path"] = str(config_path)
    futures, listing = taifex.load(artifacts / config["taifex_root"])
    futures_table = taifex.stock_futures_sessions(futures, listing)
    eligibility, audit = taifex.eligibility(
        futures, listing, min_volume=float(config["universe"]["min_volume"]),
        lookback=int(config["universe"]["lookback_sessions"]))
    tx_cfg = config["tx"]
    tx_front = taifex.continuous_front_month(
        taifex.load_contract(artifacts / tx_cfg["root"], tx_cfg["contract"]), tx_cfg["contract"])
    frames = {label: load_predictions(path, label)
              for path, label in zip(predictions, models, strict=True)}
    prices = None
    if prices_path is not None:
        from twbacktest.walkforward_data import load_prices
        prices = load_prices(prices_path)
        prices = prices.rename(columns={"raw_open_1": "_unused"})
        # load_prices drops the raw open; rebuild it from the parquet for the timing statistic.
        raw = pd.read_parquet(prices_path, columns=["timestamp", "symbol", "open", "close"])
        raw["date"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        prices = prices.merge(raw[["symbol", "date", "open"]].rename(columns={"open": "raw_open"}),
                              on=["symbol", "date"], how="left")
    return run_frames(frames, futures_table, eligibility, tx_front, config, output_root,
                      prices=prices,
                      extra={"predictions": {m: str(p) for p, m in zip(predictions, models,
                                                                        strict=True)},
                             "eligibility_audit": audit})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--prices", type=Path, help="raw market.parquet (timing-noise statistic)")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output_root = args.output_root or runner.default_output_root(args.artifacts_root,
                                                                 "futures-daytrade")
    report = run(args.config, args.predictions, args.model, args.prices, output_root,
                 artifacts=args.artifacts_root)
    print(json.dumps({"output_root": str(output_root), "selection": report["selection"],
                      "decisions": report["decisions"],
                      "forward_candidate": report["forward_candidate"],
                      "infeasible_capital": report["infeasible_capital"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
