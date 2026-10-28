"""S-A weekly band: rebalance flags, the replacement cap, the suite integration and the
decision module (protocol 13.20)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_tw_strategy import _fixture
from twbacktest import positions
from twbacktest import scenario_suite as ss
from twbacktest import strategy_decision as sd
from twbacktest import weekly_band as wb
from twbacktest.topk_dropout import period_masks


def test_weekly_flags_mark_the_last_session_of_each_iso_week():
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-08",  # Mon, Tue, Thu
                              "2026-01-12", "2026-01-16",  # Mon, Fri
                              "2026-01-21"])  # Wed alone
    flags = wb.weekly_signal_flags(dates)
    assert flags.tolist() == [False, False, True, False, True, True]
    assert wb.weekly_signal_flags(pd.DatetimeIndex([])).size == 0


def test_period_masks_decision_union_excludes_forward_periods():
    dates = pd.DatetimeIndex(["2025-01-02", "2025-07-01", "2026-07-01"])
    periods = {"P1": ["2023-01-02", "2025-06-01"], "P2": ["2025-06-01", "2026-06-01"],
               "P3": ["2026-06-01", "2026-10-23"]}
    masks = period_masks(dates, periods, ["P1", "P2"])
    assert masks["P1+P2"].tolist() == [True, True, False]
    assert masks["P3"].tolist() == [False, False, True]
    assert period_masks(dates, periods)["P1+P2"].all()
    with pytest.raises(ValueError, match="not defined"):
        period_masks(dates, periods, ["P4"])


def _ranked_market(n_dates: int, n_stocks: int):
    rows = np.arange(n_dates * n_stocks)
    symbols = np.tile(np.arange(n_stocks), n_dates)
    link = positions.Links(
        next_row=np.where(rows + n_stocks < rows.size, rows + n_stocks, -1),
        prev_row=np.where(rows - n_stocks >= 0, rows - n_stocks, -1),
    )
    move = positions.Moves(growth=np.zeros(rows.size), fallback=np.zeros(rows.size),
                           gap=np.zeros(rows.size), index_growth=np.zeros(rows.size))
    return rows, symbols, link, move


def test_weekly_band_trades_on_rebalance_dates_only_and_caps_replacements():
    # Six stocks, K = 4, buffer 1 (keep while inside the top 4), cap 0.5 -> two replacements.
    n_dates, n_stocks = 5, 6
    _rows, symbols, link, move = _ranked_market(n_dates, n_stocks)

    def day(j: int, order: list[int]) -> np.ndarray:
        return np.array([j * n_stocks + s for s in order])

    ranked = [
        day(0, [0, 1, 2, 3, 4, 5]),  # rebalance: build 0, 1, 2, 3
        day(1, [4, 5, 0, 1, 2, 3]),  # not a rebalance: nothing happens although 2, 3 fell out
        day(2, [4, 5, 0, 1, 2, 3]),  # rebalance: 2 and 3 fall out -> replaced by 4 and 5
        day(3, [4, 5, 1, 0, 2, 3]),  # not a rebalance
        day(4, [2, 3, 0, 1, 4, 5]),  # rebalance: 4 and 5 fall out, no later interval to trade
    ]
    rebalance = np.array([True, False, True, False, True])
    result = wb.simulate_weekly_band(
        ranked, k=4, buffer=1, turnover_cap=0.5, rebalance=rebalance, symbols=symbols,
        link=link, move=move, limit=0.095,
    )
    path = result.band.path
    assert sorted(symbols[path.buy[0]].tolist()) == [0, 1, 2, 3]
    assert (path.buy[1] < 0).all() and (path.sell_open[1] < 0).all()
    assert sorted(symbols[path.sell_open[2][path.sell_open[2] >= 0]].tolist()) == [2, 3]
    assert sorted(symbols[path.buy[2][path.buy[2] >= 0]].tolist()) == [4, 5]
    assert (path.buy[3] < 0).all()
    assert result.rebalances == 2 and result.replacements == 2 and result.cap_hits == 0

    # With the cap at one replacement, only the worst-ranked name (3) leaves on the rebalance.
    capped = wb.simulate_weekly_band(
        ranked, k=4, buffer=1, turnover_cap=0.25, rebalance=rebalance, symbols=symbols,
        link=link, move=move, limit=0.095,
    )
    sold = capped.band.path.sell_open[2]
    assert symbols[sold[sold >= 0]].tolist() == [3]
    assert capped.cap_hits == 1 and capped.replacements == 1
    bought = capped.band.path.buy[2]
    assert symbols[bought[bought >= 0]].tolist() == [4]


def test_weekly_band_retries_a_limit_locked_sale_on_the_next_session():
    n_dates, n_stocks = 5, 3
    rows, symbols, link, move = _ranked_market(n_dates, n_stocks)
    gap = np.zeros(rows.size)
    gap[2 * n_stocks + 0] = -0.1  # stock 0 opens limit-down on the session after date 2
    move = positions.Moves(growth=move.growth, fallback=move.fallback, gap=gap,
                           index_growth=move.index_growth)
    ranked = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([7, 8, 6]),
              np.array([10, 11, 9]), np.array([13, 14, 12])]
    rebalance = np.array([True, False, True, False, False])
    result = wb.simulate_weekly_band(
        ranked, k=1, buffer=1, turnover_cap=1.0, rebalance=rebalance, symbols=symbols,
        link=link, move=move, limit=0.095,
    )
    path = result.band.path
    assert path.blocked_sells == 1
    assert path.sell_open[2, 0] < 0 and symbols[path.sell_open[3, 0]] == 0


def _p3_config(tmp_path, fixture_dates: pd.DatetimeIndex) -> dict:
    config = json.loads((ss.CONFIG_DIR / "scenario_suite_v1_p3.json").read_text(encoding="utf-8"))
    config["cost_profiles"] = str(ss.CONFIG_DIR / "cost_profiles.json")
    config["markets"]["TWSE"]["min_dollar_volume"] = 1_000_000
    config["markets"]["TPEX"]["min_dollar_volume"] = 1_000_000
    config["multi_day"]["fixed"] = {"F0": {"h": 10, "k": 6, "buffer": 2},
                                    "F2": {"h": 10, "k": 6, "buffer": 2}}
    config["weekly"]["grid"] = {"k": [4, 6], "buffer_multiples": [2]}
    config["benchmarks"]["etf"] = {"name": "etf_0050", "symbol": "1102.TW"}
    config["bootstrap"]["replicates"] = 50
    config.pop("execution")  # no chip data in the fixture
    # The fixture ends in 2026-03; make P3 the last months so every period has rows.
    config["periods"] = {"P1": ["2023-01-02", "2025-06-01"], "P2": ["2025-06-01", "2025-12-01"],
                         "P3": ["2025-12-01", "2026-06-01"]}
    return config


def test_suite_runs_s_a_with_benchmarks_and_the_decision_module_reads_it(tmp_path):
    predictions_path, prices_path, artifacts = _fixture(tmp_path)
    config = _p3_config(tmp_path, None)
    path = tmp_path / "p3.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "suite"
    report = ss.run(path, [predictions_path, predictions_path], ["good", "good"], prices_path,
                    output, artifacts=artifacts, labels=["F0", "F2"])
    grid = pd.read_csv(output / "grid.csv")
    assert set(grid["scenario"]) == {"md_ls", "md_ls_tx", "wk_long"}
    assert set(grid["model"].dropna()) == {"F0", "F2"}
    assert set(grid["period"]) == {"P1", "P2", "P3", "P1+P2"}
    assert report["selection"]["F0"]["model"] == "good"
    weekly_choice = report["selection"]["F0"]["weekly"]
    assert weekly_choice["k"] in (4, 6) and weekly_choice["buffer"] == 2
    assert weekly_choice["turnover_cap"] == 0.5
    # The decision union stays P1 + P2: its day count is the sum of the two, without P3.
    cell = grid[(grid["stage"] == "final") & (grid["scenario"] == "wk_long")
                & (grid["model"] == "F0") & (grid["market"] == "TWSE")
                & (grid["profile"] == "institution") & grid["primary_slippage"]
                & (grid["capital"] == "fractional")]
    days = cell.set_index("period")["days"]
    assert days["P1+P2"] == days["P1"] + days["P2"]
    assert days["P3"] > 0
    # S-A is long only, weekly, within the cap, and carries the comparison columns.
    assert {"universe_ew_annual", "etf_0050_annual", "etf_0050_max_drawdown",
            "weekly_turnover", "rebalances", "cap_hits"} <= set(cell.columns)
    assert (cell["weekly_turnover"] <= 0.5 + 1e-9).all()
    assert (cell["rebalances"] > 100).all()
    # Daily turnover is bounded by the weekly cap spread over the sessions between rebalances
    # (forced exits are refilled outside the cap, hence the allowance).
    bound = 0.5 * cell["rebalances"].iloc[0] / cell["days"].iloc[0] + 0.02
    assert cell["turnover"].iloc[0] <= bound
    assert "short_net_annual" not in cell.dropna(axis=1, how="all").columns
    # The grid stage rows of S-A exist (K selected on TWSE P1) and the daily file has S-A.
    assert ((grid["stage"] == "grid") & (grid["scenario"] == "wk_long")).any()
    daily = pd.read_csv(output / "daily.csv")
    assert {"universe_ew", "etf_0050"} <= set(daily.columns)
    assert (daily["scenario"] == "wk_long").any()
    assert (daily["market"] == "TW").any()

    decision_config = json.loads(sd.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    decision_config["periods"] = config["periods"]
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision_config))
    out = tmp_path / "decision"
    verdict = sd.run(output, decision_path, out)
    assert verdict["selection"]["model"] in {"F0", "F2"}
    assert verdict["selection"]["scenario"] in {"md_ls", "md_ls_tx", "wk_long"}
    assert verdict["confirmation"]["evaluated"] is True
    assert verdict["confirmation"]["status"] in {"paper_trading_candidate", "research"}
    # Identical runs: the paired increment is exactly zero and claims nothing.
    assert verdict["increment"]
    for item in verdict["increment"]:
        assert item["P1+P2"]["diff_annual"] == pytest.approx(0.0)
        assert item["increment"] is False
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("# 最後選定與 P3 確認")


def test_decision_rules_on_a_synthetic_grid(tmp_path):
    base = {"stage": "final", "primary_slippage": True, "capital": "fractional",
            "profile": "institution", "market": "TW", "score_variant": "v1",
            "leg_matching": "none", "net_excess_annual_low": 0.0,
            "net_excess_annual_high": 0.2, "turnover": 0.03, "cost_drag_annual": 0.05,
            "hit_rate": 0.5, "days": 100, "index_total_return_annual": 0.1}
    rows = []
    for model, scenario, sharpe, p3_net, p3_dd in [
        ("F0", "md_ls", 1.0, 0.05, -0.08), ("F2", "md_ls", 1.4, -0.02, -0.05),
        ("F2", "wk_long", 1.2, 0.10, -0.20),
    ]:
        rows.append({**base, "model": model, "scenario": scenario, "period": "P1+P2",
                     "sharpe": sharpe, "net_return_annual": 0.15, "max_drawdown": -0.10})
        rows.append({**base, "model": model, "scenario": scenario, "period": "P3",
                     "sharpe": 0.5, "net_return_annual": p3_net, "max_drawdown": p3_dd})
    grid = pd.DataFrame(rows)
    config = json.loads(sd.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    chosen = sd.select_cell(grid, config)
    assert (chosen["model"], chosen["scenario"]) == ("F2", "md_ls")
    assert [c["model"] + "/" + c["scenario"] for c in chosen["candidates"]] == [
        "F2/md_ls", "F2/wk_long", "F0/md_ls"]
    verdict = sd.confirm(grid, config, chosen)
    assert verdict["positive_return"] is False and verdict["drawdown_not_deeper"] is True
    assert verdict["passed"] is False and verdict["status"] == "research"
    # A positive P3 with a shallower drawdown passes; a deeper drawdown fails rule (ii).
    passing = sd.confirm(grid, config, {"model": "F0", "scenario": "md_ls"})
    assert passing["passed"] is True and passing["status"] == "paper_trading_candidate"
    deeper = sd.confirm(grid, config, {"model": "F2", "scenario": "wk_long"})
    assert deeper["positive_return"] is True and deeper["drawdown_not_deeper"] is False
    assert deeper["passed"] is False
