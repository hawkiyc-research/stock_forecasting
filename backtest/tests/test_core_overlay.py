"""0050 core plus orthogonalized long-short overlay (protocol 13.17 supplement)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_borrow_execution import _chip_root, _config, _exec_block
from tests.test_tw_strategy import _fixture
from twbacktest import core_overlay as co
from twbacktest import scenario_suite as ss
from twbacktest.walkforward_data import load_predictions

DAYS = 252


def test_buy_and_hold_pays_costs_at_both_ends():
    growth = np.array([0.0, 0.01, 0.0])
    net = co.buy_and_hold(growth, 0.001, 0.002)
    assert net[0] == pytest.approx(-0.001)
    assert net[1] == pytest.approx(0.01)
    assert net[2] == pytest.approx(-0.002)


def _legs(n: int, long: float = 0.0, short: float = 0.0):
    return {"long": np.full(n, long), "short": np.full(n, short)}


def test_variant_a_loan_interest_and_collateral():
    core = np.zeros(10)
    rates = (0.0, 0.0)
    plain = co.variant_a(core, rates, _legs(10), _legs(10), w=0.5, leverage=0.5, loan_rate=0.025,
                         ltv_max=0.6, annualization_days=DAYS)
    assert plain.info["loan"] == 0.0 and np.allclose(plain.net, 0.0)
    assert plain.info["collateral_shortfall_days"] == 0
    levered = co.variant_a(core, rates, _legs(10, long=0.001), _legs(10, long=0.002), w=0.5,
                           leverage=1.0, loan_rate=0.025, ltv_max=0.6, annualization_days=DAYS)
    assert levered.info["loan"] == pytest.approx(0.5)
    assert levered.net[3] == pytest.approx(0.001 - 0.5 * 0.025 / DAYS)
    assert levered.gross[3] == pytest.approx(0.002)
    assert levered.info["loan_interest_annual"] == pytest.approx(0.0125)
    # 0050 halves: the 0.5 loan exceeds 60% of the marked 0.25 holding.
    crash = np.array([-0.5] + [0.0] * 9)
    short_of_collateral = co.variant_a(crash, rates, _legs(10), _legs(10), w=0.5, leverage=1.0,
                                       loan_rate=0.025, ltv_max=0.6, annualization_days=DAYS)
    assert short_of_collateral.info["collateral_shortfall_days"] == 10
    assert short_of_collateral.info["ltv_peak"] == pytest.approx(2.0)


def test_variant_b_tx_fees_rolls_and_margin_calls():
    growth = np.full(6, 0.01)
    roll = np.array([False, False, True, False, False, False])
    rate = np.full(6, 0.001)
    series = co.variant_b(growth, roll, rate, _legs(6, long=0.002), _legs(6, long=0.003),
                          margin_fraction=0.15, maintenance_fraction=0.10, overlay_fraction=0.85,
                          tx_notional=1.0, annualization_days=DAYS)
    assert series.net[0] == pytest.approx(0.01 - 0.001 + 0.85 * 0.002)  # entry side
    assert series.net[2] == pytest.approx(0.01 - 0.002 + 0.85 * 0.002)  # roll: two sides
    assert series.net[3] == pytest.approx(0.01 + 0.85 * 0.002)
    assert series.net[5] == pytest.approx(0.01 - 0.001 + 0.85 * 0.002)  # exit side
    assert series.gross[3] == pytest.approx(0.01 + 0.85 * 0.003)
    assert series.info["margin_calls"] == 0
    crash = co.variant_b(np.full(6, -0.03), roll, rate, _legs(6), _legs(6), margin_fraction=0.15,
                         maintenance_fraction=0.10, overlay_fraction=0.85, tx_notional=1.0,
                         annualization_days=DAYS)
    assert crash.info["margin_calls"] == 3  # -3% a day: a call every second interval


def test_combine_books_averages_legs_and_info():
    a = ss.Book(net=np.array([0.02]), gross=np.array([0.03]), info={"turnover": 0.02},
                legs={"long": np.array([0.03]), "short": np.array([-0.01])},
                gross_legs={"long": np.array([0.04]), "short": np.array([-0.01])})
    b = ss.Book(net=np.array([0.0]), gross=np.array([0.01]), info={"turnover": 0.04},
                legs={"long": np.array([0.01]), "short": np.array([-0.01])},
                gross_legs={"long": np.array([0.02]), "short": np.array([-0.01])})
    combined = co.combine_books({"TWSE": a, "TPEX": b}, {"TWSE": 0.5, "TPEX": 0.5})
    assert combined.legs["long"][0] == pytest.approx(0.02)
    assert combined.gross_legs["long"][0] == pytest.approx(0.03)
    assert combined.net[0] == pytest.approx(0.01) and combined.info["turnover"] == 0.03


def _with_etf(prices_path, symbol="0050.TW"):
    """Append an ETF that tracks the TAIEX index rows of the fixture (plus a 1% premium drift)."""
    prices = pd.read_parquet(prices_path)
    index = prices[prices["symbol"] == "TAIEX.TW"].copy()
    level = index["open"].to_numpy() / 100.0
    etf = index.assign(symbol=symbol, asset_type="etf", open=level, close=level * 1.001,
                       adjusted_close=level * 1.001 * np.linspace(1.0, 1.01, len(index)),
                       volume=5e6)
    pd.concat([prices, etf], ignore_index=True).to_parquet(prices_path, index=False)


def test_run_end_to_end(tmp_path):
    predictions, prices, artifacts = _fixture(tmp_path)
    _with_etf(prices)
    frame = load_predictions(predictions, "good")
    root = _chip_root(tmp_path, frame, ineligible={"1103.TW"})
    cost = json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text(encoding="utf-8"))
    cost["profiles"]["institution"]["capital_scenarios"] = []
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(json.dumps(cost))
    suite_path = _config(
        tmp_path, cost_profiles=str(cost_path), scenarios=["md_ls"],
        construction={"score_variants": ["v1"], "leg_matching": ["none"]},
        execution=_exec_block(failure_multipliers=[1], recall_probabilities=[0.0]),
    )
    suite = json.loads(suite_path.read_text())
    suite["multi_day"]["fixed"] = {"good": {"h": 10, "k": 5, "buffer": 2}}
    suite_path.write_text(json.dumps(suite))
    config = json.loads(co.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["suite_config"] = str(suite_path)
    config["bootstrap"]["replicates"] = 50
    config["capital_scenarios"] = {"retail": [{"name": "10M", "twd": 1e7, "odd_lots": True}],
                                   "institution": [{"name": "30M", "twd": 3e7}]}
    config_path = tmp_path / "core_overlay.json"
    config_path.write_text(json.dumps(config))
    report = co.run(config_path, [predictions], ["good"], prices, tmp_path / "out",
                    artifacts=artifacts, chip_data_root=root)
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    assert set(grid["variant"]) == {"A", "B", "benchmark"}
    assert set(grid["market"]) == {"TWSE", "TPEX", "TW"}
    bench = grid[grid["variant"] == "benchmark"]
    assert np.allclose(bench["net_excess_annual"], 0.0)
    assert np.allclose(bench["net_return_annual"], bench["benchmark_net_annual"])
    frac_a = grid[(grid["variant"] == "A") & (grid["capital"] == "fractional")
                  & (grid["profile"] == "institution") & (grid["market"] == "TW")
                  & (grid["period"] == "P1")]
    assert len(frac_a) == 2 * 2 * 2  # w x L x loan rate
    retail_a = grid[(grid["variant"] == "A") & (grid["profile"] == "retail")]
    assert set(retail_a["leverage"]) == {0.5}
    capital = grid[grid["capital"] != "fractional"]
    assert set(capital["capital"]) == {"10M", "30M"} and (capital["market"] == "TW").all()
    assert (capital["variant"] == "A").all() or set(capital["variant"]) <= {"A", "benchmark"}
    sel = report["selection"]["good"]["TW"]["institution"]
    assert sel["leverage"] in (0.5, 1.0) and sel["loan_rate"] == 0.025
    decisions = report["decisions"]["good"]["TW"]
    assert set(decisions) == {"institution", "retail"} and set(decisions["retail"]) == {"A", "B"}
    assert isinstance(decisions["institution"]["A"]["passes"], bool)
    # The leg books carry the execution layer: some short candidates were off the list.
    twse_a = grid[(grid["variant"] == "A") & (grid["market"] == "TWSE")]
    assert (twse_a["ineligible_share"] > 0).all()
    daily = pd.read_csv(tmp_path / "out" / "daily.csv")
    assert set(daily["series"]) >= {"B", "benchmark"} and (daily["market"] == "TW").all()
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "H-G" in summary and "0050" in summary
