"""Supplement 4 of protocol 13.23: the open-fill impact cost and the 15% trailing stop."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.test_borrow_execution import _chip_root, _exec_block
from tests.test_scenario_suite import _config
from tests.test_tw_strategy import _fixture
from tests.test_weekly_band import _ranked_market
from twbacktest import positions
from twbacktest import scenario_suite as ss
from twbacktest import weekly_band as wb
from twbacktest.tw_strategy import simulate_band
from twbacktest.walkforward_data import load_predictions

LIMIT = 0.095


def _two_stock_band(**kwargs):
    """A, B ranked for three intervals; one slot; opens flat unless ``growth`` is given."""
    symbols = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    link = positions.Links(next_row=np.array([2, 3, 4, 5, 6, 7, -1, -1]),
                           prev_row=np.array([-1, -1, 0, 1, 2, 3, 4, 5]))
    growth = kwargs.pop("growth", np.zeros(8))
    gap = kwargs.pop("gap", np.zeros(8))
    move = positions.Moves(growth=growth, fallback=np.zeros(8), gap=gap,
                           index_growth=np.zeros(8))
    ranked = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
    return simulate_band(ranked, k=1, buffer=2, max_buys=1, symbols=symbols, link=link,
                         move=move, limit=LIMIT, **kwargs)


def test_stop_triggers_intraday_and_fills_at_the_discounted_level():
    adverse = np.zeros(8)
    adverse[0] = math.log(0.8)  # A falls 20% below the open on its first session
    discount = np.full(8, 0.01)
    result = _two_stock_band(stop_fraction=0.85, adverse=adverse, favorable=np.zeros(8),
                             stop_discount=discount, stop_overshoot=0.5)
    assert result.path.buy[0, 0] == 0 and result.path.sell_close[0, 0] == 0
    assert result.stop_exits == 1 and result.stop_gap_exits == 0
    assert result.stop_rows is not None and result.stop_rows[0] and not result.stop_rows[1]
    level = 0.85
    fill = level * (1.0 - 0.01 - 0.5 * (level - 0.8) / level)
    assert result.path.growth[0, 0] == pytest.approx(math.log(fill))
    assert result.path.buy[1, 0] >= 0  # the freed slot refills at the next open


def test_stop_gap_exit_sells_at_the_next_open():
    growth = np.zeros(8)
    growth[0] = math.log(0.8)  # the next open is 20% lower, no intraday breach before it
    result = _two_stock_band(growth=growth, stop_fraction=0.85, adverse=np.zeros(8),
                             favorable=np.zeros(8))
    assert result.path.sell_open[1, 0] == 2 and result.stop_gap_exits == 1
    assert result.stop_exits == 0 and result.stop_rows[2]
    assert result.path.buy[1, 0] == 3  # B fills the slot at that open


def test_stop_gap_exit_waits_at_a_limit_down_open():
    growth, gap = np.zeros(8), np.zeros(8)
    growth[0] = math.log(0.8)
    gap[2] = math.log(0.9)  # A opens limit-down on the session that follows: no sale
    result = _two_stock_band(growth=growth, gap=gap, stop_fraction=0.85, adverse=np.zeros(8),
                             favorable=np.zeros(8))
    assert result.path.sell_open[1, 0] == -1 and result.path.blocked_sells == 1
    assert result.path.sell_open[2, 0] == 4 and result.stop_gap_exits == 1


def test_running_high_uses_the_intraday_favorable_move():
    favorable = np.zeros(8)
    favorable[0] = math.log(1.5)  # A traded 50% above its entry on the first session
    result = _two_stock_band(stop_fraction=0.85, adverse=np.zeros(8), favorable=favorable)
    # The next open (flat, value 1.0) is below 0.85 x 1.5: a gap exit at that open.
    assert result.path.sell_open[1, 0] == 2 and result.stop_gap_exits == 1


def test_no_stop_without_a_breach():
    result = _two_stock_band(stop_fraction=0.85, adverse=np.full(8, math.log(0.9)),
                             favorable=np.zeros(8))
    assert result.stop_exits == 0 and result.stop_gap_exits == 0
    assert result.path.buy[0, 0] == 0 and (result.path.sell_close < 0).all()


def test_stop_needs_the_intraday_arrays():
    with pytest.raises(ValueError, match="adverse"):
        _two_stock_band(stop_fraction=0.85)


def test_weekly_band_stop_exits_intraday_on_any_session():
    n_dates, n_stocks = 3, 6
    _rows, symbols, link, move = _ranked_market(n_dates, n_stocks)
    ranked = [np.array([j * n_stocks + s for s in range(n_stocks)]) for j in range(n_dates)]
    adverse = np.zeros(n_dates * n_stocks)
    adverse[0] = math.log(0.7)  # stock 0 breaks 15% on the first session
    result = wb.simulate_weekly_band(
        ranked, k=4, buffer=1, turnover_cap=0.5, rebalance=np.array([True, False, True]),
        symbols=symbols, link=link, move=move, limit=LIMIT, stop_fraction=0.85,
        adverse=adverse, favorable=np.zeros(n_dates * n_stocks), stop_overshoot=0.0)
    band = result.band
    assert band.stop_exits == 1 and band.stop_rows[0]
    assert band.path.sell_close[0, 0] == 0
    assert band.path.growth[0, 0] == pytest.approx(math.log(0.85))


def test_tick_sizes_follow_the_exchange_table():
    prices = np.array([5.0, 10.0, 49.95, 50.0, 99.9, 100.0, 499.5, 500.0, 999.0, 1000.0,
                       np.nan])
    expected = [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 5.0]
    ticks = ss.tick_size(prices)
    assert list(ticks[:-1]) == expected and np.isnan(ticks[-1])


def test_stop_settings_mirror_the_short_leg_and_price_the_half_tick():
    inputs = SimpleNamespace(intraday_high=np.array([math.log(1.1), np.nan]),
                             intraday_low=np.array([math.log(0.9), np.nan]),
                             raw_open_1=np.array([80.0, np.nan]))
    data = SimpleNamespace(inputs=inputs)
    config = {"stop_loss": {"fraction": 0.85, "overshoot": 0.25}}
    long_stop = ss.stop_settings(data, config)
    assert long_stop["adverse"][0] == pytest.approx(math.log(0.9))
    assert long_stop["favorable"][0] == pytest.approx(math.log(1.1))
    assert long_stop["stop_discount"][0] == pytest.approx(0.05 / 80.0)  # half of a 0.1 tick
    short_stop = ss.stop_settings(data, config, short=True)
    assert short_stop["adverse"][0] == pytest.approx(math.log(2.0 - 1.1))
    assert short_stop["favorable"][0] == pytest.approx(math.log(2.0 - 0.9))
    assert ss.stop_settings(data, {}) == {}
    with pytest.raises(ValueError, match="intraday"):
        ss.stop_settings(SimpleNamespace(inputs=SimpleNamespace(intraday_high=None,
                                                                 intraday_low=None)), config)


def test_impact_rates_follow_the_square_root_law():
    inputs = SimpleNamespace(volatility=np.array([0.02, 0.02, 0.02, np.nan]),
                             dollar_volume=np.array([1e8, 1e8, 1e8, 1e8]))
    data = SimpleNamespace(inputs=inputs)
    result = SimpleNamespace(buy_value=np.array([1e6, np.nan, 1e6, 1e6]),
                             sell_value=np.array([np.nan, 1e6, 1e6, 1e6]),
                             stop_rows=np.array([False, False, True, False]))
    config = {"execution": {"open_fill": {"impact_y": 0.7, "auction_share": 0.1}}}
    buy, sell = ss.impact_rates(result, data, config)
    assert buy[0] == pytest.approx(0.7 * 0.02 * math.sqrt(1e6 / (0.1 * 1e8)))
    assert buy[1] == 0.0 and sell[0] == 0.0  # no trade on that side
    assert sell[1] == pytest.approx(buy[0])  # an open sale pays the auction-share impact
    assert sell[2] == pytest.approx(0.7 * 0.02 * math.sqrt(1e6 / 1e8))  # a stop fill: f = 1
    assert buy[3] == 0.0  # no volatility estimate -> no impact charged
    assert ss.impact_rates(result, data, {"execution": {}}) is None
    rates = (np.zeros(4), np.zeros(4))
    assert ss.with_impact(rates, result, data, {})[0] is rates[0]


def test_run_with_fill_and_stop(tmp_path):
    predictions, prices, artifacts = _fixture(tmp_path)
    frame = load_predictions(predictions, "good")
    root = _chip_root(tmp_path, frame)
    cost = json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text(encoding="utf-8"))
    cost["profiles"]["institution"]["capital_scenarios"] = [{"name": "20M", "twd": 2e7}]
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(json.dumps(cost))
    execution = _exec_block(failure_multipliers=[1], recall_probabilities=[0.0])
    execution["open_fill"] = {"impact_y": 0.7, "auction_share": 0.1}
    config = _config(
        tmp_path, cost_profiles=str(cost_path), scenarios=["md_ls", "md_ls_tx"],
        construction={"score_variants": ["v1"], "leg_matching": ["none"]},
        execution=execution, stop_loss={"fraction": 0.97, "overshoot": 0.25, "half_tick": True},
    )
    ss.run(config, [predictions], ["good"], prices, tmp_path / "out", artifacts=artifacts,
           chip_data_root=root)
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    final = grid[(grid["stage"] == "final") & grid["primary_slippage"]
                 & (grid["market"] == "TWSE") & (grid["profile"] == "institution")]
    md = final[final["scenario"] == "md_ls"]
    assert (md["stop_exits"] + md["stop_gap_exits"] > 0).all()
    assert (md["impact_y"] == 0.7).all() and (md["stop_fraction"] == 0.97).all()
    # The impact only bites with capital: the fractional row equals a run without it.
    (tmp_path / "plain").mkdir()
    plain = _config(
        tmp_path / "plain", cost_profiles=str(cost_path), scenarios=["md_ls"],
        construction={"score_variants": ["v1"], "leg_matching": ["none"]},
        execution=_exec_block(failure_multipliers=[1], recall_probabilities=[0.0]),
        stop_loss={"fraction": 0.97, "overshoot": 0.25, "half_tick": True},
    )
    ss.run(plain, [predictions], ["good"], prices, tmp_path / "out-plain", artifacts=artifacts,
           chip_data_root=root)
    grid_plain = pd.read_csv(tmp_path / "out-plain" / "grid.csv")

    def row(frame, capital):
        part = frame[(frame["stage"] == "final") & frame["primary_slippage"]
                     & (frame["market"] == "TWSE") & (frame["profile"] == "institution")
                     & (frame["scenario"] == "md_ls") & (frame["period"] == "P1")
                     & (frame["capital"] == capital)]
        return float(part["net_return_annual"].iloc[0])

    assert row(grid, "fractional") == pytest.approx(row(grid_plain, "fractional"))
    assert row(grid, "20M") < row(grid_plain, "20M")


def test_policy_b_adds_half_tick_delay_and_full_day_impact():
    growth = np.zeros(8)
    result = _two_stock_band(growth=growth)  # A bought at the first open, held throughout
    rows = 8
    inputs = SimpleNamespace(volatility=np.full(rows, 0.02), dollar_volume=np.full(rows, 1e8),
                             raw_open_1=np.full(rows, 80.0))
    intraday = np.zeros(rows)
    intraday[0] = math.log(1.02)  # A rises 2% from open to close on its entry session
    intraday[6] = math.log(0.99)  # and falls 1% on the final session (liquidation at that open)
    data = SimpleNamespace(inputs=inputs, intraday=intraday)
    config = {"execution": {"open_fill": {"policy": "B", "impact_y": 0.7, "auction_share": 0.1,
                                          "half_tick": True, "delay_theta": 0.5}}}
    buy, sell = ss.impact_rates(result, data, config)
    half = 0.05 / 80.0
    assert buy[0] == pytest.approx(half + (1.0 - math.exp(-0.5 * math.log(1.02))))
    assert buy[1] == 0.0  # B was never bought
    assert sell[6] == pytest.approx(half + (1.0 - math.exp(0.5 * math.log(0.99))))
    # Short mirror: the same move is a gain on the short sale and a cost on the cover.
    sbuy, ssell = ss.impact_rates(result, data, config, short=True)
    assert sbuy[0] == pytest.approx(half + (1.0 - math.exp(0.5 * math.log(1.02))))
    assert ssell[6] == pytest.approx(half + (1.0 - math.exp(-0.5 * math.log(0.99))))
    # Impact under policy B uses the full day's volume (f = 1), capital rows only.
    capital = SimpleNamespace(path=result.path, fill_fraction=result.fill_fraction,
                              stop_rows=None, buy_value=np.where(np.arange(rows) == 0, 1e6, np.nan),
                              sell_value=np.full(rows, np.nan))
    cbuy, _ = ss.impact_rates(capital, data, config)
    assert cbuy[0] == pytest.approx(buy[0] + 0.7 * 0.02 * math.sqrt(1e6 / 1e8))
    with pytest.raises(ValueError, match="policy"):
        ss.impact_rates(result, data, {"execution": {"open_fill": {"policy": "C", "impact_y": 1}}})
    tags = ss.execution_tags(config)
    assert tags["fill_policy"] == "B" and tags["delay_theta"] == 0.5


def test_fill_profile_drops_the_flat_slippage_for_capital_rows_only():
    profiles = json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text(encoding="utf-8"))
    profile = next(p for p in ss.cost_profiles(profiles)
                   if p.name == "institution" and ss.is_primary(p, profiles))
    config = {"execution": {"open_fill": {"policy": "A", "impact_y": 0.7, "auction_share": 0.1}}}
    capital = ss.Capital("300M", 3e8, "institution")
    assert ss.fill_profile(profile, config, capital).slippage == (0.0, 0.0)
    assert ss.fill_profile(profile, config, ss.Capital("fractional", None, None)) is profile
    assert ss.fill_profile(profile, {"execution": {}}, capital) is profile
    kept = {"execution": {"open_fill": {"policy": "A", "impact_y": 0.7, "auction_share": 0.1,
                                        "keep_flat_slippage": True}}}
    assert ss.fill_profile(profile, kept, capital) is profile
