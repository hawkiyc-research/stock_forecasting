"""Six-scenario suite: short moves, borrow fee, day-trade costs, rolling beta, end to end."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.test_tw_strategy import _fixture
from twbacktest import positions
from twbacktest import scenario_suite as ss
from twbacktest.tw_strategy import FuturesSeries, universe_groups
from twbacktest.walkforward_data import build_inputs, load_predictions, load_prices


def _profiles() -> dict:
    return json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text(encoding="utf-8"))


def _config(tmp_path, **overrides):
    config = json.loads(ss.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["cost_profiles"] = str(ss.CONFIG_DIR / "cost_profiles.json")
    config["markets"]["TWSE"]["min_dollar_volume"] = 1_000_000
    config["markets"]["TPEX"]["min_dollar_volume"] = 1_000_000
    config["multi_day"]["grid"] = {"score_horizons": [5, 10], "k": [5], "buffer_multiples": [2]}
    config["day_trade"]["grid"] = {"fractions": [0.2]}
    config["bootstrap"]["replicates"] = 50
    config.update(overrides)
    path = tmp_path / "scenario_suite.json"
    path.write_text(json.dumps(config))
    return path


def test_cost_profiles_rates():
    profiles = _profiles()
    institution = ss._primary_profile(profiles, "institution")
    codes = np.array([0, 1], dtype=np.int8)
    buy, sell = institution.long_rates(codes)
    assert buy == pytest.approx([0.0008, 0.0008])
    assert sell == pytest.approx([0.0038, 0.0038])
    entry, exit_ = institution.short_rates(codes)
    assert entry == pytest.approx([0.0038, 0.0038]) and exit_ == pytest.approx([0.0008, 0.0008])
    assert institution.day_trade_round_trip(codes) == pytest.approx([0.0031, 0.0031])
    assert institution.borrow_fee_daily == pytest.approx(0.02 / 252)
    retail = ss._primary_profile(profiles, "retail")
    entry, _ = retail.short_rates(codes)
    assert entry == pytest.approx([0.000855 + 0.003 + 0.001 + 0.0008,
                                   0.000855 + 0.003 + 0.002 + 0.0008])
    assert retail.borrow_fee_daily == 0.0
    multipliers = {p.slippage_multiplier for p in ss.cost_profiles(profiles)}
    assert multipliers == {0.0, 1.0, 2.0}
    assert [c.name for c in ss.capital_scenarios(profiles)] == ["fractional", "300M", "1B"]


def test_short_moves_flip_direction_and_reverse_gaps():
    move = positions.Moves(growth=np.array([0.01, -0.02, np.nan, 1.0]),
                           fallback=np.array([0.0, 0.0, 0.0, 0.0]),
                           gap=np.array([0.09, -0.09, 0.0, np.nan]),
                           index_growth=np.zeros(4))
    short = ss.short_moves(move)
    assert short.growth[0] == pytest.approx(math.log(2 - math.exp(0.01)))
    assert short.growth[1] == pytest.approx(math.log(2 - math.exp(-0.02)))
    assert np.isnan(short.growth[2])
    assert short.growth[3] == pytest.approx(math.log(ss.SHORT_FLOOR))  # a doubling is floored
    assert short.gap[0] == -0.09 and short.gap[1] == 0.09 and np.isnan(short.gap[3])


def test_borrow_fee_charges_held_slots_only():
    held = np.array([[0, -1], [1, 2]])
    path = positions.Path(buy=held, sell_open=-np.ones_like(held), sell_close=-np.ones_like(held),
                          held=held, growth=np.zeros((2, 2)), final=np.array([1, 2]),
                          holding_periods=np.array([1]), blocked_buys=0, blocked_sells=0,
                          missing_exits=0)
    charged = ss.with_borrow_fee(path, 0.02 / 252)
    assert charged.growth[0, 0] == pytest.approx(math.log1p(-0.02 / 252))
    assert charged.growth[0, 1] == 0.0
    assert ss.with_borrow_fee(path, 0.0) is path


def test_rolling_beta_is_ex_ante():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 200)
    returns = 2.0 * market + rng.normal(0, 0.001, 200)
    beta = ss.rolling_beta(returns, market, window=60, min_obs=30)
    assert np.all(beta[:30] == 0.0)
    assert beta[100] == pytest.approx(2.0, abs=0.1)
    shocked = returns.copy()
    shocked[150] = 5.0  # the shock may only affect later betas
    beta2 = ss.rolling_beta(shocked, market, window=60, min_obs=30)
    assert beta2[150] == beta[150] and beta2[151] != beta[151]


def _tiny_market(tmp_path, *, limit_up_first: bool = False):
    """Four TWSE stocks over four sessions; signals on the first session only matter."""
    sessions = pd.bdate_range("2025-01-06", periods=4)
    opens = {"A": [100, 100, 100, 100], "B": [100, 100, 100, 100],
             "C": [100, 100, 100, 100], "D": [100, 100, 100, 100]}
    closes = {"A": [100, 102, 100, 100], "B": [100, 98, 100, 100],
              "C": [100, 100, 100, 100], "D": [100, 100, 100, 100]}
    if limit_up_first:
        opens["A"][1] = 110  # A opens limit-up on the trade session
        closes["A"][1] = 112
    rows = []
    for symbol in opens:
        for day, date in enumerate(sessions):
            rows.append((date, f"{symbol}.TW", "stock", float(opens[symbol][day]),
                         float(closes[symbol][day]), float(closes[symbol][day]), 1e6, "TWSE"))
    for date in sessions:
        rows.append((date, "TAIEX.TW", "index", 100.0, 100.0, 100.0, 0.0, "TWSE"))
    prices = pd.DataFrame(rows, columns=["timestamp", "symbol", "asset_type", "open", "close",
                                         "adjusted_close", "volume", "market"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize("UTC")
    scores = {"A": 4.0, "B": 1.0, "C": 2.0, "D": 3.0}
    predictions = pd.DataFrame([
        {"model": "m", "symbol": f"{s}.TW", "market": "TWSE", "signal_date": date,
         **{f"h{h:02d}": scores[s] for h in range(1, 15)}}
        for s in scores for date in sessions[:2]
    ])
    tmp_path.mkdir(parents=True, exist_ok=True)
    prices_path, predictions_path = tmp_path / "p.parquet", tmp_path / "s.parquet"
    prices.to_parquet(prices_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    frame = load_predictions(predictions_path, "m")
    price_frame = load_prices(prices_path, liquidity_lookback=2)
    inputs = build_inputs(frame, price_frame)
    intraday = ss.intraday_returns(frame, price_frame)
    mask = np.ones(inputs.index.count, dtype=bool)
    intervals = len(inputs.dates) - 1
    futures = FuturesSeries(growth=np.zeros(intervals), roll=np.zeros(intervals, dtype=bool),
                            rate=np.zeros(intervals), index_growth=np.zeros(intervals), missing=0)
    return ss.MarketData(
        market="TWSE", inputs=inputs, mask=mask, groups=universe_groups(inputs, mask),
        index_interval=np.zeros(intervals), index_intraday=np.zeros(intervals),
        index_total_return=np.zeros(intervals), futures=futures,
        tx_intraday=np.zeros(intervals), tx_notional_per_contract=np.full(intervals, 4e6),
        intraday=intraday, lot_cost=np.full(inputs.index.count, 1e5),
        masks={"P1": np.ones(intervals, dtype=bool)},
    )


def test_day_trade_book_pnl_costs_and_limit_skip(tmp_path):
    config = json.loads(ss.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    institution = ss._primary_profile(_profiles(), "institution")
    data = _tiny_market(tmp_path)
    book = ss.day_trade_book(data, config, institution, ss.Capital("fractional", None, None),
                             fraction=0.25, participation_cap=0.05)
    # Long A (+2%) and short B (-2% -> +2%), each on the full capital: gross 4%.
    assert book.gross[0] == pytest.approx(0.04)
    assert book.net[0] == pytest.approx(0.04 - 2 * 0.0031)
    assert book.info["names_mean"] > 0 and book.info["turnover"] == 1.0
    # A opens limit-up: the long leg skips it and takes D (flat) instead.
    blocked = ss.day_trade_book(_tiny_market(tmp_path / "b", limit_up_first=True), config,
                                institution, ss.Capital("fractional", None, None),
                                fraction=0.25, participation_cap=0.05)
    assert blocked.gross[0] == pytest.approx(0.02)
    assert blocked.info["limit_blocked_share"] > 0
    # Capacity: a 1B slot may only use 5% of a 100M daily dollar volume (0.5% of capital).
    capped = ss.day_trade_book(data, config, institution, ss.Capital("1B", 1e9, "institution"),
                               fraction=0.25, participation_cap=0.05)
    assert capped.gross[0] == pytest.approx(2 * 0.005 * 0.02)
    assert capped.info["liquidity_capped_share"] > 0 and capped.info["deployed_share"] < 0.01
    assert book.info["deployed_share"] == pytest.approx(1.0)


def test_tx_overlay_hedges_a_beta_two_book(tmp_path):
    config = json.loads(ss.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    institution = ss._primary_profile(_profiles(), "institution")
    data = _tiny_market(tmp_path)
    rng = np.random.default_rng(1)
    n = 120
    market = rng.normal(0, 0.01, n)
    book = ss.Book(net=2.0 * market, gross=2.0 * market, info={"turnover": 0.0})
    futures = FuturesSeries(growth=market, roll=np.zeros(n, dtype=bool), rate=np.zeros(n),
                            index_growth=market, missing=0)
    big = ss.MarketData(**{**data.__dict__, "index_interval": market, "futures": futures,
                           "tx_intraday": market, "tx_notional_per_contract": np.full(n, 4e6)})
    hedged = ss.tx_overlay(book, big, institution, config, intraday=False)
    assert hedged.info["beta_abs_mean"] > 1.0
    # After the warm-up the hedge removes most of the variance.
    assert hedged.gross[80:].std() < 0.25 * book.gross[80:].std()
    assert (hedged.net <= hedged.gross + 1e-12).all()  # fees only reduce the net


def test_end_to_end_run_writes_outputs_and_selects(tmp_path):
    predictions_path, prices_path, artifacts = _fixture(tmp_path)
    config = _config(tmp_path)
    output = tmp_path / "out"
    report = ss.run(config, [predictions_path], ["good"], prices_path, output, artifacts=artifacts)
    grid = pd.read_csv(output / "grid.csv")
    assert (output / "summary.md").is_file() and (output / "daily.csv").is_file()
    assert set(grid["scenario"]) == set(ss.SCENARIOS)
    assert set(grid["profile"]) == {"retail", "institution"}
    assert set(grid["market"]) == {"TWSE", "TPEX", "TW"}
    assert {"fractional", "300M", "1B"} <= set(grid["capital"])
    chosen = report["selection"]["good"]
    assert chosen["multi_day"]["h"] in (5, 10) and chosen["multi_day"]["k"] == 5
    assert chosen["day_trade"]["fraction"] == 0.2
    # Scores that know the next open-to-open move make S1 attractive under institution costs.
    verdict = report["attractiveness"]["good"]["md_ls"]["TWSE"]
    assert verdict["attractive"] is True
    final = grid[(grid["stage"] == "final") & (grid["capital"] == "fractional")
                 & grid["primary_slippage"] & (grid["period"] == "P1+P2")]
    s1 = final[(final["scenario"] == "md_ls") & (final["market"] == "TWSE")]
    retail = s1[s1["profile"] == "retail"].iloc[0]
    institution = s1[s1["profile"] == "institution"].iloc[0]
    assert institution["net_return_annual"] > retail["net_return_annual"]
    assert institution["cost_drag_annual"] < retail["cost_drag_annual"]
    assert {"long_net_annual", "short_net_annual", "beta_abs_mean", "index_total_return_annual",
            "hit_rate", "holding_mean", "deployed_share", "capped_share"} <= set(grid.columns)
    # Cap-and-scale (the cost profile default): the 300M book is capped, not emptied.
    big = grid[(grid["stage"] == "final") & (grid["scenario"] == "md_ls")
               & (grid["capital"] == "300M") & (grid["market"] == "TWSE")
               & grid["primary_slippage"]]
    assert (big["cap_mode"] == "scale").all()
    assert (big["deployed_share"] < 0.5).all() and (big["capped_share"] > 0).all()


def test_purified_band_construction_runs_both_variants_and_matchings(tmp_path):
    predictions_path, prices_path, artifacts = _fixture(tmp_path)
    config = json.loads(ss.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["cost_profiles"] = str(ss.CONFIG_DIR / "cost_profiles.json")
    config["markets"]["TWSE"]["min_dollar_volume"] = 1_000_000
    config["markets"]["TPEX"]["min_dollar_volume"] = 1_000_000
    config["scenarios"] = ["md_ls", "md_ls_tx"]
    config["construction"] = {"score_variants": ["v0", "v1"], "leg_matching": ["none", "momentum"],
                              "momentum_quintiles": 2}
    config["multi_day"]["fixed"] = {"good": {"h": 10, "k": 6, "buffer": 2}}
    config["bootstrap"]["replicates"] = 50
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "out-v1"
    report = ss.run(path, [predictions_path], ["good"], prices_path, output, artifacts=artifacts)
    grid = pd.read_csv(output / "grid.csv")
    assert set(grid["scenario"]) == {"md_ls", "md_ls_tx"}
    assert set(zip(grid["score_variant"], grid["leg_matching"], strict=True)) == {
        ("v0", "none"), ("v1", "none"), ("v0", "momentum"), ("v1", "momentum")}
    assert report["selection"]["good"]["multi_day"]["fixed"] is True
    assert report["selection"]["good"]["day_trade"] == {"skipped": True}
    assert (grid["stage"] == "final").all()  # no selection grid with fixed parameters
    verdict = report["attractiveness"]["good"]["md_ls"]
    assert "TWSE" in verdict and "TWSE|v1/momentum" in verdict
    daily = pd.read_csv(output / "daily.csv")
    assert {"score_variant", "leg_matching"} <= set(daily.columns)
    yearly = pd.read_csv(output / "yearly.csv")
    assert {"year", "net_return", "sharpe", "max_drawdown", "days"} <= set(yearly.columns)
    assert set(yearly["year"]) >= {2023, 2024, 2025}
    # The momentum-matched book splits k over the quintile sub-books but keeps the same k label.
    matched = grid[(grid["leg_matching"] == "momentum") & (grid["market"] == "TWSE")]
    assert matched["k"].eq(6).all() and matched["turnover"].notna().all()


def test_multi_day_default_fixes_every_run_from_the_named_run(tmp_path):
    """Protocol 13.23 supplement 1 (g): runs without a ``multi_day.fixed`` entry take the named
    run's (h, K, m) instead of selecting their own on the P1 grid; the source is recorded."""
    predictions_path, prices_path, artifacts = _fixture(tmp_path)
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["multi_day"]["fixed"] = {"good": {"h": 10, "k": 5, "buffer": 2}}
    payload["scenarios"] = ["md_ls"]
    config.write_text(json.dumps(payload))
    output = tmp_path / "out-default"
    report = ss.run(config, [predictions_path, predictions_path], ["good", "good"],
                    prices_path, output, artifacts=artifacts, labels=["good", "other"],
                    multi_day_default="good")
    for label, source in (("good", "good"), ("other", "good")):
        choice = report["selection"][label]["multi_day"]
        assert choice["fixed"] is True and choice["fixed_from"] == source
        assert (choice["h"], choice["k"], choice["buffer"]) == (10, 5, 2)
    grid = pd.read_csv(output / "grid.csv")
    assert (grid["stage"] == "final").all()  # no P1 selection grid for either run
    assert set(grid["model"]) == {"good", "other"}
    with pytest.raises(ValueError, match="multi-day-default"):
        ss.run(config, [predictions_path], ["good"], prices_path, tmp_path / "bad",
               artifacts=artifacts, labels=["other"], multi_day_default="missing")
