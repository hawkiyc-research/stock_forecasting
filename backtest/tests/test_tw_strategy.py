"""Taiwan strategy layer: band rule, smoothing, lot feasibility, futures overlay, end to end."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twbacktest import positions, tw_strategy
from twbacktest.walkforward_data import build_inputs, load_predictions, load_prices

HEADER = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,"
          "未沖銷契約數,最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,"
          "價差對單式委託成交量")
MARKETS = {"TWSE": ("TW", 40, "TAIEX.TW"), "TPEX": ("TWO", 25, "TPEX.TWO")}


def _write_tx(root: Path, sessions: pd.DatetimeIndex, level: np.ndarray) -> None:
    """A TX download root whose front month follows ``level`` (two expiries per session)."""
    lines = [HEADER]
    for day, value in zip(sessions, level, strict=True):
        month = day.to_period("M")
        expiries = [month.strftime("%Y%m"), (month + 1).strftime("%Y%m")]
        for expiry in expiries:
            open_, settle = value, value * 1.001
            lines.append(
                f"{day:%Y/%m/%d},TX,{expiry}  ,{open_:.0f},{open_ + 50:.0f},{open_ - 50:.0f},"
                f"{settle:.0f},0,0%,1000,{settle:.0f},5000,{open_:.0f},{open_:.0f},1,1,,一般,,"
            )
    content = ("\r\n".join(lines) + "\r\n").encode("cp950")
    name = f"futures-TX-{sessions[0]:%Y%m%d}-{sessions[-1]:%Y%m%d}.csv"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / name).write_bytes(content)
    manifest = {"source": "test", "commodity": "TX", "files": {
        name: hashlib.sha256(content).hexdigest()}}
    (root / "download-TX.json").write_text(json.dumps(manifest))


def _fixture(tmp_path: Path, seed: int = 5) -> tuple[Path, Path, Path]:
    """Prices with volume, scores that know the next open-to-open move, and a TX root."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2022-11-01", "2026-03-31")
    signal_dates = sessions[(sessions >= "2023-01-02") & (sessions < "2026-03-01")]
    price_rows, prediction_rows = [], []
    index_level = 15000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(sessions))))
    for market, (suffix, count, index_symbol) in MARKETS.items():
        symbols = [f"{1101 + i}.{suffix}" for i in range(count)]
        base = np.where(np.arange(count) % 7 == 0, 900.0, 60.0)  # a few expensive names
        opens = base[:, None] * np.exp(
            np.cumsum(rng.normal(0, 0.015, (count, len(sessions))), axis=1))
        closes = opens * np.exp(rng.normal(0, 0.008, opens.shape))
        volumes = np.where(np.arange(count) % 5 == 0, 200.0, 2_000_000.0)  # illiquid names
        for day, date in enumerate(sessions):
            for i, symbol in enumerate(symbols):
                price_rows.append((date, symbol, "etf" if i == 1 else "stock", opens[i, day],
                                   closes[i, day], closes[i, day], volumes[i], market))
            price_rows.append((date, index_symbol, "index", index_level[day],
                               index_level[day] * 1.001, index_level[day] * 1.001, 0.0, market))
        for date in signal_dates:
            day = sessions.get_loc(date)
            for i, symbol in enumerate(symbols):
                move = np.log(opens[i, day + 2] / opens[i, day + 1])
                ten = np.log(opens[i, min(day + 11, len(sessions) - 1)] / opens[i, day + 1])
                scores = {f"h{h:02d}": rng.normal() for h in range(1, 15)}
                scores["h05"] = move + rng.normal(0, 0.004)
                scores["h10"] = ten + rng.normal(0, 0.01)
                prediction_rows.append({"model": "good", "symbol": symbol, "market": market,
                                        "signal_date": date, **scores})
    prices = pd.DataFrame(price_rows, columns=[
        "timestamp", "symbol", "asset_type", "open", "close", "adjusted_close", "volume",
        "market"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize("UTC")
    prices_path = tmp_path / "market.parquet"
    prices.to_parquet(prices_path, index=False)
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    tx_root = tmp_path / "artifacts" / "diagnostics" / "taifex-tx"
    _write_tx(tx_root, sessions, index_level)
    return predictions_path, prices_path, tmp_path / "artifacts"


def _config(tmp_path: Path, **overrides) -> Path:
    config = json.loads(tw_strategy.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["markets"]["TWSE"]["min_dollar_volume"] = 1_000_000
    config["markets"]["TPEX"]["min_dollar_volume"] = 1_000_000
    config["grid"] = {"score_horizons": [5, 10], "k": [5], "buffer_multiples": [2]}
    config["capital"]["sensitivity_twd"] = [2_000_000]
    config["bootstrap"]["replicates"] = 50
    config["periods"] = {"P1": ["2023-01-02", "2025-06-01"], "P2": ["2025-06-01", "2026-06-01"]}
    config.update(overrides)
    path = tmp_path / "tw_strategy.json"
    path.write_text(json.dumps(config))
    return path


def test_smoothing_half_life_and_gap_reset():
    values = np.array([1.0, 0.0, 0.0, 1.0, np.nan, 0.5])
    dates = np.array([0, 1, 2, 10, 11, 11])
    symbols = np.array([0, 0, 0, 0, 1, 0])
    out = tw_strategy.smooth_by_symbol(values, dates, symbols, half_life=1.0, max_gap=5)
    assert out[0] == 1.0
    assert out[1] == pytest.approx(0.5)  # half-life 1: theta = 0.5
    assert out[2] == pytest.approx(0.25)
    assert out[3] == 1.0  # gap of 8 sessions resets the state
    assert np.isnan(out[4])
    assert out[5] == pytest.approx(0.75)


def test_band_entry_cap_and_lot_feasibility():
    # Three signal dates, six stocks, K = 4, cap 1 buy per session.
    n_dates, n_stocks = 3, 6
    rows = np.arange(n_dates * n_stocks)
    symbols = np.tile(np.arange(n_stocks), n_dates)
    link = positions.Links(
        next_row=np.where(rows + n_stocks < rows.size, rows + n_stocks, -1),
        prev_row=np.where(rows - n_stocks >= 0, rows - n_stocks, -1),
    )
    move = positions.Moves(growth=np.zeros(rows.size), fallback=np.zeros(rows.size),
                           gap=np.zeros(rows.size), index_growth=np.zeros(rows.size))
    ranked = [rows[d * n_stocks:(d + 1) * n_stocks] for d in range(n_dates)]
    result = tw_strategy.simulate_band(
        ranked, k=4, buffer=2, max_buys=1, symbols=symbols, link=link, move=move, limit=0.095,
    )
    assert (result.path.buy[0] >= 0).sum() == 1
    assert (result.path.buy[1] >= 0).sum() == 1
    # Lot feasibility: stock 0's lot costs more than a slot of 100 / 4 = 25.
    lot_cost = np.where(symbols == 0, 40.0, 10.0)
    result = tw_strategy.simulate_band(
        ranked, k=4, buffer=2, max_buys=4, symbols=symbols, link=link, move=move, limit=0.095,
        lot_cost=lot_cost, slot_capital=100.0,
    )
    assert result.lot_blocked == 1
    assert 0 not in set(symbols[result.path.buy[0][result.path.buy[0] >= 0]])


def test_hedged_overlay_is_flat_when_stocks_track_the_future_and_calls_on_a_spike():
    growth = np.full(10, 0.01)
    futures = tw_strategy.FuturesSeries(growth=growth, roll=np.zeros(10, dtype=bool),
                                        rate=np.zeros(10), index_growth=growth, missing=0)
    overlay = tw_strategy.hedged_overlay(
        growth, growth, futures, beta=1.0, reserve_fraction=0.15, call_fraction=0.10,
        reinvest_fraction=0.25, stock_buy_rate=0.0, stock_sell_rate=0.0,
    )
    # Stocks and the short cancel exactly without costs; a steady rise still drains the
    # reserve into the stocks' gains, so margin is moved back from the stocks (a call).
    assert np.allclose(overlay.net, 0.0, atol=1e-12)
    assert overlay.margin_calls >= 1
    alternating = np.tile([0.01, -0.01], 5)
    futures = tw_strategy.FuturesSeries(growth=alternating, roll=np.zeros(10, dtype=bool),
                                        rate=np.zeros(10), index_growth=alternating, missing=0)
    overlay = tw_strategy.hedged_overlay(
        alternating, alternating, futures, beta=1.0, reserve_fraction=0.15, call_fraction=0.10,
        reinvest_fraction=0.25, stock_buy_rate=0.0, stock_sell_rate=0.0,
    )
    assert overlay.margin_calls == 0 and overlay.reinvestments == 0
    spike = np.zeros(3)
    spike[1] = 0.40  # the index jumps 40%: the short loses more than the 15% reserve
    futures = tw_strategy.FuturesSeries(growth=spike, roll=np.zeros(3, dtype=bool),
                                        rate=np.zeros(3), index_growth=spike, missing=0)
    overlay = tw_strategy.hedged_overlay(
        np.zeros(3), np.zeros(3), futures, beta=1.0, reserve_fraction=0.15, call_fraction=0.10,
        reinvest_fraction=0.25, stock_buy_rate=0.0, stock_sell_rate=0.005,
    )
    assert overlay.margin_calls == 1
    assert overlay.net[1] == pytest.approx(-0.40 / 1.15 - 0.005 * (0.40 + 0.15 * (1 - 0.0)) / 1.15,
                                           abs=2e-3)


def test_cash_equitized_adds_idle_cash_exposure():
    futures = tw_strategy.FuturesSeries(growth=np.array([0.02, 0.02]), roll=np.array([False, True]),
                                        rate=np.array([0.0001, 0.0001]),
                                        index_growth=np.array([0.02, 0.02]), missing=0)
    overlay = tw_strategy.cash_equitized_overlay(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5]), futures)
    assert overlay.net[0] == pytest.approx(0.5 * 0.02 - 0.0001 * 0.5)
    assert overlay.net[1] == pytest.approx(0.5 * 0.02 - 0.0001 * (0.0 + 2 * 0.5))


def test_universe_filters_liquidity_price_and_etf(tmp_path):
    predictions_path, prices_path, _ = _fixture(tmp_path)
    inputs = build_inputs(load_predictions(predictions_path, "good"), load_prices(prices_path))
    mask = tw_strategy.universe_mask(inputs, (0,), min_dollar_volume=1_000_000, min_close=10,
                                     exclude_etf=True)
    symbols = np.array(inputs.index.symbol_names)[inputs.index.symbol_codes]
    kept = set(symbols[mask])
    assert "1102.TW" not in kept  # the ETF
    assert "1101.TW" not in kept  # illiquid (200 shares a day)
    assert "1103.TW" in kept
    assert not any(s.endswith(".TWO") for s in kept)


def test_end_to_end_run_writes_outputs_and_decisions(tmp_path):
    predictions_path, prices_path, artifacts = _fixture(tmp_path)
    config = _config(tmp_path)
    output = tmp_path / "out"
    report = tw_strategy.run(config, [predictions_path], ["good"], prices_path, output,
                             artifacts=artifacts, workers=1)
    grid = pd.read_csv(output / "grid.csv")
    assert (output / "summary.md").is_file() and (output / "daily_series.csv").is_file()
    assert set(grid["version"].dropna()) >= {"long_only", "hedged", "cash_equitized",
                                             "tx_long_reference"}
    chosen = report["selection"]["good"]
    assert chosen["h"] in (5, 10) and chosen["k"] == 5 and chosen["buffer"] == 2
    verdicts = report["decisions"]["good"]
    assert set(verdicts) == {"H-D", "H-D-hedged"}
    # Scores that know the next move must beat the equal-weight universe net of costs.
    assert verdicts["H-D"]["passed"] is True
    assert report["forward_candidate"] == "good"
    primary = grid[(grid["version"] == "long_only") & (grid["discount"] == 0.6)
                   & (grid["slippage_multiplier"] == 1) & (grid["period"] == "P1+P2")
                   & (grid["market"] == "TWSE")]
    assert {"net_return_annual", "net_cagr", "sharpe", "max_drawdown", "index_excess_annual",
            "lot_blocked_share", "margin_calls"} <= set(primary.columns)
    lots = primary[primary["capital"] == "2000000"] if primary["capital"].dtype == object \
        else primary[primary["capital"] == 2000000]
    assert (lots["lot_blocked"] > 0).all()  # the 900-dollar names cost more than a slot
    reference = grid[grid["version"] == "tx_long_reference"]
    assert reference["rolls"].ge(1).all()
