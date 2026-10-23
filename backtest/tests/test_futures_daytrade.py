"""Stock-futures short day trade on synthetic sessions, contracts and TX prices."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from twbacktest import futures_daytrade as fd


def _config(**overrides):
    config = json.loads(fd.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["bootstrap"]["replicates"] = 50
    config.update(overrides)
    return config


def _session(scores, opens, settles, volumes=None, shares=None, tx=(20000.0, 20000.0)):
    n = len(scores)
    opens = np.asarray(opens, dtype=float)
    return fd.Session(
        signal_date=pd.Timestamp("2025-01-02"), trade_date="2025-01-03",
        scores=np.asarray(scores, dtype=float), open=opens,
        settlement=np.asarray(settles, dtype=float),
        volume=np.asarray(volumes if volumes is not None else [1000.0] * n, dtype=float),
        shares=np.asarray(shares if shares is not None else [2000.0] * n, dtype=float),
        tick=np.array([fd.tick_size(p, _config()["costs"]["tick_bands"]) for p in opens]),
        tx_open=tx[0], tx_settlement=tx[1], skipped_no_price=0, small_contract=0,
    )


def test_tick_size_bands():
    bands = _config()["costs"]["tick_bands"]
    assert fd.tick_size(5.0, bands) == 0.01
    assert fd.tick_size(49.9, bands) == 0.05
    assert fd.tick_size(50.0, bands) == 0.1
    assert fd.tick_size(250.0, bands) == 0.5
    assert fd.tick_size(999.0, bands) == 1.0
    assert fd.tick_size(1500.0, bands) == 5.0


def test_short_leg_pnl_costs_and_contract_rounding():
    tx = {"multiplier": 200.0, "tax_rate": 0.00002, "commission": 50.0, "slippage_points": 1.0}
    cost = fd.CostScenario(commission=30.0, ticks=1.0)
    # Ten names; the two lowest scores are shorted with q = 0.2; they fall 2% intraday.
    scores = np.arange(10, dtype=float)
    opens = np.full(10, 100.0)
    settles = opens.copy()
    settles[:2] = 98.0
    session = _session(scores, opens, settles)
    frac = fd.trade_session(session, fraction=0.2, version="short_only", capital=None,
                            leverage=1.0, cost=cost, tax_rate=0.00002, participation_cap=0.05,
                            margin_rate=0.2025, tx=tx)
    assert frac.gross == pytest.approx(0.02)  # fully invested in the two shorts
    side = 0.00002 + 30.0 / (100.0 * 2000.0) + 1.0 * 0.5 / 100.0
    assert frac.net == pytest.approx(0.02 - 2 * side)
    assert frac.margin_usage == pytest.approx(0.2025)
    assert frac.deployed == pytest.approx(1.0)
    # NT$1,000,000: two names, NT$500k each -> 2 contracts of NT$200k each per name.
    cap = fd.trade_session(session, fraction=0.2, version="short_only", capital=1_000_000.0,
                           leverage=1.0, cost=cost, tax_rate=0.00002, participation_cap=0.05,
                           margin_rate=0.2025, tx=tx)
    assert cap.contracts == 4
    notional = 4 * 2000 * 100.0
    assert cap.gross == pytest.approx(4 * 2000 * 2.0 / 1_000_000)
    assert cap.net == pytest.approx((4 * 2000 * 2.0 - 2 * notional * side) / 1_000_000)
    assert cap.margin_usage == pytest.approx(notional * 0.2025 / 1_000_000)
    # NT$300,000: NT$150k per name buys no NT$200k contract -> both skipped.
    small = fd.trade_session(session, fraction=0.2, version="short_only", capital=300_000.0,
                             leverage=1.0, cost=cost, tax_rate=0.00002, participation_cap=0.05,
                             margin_rate=0.2025, tx=tx)
    assert small.skipped_size == 2 and small.contracts == 0 and small.net == 0.0
    # Participation cap: 5% of a 20-contract day allows 1 contract, not 2.
    thin = _session(scores, opens, settles, volumes=[20.0] * 10)
    capped = fd.trade_session(thin, fraction=0.2, version="short_only", capital=1_000_000.0,
                              leverage=1.0, cost=cost, tax_rate=0.00002, participation_cap=0.05,
                              margin_rate=0.2025, tx=tx)
    assert capped.contracts == 2 and capped.skipped_participation == 0
    tiny = _session(scores, opens, settles, volumes=[10.0] * 10)
    zero = fd.trade_session(tiny, fraction=0.2, version="short_only", capital=1_000_000.0,
                            leverage=1.0, cost=cost, tax_rate=0.00002, participation_cap=0.05,
                            margin_rate=0.2025, tx=tx)
    assert zero.skipped_participation == 2 and zero.contracts == 0


def test_long_short_and_hedged_versions():
    tx = {"multiplier": 200.0, "tax_rate": 0.00002, "commission": 50.0, "slippage_points": 1.0}
    cost = fd.CostScenario(commission=0.0, ticks=0.0)
    scores = np.arange(10, dtype=float)
    opens = np.full(10, 100.0)
    settles = opens.copy()
    settles[:2] = 98.0  # shorts gain 2%
    settles[-2:] = 101.0  # longs gain 1%
    session = _session(scores, opens, settles, tx=(20000.0, 20200.0))  # TX +1%
    ls = fd.trade_session(session, fraction=0.2, version="long_short", capital=None, leverage=1.0,
                          cost=cost, tax_rate=0.0, participation_cap=0.05, margin_rate=0.2,
                          tx=tx)
    assert ls.gross == pytest.approx(0.5 * 0.02 + 0.5 * 0.01)
    hedged = fd.trade_session(session, fraction=0.2, version="short_only", capital=None,
                              leverage=1.0, cost=cost, tax_rate=0.0, participation_cap=0.05,
                              margin_rate=0.2, tx=tx)
    hedged_v = fd.trade_session(session, fraction=0.2, version="hedged", capital=None,
                                leverage=1.0, cost=cost, tax_rate=0.0, participation_cap=0.05,
                                margin_rate=0.2, tx=tx)
    assert hedged_v.gross == pytest.approx(hedged.gross + 1.0 * 0.01)
    tx_side = 0.00002 + (50.0 + 200.0) / (20000.0 * 200.0)
    assert hedged_v.net == pytest.approx(hedged.net + 0.01 - 2 * tx_side)


def _frames():
    """Twenty sessions, eight names: name 0 is always the worst-scored and falls 3% intraday."""
    dates = pd.bdate_range("2025-01-02", periods=21).strftime("%Y-%m-%d").tolist()
    symbols = [f"{1101 + i}.TW" for i in range(8)]
    rows, pred_rows, elig_rows, tx_rows = [], [], [], []
    rng = np.random.default_rng(3)
    for i, date in enumerate(dates):
        for j, symbol in enumerate(symbols):
            open_ = 100.0 + j
            settle = open_ * (0.97 if j == 0 else 1.0 + rng.normal(0, 0.002))
            rows.append((date, symbol, "CDF", "202501", 2000.0, True, open_, settle, 500.0, 1000.0))
            elig_rows.append((date, symbol, j != 7))  # name 7 is never liquid
            if i < len(dates) - 1:
                pred_rows.append({"model": "m", "symbol": symbol, "market": "TWSE",
                                  "signal_date": pd.Timestamp(date),
                                  **{f"h{h:02d}": (float(j) if h == 1 else 0.0)
                                     for h in range(1, 15)}})
        tx_rows.append((date, "202501", 20000.0, 20000.0 * (1 + rng.normal(0, 0.005)), True))
    futures_table = pd.DataFrame(rows, columns=["date", "symbol", "contract", "expiry", "shares",
                                                "is_standard", "open", "settlement", "volume",
                                                "open_interest"])
    eligibility = pd.DataFrame(elig_rows, columns=["date", "symbol", "liquid"])
    tx_front = pd.DataFrame(tx_rows, columns=["date", "expiry", "open", "settlement", "roll"])
    return pd.DataFrame(pred_rows), futures_table, eligibility, tx_front


def test_build_sessions_uses_next_session_prices_and_liquidity():
    predictions, futures_table, eligibility, tx_front = _frames()
    sessions = fd.build_sessions(predictions, futures_table, eligibility, tx_front,
                                 horizon_column="h01", tick_bands=_config()["costs"]["tick_bands"],
                                 require_open=True)
    assert len(sessions) == 20
    first = sessions[0]
    assert first.trade_date == "2025-01-03" and first.scores.size == 7  # name 7 excluded
    assert first.open[0] == 100.0 and first.settlement[0] == pytest.approx(97.0)


def test_run_frames_selects_and_decides(tmp_path):
    predictions, futures_table, eligibility, tx_front = _frames()
    config = _config()
    config["periods"] = {"P1": ["2025-01-01", "2025-01-20"], "P2": ["2025-01-20", "2025-03-01"]}
    report = fd.run_frames({"m": predictions}, futures_table, eligibility, tx_front, config,
                           tmp_path / "out")
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    assert {"short_only", "long_short", "hedged"} <= set(grid["version"])
    assert set(grid["capital"]) == {"fractional", "10M", "30M", "100M", "300M", "1B"}
    chosen = report["selection"]["m"]["fraction"]
    assert chosen in (0.1, 0.2)
    verdict = report["decisions"]["m"]
    assert verdict["hypothesis"] == "H-E" and verdict["passed"] is True  # name 0 falls 3%
    assert report["forward_candidate"] == "m"
    assert (tmp_path / "out" / "summary.md").is_file()
    row = grid[(grid["version"] == "short_only") & (grid["capital"] == "fractional")
               & (grid["period"] == "P1+P2") & (grid["fraction"] == chosen)
               & (grid["commission"] == 30) & (grid["ticks"] == 1)].iloc[0]
    assert row["gross_return_annual"] > row["net_return_annual"] > 0
    big = grid[(grid["version"] == "short_only") & (grid["capital"] == "1B")
               & (grid["period"] == "P1+P2") & (grid["fraction"] == chosen)
               & (grid["commission"] == 30) & (grid["ticks"] == 1)].iloc[0]
    # 500 contracts a day cap the NT$1B slot at 25 contracts: little of the notional deploys.
    assert big["contracts_mean"] == pytest.approx(25.0)
    assert big["deployed_share_mean"] < 0.01
    assert row["deployed_share_mean"] == pytest.approx(1.0)
