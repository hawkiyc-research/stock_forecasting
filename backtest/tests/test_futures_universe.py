"""Stock-futures universe diagnostic on synthetic predictions, prices and TAIFEX files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twbacktest import futures_universe
from twbacktest.walkforward_data import build_holding_returns, load_predictions, load_prices

HEADER = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,"
          "未沖銷契約數,最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,"
          "價差對單式委託成交量")


def _listing_html(entries):
    def row(prefix, stock, short, board, shares):
        cells = [prefix, f"{short}公司", stock, short, "● 是股票期貨標的", "", "",
                 "◎" if board == "twse" else "", "◎" if board == "tpex" else "", "", "",
                 shares, "8:45~13:45", "-"]
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
    return "<table>" + "".join(row(*entry) for entry in entries) + "</table>"


def _taifex_root(root: Path, sessions: pd.DatetimeIndex, contracts: dict[str, int]) -> None:
    """Every contract trades every session with the given volume (standard 2,000-share)."""
    lines = [HEADER]
    for day in sessions:
        expiry = day.to_period("M").strftime("%Y%m")
        for code, volume in contracts.items():
            lines.append(f"{day:%Y/%m/%d},{code},{expiry}  ,10,11,9,10,0,0%,{volume},10,5,10,10,"
                         f"12,8,,一般,,")
    content = ("\r\n".join(lines) + "\r\n").encode("cp950")
    name = f"futures-{sessions[0]:%Y%m%d}-{sessions[-1]:%Y%m%d}.csv"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / name).write_bytes(content)
    listing = _listing_html([("CD", "1101", "甲", "twse", "2,000"),
                             ("DE", "1102", "乙", "twse", "2,000")])
    (root / "raw" / "stock-lists-20261023.html").write_text(listing, encoding="utf-8")
    files = {name: hashlib.sha256(content).hexdigest(),
             "stock-lists-20261023.html": hashlib.sha256(listing.encode("utf-8")).hexdigest()}
    (root / "download.json").write_text(json.dumps({"files": files}))


def test_segments_and_decile_returns(tmp_path):
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2024-12-02", "2025-08-29")
    signal_dates = sessions[(sessions >= "2025-01-02") & (sessions < "2025-08-01")]
    symbols = [f"{1101 + i}.TW" for i in range(30)]
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (30, len(sessions))), axis=1))
    closes = opens * np.exp(rng.normal(0, 0.005, opens.shape))
    rows = []
    for day, date in enumerate(sessions):
        for i, symbol in enumerate(symbols):
            rows.append((date, symbol, "stock", opens[i, day], closes[i, day], closes[i, day],
                         1_000_000.0, "TWSE"))
        rows.append((date, "TAIEX.TW", "index", 100.0, 100.0, 100.0, 0.0, "TWSE"))
        rows.append((date, "TPEX.TWO", "index", 100.0, 100.0, 100.0, 0.0, "TPEX"))
    prices = pd.DataFrame(rows, columns=["timestamp", "symbol", "asset_type", "open", "close",
                                         "adjusted_close", "volume", "market"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize("UTC")
    prices.to_parquet(tmp_path / "market.parquet", index=False)
    predictions = []
    for date in signal_dates:
        day = sessions.get_loc(date)
        for i, symbol in enumerate(symbols):
            label = np.log(closes[i, day + 1] / opens[i, day + 1])  # the h = 1 label
            scores = {f"h{h:02d}": rng.normal() for h in range(1, 15)}
            scores["h01"] = label  # a perfect h = 1 ranking
            predictions.append({"model": "m", "symbol": symbol, "market": "TWSE",
                                "signal_date": date, **scores})
    pd.DataFrame(predictions).to_parquet(tmp_path / "predictions.parquet", index=False)
    root = tmp_path / "taifex"
    _taifex_root(root, sessions, {"CDF": 500, "DEF": 10})  # 1101 liquid, 1102 listed only

    frame = load_predictions(tmp_path / "predictions.parquet", "m")
    eligibility, audit = futures_universe.eligibility_frame(root, min_volume=100, lookback=20)
    masks = futures_universe.segment_masks(frame, eligibility)
    assert masks["futures_listed"].sum() == 2 * len(signal_dates)
    assert masks["futures_liquid"].sum() == len(signal_dates)
    assert audit["list_codes"] == 2

    price_frame = load_prices(tmp_path / "market.parquet")
    log_return, log_alpha = build_holding_returns(frame, price_frame)
    from twbacktest.walkforward_data import build_inputs
    inputs = build_inputs(frame, price_frame)
    daily = futures_universe.daily_table(frame, inputs.scores[:, 0], log_return[:, 0],
                                         log_alpha[:, 0], masks["all"], ("TWSE",))
    assert len(daily) == len(signal_dates)
    assert (daily["rank_ic"] > 0.99).all()
    assert (daily["top_return"] >= daily["bottom_return"]).all()
    summary = futures_universe.summarize(daily, h=1)
    assert summary["rank_ic_low"] > 0.99 and summary["spread_low"] > 0

    report = futures_universe.run([tmp_path / "predictions.parquet"], ["m"],
                                  tmp_path / "market.parquet", root, tmp_path / "out",
                                  min_volume=100, lookback=20)
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    assert set(grid["segment"]) == {"all"}  # the futures segments have fewer than 10 names
    assert (tmp_path / "out" / "summary.md").is_file()
    assert report["eligibility_audit"]["list_codes"] == 2
