"""Protocol 13.17 step 0 diagnostic on synthetic predictions, residual targets, prices, TAIFEX
files and an industry reference table."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from twbacktest import exposure_diagnostics as diag

HEADER = (
    "交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,"
    "未沖銷契約數,最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,"
    "價差對單式委託成交量"
)


def _listing_html(entries):
    def row(prefix, stock, short, shares):
        cells = [
            prefix,
            f"{short}公司",
            stock,
            short,
            "● 是股票期貨標的",
            "",
            "",
            "◎",
            "",
            "",
            "",
            shares,
            "8:45~13:45",
            "-",
        ]
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

    return "<table>" + "".join(row(*entry) for entry in entries) + "</table>"


def _taifex_root(root, sessions, contracts):
    lines = [HEADER]
    for day in sessions:
        expiry = day.to_period("M").strftime("%Y%m")
        for code, volume in contracts.items():
            lines.append(
                f"{day:%Y/%m/%d},{code},{expiry}  ,100,101,99,100,0,0%,{volume},100,5,"
                f"100,100,120,80,,一般,,"
            )
    content = ("\r\n".join(lines) + "\r\n").encode("cp950")
    name = f"futures-{sessions[0]:%Y%m%d}-{sessions[-1]:%Y%m%d}.csv"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / name).write_bytes(content)
    entries = [
        (code[:2], code_stock, f"甲{i}", "2,000")
        for i, (code, code_stock) in enumerate(contracts_to_stocks(contracts).items())
    ]
    listing = _listing_html(entries)
    (root / "raw" / "stock-lists-20261023.html").write_text(listing, encoding="utf-8")
    files = {
        name: hashlib.sha256(content).hexdigest(),
        "stock-lists-20261023.html": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
    }
    (root / "download.json").write_text(json.dumps({"files": files}))


def contracts_to_stocks(contracts):
    # Contract AAF -> stock 1101, ABF -> 1102, ... (deterministic two-letter prefixes).
    return {code: str(1101 + i) for i, code in enumerate(contracts)}


def test_quantile_long_short_is_class_neutral_and_cost_formulas():
    score = np.array([5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4], dtype=float)
    ret = np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])
    classes = np.array(["a"] * 6 + ["b"] * 6, dtype=object)
    spread, used, weights = diag.quantile_long_short(score, ret, classes)
    assert used == 12 and weights == {"a": 0.5, "b": 0.5}
    # Top 2 minus bottom 2 inside each class: (0.045 - 0.005) and (0.085 - 0.045).
    assert spread == pytest_approx(0.5 * (0.04 + 0.04))
    plain, used_plain, _ = diag.quantile_long_short(score, ret, None)
    assert used_plain == 12 and plain > 0
    # Small classes are pooled; a lone class below MIN_CLASS is skipped.
    tiny = np.array(["a"] * 3 + ["b"] * 9, dtype=object)
    _spread_tiny, used_tiny, weights_tiny = diag.quantile_long_short(score, ret, tiny)
    assert (used_tiny == 12 and set(weights_tiny) == {"b", diag.OTHER}) or used_tiny == 9
    assert (
        diag.tick_size(9.0) == 0.01
        and diag.tick_size(100.0) == 0.5
        and diag.tick_size(2000.0) == 5.0
    )
    rate = diag.per_side_cost_rate(np.array([100.0]), np.array([2000.0]))[0]
    assert rate == pytest_approx(2e-5 + 0.5 / 100.0 + 30.0 / (100.0 * 2000.0))
    assert diag.annual_cost_drag(0.001, 10) == pytest_approx((252 * 0.2 + 24) * 0.001)


def pytest_approx(value, rel=1e-9):
    import pytest

    return pytest.approx(value, rel=rel, abs=1e-12)


def test_run_reports_universes_spreads_and_stop_rule(tmp_path):
    rng = np.random.default_rng(3)
    sessions = pd.bdate_range("2024-11-01", "2025-09-30")
    signal_dates = sessions[(sessions >= "2025-01-02") & (sessions < "2025-08-01")]
    n = 40
    symbols = [f"{1101 + i}.TW" for i in range(n)]
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(sessions))), axis=1))
    closes = opens * np.exp(rng.normal(0, 0.005, opens.shape))
    rows = []
    for day, date in enumerate(sessions):
        for i, symbol in enumerate(symbols):
            rows.append(
                (
                    date,
                    symbol,
                    "stock",
                    opens[i, day],
                    closes[i, day],
                    closes[i, day],
                    2_000_000.0,
                    "TWSE",
                )
            )
        rows.append((date, "TAIEX.TW", "index", 100.0, 100.0, 100.0, 0.0, "TWSE"))
        rows.append((date, "TPEX.TWO", "index", 100.0, 100.0, 100.0, 0.0, "TPEX"))
    prices = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "symbol",
            "asset_type",
            "open",
            "close",
            "adjusted_close",
            "volume",
            "market",
        ],
    )
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize("UTC")
    prices.to_parquet(tmp_path / "market.parquet", index=False)
    industry = {s: ("24" if i % 2 == 0 else "17") for i, s in enumerate(symbols)}
    predictions, targets = [], []
    for date in signal_dates:
        day = sessions.get_loc(date)
        labels = {h: np.log(closes[:, day + h] / opens[:, day + 1]) for h in (5, 10)}
        for i, symbol in enumerate(symbols):
            scores = {f"h{h:02d}": rng.normal() for h in range(1, 15)}
            scores["h10"] = labels[10][i]  # perfect h = 10 ranking
            scores["h05"] = labels[5][i]
            predictions.append(
                {"model": "m", "symbol": symbol, "market": "TWSE", "signal_date": date, **scores}
            )
            same = [j for j in range(n) if industry[symbols[j]] == industry[symbol]]
            targets.append(
                {
                    "symbol": symbol,
                    "signal_date": date,
                    "residual_h05": labels[5][i] - labels[5][same].mean(),
                    "residual_h10": labels[10][i] - labels[10][same].mean(),
                }
            )
    pd.DataFrame(predictions).to_parquet(tmp_path / "predictions.parquet", index=False)
    pd.DataFrame(targets).to_parquet(tmp_path / "targets.parquet", index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "symbol": symbols,
            "market": "TWSE",
            "industry_code": [industry[s] for s in symbols],
            "issued_shares": 1_000_000,
            "par_value_twd": 10.0,
            "snapshot_date": "2026-10-22",
        }
    ).to_csv(reference, index=False)
    # 24 of the 40 names have a liquid stock-futures contract (contract AAF -> 1101, ...).
    codes = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}F" for i in range(24)]
    root = tmp_path / "taifex"
    _taifex_root(root, sessions, dict.fromkeys(codes, 500))

    report = diag.run(
        [tmp_path / "predictions.parquet"],
        [tmp_path / "targets.parquet"],
        ["m"],
        tmp_path / "market.parquet",
        root,
        reference,
        tmp_path / "out",
        min_volume=100,
        lookback=20,
    )
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    assert set(grid["universe"]) == {"liquidity", "futures_liquid"} and set(grid["h"]) == {5, 10}
    decision = grid[
        (grid["universe"] == "futures_liquid") & (grid["h"] == 10) & (grid["period"] == "P1+P2")
    ].iloc[0]
    assert 20 <= decision["names_mean"] <= 24
    assert decision["rank_ic_raw"] > 0.99 and decision["rank_ic_residual"] > 0.5
    assert decision["quartile_spread_raw"] > 0 and decision["quartile_spread_residual"] > 0
    assert decision["ls_industry_gross_annual_low"] > 0 and decision["ls_plain_gross_annual"] > 0
    assert decision["ls_size_gross_annual"] > 0
    # Prices wander around 100 (tick 0.1 below, 0.5 above) on 2,000-share contracts.
    assert 0.001 < decision["cost_rate_per_side"] < 0.007
    assert decision["cost_drag_annual"] == pytest_approx(
        diag.annual_cost_drag(decision["cost_rate_per_side"], 10), rel=1e-6
    )
    verdict = report["stop_rule"]["verdicts"][0]
    assert verdict["evaluated"] and verdict["proceed"] is True
    assert (tmp_path / "out" / "summary.md").read_text(encoding="utf-8").count(
        "繼續 / proceed"
    ) == 2
    liquidity = grid[
        (grid["universe"] == "liquidity") & (grid["h"] == 5) & (grid["period"] == "P1")
    ]
    assert len(liquidity) == 1 and liquidity.iloc[0]["names_mean"] == 40
