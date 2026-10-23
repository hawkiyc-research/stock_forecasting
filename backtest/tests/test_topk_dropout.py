"""Top-k dropout on synthetic walk-forward predictions and raw prices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twbacktest import positions, topk_dropout
from twbacktest.walkforward_data import build_inputs, load_predictions, load_prices

MARKETS = {"TWSE": ("TW", 60, "TAIEX.TW"), "TPEX": ("TWO", 40, "TPEX.TWO")}


def _fixture(tmp_path: Path, seed: int = 3) -> tuple[Path, Path]:
    """Prices with a known next-open-to-open move; scores that know it (h01) or are noise."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2022-12-01", "2026-03-31")
    price_rows, prediction_rows = [], []
    signal_dates = sessions[(sessions >= "2023-01-02") & (sessions < "2026-03-01")]
    for market, (suffix, count, index_symbol) in MARKETS.items():
        symbols = [f"{1101 + i}.{suffix}" for i in range(count)]
        opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, (count, len(sessions))), axis=1))
        closes = opens * np.exp(rng.normal(0, 0.008, opens.shape))
        index_open = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(sessions))))
        index_close = index_open * np.exp(rng.normal(0, 0.003, len(sessions)))
        for day, date in enumerate(sessions):
            for i, symbol in enumerate(symbols):
                price_rows.append(
                    (
                        date,
                        symbol,
                        "etf" if i == 0 else "stock",
                        opens[i, day],
                        closes[i, day],
                        closes[i, day],
                        market,
                    )
                )
            price_rows.append(
                (
                    date,
                    index_symbol,
                    "index",
                    index_open[day],
                    index_close[day],
                    index_close[day],
                    market,
                )
            )
        for date in signal_dates:
            day = sessions.get_loc(date)
            for i, symbol in enumerate(symbols):
                move = np.log(opens[i, day + 2] / opens[i, day + 1])
                scores = {"h01": move + rng.normal(0, 0.004)}
                for h in range(2, 15):
                    scores[f"h{h:02d}"] = rng.normal()
                prediction_rows.append(
                    {
                        "model": "good",
                        "symbol": symbol,
                        "market": market,
                        "signal_date": date,
                        **scores,
                    }
                )
    prices = pd.DataFrame(
        price_rows,
        columns=["timestamp", "symbol", "asset_type", "open", "close", "adjusted_close", "market"],
    )
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize("UTC")
    prices_path = tmp_path / "market.parquet"
    prices.to_parquet(prices_path, index=False)
    predictions = pd.DataFrame(prediction_rows)
    noise = predictions.copy()
    noise["model"] = "noise"
    for h in range(1, 15):
        noise[f"h{h:02d}"] = rng.normal(size=len(noise))
    predictions_path = tmp_path / "predictions.parquet"
    pd.concat([predictions, noise]).to_parquet(predictions_path, index=False)
    return predictions_path, prices_path


def test_inputs_moves_match_the_price_paths(tmp_path: Path) -> None:
    predictions_path, prices_path = _fixture(tmp_path)
    prices = load_prices(prices_path)
    inputs = build_inputs(load_predictions(predictions_path, "good"), prices)
    assert inputs.index.count == len(load_predictions(predictions_path, "good"))
    assert set(inputs.horizons) == set(range(1, 15))
    # The h01 score of the fixture is the realized open-to-open move plus small noise.
    finite = np.isfinite(inputs.move.growth)
    assert finite.mean() > 0.99
    corr = np.corrcoef(inputs.scores[finite, 0], inputs.move.growth[finite])[0, 1]
    assert corr > 0.9
    assert (inputs.index.is_etf.sum()) == len(inputs.dates) * 2  # one ETF per market per date
    assert np.isfinite(inputs.move.index_growth[finite]).all()


def test_dropout_rule_keeps_k_slots_and_rotates_at_most_drop_per_day(tmp_path: Path) -> None:
    predictions_path, prices_path = _fixture(tmp_path)
    inputs = build_inputs(load_predictions(predictions_path, "good"), load_prices(prices_path))
    groups = positions.date_groups(inputs.index, (0,))
    ranked = topk_dropout.engine.rankings(inputs.scores[:, 0], groups)
    path = topk_dropout.simulate_dropout(
        ranked,
        k=10,
        drop=2,
        symbols=inputs.index.symbol_codes,
        link=inputs.link,
        move=inputs.move,
        limit=0.095,
    )
    assert path.buy.shape == (len(groups) - 1, 10)
    sells = (path.sell_open >= 0).sum(axis=1)
    assert sells.max() <= 2 and (path.buy[0] >= 0).all()
    for j in range(path.held.shape[0]):
        rows = path.held[j][path.held[j] >= 0]
        symbols = inputs.index.symbol_codes[rows]
        assert len(set(symbols)) == len(symbols)
    free = np.zeros(inputs.index.count)
    gross = positions.replay(path, free, free)
    assert np.isfinite(gross.daily).all()
    with pytest.raises(ValueError):
        topk_dropout.simulate_dropout(
            ranked,
            k=10,
            drop=0,
            symbols=inputs.index.symbol_codes,
            link=inputs.link,
            move=inputs.move,
            limit=0.095,
        )


def test_run_selects_the_informative_horizon_and_judges_h_b(tmp_path: Path) -> None:
    predictions_path, prices_path = _fixture(tmp_path)
    config = json.loads(topk_dropout.DEFAULT_CONFIG.read_text())
    config["bootstrap"]["replicates"] = 200
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    report = topk_dropout.run(
        config_path,
        predictions_path,
        "good",
        prices_path,
        tmp_path / "out",
        compare_predictions=predictions_path,
        compare_model="noise",
    )
    assert report["selection"]["h"] == 1
    decision = report["decision"]
    assert decision["hypothesis"] == "H-B" and decision["passed"] is True
    assert decision["ir_not_below_baseline"] is True
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    assert set(grid["period"]) == {"P1", "P2", "P1+P2"}
    assert set(grid["market"]) == {"TWSE", "TPEX"} and set(grid["model"]) == {"good", "noise"}
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# Top-k dropout")
    assert (tmp_path / "out" / "daily_series.csv").is_file()


def test_holding_backtest_finds_the_intraday_signal(tmp_path: Path) -> None:
    from twbacktest import holding_walkforward
    from twbacktest.walkforward_data import build_holding_returns

    predictions_path, prices_path = _fixture(tmp_path)
    prices = load_prices(prices_path)
    predictions = load_predictions(predictions_path, "good")
    log_return, log_alpha = build_holding_returns(predictions, prices)
    assert (
        log_return.shape == (len(predictions), 14) and np.isfinite(log_return[:, 0]).mean() > 0.99
    )
    assert np.isfinite(log_alpha[:, 4]).mean() > 0.99
    # An "intraday" model that knows the next open-to-close move must win at h = 1.
    intraday = predictions.copy()
    rng = np.random.default_rng(9)
    for h in range(1, 15):
        intraday[f"h{h:02d}"] = rng.normal(size=len(intraday))
    intraday["h01"] = np.nan_to_num(log_return[:, 0]) + rng.normal(0, 0.002, len(intraday))
    intraday["model"] = "intraday"
    both = tmp_path / "both.parquet"
    pd.concat([pd.read_parquet(predictions_path), intraday]).to_parquet(both, index=False)
    config = json.loads(holding_walkforward.DEFAULT_CONFIG.read_text())
    config["bootstrap"]["replicates"] = 100
    config["holding_days"] = [1, 5]
    config["top_k"] = [10]
    config_path = tmp_path / "holding.json"
    config_path.write_text(json.dumps(config))
    report = holding_walkforward.run(config_path, both, "intraday", prices_path, tmp_path / "hold")
    assert report["selection"]["h"] == 1 and report["selection"]["k"] == 10
    assert report["selection"]["P1+P2"]["gross_excess_mean"] > 0
    grid = pd.read_csv(tmp_path / "hold" / "grid.csv")
    assert set(grid["period"]) == {"P1", "P2", "P1+P2"} and set(grid["market"]) == {
        "TWSE",
        "TPEX",
        "TW",
    }
