"""Protocol 13.18 G1 / H-H scoring on synthetic runs: a perfect ranker paired against a weak one
must pass both decisions, the paired tables must cover every universe and segment, and a noisy
run must not pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
from test_exposure_diagnostics import _taifex_root

from twbacktest import chip_increment as inc


def test_grouped_spearman_and_quartiles_match_direct_formulas():
    rng = np.random.default_rng(1)
    rows = []
    for date in pd.bdate_range("2025-01-02", periods=3):
        for group in ("TWSE", "TPEX"):
            score = rng.normal(size=30)
            rows.append(
                pd.DataFrame(
                    {
                        "signal_date": date,
                        "group": group,
                        "h10": score,
                        "residual_h10": score + rng.normal(size=30),
                        "alpha_h10": rng.normal(size=30),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    ic = inc.grouped_spearman(frame, "h10", "residual_h10")
    first = frame[(frame["signal_date"] == frame["signal_date"].min()) & (frame["group"] == "TPEX")]
    expected = first["h10"].rank().corr(first["residual_h10"].rank())
    assert abs(ic.loc[(frame["signal_date"].min(), "TPEX")] - expected) < 1e-12
    universe_mean = frame.groupby(inc.KEYS)["alpha_h10"].mean()
    quartiles = inc.quartile_statistics(
        frame, "h10", {"residual": "residual_h10", "raw": "alpha_h10"}, universe_mean
    )
    k = 8  # ceil(30 x 0.25)
    top = first.nlargest(k, "h10")["residual_h10"].mean()
    bottom = first.nsmallest(k, "h10")["residual_h10"].mean()
    key = (frame["signal_date"].min(), "TPEX")
    assert abs(quartiles.loc[key, "spread_residual"] - (top - bottom)) < 1e-12
    excess = first.nsmallest(k, "h10")["alpha_h10"].mean() - first["alpha_h10"].mean()
    assert abs(quartiles.loc[key, "bottom_excess"] - excess) < 1e-12
    # Groups below MIN_NAMES are NaN, not dropped silently.
    small = frame.groupby("group").head(10)
    assert inc.grouped_spearman(small, "h10", "residual_h10").isna().all()


def _synthetic(tmp_path):
    rng = np.random.default_rng(3)
    sessions = pd.bdate_range("2024-10-01", "2025-10-31")
    signal_dates = sessions[(sessions >= "2025-01-02") & (sessions < "2025-09-01")]
    n = 120
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
    horizons = sorted(set(inc.IC_HORIZONS) | set(inc.SPREAD_HORIZONS))
    runs = {"F0": 3.0, "F1": 30.0, "F2": 0.0}  # noise scale added to the true label
    targets, predictions = [], {label: [] for label in runs}
    for date in signal_dates:
        day = sessions.get_loc(date)
        labels = {h: np.log(closes[:, day + h] / opens[:, day + 1]) for h in horizons}
        for i, symbol in enumerate(symbols):
            target = {"symbol": symbol, "signal_date": date}
            for h in horizons:
                target[f"alpha_h{h:02d}"] = labels[h][i]
                target[f"residual_h{h:02d}"] = labels[h][i] - labels[h].mean()
            targets.append(target)
        for label, noise in runs.items():
            for i, symbol in enumerate(symbols):
                scores = {f"h{h:02d}": 0.0 for h in range(1, 15)}
                for h in horizons:
                    scores[f"h{h:02d}"] = labels[h][i] + noise * labels[h].std() * rng.normal()
                predictions[label].append(
                    {
                        "model": "m",
                        "symbol": symbol,
                        "market": "TWSE",
                        "signal_date": date,
                        **scores,
                    }
                )
    pd.DataFrame(targets).to_parquet(tmp_path / "targets.parquet", index=False)
    for label in runs:
        pd.DataFrame(predictions[label]).to_parquet(tmp_path / f"{label}.parquet", index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "symbol": symbols,
            "market": "TWSE",
            "industry_code": ["24" if i % 2 else "17" for i in range(n)],
            "issued_shares": 1_000_000,
            "par_value_twd": 10.0,
            "snapshot_date": "2026-10-22",
        }
    ).to_csv(reference, index=False)
    codes = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}F" for i in range(30)]
    root = tmp_path / "taifex"
    _taifex_root(root, sessions, dict.fromkeys(codes, 500))
    return reference, root


def test_run_scores_pairs_universes_segments_and_decisions(tmp_path):
    reference, taifex_root = _synthetic(tmp_path)
    report = inc.run(
        [tmp_path / f"{label}.parquet" for label in ("F0", "F1", "F2")],
        [tmp_path / "targets.parquet"] * 3,
        ["m", "m", "m"],
        ["F0", "F1", "F2"],
        tmp_path / "market.parquet",
        taifex_root,
        reference,
        tmp_path / "out",
        baseline="F0",
        min_volume=100,
        lookback=20,
    )
    levels = pd.read_csv(tmp_path / "out" / "grid.csv")
    paired = pd.read_csv(tmp_path / "out" / "paired.csv")
    assert set(levels["universe"]) == set(inc.UNIVERSES)
    assert set(levels["segment"]) == set(inc.SEGMENTS)
    assert set(paired["model"]) == {"F1", "F2"} and set(paired["baseline"]) == {"F0"}
    assert {
        "ic_residual",
        "ic_raw",
        "spread_residual",
        "spread_raw",
        "bottom_excess",
        "ls_industry",
    } <= set(levels["metric"])
    verdicts = {v["model"]: v for v in report["verdicts"]}
    f2, f1 = verdicts["F2"], verdicts["F1"]
    assert f2["H-H"]["condition_i"] and f2["H-H"]["condition_ii"] and f2["H-H"]["pass"]
    assert f2["G1"]["liquidity"]["pass"] and f2["G1"]["pass"]
    assert not f1["H-H"]["pass"] and not f1["G1"]["pass"]
    # The paired holding-cell difference of the perfect ranker is positive with a tight interval.
    row = paired[
        (paired["model"] == "F2")
        & (paired["universe"] == "all")
        & (paired["segment"] == "all")
        & (paired["metric"] == "ic_residual")
        & (paired["scope"] == "holding")
        & (paired["period"] == "P1+P2")
    ].iloc[0]
    assert row["mean"] > 0.3 and row["ci95_low"] > 0
    # Winners are a fifth of the group (120 x 0.2 = 24 >= MIN_NAMES), so both segments score;
    # in the liquidity universe the perfect ranker keeps its edge inside the winners too.
    winners = levels[
        (levels["segment"] == "winners")
        & (levels["metric"] == "ic_residual")
        & (levels["model"] == "F2")
        & (levels["universe"] == "liquidity")
        & (levels["scope"] == "holding")
        & (levels["period"] == "P1+P2")
    ]
    assert len(winners) == 1 and winners.iloc[0]["mean"] > 0.9
    assert not levels[(levels["segment"] == "rest") & (levels["metric"] == "ic_residual")].empty
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "通過 / pass" in summary and "未通過 / fail" in summary
