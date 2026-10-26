"""Execution-uncertainty layer of the short book (protocol 13.19 supplement 4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_scenario_suite import _config, _tiny_market
from tests.test_tw_strategy import _fixture
from twbacktest import borrow_execution as be
from twbacktest import chip_data, positions
from twbacktest import scenario_suite as ss
from twbacktest.tw_strategy import simulate_band
from twbacktest.walkforward_data import load_predictions


def _write_curated(root: Path, rows: pd.DataFrame, dataset: str) -> None:
    """One partition per year with the _SUCCESS marker, like tw_chip_data publishes."""
    for year, part in rows.groupby(rows["date"].dt.year):
        folder = root / "curated" / dataset / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        part.to_parquet(folder / "part-0.parquet", index=False)
        (folder / "_SUCCESS.json").write_text(json.dumps({"sha256": f"{dataset}{year}"}))


def _chip_root(tmp_path: Path, predictions: pd.DataFrame, *, ineligible: set[str] = frozenset(),
               quota_shares: float = 1e9, drop_dates: int = 0) -> Path:
    """A synthetic curated layer covering every prediction row (minus ``ineligible`` symbols
    and the last ``drop_dates`` signal dates, which are then carried forward)."""
    root = tmp_path / "chip"
    base = predictions[["symbol", "market", "signal_date"]].drop_duplicates()
    base = base[~base["symbol"].isin(ineligible)]
    dates = sorted(base["signal_date"].unique())
    if drop_dates:
        base = base[~base["signal_date"].isin(dates[-drop_dates:])]
    code = base["symbol"].str.split(".").str[0]
    stamp = pd.Timestamp("2000-01-01", tz="Asia/Taipei")
    eligibility = pd.DataFrame({
        "date": base["signal_date"].to_numpy(), "symbol": code.to_numpy(),
        "market": base["market"].to_numpy(), "name": "x",
        "margin_eligible": True, "short_suspended": False, "sbl_suspended": False,
        "limit_down_block": False, "sbl_eligible": True, "available_at": stamp,
    })
    sbl = pd.DataFrame({
        "date": base["signal_date"].to_numpy(), "symbol": code.to_numpy(),
        "market": base["market"].to_numpy(), "sbl_next_quota": int(quota_shares),
        "short_limit": int(quota_shares), "short_balance": 0, "available_at": stamp,
    })
    _write_curated(root, eligibility, "eligibility")
    _write_curated(root, sbl, "sbl")
    return root


def _exec_block(**overrides) -> dict:
    block = {
        "borrow_list": {"institution": "sbl", "retail": "margin"},
        "borrow_failure": {"institution": [0.02, 0.10, 0.25], "retail": [0.05, 0.20, 0.50]},
        "failure_multipliers": [0, 1, 2], "primary_failure_multiplier": 1,
        "recall_probabilities": [0.0, 0.005], "primary_recall_probability": 0.0,
        "borrow_fee_tiers": {"institution": [0.015, 0.03, 0.06]},
        "quota_cap": True, "replicates": 2, "seed": 11,
    }
    block.update(overrides)
    return block


# --------------------------------------------------------------------------- engine


def _two_stock_band(**kwargs):
    """A, B ranked for two intervals; one slot; every open flat (growth 0)."""
    symbols = np.array([0, 1, 0, 1, 0, 1])
    link = positions.Links(next_row=np.array([2, 3, 4, 5, -1, -1]),
                           prev_row=np.array([-1, -1, 0, 1, 2, 3]))
    move = positions.Moves(growth=np.zeros(6), fallback=np.zeros(6), gap=np.zeros(6),
                           index_growth=np.zeros(6))
    ranked = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    return simulate_band(ranked, k=1, buffer=2, max_buys=1, symbols=symbols, link=link,
                         move=move, limit=0.095, **kwargs)


def test_ineligible_candidates_are_skipped_and_counted():
    result = _two_stock_band(eligible=np.array([False, True, False, True, False, True]))
    assert result.path.buy[0, 0] == 1  # B instead of the ineligible A
    assert result.ineligible_skipped == 1 and result.borrow_failed == 0


def test_certain_borrow_failure_leaves_the_book_empty():
    result = _two_stock_band(fail_prob=np.ones(6), rng=np.random.default_rng(0))
    assert (result.path.buy < 0).all() and (result.path.held < 0).all()
    assert result.borrow_failed == result.buy_attempts > 0


def test_random_arguments_need_a_generator():
    with pytest.raises(ValueError, match="random generator"):
        _two_stock_band(fail_prob=np.full(6, 0.5))


def test_recall_forces_the_exit_at_the_next_open():
    result = _two_stock_band(recall_prob=1.0, rng=np.random.default_rng(0))
    assert result.path.buy[0, 0] == 0 and result.recalls >= 1
    assert result.path.sell_open[1, 0] == 2  # A's next row is sold at the second open
    assert result.path.buy[1, 0] == 3  # and the slot refills with B's row of that date


def test_value_cap_scales_the_position_under_the_quota():
    cap = np.array([2e5, np.nan, 2e5, np.nan, 2e5, np.nan])  # A: NT$200k of a NT$1M slot
    result = _two_stock_band(slot_capital=1e6, lot_cost=np.full(6, 1e4), value_cap=cap,
                             cap_mode="scale")
    assert result.path.buy[0, 0] == 0 and result.fill_fraction[0] == pytest.approx(0.2)
    assert result.quota_capped == 1 and result.capped_buys == 1
    assert result.deployed[0] == pytest.approx(0.2)
    skipped = _two_stock_band(slot_capital=1e6, lot_cost=np.full(6, 1e4), value_cap=cap,
                              cap_mode="skip")
    assert skipped.path.buy[0, 0] == -1 and skipped.liquidity_blocked == 4  # B's cap is NaN


def test_tiered_borrow_fee_charges_per_row():
    path = positions.Path(
        buy=np.array([[0], [-1]]), sell_open=np.full((2, 1), -1), sell_close=np.full((2, 1), -1),
        held=np.array([[0], [2]]), growth=np.zeros((2, 1)), final=np.array([2]),
        holding_periods=np.array([2]), blocked_buys=0, blocked_sells=0, missing_exits=0,
    )
    fee = np.array([0.06, 0.0, 0.015]) / 252
    charged = ss.with_borrow_fee(path, fee)
    assert charged.growth[0, 0] == pytest.approx(math.log1p(-0.06 / 252))
    assert charged.growth[1, 0] == pytest.approx(math.log1p(-0.015 / 252))


# --------------------------------------------------------------------------- specs and data


def test_execution_specs_primary_first_then_grid():
    config = {"execution": _exec_block()}
    specs = be.execution_specs(config, "institution")
    assert [s.key for s in specs] == [(1.0, 0.0), (0.0, 0.0), (0.0, 0.005), (1.0, 0.005),
                                      (2.0, 0.0), (2.0, 0.005)]
    assert specs[0].primary and not any(s.primary for s in specs[1:])
    tercile = np.array([0, 1, 2, -1])
    assert specs[0].fail_prob(tercile) == pytest.approx([0.02, 0.10, 0.25, 0.0])
    assert specs[4].fail_prob(tercile) == pytest.approx([0.04, 0.20, 0.50, 0.0])
    assert specs[0].fee_daily(tercile, 0.02) == pytest.approx(
        np.array([0.015, 0.03, 0.06, 0.02]) / 252)
    retail = be.execution_specs(config, "retail")[0]
    assert retail.borrow_list == "margin" and retail.fee_daily(tercile, 0.0) == 0.0
    assert specs[0].replicate_seeds() == [11, 12]


def test_carry_dates_maps_missing_lists_to_the_previous_date():
    table = np.array(["2025-01-06", "2025-01-07"], dtype="datetime64[ns]")
    signal = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-08", "2025-01-08"]))
    mapped, carried = be._carry_dates(signal, table)
    assert pd.isna(mapped.iloc[0])  # before the first list: nothing to carry
    assert mapped.iloc[1] == pd.Timestamp("2025-01-06")
    assert mapped.iloc[2] == pd.Timestamp("2025-01-07") and carried == 1


def test_load_borrow_data_joins_lists_quota_and_terciles(tmp_path):
    data = _tiny_market(tmp_path / "m")
    frame = load_predictions(tmp_path / "m" / "s.parquet", "m")
    root = _chip_root(tmp_path, frame, ineligible={"C.TW"}, quota_shares=500, drop_dates=1)
    borrow = be.load_borrow_data(root, frame, data.inputs, data.groups)
    symbols = frame["symbol"].to_numpy()
    assert not borrow.sbl_eligible[symbols == "C.TW"].any()
    assert borrow.sbl_eligible[symbols != "C.TW"].all()
    assert borrow.margin_eligible[symbols == "A.TW"].all()
    # 500 shares x NT$100 open = NT$50,000 of lending quota.
    assert np.nanmax(borrow.sbl_quota_value) == pytest.approx(5e4)
    assert borrow.carried_dates == 2  # the dropped date, in both tables
    assert borrow.list_matched_share == pytest.approx(0.75)  # C's rows have no list row
    assert set(borrow.tercile[borrow.tercile >= 0]) <= {0, 1, 2}
    assert len(borrow.digests) == 2
    summary = be.summarize_borrow(borrow, data.groups)
    assert summary["sbl_eligible_share"] == pytest.approx(0.75)
    missing = tmp_path / "nothing"
    with pytest.raises(FileNotFoundError):
        chip_data.partition_files(missing, "eligibility")


# --------------------------------------------------------------------------- books


def _book(data, config, profile, execution, cache=None):
    return ss.multi_day_book(data, config, profile, ss.Capital("fractional", None, None),
                             h=5, k=1, buffer=2, participation_cap=0.05,
                             execution=execution, cache=cache)


def test_zero_probabilities_reproduce_the_plain_short_book(tmp_path):
    config = json.loads(ss.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["execution"] = _exec_block()
    profile = ss._primary_profile(
        json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text()), "institution")
    data = _tiny_market(tmp_path / "m")
    frame = load_predictions(tmp_path / "m" / "s.parquet", "m")
    from dataclasses import replace
    data = replace(data, borrow=be.load_borrow_data(
        _chip_root(tmp_path, frame), frame, data.inputs, data.groups))
    plain = _book(data, config, profile, None)
    flat = ss.ExecutionSpec("institution", "sbl", (0.0, 0.0, 0.0), 0.0, 0.0, None, True, 3, 11,
                            True)
    same = _book(data, config, profile, flat)
    np.testing.assert_allclose(same.legs["short"], plain.legs["short"])
    np.testing.assert_allclose(same.net, plain.net)
    assert same.info["replicates"] == 3 and same.info["borrow_failed_share"] == 0.0
    assert same.info["short_net_annual_rep_min"] == same.info["short_net_annual_rep_max"]
    tiered = replace(flat, fee_tiers=(0.015, 0.03, 0.06))
    fees = _book(data, config, profile, tiered)
    assert fees.legs["short"].sum() != pytest.approx(plain.legs["short"].sum())
    certain = replace(flat, failure_probs=(1.0, 1.0, 1.0), failure_multiplier=1.0)
    empty = _book(data, config, profile, certain)
    assert np.all(empty.legs["short"] == 0.0) and empty.info["borrow_failed_share"] == 1.0


def test_run_with_the_execution_layer(tmp_path):
    predictions, prices, artifacts = _fixture(tmp_path)
    frame = load_predictions(predictions, "good")
    root = _chip_root(tmp_path, frame, ineligible={"1103.TW"})
    cost = json.loads((ss.CONFIG_DIR / "cost_profiles.json").read_text(encoding="utf-8"))
    cost["profiles"]["institution"]["capital_scenarios"] = [{"name": "20M", "twd": 2e7}]
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(json.dumps(cost))
    config = _config(
        tmp_path, cost_profiles=str(cost_path), scenarios=["md_ls", "md_ls_tx"],
        construction={"score_variants": ["v0"], "leg_matching": ["none"]},
        execution=_exec_block(failure_multipliers=[0, 1], recall_probabilities=[0.0, 0.5]),
    )
    with pytest.raises(ValueError, match="chip-data-root"):
        ss.run(config, [predictions], ["good"], prices, tmp_path / "no-root",
               artifacts=artifacts)
    report = ss.run(config, [predictions], ["good"], prices, tmp_path / "out",
                    artifacts=artifacts, chip_data_root=root)
    grid = pd.read_csv(tmp_path / "out" / "grid.csv")
    final = grid[(grid["stage"] == "final") & (grid["scenario"] == "md_ls")]
    assert set(final["failure_multiplier"].dropna()) == {1.0}
    sens = grid[grid["stage"] == "execution"]
    assert set(zip(sens["failure_multiplier"], sens["recall_prob"], strict=True)) == {
        (0.0, 0.0), (0.0, 0.5), (1.0, 0.5)}
    assert (sens["capital"] == "fractional").all() and sens["primary_slippage"].all()
    twse = final[(final["market"] == "TWSE") & (final["profile"] == "institution")
                 & (final["capital"] == "fractional") & (final["primary_slippage"])]
    assert (twse["ineligible_share"] > 0).all() and (twse["borrow_failed_share"] > 0).all()
    assert twse["replicates"].iloc[0] == 2
    recalled = sens[(sens["recall_prob"] == 0.5) & (sens["market"] == "TWSE")
                    & (sens["scenario"] == "md_ls")]
    assert (recalled["recalls"] > 0).all()
    capital = final[(final["capital"] == "20M") & (final["market"] == "TWSE")]
    assert (capital["short_deployed_share"] <= 1.0).all()
    coverage = report["execution"]["coverage"]["good"]["TWSE"]
    assert 0 < coverage["sbl_eligible_share"] < 1 and coverage["carried_dates"] == 0
    assert report["execution"]["points"]["institution"][0]["primary"]
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "Execution layer" in summary
