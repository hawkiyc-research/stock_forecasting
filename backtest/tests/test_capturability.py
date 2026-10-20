"""Capturability part A (protocol section 13.9) on the price fixture."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_config

from twbacktest import capturability, data


def _segment_inputs(config, artifacts, index):
    symbols = np.asarray(index.symbol_names)
    rng = np.random.default_rng(3)
    futures = pd.DataFrame(
        [{"date": day, "symbol": symbol, "liquid": symbol.endswith(("-1", "-2"))}
         for day in index.dates for symbol in symbols
         if symbol.split("-")[1] in {"1", "2", "3", "4", "5", "6"}]
    )
    liquidity = pd.DataFrame([{"date": day, "symbol": symbol, "mean_value": rng.uniform(1, 9)}
                              for day in index.dates for symbol in symbols])
    return futures, liquidity, {"fixture": True}


def config_file(tmp_path):
    return fixture_config(capturability.DEFAULT_CONFIG, tmp_path, "capturability.json")


def test_segment_masks_split_futures_and_liquidity_within_dates(price_artifacts, tmp_path):
    config = json.loads(config_file(tmp_path).read_text())
    index = data.build_split_index(price_artifacts / config["inputs"]["alpha_baseline"],
                                   "validation")
    futures, liquidity, _ = _segment_inputs(config, price_artifacts, index)
    masks = capturability.segment_masks(index, futures, liquidity, 3)
    assert (masks["futures"] ^ masks["no_futures"]).all()
    assert not (masks["liquid_futures"] & ~masks["futures"]).any()
    total = masks["liquidity_low"] | masks["liquidity_mid"] | masks["liquidity_high"]
    assert total.all()
    first = (index.date_codes == 0) & (index.market_codes == 0)  # 12 TWSE rows on day one
    assert [int((masks[name] & first).sum()) for name in
            ("liquidity_low", "liquidity_mid", "liquidity_high")] == [4, 4, 4]
    with pytest.raises(ValueError):
        capturability.segment_masks(index, futures, liquidity, 4)


def test_decile_curve_is_monotone_for_a_perfect_score():
    simple = np.linspace(-0.05, 0.05, 20)
    curve = capturability.decile_curve(simple.copy(), simple, (np.arange(20),), 10)
    assert np.all(np.diff(curve[0]) < 0) and abs(curve[0].sum()) < 1e-12


def test_end_to_end_tables_and_summary(price_artifacts, tmp_path):
    config = config_file(tmp_path)
    output = tmp_path / "run"
    report = capturability.run(config, price_artifacts, output, workers=1,
                               segment_inputs=_segment_inputs)
    segments = pd.read_csv(output / "segments.csv")
    deciles = pd.read_csv(output / "deciles.csv")
    decay = pd.read_csv(output / "decay.csv")
    signals = 3
    assert len(segments) == signals * 2 * len(capturability.SEGMENTS) * 2
    assert len(deciles) == signals * 2 * 2 * 2 * 10
    assert len(decay) == signals * 2 * 13
    all_twse = segments[(segments["segment"] == "all") & (segments["market"] == "TWSE")
                        & (segments["h"] == 5)].set_index("signal")
    assert all_twse.loc["GOOD", "ic_mean"] > all_twse.loc["NOISE", "ic_mean"]
    assert report["stocks_per_date"]["TWSE"]["futures"] == 6
    assert (output / "summary.md").read_text().startswith("# 可捕捉性分析 A")
    with pytest.raises(FileExistsError):
        capturability.run(config, price_artifacts, output, workers=1,
                          segment_inputs=_segment_inputs)


def test_only_validation_may_be_used(price_artifacts, tmp_path):
    path = fixture_config(capturability.DEFAULT_CONFIG, tmp_path, "bad.json",
                          splits=["validation", "test"])
    with pytest.raises(ValueError, match="validation only"):
        capturability.run(path, price_artifacts, tmp_path / "out", workers=1,
                          segment_inputs=_segment_inputs)
