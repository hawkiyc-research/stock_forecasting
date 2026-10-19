import json

import numpy as np
import pandas as pd
import pytest
from conftest import DATES, DIRECTION_ID, fixture_config

from twbacktest import stage1b


def config_file(tmp_path):
    costs = json.loads(stage1b.DEFAULT_CONFIG.read_text())["costs"]
    costs.update(discounts=[0.6], slippage_multipliers=[1])
    return fixture_config(stage1b.DEFAULT_CONFIG, tmp_path, "stage1b.json",
                          score_horizons=[1, 3], top_k=[2, 4], buffer_multiples=[1, 2],
                          costs=costs)


def test_end_to_end_selection_gate_and_daily_series(price_artifacts, tmp_path):
    config = config_file(tmp_path)
    output = tmp_path / "run"
    report = stage1b.run(config, price_artifacts, output, workers=1)
    assert (output / "summary.md").read_text().startswith("# 台股回測第一階段 1b 摘要")
    grid = pd.read_csv(output / "grid.csv")
    # 2 splits x 3 signals x 3 markets x 2 s x 2 K x 2 b x 1 scenario x 2 versions
    assert len(grid) == report["configurations"] == 2 * 3 * 3 * 2 * 2 * 2 * 2
    for split in ("validation", "test"):
        identity = report["data"][split]["identity"]
        assert identity["stock_fraction"] == 1.0 and identity["index_fraction"] == 1.0
        assert identity["missing_next_rows"] == 0
    for market in ("TWSE", "TPEX", "TW"):
        item = report["selection"][market]
        assert item["selected"]["signal"] == "GOOD" and item["candidates"] == 24
        assert item["test"] is not None
    daily = pd.read_csv(output / "daily.csv")
    assert set(daily["market"]) == {"TWSE", "TPEX", "TW"}
    assert len(daily) == 3 * 2 * (DATES - 1)
    with pytest.raises(FileExistsError):
        stage1b.run(config, price_artifacts, output, workers=1)


def test_broken_identity_stops_the_run(price_artifacts, tmp_path):
    path = price_artifacts / DIRECTION_ID / "inputs" / "test" / "absolute_targets.npy"
    absolute = np.load(path)
    absolute[::7, 4] += 0.01
    np.save(path, absolute)
    with pytest.raises(ValueError, match="identity"):
        stage1b.run(config_file(tmp_path), price_artifacts, tmp_path / "run", workers=1)
    assert not (tmp_path / "run").exists()


def test_process_pool_matches_the_serial_run(price_artifacts, tmp_path):
    config = config_file(tmp_path)
    serial = stage1b.run(config, price_artifacts, tmp_path / "serial", workers=1)
    try:
        parallel = stage1b.run(config, price_artifacts, tmp_path / "parallel", workers=2)
    except PermissionError:
        pytest.skip("Process pools are not permitted in this sandbox")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "serial" / "grid.csv"),
                                  pd.read_csv(tmp_path / "parallel" / "grid.csv"))
    assert parallel["selection"] == serial["selection"]
