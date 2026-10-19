import json

import pandas as pd
import pytest
from conftest import fixture_config

from twbacktest import holding_eval


def config_file(tmp_path):
    return fixture_config(holding_eval.DEFAULT_CONFIG, tmp_path, "holding_eval.json")


def test_end_to_end_tables_watch_list_and_summary(price_artifacts, tmp_path):
    config = config_file(tmp_path)
    output = tmp_path / "run"
    report = holding_eval.run(config, price_artifacts, output, workers=1)
    assert (output / "summary.md").read_text().startswith("# 與持有方式對齊的評估摘要")
    ranking = pd.read_csv(output / "ranking.csv")
    portfolio = pd.read_csv(output / "portfolio.csv")
    # 2 splits x 3 signals x {TWSE, TPEX} x 3 horizons; the portfolio adds TW.
    assert len(ranking) == report["ranking_rows"] == 2 * 3 * 2 * 3
    assert len(portfolio) == report["portfolio_rows"] == 2 * 3 * 3 * 3
    for split in ("validation", "test"):
        assert report["data"][split]["identity"]["stock_fraction"] == 1.0
    validation = ranking[(ranking["split"] == "validation") & (ranking["h"] == 3)
                         & (ranking["market"] == "TWSE")].set_index("signal")
    # GOOD knows the first day of the window, so it ranks the 3-day return better than noise.
    assert validation.loc["GOOD", "ic_oo_mean"] > validation.loc["NOISE", "ic_oo_mean"]
    assert (ranking["label_coverage"] > 0.5).all()
    rule = json.loads(config.read_text())["watch_rule"]
    for item in report["watch_list"]:
        assert item["validation_top_low"] > rule["lower_bound_above"] and item["h"] in (5, 10)
    with pytest.raises(FileExistsError):
        holding_eval.run(config, price_artifacts, output, workers=1)


def test_process_pool_matches_the_serial_run(price_artifacts, tmp_path):
    config = config_file(tmp_path)
    serial = holding_eval.run(config, price_artifacts, tmp_path / "serial", workers=1)
    try:
        parallel = holding_eval.run(config, price_artifacts, tmp_path / "parallel", workers=2)
    except PermissionError:
        pytest.skip("Process pools are not permitted in this sandbox")
    for name in ("ranking.csv", "portfolio.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "serial" / name),
                                      pd.read_csv(tmp_path / "parallel" / name))
    assert parallel["watch_list"] == serial["watch_list"]
