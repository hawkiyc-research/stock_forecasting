import json

import numpy as np
import pandas as pd
import pytest

from twbacktest import data, stage1

ALPHA_ID = "audit/baselines/alpha"
DIRECTION_ID = "audit/baselines/direction"
RUN = "run-fixture-alpha"
CHECKPOINT = "checkpoint-000010"
DIGESTS = {"validation": "digest-validation", "test": "digest-test"}
STOCKS = {"TWSE": 12, "TPEX": 8, "US": 3}


def write_split(root, split, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-02", periods=40)
    rows = [(f"{market}-{number}", f"{date.date()}T00:00:00+00:00", market)
            for date in dates for market, count in STOCKS.items() for number in range(count)]
    metadata = np.array(
        [(symbol, date, market, "etf" if symbol.endswith("-0") else "stock", "fixture")
         for symbol, date, market in rows],
        dtype=[("symbol", "<U48"), ("date", "<U40"), ("market", "<U24"),
               ("asset_type", "<U24"), ("provider", "<U32")],
    )
    count = len(rows)
    alpha = rng.normal(0, 0.02, size=(count, 14)).astype(np.float32)
    bench = np.repeat(rng.normal(0, 0.01, size=(len(dates), 1, 14)), sum(STOCKS.values()), axis=1)
    absolute = (alpha + bench.reshape(count, 14)).astype(np.float32)
    good = (alpha + rng.normal(0, 0.01, size=alpha.shape)).astype(np.float32)
    noise = rng.normal(size=alpha.shape).astype(np.float32)
    logits = rng.normal(size=(count, 14, 3)).astype(np.float32)
    for baseline, arrays in (
        (ALPHA_ID, {"inputs/{s}/metadata.npy": metadata, "inputs/{s}/targets.npy": alpha,
                    "jobs/gru-42/{s}/predictions.npy": noise}),
        (DIRECTION_ID, {"inputs/{s}/absolute_targets.npy": absolute,
                        "jobs/gru-42/{s}/predictions.npy": logits}),
    ):
        for pattern, array in arrays.items():
            path = root / baseline / pattern.format(s=split)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
    export = root / "diagnostics/signal-predictions" / RUN / CHECKPOINT / "export-1"
    (export / split).mkdir(parents=True, exist_ok=True)
    np.save(export / split / "predictions.npy", good)
    return count


@pytest.fixture()
def artifacts(tmp_path):
    root = tmp_path / "artifacts"
    for split, seed in (("validation", 1), ("test", 2)):
        write_split(root, split, seed)
    membership = {"validation_membership": {"ordered_symbol_dates_sha256": DIGESTS["validation"]},
                  "evaluation_membership": {"ordered_symbol_dates_sha256": DIGESTS["test"]}}
    for baseline in (ALPHA_ID, DIRECTION_ID):
        (root / baseline / "complete.json").write_text(json.dumps(membership))
    export = root / "diagnostics/signal-predictions" / RUN / CHECKPOINT / "export-1"
    (export / "status.json").write_text(json.dumps({"state": "complete"}))
    (export / "export.json").write_text(json.dumps({"splits": {
        split: {"sample_membership": {"ordered_symbol_dates_sha256": digest}}
        for split, digest in DIGESTS.items()
    }}))
    return root


def config_file(tmp_path):
    config = json.loads((stage1.DEFAULT_CONFIG).read_text())
    config["inputs"] = {"alpha_baseline": ALPHA_ID, "direction_baseline": DIRECTION_ID,
                        "predictions_root": "diagnostics/signal-predictions"}
    config["signals"] = {
        "GOOD": {"kind": "alpha", "members": [[RUN, CHECKPOINT]]},
        "NOISE": {"kind": "alpha", "baseline": "gru-42"},
        "DIRECTION": {"kind": "direction", "baseline": "gru-42"},
    }
    config["holding_days"], config["top_k"] = [1, 2], [2, 4]
    config["costs"].update(discounts=[0.6], slippage_multipliers=[1],
                           turnover_modes=["conservative", "optimistic"])
    config["bootstrap"]["replicates"] = 50
    path = tmp_path / "stage1.json"
    path.write_text(json.dumps(config))
    return path


def test_split_index_keeps_taiwan_rows_with_stable_symbol_codes(artifacts):
    index = data.build_split_index(artifacts / ALPHA_ID, "validation")
    assert index.count == 40 * 20 and len(index.dates) == 40
    assert set(np.unique(index.market_codes)) == {0, 1}
    assert index.is_etf.sum() == 40 * 2
    first_day = index.symbol_codes[index.date_codes == 0]
    last_day = index.symbol_codes[index.date_codes == 39]
    assert np.array_equal(np.sort(first_day), np.sort(last_day))


def test_exports_with_another_row_order_are_rejected(artifacts):
    export = data.export_directory(artifacts / "diagnostics/signal-predictions", RUN, CHECKPOINT)
    data.verify_export(export, "test", DIGESTS["test"])
    with pytest.raises(ValueError, match="row order"):
        data.verify_export(export, "test", "another-digest")


def test_end_to_end_selection_and_gate(artifacts, tmp_path):
    config = config_file(tmp_path)
    report = stage1.run(config, artifacts, tmp_path / "serial", workers=1)
    assert (tmp_path / "serial" / "summary.md").read_text().startswith("# 台股回測第一階段摘要")
    grid = pd.read_csv(tmp_path / "serial" / "grid.csv")
    # 2 splits x 3 signals x 3 markets x 2 h x 2 K x 2 scenarios x 2 versions
    assert len(grid) == report["configurations"] == 2 * 3 * 3 * 2 * 2 * 2 * 2
    for market in ("TWSE", "TPEX", "TW"):
        item = report["selection"][market]
        assert item["selected"]["signal"] == "GOOD"
        assert item["validation"]["kind"] == "alpha" and item["test"] is not None
    assert set(report["gate_passed_markets"]) <= {"TWSE", "TPEX", "TW"}
    assert report["data"]["validation"]["dates"] == 40
    with pytest.raises(FileExistsError):
        stage1.run(config, artifacts, tmp_path / "serial", workers=1)


def test_process_pool_matches_the_serial_run(artifacts, tmp_path):
    config = config_file(tmp_path)
    serial = stage1.run(config, artifacts, tmp_path / "serial", workers=1)
    try:
        parallel = stage1.run(config, artifacts, tmp_path / "parallel", workers=2)
    except PermissionError:
        pytest.skip("Process pools are not permitted in this sandbox")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "serial" / "grid.csv"),
                                  pd.read_csv(tmp_path / "parallel" / "grid.csv"))
    assert parallel["selection"] == serial["selection"]
