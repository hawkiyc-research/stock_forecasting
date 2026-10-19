import json

import numpy as np
import pandas as pd
import pytest

ALPHA_ID = "audit/baselines/alpha"
DIRECTION_ID = "audit/baselines/direction"
RUN = "run-fixture-alpha"
CHECKPOINT = "checkpoint-000010"
DIGESTS = {"validation": "digest-validation", "test": "digest-test"}
STOCKS = {"TWSE": 12, "TPEX": 8, "US": 3}
DATES = 40


def write_price_split(root, split, seed):
    """Targets from consistent open/close paths, so the open-to-open identity holds."""
    rng = np.random.default_rng(seed)
    sessions = DATES + 16
    symbols = [(f"{market}-{number}", market) for market, count in STOCKS.items()
               for number in range(count)]
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, size=(len(symbols), sessions)), axis=1))
    closes = opens * np.exp(rng.normal(0, 0.01, size=opens.shape))
    index_open = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(3, sessions)), axis=1))
    index_close = index_open * np.exp(rng.normal(0, 0.005, size=index_open.shape))
    markets = list(STOCKS)
    dates = pd.bdate_range("2025-06-02", periods=DATES)
    horizons = np.arange(1, 15)
    meta, absolute, bench, good = [], [], [], []
    for day, date in enumerate(dates):
        for number, (symbol, market) in enumerate(symbols):
            meta.append((symbol, f"{date.date()}T00:00:00+00:00", market,
                         "etf" if symbol.endswith("-0") else "stock", "fixture"))
            absolute.append(np.log(closes[number, day + horizons] / opens[number, day + 1]))
            m = markets.index(market)
            bench.append(np.log(index_close[m, day + horizons] / index_open[m, day + 1]))
            # GOOD knows the next open-to-open move, with noise.
            move = np.log(opens[number, day + 2] / opens[number, day + 1])
            good.append(np.full(14, move) + rng.normal(0, 0.005, size=14))
    metadata = np.array(meta, dtype=[("symbol", "<U48"), ("date", "<U40"), ("market", "<U24"),
                                     ("asset_type", "<U24"), ("provider", "<U32")])
    absolute = np.array(absolute, dtype=np.float32)
    alpha = (absolute - np.array(bench)).astype(np.float32)
    count = len(meta)
    noise = rng.normal(size=(count, 14)).astype(np.float32)
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
    np.save(export / split / "predictions.npy", np.array(good, dtype=np.float32))


@pytest.fixture()
def price_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    for split, seed in (("validation", 1), ("test", 2)):
        write_price_split(root, split, seed)
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


def fixture_config(default_config, tmp_path, name, **overrides):
    """A copy of a run's default config pointed at the fixture artifacts and signals."""
    config = json.loads(default_config.read_text())
    config["inputs"] = {"alpha_baseline": ALPHA_ID, "direction_baseline": DIRECTION_ID,
                        "predictions_root": "diagnostics/signal-predictions"}
    config["signals"] = {
        "GOOD": {"kind": "alpha", "members": [[RUN, CHECKPOINT]]},
        "NOISE": {"kind": "alpha", "baseline": "gru-42"},
        "DIRECTION": {"kind": "direction", "baseline": "gru-42"},
    }
    config["bootstrap"]["replicates"] = 50
    config.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path
