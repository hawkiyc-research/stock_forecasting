"""Cross-sectional rank baselines: contract, masked loss, Rank IC selection and CPU jobs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MIN_GROUP = 3


def rank_parameters():
    from stock_forecasting.baseline_signal_contract import parameters_for_task

    parameters = parameters_for_task(
        json.loads((ROOT / "configs/baseline.json").read_text()),
        {
            "baseline_task": {
                "prediction_task": "cross_sectional_rank",
                "rank_min_group_size": MIN_GROUP,
            }
        },
    )
    parameters.update(epochs=1, batch_size=16, evaluations_per_epoch=2)
    parameters["resources"].update(auto_batch=False, checkpoint_seconds=0)
    parameters["gbdt"].update(max_iter=2, validation_every=1, max_leaf_nodes=3)
    return parameters


@pytest.fixture
def ranked(rank_config):
    from stock_forecasting.data.rank_labels import ensure_split_rank_labels
    from stock_forecasting.data.sample_universe import validated_split_roots

    rank_config.model.prediction_task = "cross_sectional_rank"
    rank_config.model.rank_min_group_size = MIN_GROUP
    roots = {split: str(root) for split, root in validated_split_roots(rank_config).items()}
    stores = ensure_split_rank_labels(rank_config, roots, workers=1)
    assert set(stores) == {"train", "validation", "test"}
    return rank_config


def test_rank_task_contract_and_objectives():
    from stock_forecasting.baseline_signal_contract import signal_task

    parameters = rank_parameters()
    assert parameters["signal_task"] == {
        "prediction_task": "cross_sectional_rank",
        "rank_min_group_size": MIN_GROUP,
    }
    assert parameters["objectives"] == {
        "neural": "masked_mse_gauss_rank",
        "gbdt": "squared_error_gauss_rank",
        "rules": "raw_rule_score",
    }
    assert parameters["direction"] == {"enabled": False}
    default = signal_task({"baseline_task": {"prediction_task": "cross_sectional_rank"}})
    assert default["rank_min_group_size"] == 20
    for invalid in (1, 2.5, True, "20"):
        with pytest.raises(ValueError, match="rank_min_group_size"):
            signal_task(
                {
                    "baseline_task": {
                        "prediction_task": "cross_sectional_rank",
                        "rank_min_group_size": invalid,
                    }
                }
            )


@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
def test_masked_rank_loss_ignores_unlabelled_values(name):
    from stock_forecasting.baseline_signal import SignalBaseline, signal_forward_loss
    from stock_forecasting.baselines import CausalGRUBaseline, DLinearBaseline, PatchTSTBaseline

    torch.manual_seed(3)
    horizons = tuple(range(1, 15))
    base = {
        "gru": lambda: CausalGRUBaseline(horizons=horizons),
        "dlinear": lambda: DLinearBaseline(32, horizons=horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=horizons),
    }[name]()
    model = SignalBaseline(
        base, {"prediction_task": "cross_sectional_rank", "rank_min_group_size": MIN_GROUP}
    )
    target = torch.randn(6, len(horizons))
    target[1] = float("nan")
    target[4, :3] = float("nan")
    batch = {"sequences": torch.randn(6, 2, 32, 5), "target_rank": target}
    prediction, logits, loss, losses = signal_forward_loss(
        model, batch, torch.ones(len(horizons)), "cpu"
    )
    assert logits is None and len(losses) == 1
    assert prediction.shape == (6, len(horizons))
    labelled = torch.isfinite(target)
    torch.testing.assert_close(loss, F.mse_loss(prediction[labelled], target[labelled]))
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = {"sequences": batch["sequences"], "target_rank": torch.full_like(target, float("nan"))}
    _prediction, _logits, empty_loss, _losses = signal_forward_loss(
        model, empty, torch.ones(len(horizons)), "cpu"
    )
    assert float(empty_loss.detach()) == 0.0


def test_rank_evaluation_selects_on_market_equal_rank_ic(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import stock_forecasting.signal_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "detect_available_memory",
        lambda: SimpleNamespace(available_bytes=8 * 1024**3),
    )
    rng = np.random.default_rng(11)
    horizons, dates, symbols = [1, 5], [f"2025-06-{d:02d}" for d in range(2, 8)], 6
    rows = [(m, date, s) for m in ("TWSE", "US") for date in dates for s in range(symbols)]
    alpha = rng.normal(size=(len(rows), len(horizons)))
    markets = np.asarray([row[0] for row in rows])
    # TWSE predictions are perfectly ordered, US predictions are perfectly reversed.
    predictions = np.where(markets[:, None] == "TWSE", alpha, -alpha)
    ranks = rng.normal(size=alpha.shape).astype(np.float32)
    ranks[:3] = np.nan
    store = evaluation.SignalEvaluationStore(
        tmp_path / "eval",
        len(rows),
        horizons,
        [1.0, 1.0],
        "cross_sectional_rank",
        rank_min_group_size=MIN_GROUP,
    )
    try:
        store.append(
            alpha,
            predictions,
            symbols=[f"{m}-{s}" for m, _d, s in rows],
            dates=[d for _m, d, _s in rows],
            markets=markets.tolist(),
            asset_types=["stock"] * len(rows),
            providers=["fixture"] * len(rows),
            target_rank=ranks,
        )
        result = store.finish()
    finally:
        store.close()
    by_market = result["cross_sectional"]["by_market"]
    for horizon in ("1d", "5d"):
        assert by_market["TWSE"][horizon]["rank_ic_mean"] == pytest.approx(1.0)
        assert by_market["US"][horizon]["rank_ic_mean"] == pytest.approx(-1.0)
        assert result["per_horizon"][horizon]["selection_score"] == pytest.approx(0.0, abs=1e-12)
    assert result["rank_ic_selection"]["markets"] == ["TWSE", "US"]
    assert result["aggregate"]["selection_score"] == pytest.approx(0.0, abs=1e-12)
    assert result["primary_5d"]["selection_score"] == result["aggregate"]["selection_score"]
    assert result["rank_min_group_size"] == MIN_GROUP
    labelled = np.isfinite(ranks)
    expected = ((predictions - np.nan_to_num(ranks)) ** 2 * labelled).sum(0) / labelled.sum(0)
    assert result["per_horizon"]["1d"]["rank_mse"] == pytest.approx(expected[0])
    assert result["per_horizon"]["1d"]["labelled_fraction"] == pytest.approx(labelled[:, 0].mean())


def test_rank_neural_batches_carry_dataset_labels(ranked):
    from stock_forecasting.baseline_signal import SignalBatchDataset
    from stock_forecasting.baseline_storage import lazy_dataset

    source = lazy_dataset(ranked, "validation", relative=True)
    try:
        indices = [0, 7, len(source) - 1]
        expected = source.array_batch(indices)["rank_targets"]
        batch = SignalBatchDataset(
            source, neural=True, metadata=True, task="cross_sectional_rank"
        ).__getitems__(indices)
        np.testing.assert_array_equal(batch["target_rank"].numpy(), expected)
        assert batch["sequences"].shape == (3, 2, 32, 5)
        assert "target_absolute" not in batch
    finally:
        source.close()


def test_rank_cache_rules_and_tree(ranked, tmp_path):
    import stock_forecasting.baseline_build as build
    import stock_forecasting.baseline_storage as storage
    from stock_forecasting.baseline_signal_contract import validate_signal_metrics

    p = rank_parameters()
    root = tmp_path / "suite"
    cache = storage.build_tabular_cache(ranked, root / "inputs", workers=1, batch_size=23)
    assert storage.build_tabular_cache(ranked, root / "inputs", workers=1) == cache
    assert cache["identity"]["signal_task"] == p["signal_task"]
    assert all("rank_labels" in row for row in cache["identity"]["split_sources"].values())
    horizons = list(ranked.data.alpha_horizons)
    scales = [0.03] * len(horizons)
    for split, count in cache["counts"].items():
        storage.validate_tabular(root / "inputs", split, count, len(horizons), rank_signal=True)
        arrays = storage.open_tabular(root / "inputs", split)
        source = storage.lazy_dataset(ranked, split)
        try:
            np.testing.assert_array_equal(
                arrays["rank_targets"], source.array_batch(np.arange(count))["rank_targets"]
            )
        finally:
            source.close()
            for array in arrays.values():
                array._mmap.close()
    tree = root / "tree"
    tree.mkdir()
    result = build._train_gbdt(root, tree, 42, p, horizons, scales)
    assert build._train_gbdt(root, tree, 42, p, horizons, scales) == result
    validate_signal_metrics(result, p["signal_task"], cache["counts"]["test"], horizons)
    assert result["aggregate"]["selection_score"] == pytest.approx(
        -result["rank_ic_selection"]["rank_ic_market_macro"]
    )
    rules = root / "rules"
    rules.mkdir()
    results = build._train_rules(root, rules, horizons, scales, p)
    assert len(results) == 8
    assert build._train_rules(root, rules, horizons, scales, p) == results
    test_inputs = storage.open_tabular(root / "inputs", "test")
    try:
        momentum = np.load(rules / "momentum_5d/test/predictions.npy")
        signal = test_inputs["signals"][:, 0]
        np.testing.assert_allclose(momentum, np.repeat(signal[:, None], len(horizons), 1))
    finally:
        for array in test_inputs.values():
            array._mmap.close()
    for name, row in results.items():
        validate_signal_metrics(row["metrics"], p["signal_task"], cache["counts"]["test"], horizons)
        assert row["metrics"]["sample_membership"] == result["sample_membership"]
        assert not (rules / name / "test/rank_targets.npy").exists()
    zero = results["zero_return"]["metrics"]
    assert zero["aggregate"]["rank_ic_market_macro"] == 0.0
    assert np.all(np.load(rules / "zero_return/test/predictions.npy") == 0)
    assert not (tree / "test/rank_targets.npy").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Authorized cloud CUDA only")
@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
def test_rank_neural_job_full_splits(ranked, tmp_path, name):
    import stock_forecasting.baseline_build as build
    from stock_forecasting.baseline_signal_contract import validate_signal_metrics

    p = rank_parameters()
    directory = tmp_path / f"rank-{name}"
    directory.mkdir()
    horizons = list(ranked.data.alpha_horizons)
    scales = [0.03] * len(horizons)
    plan = {"loader_workers": 2, "prefetch_factor": 2, "gpu_fraction_per_job": 0.35}
    result = build._train_neural(ranked, directory, name, 42, p, scales, plan)
    source = build.lazy_dataset(ranked, "test")
    validate_signal_metrics(result, p["signal_task"], len(source), horizons)
    source.close()
    history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert all(row["task_loss_order"] == [p["objectives"]["neural"]] for row in history)
    assert not (directory / "test/rank_targets.npy").exists()
