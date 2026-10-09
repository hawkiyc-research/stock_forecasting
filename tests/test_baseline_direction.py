"""Numerical parity, full-population outputs, and resume tests for direction baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_full_workflow_models import small_lazy_config as _small_lazy_config

from stock_forecasting.baseline_direction import (
    BaselineDirectionEvaluation,
    DirectionalBaseline,
    forward_loss,
)
from stock_forecasting.baseline_direction_data import direction_array_batch, numpy_direction_labels
from stock_forecasting.baseline_direction_tree import (
    ConstantDirectionClassifier,
    predict_probabilities,
)
from stock_forecasting.baseline_input import NeuralBatchDataset
from stock_forecasting.baseline_storage import (
    build_tabular_cache,
    lazy_dataset,
    open_tabular,
    validate_tabular,
)
from stock_forecasting.baselines import CausalGRUBaseline, DLinearBaseline, PatchTSTBaseline
from stock_forecasting.config import DirectionConfig
from stock_forecasting.direction_data import DirectionDataset
from stock_forecasting.models.direction import direction_labels

# Re-export a fixture while avoiding name-shadowing imports in test signatures.
small_lazy_config = _small_lazy_config

ROOT = Path(__file__).resolve().parents[1]


def parameters():
    result = json.loads((ROOT / "configs/baseline.json").read_text())
    result.update(epochs=1, batch_size=16, evaluations_per_epoch=2)
    result["direction"]["loss_reference_samples"] = 32
    result["resources"].update(auto_batch=False, checkpoint_seconds=0)
    result["gbdt"].update(max_iter=2, validation_every=1, max_leaf_nodes=3)
    return result


def enable(config):
    config.model.direction = DirectionConfig(**parameters()["direction"])
    return config


@pytest.mark.parametrize("count", [1, 3])
def test_loss_history_orders_match_both_quantile_only_and_direction_tasks(tmp_path, count):
    from stock_forecasting.baseline_build import _history

    losses = [0.3, 1.1, 1.1][:count]
    for record in (
        {"phase": "training", "task_losses": losses},
        {"phase": "validation", "training_task_losses": losses},
        {"phase": "validation", "training": {"task_losses": losses}},
    ):
        _history(tmp_path, record)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(len(row["task_loss_order"]) == count for row in rows)
    assert all(row["task_loss_order"][0] == "normalized_pinball" for row in rows)


def test_numpy_labels_match_torch_including_edges_and_market_crash():
    bands = np.asarray([[-0.005, 0.005]], dtype="float32")
    y = np.asarray([[-0.006, -0.005, 0, 0.005, 0.006]], dtype="float32")
    np.testing.assert_array_equal(
        numpy_direction_labels(y, bands),
        direction_labels(torch.tensor(y), torch.tensor(bands)).numpy(),
    )
    assert numpy_direction_labels(np.asarray([[np.log(0.96)]]), bands).item() == 0
    assert numpy_direction_labels(np.asarray([[np.log(0.96) - np.log(0.92)]]), bands).item() == 2


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_lazy_baseline_labels_match_main_without_changing_membership(small_lazy_config, split):
    config = enable(small_lazy_config)
    source = lazy_dataset(config, split)
    indices = [0, min(13, len(source) - 1), len(source) - 1]
    expected = DirectionDataset(source).__getitems__(indices)
    arrays = direction_array_batch(source, indices, config.model.direction.costs)
    np.testing.assert_array_equal(
        arrays["absolute_targets"],
        torch.stack([row["target_absolute"] for row in expected]).numpy(),
    )
    np.testing.assert_array_equal(
        arrays["targets"].astype("float32"),
        torch.stack([row["target_alpha"] for row in expected]).numpy(),
    )
    from stock_forecasting.training import ModelBatchCollator

    batch = ModelBatchCollator.for_model(config, training=False)(expected)
    np.testing.assert_array_equal(arrays["direction_bands"], batch["direction_bands"].numpy())
    actual = NeuralBatchDataset(
        source, metadata=True, direction_costs=config.model.direction.costs
    ).__getitems__(indices)
    torch.testing.assert_close(actual["target_absolute"], batch["target_absolute"], rtol=0, atol=0)
    assert actual["symbols"] == batch["symbols"] and actual["cutoff_at"] == batch["cutoff_at"]
    source.close()


@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Cloud CUDA")
        ),
    ],
)
def test_dual_heads_backward_preserve_quantiles_and_save_restore(name, device, tmp_path):
    horizons = tuple(range(1, 15))
    base = {
        "gru": lambda: CausalGRUBaseline(horizons=horizons),
        "dlinear": lambda: DLinearBaseline(32, horizons=horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=horizons),
    }[name]()
    model = DirectionalBaseline(base, DirectionConfig(enabled=True)).to(device)
    model.restore_references({"schema_version": 1, "samples": 32, "references": [0.3, 1.2, 0.9]})
    x = torch.randn(4, 2, 32, 5, device=device)
    batch = {
        "sequences": x,
        "target_alpha": torch.randn(4, 14, device=device) * 0.03,
        "target_absolute": torch.randn(4, 14, device=device) * 0.04,
        "direction_bands": torch.tensor([[-0.005, 0.005]], device=device).expand(4, -1),
    }
    q, logits, loss, losses = forward_loss(
        model, batch, torch.ones(14, device=device) * 0.03, device
    )
    assert q.shape == (4, 14, 3) and all(item.shape == q.shape for item in logits)
    assert torch.all(q[..., 0] <= q[..., 1]) and torch.all(q[..., 1] <= q[..., 2])
    torch.testing.assert_close(loss, losses[0] + 0.25 * losses[1] + (1 / 3) * losses[2])
    loss.backward()
    assert model.absolute.weight.grad.abs().sum() > 0 and model.alpha.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        before = model.forward_tasks(x, batch["direction_bands"])
        torch.testing.assert_close(before[0], model(x), rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    with torch.no_grad():
        for first, second in zip(
            before, model.forward_tasks(x, batch["direction_bands"]), strict=True
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.reference_state()["samples"] == 32


def test_direction_cache_full_resume_and_cpu_rules_trees(small_lazy_config, tmp_path, monkeypatch):
    import stock_forecasting.baseline_build as build

    config = enable(small_lazy_config)
    p = parameters()
    root = tmp_path / "suite"
    cache = build_tabular_cache(config, root / "inputs", workers=2, batch_size=32)
    assert build_tabular_cache(config, root / "inputs", workers=2) == cache
    horizons = list(config.data.alpha_horizons)
    scales = [0.03] * len(horizons)
    for split, count in cache["counts"].items():
        validate_tabular(root / "inputs", split, count, len(horizons), direction=True)
        arrays = open_tabular(root / "inputs", split)
        assert arrays["features"].shape == (count, 32)
        assert arrays["absolute_targets"].shape == (count, len(horizons))
    tree = root / "gbdt"
    tree.mkdir()
    original_save = build._save_pickle

    def interrupt(path, payload):
        original_save(path, payload)
        if path.name == "resume.pkl" and payload["next_model"] == len(horizons) * 3 + 1:
            raise InterruptedError("committed direction tree checkpoint")

    monkeypatch.setattr(build, "_save_pickle", interrupt)
    with pytest.raises(InterruptedError, match="direction tree checkpoint"):
        build._train_gbdt(root, tree, 42, p, horizons, scales)
    monkeypatch.setattr(build, "_save_pickle", original_save)
    first = build._train_gbdt(root, tree, 42, p, horizons, scales)
    second = build._train_gbdt(root, tree, 42, p, horizons, scales)
    assert first == second and first["direction"]["samples"] == cache["counts"]["test"]
    assert first["direction"]["prediction_source"] == "supervised"
    assert np.load(tree / "test/direction-probabilities.npy", mmap_mode="r").shape == (
        cache["counts"]["test"],
        2,
        len(horizons),
        3,
    )
    rules = root / "rules"
    rules.mkdir()
    results = build._train_rules(root, rules, horizons, scales, p)
    assert len(results) == 8
    for name, result in results.items():
        direction = result["metrics"]["direction"]
        assert direction["samples"] == cache["counts"]["test"]
        assert direction["overall"]["absolute"]["aggregate"]["cross_entropy"] is None
        assert "selection_loss" not in direction
        assert (rules / name / "test/direction-classes.npy").is_file()
    for name, code in (("always_buy", 2), ("zero_return", 1)):
        assert np.all(np.load(rules / name / "test/direction-classes.npy") == code)


def test_missing_training_classes_map_to_fixed_three_class_axis():
    classifiers = [[ConstantDirectionClassifier(2)], [ConstantDirectionClassifier(0)]]
    probabilities = predict_probabilities(classifiers, np.zeros((5, 32)))
    assert probabilities.shape == (5, 2, 1, 3)
    np.testing.assert_array_equal(probabilities[:, 0, 0, 2], 1)
    np.testing.assert_array_equal(probabilities[:, 1, 0, 0], 1)


def test_baseline_and_main_direction_metrics_match_including_extreme_logits():
    from types import SimpleNamespace

    from stock_forecasting.baseline_build import _append_direction
    from stock_forecasting.baseline_direction import reference_head
    from stock_forecasting.direction_runtime import DirectionMetrics

    config = DirectionConfig(enabled=True)
    head = reference_head(
        {"schema_version": 1, "samples": 32, "references": [0.3, 1.1, 1.1]}, config.model_dump()
    )
    logits = (
        torch.tensor([[-100.0, 0.0, 100.0], [10.0, 0.0, -10.0], [0.0, 20.0, 0.0]])
        .unsqueeze(1)
        .expand(-1, 14, -1)
    )
    targets = torch.tensor([-0.05, 0.1, 0.0]).unsqueeze(1).expand(-1, 14)
    batch = {
        "target_absolute": targets,
        "target_alpha": -targets,
        "direction_bands": torch.tensor([[-0.005, 0.005]]).expand(3, -1),
        "markets": ["US", "TWSE", "US"],
        "cutoff_at": ["2026-01-02"] * 3,
    }
    main = DirectionMetrics(range(1, 15))
    main.append(
        SimpleNamespace(
            absolute_direction_logits=logits, alpha_direction_logits=-logits, pinball_loss=0.3
        ),
        batch,
    )
    baseline = BaselineDirectionEvaluation(range(1, 15), config.costs, 3, head=head)
    _append_direction(baseline, (logits, -logits), batch, 0.3)
    result = baseline.finish()
    for key, value in main.finish(head, config.costs).items():
        assert result[key] == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cloud CUDA")
def test_direction_runtime_measures_joint_path_without_updating_weights(small_lazy_config):
    from stock_forecasting.baseline_runtime import tune_baseline_runtime

    config, p = enable(small_lazy_config), parameters()
    source = lazy_dataset(config, "train")
    model = DirectionalBaseline(
        DLinearBaseline(config.data.input_length, horizons=tuple(config.data.alpha_horizons)),
        config.model.direction,
    ).cuda()
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    p["resources"].update(
        auto_batch=True,
        max_training_batch_size=32,
        max_evaluation_batch_size=32,
        batch_probe_repetitions=1,
        runtime_probe_warmup_samples=16,
        runtime_probe_samples=32,
        loader_probe_batches=2,
    )
    plan = {
        "loader_workers": 2,
        "prefetch_factor": 2,
        "gpu_fraction_per_job": 0.35,
        "gpu_job_host_bytes": 6 * 1024**3,
    }
    runtime = tune_baseline_runtime(
        model, source, p, plan, scales=[0.03] * len(config.data.alpha_horizons)
    )
    assert runtime["source"] == "end_to_end_real_windows"
    assert runtime["loader_workers"] >= 1 and runtime["prefetch_factor"] >= 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    source.close()


def test_direction_comparison_reuses_metrics_and_rejects_different_costs():
    from copy import deepcopy

    from stock_forecasting.validation_benchmark import FULL_MODEL_NAME, _direction_comparison

    full = {
        "samples": 2,
        "class_order": ["down", "neutral", "up"],
        "task_order": ["absolute", "alpha"],
        "cost_scenario": DirectionConfig().costs.model_dump(),
        "overall": {
            task: {"horizons": {str(h): {"count": 2} for h in range(1, 15)}}
            for task in ("absolute", "alpha")
        },
    }
    results = {
        FULL_MODEL_NAME: {"metrics": {"direction": full}},
        "gru": {"seed_results": {"42": {"direction": deepcopy(full)}}},
    }
    assert _direction_comparison(results)["models"]["gru"]["42"]["overall"] == full["overall"]
    results["gru"]["seed_results"]["42"]["direction"]["cost_scenario"]["us_shares"] = 100
    with pytest.raises(ValueError, match="matching complete"):
        _direction_comparison(results)


def test_direction_evaluation_rejects_incomplete_and_invalid_probabilities(tmp_path):
    costs = DirectionConfig().costs
    evaluation = BaselineDirectionEvaluation([1], costs, 3, output=tmp_path)
    try:
        with pytest.raises(ValueError, match="full split"):
            evaluation.finish()
        with pytest.raises(ValueError, match="probabilities"):
            evaluation.append(
                np.ones((2, 2, 1, 3)),
                np.zeros((2, 2, 1)),
                np.zeros((2, 2)),
                ["US"] * 2,
                ["2026-01-01"] * 2,
            )
    finally:
        evaluation.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cloud CUDA")
@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
def test_neural_direction_full_splits_and_interrupted_resume(
    small_lazy_config, tmp_path, monkeypatch, name
):
    import stock_forecasting.baseline_build as build

    config, p = enable(small_lazy_config), parameters()
    plan = {"loader_workers": 2, "prefetch_factor": 2, "gpu_fraction_per_job": 0.35}
    directory = tmp_path / name
    directory.mkdir()
    scales = [0.03] * len(config.data.alpha_horizons)
    original = build._save_torch

    def interrupt(path, payload):
        original(path, payload)
        if path.name == "resume.pt" and payload["sample_cursor"] > 0:
            raise InterruptedError("committed direction checkpoint")

    monkeypatch.setattr(build, "_save_torch", interrupt)
    with pytest.raises(InterruptedError):
        build._train_neural(config, directory, name, 42, p, scales, plan)
    before = torch.load(directory / "resume.pt", weights_only=False)
    refs = before["model"]["references"].clone()
    assert before["model"]["reference_count"].item() == 32
    monkeypatch.setattr(build, "_save_torch", original)
    p["batch_size"] = 23
    result = build._train_neural(config, directory, name, 42, p, scales, plan)
    after = torch.load(directory / "resume.pt", weights_only=False)
    torch.testing.assert_close(refs, after["model"]["references"], rtol=0, atol=0)
    assert result["direction"]["samples"] == result["samples"]
    assert result["direction"]["loss_reference"]["samples"] == 32
    assert (directory / "validation/direction-probabilities.npy").is_file()
    assert (directory / "test/direction-probabilities.npy").is_file()
    assert build._train_neural(config, directory, name, 42, p, scales, plan) == result
