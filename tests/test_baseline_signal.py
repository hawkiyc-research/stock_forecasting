"""Cloud numerical regression for independent baseline heads and fixed labels."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stock_forecasting.baseline_signal import (
    SignalBaseline,
    fixed_return_labels,
    signal_forward_loss,
)
from stock_forecasting.baselines import CausalGRUBaseline, DLinearBaseline, PatchTSTBaseline

ROOT = Path(__file__).resolve().parents[1]


def signal_parameters(task):
    from stock_forecasting.baseline_signal_contract import parameters_for_task

    parameters = parameters_for_task(
        json.loads((ROOT / "configs/baseline.json").read_text()),
        {"baseline_task": {"prediction_task": task}},
    )
    parameters.update(epochs=1, batch_size=16, evaluations_per_epoch=2)
    parameters["resources"].update(auto_batch=False, checkpoint_seconds=0)
    parameters["gbdt"].update(max_iter=2, validation_every=1, max_leaf_nodes=3)
    return parameters


@pytest.fixture
def signal_config(tmp_path, market_frame):
    from stock_forecasting.config import ExperimentConfig
    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.manifest import artifact_metadata

    # Intraday moves cover all three fixed classes in each market/horizon.
    frame = market_frame.copy()
    position = frame.groupby("symbol", sort=False).cumcount().to_numpy()
    frame["open"] = frame["close"] / np.exp(0.05 * np.sin(position * 0.71))
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.006
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.994
    raw, store = tmp_path / "signal.parquet", tmp_path / "signal-bars"
    frame.to_parquet(raw, index=False)
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(frame))}
        },
        window_size=32,
        bucket_count=4,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.bar_store_path, config.data.input_length = store, 32
    config.data.h_start = 1
    config.model.direction.enabled = False
    return config


def test_fixed_threshold_edges_and_no_cost_features():
    threshold = 0.01
    lower, upper = np.float32(np.log1p(-threshold)), np.float32(np.log1p(threshold))
    values = np.asarray([[lower - 0.001, lower, 0, upper, upper + 0.001]], dtype=np.float32)
    np.testing.assert_array_equal(fixed_return_labels(values, threshold), [[0, 1, 1, 1, 2]])


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_absolute_labels_exactly_match_main_next_open_convention(signal_config, split):
    from stock_forecasting.baseline_signal import SignalBatchDataset, signal_array_batch
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.direction_data import DirectionDataset

    source = lazy_dataset(signal_config, split)
    try:
        indices = [0, min(13, len(source) - 1), -1, 0]
        records = DirectionDataset(source).__getitems__(indices)
        expected = torch.stack([record["target_absolute"] for record in records]).numpy()
        actual = signal_array_batch(source, indices)
        np.testing.assert_array_equal(actual["absolute_targets"], expected)
        np.testing.assert_array_equal(
            actual["targets"].astype(np.float32),
            torch.stack([record["target_alpha"] for record in records]).numpy(),
        )
        batch = SignalBatchDataset(
            source, neural=True, metadata=True, task="absolute_direction"
        ).__getitems__(indices)
        np.testing.assert_array_equal(batch["target_absolute"].numpy(), expected)
        assert "direction_bands" not in batch
    finally:
        source.close()


@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_neural_batch_retains_fast_path_without_tree_features(signal_config, monkeypatch, task):
    import stock_forecasting.baselines as baselines
    from stock_forecasting.baseline_input import NeuralBatchDataset
    from stock_forecasting.baseline_signal import SignalBatchDataset
    from stock_forecasting.baseline_storage import lazy_dataset

    source = lazy_dataset(signal_config, "train")
    try:
        indices = [0, min(13, len(source) - 1), -1]
        expected = NeuralBatchDataset(source).__getitems__(indices)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Neural batches must not compute unused tree features")

        monkeypatch.setattr(baselines, "baseline_arrays_from_windows", forbidden)
        actual = SignalBatchDataset(source, neural=True, task=task).__getitems__(indices)
        torch.testing.assert_close(actual["sequences"], expected["sequences"], rtol=0, atol=0)
        torch.testing.assert_close(actual["target_alpha"], expected["target_alpha"], rtol=0, atol=0)
        assert "symbols" not in actual
        assert ("target_absolute" in actual) == (task == "absolute_direction")
    finally:
        source.close()


@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_single_task_loss_backward_and_roundtrip(name, task):
    torch.manual_seed(42)
    horizons = tuple(range(1, 15))
    constructors = {
        "gru": lambda: CausalGRUBaseline(horizons=horizons),
        "dlinear": lambda: DLinearBaseline(32, horizons=horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=horizons),
    }
    parameters = {"prediction_task": task, "point_loss_beta": 1.0, "neutral_return_threshold": 0.01}
    model = SignalBaseline(constructors[name](), parameters)
    batch = {
        "sequences": torch.randn(6, 2, 32, 5),
        "target_alpha": torch.randn(6, len(horizons)) * 0.03,
        "target_absolute": torch.tensor([-0.02, 0.0, 0.02])[torch.arange(len(horizons)) % 3].expand(
            6, -1
        ),
    }
    scales = torch.linspace(0.02, 0.08, len(horizons))
    prediction, logits, loss, losses = signal_forward_loss(model, batch, scales, "cpu")
    assert logits is None and len(losses) == 1
    if task == "alpha_point":
        assert prediction.shape == (6, len(horizons))
        torch.testing.assert_close(
            loss, F.smooth_l1_loss(prediction / scales, batch["target_alpha"] / scales)
        )
    else:
        assert prediction.shape == (6, len(horizons), 3)
        torch.testing.assert_close(
            loss,
            F.cross_entropy(prediction.flatten(0, 1), (torch.arange(len(horizons)) % 3).repeat(6)),
        )
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SignalBaseline(constructors[name](), parameters)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    model.eval()
    restored.eval()
    torch.testing.assert_close(
        model(batch["sequences"]), restored(batch["sequences"]), rtol=0, atol=0
    )


@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_full_signal_cache_resume_rules_and_tree(signal_config, tmp_path, monkeypatch, task):
    import stock_forecasting.baseline_build as build
    import stock_forecasting.baseline_storage as storage
    from stock_forecasting.baseline_signal_contract import (
        validate_class_audit,
        validate_signal_metrics,
    )
    from stock_forecasting.data.manifest import sha256_file

    config = signal_config
    config.model.prediction_task = task
    p = signal_parameters(task)
    root = tmp_path / "suite"
    source_sha = sha256_file(config.data.bar_store_path / "bar-store.json")
    original_commit = storage._commit_tabular_progress

    def interrupt_inputs(directory, arrays, identity, count, offset, class_label_counts=None):
        original_commit(directory, arrays, identity, count, offset, class_label_counts)
        if directory.name == "train" and offset > 0:
            raise InterruptedError("committed signal input batch")

    monkeypatch.setattr(storage, "_commit_tabular_progress", interrupt_inputs)
    with pytest.raises(InterruptedError, match="committed signal input"):
        storage.build_tabular_cache(
            config, root / "inputs", workers=2, batch_size=32, checkpoint_seconds=0
        )
    monkeypatch.setattr(storage, "_commit_tabular_progress", original_commit)
    cache = storage.build_tabular_cache(config, root / "inputs", workers=2, batch_size=23)
    assert storage.build_tabular_cache(config, root / "inputs", workers=2) == cache
    assert sha256_file(config.data.bar_store_path / "bar-store.json") == source_sha
    horizons = list(config.data.alpha_horizons)
    scales = [0.03] * len(horizons)
    for split, count in cache["counts"].items():
        storage.validate_tabular(
            root / "inputs",
            split,
            count,
            len(horizons),
            absolute_signal=task == "absolute_direction",
        )
        arrays = storage.open_tabular(root / "inputs", split)
        assert arrays["features"].shape == (count, 30)
        assert "direction_bands" not in arrays
        for array in arrays.values():
            array._mmap.close()
    if task == "absolute_direction":
        assert not validate_class_audit(
            cache["class_label_audit"], p["signal_task"], cache["counts"]["train"], horizons
        )
        assert set(cache["class_label_audit"]["counts"]) == {"overall", "US", "TWSE"}

    tree = root / "tree"
    tree.mkdir()
    original_save = build._save_pickle

    def interrupt_tree(path, payload):
        original_save(path, payload)
        if path.name == "resume.pkl" and payload["cursor"] == 1:
            raise InterruptedError("committed signal tree")

    monkeypatch.setattr(build, "_save_pickle", interrupt_tree)
    with pytest.raises(InterruptedError, match="committed signal tree"):
        build._train_gbdt(root, tree, 42, p, horizons, scales)
    monkeypatch.setattr(build, "_save_pickle", original_save)
    result = build._train_gbdt(root, tree, 42, p, horizons, scales)
    assert build._train_gbdt(root, tree, 42, p, horizons, scales) == result
    validate_signal_metrics(result, p["signal_task"], cache["counts"]["test"], horizons)
    rules = root / "rules"
    rules.mkdir()
    results = build._train_rules(root, rules, horizons, scales, p)
    assert len(results) == 8
    assert build._train_rules(root, rules, horizons, scales, p) == results
    for name, row in results.items():
        validate_signal_metrics(row["metrics"], p["signal_task"], cache["counts"]["test"], horizons)
        assert row["metrics"]["sample_membership"] == result["sample_membership"]
        assert not (rules / name / "test/absolute_targets.npy").exists()
    zero = np.load(rules / "zero_return/test/predictions.npy")
    if task == "alpha_point":
        assert np.all(zero == 0)
        assert np.all(np.load(rules / "always_buy/test/predictions.npy") > 0)
    else:
        assert np.all(zero.argmax(-1) == 1)
    assert not (tree / "test/absolute_targets.npy").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Authorized cloud CUDA only")
@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
@pytest.mark.parametrize("name", ["gru", "dlinear", "patchtst"])
def test_neural_signal_full_splits_and_resume(signal_config, tmp_path, monkeypatch, task, name):
    import stock_forecasting.baseline_build as build
    from stock_forecasting.baseline_signal_contract import validate_signal_metrics

    config = signal_config
    config.model.prediction_task = task
    p = signal_parameters(task)
    directory = tmp_path / f"{task}-{name}"
    directory.mkdir()
    horizons = list(config.data.alpha_horizons)
    scales = [0.03] * len(horizons)
    plan = {"loader_workers": 2, "prefetch_factor": 2, "gpu_fraction_per_job": 0.35}
    original = build._save_torch

    def interrupt(path, payload):
        original(path, payload)
        if path.name == "resume.pt" and payload["sample_cursor"] > 0:
            raise InterruptedError("committed single-task checkpoint")

    monkeypatch.setattr(build, "_save_torch", interrupt)
    with pytest.raises(InterruptedError, match="single-task checkpoint"):
        build._train_neural(config, directory, name, 42, p, scales, plan)
    before = torch.load(directory / "resume.pt", weights_only=False)
    assert before["sample_cursor"] > 0
    monkeypatch.setattr(build, "_save_torch", original)
    p["batch_size"] = 23
    result = build._train_neural(config, directory, name, 42, p, scales, plan)
    source = build.lazy_dataset(config, "test")
    validate_signal_metrics(result, p["signal_task"], len(source), horizons)
    source.close()
    assert build._train_neural(config, directory, name, 42, p, scales, plan) == result
    assert (directory / "model.pt").is_file()
    history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert all(row["task_loss_order"] == [p["objectives"]["neural"]] for row in history)
    assert not (directory / "test/absolute_targets.npy").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Authorized cloud CUDA only")
@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_signal_suite_parallel_publication_and_reuse(signal_config, tmp_path, monkeypatch, task):
    """Exercise actual spawned CPU/GPU jobs, the final gate, and canonical labels."""
    import stock_forecasting.baseline_build as build
    from stock_forecasting.baseline_contract import validate_complete, validate_signal_admission
    from stock_forecasting.baseline_result_storage import finalize_baseline_storage

    config, p = signal_config, signal_parameters(task)
    config.model.prediction_task = task
    p["label_scale_calibration_samples"] = 32
    p["resources"].update(
        checkpoint_seconds=30,
        max_loader_workers_per_experiment=1,
        worker_bytes=512 * 1024**2,
        parquet_cache_bytes=128 * 1024**2,
        host_bytes_per_gpu_experiment=1024**3,
        gpu_bytes_per_experiment=2 * 1024**3,
        max_input_batch_size=32,
    )
    for name in (
        "evaluations_per_epoch",
        "early_stopping_patience_evaluations",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
        "plateau_patience_evaluations",
        "plateau_factor",
        "plateau_min_ratio",
        "plateau_min_low_lr_evaluations",
    ):
        setattr(config.training, name, p[name])
    config.training.learning_rate_schedule = "validation_plateau"
    config.training.evaluation_max_samples = None
    config.data.calibration_seed = p["calibration_seed"]
    config.data.label_scale_calibration_samples = p["label_scale_calibration_samples"]
    config.validation.models = ["kronos_full", *p["models"]]
    config.validation.seeds = p["seeds"]
    config.validation.baseline_max_samples_per_split = None
    project = tmp_path / "control"
    (project / "configs").mkdir(parents=True)
    (project / "configs/baseline.json").write_text(json.dumps(p))
    numerical = {key: value for key, value in p.items() if key != "resources"}
    identity = {
        "baseline_id": f"baseline-fixture-{task}",
        "contract": {"parameters": numerical, "data": {"preparation": {"h_start": 1}}},
    }
    monkeypatch.setattr(build, "runtime_contract", lambda: (project, identity))
    monkeypatch.setenv("NETWORK_VOLUME_ROOT", str(tmp_path / "volume"))
    payload = build.build_baselines(config)
    root = tmp_path / "volume/baselines" / identity["baseline_id"]
    validate_complete(payload, identity)
    assert set(payload["models"]) == set(p["models"])
    if task == "absolute_direction":
        validate_signal_admission(payload, numerical, config.data.alpha_horizons)
    executions = {
        path.parent.name: json.loads(path.read_text())
        for path in (root / "jobs").glob("*/execution.json")
    }
    assert all(row["finished_at"] > row["started_at"] for row in executions.values())
    assert all(
        executions[f"{name}-42"]["peak_gpu_bytes"] > 0 for name in ("gru", "dlinear", "patchtst")
    )
    allocation = json.loads((root / "resource-plan.json").read_text())

    def overlaps(first, second):
        return max(first["started_at"], second["started_at"]) < min(
            first["finished_at"], second["finished_at"]
        )

    if allocation["gpu_slots"] > 1:
        assert overlaps(executions["gru-42"], executions["dlinear-42"])
    assert any(
        overlaps(executions[cpu], executions[gpu])
        for cpu in ("gbdt-42", "rules")
        for gpu in ("gru-42", "dlinear-42", "patchtst-42")
    )
    before = {path: path.stat().st_mtime_ns for path in root.rglob("model.*")}
    assert build.build_baselines(config) == payload
    assert {path: path.stat().st_mtime_ns for path in before} == before
    finalized = finalize_baseline_storage(root, apply=True, workers=2, progress=lambda *_: None)
    validate_complete(finalized["payload"], identity, require_shared=True)
    assert build.build_baselines(config) == finalized["payload"]
    assert not list((root / "jobs").rglob("targets.npy"))
    assert not list((root / "jobs").rglob("membership.npy"))
    assert not list((root / "jobs").rglob("absolute_targets.npy"))
    assert (root / "inputs/test/targets.npy").is_file()
    if task == "absolute_direction":
        assert (root / "inputs/test/absolute_targets.npy").is_file()
