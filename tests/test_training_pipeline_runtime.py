"""Cloud ML regression for input overlap and reversible end-to-end CUDA probes."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stock_forecasting import training
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import BlockwisePermutationSampler, FinancialBatchCollator
from stock_forecasting.models.ranking import same_date_ranking_loss

ROOT = Path(__file__).resolve().parents[1]


class ProbeDataset(Dataset):
    """Small generated windows with real cross-security, same-date ranking pairs."""

    def __len__(self):
        return 256

    def __getitem__(self, index):
        series = torch.arange(80, dtype=torch.float32).reshape(16, 5) / 100 + index / 500
        timestamps = torch.zeros(16, 5, dtype=torch.long)
        return {
            "asset_series": series, "benchmark_series": series + 1,
            "asset_timestamp_features": timestamps, "benchmark_timestamp_features": timestamps,
            "target_alpha": torch.arange(14, dtype=torch.float32) / 100 + index / 1000,
            "sample_id": str(index), "symbol": f"S{index % 64}", "benchmark_symbol": "INDEX",
            "asset_type": "stock", "market": "US", "provider": "fixture",
            "dataset_profile": "fixture", "cutoff_at": "2025-01-03", "diagnostics": {},
        }


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(5, 42)
        self.register_buffer("calls", torch.tensor(0))
        self.dropout = nn.Dropout(.1)

    def forward(self, asset, benchmark, *, target_alpha, ranking_group_ids, security_ids, **_kw):
        predictions = self.head(self.dropout(asset.mean(1))).reshape(-1, 14, 3)
        pinball = (predictions[:, :, 1] - target_alpha).square().mean()
        ranking = same_date_ranking_loss(
            predictions, target_alpha, predictions.new_ones(14),
            ranking_group_ids, security_ids, 16,
        ) if self.training else None
        if self.training:
            self.calls.add_(1)
        return SimpleNamespace(
            loss=pinball if ranking is None else pinball + .05 * ranking,
            pinball_loss=pinball, ranking_loss=ranking, alpha_quantiles=predictions,
        )


def test_worker_metadata_preserves_inputs_predictions_and_gradients():
    samples = [ProbeDataset()[index] for index in (1, 2, 4, 7)]
    old = FinancialBatchCollator()(samples)
    new = training.ModelBatchCollator()(samples)
    for name, value in old.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(new[name], value, rtol=0, atol=0)
        else:
            assert new[name] == value
    model = ProbeModel()
    model.dropout.eval()
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    bundle = SimpleNamespace(model=model)
    outputs = []
    gradients = []
    for batch in (old, new):
        torch.manual_seed(5)
        model.zero_grad(set_to_none=True)
        output = training.forward_batch(
            bundle, training._move_batch_to_device(batch, torch.device("cpu")),
            config, torch.device("cpu"),
        )
        output.loss.backward()
        outputs.append(output.loss.detach())
        gradients.append(model.head.weight.grad.clone())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_probe_rolls_back_parameters_buffers_modes_and_rng_even_on_failure(fail):
    model = ProbeModel()
    model.dropout.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    modes = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match="injected") if fail else nullcontext():
        with training._preserve_probe_state(model):
            optimizer = torch.optim.AdamW(model.parameters())
            model.train()
            batch = training.ModelBatchCollator()([ProbeDataset()[i] for i in range(4)])
            output = training.forward_batch(
                SimpleNamespace(model=model),
                training._move_batch_to_device(batch, torch.device("cpu")),
                None, torch.device("cpu"),
            )
            output.loss.backward()
            optimizer.step()
            random.random()
            np.random.random()
            if fail:
                raise RuntimeError("injected")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert modes == [module.training for module in model.modules()]
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), cpu_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
@pytest.mark.parametrize("training_mode", [True, False])
@pytest.mark.parametrize("cpu_guard", [True, False])
def test_cuda_pipeline_probe_has_real_ranking_and_optimizer_work_without_state_changes(
    monkeypatch, capsys, training_mode, cpu_guard,
):
    device = torch.device("cuda")
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.training.gradient_accumulation_steps = 2
    config.model.lora.enabled = False
    config.training.auto_batch_probe_steps = 2
    config.training.mixed_precision = "no"
    model = ProbeModel().to(device)
    model.dropout.eval()
    source = ProbeDataset()
    monkeypatch.setattr(training, "_training_sampler", lambda *_args: BlockwisePermutationSampler(
        len(source), seed=77, block_size=128
    ))
    if cpu_guard:
        monkeypatch.setattr(training, "_probe_cpu_pressure", lambda *_args: {
            "cpu_fraction": .99, "throttled_period_fraction": .20,
        })
    before = {name: value.clone() for name, value in model.state_dict().items()}
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    workers = training.plan_dataloader_workers(
        2, visible_cpu_count=4, available_memory_bytes=16 * 1024**3,
    )
    result = training._measure_cuda_batch(
        bundle=SimpleNamespace(model=model), sample=source[0], batch_size=4,
        config=config, device=device, training=training_mode,
        optimizer_state_reserve_bytes=0, device_memory_limit_bytes=1024**3,
        dataset=source, worker_plan=workers, require_cpu_headroom=cpu_guard,
    )
    assert result.accepted == (not cpu_guard) and result.samples_per_second > 0
    if cpu_guard:
        assert result.outcome == "cpu_quota_guard"
    rows = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    evidence = next(row["cuda_pipeline_probe"] for row in rows if "cuda_pipeline_probe" in row)
    assert evidence["measured_batches"] >= 16
    assert evidence["workers"] == 2
    assert (evidence["optimizer_updates"] > 0) == training_mode
    if training_mode:
        assert evidence["ranking_loss_mean"] > 0
    else:
        assert evidence["ranking_loss_mean"] is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not model.dropout.training
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(torch.equal(left, right) for left, right in zip(
        torch.cuda.get_rng_state_all(), gpu_rng, strict=True
    ))
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
def test_cuda_complete_input_order():
    source = ProbeDataset()
    loader = DataLoader(source, batch_size=7, num_workers=2, prefetch_factor=2,
                        multiprocessing_context="spawn", pin_memory=True,
                        collate_fn=training.ModelBatchCollator())
    completed = []
    for batch in training.iter_device_batches(loader, torch.device("cuda")):
        assert batch["market_ids"].device.type == "cuda"
        completed.extend(batch["sample_ids"])
    torch.cuda.synchronize()
    assert completed == [str(index) for index in range(len(source))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires isolated cloud CUDA runner")
@pytest.mark.parametrize("oom", [False, True])
def test_cuda_probe_failure_restores_state_and_stops_workers(monkeypatch, oom):
    import multiprocessing

    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.training.gradient_accumulation_steps = 2
    config.model.lora.enabled = False
    model = ProbeModel().cuda()
    source = ProbeDataset()
    monkeypatch.setattr(training, "_training_sampler", lambda *_args: BlockwisePermutationSampler(
        len(source), seed=77, block_size=128
    ))
    original_forward = training.forward_batch
    calls = 0

    def fail_after_optimizer_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            if oom:
                raise torch.cuda.OutOfMemoryError("injected CUDA OOM")
            raise RuntimeError("injected worker/compute failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(training, "forward_batch", fail_after_optimizer_step)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    children = {child.pid for child in multiprocessing.active_children()}
    context = nullcontext() if oom else pytest.raises(RuntimeError, match="injected")
    with context:
        result = training._measure_cuda_batch(
            bundle=SimpleNamespace(model=model), sample=source[0], batch_size=4,
            config=config, device=torch.device("cuda"), training=True,
            optimizer_state_reserve_bytes=0, device_memory_limit_bytes=1024**3,
            dataset=source, worker_plan=training.plan_dataloader_workers(
                2, visible_cpu_count=4, available_memory_bytes=16 * 1024**3,
            ),
        )
        assert not result.accepted and result.outcome == "cuda_out_of_memory"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {child.pid for child in multiprocessing.active_children()} <= children
