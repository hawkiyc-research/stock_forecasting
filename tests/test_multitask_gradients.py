"""Cloud numerical checks for LoRA-only conflict projection and accumulation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stock_forecasting.multitask_gradients import SharedTaskGradientManager

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA"),
    ),
]


def make_parameters(device="cpu"):
    shared = nn.Parameter(torch.tensor([1.0, 2.0], device=device))
    private = nn.Parameter(torch.tensor([3.0], device=device))
    technical = nn.Parameter(torch.tensor([4.0], device=device))
    named = [
        ("encoder.attn.lora_a.weight", shared),
        ("head.weight", private),
        ("technical_adapter.weight", technical),
    ]
    return shared, private, technical, named


@pytest.mark.parametrize("device", DEVICES)
def test_negative_conflict_projects_only_lora_and_retains_private_regularization(device):
    shared, private, technical, named = make_parameters(device)
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    losses = {
        "regression": shared[0] + private.sum(),
        "absolute": -shared[0] + shared[1] + technical.sum(),
    }
    total = sum(losses.values()) + 0.2 * shared.square().sum() + private.square().sum()
    correction = manager.prepare_correction(losses)
    total.backward()
    manager.apply_correction(correction)
    # [1,0] projects to [.5,.5], while [-1,1] projects to [0,1].
    torch.testing.assert_close(shared.grad, torch.tensor([0.9, 2.3], device=device))
    torch.testing.assert_close(private.grad, torch.tensor([7.0], device=device))
    torch.testing.assert_close(technical.grad, torch.tensor([1.0], device=device))
    diagnostics = manager.optimizer_step_completed()
    assert diagnostics["gradient/conflict_fraction/regression__absolute"] == 1
    assert diagnostics["gradient/projected_pairs_per_microbatch"] == 2
    with pytest.raises(ValueError, match="applied once"):
        manager.apply_correction(correction)


@pytest.mark.parametrize("device", DEVICES)
def test_no_conflict_is_exactly_ordinary_backward(device):
    shared, private, technical, named = make_parameters(device)
    losses = {
        "regression": shared.sum() + private.sum(),
        "absolute": 2 * shared.sum() + technical.sum(),
    }
    total = sum(losses.values())
    expected = torch.autograd.grad(total, (shared, private, technical), retain_graph=True)
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    correction = manager.prepare_correction(losses)
    total.backward()
    manager.apply_correction(correction)
    for parameter, gradient in zip((shared, private, technical), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert manager.optimizer_step_completed()["gradient/projected_pairs_per_microbatch"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_microbatch_accumulation_and_partial_divisor_preserve_prior_gradients(device):
    shared, private, _, named = make_parameters(device)
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    expected = shared.new_zeros(2)
    for factor, divisor in [(1.0, 2), (3.0, 2), (2.0, 1)]:
        losses = {
            "regression": factor * shared[0] + private.sum(),
            "absolute": factor * (-shared[0] + shared[1]),
        }
        correction = manager.prepare_correction(losses, divisor=divisor)
        (sum(losses.values()) / divisor).backward()
        manager.apply_correction(correction)
        expected += shared.new_tensor([0.5, 1.5]) * factor / divisor
    torch.testing.assert_close(shared.grad, expected)
    torch.testing.assert_close(private.grad, private.new_tensor([2.0]))
    assert manager.optimizer_step_completed()["gradient/projected_pairs_per_microbatch"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_unused_parameters_and_zero_task_are_supported(device):
    shared, _, _, named = make_parameters(device)
    unused = nn.Parameter(torch.ones(3, device=device))
    named.append(("encoder.ffn.lora_b.weight", unused))
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    losses = {"regression": shared.sum(), "absolute": shared.new_zeros(())}
    correction = manager.prepare_correction(losses)
    sum(losses.values()).backward()
    manager.apply_correction(correction)
    assert unused.grad is None
    torch.testing.assert_close(shared.grad, torch.ones_like(shared))
    diagnostics = manager.optimizer_step_completed()
    assert diagnostics["gradient/norm/absolute"] == 0
    assert diagnostics["gradient/cosine/regression__absolute"] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_task_is_not_masked_and_cannot_update_parameters(device, invalid):
    shared, _, _, named = make_parameters(device)
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    optimizer = torch.optim.AdamW([shared], lr=0.1)
    original = shared.detach().clone()
    losses = {"regression": shared.sum(), "absolute": shared.sum() * invalid}
    correction = manager.prepare_correction(losses)
    sum(losses.values()).backward()
    manager.apply_correction(correction)
    assert not torch.isfinite(shared.grad).all()
    # Production and the CUDA probe both reject the norm before optimizer.step.
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_([shared], 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.testing.assert_close(shared, original, rtol=0, atol=0)
    assert not optimizer.state
    assert manager.optimizer_step_completed()["gradient/nonfinite_microbatch_fraction"] == 1


def test_monitor_samples_only_due_optimizer_step_and_does_not_modify_gradients():
    shared, _, _, named = make_parameters()
    manager = SharedTaskGradientManager(named, mode="monitor", diagnostics_interval=2)
    for step in range(4):
        losses = {"regression": shared[0], "absolute": -shared[0] + shared[1]}
        assert manager.prepare_correction(losses) is None
        sum(losses.values()).backward()
        torch.testing.assert_close(shared.grad, shared.new_tensor([0.0, 1.0]))
        shared.grad = None
        diagnostics = manager.optimizer_step_completed()
        assert (diagnostics is not None) == ((step + 1) % 2 == 0)


def test_selection_memory_budget_and_task_contract_are_bounded():
    shared, _, _, named = make_parameters()
    manager = SharedTaskGradientManager(named, max_extra_bytes=1)
    assert manager.names == ("encoder.attn.lora_a.weight",)
    assert manager.estimated_extra_bytes(3) == 56
    with pytest.raises(ValueError, match="memory budget"):
        manager.prepare_correction({"a": shared.sum(), "b": shared.sum()})
    with pytest.raises(ValueError, match="trainable LoRA"):
        SharedTaskGradientManager([("head.weight", shared)])
    manager = SharedTaskGradientManager(named)
    with pytest.raises(ValueError, match="task range"):
        manager.prepare_correction({str(i): shared.sum() for i in range(4)})
    with pytest.raises(ValueError, match="divisor"):
        manager.prepare_correction({"a": shared.sum(), "b": shared.sum()}, divisor=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cloud CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_scaler_accumulation_and_optimizer_update(dtype):
    shared, private, _, named = make_parameters("cuda")
    manager = SharedTaskGradientManager(named, diagnostics_interval=1)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128)
    optimizer = torch.optim.SGD([shared, private], lr=0.1)
    for _ in range(2):
        with torch.autocast("cuda", dtype=dtype):
            losses = {
                "regression": shared @ shared.new_tensor([1.0, 0.0]) + private.sum(),
                "absolute": shared @ shared.new_tensor([-1.0, 1.0]),
            }
            total = sum(losses.values())
        correction = manager.prepare_correction(losses, divisor=2, scaler=scaler)
        scaler.scale(total / 2).backward()
        manager.apply_correction(correction)
    scaler.unscale_(optimizer)
    torch.testing.assert_close(shared.grad, shared.new_tensor([0.5, 1.5]))
    torch.testing.assert_close(private.grad, private.new_tensor([1.0]))
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(shared, shared.new_tensor([0.95, 1.85]))
    assert manager.optimizer_step_completed()["gradient/projected_pairs_per_microbatch"] == 2


def test_scaler_scale_is_read_without_get_scale_host_sync():
    shared, _, _, named = make_parameters()
    manager = SharedTaskGradientManager(named)
    scaler = SimpleNamespace(scale=lambda value: value * 32)
    losses = {"regression": shared[0], "absolute": -shared[0] + shared[1]}
    correction = manager.prepare_correction(losses, divisor=2, scaler=scaler)
    scaler.scale(sum(losses.values()) / 2).backward()
    manager.apply_correction(correction)
    torch.testing.assert_close(shared.grad / 32, shared.new_tensor([0.25, 0.75]))


def test_checkpoint_resumes_three_task_projection_order_exactly():
    shared, _, _, named = make_parameters()
    manager = SharedTaskGradientManager(named, diagnostics_interval=1, seed=42)
    losses = {"regression": shared[0], "absolute": -shared[0] + shared[1], "alpha": -shared[1]}
    correction = manager.prepare_correction(losses)
    sum(losses.values()).backward()
    manager.apply_correction(correction)
    manager.optimizer_step_completed()
    shared.grad = None
    restored = SharedTaskGradientManager(named, diagnostics_interval=1, seed=42)
    restored.load_state_dict(manager.state_dict())
    continued_losses = {
        "regression": shared[0],
        "absolute": -shared[0] + shared[1],
        "alpha": -shared[1],
    }
    expected = manager.prepare_correction(continued_losses)
    actual = restored.prepare_correction(continued_losses)
    for left, right in zip(expected.values, actual.values, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert restored.microbatches == manager.microbatches
