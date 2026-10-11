"""Measure task gradients and apply PCGrad only to shared trainable LoRA weights."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class SharedGradientCorrection:
    """A single-use, scaled microbatch correction for the owning manager."""

    owner: int
    values: tuple[Tensor, ...]
    applied: bool = False


def _dot(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> Tensor:
    # Reduce individual tensors without concatenating a model-sized flat buffer.
    return torch.stack([torch.sum(a * b) for a, b in zip(left, right, strict=True)]).sum()


class SharedTaskGradientManager:
    """Correct shared gradients while leaving ordinary backward for all private weights.

    Call prepare_correction before total_loss/divisor.backward, then apply_correction
    immediately afterwards for each microbatch. With GradScaler, use its ordinary
    scale(total_loss/divisor).backward; the returned correction has the same scale.
    Projection is per microbatch, and therefore preserves accumulation and partial
    final accumulation divisors without overwriting previously accumulated gradients.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        mode: str = "pcgrad",
        diagnostics_interval: int = 100,
        seed: int = 0,
        max_tasks: int = 3,
        max_extra_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if mode not in {"monitor", "pcgrad"}:
            raise ValueError("Shared gradient mode must be monitor or pcgrad")
        if diagnostics_interval < 1 or max_tasks < 2 or max_extra_bytes < 1:
            raise ValueError("Gradient diagnostics and memory limits must be positive")
        selected = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
            and any(part.lower() in {"lora_a", "lora_b"} for part in name.split("."))
        ]
        if not selected:
            raise ValueError("Shared gradient management requires trainable LoRA parameters")
        if len({id(parameter) for _, parameter in selected}) != len(selected):
            raise ValueError("Shared LoRA parameters must not be duplicated")
        self.names = tuple(name for name, _ in selected)
        self.parameters = tuple(parameter for _, parameter in selected)
        self.device = self.parameters[0].device
        if any(parameter.device != self.device for parameter in self.parameters):
            raise ValueError("Shared LoRA parameters must use one device")
        self.numel = sum(parameter.numel() for parameter in self.parameters)
        self.mode = mode
        self.diagnostics_interval = diagnostics_interval
        self.seed = seed
        self.max_tasks = max_tasks
        self.max_extra_bytes = max_extra_bytes
        self.optimizer_steps = 0
        self.microbatches = 0
        self.task_names: tuple[str, ...] | None = None
        self._diagnostic_sum: Tensor | None = None
        self._diagnostic_count = 0

    def estimated_extra_bytes(self, task_count: int = 3) -> int:
        # Original tasks, one projected task, correction, and two temporary buffers.
        return (task_count + 4) * self.numel * 4

    def _diagnostics_due(self) -> bool:
        return (self.optimizer_steps + 1) % self.diagnostics_interval == 0

    def state_dict(self) -> dict[str, Any]:
        """Persist ordering counters at the existing optimizer-step checkpoint boundary."""
        if self._diagnostic_count:
            raise ValueError("Save shared gradient state after completing the optimizer step")
        return {
            "schema_version": 1,
            "mode": self.mode,
            "seed": self.seed,
            "optimizer_steps": self.optimizer_steps,
            "microbatches": self.microbatches,
            "task_names": None if self.task_names is None else list(self.task_names),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore deterministic task-projection order without checkpointing gradients."""
        if (state.get("schema_version") != 1 or state.get("mode") != self.mode
                or state.get("seed") != self.seed):
            raise ValueError("Incompatible shared gradient checkpoint state")
        for field in ("optimizer_steps", "microbatches"):
            value = state.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Invalid shared gradient checkpoint counter")
        names = state.get("task_names")
        if names is not None and (
            not isinstance(names, list) or not 2 <= len(names) <= self.max_tasks
            or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid shared gradient checkpoint tasks")
        self.optimizer_steps = state["optimizer_steps"]
        self.microbatches = state["microbatches"]
        self.task_names = None if names is None else tuple(names)
        self._diagnostic_sum = None
        self._diagnostic_count = 0

    def prepare_correction(
        self,
        losses: Mapping[str, Tensor],
        *,
        divisor: float = 1.0,
        scaler: Any | None = None,
    ) -> SharedGradientCorrection | None:
        """Inspect weighted task losses; retain their graph for ordinary backward."""
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("Gradient accumulation divisor must be finite and positive")
        names = tuple(losses)
        if not 2 <= len(names) <= self.max_tasks:
            raise ValueError("Shared task count exceeds the configured bounded task range")
        if self.task_names is not None and names != self.task_names:
            raise ValueError("Shared task names and order must remain stable")
        self.task_names = names
        self.microbatches += 1
        collect = self._diagnostics_due()
        if self.mode == "monitor" and not collect:
            return None
        if self.estimated_extra_bytes(len(names)) > self.max_extra_bytes:
            raise ValueError("Shared gradient buffers exceed the configured memory budget")
        for loss in losses.values():
            if loss.numel() != 1 or loss.device != self.device:
                raise ValueError("Weighted task losses must be scalar tensors on the LoRA device")

        # scale(ones) reads GradScaler's device scale without a per-batch get_scale sync.
        scale = torch.ones((), device=self.device, dtype=torch.float32)
        if scaler is not None:
            scale = scaler.scale(scale)
        originals: list[tuple[Tensor, ...]] = []
        for loss in losses.values():
            if loss.requires_grad:
                scaled = loss if scaler is None else scaler.scale(loss)
                raw = torch.autograd.grad(
                    scaled, self.parameters, retain_graph=True, allow_unused=True
                )
            else:
                raw = (None,) * len(self.parameters)
            originals.append(
                tuple(
                    torch.zeros_like(parameter, dtype=torch.float32)
                    if gradient is None
                    else gradient.detach().float() / scale
                    for parameter, gradient in zip(self.parameters, raw, strict=True)
                )
            )
            del raw
        norms_squared = torch.stack([_dot(gradient, gradient) for gradient in originals])
        valid = torch.isfinite(norms_squared).all()
        pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
        pair_dots = torch.stack([_dot(originals[i], originals[j]) for i, j in pairs])
        valid = valid & torch.isfinite(pair_dots).all()
        projected_pairs = scale.new_zeros(())

        correction = None
        if self.mode == "pcgrad":
            delta = tuple(torch.zeros_like(parameter, dtype=torch.float32)
                          for parameter in self.parameters)
            rng = random.Random(self.seed + self.microbatches)
            for i, original in enumerate(originals):
                projected = tuple(gradient.clone() for gradient in original)
                others = [j for j in range(len(names)) if j != i]
                rng.shuffle(others)
                for j in others:
                    dot = _dot(projected, originals[j])
                    negative = valid & torch.isfinite(dot) & (dot < 0)
                    coefficient = torch.where(
                        negative,
                        dot / norms_squared[j].clamp_min(torch.finfo(torch.float32).tiny),
                        dot.new_zeros(()),
                    )
                    projected_pairs = projected_pairs + negative.float()
                    for current, other in zip(projected, originals[j], strict=True):
                        current.sub_(coefficient * other)
                for destination, current, source in zip(delta, projected, original, strict=True):
                    destination.add_(current - source)
                del projected
            # Invalid task gradients receive no correction. Ordinary backward retains
            # its non-finite gradients for the existing scaler/error handling.
            delta = tuple(torch.where(valid, value, torch.zeros_like(value)) for value in delta)
            for value in delta:
                value.mul_(scale / divisor)
            correction = SharedGradientCorrection(id(self), delta)

        if collect:
            cosines = torch.stack([
                dot / (norms_squared[i].sqrt() * norms_squared[j].sqrt()).clamp_min(1e-12)
                for dot, (i, j) in zip(pair_dots, pairs, strict=True)
            ]).clamp(-1, 1)
            metrics = torch.cat([
                norms_squared.sqrt(), cosines,
                (pair_dots < 0).float(),
                torch.stack([projected_pairs, (~valid).float()]),
            ]).detach()
            self._diagnostic_sum = (
                metrics if self._diagnostic_sum is None else self._diagnostic_sum + metrics
            )
            self._diagnostic_count += 1
        return correction

    def apply_correction(self, correction: SharedGradientCorrection | None) -> None:
        """Add only this microbatch's LoRA delta to existing accumulated gradients."""
        if correction is None:
            return
        if correction.owner != id(self) or correction.applied:
            raise ValueError("Shared gradient correction must be owned and applied once")
        with torch.no_grad():
            for parameter, delta in zip(self.parameters, correction.values, strict=True):
                if parameter.grad is not None:
                    parameter.grad.add_(delta.to(parameter.grad.dtype))
        correction.applied = True

    def optimizer_step_completed(self) -> dict[str, float] | None:
        """Export diagnostic scalars at a bounded optimizer-step interval only."""
        self.optimizer_steps += 1
        if self.optimizer_steps % self.diagnostics_interval or self._diagnostic_sum is None:
            return None
        assert self.task_names is not None
        count = len(self.task_names)
        pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
        values = (self._diagnostic_sum / self._diagnostic_count).cpu().tolist()
        result = {
            f"gradient/norm/{name}": values[i] for i, name in enumerate(self.task_names)
        }
        for offset, (i, j) in enumerate(pairs):
            suffix = f"{self.task_names[i]}__{self.task_names[j]}"
            result[f"gradient/cosine/{suffix}"] = values[count + offset]
            result[f"gradient/conflict_fraction/{suffix}"] = values[count + len(pairs) + offset]
        result["gradient/projected_pairs_per_microbatch"] = values[-2]
        result["gradient/nonfinite_microbatch_fraction"] = values[-1]
        result["gradient/shared_lora_numel"] = float(self.numel)
        result["gradient/extra_memory_upper_bytes"] = float(self.estimated_extra_bytes(count))
        self._diagnostic_sum = None
        self._diagnostic_count = 0
        return result
