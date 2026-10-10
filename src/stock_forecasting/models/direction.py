"""Horizon-specific absolute and relative direction supervision."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DIRECTION_CLASSES = ("down", "neutral", "up")


def direction_labels(targets: Tensor, bands: Tensor) -> Tensor:
    if targets.ndim != 2 or bands.shape != (targets.shape[0], 2):
        raise ValueError("Direction labels require [batch,horizon] returns and [batch,2] bands")
    return torch.where(targets < bands[:, 0:1], 0, torch.where(targets > bands[:, 1:2], 2, 1))


def direction_cost_features(bands: Tensor) -> Tensor:
    """Bound predictor covariates, never the exact label thresholds or returns."""
    return bands.tanh() * 100


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.float().reshape(-1, 3), labels.reshape(-1))


class DualDirectionHead(nn.Module):
    """Learn a separate horizon query without changing alpha's median or ranking."""

    def __init__(self, dimension: int, horizons: int, config, dropout: float) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Embedding(horizons, dimension)
        self.market = nn.Embedding(4, dimension)
        self.stream = nn.Parameter(torch.zeros(3, dimension))
        self.cost_projection = nn.Linear(2, dimension, bias=False)
        self.attention = nn.MultiheadAttention(
            dimension, config.attention_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dimension)
        self.absolute = nn.Linear(dimension, 3)
        self.alpha = nn.Linear(dimension, 3)
        self.register_buffer(
            "loss_references", torch.tensor([1.0, math.log(3), math.log(3)]), persistent=False
        )
        self.register_buffer("reference_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def read(
        self,
        asset: Tensor,
        benchmark: Tensor,
        conditioned: Tensor,
        market_ids: Tensor,
        bands: Tensor,
    ) -> Tensor:
        context = torch.cat(
            [tokens + self.stream[i] for i, tokens in enumerate((asset, benchmark, conditioned))], 1
        )
        query = (
            self.query.weight[None]
            + self.market(market_ids)[:, None]
            + self.cost_projection(
                direction_cost_features(bands).to(self.cost_projection.weight.dtype)
            )[:, None]
        )
        hidden, _ = self.attention(query, context, context, need_weights=False)
        return self.norm(query + hidden)

    def logits(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.absolute(hidden).float(), self.alpha(hidden).float()

    def weights(self) -> Tensor:
        return (
            self.loss_references[0]
            / self.loss_references[1:]
            * self.loss_references.new_tensor(
                [self.config.absolute_weight, self.config.alpha_weight]
            )
        )

    def reference_state(self) -> dict:
        return {
            "schema_version": 1,
            "samples": int(self.reference_count.cpu()),
            "references": self.loss_references.float().cpu().tolist(),
        }

    def restore_references(self, state: dict) -> None:
        values = state.get("references", [])
        if (
            state.get("schema_version") != 1
            or len(values) != 3
            or any(not math.isfinite(v) or v <= 0 for v in values)
            or not isinstance(state.get("samples"), int)
            or state["samples"] <= 0
        ):
            raise ValueError("Invalid train-only direction loss reference state")
        self.loss_references.copy_(self.loss_references.new_tensor(values))
        self.reference_count.fill_(state["samples"])

    def losses(
        self,
        hidden: Tensor,
        logits: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
        bands: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        labels = tuple(direction_labels(target, bands) for target in targets)
        clean = tuple(
            classification_loss(logit, label) for logit, label in zip(logits, labels, strict=True)
        )
        adversarial = None
        if self.training and torch.is_grad_enabled() and self.config.representation_weight:
            normalized = clean[0] / self.loss_references[1] + clean[1] / self.loss_references[2]
            gradient = (
                torch.autograd.grad(normalized, hidden, retain_graph=True)[0].detach().float()
            )
            radius = self.config.representation_radius * math.sqrt(hidden.shape[-1])
            delta = radius * gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            perturbed = self.logits(hidden + delta.to(hidden.dtype))
            adversarial = 0.5 * sum(
                classification_loss(logit, label) / reference
                for logit, label, reference in zip(
                    perturbed, labels, self.loss_references[1:], strict=True
                )
            )
        return clean[0], clean[1], adversarial
