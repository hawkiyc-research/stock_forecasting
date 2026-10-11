"""Horizon-specific absolute and relative direction supervision."""

from __future__ import annotations

import copy
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


class MarketResidualDualDirectionHead(DualDirectionHead):
    """Separate train priors from historical signal and retain population logits.

    The historical branch emits f. Population logits are z=f+tau*log(prior),
    and the Menon decision logits are f=z-tau*log(prior). Training uses CE(z)
    on the natural class population, never a second inverse-class multiplier.
    Market weights are fixed from train and independent of microbatch contents.
    """

    architecture = "market_residual_v2"

    def __init__(self, dimension: int, horizons, config, dropout: float) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.horizons = tuple(int(h) for h in horizons)
        self.query = nn.Embedding(len(self.horizons), dimension)
        nn.init.normal_(self.query.weight, std=dimension ** -0.5)
        self.stream = nn.Parameter(torch.zeros(3, dimension))
        self.attention = nn.MultiheadAttention(
            dimension, config.attention_heads, dropout=dropout, batch_first=True
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(24, config.market_residual_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(config.market_residual_hidden_dim, dimension),
        )
        self.norm = nn.LayerNorm(dimension)
        self.absolute = nn.Linear(dimension, 3)
        self.alpha = nn.Linear(dimension, 3)
        self.market_signal = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, config.market_residual_hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(config.market_residual_hidden_dim, 4 * 2 * 3),
        )
        nn.init.zeros_(self.market_signal[-1].weight)
        nn.init.zeros_(self.market_signal[-1].bias)
        self.register_buffer(
            "loss_references", torch.tensor([1.0, math.log(3), math.log(3)]), persistent=False
        )
        self.register_buffer("reference_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer(
            "train_priors", torch.full((4, 2, len(self.horizons), 3), 1 / 3), persistent=False
        )
        self.register_buffer("train_market_weights", torch.ones(4), persistent=False)
        self.register_buffer("horizon_values", torch.tensor(self.horizons), persistent=False)
        self._reference_metadata: dict = {}
        self.pinball_guard_reference: float | None = None
        self.pinball_guard_basis: str | None = None

    def runtime_float32(self) -> None:
        """Preserve runtime normalization precision after compact model conversion."""
        device = self.query.weight.device
        self.loss_references = torch.tensor(
            [1.0, math.log(3), math.log(3)], dtype=torch.float32, device=device
        )
        self.train_priors = torch.full(
            (4, 2, len(self.horizons), 3), 1 / 3, dtype=torch.float32, device=device
        )
        self.train_market_weights = torch.ones(4, dtype=torch.float32, device=device)

    def read(
        self, asset: Tensor, benchmark: Tensor, conditioned: Tensor,
        market_ids: Tensor, bands: Tensor, *, standardized_scales: Tensor,
        scale_features: Tensor,
    ) -> Tensor:
        if standardized_scales.shape != (asset.shape[0], 20) or scale_features.shape != (
            asset.shape[0], 20
        ):
            raise ValueError("Direction v2 requires the existing 20 past-only scale features")
        context = torch.cat(
            [tokens + self.stream[i] for i, tokens in enumerate((asset, benchmark, conditioned))], 1
        )
        query = self.query.weight[None].expand(asset.shape[0], -1, -1)
        attended, _ = self.attention(query, context, context, need_weights=False)
        # The decision hurdle depends on return scale as well as its raw fee value.
        # An unreachable lower band stays exact in labels and finite in predictors.
        volatility = scale_features[:, (0, 2)].float().clamp_min(1e-8)
        volatility = volatility[:, None, :, None] * self.horizon_values.float().sqrt()[
            None, :, None, None
        ]
        ratios = bands.float()[:, None, None, :] / volatility
        ratios = torch.nan_to_num(ratios, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
        numerical = torch.cat(
            (
                standardized_scales.float()[:, None].expand(-1, len(self.horizons), -1),
                ratios.asinh().flatten(-2),
            ), -1,
        )
        numerical = self.numeric_projection(numerical.to(self.numeric_projection[0].weight.dtype))
        # Queries route attention only. Market/cost metadata has no output residual shortcut.
        return self.norm(attended + numerical)

    def logits(self, hidden: Tensor, market_ids: Tensor) -> tuple[Tensor, Tensor]:
        common = torch.stack((self.absolute(hidden), self.alpha(hidden)), dim=2).float()
        residual = self.market_signal(hidden).reshape(*hidden.shape[:2], 4, 2, 3)
        index = market_ids[:, None, None, None, None].expand(-1, hidden.shape[1], 1, 2, 3)
        historical = common + residual.gather(2, index).squeeze(2).float()
        prior = self.train_priors[market_ids].permute(0, 2, 1, 3).float().log()
        population = historical + self.config.class_balance_power * prior
        return population[:, :, 0], population[:, :, 1]

    def decision_logits(
        self, logits: tuple[Tensor, Tensor], market_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        prior = self.train_priors[market_ids].float().log()
        return tuple(
            logit.float() - self.config.class_balance_power * prior[:, task]
            for task, logit in enumerate(logits)
        )

    def reference_state(self) -> dict:
        return {
            **copy.deepcopy(self._reference_metadata),
            "schema_version": 2,
            "architecture": self.architecture,
            "samples": int(self.reference_count.cpu()),
            "references": self.loss_references.float().cpu().tolist(),
            "priors": self.train_priors.float().cpu().tolist(),
            "market_weights": self.train_market_weights.float().cpu().tolist(),
            "pinball_guard_reference": self.pinball_guard_reference,
            "pinball_guard_basis": self.pinball_guard_basis,
        }

    def set_pinball_guard_reference(self, value: float) -> None:
        """Freeze the first complete validation anchor once; never update on test."""
        if not math.isfinite(value) or value <= 0:
            raise ValueError("The full-validation pinball guard must be finite and positive")
        if self.pinball_guard_reference is not None:
            raise ValueError("The full-validation pinball guard is already frozen")
        self.pinball_guard_reference = float(value)
        self.pinball_guard_basis = "first_full_validation"

    def restore_references(self, state: dict) -> None:
        values = state.get("references", [])
        samples = state.get("samples")
        if (
            state.get("schema_version") != 2
            or state.get("architecture", self.architecture) != self.architecture
            or len(values) != 3
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values)
            or not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0
        ):
            raise ValueError("Invalid direction v2 train-only reference state")
        device = self.query.weight.device
        priors = torch.tensor(state.get("priors", []), dtype=torch.float32, device=device)
        weights = torch.tensor(state.get("market_weights", []), dtype=torch.float32, device=device)
        if (
            priors.shape != (4, 2, len(self.horizons), 3)
            or not bool((torch.isfinite(priors) & (priors > 0)).all())
            or not torch.allclose(priors.sum(-1), torch.ones_like(priors[..., 0]), atol=1e-6, rtol=0)
            or weights.shape != (4,)
            or not bool((torch.isfinite(weights) & (weights > 0)).all())
        ):
            raise ValueError("Direction v2 priors and market weights must match the train population")
        guard = state.get("pinball_guard_reference")
        if guard is not None and (
            not isinstance(guard, (int, float)) or isinstance(guard, bool)
            or not math.isfinite(guard) or guard <= 0
        ):
            raise ValueError("Direction v2 pinball guard reference must be finite and positive")
        if (guard is None) != (state.get("pinball_guard_basis") is None) or (
            guard is not None and state.get("pinball_guard_basis") != "first_full_validation"
        ):
            raise ValueError("Direction v2 pinball guard must come from the first full validation")
        self.loss_references = torch.tensor(values, dtype=torch.float32, device=device)
        self.reference_count.fill_(samples)
        self.train_priors = priors
        self.train_market_weights = weights
        self.pinball_guard_reference = guard
        self.pinball_guard_basis = state.get("pinball_guard_basis")
        owned = {
            "schema_version", "architecture", "samples", "references", "priors",
            "market_weights", "pinball_guard_reference", "pinball_guard_basis",
        }
        self._reference_metadata = copy.deepcopy({k: v for k, v in state.items() if k not in owned})

    def _task_loss(self, logits: Tensor, labels: Tensor, market_ids: Tensor) -> Tensor:
        losses = F.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape_as(labels)
        return (losses * self.train_market_weights[market_ids, None]).mean()

    def losses(
        self, hidden: Tensor, logits: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor], bands: Tensor, *, market_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        labels = tuple(direction_labels(target, bands) for target in targets)
        clean = tuple(
            self._task_loss(logit, label, market_ids)
            for logit, label in zip(logits, labels, strict=True)
        )
        adversarial = None
        if self.training and torch.is_grad_enabled() and self.config.representation_weight:
            normalized = clean[0] / self.loss_references[1] + clean[1] / self.loss_references[2]
            gradient = torch.autograd.grad(normalized, hidden, retain_graph=True)[0].detach().float()
            radius = self.config.representation_radius * math.sqrt(hidden.shape[-1])
            delta = radius * gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            perturbed = self.logits(hidden + delta.to(hidden.dtype), market_ids)
            adversarial = 0.5 * sum(
                self._task_loss(logit, label, market_ids) / reference
                for logit, label, reference in zip(
                    perturbed, labels, self.loss_references[1:], strict=True
                )
            )
        return clean[0], clean[1], adversarial
