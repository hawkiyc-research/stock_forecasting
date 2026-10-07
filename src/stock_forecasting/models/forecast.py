"""Dynamic benchmark conditioning and multi-horizon alpha quantile head."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stock_forecasting.data.horizons import (
    DEFAULT_ALPHA_HORIZONS,
    validate_alpha_horizons,
)
from stock_forecasting.models.scale_features import NumericalResidualBranch


class GatedBenchmarkConditioner(nn.Module):
    """Condition asset latents on historical benchmark latents with a learned gate."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_dim = hidden_dim
        self.asset_norm = nn.LayerNorm(hidden_dim)
        self.benchmark_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        gate_linear = cast(nn.Linear, self.gate[1])
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -2.0)

    def forward(self, asset_tokens: Tensor, benchmark_tokens: Tensor) -> tuple[Tensor, Tensor]:
        expected_rank = asset_tokens.ndim == benchmark_tokens.ndim == 3
        if (
            not expected_rank
            or asset_tokens.shape[0] != benchmark_tokens.shape[0]
            or asset_tokens.shape[-1] != self.hidden_dim
            or benchmark_tokens.shape[-1] != self.hidden_dim
        ):
            raise ValueError("Conditioner inputs must be paired [batch, tokens, hidden] tensors")
        attended, _ = self.cross_attention(
            query=self.asset_norm(asset_tokens),
            key=self.benchmark_norm(benchmark_tokens),
            value=self.benchmark_norm(benchmark_tokens),
            need_weights=False,
        )
        gate = self.gate(torch.cat([asset_tokens, attended], dim=-1))
        conditioned = self.output_norm(asset_tokens + gate * self.dropout(attended))
        return cast(Tensor, conditioned), cast(Tensor, gate)


class MultiHorizonAlphaHead(nn.Module):
    """Predict ordered q10/q50/q90 benchmark-relative log-return quantiles."""

    def __init__(
        self,
        input_dim: int,
        *,
        horizons: Sequence[int] = DEFAULT_ALPHA_HORIZONS,
        hidden_dim: int | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        robust_scales: Sequence[float] | None = None,
        dropout: float = 0.0,
        feature_mode: str = "baseline",
        fp32_head: bool = False,
        market_aware: bool = False,
        explicit_output_scale: bool = False,
        decoupled_output_scale: bool = False,
        independent_ranking_head: bool = False,
        market_residual_hidden_dim: int = 0,
        ranking_numeric_features: bool = False,
    ) -> None:
        super().__init__()
        ordered_horizons = validate_alpha_horizons(horizons)
        ordered_quantiles = tuple(float(value) for value in quantiles)
        if ordered_quantiles != (0.1, 0.5, 0.9):
            raise ValueError("MultiHorizonAlphaHead requires quantiles (0.1, 0.5, 0.9)")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        hidden_dim = hidden_dim or max(32, input_dim // 2)
        scales = tuple(float(value) for value in (robust_scales or [1.0] * len(ordered_horizons)))
        if len(scales) != len(ordered_horizons) or any(
            not math.isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError("robust_scales must contain one finite positive value per horizon")

        self.input_dim = input_dim
        self.horizons = ordered_horizons
        self.quantiles = ordered_quantiles
        self.horizon_embeddings = nn.Embedding(len(ordered_horizons), input_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quantile_parameters = nn.Linear(hidden_dim, 3)
        self.explicit_output_scale = explicit_output_scale
        self.decoupled_output_scale = decoupled_output_scale
        if decoupled_output_scale and not explicit_output_scale:
            raise ValueError("Decoupled widths require historical output scales")
        self.ranking_head = nn.Linear(hidden_dim, 1) if independent_ranking_head else None
        if explicit_output_scale and feature_mode not in ("scales", "combined"):
            raise ValueError("Explicit output scale requires historical scale features")
        self.market_embedding = nn.Embedding(4, input_dim) if market_aware else None
        if self.market_embedding is not None:
            nn.init.zeros_(self.market_embedding.weight)
        self.scale_gate = (
            nn.Linear(hidden_dim, 2 if decoupled_output_scale else 1)
            if explicit_output_scale
            else None
        )
        if self.scale_gate is not None:
            nn.init.zeros_(self.scale_gate.weight)
            nn.init.zeros_(self.scale_gate.bias)
        self.fp32_head = fp32_head or feature_mode != "baseline"
        self.numeric_branch = (
            None
            if feature_mode == "baseline"
            else NumericalResidualBranch(
                hidden_dim, input_dim, feature_mode, extended=explicit_output_scale
            )
        )
        feature_dim = 0 if self.numeric_branch is None else self.numeric_branch.encoded_dim
        if ranking_numeric_features and (self.ranking_head is None or not feature_dim):
            raise ValueError("Ranking feature fusion requires numerical features and ranking head")
        self.ranking_feature_head = (
            nn.Sequential(
                nn.Linear(feature_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1)
            )
            if ranking_numeric_features else None
        )
        if market_residual_hidden_dim and self.market_embedding is None:
            raise ValueError("Market residual heads require explicit market conditioning")
        self.market_residual = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim + feature_dim),
                nn.Linear(hidden_dim + feature_dim, market_residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(market_residual_hidden_dim, 4 * 3),
            )
            if market_residual_hidden_dim else None
        )
        for branch in (self.market_residual, self.ranking_feature_head):
            if branch is not None:
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
        self.register_buffer(
            "robust_scales",
            torch.tensor(scales, dtype=torch.float32),
            persistent=False,
        )
        # Integer buffers survive model dtype conversions. Store the exact FP32
        # level bits: GPU division by ten rounds q90 differently from the literal.
        # Nonpersistent buffers keep existing checkpoint state keys intact.
        self.register_buffer(
            "_horizon_values", torch.tensor(ordered_horizons, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_quantile_level_bits",
            torch.tensor(ordered_quantiles, dtype=torch.float32).view(torch.int32),
            persistent=False,
        )

    def forward(
        self,
        conditioned_tokens: Tensor,
        *,
        scale_features: Tensor | None = None,
        benchmark_tokens: Tensor | None = None,
        market_ids: Tensor | None = None,
        return_ranking: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        if self.fp32_head:
            with torch.autocast(device_type=conditioned_tokens.device.type, enabled=False):
                return self._forward(
                    conditioned_tokens.float(),
                    scale_features,
                    benchmark_tokens,
                    market_ids,
                    return_ranking,
                )
        return self._forward(
            conditioned_tokens, scale_features, benchmark_tokens, market_ids, return_ranking
        )

    def _forward(
        self,
        conditioned_tokens: Tensor,
        scale_features: Tensor | None,
        benchmark_tokens: Tensor | None,
        market_ids: Tensor | None = None,
        return_ranking: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        if conditioned_tokens.ndim != 3 or conditioned_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"conditioned_tokens must have shape [batch, tokens, {self.input_dim}]"
            )
        pooled = conditioned_tokens.mean(dim=1)
        if self.market_embedding is not None:
            if market_ids is None:
                raise ValueError("Market-aware forecasts require explicit market IDs")
            pooled = pooled + self.market_embedding(market_ids)
        horizon_ids = torch.arange(
            len(self.horizons),
            device=conditioned_tokens.device,
            dtype=torch.long,
        )
        horizon_inputs = pooled[:, None, :] + self.horizon_embeddings(horizon_ids)[None, :, :]
        hidden = self.trunk(horizon_inputs)
        raw = self.quantile_parameters(hidden)
        features = None
        if self.numeric_branch is not None:
            features = self.numeric_branch.encode(hidden, scale_features, benchmark_tokens)
            raw = raw + self.numeric_branch.fusion(features)
        if self.market_residual is not None:
            if market_ids is None:
                raise ValueError("Market residual heads require market IDs")
            inputs = hidden if features is None else torch.cat((hidden, features), dim=-1)
            residual = self.market_residual(inputs).reshape(*hidden.shape[:2], 4, 3)
            index = market_ids[:, None, None, None].expand(-1, hidden.shape[1], 1, 3)
            raw = raw + residual.gather(2, index).squeeze(2)
        median = raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        quantiles = torch.stack([lower, median, upper], dim=-1)
        if self.scale_gate is not None:
            if scale_features is None:
                raise ValueError("Output scale requires past-only relative volatility")
            # A strictly positive historical anchor controls both location and interval widths.
            horizon = self._horizon_values.to(dtype=quantiles.dtype).sqrt()
            anchor = scale_features[:, 2:3] * horizon[None, :]
            floor = self.robust_scales[None, :] * 0.1
            anchor = torch.minimum(torch.maximum(anchor, floor), self.robust_scales[None, :] * 10)
            multiplier = (math.log(4) * torch.tanh(self.scale_gate(hidden))).exp()
            if self.decoupled_output_scale:
                # The median cannot collapse merely because interval widths shrink.
                median = raw[..., 1] * self.robust_scales[None, :]
                lower = median - F.softplus(raw[..., 0]) * anchor * multiplier[..., 0]
                upper = median + F.softplus(raw[..., 2]) * anchor * multiplier[..., 1]
                quantiles = torch.stack([lower, median, upper], dim=-1)
            else:
                quantiles = quantiles * anchor[..., None] * multiplier
        if return_ranking:
            scores = None if self.ranking_head is None else self.ranking_head(hidden).squeeze(-1)
            if self.ranking_feature_head is not None:
                scores = scores + self.ranking_feature_head(features).squeeze(-1)
            return quantiles, scores
        return quantiles

    def pinball_loss(
        self, predictions: Tensor, target: Tensor, sample_weights: Tensor | None = None
    ) -> Tensor:
        predictions = predictions.float()
        expected = (predictions.shape[0], len(self.horizons), len(self.quantiles))
        if tuple(predictions.shape) != expected:
            raise ValueError(f"alpha_quantiles must have shape {expected}")
        target = target.to(device=predictions.device, dtype=predictions.dtype)
        if tuple(target.shape) != (predictions.shape[0], len(self.horizons)):
            raise ValueError("target_alpha must have shape [batch, horizons]")
        valid = torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        scales = self.robust_scales.to(device=predictions.device, dtype=predictions.dtype)
        errors = (safe_target[..., None] - predictions) / scales[None, :, None]
        levels = self._quantile_level_bits.view(torch.float32).to(dtype=predictions.dtype)
        losses = torch.maximum(levels * errors, (levels - 1.0) * errors)
        weights = valid[..., None].expand_as(losses).to(dtype=losses.dtype)
        denominator = weights.sum().clamp_min(1.0)
        if sample_weights is not None:
            if sample_weights.shape != (predictions.shape[0],):
                raise ValueError("Market weights must contain one value per sample")
            weights = weights * sample_weights[:, None, None]
        return (losses * weights).sum() / denominator
