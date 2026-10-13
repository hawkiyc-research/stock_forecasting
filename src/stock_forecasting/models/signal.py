"""Independent alpha-point or absolute-direction models on paired Kronos histories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stock_forecasting.data.horizons import DEFAULT_ALPHA_HORIZONS, validate_alpha_horizons

from .forecast import GatedBenchmarkConditioner
from .outputs import OutputMapping, QuantEncoderOutput
from .projector import CausalPerceiverResampler
from .quant import QuantForecastModel
from .scale_features import NumericalResidualBranch, historical_scale_features

SIGNAL_OUTPUT_SCHEMA_VERSION = "signal-1.0"
SIGNAL_TASKS = ("alpha_point", "absolute_direction")


@dataclass(frozen=True)
class SignalForecastOutput(OutputMapping):
    """Expose only the selected prediction task, never synthetic quantiles."""

    loss: Tensor | None
    point_loss: Tensor | None
    classification_loss: Tensor | None
    alpha_point: Tensor | None
    absolute_direction_logits: Tensor | None
    asset_last_hidden_state: Tensor
    benchmark_last_hidden_state: Tensor
    asset_attention_mask: Tensor
    benchmark_attention_mask: Tensor
    asset_latent_tokens: Tensor
    benchmark_latent_tokens: Tensor
    conditioned_latent_tokens: Tensor
    conditioning_gate: Tensor
    scale_features: Tensor | None = None


def fixed_direction_labels(target_absolute: Tensor, threshold: float = 0.01) -> Tensor:
    """Map cumulative absolute log returns to down/flat/up using simple-return edges.

    Equality with either boundary belongs to the flat class. Nonfinite targets
    must be masked by the caller before a supervised loss is reduced.
    """
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("neutral_return_threshold must be finite and in (0, 1)")
    lower = target_absolute.new_tensor(math.log1p(-threshold))
    upper = target_absolute.new_tensor(math.log1p(threshold))
    return (target_absolute >= lower).long() + (target_absolute > upper).long()


class SingleTaskHead(nn.Module):
    """Keep horizon/market/scale readout while fitting one statistical objective."""

    def __init__(
        self,
        input_dim: int,
        *,
        prediction_task: str,
        horizons: Sequence[int] = DEFAULT_ALPHA_HORIZONS,
        hidden_dim: int | None = None,
        robust_scales: Sequence[float] | None = None,
        dropout: float = 0.0,
        feature_mode: str = "combined",
        market_aware: bool = True,
        explicit_output_scale: bool = True,
        fp32_head: bool = True,
        point_loss_beta: float = 1.0,
        neutral_return_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        if prediction_task not in SIGNAL_TASKS:
            raise ValueError(f"prediction_task must be one of {SIGNAL_TASKS}")
        if input_dim <= 0 or (hidden_dim is not None and hidden_dim <= 0):
            raise ValueError("Signal head dimensions must be positive")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(point_loss_beta) or point_loss_beta <= 0:
            raise ValueError("point_loss_beta must be finite and positive")
        if not math.isfinite(neutral_return_threshold) or not 0 < neutral_return_threshold < 1:
            raise ValueError("neutral_return_threshold must be finite and in (0, 1)")
        if feature_mode not in ("baseline", "scales", "benchmark", "combined"):
            raise ValueError("Unknown numerical residual feature mode")
        if explicit_output_scale and feature_mode not in ("scales", "combined"):
            raise ValueError("Extended scale features require scales or combined feature mode")
        self.horizons = validate_alpha_horizons(horizons)
        scales = tuple(
            float(value)
            for value in ([1.0] * len(self.horizons) if robust_scales is None else robust_scales)
        )
        if len(scales) != len(self.horizons) or any(
            not math.isfinite(value) or value <= 0 for value in scales
        ):
            raise ValueError("robust_scales must contain one finite positive value per horizon")
        hidden_dim = hidden_dim or max(32, input_dim // 2)
        self.input_dim = input_dim
        self.prediction_task = prediction_task
        self.point_loss_beta = float(point_loss_beta)
        self.neutral_return_threshold = float(neutral_return_threshold)
        # This flag retains the existing extended-feature calibration contract;
        # point locations use fixed train-fitted scales, never per-sample volatility.
        self.explicit_output_scale = explicit_output_scale
        self.fp32_head = fp32_head or feature_mode != "baseline"
        self.horizon_embeddings = nn.Embedding(len(self.horizons), input_dim)
        self.market_embedding = nn.Embedding(4, input_dim) if market_aware else None
        if self.market_embedding is not None:
            nn.init.zeros_(self.market_embedding.weight)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        output_dim = 1 if prediction_task == "alpha_point" else 3
        self.prediction = nn.Linear(hidden_dim, output_dim)
        self.numeric_branch = (
            None
            if feature_mode == "baseline"
            else NumericalResidualBranch(
                hidden_dim, input_dim, feature_mode, extended=explicit_output_scale
            )
        )
        if self.numeric_branch is not None:
            self.numeric_branch.fusion[-1] = nn.Linear(32, output_dim)
            nn.init.zeros_(self.numeric_branch.fusion[-1].weight)
            nn.init.zeros_(self.numeric_branch.fusion[-1].bias)
        self.register_buffer(
            "robust_scales", torch.tensor(scales, dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        conditioned_tokens: Tensor,
        *,
        scale_features: Tensor | None = None,
        benchmark_tokens: Tensor | None = None,
        market_ids: Tensor | None = None,
    ) -> Tensor:
        if self.fp32_head:
            with torch.autocast(device_type=conditioned_tokens.device.type, enabled=False):
                return self._forward(
                    conditioned_tokens.float(), scale_features, benchmark_tokens, market_ids
                )
        return self._forward(conditioned_tokens, scale_features, benchmark_tokens, market_ids)

    def _forward(
        self,
        conditioned_tokens: Tensor,
        scale_features: Tensor | None,
        benchmark_tokens: Tensor | None,
        market_ids: Tensor | None,
    ) -> Tensor:
        if conditioned_tokens.ndim != 3 or conditioned_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"conditioned_tokens must be [batch, tokens, {self.input_dim}]")
        pooled = conditioned_tokens.mean(dim=1)
        if self.market_embedding is not None:
            if market_ids is None or market_ids.shape != (pooled.shape[0],):
                raise ValueError("Market-aware signal forecasts require one market ID per sample")
            pooled = pooled + self.market_embedding(market_ids)
        horizon_ids = torch.arange(len(self.horizons), device=pooled.device, dtype=torch.long)
        hidden = self.trunk(pooled[:, None] + self.horizon_embeddings(horizon_ids)[None])
        raw = self.prediction(hidden)
        if self.numeric_branch is not None:
            raw = raw + self.numeric_branch(hidden, scale_features, benchmark_tokens)
        if self.prediction_task == "alpha_point":
            return raw.squeeze(-1).float() * self.robust_scales.float()[None]
        return raw.float()

    def point_loss(self, predictions: Tensor, target: Tensor) -> Tensor:
        if self.prediction_task != "alpha_point":
            raise ValueError("Point regression loss requires the alpha_point task")
        expected = (predictions.shape[0], len(self.horizons))
        if predictions.shape != expected or target.shape != expected:
            raise ValueError("Point predictions and target_alpha must be [batch, horizons]")
        predictions = predictions.float()
        target = target.to(device=predictions.device, dtype=torch.float32)
        valid = torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        residual = (predictions - safe_target) / self.robust_scales.float()[None]
        values = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=self.point_loss_beta, reduction="none"
        )
        return self._equal_horizon_mean(values, valid)

    def direction_labels(self, target_absolute: Tensor) -> Tensor:
        return fixed_direction_labels(target_absolute, self.neutral_return_threshold)

    def classification_loss(self, logits: Tensor, target_absolute: Tensor) -> Tensor:
        if self.prediction_task != "absolute_direction":
            raise ValueError("Classification loss requires the absolute_direction task")
        expected = (logits.shape[0], len(self.horizons))
        if logits.shape != (*expected, 3) or target_absolute.shape != expected:
            raise ValueError("Direction logits must be [batch, horizons, 3] with aligned targets")
        target = target_absolute.to(device=logits.device, dtype=torch.float32)
        valid = torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        labels = self.direction_labels(safe_target)
        values = F.cross_entropy(
            logits.float().reshape(-1, 3), labels.reshape(-1), reduction="none"
        ).reshape(expected)
        return self._equal_horizon_mean(values, valid)

    @staticmethod
    def _equal_horizon_mean(values: Tensor, valid: Tensor) -> Tensor:
        # Missing labels never change the relative weight of observed horizons.
        count = valid.sum(dim=0)
        per_horizon = torch.where(valid, values, torch.zeros_like(values)).sum(0) / count.clamp_min(
            1
        )
        available = count > 0
        return (per_horizon * available).sum() / available.sum().clamp_min(1)


class SignalForecastModel(QuantForecastModel):
    """Reuse paired OHLCV encoding without invoking the multi-task forward path."""

    def __init__(
        self,
        backbone: nn.Module,
        resampler: CausalPerceiverResampler,
        benchmark_conditioner: GatedBenchmarkConditioner,
        alpha_head: SingleTaskHead,
    ) -> None:
        # Preserve the alpha_head attribute used by existing statistics/checkpoint
        # plumbing while constructing no quantile, ranking, or auxiliary head.
        super().__init__(backbone, resampler, benchmark_conditioner, alpha_head)
        self.prediction_task = alpha_head.prediction_task

    def forward(
        self,
        asset_ohlcv: Tensor,
        benchmark_ohlcv: Tensor,
        *,
        asset_attention_mask: Tensor | None = None,
        benchmark_attention_mask: Tensor | None = None,
        asset_timestamps: Tensor | None = None,
        benchmark_timestamps: Tensor | None = None,
        target_alpha: Tensor | None = None,
        target_absolute: Tensor | None = None,
        market_ids: Tensor | None = None,
        scale_inputs_validated: bool = False,
        ranking_group_ids: Tensor | None = None,
        security_ids: Tensor | None = None,
        ranking_pairs: Tensor | None = None,
        direction_bands: Tensor | None = None,
    ) -> SignalForecastOutput:
        # Shared batch containers may include legacy metadata. Neither objective
        # reads ranking pairs, transaction costs, or the other task's targets.
        del ranking_group_ids, security_ids, ranking_pairs, direction_bands
        head = self.alpha_head
        scales = None
        branch = head.numeric_branch
        if branch is not None and branch.scale_projection is not None:
            if (asset_attention_mask is None) != (benchmark_attention_mask is None):
                raise ValueError("Scale branch requires aligned asset and benchmark masks")
            if (
                not scale_inputs_validated
                and asset_attention_mask is not None
                and not torch.equal(asset_attention_mask, benchmark_attention_mask)
            ):
                raise ValueError("Scale branch requires aligned asset and benchmark masks")
            with torch.autocast(device_type=asset_ohlcv.device.type, enabled=False):
                scales = historical_scale_features(
                    asset_ohlcv,
                    benchmark_ohlcv,
                    mask=asset_attention_mask,
                    extended=head.explicit_output_scale,
                    inputs_validated=scale_inputs_validated,
                )
        can_fuse = (
            asset_ohlcv.shape == benchmark_ohlcv.shape
            and (asset_attention_mask is None) == (benchmark_attention_mask is None)
            and (asset_timestamps is None) == (benchmark_timestamps is None)
        )
        if can_fuse:
            batch = asset_ohlcv.shape[0]
            mask = (
                None
                if asset_attention_mask is None
                else torch.cat((asset_attention_mask, benchmark_attention_mask))
            )
            timestamps = (
                None
                if asset_timestamps is None
                else torch.cat((asset_timestamps, benchmark_timestamps))
            )
            encoded = self.encode_ohlcv(
                torch.cat((asset_ohlcv, benchmark_ohlcv)),
                attention_mask=mask,
                timestamps=timestamps,
            )
            asset = QuantEncoderOutput(
                encoded.last_hidden_state[:batch],
                encoded.attention_mask[:batch],
                encoded.latent_tokens[:batch],
            )
            benchmark = QuantEncoderOutput(
                encoded.last_hidden_state[batch:],
                encoded.attention_mask[batch:],
                encoded.latent_tokens[batch:],
            )
        else:
            asset = self.encode_ohlcv(
                asset_ohlcv, attention_mask=asset_attention_mask, timestamps=asset_timestamps
            )
            benchmark = self.encode_ohlcv(
                benchmark_ohlcv,
                attention_mask=benchmark_attention_mask,
                timestamps=benchmark_timestamps,
            )
        conditioned, gate = self.benchmark_conditioner(asset.latent_tokens, benchmark.latent_tokens)
        prediction = head(
            conditioned,
            scale_features=scales,
            benchmark_tokens=benchmark.latent_tokens,
            market_ids=market_ids,
        )
        point = logits = point_loss = classification_loss = None
        if self.prediction_task == "alpha_point":
            point = prediction
            if target_alpha is not None:
                point_loss = head.point_loss(point, target_alpha)
        else:
            logits = prediction
            if target_absolute is not None:
                classification_loss = head.classification_loss(logits, target_absolute)
        return SignalForecastOutput(
            loss=point_loss if self.prediction_task == "alpha_point" else classification_loss,
            point_loss=point_loss,
            classification_loss=classification_loss,
            alpha_point=point,
            absolute_direction_logits=logits,
            asset_last_hidden_state=asset.last_hidden_state,
            benchmark_last_hidden_state=benchmark.last_hidden_state,
            asset_attention_mask=asset.attention_mask,
            benchmark_attention_mask=benchmark.attention_mask,
            asset_latent_tokens=asset.latent_tokens,
            benchmark_latent_tokens=benchmark.latent_tokens,
            conditioned_latent_tokens=conditioned,
            conditioning_gate=gate,
            scale_features=scales,
        )
