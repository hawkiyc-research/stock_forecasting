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
        signal_architecture: str = "legacy",
        signal_attention_heads: int = 4,
        alpha_ic_loss_weight: float = 0.0,
        alpha_bias_penalty_weight: float = 0.0,
        direction_label_smoothing: float = 0.0,
        direction_focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if prediction_task not in SIGNAL_TASKS:
            raise ValueError(f"prediction_task must be one of {SIGNAL_TASKS}")
        if signal_architecture not in ("legacy", "v2"):
            raise ValueError("signal_architecture must be 'legacy' or 'v2'")
        if input_dim <= 0 or (hidden_dim is not None and hidden_dim <= 0):
            raise ValueError("Signal head dimensions must be positive")
        if signal_architecture == "v2" and (
            signal_attention_heads <= 0 or input_dim % signal_attention_heads != 0
        ):
            raise ValueError("input_dim must be divisible by signal_attention_heads")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(point_loss_beta) or point_loss_beta <= 0:
            raise ValueError("point_loss_beta must be finite and positive")
        if not math.isfinite(neutral_return_threshold) or not 0 < neutral_return_threshold < 1:
            raise ValueError("neutral_return_threshold must be finite and in (0, 1)")
        if not math.isfinite(alpha_ic_loss_weight) or not 0.0 <= alpha_ic_loss_weight <= 2.0:
            raise ValueError("alpha_ic_loss_weight must be finite and in [0.0, 2.0]")
        if not math.isfinite(alpha_bias_penalty_weight) or not 0.0 <= alpha_bias_penalty_weight <= 2.0:
            raise ValueError("alpha_bias_penalty_weight must be finite and in [0.0, 2.0]")
        if not math.isfinite(direction_label_smoothing) or not 0.0 <= direction_label_smoothing < 0.25:
            raise ValueError("direction_label_smoothing must be finite and in [0.0, 0.25)")
        if not math.isfinite(direction_focal_gamma) or not 0.0 <= direction_focal_gamma <= 5.0:
            raise ValueError("direction_focal_gamma must be finite and in [0.0, 5.0]")
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
        self.hidden_dim = hidden_dim
        self.prediction_task = prediction_task
        self.signal_architecture = signal_architecture
        self.point_loss_beta = float(point_loss_beta)
        self.neutral_return_threshold = float(neutral_return_threshold)
        self.alpha_ic_loss_weight = float(alpha_ic_loss_weight)
        self.alpha_bias_penalty_weight = float(alpha_bias_penalty_weight)
        self.direction_label_smoothing = float(direction_label_smoothing)
        self.direction_focal_gamma = float(direction_focal_gamma)
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
        if signal_architecture == "v2":
            output_dim = 1 if prediction_task == "alpha_point" else 5
            self.recency_gate = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.recency_gate[1].weight)
            nn.init.constant_(self.recency_gate[1].bias, -1.0)
            self.query_norm = nn.LayerNorm(input_dim)
            self.token_norm = nn.LayerNorm(input_dim)
            self.horizon_attention = nn.MultiheadAttention(
                input_dim,
                signal_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_gate = nn.Parameter(torch.full((1, len(self.horizons), 1), 0.25))
            self.trunk_residual = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.normal_(self.trunk_residual[-1].weight, std=0.01)
            nn.init.zeros_(self.trunk_residual[-1].bias)
            regime_dim = 1 if prediction_task == "alpha_point" else 2
            regime_hidden = max(16, hidden_dim // 2)
            self.benchmark_regime_head = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, regime_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(regime_hidden, len(self.horizons) * regime_dim),
            )
            nn.init.normal_(self.benchmark_regime_head[-1].weight, std=0.01)
            nn.init.zeros_(self.benchmark_regime_head[-1].bias)
            if prediction_task == "absolute_direction":
                self.market_horizon_log_temp = nn.Parameter(
                    torch.zeros(4, len(self.horizons), 1)
                )
                self.market_horizon_prior = nn.Parameter(
                    torch.zeros(4, len(self.horizons), 3)
                )
            else:
                self.market_horizon_log_temp = None
                self.market_horizon_prior = None
        else:
            output_dim = 1 if prediction_task == "alpha_point" else 3
            self.recency_gate = None
            self.query_norm = None
            self.token_norm = None
            self.horizon_attention = None
            self.attention_gate = None
            self.trunk_residual = None
            self.benchmark_regime_head = None
            self.market_horizon_log_temp = None
            self.market_horizon_prior = None

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
        self.register_buffer(
            "_horizon_values",
            torch.tensor(self.horizons, dtype=torch.float32),
            persistent=False,
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
                    conditioned_tokens.float(),
                    None if scale_features is None else scale_features.float(),
                    None if benchmark_tokens is None else benchmark_tokens.float(),
                    market_ids,
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
        mean_pooled = conditioned_tokens.mean(dim=1)
        if self.signal_architecture == "v2":
            last_token = conditioned_tokens[:, -1]
            recency_weight = self.recency_gate(torch.cat((mean_pooled, last_token), dim=-1))
            pooled = mean_pooled + recency_weight * (last_token - mean_pooled)
        else:
            pooled = mean_pooled
        market_offset = None
        if self.market_embedding is not None:
            if market_ids is None or market_ids.shape != (pooled.shape[0],):
                raise ValueError("Market-aware signal forecasts require one market ID per sample")
            market_offset = self.market_embedding(market_ids)
            pooled = pooled + market_offset
        horizon_ids = torch.arange(len(self.horizons), device=pooled.device, dtype=torch.long)
        horizon_queries = pooled[:, None] + self.horizon_embeddings(horizon_ids)[None]
        if self.signal_architecture == "v2":
            attended, _ = self.horizon_attention(
                query=self.query_norm(horizon_queries),
                key=self.token_norm(conditioned_tokens),
                value=self.token_norm(conditioned_tokens),
                need_weights=False,
            )
            horizon_queries = horizon_queries + torch.tanh(self.attention_gate) * attended
        hidden = self.trunk(horizon_queries)
        if self.trunk_residual is not None:
            hidden = hidden + self.trunk_residual(hidden)
        raw = self.prediction(hidden)
        if self.numeric_branch is not None:
            raw = raw + self.numeric_branch(hidden, scale_features, benchmark_tokens)
        if self.signal_architecture == "legacy":
            if self.prediction_task == "alpha_point":
                return raw.squeeze(-1).float() * self.robust_scales.float()[None]
            return raw.float()

        # V2 regime conditioning from benchmark tokens (or conditioned tokens if absent).
        source_tokens = conditioned_tokens if benchmark_tokens is None else benchmark_tokens
        bench_mean = source_tokens.mean(dim=1)
        bench_last = source_tokens[:, -1]
        if market_offset is not None:
            bench_mean = bench_mean + market_offset
            bench_last = bench_last + market_offset
        bench_input = torch.cat((bench_mean, bench_last), dim=-1)
        regime_out = self.benchmark_regime_head(bench_input)
        scales = self.robust_scales.float().to(device=hidden.device)[None, :]

        if self.prediction_task == "alpha_point":
            regime_drift = 0.25 * torch.tanh(regime_out)
            z_total = 8.0 * torch.tanh((raw.squeeze(-1) + regime_drift) / 8.0)
            return z_total.float() * scales

        # V2 Volatility-Aware Gaussian-CDF Interval Logit Head for absolute_direction.
        batch_size = hidden.shape[0]
        regime = regime_out.reshape(batch_size, len(self.horizons), 2)
        z_bench = 2.0 * torch.tanh(regime[..., 0])
        v_bench = 0.5 * torch.tanh(regime[..., 1])
        z_loc = 6.0 * torch.tanh((raw[..., 0] + z_bench) / 6.0)
        mu = z_loc.float() * scales

        if scale_features is not None:
            horizon_sqrt = self._horizon_values.to(
                device=hidden.device, dtype=torch.float32
            ).sqrt()[None, :]
            asset_vol = scale_features[:, 0:1].float().clamp(min=1e-4, max=0.25) * horizon_sqrt
            bench_vol = scale_features[:, 1:2].float().clamp(min=1e-4, max=0.25) * horizon_sqrt
            hist_anchor = (0.7 * asset_vol + 0.3 * bench_vol).clamp(
                min=0.25 * scales, max=4.0 * scales
            )
            base_sigma = 0.5 * scales + 0.5 * hist_anchor
        else:
            base_sigma = scales.expand(batch_size, -1)

        log_vol_mult = math.log(3.0) * torch.tanh(raw[..., 1].float() + v_bench.float())
        sigma = (base_sigma * log_vol_mult.exp()).clamp(min=1e-4)

        lower_edge = math.log1p(-self.neutral_return_threshold)
        upper_edge = math.log1p(self.neutral_return_threshold)
        a = ((lower_edge - mu) / sigma).clamp(min=-8.0, max=8.0)
        b = ((upper_edge - mu) / sigma).clamp(min=-8.0, max=8.0)
        b = torch.maximum(b, a + 1e-4)
        p_down = torch.special.ndtr(a).clamp(min=1e-7, max=1.0)
        p_neutral = (torch.special.ndtr(b) - torch.special.ndtr(a)).clamp(min=1e-7, max=1.0)
        p_up = torch.special.ndtr(-b).clamp(min=1e-7, max=1.0)
        cdf_logits = torch.stack((p_down.log(), p_neutral.log(), p_up.log()), dim=-1)

        if market_ids is not None:
            m_ids = market_ids.clamp(min=0, max=3)
            log_temp = self.market_horizon_log_temp[m_ids].float().clamp(min=-1.0, max=1.0)
            prior = self.market_horizon_prior[m_ids].float()
        else:
            log_temp = self.market_horizon_log_temp[0:1].float().clamp(min=-1.0, max=1.0)
            prior = self.market_horizon_prior[0:1].float()

        residual_logits = 1.5 * torch.tanh(raw[..., 2:5].float() / 1.5)
        logits = cdf_logits * (-log_temp).exp() + prior + residual_logits
        return logits.float()

    def point_loss(
        self,
        predictions: Tensor,
        target: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
    ) -> Tensor:
        if self.prediction_task != "alpha_point":
            raise ValueError("Point regression loss requires the alpha_point task")
        expected = (predictions.shape[0], len(self.horizons))
        if predictions.shape != expected or target.shape != expected:
            raise ValueError("Point predictions and target_alpha must be [batch, horizons]")
        predictions = predictions.float()
        target = target.to(device=predictions.device, dtype=torch.float32)
        valid = torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        scales = self.robust_scales.float().to(device=predictions.device)[None]
        residual = (predictions - safe_target) / scales
        values = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=self.point_loss_beta, reduction="none"
        )
        base_loss = self._equal_horizon_mean(values, valid)
        if not self.training or (
            self.alpha_ic_loss_weight == 0.0 and self.alpha_bias_penalty_weight == 0.0
        ):
            return base_loss

        pred_norm = predictions / scales
        target_norm = safe_target / scales
        demeaned_loss = self._cross_sectional_demeaned_smooth_l1(
            pred_norm,
            target_norm,
            valid,
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
        )
        total_loss = 0.5 * base_loss + 0.5 * demeaned_loss
        if self.alpha_ic_loss_weight > 0.0:
            ic_loss = self._cross_sectional_ic_loss(
                pred_norm,
                target_norm,
                valid,
                market_ids=market_ids,
                ranking_group_ids=ranking_group_ids,
            )
            total_loss = total_loss + self.alpha_ic_loss_weight * ic_loss
        if self.alpha_bias_penalty_weight > 0.0:
            bias_loss = self._batch_bias_penalty(
                pred_norm,
                valid,
                market_ids=market_ids,
                ranking_group_ids=ranking_group_ids,
            )
            total_loss = total_loss + self.alpha_bias_penalty_weight * bias_loss
        return total_loss

    def _iter_cross_sectional_groups(
        self,
        batch_size: int,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
        min_group_size: int = 4,
    ) -> list[Tensor]:
        masks: list[Tensor] = []
        for group_tensor in (ranking_group_ids, market_ids):
            if group_tensor is None or group_tensor.shape != (batch_size,):
                continue
            for group_id in torch.unique(group_tensor):
                mask = group_tensor == group_id
                if int(mask.sum()) >= min_group_size:
                    masks.append(mask)
            if masks:
                return masks
        return masks

    def _cross_sectional_demeaned_smooth_l1(
        self,
        pred_norm: Tensor,
        target_norm: Tensor,
        valid: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
    ) -> Tensor:
        """Remove date-market common target shocks so regression focuses on relative alpha."""
        demeaned_residual = pred_norm - target_norm
        masks = self._iter_cross_sectional_groups(
            pred_norm.shape[0],
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            min_group_size=2,
        )
        if not masks and pred_norm.shape[0] >= 2:
            masks = [torch.ones(pred_norm.shape[0], dtype=torch.bool, device=pred_norm.device)]
        if masks:
            demeaned_residual = demeaned_residual.clone()
            for mask in masks:
                group_valid = valid[mask]
                count = group_valid.sum(dim=0, keepdim=True)
                active = count >= 2
                safe_count = count.clamp_min(1).to(dtype=pred_norm.dtype)
                p_group = torch.where(
                    group_valid, pred_norm[mask], torch.zeros_like(pred_norm[mask])
                )
                y_group = torch.where(
                    group_valid, target_norm[mask], torch.zeros_like(target_norm[mask])
                )
                p_mean = p_group.sum(dim=0, keepdim=True) / safe_count
                y_mean = y_group.sum(dim=0, keepdim=True) / safe_count
                centered = (pred_norm[mask] - p_mean) - (target_norm[mask] - y_mean)
                demeaned_residual[mask] = torch.where(active, centered, demeaned_residual[mask])
        values = F.smooth_l1_loss(
            demeaned_residual,
            torch.zeros_like(demeaned_residual),
            beta=self.point_loss_beta,
            reduction="none",
        )
        return self._equal_horizon_mean(values, valid)

    @staticmethod
    def _group_ic_term(pred_norm: Tensor, target_norm: Tensor, valid: Tensor) -> Tensor | None:
        count = valid.sum(dim=0)
        active = count >= 4
        if not bool(active.any()):
            return None
        safe_count = count.clamp_min(1).to(dtype=pred_norm.dtype)
        p_valid = torch.where(valid, pred_norm, torch.zeros_like(pred_norm))
        y_valid = torch.where(valid, target_norm, torch.zeros_like(target_norm))
        p_mean = p_valid.sum(dim=0, keepdim=True) / safe_count[None, :]
        y_mean = y_valid.sum(dim=0, keepdim=True) / safe_count[None, :]
        p_cent = torch.where(valid, pred_norm - p_mean, torch.zeros_like(pred_norm))
        y_cent = torch.where(valid, target_norm - y_mean, torch.zeros_like(target_norm))
        p_var = (p_cent * p_cent).sum(dim=0) / safe_count
        y_var = (y_cent * y_cent).sum(dim=0) / safe_count
        informative = active & (y_var > 1e-6)
        if not bool(informative.any()):
            return None
        p_std = (p_var + 1e-6).sqrt()[None, :]
        y_std = (y_var + 1e-6).sqrt()[None, :]
        # Linear cosine IC + Huberized soft-rank cosine IC for outlier robustness.
        cos_linear = (p_cent * y_cent).sum(dim=0) / (
            (p_cent.norm(dim=0) * y_cent.norm(dim=0)).clamp_min(1e-6)
        )
        p_soft = torch.where(
            valid, 3.0 * torch.tanh((p_cent / p_std) / 3.0), torch.zeros_like(p_cent)
        )
        y_soft = torch.where(
            valid, 3.0 * torch.tanh((y_cent / y_std) / 3.0), torch.zeros_like(y_cent)
        )
        p_soft = torch.where(
            valid,
            p_soft - p_soft.sum(dim=0, keepdim=True) / safe_count[None, :],
            torch.zeros_like(p_soft),
        )
        y_soft = torch.where(
            valid,
            y_soft - y_soft.sum(dim=0, keepdim=True) / safe_count[None, :],
            torch.zeros_like(y_soft),
        )
        cos_soft = (p_soft * y_soft).sum(dim=0) / (
            (p_soft.norm(dim=0) * y_soft.norm(dim=0)).clamp_min(1e-6)
        )
        per_horizon = 1.0 - 0.5 * (cos_linear + cos_soft)
        return (per_horizon * informative).sum() / informative.sum().clamp_min(1)

    def _cross_sectional_ic_loss(
        self,
        pred_norm: Tensor,
        target_norm: Tensor,
        valid: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
    ) -> Tensor:
        terms: list[Tensor] = []
        for mask in self._iter_cross_sectional_groups(
            pred_norm.shape[0],
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            min_group_size=4,
        ):
            term = self._group_ic_term(pred_norm[mask], target_norm[mask], valid[mask])
            if term is not None:
                terms.append(term)
        if not terms:
            fallback = self._group_ic_term(pred_norm, target_norm, valid)
            if fallback is not None:
                terms.append(fallback)
        if not terms:
            return pred_norm.sum() * 0.0
        return torch.stack(terms).mean()

    def _batch_bias_penalty(
        self,
        pred_norm: Tensor,
        valid: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
    ) -> Tensor:
        """Shrink same-date-market group prediction means toward zero without chasing date shocks."""
        penalties: list[Tensor] = []
        masks = self._iter_cross_sectional_groups(
            pred_norm.shape[0],
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            min_group_size=2,
        )
        if not masks:
            masks = [torch.ones(pred_norm.shape[0], dtype=torch.bool, device=pred_norm.device)]
        for mask in masks:
            group_valid = valid[mask]
            count = group_valid.sum(dim=0)
            available = count > 0
            if not bool(available.any()):
                continue
            safe_count = count.clamp_min(1).to(dtype=pred_norm.dtype)
            mean_pred = (
                torch.where(group_valid, pred_norm[mask], torch.zeros_like(pred_norm[mask])).sum(
                    dim=0
                )
                / safe_count
            )
            penalty = F.smooth_l1_loss(
                mean_pred, torch.zeros_like(mean_pred), beta=0.25, reduction="none"
            )
            penalties.append((penalty * available).sum() / available.sum().clamp_min(1))
        if not penalties:
            return pred_norm.sum() * 0.0
        return torch.stack(penalties).mean()

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
        flat_logits = logits.float().reshape(-1, 3)
        flat_labels = labels.reshape(-1)
        ce_values = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(expected)
        if not self.training or (
            self.direction_label_smoothing == 0.0 and self.direction_focal_gamma == 0.0
        ):
            return self._equal_horizon_mean(ce_values, valid)
        smoothed = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            label_smoothing=self.direction_label_smoothing,
        ).reshape(expected)
        if self.direction_focal_gamma > 0.0:
            pt = (-ce_values).exp().clamp(min=1e-6, max=1.0)
            focal_weight = (1.0 - pt) ** self.direction_focal_gamma
            values = 0.7 * smoothed + 0.3 * focal_weight * smoothed
        else:
            values = smoothed
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
        del security_ids, ranking_pairs, direction_bands
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
                point_loss = head.point_loss(
                    point,
                    target_alpha,
                    market_ids=market_ids,
                    ranking_group_ids=ranking_group_ids,
                )
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
