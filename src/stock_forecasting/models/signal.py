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


def _extract_last_bar_hidden(
    hidden_state: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Extract the true chronological last valid bar hidden state before Perceiver resampling."""
    if hidden_state.ndim != 3 or hidden_state.shape[1] == 0:
        raise ValueError("hidden_state must have shape [batch, sequence_length, hidden_dim]")
    if attention_mask is None:
        return hidden_state[:, -1, :]
    valid = attention_mask.to(device=hidden_state.device, dtype=torch.bool)
    positions = torch.arange(hidden_state.shape[1], device=hidden_state.device).unsqueeze(0)
    last_indices = (positions * valid.long()).amax(dim=1)
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, last_indices]


def _past_drawdown_tensor(
    close: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute causal historical maximum drawdown in [0, 1] from context close prices."""
    close_f = close.float()
    tiny = torch.finfo(close_f.dtype).tiny
    if attention_mask is None:
        peak = close_f.cummax(dim=1).values.clamp_min(tiny)
        return (1.0 - close_f / peak).clamp(min=0.0, max=1.0).amax(dim=1)
    valid = attention_mask.to(device=close_f.device, dtype=torch.bool)
    peak = close_f.masked_fill(~valid, 0.0).cummax(dim=1).values.clamp_min(tiny)
    declines = (1.0 - close_f / peak).masked_fill(~valid, 0.0).clamp(min=0.0, max=1.0)
    return declines.amax(dim=1)


def _student_t4_cdf(z: Tensor) -> Tensor:
    """Exact closed-form CDF of the heavy-tailed Student-t distribution with df=4."""
    w = 4.0 + z.square()
    return 0.5 + (0.5 * z * w.rsqrt()) * (1.0 + 2.0 / w)


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
        backbone_dim: int | None = None,
    ) -> None:
        super().__init__()
        if prediction_task not in SIGNAL_TASKS:
            raise ValueError(f"prediction_task must be one of {SIGNAL_TASKS}")
        if signal_architecture not in ("legacy", "v2"):
            raise ValueError("signal_architecture must be 'legacy' or 'v2'")
        if input_dim <= 0 or (hidden_dim is not None and hidden_dim <= 0):
            raise ValueError("Signal head dimensions must be positive")
        if backbone_dim is not None and backbone_dim <= 0:
            raise ValueError("backbone_dim must be positive")
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
        self.backbone_dim = int(backbone_dim or input_dim)
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
        self.benchmark_regime_head = None
        if signal_architecture == "v2":
            output_dim = 1 if prediction_task == "alpha_point" else 2
            self.last_bar_projection = nn.Sequential(
                nn.LayerNorm(self.backbone_dim),
                nn.Linear(self.backbone_dim, input_dim),
                nn.LayerNorm(input_dim),
            )
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
            if prediction_task == "alpha_point":
                init_theta = math.log(0.10 / (0.22 - 0.10))
                self.alpha_shrinkage_logit = nn.Parameter(
                    torch.full((1, len(self.horizons)), init_theta, dtype=torch.float32)
                )
                self.alpha_damping_mlp = nn.Sequential(
                    nn.LayerNorm(4),
                    nn.Linear(4, 16),
                    nn.GELU(),
                    nn.Linear(16, len(self.horizons)),
                )
                nn.init.zeros_(self.alpha_damping_mlp[-1].weight)
                nn.init.zeros_(self.alpha_damping_mlp[-1].bias)
                self.direction_vol_mlp = None
                self.market_horizon_boundary_delta = None
                self.market_horizon_loc_bias = None
                self.market_horizon_log_temp = None
                self.market_horizon_prior = None
            else:
                self.alpha_shrinkage_logit = None
                self.alpha_damping_mlp = None
                self.direction_vol_mlp = nn.Sequential(
                    nn.LayerNorm(4),
                    nn.Linear(4, 16),
                    nn.GELU(),
                    nn.Linear(16, len(self.horizons)),
                )
                nn.init.zeros_(self.direction_vol_mlp[-1].weight)
                nn.init.zeros_(self.direction_vol_mlp[-1].bias)
                self.market_horizon_boundary_delta = nn.Parameter(
                    torch.zeros(4, len(self.horizons))
                )
                self.market_horizon_loc_bias = nn.Parameter(
                    torch.zeros(4, len(self.horizons))
                )
                self.market_horizon_log_temp = nn.Parameter(
                    torch.full((4, len(self.horizons), 1), -2.0)
                )
                self.market_horizon_prior = nn.Parameter(
                    torch.zeros(4, len(self.horizons), 3)
                )
        else:
            output_dim = 1 if prediction_task == "alpha_point" else 3
            self.last_bar_projection = None
            self.recency_gate = None
            self.query_norm = None
            self.token_norm = None
            self.horizon_attention = None
            self.attention_gate = None
            self.trunk_residual = None
            self.alpha_shrinkage_logit = None
            self.alpha_damping_mlp = None
            self.direction_vol_mlp = None
            self.market_horizon_boundary_delta = None
            self.market_horizon_loc_bias = None
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
        self._last_eval_alpha: tuple[Tensor, Tensor] | None = None
        self._last_eval_direction: tuple[Tensor, Tensor, Tensor] | None = None

    def set_backbone_dim(self, backbone_dim: int) -> None:
        """Align the pre-Perceiver last-bar projection dimension with the backbone."""
        if backbone_dim <= 0:
            raise ValueError("backbone_dim must be positive")
        if self.signal_architecture != "v2" or int(backbone_dim) == self.backbone_dim:
            return
        device = self.prediction.weight.device
        dtype = self.prediction.weight.dtype
        self.backbone_dim = int(backbone_dim)
        self.last_bar_projection = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
        ).to(device=device, dtype=dtype)

    def forward(
        self,
        conditioned_tokens: Tensor,
        *,
        scale_features: Tensor | None = None,
        benchmark_tokens: Tensor | None = None,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
        asset_last_hidden: Tensor | None = None,
        benchmark_last_hidden: Tensor | None = None,
        past_drawdown: Tensor | None = None,
    ) -> Tensor:
        if self.fp32_head:
            with torch.autocast(device_type=conditioned_tokens.device.type, enabled=False):
                return self._forward(
                    conditioned_tokens.float(),
                    None if scale_features is None else scale_features.float(),
                    None if benchmark_tokens is None else benchmark_tokens.float(),
                    market_ids,
                    ranking_group_ids=ranking_group_ids,
                    asset_last_hidden=(
                        None if asset_last_hidden is None else asset_last_hidden.float()
                    ),
                    benchmark_last_hidden=(
                        None if benchmark_last_hidden is None else benchmark_last_hidden.float()
                    ),
                    past_drawdown=None if past_drawdown is None else past_drawdown.float(),
                )
        return self._forward(
            conditioned_tokens,
            scale_features,
            benchmark_tokens,
            market_ids,
            ranking_group_ids=ranking_group_ids,
            asset_last_hidden=asset_last_hidden,
            benchmark_last_hidden=benchmark_last_hidden,
            past_drawdown=past_drawdown,
        )

    def _project_true_last_bar(
        self,
        conditioned_tokens: Tensor,
        benchmark_tokens: Tensor | None,
        asset_last_hidden: Tensor | None,
        benchmark_last_hidden: Tensor | None,
    ) -> Tensor:
        assert self.last_bar_projection is not None
        if asset_last_hidden is not None:
            rel_last_raw = (
                asset_last_hidden
                if benchmark_last_hidden is None
                else asset_last_hidden - benchmark_last_hidden
            )
            if rel_last_raw.shape[-1] == self.backbone_dim:
                return self.last_bar_projection(rel_last_raw)
            if rel_last_raw.shape[-1] == self.input_dim:
                return F.layer_norm(rel_last_raw, (self.input_dim,))
            raise ValueError(
                f"Expected last_hidden dim {self.backbone_dim} or {self.input_dim}, "
                f"got {rel_last_raw.shape[-1]}"
            )
        fallback_raw = (
            conditioned_tokens[:, -1]
            if benchmark_tokens is None
            else conditioned_tokens[:, -1] - benchmark_tokens[:, -1]
        )
        if fallback_raw.shape[-1] == self.backbone_dim:
            return self.last_bar_projection(fallback_raw)
        return F.layer_norm(fallback_raw, (self.input_dim,))

    @staticmethod
    def _build_anchor_features(
        batch_size: int,
        scale_features: Tensor | None,
        past_drawdown: Tensor | None,
        *,
        device: torch.device,
    ) -> Tensor:
        if scale_features is not None and scale_features.shape[1] >= 3:
            asset_vol = scale_features[:, 0:1].float().clamp(min=1e-4, max=0.25)
            bench_vol = scale_features[:, 1:2].float().clamp(min=1e-4, max=0.25)
            rel_vol = scale_features[:, 2:3].float().clamp(min=1e-4, max=0.25)
            a0 = (asset_vol / 0.02).log().clamp(min=-3.0, max=3.0)
            a1 = (bench_vol / 0.015).log().clamp(min=-3.0, max=3.0)
            a2 = (rel_vol / 0.02).log().clamp(min=-3.0, max=3.0)
        else:
            a0 = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            a1 = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            a2 = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        if past_drawdown is not None:
            dd = past_drawdown.float().clamp(min=0.0, max=1.0).reshape(batch_size, 1)
            a3 = ((dd - 0.20) / 0.20).clamp(min=-1.0, max=4.0)
        else:
            a3 = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        return torch.cat((a0, a1, a2, a3), dim=-1)

    def _cross_sectional_standardize(
        self,
        scores: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
        eps_std: float = 0.1,
    ) -> Tensor:
        """Standardize raw scores to zero mean and unit std within same-(date, market) groups."""
        z_tilde = scores.clone()
        min_size = 2 if ranking_group_ids is not None else 4
        masks = self._iter_cross_sectional_groups(
            scores.shape[0],
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            min_group_size=min_size,
        )
        for mask in masks:
            group_scores = scores[mask]
            mean = group_scores.mean(dim=0, keepdim=True)
            var = (group_scores - mean).square().mean(dim=0, keepdim=True)
            z_tilde[mask] = (group_scores - mean) / (var + eps_std * eps_std).sqrt()
        return z_tilde

    def _cross_sectional_center(
        self,
        values: Tensor,
        *,
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
    ) -> Tensor:
        """Center location scores to zero mean within same-(date, market) groups."""
        centered = values.clone()
        min_size = 2 if ranking_group_ids is not None else 4
        masks = self._iter_cross_sectional_groups(
            values.shape[0],
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            min_group_size=min_size,
        )
        for mask in masks:
            group_values = values[mask]
            centered[mask] = group_values - group_values.mean(dim=0, keepdim=True)
        return centered

    def compute_alpha_from_standardized(
        self,
        z_tilde: Tensor,
        damping: Tensor,
    ) -> Tensor:
        """Apply Grinold-Kahn / James-Stein shrinkage gate and anchor damping to standardized z."""
        assert self.alpha_shrinkage_logit is not None
        scales = self.robust_scales.float().to(device=z_tilde.device)[None, :]
        gamma = 0.22 * torch.sigmoid(self.alpha_shrinkage_logit.float().to(device=z_tilde.device))
        return (gamma * damping.float() * torch.tanh(z_tilde.float())) * scales

    def compute_direction_cdf_logits(
        self,
        mu_centered: Tensor,
        sigma: Tensor,
        market_ids: Tensor | None = None,
    ) -> Tensor:
        """Compute heavy-tailed Student-t (df=4) interval log-probabilities."""
        assert self.market_horizon_boundary_delta is not None
        assert self.market_horizon_loc_bias is not None
        device = mu_centered.device
        scales = self.robust_scales.float().to(device=device)[None, :]
        if market_ids is not None:
            m_ids = market_ids.to(device=device, dtype=torch.long).clamp(min=0, max=3)
            delta_b = 0.7 * torch.tanh(
                self.market_horizon_boundary_delta.to(device=device)[m_ids].float() / 0.7
            )
            beta = 1.5 * torch.tanh(
                self.market_horizon_loc_bias.to(device=device)[m_ids].float() / 1.5
            )
        else:
            delta_b = 0.7 * torch.tanh(
                self.market_horizon_boundary_delta.to(device=device)[0:1].float() / 0.7
            )
            beta = 1.5 * torch.tanh(
                self.market_horizon_loc_bias.to(device=device)[0:1].float() / 1.5
            )
        b_low = (math.log1p(-self.neutral_return_threshold) / scales) * delta_b.exp()
        b_high = (math.log1p(self.neutral_return_threshold) / scales) * delta_b.exp()
        loc = mu_centered.float() + beta
        sigma_safe = sigma.float().clamp(min=0.25, max=6.0)
        z_low = ((b_low - loc) / sigma_safe).clamp(min=-12.0, max=12.0)
        z_high = ((b_high - loc) / sigma_safe).clamp(min=-12.0, max=12.0)
        z_high = torch.maximum(z_high, z_low + 1e-4)
        p_down = _student_t4_cdf(z_low).clamp(min=1e-7, max=1.0)
        p_neutral = (_student_t4_cdf(z_high) - _student_t4_cdf(z_low)).clamp(min=1e-7, max=1.0)
        p_up = _student_t4_cdf(-z_high).clamp(min=1e-7, max=1.0)
        return torch.stack((p_down.log(), p_neutral.log(), p_up.log()), dim=-1)

    def apply_direction_calibration(
        self,
        cdf_logits: Tensor,
        market_ids: Tensor | None = None,
    ) -> Tensor:
        """Apply lower-bounded temperature scaling (tau >= 1.0) and static log-prior adjustment."""
        assert self.market_horizon_log_temp is not None
        assert self.market_horizon_prior is not None
        device = cdf_logits.device
        if market_ids is not None:
            m_ids = market_ids.to(device=device, dtype=torch.long).clamp(min=0, max=3)
            phi = self.market_horizon_log_temp.to(device=device)[m_ids].float()
            raw_prior = self.market_horizon_prior.to(device=device)[m_ids].float()
        else:
            phi = self.market_horizon_log_temp.to(device=device)[0:1].float()
            raw_prior = self.market_horizon_prior.to(device=device)[0:1].float()
        tau = 1.0 + F.softplus(phi)
        prior = raw_prior - raw_prior.mean(dim=-1, keepdim=True)
        return (cdf_logits.float() / tau + prior).float()

    @torch.no_grad()
    def fit_validation_calibration(
        self,
        cdf_logits: Tensor,
        target_absolute: Tensor,
        market_ids: Tensor,
        *,
        min_samples: int = 64,
    ) -> None:
        """Calibrate per-(market, horizon) temperature tau >= 1.0 and log-prior on validation."""
        if (
            self.signal_architecture != "v2"
            or self.prediction_task != "absolute_direction"
            or self.market_horizon_log_temp is None
            or self.market_horizon_prior is None
        ):
            return
        device = self.market_horizon_log_temp.device
        logits_dev = cdf_logits.to(device=device, dtype=torch.float32)
        targets_dev = target_absolute.to(device=device, dtype=torch.float32)
        markets_dev = market_ids.to(device=device, dtype=torch.long).clamp(min=0, max=3)
        valid_mask = torch.isfinite(targets_dev)
        safe_targets = torch.where(valid_mask, targets_dev, torch.zeros_like(targets_dev))
        all_labels = self.direction_labels(safe_targets)
        tau_grid = torch.linspace(1.0, 2.5, 25, device=device, dtype=torch.float32)

        for m in range(4):
            m_rows = markets_dev == m
            if int(m_rows.sum()) < min_samples:
                continue
            for h in range(len(self.horizons)):
                valid_rows = m_rows & valid_mask[:, h]
                n_valid = int(valid_rows.sum())
                if n_valid < min_samples:
                    continue
                l_mh = logits_dev[valid_rows, h, :]
                y_mh = all_labels[valid_rows, h]

                # Baseline candidate: current (tau, prior) parameters.
                cur_phi = self.market_horizon_log_temp.data[m, h, 0].float()
                cur_tau = 1.0 + F.softplus(cur_phi)
                cur_raw_prior = self.market_horizon_prior.data[m, h, :].float()
                cur_prior = cur_raw_prior - cur_raw_prior.mean()
                best_ce = F.cross_entropy(l_mh / cur_tau + cur_prior[None, :], y_mh)
                best_phi = cur_phi
                best_prior = cur_prior

                # Empirical class frequencies with Dirichlet(2.0) smoothing.
                counts = torch.bincount(y_mh, minlength=3).float()
                pi_emp = (counts + 2.0) / (float(n_valid) + 6.0)
                log_pi_emp = pi_emp.log()

                for tau_val in tau_grid:
                    scaled = l_mh / tau_val
                    delta = torch.zeros(3, device=device, dtype=torch.float32)
                    for _ in range(5):
                        probs = F.softmax(scaled + delta[None, :], dim=-1)
                        mean_p = probs.mean(dim=0).clamp_min(1e-6)
                        delta = (0.85 * (delta + log_pi_emp - mean_p.log())).clamp(
                            min=-0.40, max=0.40
                        )
                        delta = delta - delta.mean()
                    ce_val = F.cross_entropy(scaled + delta[None, :], y_mh)
                    if bool(ce_val < best_ce - 1e-6):
                        best_ce = ce_val
                        excess = (tau_val - 1.0).clamp_min(1e-4)
                        best_phi = torch.log(torch.expm1(excess))
                        best_prior = delta

                self.market_horizon_log_temp.data[m, h, 0] = best_phi.to(
                    dtype=self.market_horizon_log_temp.dtype
                )
                self.market_horizon_prior.data[m, h, :] = best_prior.to(
                    dtype=self.market_horizon_prior.dtype
                )

    def _forward(
        self,
        conditioned_tokens: Tensor,
        scale_features: Tensor | None,
        benchmark_tokens: Tensor | None,
        market_ids: Tensor | None,
        *,
        ranking_group_ids: Tensor | None = None,
        asset_last_hidden: Tensor | None = None,
        benchmark_last_hidden: Tensor | None = None,
        past_drawdown: Tensor | None = None,
    ) -> Tensor:
        if conditioned_tokens.ndim != 3 or conditioned_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"conditioned_tokens must be [batch, tokens, {self.input_dim}]")
        batch_size = conditioned_tokens.shape[0]
        mean_pooled = conditioned_tokens.mean(dim=1)
        if self.signal_architecture == "v2":
            rel_last_token = self._project_true_last_bar(
                conditioned_tokens,
                benchmark_tokens,
                asset_last_hidden,
                benchmark_last_hidden,
            )
            recency_weight = self.recency_gate(torch.cat((mean_pooled, rel_last_token), dim=-1))
            pooled = mean_pooled + recency_weight * (rel_last_token - mean_pooled)
        else:
            pooled = mean_pooled
        if self.market_embedding is not None:
            if market_ids is None or market_ids.shape != (pooled.shape[0],):
                raise ValueError("Market-aware signal forecasts require one market ID per sample")
            pooled = pooled + self.market_embedding(market_ids)
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

        scales = self.robust_scales.float().to(device=hidden.device)[None, :]
        anchors = self._build_anchor_features(
            batch_size,
            scale_features,
            past_drawdown,
            device=hidden.device,
        )

        if self.prediction_task == "alpha_point":
            assert self.alpha_damping_mlp is not None
            raw_u = raw.squeeze(-1).float()
            damping = 2.0 * torch.sigmoid(self.alpha_damping_mlp(anchors).float())
            z_tilde = self._cross_sectional_standardize(
                raw_u,
                market_ids=market_ids,
                ranking_group_ids=ranking_group_ids,
                eps_std=0.1,
            )
            if not self.training:
                self._last_eval_alpha = (raw_u.detach(), damping.detach())
            return self.compute_alpha_from_standardized(z_tilde, damping).float()

        # V3 Heavy-Tailed Student-t (df=4) Ordinal Cumulative Head for absolute_direction.
        assert self.direction_vol_mlp is not None
        mu_raw = 3.0 * torch.tanh(raw[..., 0].float() / 3.0)
        mu_centered = self._cross_sectional_center(
            mu_raw,
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
        )
        if scale_features is not None:
            horizon_sqrt = self._horizon_values.to(
                device=hidden.device, dtype=torch.float32
            ).sqrt()[None, :]
            asset_vol = scale_features[:, 0:1].float().clamp(min=1e-4, max=0.25) * horizon_sqrt
            bench_vol = scale_features[:, 1:2].float().clamp(min=1e-4, max=0.25) * horizon_sqrt
            hist_ratio = ((0.7 * asset_vol + 0.3 * bench_vol) / scales).clamp(min=0.35, max=3.5)
            dd_boost = (
                1.0
                if past_drawdown is None
                else 1.0 + 0.5 * past_drawdown.float().clamp(min=0.0, max=1.0).reshape(batch_size, 1)
            )
            base_sigma = (0.5 + 0.5 * hist_ratio) * dd_boost
        else:
            base_sigma = torch.ones(
                (batch_size, len(self.horizons)),
                device=hidden.device,
                dtype=torch.float32,
            )

        vol_anchor_shift = self.direction_vol_mlp(anchors).float()
        log_vol_mult = math.log(2.5) * torch.tanh(0.5 * (raw[..., 1].float() + vol_anchor_shift))
        sigma = (base_sigma * log_vol_mult.exp()).clamp(min=0.25, max=6.0)

        if not self.training:
            m_ids_saved = (
                torch.zeros(batch_size, dtype=torch.long, device=hidden.device)
                if market_ids is None
                else market_ids.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=3)
            )
            self._last_eval_direction = (mu_raw.detach(), sigma.detach(), m_ids_saved.detach())

        cdf_logits = self.compute_direction_cdf_logits(mu_centered, sigma, market_ids)
        return self.apply_direction_calibration(cdf_logits, market_ids)

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
        if ranking_group_ids is not None and ranking_group_ids.shape == (batch_size,):
            for group_id in torch.unique(ranking_group_ids):
                mask = ranking_group_ids == group_id
                if int(mask.sum()) >= min_group_size:
                    masks.append(mask)
            return masks
        if market_ids is not None and market_ids.shape == (batch_size,):
            for group_id in torch.unique(market_ids):
                mask = market_ids == group_id
                if int(mask.sum()) >= min_group_size:
                    masks.append(mask)
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
        alpha_head.set_backbone_dim(resampler.input_dim)
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
        asset_last_hidden = None
        benchmark_last_hidden = None
        past_drawdown = None
        if head.signal_architecture == "v2":
            asset_last_hidden = _extract_last_bar_hidden(
                asset.last_hidden_state, asset.attention_mask
            )
            benchmark_last_hidden = _extract_last_bar_hidden(
                benchmark.last_hidden_state, benchmark.attention_mask
            )
            with torch.autocast(device_type=asset_ohlcv.device.type, enabled=False):
                past_drawdown = _past_drawdown_tensor(
                    asset_ohlcv[..., 3].float(), asset.attention_mask
                )
        conditioned, gate = self.benchmark_conditioner(asset.latent_tokens, benchmark.latent_tokens)
        prediction = head(
            conditioned,
            scale_features=scales,
            benchmark_tokens=benchmark.latent_tokens,
            market_ids=market_ids,
            ranking_group_ids=ranking_group_ids,
            asset_last_hidden=asset_last_hidden,
            benchmark_last_hidden=benchmark_last_hidden,
            past_drawdown=past_drawdown,
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
