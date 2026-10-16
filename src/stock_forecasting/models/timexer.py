"""Market-Guided TimeXer causal time-series backbone for continuous OHLCV forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import _valid_mask
from .outputs import TimeSeriesBackboneOutput


class _CausalMultiScalePatchStem(nn.Module):
    """Extract multi-scale causal patch embeddings while preserving bar resolution."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_lengths: Sequence[int] = (1, 4, 8),
    ) -> None:
        super().__init__()
        if not patch_lengths or any(length < 1 for length in patch_lengths):
            raise ValueError("patch_lengths must be a non-empty sequence of positive integers")
        self.patch_lengths = tuple(int(length) for length in patch_lengths)
        branch_dim = max(16, hidden_size // len(self.patch_lengths))
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, branch_dim, kernel_size=length, padding=0)
                for length in self.patch_lengths
            ]
        )
        self.projection = nn.Linear(branch_dim * len(self.patch_lengths), hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features: Tensor) -> Tensor:
        # features: [batch, bars, channels] -> conv input: [batch, channels, bars]
        transposed = features.transpose(1, 2)
        outputs: list[Tensor] = []
        for length, conv in zip(self.patch_lengths, self.branches, strict=True):
            padded = F.pad(transposed, (length - 1, 0)) if length > 1 else transposed
            outputs.append(F.gelu(conv(padded)).transpose(1, 2))
        concatenated = torch.cat(outputs, dim=-1)
        return self.norm(self.projection(concatenated))


class _TimeXerBlock(nn.Module):
    """Causal temporal self-attention with exogenous variate-summary gating."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        variate_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm_var = nn.LayerNorm(hidden_size)
        self.variate_proj = nn.Linear(variate_dim, hidden_size * 2)
        self.variate_out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.variate_out.weight)
        nn.init.zeros_(self.variate_out.bias)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        causal_variate_summary: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        normed = self.norm1(hidden)
        attended, _ = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout1(attended)

        # Endogenous-Exogenous variate bridge (TimeXer + MASTER style causal modulation)
        scale_shift = self.variate_proj(causal_variate_summary)
        var_gate, var_bias = scale_shift.chunk(2, dim=-1)
        modulated = F.silu(self.norm_var(hidden) * torch.sigmoid(var_gate) + var_bias)
        hidden = hidden + self.variate_out(modulated)

        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden


class MarketGuidedTimeXerBackbone(nn.Module):
    """Endogenous-Exogenous Multi-Scale Patch Transformer with MASTER regime gating.

    Operates directly on continuous, scale-free OHLCV log-return, range, volume,
    and realized-volatility dynamics without lossy VQ quantization.
    """

    NUM_ENDOGENOUS_FEATURES = 12
    NUM_REGIME_FEATURES = 6

    def __init__(
        self,
        input_dim: int = 5,
        hidden_size: int = 256,
        max_context: int = 128,
        *,
        num_layers: int = 4,
        num_heads: int = 8,
        patch_lengths: Sequence[int] = (1, 4, 8),
        dropout: float = 0.1,
        clip_value: float = 8.0,
    ) -> None:
        super().__init__()
        if input_dim != 5:
            raise ValueError("MarketGuidedTimeXerBackbone expects 5 OHLCV features")
        if hidden_size <= 0 or max_context <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("hidden_size, max_context, num_layers, and num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0 or clip_value <= 0:
            raise ValueError("dropout must be in [0, 1) and clip_value must be positive")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.max_context = max_context
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.clip_value = float(clip_value)
        self.technical_adapter: nn.Module | None = None
        self.lora_module_names: tuple[str, ...] = ()
        self.unfrozen_parameter_names: tuple[str, ...] = ()

        # MASTER market-guided feature gate: initialized so sigmoid(0) * 2 = 1.0
        self.market_gate = nn.Sequential(
            nn.Linear(self.NUM_REGIME_FEATURES, 24),
            nn.SiLU(),
            nn.Linear(24, self.NUM_ENDOGENOUS_FEATURES),
        )
        nn.init.zeros_(self.market_gate[-1].weight)
        nn.init.zeros_(self.market_gate[-1].bias)

        self.patch_stem = _CausalMultiScalePatchStem(
            self.NUM_ENDOGENOUS_FEATURES,
            hidden_size,
            patch_lengths=patch_lengths,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_context, hidden_size)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.timestamp_projection = nn.Linear(5, hidden_size)
        nn.init.zeros_(self.timestamp_projection.weight)
        nn.init.zeros_(self.timestamp_projection.bias)

        self.blocks = nn.ModuleList(
            [
                _TimeXerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    variate_dim=self.NUM_ENDOGENOUS_FEATURES * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _causal_window_sum(values: Tensor, window: int) -> Tensor:
        if window <= 1:
            return values
        padded = F.pad(values.unsqueeze(1), (window - 1, 0)).squeeze(1)
        return padded.unfold(dimension=1, size=window, step=1).sum(dim=-1)

    @staticmethod
    def _causal_window_mean(values: Tensor, window: int) -> Tensor:
        if window <= 1:
            return values
        padded = F.pad(values.unsqueeze(1), (window - 1, 0)).squeeze(1)
        return padded.unfold(dimension=1, size=window, step=1).mean(dim=-1)

    def _extract_causal_features(
        self,
        ohlcv: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid = mask.to(dtype=torch.float32)
        valid_3d = valid.unsqueeze(-1)
        prices = ohlcv[..., :4].float().clamp_min(1e-6)
        log_p = prices.log()
        log_o, log_h, log_l, log_c = (
            log_p[..., 0],
            log_p[..., 1],
            log_p[..., 2],
            log_p[..., 3],
        )
        prev_log_c = torch.cat([log_c[:, :1], log_c[:, :-1]], dim=1)

        r_close = (log_c - prev_log_c) * valid
        r_close[:, 0] = 0.0
        r_oc = (log_c - log_o) * valid
        r_hl = (log_h - log_l).clamp_min(0.0) * valid
        r_upper = (log_h - torch.maximum(log_o, log_c)).clamp_min(0.0) * valid
        r_lower = (torch.minimum(log_o, log_c) - log_l).clamp_min(0.0) * valid
        r_gap = (log_o - prev_log_c) * valid
        r_gap[:, 0] = 0.0

        r_5 = self._causal_window_sum(r_close, 5) * valid
        r_20 = self._causal_window_sum(r_close, 20) * valid
        rv_5 = self._causal_window_mean(r_close.square(), 5).clamp_min(1e-8).sqrt() * valid
        rv_20 = self._causal_window_mean(r_close.square(), 20).clamp_min(1e-8).sqrt() * valid

        log_v = ohlcv[..., 4].float().clamp_min(0.0).log1p() * valid
        counts = valid.cumsum(dim=1).clamp_min(1.0)
        v_mean = log_v.cumsum(dim=1) / counts
        v_var = ((log_v - v_mean).square() * valid).cumsum(dim=1) / counts
        vol_z = ((log_v - v_mean) / (v_var + 1e-4).sqrt()) * valid

        masked_close = prices[..., 3].masked_fill(~mask, 0.0)
        running_peak = masked_close.cummax(dim=1).values.clamp_min(1e-6)
        dd_causal = (1.0 - prices[..., 3] / running_peak).clamp(0.0, 1.0) * valid

        endogenous = torch.stack(
            [
                r_close * 25.0,
                r_oc * 25.0,
                r_hl * 25.0,
                r_upper * 25.0,
                r_lower * 25.0,
                r_gap * 25.0,
                r_5 * 10.0,
                r_20 * 5.0,
                vol_z,
                rv_5 * 25.0,
                rv_20 * 25.0,
                dd_causal * 5.0,
            ],
            dim=-1,
        ).clamp(-self.clip_value, self.clip_value)
        endogenous = endogenous * valid_3d

        regime = torch.stack(
            [
                r_5 * 10.0,
                r_20 * 5.0,
                rv_5 * 25.0,
                rv_20 * 25.0,
                vol_z,
                dd_causal * 5.0,
            ],
            dim=-1,
        ).clamp(-self.clip_value, self.clip_value)
        regime = regime * valid_3d
        return endogenous, regime

    def forward(
        self,
        ohlcv: Tensor,
        attention_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> TimeSeriesBackboneOutput:
        if ohlcv.ndim != 3:
            raise ValueError("ohlcv must have shape [batch, bars, features]")
        batch_size, sequence_length, feature_count = ohlcv.shape
        if feature_count != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} OHLCV features, got {feature_count}")
        if sequence_length > self.max_context:
            raise ValueError(
                f"Input contains {sequence_length} bars, exceeding max_context={self.max_context}"
            )
        if ohlcv.device.type == "cpu" and not torch.isfinite(ohlcv).all():
            raise ValueError("ohlcv contains non-finite values")
        if timestamps is not None and timestamps.shape != (batch_size, sequence_length, 5):
            raise ValueError("timestamps must have shape [batch, bars, 5]")

        mask = _valid_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=ohlcv.device,
        )
        endogenous, regime = self._extract_causal_features(ohlcv, mask)
        param_dtype = self.position_embedding.dtype
        endogenous = endogenous.to(dtype=param_dtype)
        regime = regime.to(dtype=param_dtype)
        valid = mask.unsqueeze(-1).to(dtype=param_dtype)

        gate = 2.0 * torch.sigmoid(self.market_gate(regime))
        gated = (endogenous * gate) * valid

        counts = valid.cumsum(dim=1).clamp_min(1.0)
        causal_mean = gated.cumsum(dim=1) / counts
        causal_rms = (gated.square().cumsum(dim=1) / counts + 1e-6).sqrt()
        causal_variate_summary = torch.cat([causal_mean, causal_rms], dim=-1) * valid

        hidden = self.patch_stem(gated) + self.position_embedding[:, :sequence_length, :]
        if timestamps is not None:
            stamp_scales = torch.tensor(
                [60.0, 24.0, 7.0, 31.0, 12.0],
                device=ohlcv.device,
                dtype=param_dtype,
            )
            normalized_stamps = (timestamps.to(dtype=param_dtype) / stamp_scales).clamp(-2.0, 2.0)
            hidden = hidden + self.timestamp_projection(normalized_stamps)
        hidden = hidden * valid

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=ohlcv.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, causal_variate_summary, causal_mask)
            hidden = hidden * valid

        hidden = self.final_norm(hidden)
        if self.technical_adapter is not None:
            hidden = hidden + self.technical_adapter(ohlcv).to(dtype=hidden.dtype)
        hidden = hidden * valid
        return TimeSeriesBackboneOutput(last_hidden_state=hidden, attention_mask=mask)

