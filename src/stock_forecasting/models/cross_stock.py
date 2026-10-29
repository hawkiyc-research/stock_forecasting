"""D0 ``t-xs-scratch``: cross-stock transformer for daily ranking (protocol 13.13 item 1).

Intra-stock encoder (patch transformer over the OHLCV window) + feature side input → every stock
of one signal date and market as one batch → inter-stock attention → market-state gating →
ranking scores per horizon. Ablations A1-A4 are single switches on ``CrossStockConfig``. The
encoder follows the pluggable contract ``(B, L, F) -> (B, D_enc)``; a frozen encoder's cached
embeddings can replace it through ``encoded``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

MODEL_SCHEMA = "cross-stock-transformer-v1"


@dataclass(frozen=True)
class CrossStockConfig:
    window: int = 128
    channels: int = 5
    patch: int = 8
    d_model: int = 128
    heads: int = 4
    encoder_layers: int = 2
    cross_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.1
    feature_dim: int = 183
    horizons: int = 14
    # Ablations (protocol 13.13): A1 cross attention off, A3 features off, A4 gating off;
    # A2 (lambda = 0) lives in the loss.
    inter_stock_attention: bool = True
    feature_input: bool = True
    market_gating: bool = True
    encoder_dim: int | None = None  # D_enc of an external (cached) encoder; None = scratch

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_windows(windows: torch.Tensor) -> torch.Tensor:
    """Scale-free channels: log price relative to the last close, log volume to its median."""
    x = windows.float()
    close_last = x[:, -1:, 3:4].clamp_min(1e-8)
    prices = torch.log(x[:, :, :4].clamp_min(1e-8) / close_last)
    volume = x[:, :, 4:5].clamp_min(0.0)
    median = volume.median(dim=1, keepdim=True).values.clamp_min(1e-8)
    log_volume = torch.log1p(volume / median) - torch.log1p(torch.ones_like(median))
    return torch.cat([prices, log_volume], dim=-1).clamp(-5.0, 5.0)


def cross_sectional_zscore(features: torch.Tensor) -> torch.Tensor:
    """Standardize every feature across the stocks of the batch; NaN → 0, clipped to ±5."""
    x = features.float()
    valid = torch.isfinite(x)
    filled = torch.where(valid, x, torch.zeros_like(x))
    count = valid.sum(dim=0, keepdim=True).clamp_min(1)
    mean = filled.sum(dim=0, keepdim=True) / count
    centered = torch.where(valid, filled - mean, torch.zeros_like(x))
    std = torch.sqrt((centered**2).sum(dim=0, keepdim=True) / count).clamp_min(1e-6)
    return (centered / std).clamp(-5.0, 5.0)


class PatchEncoder(nn.Module):
    """Scratch intra-stock encoder: non-overlapping patches → transformer → mean pool."""

    def __init__(self, config: CrossStockConfig) -> None:
        super().__init__()
        if config.window % config.patch:
            raise ValueError("window must be a multiple of patch")
        self.config = config
        tokens = config.window // config.patch
        self.embed = nn.Linear(config.patch * config.channels, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, tokens, config.d_model))
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.heads,
            config.feedforward,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers)
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = normalize_windows(windows)
        batch, length, channels = x.shape
        patches = x.reshape(batch, length // self.config.patch, self.config.patch * channels)
        tokens = self.embed(patches) + self.position
        return self.norm(self.encoder(tokens).mean(dim=1))


class CrossStockModel(nn.Module):
    def __init__(self, config: CrossStockConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if encoder is None and config.encoder_dim is None:
            encoder = PatchEncoder(config)
        self.encoder = encoder
        encoder_dim = config.encoder_dim or config.d_model
        self.project = (
            nn.Identity()
            if encoder_dim == config.d_model
            else nn.Linear(encoder_dim, config.d_model)
        )
        if config.feature_input:
            self.feature_mlp = nn.Sequential(
                nn.Linear(config.feature_dim, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
            self.fuse = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.GELU())
        if config.inter_stock_attention:
            layer = nn.TransformerEncoderLayer(
                config.d_model,
                config.heads,
                config.feedforward,
                config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross = nn.TransformerEncoder(layer, config.cross_layers)
        if config.market_gating:
            self.gate = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
                nn.Sigmoid(),
            )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.horizons)

    def encode(self, windows: torch.Tensor | None, encoded: torch.Tensor | None) -> torch.Tensor:
        if encoded is None:
            if self.encoder is None or windows is None:
                raise ValueError(
                    "Either raw windows with an encoder or cached embeddings are required"
                )
            encoded = self.encoder(windows)
        return self.project(encoded.float())

    def forward(
        self,
        windows: torch.Tensor | None,
        features: torch.Tensor | None = None,
        *,
        encoded: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores (S, horizons) for the S stocks of one signal date and market.

        ``valid`` (S,) bool marks the real rows when the group is padded to a fixed batch size
        (static shapes let the trainer replay CUDA graphs). Padded rows carry NaN features and
        labels, are left out of the NaN-aware cross-sectional z-score, are never attended to as
        keys and do not enter the market-state mean, so every valid row's score equals the
        unpadded computation.
        """
        tokens = self.encode(windows, encoded)
        if self.config.feature_input:
            if features is None:
                raise ValueError("feature_input requires the feature-library rows")
            side = self.feature_mlp(cross_sectional_zscore(features))
            tokens = self.fuse(torch.cat([tokens, side], dim=-1))
        if self.config.inter_stock_attention:
            padding = None if valid is None else (~valid).unsqueeze(0)
            tokens = self.cross(tokens.unsqueeze(0), src_key_padding_mask=padding).squeeze(0)
        if self.config.market_gating:
            if valid is None:
                state = tokens.mean(dim=0, keepdim=True)
            else:
                weight = valid.to(tokens.dtype).unsqueeze(1)
                state = (tokens * weight).sum(dim=0, keepdim=True) / weight.sum().clamp_min(1.0)
            tokens = tokens * self.gate(state)
        return self.head(self.norm(tokens))


def cross_sectional_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    lambda_corr: float,
    *,
    check_labels: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MSE on the gauss-rank labels + lambda * (1 - cross-sectional Pearson correlation).

    NaN targets (unlabelled groups) are masked; both terms average over horizons. The returned
    details are detached 0-d tensors, and with ``check_labels=False`` (training groups are
    pre-filtered to labelled ones) nothing here reads a device value back to the host, so the
    per-group steps queue without a sync.
    """
    mask = torch.isfinite(targets)
    if check_labels and not bool(mask.any()):
        raise ValueError("A batch without any labelled sample cannot contribute to the loss")
    scores = scores.float()
    targets = torch.where(mask, targets.float(), torch.zeros_like(scores))
    weight = mask.float()
    count = weight.sum(dim=0).clamp_min(1.0)
    mse = (((scores - targets) ** 2) * weight).sum(dim=0) / count
    score_mean = (scores * weight).sum(dim=0) / count
    target_mean = (targets * weight).sum(dim=0) / count
    ds = (scores - score_mean) * weight
    dt = (targets - target_mean) * weight
    corr = (ds * dt).sum(dim=0) / torch.sqrt((ds**2).sum(dim=0) * (dt**2).sum(dim=0) + 1e-12)
    usable = (count > 2).float()
    usable_count = usable.sum().clamp_min(1.0)
    # Horizons with fewer than three labelled rows contribute nothing (data-independent form).
    corr_term = ((1.0 - corr) * usable).sum() / usable_count
    loss = mse.mean() + lambda_corr * corr_term
    corr_mean = (corr * usable).sum() / usable_count
    return loss, {"mse": mse.mean().detach(), "corr": corr_mean.detach()}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
