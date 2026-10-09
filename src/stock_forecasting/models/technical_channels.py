"""Five causal channels per stream; see the README direction model definitions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TECHNICAL_CHANNEL_NAMES = (
    "close_sma20",
    "sma5_sma20",
    "sma20_sma60",
    "prior20_range",
    "signed_volume20",
)


def technical_channels(ohlcv: Tensor) -> Tensor:
    """Use only each bar's prefix; zero channels until their full lookback exists."""
    with torch.autocast(device_type=ohlcv.device.type, enabled=False):
        values = ohlcv.float()
        close, volume = values[..., 3].clamp_min(1e-8), values[..., 4].clamp_min(0)
        time = torch.arange(close.shape[1], device=close.device)

        def mean(x: Tensor, period: int) -> Tensor:
            return F.avg_pool1d(F.pad(x.unsqueeze(1), (period - 1, 0)), period, stride=1)[:, 0]

        ma5, ma20, ma60 = (mean(close, period).clamp_min(1e-8) for period in (5, 20, 60))
        prior_high = F.pad(values[:, :-1, 1], (20, 0), value=-torch.inf).unfold(1, 20, 1).amax(-1)
        prior_low = F.pad(values[:, :-1, 2], (20, 0), value=torch.inf).unfold(1, 20, 1).amin(-1)
        span = (prior_high - prior_low).clamp_min(1e-8)
        position = (2 * close - prior_high - prior_low) / span
        change = torch.cat([torch.zeros_like(close[:, :1]), close[:, 1:] - close[:, :-1]], 1)
        signed_volume = mean(change.sign() * volume, 20) / mean(volume, 20).clamp_min(1e-8)
        channels = (
            torch.where(time >= 19, (close / ma20).log(), 0),
            torch.where(time >= 19, (ma5 / ma20).log(), 0),
            torch.where(time >= 59, (ma20 / ma60).log(), 0),
            torch.where((time >= 20) & (prior_high > prior_low), position, 0),
            torch.where(time >= 20, signed_volume, 0),
        )
        # Bound feature leverage only; labels and extreme market returns are untouched.
        return torch.stack(channels, -1).clamp(-5, 5)


class TechnicalTokenAdapter(nn.Module):
    """Zero-initialized residual in predictor embeddings, not a head-only shortcut."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(len(TECHNICAL_CHANNEL_NAMES), hidden_size, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, ohlcv: Tensor) -> Tensor:
        return self.projection(technical_channels(ohlcv).to(self.projection.weight.dtype))
