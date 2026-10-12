"""Cost-ordered conditional direction probabilities and market-guided reading."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .direction import MarketResidualDualDirectionHead
from .scale_features import EXTENDED_FEATURE_NAMES

# Only benchmark state observed at the prediction cutoff conditions feature selection.
MARKET_STATE_NAMES = (
    "benchmark_log_return_std_20d",
    "benchmark_log_return_std_60d",
    "benchmark_close_cv_context",
    "benchmark_return_mean_5d",
    "benchmark_return_mean_20d",
    "benchmark_volume_ratio_20d",
    "benchmark_drawdown_60d",
)
MARKET_STATE_INDICES = tuple(EXTENDED_FEATURE_NAMES.index(name) for name in MARKET_STATE_NAMES)


def conditional_log_probabilities(conditional_logits: Tensor) -> Tensor:
    """Chain rule: P(up)<=P(neutral or up), including saturated FP32 logits.

    NLL of these probabilities equals BCE for crossing the lower boundary on all
    samples plus BCE for crossing the upper boundary only on non-down samples.
    This is a proper conditional likelihood, not two equally weighted BCE means.
    """
    lower, upper_given_lower = conditional_logits.float().unbind(-1)
    lower_yes = F.logsigmoid(lower)
    return torch.stack(
        (
            F.logsigmoid(-lower),
            lower_yes + F.logsigmoid(-upper_given_lower),
            lower_yes + F.logsigmoid(upper_given_lower),
        ),
        -1,
    )


class MarketOrdinalDualDirectionHead(MarketResidualDualDirectionHead):
    """Retain population/decision outputs with conditional rather than flat logits."""

    architecture = "market_ordinal_v3"

    def __init__(self, dimension: int, horizons, config, dropout: float) -> None:
        super().__init__(dimension, horizons, config, dropout)
        self.absolute = nn.Linear(dimension, 2)
        self.alpha = nn.Linear(dimension, 2)
        self.market_signal[-1] = nn.Linear(config.market_residual_hidden_dim, 4 * 2 * 2)
        nn.init.zeros_(self.market_signal[-1].weight)
        nn.init.zeros_(self.market_signal[-1].bias)
        self.state_gate = nn.Sequential(
            nn.Linear(len(MARKET_STATE_NAMES), config.market_residual_hidden_dim),
            nn.GELU(),
            nn.Linear(config.market_residual_hidden_dim, 23),
        )
        # Near-neutral routing avoids suppressing pretrained streams at initialization.
        nn.init.normal_(self.state_gate[-1].weight, std=0.01)
        nn.init.zeros_(self.state_gate[-1].bias)

    def feature_gates(self, standardized_scales: Tensor) -> tuple[Tensor, Tensor]:
        if standardized_scales.ndim != 2 or standardized_scales.shape[-1] != 20:
            raise ValueError("Market-state gating requires 20 train-standardized past features")
        state = standardized_scales[:, MARKET_STATE_INDICES]
        gates = self.state_gate(state.to(self.state_gate[0].weight.dtype)).float()
        # Bounded gates use no labels, cost metadata, future bars or current batch statistics.
        return 3 * gates[:, :3].softmax(-1), 2 * gates[:, 3:].sigmoid()

    def read(
        self,
        asset,
        benchmark,
        conditioned,
        market_ids,
        bands,
        *,
        standardized_scales,
        scale_features,
    ):
        streams, features = self.feature_gates(standardized_scales)
        return super().read(
            asset * streams[:, 0, None, None].to(asset.dtype),
            benchmark * streams[:, 1, None, None].to(benchmark.dtype),
            conditioned * streams[:, 2, None, None].to(conditioned.dtype),
            market_ids,
            bands,
            standardized_scales=standardized_scales * features,
            scale_features=scale_features,
        )

    def logits(self, hidden: Tensor, market_ids: Tensor) -> tuple[Tensor, Tensor]:
        common = torch.stack((self.absolute(hidden), self.alpha(hidden)), dim=2).float()
        residual = self.market_signal(hidden).reshape(*hidden.shape[:2], 4, 2, 2)
        index = market_ids[:, None, None, None, None].expand(-1, hidden.shape[1], 1, 2, 2)
        signal = common + residual.gather(2, index).squeeze(2).float()
        priors = self.train_priors[market_ids].permute(0, 2, 1, 3).float()
        # Express market priors in the same conditional coordinates as the signal.
        non_down = priors[..., 1] + priors[..., 2]
        offset = torch.stack(
            (non_down.log() - priors[..., 0].log(), priors[..., 2].log() - priors[..., 1].log()), -1
        )
        log_probability = conditional_log_probabilities(
            signal + self.config.class_balance_power * offset
        )
        return log_probability[:, :, 0], log_probability[:, :, 1]

    def reference_state(self) -> dict:
        return {
            **super().reference_state(),
            "schema_version": 3,
            "conditional_order": ["above_lower", "above_upper_given_above_lower"],
        }

    def restore_references(self, state: dict) -> None:
        if (
            state.get("schema_version") != 3
            or state.get("architecture") != self.architecture
            or state.get("conditional_order") != ["above_lower", "above_upper_given_above_lower"]
        ):
            raise ValueError("Invalid ordinal-v3 train reference architecture or conditional order")
        # Reuse strict array/anchor validation without accepting a v2 checkpoint as v3.
        super().restore_references({**state, "schema_version": 2})
