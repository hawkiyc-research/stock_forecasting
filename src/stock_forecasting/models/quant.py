"""Quant-only conditional alpha model built on reusable OHLCV representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .forecast import GatedBenchmarkConditioner, MultiHorizonAlphaHead
from .outputs import (
    DirectionForecastOutput,
    DirectionV2ForecastOutput,
    QuantEncoderOutput,
    QuantForecastOutput,
)
from .projector import CausalPerceiverResampler
from .ranking import same_date_ranking_loss
from .scale_features import historical_scale_features


class QuantForecastModel(nn.Module):
    """Encode paired histories for alpha quantiles and optional direction supervision."""

    def __init__(
        self,
        backbone: nn.Module,
        resampler: CausalPerceiverResampler,
        benchmark_conditioner: GatedBenchmarkConditioner,
        alpha_head: MultiHorizonAlphaHead,
        ranking_loss_weight: float = 0.0,
        ranking_max_pairs: int = 256,
        market_loss_weights: list[float] | None = None,
        direction_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.resampler = resampler
        self.benchmark_conditioner = benchmark_conditioner
        self.alpha_head = alpha_head
        self.direction_head = direction_head
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_max_pairs = ranking_max_pairs
        self.use_market_loss_weights = market_loss_weights is not None and any(
            weight != 1.0 for weight in market_loss_weights
        )
        self.register_buffer(
            "market_loss_weights", torch.tensor(market_loss_weights or [1.0] * 4), persistent=False
        )

    def encode_ohlcv(
        self,
        ohlcv: Tensor,
        *,
        attention_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> QuantEncoderOutput:
        """Expose reusable representations for a possible future multimodal model."""

        backbone_output = self.backbone(
            ohlcv,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
        resampler_dtype = self.resampler.context_projection.weight.dtype
        latent_tokens = self.resampler(
            backbone_output.last_hidden_state.to(dtype=resampler_dtype),
            None if attention_mask is None else backbone_output.attention_mask,
        )
        return QuantEncoderOutput(
            last_hidden_state=backbone_output.last_hidden_state,
            attention_mask=backbone_output.attention_mask,
            latent_tokens=latent_tokens,
        )

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
        market_ids: Tensor | None = None,
        ranking_group_ids: Tensor | None = None,
        security_ids: Tensor | None = None,
        ranking_pairs: Tensor | None = None,
        scale_inputs_validated: bool = False,
        target_absolute: Tensor | None = None,
        direction_bands: Tensor | None = None,
    ) -> QuantForecastOutput:
        scales = None
        branch = self.alpha_head.numeric_branch
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
                    extended=self.alpha_head.explicit_output_scale,
                    inputs_validated=scale_inputs_validated,
                )
        can_fuse_pair = (
            asset_ohlcv.shape == benchmark_ohlcv.shape
            and (asset_attention_mask is None) == (benchmark_attention_mask is None)
            and (asset_timestamps is None) == (benchmark_timestamps is None)
        )
        if can_fuse_pair:
            asset_batch_size = asset_ohlcv.shape[0]
            if asset_attention_mask is None:
                combined_mask = None
            else:
                assert benchmark_attention_mask is not None
                combined_mask = torch.cat(
                    [asset_attention_mask, benchmark_attention_mask],
                    dim=0,
                )
            if asset_timestamps is None:
                combined_timestamps = None
            else:
                assert benchmark_timestamps is not None
                combined_timestamps = torch.cat(
                    [asset_timestamps, benchmark_timestamps],
                    dim=0,
                )
            combined = self.encode_ohlcv(
                torch.cat([asset_ohlcv, benchmark_ohlcv], dim=0),
                attention_mask=combined_mask,
                timestamps=combined_timestamps,
            )
            asset_encoded = QuantEncoderOutput(
                last_hidden_state=combined.last_hidden_state[:asset_batch_size],
                attention_mask=combined.attention_mask[:asset_batch_size],
                latent_tokens=combined.latent_tokens[:asset_batch_size],
            )
            benchmark_encoded = QuantEncoderOutput(
                last_hidden_state=combined.last_hidden_state[asset_batch_size:],
                attention_mask=combined.attention_mask[asset_batch_size:],
                latent_tokens=combined.latent_tokens[asset_batch_size:],
            )
        else:
            asset_encoded = self.encode_ohlcv(
                asset_ohlcv,
                attention_mask=asset_attention_mask,
                timestamps=asset_timestamps,
            )
            benchmark_encoded = self.encode_ohlcv(
                benchmark_ohlcv,
                attention_mask=benchmark_attention_mask,
                timestamps=benchmark_timestamps,
            )
        conditioned, gate = self.benchmark_conditioner(
            asset_encoded.latent_tokens,
            benchmark_encoded.latent_tokens,
        )
        alpha_quantiles, ranking_scores = self.alpha_head(
            conditioned,
            scale_features=scales,
            benchmark_tokens=benchmark_encoded.latent_tokens,
            market_ids=market_ids,
            return_ranking=True,
        )
        alpha_quantiles = alpha_quantiles.float()
        pinball_loss = (
            None
            if target_alpha is None
            else self.alpha_head.pinball_loss(alpha_quantiles, target_alpha)
        )
        ranking_loss = None
        weighted_pinball = pinball_loss
        if self.training and target_alpha is not None and self.use_market_loss_weights:
            if market_ids is None:
                raise ValueError("Market-weighted training requires market IDs")
            weighted_pinball = self.alpha_head.pinball_loss(
                alpha_quantiles, target_alpha, self.market_loss_weights[market_ids].float()
            )
        if self.training and self.ranking_loss_weight and target_alpha is not None:
            if ranking_group_ids is None or security_ids is None:
                raise ValueError(
                    "Ranking training requires same-date/market and security identities"
                )
            ranking_loss = same_date_ranking_loss(
                alpha_quantiles if ranking_scores is None else ranking_scores,
                target_alpha,
                self.alpha_head.robust_scales,
                ranking_group_ids,
                security_ids,
                self.ranking_max_pairs,
                eligible_pairs=ranking_pairs,
            )
        total_loss = (
            weighted_pinball
            if ranking_loss is None
            else weighted_pinball + self.ranking_loss_weight * ranking_loss
        )
        absolute_logits = alpha_logits = absolute_loss = alpha_loss = None
        absolute_decision_logits = alpha_decision_logits = None
        representation_loss = direction_weighted_loss = None
        if self.direction_head is not None:
            if market_ids is None or direction_bands is None:
                raise ValueError("Direction predictions require market IDs and ex-ante cost bands")
            head = self.direction_head
            direction_v2 = getattr(head, "architecture", None) in (
                "market_residual_v2",
                "market_ordinal_v3",
            )
            read_kwargs = {}
            if direction_v2:
                from .scale_features import SCALE_STANDARDIZED_CLIP, _transform

                if scales is None or branch is None or branch.statistics is None:
                    raise ValueError(
                        "Direction v2 requires the existing train-fitted scale statistics"
                    )
                # Reuse the numerical branch's exact train-fitted transform and buffers.
                standardized = (
                    _transform(scales.float()) - branch.feature_center
                ) / branch.feature_scale
                read_kwargs = {
                    "standardized_scales": standardized.clamp(
                        -SCALE_STANDARDIZED_CLIP, SCALE_STANDARDIZED_CLIP
                    ),
                    "scale_features": scales,
                }
            hidden = head.read(
                asset_encoded.latent_tokens,
                benchmark_encoded.latent_tokens,
                conditioned,
                market_ids,
                direction_bands,
                **read_kwargs,
            )
            logits = head.logits(hidden, market_ids) if direction_v2 else head.logits(hidden)
            absolute_logits, alpha_logits = logits
            if direction_v2:
                absolute_decision_logits, alpha_decision_logits = head.decision_logits(
                    logits, market_ids
                )
            if (target_alpha is None) != (target_absolute is None):
                raise ValueError("Direction supervision requires both absolute and alpha labels")
            if target_absolute is not None:
                absolute_loss, alpha_loss, representation_loss = head.losses(
                    hidden,
                    (absolute_logits, alpha_logits),
                    (target_absolute, target_alpha),
                    direction_bands,
                    **({"market_ids": market_ids} if direction_v2 else {}),
                )
                weights = head.weights()
                direction_weighted_loss = weights[0] * absolute_loss + weights[1] * alpha_loss
                total_loss = total_loss + direction_weighted_loss
                if representation_loss is not None:
                    total_loss = total_loss + (
                        head.config.representation_weight
                        * head.loss_references[0]
                        * representation_loss
                    )
        direction_fields = (
            {}
            if self.direction_head is None
            else {
                "absolute_direction_logits": absolute_logits,
                "alpha_direction_logits": alpha_logits,
                "absolute_direction_loss": absolute_loss,
                "alpha_direction_loss": alpha_loss,
                "representation_loss": representation_loss,
                "direction_weighted_loss": direction_weighted_loss,
            }
        )
        if absolute_decision_logits is not None:
            direction_fields.update(
                absolute_direction_decision_logits=absolute_decision_logits,
                alpha_direction_decision_logits=alpha_decision_logits,
            )
        output_type = (
            QuantForecastOutput
            if self.direction_head is None
            else DirectionV2ForecastOutput
            if absolute_decision_logits is not None
            else DirectionForecastOutput
        )
        return output_type(
            loss=total_loss,
            pinball_loss=pinball_loss,
            alpha_quantiles=alpha_quantiles,
            asset_last_hidden_state=asset_encoded.last_hidden_state,
            benchmark_last_hidden_state=benchmark_encoded.last_hidden_state,
            asset_attention_mask=asset_encoded.attention_mask,
            benchmark_attention_mask=benchmark_encoded.attention_mask,
            asset_latent_tokens=asset_encoded.latent_tokens,
            benchmark_latent_tokens=benchmark_encoded.latent_tokens,
            conditioned_latent_tokens=conditioned,
            conditioning_gate=gate,
            ranking_loss=ranking_loss,
            ranking_scores=ranking_scores,
            weighted_pinball_loss=weighted_pinball,
            scale_features=scales,
            **direction_fields,
        )
