"""Pinned Chronos-2 historical encoder for the existing independent signal heads."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor, nn

from .backbones import _valid_mask, freeze_module
from .exceptions import ModelCapabilityError, OptionalDependencyError
from .lora import inject_lora
from .outputs import TimeSeriesBackboneOutput

CHRONOS2_PACKAGE_VERSION = "2.3.2"
CHRONOS2_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def verify_chronos2_dependency() -> str:
    """Fail closed on untested encoder API versions without downloading anything."""
    try:
        installed = version("chronos-forecasting")
    except PackageNotFoundError as error:
        raise OptionalDependencyError(
            "Chronos-2 requires the cloud-only Poetry extra: chronos2"
        ) from error
    if installed != CHRONOS2_PACKAGE_VERSION:
        raise OptionalDependencyError(
            f"Chronos-2 requires chronos-forecasting=={CHRONOS2_PACKAGE_VERSION}; "
            f"found {installed}"
        )
    return installed


class Chronos2Backbone(nn.Module):
    """Use native scaling and within-window group attention without batch leakage.

    Only observed OHLCV history enters the pretrained encoder. No future target
    or covariate is supplied. Its context patches feed the existing resampler;
    native regression/future placeholder tokens are excluded from that output.
    """

    def __init__(self, model: nn.Module, *, max_context: int = 128):
        super().__init__()
        if not all(callable(getattr(model, name, None)) for name in ("encode", "patch")):
            raise ModelCapabilityError("Chronos-2 must expose its native encode and patch methods")
        self.hidden_size = int(model.config.d_model)
        if not 0 < max_context <= model.chronos_config.context_length:
            raise ValueError("Requested context exceeds the pretrained Chronos-2 context")
        self.max_context = max_context
        self.model = freeze_module(model)
        self._predictor_trainable = False
        self.lora_module_names: tuple[str, ...] = ()
        self.unfrozen_parameter_names: tuple[str, ...] = ()
        self.technical_adapter = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "amazon/chronos-2",
        *,
        model_revision: str = CHRONOS2_MODEL_REVISION,
        local_files_only: bool = True,
        max_context: int = 128,
    ) -> Chronos2Backbone:
        verify_chronos2_dependency()
        from chronos import Chronos2Model

        model = Chronos2Model.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        return cls(model, max_context=max_context)

    def enable_lora(
        self, *, target_modules: Sequence[str], rank: int, alpha: float, dropout: float
    ) -> tuple[str, ...]:
        if self._predictor_trainable:
            raise RuntimeError("Chronos-2 encoder LoRA is already enabled")
        self.lora_module_names = inject_lora(
            self.model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            allowed_prefixes=("encoder.",),
        )
        self._predictor_trainable = True
        return self.lora_module_names

    @property
    def predictor_trainable(self) -> bool:
        return self._predictor_trainable

    def train(self, mode: bool = True) -> Chronos2Backbone:
        super().train(mode)
        self.model.train(mode if self._predictor_trainable else False)
        return self

    def forward(
        self, ohlcv: Tensor, attention_mask: Tensor | None = None, timestamps: Tensor | None = None
    ) -> TimeSeriesBackboneOutput:
        del timestamps
        if ohlcv.ndim != 3 or ohlcv.shape[-1] != 5:
            raise ValueError("Chronos-2 expects OHLCV shaped [batch, bars, 5]")
        batch, bars, channels = ohlcv.shape
        if bars > self.max_context:
            raise ValueError("Input exceeds the configured Chronos-2 context")
        mask = _valid_mask(
            attention_mask, batch_size=batch, sequence_length=bars, device=ohlcv.device
        )
        if attention_mask is not None:
            # Native Chronos places the final observation next to its forecast tokens.
            positions = torch.arange(bars, device=ohlcv.device)[None, :].expand(batch, bars)
            left_padding = bars - mask.sum(-1, keepdim=True)
            indices = (positions - left_padding).clamp_min(0)
            ohlcv = ohlcv.gather(1, indices[..., None].expand(batch, bars, channels))
            mask = positions >= left_padding
        context = ohlcv.transpose(1, 2).reshape(batch * channels, bars).float()
        context_mask = mask[:, None, :].expand(batch, channels, bars).reshape_as(context)
        # Native instance normalization ignores NaNs, not the separate mask.
        context = context.masked_fill(~context_mask, float("nan"))
        group_ids = torch.arange(batch, device=ohlcv.device).repeat_interleave(channels)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._predictor_trainable):
            encoded, _, _, patches = self.model.encode(
                context=context,
                context_mask=context_mask,
                group_ids=group_ids,
                num_output_patches=1,
            )
        hidden = encoded.last_hidden_state[:, :patches]
        hidden = hidden.reshape(batch, channels, patches, self.hidden_size)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, patches * channels, self.hidden_size)
        patch_mask = torch.nan_to_num(self.model.patch(context_mask.float()), nan=0.0).sum(-1) > 0
        patch_mask = patch_mask.reshape(batch, channels, patches).transpose(1, 2)
        patch_mask = patch_mask.reshape(batch, patches * channels)
        hidden = hidden.masked_fill(~patch_mask[..., None], 0.0)
        return TimeSeriesBackboneOutput(hidden, patch_mask)
