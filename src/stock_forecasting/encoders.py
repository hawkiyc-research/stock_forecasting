"""Pluggable frozen encoders for the cross-stock model (protocol 13.13 D1).

Contract: raw OHLCV window ``(B, L, 5)`` in, embedding ``(B, D_enc)`` out; every adapter
normalizes inside. Frozen encoders embed every sample of a split once; the embeddings are
cached beside the cross-stock cache as ``embeddings.npy`` plus a Parquet index keyed by
(symbol, cutoff), so walk-forward refits train only the cross-stock module.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from stock_forecasting.data.manifest import atomic_write_json, sha256_file

EMBEDDING_SCHEMA = "frozen-encoder-embeddings-v1"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    repositories: tuple[str, ...]
    family: str  # kronos | chronos2
    pooling: str


ENCODERS: dict[str, EncoderSpec] = {
    "kronos-small": EncoderSpec(
        name="kronos-small",
        repositories=("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
        family="kronos",
        pooling="last_position",
    ),
    "chronos2-small": EncoderSpec(
        name="chronos2-small",
        repositories=("autogluon/chronos-2-small",),
        family="chronos2",
        pooling="mean_over_variates_and_patches",
    ),
}


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prefetch(spec: EncoderSpec, *, local_files_only: bool = False) -> dict[str, str]:
    """Download (or verify) the encoder's repositories; returns repo → resolved revision."""
    from huggingface_hub import snapshot_download

    revisions = {}
    for repository in spec.repositories:
        path = Path(snapshot_download(repository, local_files_only=local_files_only))
        # snapshot_download returns .../snapshots/<commit>; record that commit.
        revisions[repository] = path.name
    return revisions


class KronosEncoder(nn.Module):
    """Frozen Kronos hidden state of the last window position."""

    def __init__(self, spec: EncoderSpec, *, kronos_source_root: str, kronos_source_revision: str):
        super().__init__()
        from stock_forecasting.factory import _prepare_kronos_import
        from stock_forecasting.models.backbones import KronosBackbone

        _prepare_kronos_import(kronos_source_root, kronos_source_revision)
        self.backbone = KronosBackbone.from_pretrained(
            spec.repositories[0], spec.repositories[1], local_files_only=True
        )
        self.backbone.eval()
        self.dim = int(self.backbone.hidden_size)

    @torch.no_grad()
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        output = self.backbone(windows.float())
        return output.last_hidden_state[:, -1, :].float()


class Chronos2Encoder(nn.Module):
    """Frozen Chronos-2 encoder; the five OHLCV channels are the variates of one series."""

    def __init__(self, spec: EncoderSpec, *, device: torch.device):
        super().__init__()
        try:
            from chronos import Chronos2Pipeline
        except ImportError as error:  # pragma: no cover - depends on the Pod environment
            raise ImportError(
                "chronos-forecasting is required for the Chronos-2 encoder"
            ) from error
        self.pipeline = Chronos2Pipeline.from_pretrained(
            spec.repositories[0], device_map=str(device), local_files_only=True
        )
        if not callable(getattr(self.pipeline, "embed", None)):
            raise RuntimeError("Chronos2Pipeline.embed is unavailable in the installed version")
        self.dim: int | None = None

    @torch.no_grad()
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # Scale-free inputs: log prices relative to the last close, volume relative to its median.
        x = windows.float()
        close_last = x[:, -1:, 3:4].clamp_min(1e-8)
        prices = torch.log(x[:, :, :4].clamp_min(1e-8) / close_last)
        volume = x[:, :, 4:5].clamp_min(0.0)
        median = volume.median(dim=1, keepdim=True).values.clamp_min(1e-8)
        series = torch.cat([prices, torch.log1p(volume / median)], dim=-1).transpose(1, 2)
        embeddings, _scales = self.pipeline.embed(
            [row.cpu() for row in series], batch_size=len(series)
        )
        pooled = torch.stack([e.float().mean(dim=(0, 1)) for e in embeddings])
        self.dim = int(pooled.shape[1])
        return pooled


def load_encoder(name: str, *, device: torch.device, **kronos_kwargs: Any) -> nn.Module:
    spec = ENCODERS[name]
    if spec.family == "kronos":
        encoder = KronosEncoder(spec, **kronos_kwargs)
    elif spec.family == "chronos2":
        encoder = Chronos2Encoder(spec, device=device)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported encoder family: {spec.family}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder.to(device).eval()


def embeddings_root(cache_root: Path, name: str) -> Path:
    return cache_root / "embeddings" / name


def embeddings_ready(root: Path, *, samples: int) -> bool:
    marker = root / "embeddings.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    return (
        payload.get("state") == "ready"
        and payload.get("samples") == samples
        and (root / "embeddings.npy").is_file()
    )


def ensure_embeddings(
    cache,
    name: str,
    *,
    encoder: nn.Module | None = None,
    device: torch.device,
    batch_size: int = 512,
    revisions: dict[str, str] | None = None,
    encoder_factory=None,
) -> Path:
    """Embed every window of the split cache once; reuse the store when it is ready."""
    root = embeddings_root(cache.root, name)
    samples = len(cache)
    if embeddings_ready(root, samples=samples):
        return root
    root.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        encoder = (encoder_factory or load_encoder)(name, device=device)
    started = time.monotonic()
    pending = root / "embeddings.pending.npy"
    output = None
    for start in range(0, samples, batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(cache.windows[start : start + batch_size]))
        values = encoder(chunk.to(device)).float().cpu().numpy().astype(np.float32)
        if output is None:
            output = np.lib.format.open_memmap(
                pending, mode="w+", dtype="float32", shape=(samples, values.shape[1])
            )
        output[start : start + len(values)] = values
    if output is None:
        raise ValueError("The split cache has no samples to embed")
    if not np.isfinite(output).all():
        raise ValueError(f"{name} produced non-finite embeddings")
    output.flush()
    dim = int(output.shape[1])
    del output
    pending.replace(root / "embeddings.npy")
    import pandas as pd

    pd.DataFrame(
        {
            "ordinal": np.arange(samples, dtype=np.int64),
            "symbol": np.asarray(cache.symbols, dtype=object)[np.asarray(cache.symbol_codes)],
            "timestamp": pd.to_datetime(np.asarray(cache.cutoff_ns, dtype="int64"), utc=True),
        }
    ).to_parquet(root / "embeddings-index.parquet", index=False)
    atomic_write_json(
        root / "embeddings.json",
        {
            "state": "ready",
            "schema": EMBEDDING_SCHEMA,
            "encoder": name,
            "spec": ENCODERS[name].__dict__,
            "revisions": revisions or {},
            "split": cache.split,
            "samples": samples,
            "dim": dim,
            "implementation_sha256": implementation_sha256(),
            "embeddings_sha256": sha256_file(root / "embeddings.npy"),
            "seconds": time.monotonic() - started,
        },
    )
    return root


def open_embeddings(root: Path) -> np.ndarray:
    payload = json.loads((root / "embeddings.json").read_text())
    if payload.get("state") != "ready":
        raise ValueError("Embedding store is not ready")
    return np.load(root / "embeddings.npy", mmap_mode="r")
