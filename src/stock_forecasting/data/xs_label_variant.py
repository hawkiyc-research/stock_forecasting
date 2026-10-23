"""Label-variant arrays beside a cross-stock sample cache (protocol 13.16).

The sample cache (``xs_cache``) carries the gauss-rank labels of the standard store. A
residualized label variant supplies different ranks and, in addition, the residual targets the
model is scored against. Both are written once per (cache, variant) under
``<cache root>/label-variants/<identity digest>/`` as memory-mapped ``.npy`` arrays. The cache
itself is never touched, so its identity, and the frozen-encoder embedding caches keyed on it,
stay valid across label variants.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.data.residual_labels import (
    attach_residual_labels,
    residual_column,
    residual_label_identity,
)

LABEL_VARIANT_SCHEMA = "cross-stock-label-variant-v1"
ARRAYS = ("ranks", "residual")
MARKER_NAME = "xs-label-variant.json"


def variant_identity(cache_root: Path, source, min_group_size: int, spec: dict) -> dict:
    return {
        "schema": LABEL_VARIANT_SCHEMA,
        "cache": Path(cache_root).name,
        "labels": residual_label_identity(source, min_group_size, spec),
    }


def variant_root(cache_root: Path, source, min_group_size: int, spec: dict) -> tuple[Path, dict]:
    identity = variant_identity(cache_root, source, min_group_size, spec)
    root = Path(cache_root) / "label-variants" / canonical_json_sha256(identity)
    return root, identity


def _ready(root: Path, identity: dict) -> bool:
    marker = root / MARKER_NAME
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    if payload.get("identity") != identity or payload.get("state") != "ready":
        raise ValueError("Cross-stock label-variant arrays have an inconsistent identity")
    for name in ARRAYS:
        if not (root / f"{name}.npy").is_file():
            raise ValueError(f"Cross-stock label-variant arrays are missing {name}.npy")
    return True


def ensure_label_variant(cache, source, min_group_size: int, spec: dict) -> Path:
    """Write the variant's ranks and residual targets beside the cache; reuse them when ready.

    The residual-label store must already be built (``cpu-features --label-variant``); the
    rows are checked against the cache's cutoffs so the arrays line up sample for sample.
    """
    from stock_forecasting.data.sample_universe import _exclusive_build
    from stock_forecasting.walk_forward_lgbm import read_split_rank_labels

    root, identity = variant_root(cache.root, source, min_group_size, spec)
    if _ready(root, identity):
        return root
    label_root = attach_residual_labels(source, min_group_size, spec)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_build(root):
        if _ready(root, identity):
            return root
        started = time.monotonic()
        horizons = [int(h) for h in cache.horizons]
        labels = read_split_rank_labels(label_root)
        if len(labels) != len(cache):
            raise ValueError(f"{cache.split}: label-variant rows differ from the sample cache")
        stamps = np.asarray(labels["timestamp"].astype("int64"))
        if (stamps != np.asarray(cache.cutoff_ns, dtype=np.int64)).any():
            raise ValueError(f"{cache.split}: label-variant cutoffs differ from the sample cache")
        missing = [
            column
            for h in horizons
            for column in (f"rank_h{h:02d}", residual_column(h))
            if column not in labels.columns
        ]
        if missing:
            raise ValueError(f"{cache.split}: label-variant store lacks {missing}")
        pending = root / "pending"
        pending.mkdir(exist_ok=True)
        np.save(
            pending / "ranks.npy",
            labels[[f"rank_h{h:02d}" for h in horizons]].to_numpy(dtype=np.float32),
        )
        np.save(
            pending / "residual.npy",
            labels[[residual_column(h) for h in horizons]].to_numpy(dtype=np.float32),
        )
        files = {}
        for name in ARRAYS:
            (pending / f"{name}.npy").replace(root / f"{name}.npy")
            files[f"{name}.npy"] = {
                "bytes": (root / f"{name}.npy").stat().st_size,
                "sha256": sha256_file(root / f"{name}.npy"),
            }
        pending.rmdir()
        atomic_write_json(
            root / MARKER_NAME,
            {
                "state": "ready",
                "identity": identity,
                "split": cache.split,
                "variant": spec["name"],
                "label_root": str(label_root),
                "samples": len(labels),
                "horizons": horizons,
                "files": files,
                "seconds": time.monotonic() - started,
            },
        )
    return root


@dataclass(frozen=True)
class LabelVariantArrays:
    """Memory-mapped view of one split's variant ranks and residual targets."""

    root: Path
    split: str
    variant: str
    ranks: np.ndarray
    residual: np.ndarray

    def __len__(self) -> int:
        return len(self.ranks)


def open_label_variant(root: Path) -> LabelVariantArrays:
    payload = json.loads((Path(root) / MARKER_NAME).read_text())
    if payload.get("state") != "ready":
        raise ValueError("Cross-stock label-variant arrays are not ready")
    arrays = {name: np.load(Path(root) / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    return LabelVariantArrays(
        root=Path(root), split=payload["split"], variant=payload["variant"], **arrays
    )


__all__ = [
    "LABEL_VARIANT_SCHEMA",
    "LabelVariantArrays",
    "ensure_label_variant",
    "open_label_variant",
    "variant_identity",
    "variant_root",
]
