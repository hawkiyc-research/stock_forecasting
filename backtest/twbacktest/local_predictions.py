"""Place local Kronos predictions (the inference/ subproject) on the Taiwan rows of a split.

Local runs cover only Taiwan rows and record each row's stock and date, so they are
joined to the evaluation arrays by (symbol, date) instead of by row order. The join
must be one-to-one and complete (protocol section 13.7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest.data import DOWN, UP, SplitIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direction_score(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities[..., UP] - probabilities[..., DOWN]


def load_local_scores(directory: Path, index: SplitIndex) -> tuple[np.ndarray, dict[str, Any]]:
    """Scores [rows, horizons] for every Taiwan row of the split, and the run record."""
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    if status.get("state") != "complete" or run.get("kind") != "local-signal-predictions":
        raise ValueError(f"{directory} is not a completed local prediction run")
    if run.get("split") != index.split:
        raise ValueError(f"{directory} predicts {run.get('split')}, not {index.split}")
    path = directory / "predictions.npy"
    if _sha256(path) != run.get("predictions_sha256"):
        raise ValueError(f"{path} differs from its recorded SHA-256")
    values = np.load(path).astype(np.float64)
    keys = pd.read_parquet(directory / "keys.parquet")
    if len(keys) != len(values):
        raise ValueError("Local prediction keys and values have different lengths")
    if not index.symbol_names:
        raise ValueError("The split index carries no symbol names to join on")
    rows = pd.DataFrame({
        "symbol": np.asarray(index.symbol_names)[index.symbol_codes],
        "date": np.asarray(index.dates)[index.date_codes],
        "row": np.arange(index.count),
    })
    predicted = pd.DataFrame({
        "symbol": keys["symbol"].astype(str).to_numpy(),
        "date": keys["cutoff_at"].astype(str).str[:10].to_numpy(),
        "position": np.arange(len(keys)),
    })
    if predicted.duplicated(["symbol", "date"]).any():
        raise ValueError("Local predictions repeat a stock and date")
    joined = predicted.merge(rows, on=["symbol", "date"], how="left", validate="one_to_one")
    if joined["row"].isna().any():
        raise ValueError("Local predictions include rows outside the evaluation population")
    if len(joined) != index.count:
        raise ValueError(
            f"Local predictions cover {len(joined)} of {index.count} Taiwan rows"
        )
    scores = _direction_score(values) if values.ndim == 3 else values
    output = np.full((index.count, scores.shape[1]), np.nan)
    output[joined["row"].astype(np.int64).to_numpy()] = scores[joined["position"].to_numpy()]
    return output, run
