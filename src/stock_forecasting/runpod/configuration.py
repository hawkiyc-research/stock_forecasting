"""Apply explicit, selection-pinned launch overrides at main-model CLI boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stock_forecasting.config import ExperimentConfig


def training_seed_override(path: str | Path, environment: Mapping[str, str]) -> int | None:
    """Never override a saved resolved config or silently trust a free-form seed env."""
    requested = environment.get("RUNPOD_TRAINING_SEED", "")
    if not requested:
        return None
    configured = environment.get("RUNPOD_CONFIG", "")
    if not configured:
        raise ValueError("A training seed override requires the selected RUNPOD_CONFIG")
    config_path = Path(configured)
    if not config_path.is_absolute():
        config_path = Path(environment.get("PROJECT_ROOT", str(Path.cwd()))) / config_path
    if Path(path).resolve() != config_path.resolve():
        # Checkpoint readers must retain the seed saved in that checkpoint.
        return None
    if re.fullmatch(r"0|[1-9][0-9]*", requested) is None or not 0 <= int(requested) < 2**32:
        raise ValueError("Training seed must be an integer from 0 through 4294967295")
    selection_path = environment.get("RUNPOD_REMOTE_SELECTION_PATH") or environment.get(
        "RUNPOD_SELECTION_FILE"
    )
    if not selection_path:
        raise ValueError("A training seed override requires the run's immutable selection")
    source = Path(selection_path)
    if not source.is_file() or source.stat().st_size > 1024 * 1024:
        raise ValueError("The run's immutable selection is missing or oversized")
    selection = json.loads(source.read_text())
    if not isinstance(selection, dict) or not isinstance(selection.get("stage"), dict):
        raise ValueError("The run's immutable selection has an invalid structure")
    core = {
        key: selection.get(key)
        for key in ("schema_version", "stage", "dataset_request", "runtime")
    }
    digest = hashlib.sha256(
        json.dumps(
            core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    stage = selection["stage"]
    seed = stage.get("training_seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != int(requested)
        or stage.get("config_path") != configured
        or stage.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()
        or selection.get("selection_sha256") != digest
        or selection.get("selection_id") != "selection-" + digest[:16]
        or environment.get("RUNPOD_SELECTION_SHA256") != digest
        or environment.get("RUNPOD_SELECTION_ID") != selection.get("selection_id")
    ):
        raise ValueError("Training seed/config does not match the run's immutable selection")
    return seed


def load_run_config(path: str | Path) -> ExperimentConfig:
    """Resolve launch-only seed overrides before constructing a training/resume contract."""
    from stock_forecasting.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(path)
    seed = training_seed_override(path, os.environ)
    if seed is not None:
        config.training.seed = seed
    return config
