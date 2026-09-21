"""Build and persist matching full-data baselines on a dedicated GPU Pod."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from datetime import UTC, datetime
from pathlib import Path

import torch

from stock_forecasting.baseline_build import build_baselines
from stock_forecasting.baseline_contract import (
    baseline_runtime_payload,
    validate_local_configuration,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=os.environ.get("RUNPOD_SELECTION_FILE"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or os.environ.get("RUNPOD_ROLE") != "gpu-baseline":
        raise ValueError("Full baseline building requires the dedicated CUDA Pod workflow")
    root = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    if args.selection is None:
        parser.error("The baseline workflow requires an immutable data selection")
    project = Path(__file__).resolve().parents[3]
    selection_tools = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    selection_path, selection = selection_tools["_resolve_selection_path"](
        project, str(args.selection), validate_local_config=False
    )
    os.environ["RUNPOD_SELECTION_FILE"] = str(selection_path)
    parameters = json.loads((project / "configs/baseline.json").read_text())
    validate_local_configuration(project, selection, parameters)
    config = ExperimentConfig.model_validate(baseline_runtime_payload(selection, parameters, root))
    lifecycle = root / "lifecycle/stage1/baseline.json"

    def publish(state, error=None):
        atomic_write_json(
            lifecycle,
            {
                "schema_version": 1,
                "kind": "stage1-baseline",
                "state": state,
                "pod_id": os.environ["RUNPOD_POD_ID"],
                "wandb_run_id": os.environ["WANDB_RUN_ID"],
                "launch_id": os.environ.get("RUNPOD_LAUNCH_ID", "manual-baseline"),
                "generated_at": datetime.now(UTC).isoformat(),
                "baseline_completed": state == "ready",
                "error": error,
            },
        )

    publish("preparing")
    try:
        payload = build_baselines(config)
        publish("finalizing" if os.environ.get("RUNPOD_TMUX_LOG_FILE") else "ready")
        print(json.dumps({"state": "complete", "baseline_id": payload["identity"]["baseline_id"]}))
    except BaseException as error:
        publish(
            "finalizing" if os.environ.get("RUNPOD_TMUX_LOG_FILE") else "failed",
            f"{type(error).__name__}: {error}",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
