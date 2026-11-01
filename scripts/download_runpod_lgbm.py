#!/usr/bin/env python3
"""Download the results of one cpu-lgbm (B0) or xs-train (D0/D1) job (protocol 13.13).

Every file is verified against the SHA-256 recorded in the job's walk-forward manifest; the
transfers go through the credential-isolated S3 wrapper. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))
from download_runpod_features import LAUNCH_PATTERN, Reader, transfer_timeout

from stock_forecasting.model_lineage import LINEAGE_NAME, REGISTRY_DIR, update_registry

WORKFLOWS = {
    "lgbm": {
        "lifecycle_key": "lifecycle/research/lgbm.json",
        "lifecycle_kind": "research-lgbm",
        "remote_root": "diagnostics/lgbm",
        "session": "stock-forecasting-cpu-lgbm",
        "manifest_kind": "walk-forward-lgbm",
        "local_root": "artifacts/lgbm",
        "label": "LightGBM baseline",
    },
    "xs": {
        "lifecycle_key": "lifecycle/research/xs-train.json",
        "lifecycle_kind": "research-xs-train",
        "remote_root": "diagnostics/xs",
        "session": "stock-forecasting-xs-train",
        "manifest_kind": "cross-stock-walk-forward",
        "local_root": "artifacts/xs",
        "label": "cross-stock experiment",
    },
}
MANIFEST_NAME = "walk-forward-manifest.json"
FILE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,80}")
# Saved P3-fold models (2026-11-02 rule): models/fold-XX/<file>.
MODEL_PATTERN = re.compile(r"models/fold-[0-9]{2}/[A-Za-z0-9_.-]{1,80}")


def summary_lines(results: dict) -> list[str]:
    lines = [f"backend={results.get('backend')} fits={results.get('fits', {}).get('count')}"]
    decision = results.get("decision")
    if isinstance(decision, dict):
        lines.append(
            f"{decision.get('hypothesis')}: cells={decision.get('cells')} "
            f"any_cell={decision.get('passed_any_cell')} -> {decision.get('disposition')}"
        )
    for cell in results.get("cells", []):
        if cell.get("model") == results.get("model") and cell.get("period") == "P1+P2":
            icir = cell["rank_icir"]
            lines.append(
                f"{cell['cell']}: rank_ic={cell['rank_ic']:.4f} "
                f"ci95=[{cell['ci95_low']:.4f}, {cell['ci95_high']:.4f}] "
                f"icir={None if icir is None else round(icir, 3)} days={cell['days']}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workflow", choices=sorted(WORKFLOWS), default="lgbm")
    parser.add_argument("launch_id", nargs="?")
    args = parser.parse_args(argv)
    workflow = WORKFLOWS[args.workflow]
    label, remote_root, session = workflow["label"], workflow["remote_root"], workflow["session"]
    volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id):
        print("RUNPOD_NETWORK_VOLUME_ID must be loaded from .env", file=sys.stderr)
        return 2
    if args.launch_id is not None and LAUNCH_PATTERN.fullmatch(args.launch_id) is None:
        print("LAUNCH_ID must look like launch-...", file=sys.stderr)
        return 2
    reader = Reader(args.project_root.resolve(), volume_id, transfer_timeout())
    launch_id, lifecycle = args.launch_id, None
    if launch_id is None:
        lifecycle = reader.read_json(workflow["lifecycle_key"])
        if lifecycle is None:
            print(f"No {label} lifecycle marker is on the volume", file=sys.stderr)
            return 4
        launch_id = str(lifecycle.get("launch_id", ""))
        if (
            lifecycle.get("kind") != workflow["lifecycle_kind"]
            or LAUNCH_PATTERN.fullmatch(launch_id) is None
        ):
            print(f"The {label} lifecycle marker is invalid", file=sys.stderr)
            return 4
        if lifecycle.get("state") == "preparing":
            print(f"{label} {launch_id} is still running; try again later", file=sys.stderr)
            return 3
        print(f"Latest {label}: {launch_id} (state={lifecycle.get('state')})")
    local = args.project_root / workflow["local_root"] / launch_id
    local.mkdir(parents=True, exist_ok=True)
    if lifecycle is not None:
        (local / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
    for name in ("combined.log", "status.json"):
        if not reader.download(
            f"logs/tmux/{session}/{launch_id}/{name}", local / "logs" / name, None
        ):
            print(f"Not found on the volume: {name}", file=sys.stderr)
    manifest_key = f"{remote_root}/{launch_id}/{MANIFEST_NAME}"
    if not reader.download(manifest_key, local / MANIFEST_NAME, None):
        print(f"The {label} is incomplete; read logs/combined.log for the failure", file=sys.stderr)
        return 4
    manifest = json.loads((local / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("kind") != workflow["manifest_kind"] or not isinstance(
        manifest.get("files"), dict
    ):
        print("The walk-forward manifest is unsupported", file=sys.stderr)
        return 4
    for name, entry in manifest["files"].items():
        if FILE_PATTERN.fullmatch(name) is None or not isinstance(entry.get("sha256"), str):
            print(f"The manifest names an unexpected file: {name!r}", file=sys.stderr)
            return 4
        if not reader.download(f"{remote_root}/{launch_id}/{name}", local / name, entry["sha256"]):
            print(f"Missing on the volume: {name}", file=sys.stderr)
            return 4
    models = manifest.get("models") or {}
    if not isinstance(models, dict):
        print("The walk-forward manifest lists models in an unsupported way", file=sys.stderr)
        return 4
    for name, entry in models.items():
        if MODEL_PATTERN.fullmatch(name) is None or not isinstance(entry.get("sha256"), str):
            print(f"The manifest names an unexpected model file: {name!r}", file=sys.stderr)
            return 4
        if not reader.download(f"{remote_root}/{launch_id}/{name}", local / name, entry["sha256"]):
            print(f"Missing on the volume: {name}", file=sys.stderr)
            return 4
    results = json.loads((local / "results.json").read_text(encoding="utf-8"))
    print(
        f"Verified {len(manifest['files'])} files and {len(models)} model files for "
        f"{launch_id} under {local}"
    )
    for line in summary_lines(results):
        print(line)
    if (local / LINEAGE_NAME).is_file():
        # The registry (committed with the repository) now points at this launch for its
        # model; it says which launch holds each fold's predictions and models.
        path, registry = update_registry(args.project_root / REGISTRY_DIR, local)
        print(
            f"Registry {path.relative_to(args.project_root)}: current launch "
            f"{registry['current']['launch']}, model folds {registry['model_folds']}"
        )
    else:
        print("No lineage.json: the launch predates the model rule; registry unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
