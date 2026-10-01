"""Coordinate a local runtime cutoff with durable remote section boundaries."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
SAFE_SECTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


class RuntimeStopRequested(Exception):
    """The current section was saved after a matching local stop request."""


def stop_at_saved_boundary(
    *,
    section_kind: str,
    section_id: str,
    artifact: Path,
) -> None:
    """Acknowledge a matching request only after the section artifact is durable."""

    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    run_id = os.environ.get("RUNPOD_RUN_KEY", "")
    if not pod_id or not run_id:
        return
    if SAFE_ID.fullmatch(pod_id) is None or SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Runtime stop identity is invalid")
    if "--" in run_id or SAFE_SECTION.fullmatch(section_id) is None:
        raise ValueError("Runtime stop section identity is invalid")
    if section_kind not in {"training-checkpoint", "validation-model", "validation-seed"}:
        raise ValueError("Runtime stop section kind is unsupported")
    raw_volume_root = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    if not raw_volume_root.is_absolute() or raw_volume_root == Path("/workspace"):
        raise ValueError("Runtime stop requires a persistent network volume")
    volume_root = raw_volume_root.resolve()
    signal_dir = volume_root / "lifecycle" / "runs" / run_id / "pods" / pod_id
    request_path = signal_dir / "stop-request.json"
    try:
        if request_path.is_symlink() or request_path.stat().st_size > 8192:
            return
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return
    if not isinstance(request, dict) or request != {
        "schema_version": 1,
        "kind": "runtime-stop-request",
        "pod_id": pod_id,
        "run_id": run_id,
    }:
        return
    artifact = artifact.resolve(strict=True)
    if section_kind == "training-checkpoint":
        expected = volume_root / "savedModel" / run_id / section_id
        state_path = artifact / "trainer-state.json"
        if artifact != expected or not state_path.is_file():
            raise ValueError("Runtime stop checkpoint is not committed")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("global_step") != int(section_id[11:]):
            raise ValueError("Runtime stop checkpoint step is inconsistent")
    else:
        expected = volume_root / "evaluations" / run_id / "validation-benchmark.json"
        if artifact != expected or not artifact.is_file() or artifact.stat().st_size < 1:
            raise ValueError("Runtime stop validation result is not committed")
        result = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("run_id") != run_id:
            raise ValueError("Runtime stop validation result has the wrong run")
        models = result.get("models")
        if not isinstance(models, dict):
            raise ValueError("Runtime stop validation models are missing")
        if section_kind == "validation-model":
            section_saved = isinstance(models.get(section_id), dict) and (
                models[section_id].get("state") == "complete"
            )
        else:
            model_name, _, seed = section_id.rpartition(".")
            model = models.get(model_name)
            section_saved = isinstance(model, dict) and isinstance(
                model.get("seed_results"), dict
            ) and seed in model["seed_results"]
        if not section_saved:
            raise ValueError("Runtime stop validation section is incomplete")
    acknowledgement = {
        "schema_version": 1,
        "kind": "runtime-stop-ready",
        "pod_id": pod_id,
        "run_id": run_id,
        "section_kind": section_kind,
        "section_id": section_id,
        "artifact": str(artifact),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    temporary = signal_dir / f"stop-ready.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(acknowledgement, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, signal_dir / "stop-ready.json")
    raise RuntimeStopRequested(f"Saved {section_kind} {section_id} after local cutoff")
