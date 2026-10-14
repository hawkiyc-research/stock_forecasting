#!/usr/bin/env python3
"""Record capacity-request outcomes locally; never publish failed attempts remotely."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def create(root, payload):
    # Store an explicit allowlist, never the full request/response environment.
    directory = root / ".runpod/launch-attempts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"attempt-{uuid.uuid4().hex}.json"
    env = payload.get("env", {})
    record = {
        "schema_version": 1, "state": "requesting",
        "run_id": env.get("WANDB_RUN_ID", ""), "pod_name": payload.get("name"),
        "gpu_id": payload.get("gpuId"), "volume_id": payload.get("networkVolumeId"),
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    save(path, record)
    try:
        result = subprocess.run(
            ["bash", str(root / "scripts/runpodctl_project.sh"), "pod", "create-gpu"],
            input=json.dumps(payload), text=True, capture_output=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError):
        record.update(state="unknown", finished_at=datetime.now(timezone.utc).isoformat())
        save(path, record)
        raise
    if result.returncode:
        try:
            error = json.loads(result.stderr)
        except (ValueError, TypeError):
            error = {}
        status = error.get("status") if isinstance(error, dict) else None
        record["state"] = "rejected" if isinstance(status, int) and 400 <= status < 500 else "unknown"
        record["http_status"] = status
    else:
        try:
            response = json.loads(result.stdout)
        except ValueError:
            response = {}
        pod_id = response.get("id") if isinstance(response, dict) else None
        if isinstance(pod_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            record.update(state="created", pod_id=pod_id)
        else:
            record["state"] = "unknown"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        save(path, record)
    except OSError:
        # A full local disk must not hide a created Pod ID from guard startup.
        print("Warning: could not update local launch receipt; inspect Pod inventory before retrying.", file=sys.stderr)
    # Preserve all provider diagnostics; the existing wrapper handles redaction.
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Pod creation payload must be an object")
    return create(args.project_root.resolve(), payload)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Pod creation outcome is uncertain; inspect Pod inventory before retrying: {error}", file=sys.stderr)
        raise SystemExit(2) from error
