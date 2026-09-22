#!/usr/bin/env python3
"""Verify the selected prepared dataset and pinned model cache as independent inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA_GATE = runpy.run_path(str(ROOT / "scripts/runpod_dataset_readiness.py"))
SELECTION, READINESS = DATA_GATE["SELECTION"], DATA_GATE["READINESS"]


def model_contract(config_path):
    # Use the same dependency-free scalar rules as the existing control-host gate.
    values, section = {}, None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            section = line.strip()
        elif indent == 2 and section in {"data:", "model:"} and ":" in line:
            key, value = line.strip().split(":", 1)
            if key in values:
                raise ValueError(f"Repeated model readiness scalar: {key}")
            values[key] = READINESS["_resolve_environment_default"](
                READINESS["_strip_yaml_scalar"](value)
            )
    if values.get("time_series_backend") != "kronos" or values.get("local_files_only") != "true":
        raise ValueError("Production training requires the offline Kronos backend")
    for name in (
        "kronos_source_revision", "time_series_model_revision", "time_series_tokenizer_revision"
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", values.get(name, "")):
            raise ValueError(f"Model readiness requires an immutable revision: {name}")
    for name in ("time_series_model_id", "time_series_tokenizer_id"):
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", values.get(name, "")):
            raise ValueError(f"Invalid model repository: {name}")
    if not re.fullmatch(r"[1-9][0-9]*", values.get("input_length", "")):
        raise ValueError("Model readiness requires a positive input length")
    return values


def verify_snapshot_file(reader, repository, revision, filename):
    """Check HF links without downloading weights to the control host."""
    base = "cache/huggingface/hub/models--" + repository.replace("/", "--") + "/"
    key = base + "snapshots/" + revision + "/" + filename
    if reader.volume is not None:
        snapshot = reader.volume / key
        resolved = snapshot.resolve(strict=True)
        repository_root = (reader.volume / base).resolve(strict=True)
        if not resolved.is_relative_to(repository_root) or not resolved.is_file():
            raise ValueError(f"Cached model file escapes its repository: {key}")
        size = resolved.stat().st_size
        # HF snapshots normally link to content-addressed blobs. Verify these in bounded chunks.
        if snapshot.is_symlink():
            if resolved.parent != repository_root / "blobs" or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name
            ):
                raise ValueError(f"Invalid cached model blob target: {key}")
            hasher = hashlib.sha256() if len(resolved.name) == 64 else hashlib.sha1()
            if len(resolved.name) == 40:
                hasher.update(f"blob {size}\0".encode())
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != resolved.name:
                raise ValueError(f"Cached model blob checksum mismatch: {key}")
    else:
        # HEAD resolves RunPod's HF symlinks; list-objects sizes describe the links,
        # not the target weights. Never download model weights to the control host.
        size = reader.size(key)
    if size < (8 if filename == "model.safetensors" else 2):
        raise ValueError(f"Cached model file is empty or truncated: {key}")
    return {"key": key, "size_bytes": size}


def verify_models(config_path, reader, *, workers):
    values = model_contract(config_path)
    payload = json.loads(reader.read("cache/hf-models.json"))
    expected = {
        values["time_series_model_id"]: values["time_series_model_revision"],
        values["time_series_tokenizer_id"]: values["time_series_tokenizer_revision"],
    }
    if (
        payload.get("local_files_only_verified") is not True
        or payload.get("kronos_source_revision") != values["kronos_source_revision"]
        or payload.get("repository_revisions") != expected
        or set(payload.get("repositories", {})) != set(expected)
    ):
        raise ValueError("Offline model cache differs from selected model repositories/revisions")
    smoke = payload.get("time_series_smoke_test", {})
    for key, expected_value in {
        "passed": True, "local_files_only": True, "backend": "kronos",
        "input_bars": int(values["input_length"]),
        **{key: values[key] for key in (
            "kronos_source_revision", "time_series_model_revision", "time_series_tokenizer_revision"
        )},
    }.items():
        if smoke.get(key) != expected_value:
            raise ValueError(f"Offline model smoke-test mismatch: {key}")
    requests = []
    for repository, revision in expected.items():
        relative = (
            "cache/huggingface/hub/models--" + repository.replace("/", "--")
            + "/snapshots/" + revision
        )
        # The persisted cache uses the canonical Pod mount, not a control-host path.
        if payload["repositories"][repository] != str(PurePosixPath("/runpod-volume") / relative):
            raise ValueError("Model snapshot is outside its pinned repository/revision")
        requests.extend(
            (repository, revision, name) for name in ("config.json", "model.safetensors")
        )
    files = list(DATA_GATE["bounded_map"](
        lambda args: verify_snapshot_file(reader, *args), requests, workers
    ))
    return {"state": "ready", "repository_revisions": expected, "verified_files": len(files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--selection", default=os.environ.get("RUNPOD_SELECTION_FILE"))
    parser.add_argument("--network-volume-root", type=Path)
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    _, selected = SELECTION["_resolve_selection_path"](project, args.selection)
    config_path = project / selected["stage"]["config_path"]
    contract = READINESS["_load_stage_contract"](config_path)
    if contract.dataset_profile != selected["dataset_request"]["profile"]:
        raise ValueError("Training config differs from the selected dataset profile")
    if contract.training_stage != selected["stage"]["name"]:
        raise ValueError("Training config differs from the selected stage")
    reader = DATA_GATE["ArtifactReader"](
        project, volume=args.network_volume_root, bucket=os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )
    workers = DATA_GATE["worker_count"]()
    data = DATA_GATE["verify_data"](
        project, selected, reader, workers=workers, training_artifacts=True
    )
    print(json.dumps({"dataset": data}, sort_keys=True), flush=True)
    models = verify_models(config_path, reader, workers=workers)
    print(json.dumps({"models": models}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError, RuntimeError, KeyError, TypeError, OSError, subprocess.SubprocessError
    ) as error:
        print(f"Training readiness failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
