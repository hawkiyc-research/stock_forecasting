#!/usr/bin/env python3
"""Shared, stdlib-only contract for one Pod owning two independent signal tasks."""

from __future__ import annotations

import copy
import json
import os
import re
import runpy
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_EXPERIMENTS = ("a-signal64", "b-signal64")
TASKS = ("alpha_point", "absolute_direction")
VARIANTS = {"alpha_point": "alpha-point64", "absolute_direction": "absolute-direction64"}
SUFFIXES = {"alpha_point": "alpha", "absolute_direction": "direction"}


@lru_cache(maxsize=8)
def selection_api(root: Path = ROOT) -> dict:
    return runpy.run_path(str(root / "scripts/runpod_selection.py"))


def safe_run_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value) or "--" in value:
        raise ValueError("Paired experiment requires a canonical run ID")
    return value


def manifest_key(run_id: str) -> str:
    return f"lifecycle/runs/{safe_run_id(run_id)}/pair.json"


def validate_manifest(payload: dict, run_id: str | None = None) -> dict:
    """Validate task/data/seed ownership without changing numerical identities."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("kind") != "paired-training":
        raise ValueError("Unsupported paired experiment manifest")
    parent = safe_run_id(payload.get("run_id"))
    if run_id is not None and parent != safe_run_id(run_id):
        raise ValueError("Paired experiment run ID mismatch")
    experiment = payload.get("experiment")
    if experiment not in PAIR_EXPERIMENTS or payload.get("workflow") not in ("train", "baseline"):
        raise ValueError("Unsupported paired experiment or workflow")
    seed = payload.get("seed")
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError("Paired experiment seed must be an unsigned 32-bit integer")
    if not isinstance(payload.get("created_at"), str):
        raise ValueError("Paired experiment creation time is missing")
    if datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Paired experiment creation time must include a timezone")
    children = payload.get("children")
    if not isinstance(children, dict) or set(children) != set(TASKS):
        raise ValueError("Paired experiment must contain both signal tasks exactly once")
    requests = []
    for task in TASKS:
        child = children[task]
        if not isinstance(child, dict) or child.get("task") != task:
            raise ValueError("Paired child task mismatch")
        if child.get("run_id") != f"{parent}-{SUFFIXES[task]}":
            raise ValueError("Paired child is not owned by its parent run")
        safe_run_id(child["run_id"])
        expected = f"{experiment[0]}-{VARIANTS[task]}"
        selected = child.get("selection")
        if not isinstance(selected, dict):
            raise ValueError("Paired child selection is missing")
        selection_api()["_validate_selection"](selected)
        if child.get("experiment") != expected or selected["stage"].get("experiment") != expected:
            raise ValueError("Paired child experiment mismatch")
        if child.get("config_path") != selected["stage"]["config_path"]:
            raise ValueError("Paired child config mismatch")
        if child.get("selection_path") != f"lifecycle/selections/{selected['selection_id']}.json":
            raise ValueError("Paired child selection path mismatch")
        if selected["stage"].get("training_seed") != seed:
            raise ValueError("Paired tasks must use the same training seed")
        if selected.get("baseline_task", {}).get("prediction_task") != task:
            raise ValueError("Paired task baseline contract mismatch")
        requests.append(selected["dataset_request"])
    if requests[0] != requests[1]:
        raise ValueError("Paired tasks must use the same data group and evaluation contract")
    return payload


def build_plan(root: Path, original_selection: dict, experiment: str, seed: int | None = None,
               workflow: str = "train", run_id: str | None = None) -> dict:
    root = Path(root).resolve()
    if experiment not in PAIR_EXPERIMENTS:
        raise ValueError("Unknown paired experiment: " + str(experiment))
    api = selection_api(root)
    if seed is None:
        seed = original_selection.get("stage", {}).get("training_seed")
    if seed is None:
        # The current preset default is explicit and must agree across both tasks.
        values = []
        for variant in VARIANTS.values():
            source = (root / f"configs/experiments/{experiment[0]}_{variant.replace('-', '_')}.yaml").read_text()
            training = source.split("\ntraining:\n", 1)[1]
            found = re.search(r"(?m)^  seed: (\d+)\s*$", training)
            if found is None:
                raise ValueError("Paired preset has no explicit training seed")
            values.append(int(found.group(1)))
        if values[0] != values[1]:
            raise ValueError("Paired preset default seeds disagree; specify --seed")
        seed = values[0]
    api["_validate_training_seed"](seed)
    now = datetime.now(timezone.utc)
    prefix = "baseline-run" if workflow == "baseline" else "run"
    parent = run_id or f"{prefix}-{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:16]}"
    safe_run_id(parent)
    children = {}
    for task in TASKS:
        name = f"{experiment[0]}-{VARIANTS[task]}"
        selected = api["_with_experiment"](copy.deepcopy(original_selection), name, root)
        selected = api["_with_training_seed"](selected, seed, root)
        children[task] = {
            "task": task, "run_id": f"{parent}-{SUFFIXES[task]}", "experiment": name,
            "config_path": selected["stage"]["config_path"],
            "selection_path": f"lifecycle/selections/{selected['selection_id']}.json",
            "selection": selected,
        }
    return validate_manifest({
        "schema_version": 1, "kind": "paired-training", "workflow": workflow,
        "run_id": parent, "experiment": experiment, "seed": seed,
        "created_at": now.isoformat(), "children": children,
    })


def write_json(path: Path, payload: dict) -> None:
    """Atomically publish metadata without mutating any dataset or model cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def store_plan(root: Path, plan: dict) -> Path:
    validate_manifest(plan)
    api = selection_api(root)
    for child in plan["children"].values():
        api["_store_selection"](root, child["selection"])
    path = root / ".runpod/paired" / f"{plan['run_id']}.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Existing paired run manifest must not be overwritten")
    else:
        write_json(path, plan)
    return path


def load_manifest(path: Path, run_id: str | None = None) -> dict:
    if path.is_symlink():
        raise ValueError("Paired manifest must not be a symlink")
    return validate_manifest(json.loads(path.read_text()), run_id=run_id)
