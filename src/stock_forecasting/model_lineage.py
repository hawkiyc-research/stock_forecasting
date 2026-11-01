"""Saved models, launch lineage and the model registry of the walk-forward trainers.

The 2026-11-02 rule: every walk-forward launch (LightGBM and cross-stock) persists the models
of the folds the out-of-sample period P3 needs (``models/fold-XX/``), a later launch of the same
model extends an earlier one instead of retraining it (the earlier folds' partials are linked,
not copied; P3 folds get their models), the earlier launch is stamped ``superseded_by.json`` and
loses its model files once the new ones are verified, and every launch writes ``lineage.json``
saying where each fold's predictions and models live. The local download step turns the lineage
into ``registry/<kind>/<key>.json``: which launch is the current one, which launch holds each
fold, which folds have models.

Standard library only: the download scripts run with the control machine's ``python3``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUPERSEDED_BY_NAME = "superseded_by.json"
LINEAGE_NAME = "lineage.json"
MODELS_DIR = "models"
MODEL_MANIFEST_NAME = "model-manifest.json"
LINEAGE_SCHEMA = "walk-forward-lineage-v1"
REGISTRY_SCHEMA = "model-registry-v1"
REGISTRY_DIR = "registry"
TRAINING_DEFINITION = ("train_start", "validation_start", "train_end")
PREDICTION_DEFINITION = ("predict_start", "predict_end")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(target)
    return target


def read_json(path: str | Path) -> dict[str, Any] | None:
    file = Path(path)
    if not file.is_file():
        return None
    payload = json.loads(file.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _field(fold: Any, name: str) -> str:
    value = fold[name] if isinstance(fold, Mapping) else getattr(fold, name)
    return str(value)


def training_key(fold: Any) -> tuple[str, ...]:
    """What decides a fold's model: its training window (the schedule's dates)."""
    return tuple(_field(fold, name) for name in TRAINING_DEFINITION)


def prediction_key(fold: Any) -> tuple[str, ...]:
    return tuple(_field(fold, name) for name in PREDICTION_DEFINITION)


def overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    """Whether the half-open date ranges [start, end) intersect (ISO date strings)."""
    return start_a < end_b and start_b < end_a


def model_fold_indexes(folds: Iterable[Any], schedule: Mapping[str, Any]) -> set[int]:
    """Folds whose models a launch keeps: those whose prediction window intersects the
    schedule's P3 (the out-of-sample period), or the last fold when no P3 is defined."""
    folds = list(folds)
    if not folds:
        return set()
    p3 = (schedule.get("periods") or {}).get("P3")
    if not p3:
        return {int(_field(folds[-1], "index"))}
    start, end = str(p3[0]), str(p3[1])
    return {
        int(_field(fold, "index"))
        for fold in folds
        if overlaps(_field(fold, "predict_start"), _field(fold, "predict_end"), start, end)
    }


def select_model_fold(folds: Iterable[Any], model_folds: Iterable[int], date: str) -> int:
    """The saved-model fold that scores ``date`` (ISO): the model fold whose prediction window
    holds the date, else the latest model fold whose window started before it (a date after
    the data end is scored by the newest model). A date before every model fold's window has
    no model that was trained without seeing it, so it is refused."""
    kept = sorted(int(index) for index in model_folds)
    by_index = {int(_field(fold, "index")): fold for fold in folds}
    if not kept:
        raise ValueError("The launch kept no models")
    latest = None
    for index in kept:
        fold = by_index[index]
        start, end = _field(fold, "predict_start"), _field(fold, "predict_end")
        if start <= date < end:
            return index
        if start <= date:
            latest = index
    if latest is None:
        raise ValueError(
            f"{date} precedes every saved model's prediction window (first model fold "
            f"{kept[0]} starts {_field(by_index[kept[0]], 'predict_start')})"
        )
    return latest


def fold_models_dir(root: str | Path, fold_index: int) -> Path:
    return Path(root) / MODELS_DIR / f"fold-{int(fold_index):02d}"


def write_model_manifest(
    fold_dir: str | Path,
    *,
    kind: str,
    fold: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """Record the model files of one fold with their SHA-256 (``files`` maps a file name in
    ``fold_dir`` to its details: horizon, seed, best iteration, ...)."""
    fold_dir = Path(fold_dir)
    payload = {
        "schema": "walk-forward-models-v1",
        "kind": kind,
        "fold": dict(fold),
        "files": {
            name: {
                **dict(details),
                "sha256": sha256_file(fold_dir / name),
                "bytes": (fold_dir / name).stat().st_size,
            }
            for name, details in files.items()
        },
        **extra,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(fold_dir / MODEL_MANIFEST_NAME, payload)
    return payload


def read_model_manifest(fold_dir: str | Path, *, verify: bool = True) -> dict[str, Any] | None:
    """The fold's model manifest when every listed file is present (and, with ``verify``,
    matches its SHA-256); None otherwise."""
    fold_dir = Path(fold_dir)
    manifest = read_json(fold_dir / MODEL_MANIFEST_NAME)
    if manifest is None or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        return None
    for name, details in manifest["files"].items():
        file = fold_dir / name
        if "/" in name or not file.is_file():
            return None
        if verify and sha256_file(file) != details.get("sha256"):
            return None
    return manifest


def copy_fold_models(source_dir: str | Path, target_dir: str | Path) -> dict[str, Any]:
    """Copy a verified fold model directory (files and manifest); returns the manifest."""
    manifest = read_model_manifest(source_dir)
    if manifest is None:
        raise ValueError(f"No verified model manifest in {source_dir}")
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        shutil.copyfile(source_dir / name, target_dir / name)
    shutil.copyfile(source_dir / MODEL_MANIFEST_NAME, target_dir / MODEL_MANIFEST_NAME)
    if read_model_manifest(target_dir) is None:
        raise RuntimeError(f"Copied models failed verification: {target_dir}")
    return manifest


def launch_models(root: str | Path) -> dict[int, dict[str, Any]]:
    """Verified model manifests of a launch by fold index."""
    models_root = Path(root) / MODELS_DIR
    found: dict[int, dict[str, Any]] = {}
    if not models_root.is_dir():
        return found
    for fold_dir in sorted(models_root.glob("fold-*")):
        if not fold_dir.is_dir() or fold_dir.is_symlink():
            continue
        manifest = read_model_manifest(fold_dir)
        if manifest is not None:
            found[int(fold_dir.name.split("-")[1])] = manifest
    return found


def real_file(path: str | Path) -> Path:
    """The file a (possibly linked) partial resolves to."""
    return Path(os.path.realpath(Path(path)))


def owning_launch(path: str | Path) -> str:
    """Name of the launch directory that holds the real file (``<launch>/partial/<file>``)."""
    return real_file(path).parent.parent.name


def link_file(target: str | Path, source: str | Path) -> str:
    """Make ``target`` a relative symbolic link to the real file behind ``source`` (so a chain
    of extensions always points at the launch that trained the fold); the bytes are copied
    when the file system refuses links. Returns ``"link"`` or ``"copy"``."""
    target, resolved = Path(target), real_file(source)
    if not resolved.is_file():
        raise FileNotFoundError(f"Link source is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        relative = os.path.relpath(resolved, target.parent)
        os.symlink(relative, target)
        if not target.is_file():
            target.unlink()
            raise OSError("dangling link")
        return "link"
    except OSError:
        shutil.copyfile(resolved, target)
        return "copy"


def stamp_superseded(
    source: str | Path, *, superseded_by: str | Path, folds: list[int], **details: Any
) -> dict[str, Any]:
    """Record in ``source`` that ``superseded_by`` is now the current launch of this model and
    which of the source's folds it carries on (by link or by reuse of their models)."""
    payload = {
        "superseded_by": Path(superseded_by).name,
        "folds": sorted(int(index) for index in folds),
        **details,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(Path(source) / SUPERSEDED_BY_NAME, payload)
    return payload


def prune_models(source: str | Path) -> list[str]:
    """Delete the model files of a superseded launch (its P3 moved on: the current launch
    holds every model P3 needs). Returns the removed paths relative to the launch."""
    models_root = Path(source) / MODELS_DIR
    if models_root.is_symlink() or not models_root.is_dir():
        return []
    removed: list[str] = []
    for dirpath, dirnames, filenames in os.walk(models_root):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for name in filenames:
            removed.append(str((Path(dirpath) / name).relative_to(Path(source))))
    shutil.rmtree(models_root)
    return sorted(removed)


def write_lineage(root: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    record = {"schema": LINEAGE_SCHEMA, **payload}
    record.setdefault("written_at", datetime.now(UTC).isoformat(timespec="seconds"))
    atomic_write_json(Path(root) / LINEAGE_NAME, record)
    return record


def read_lineage(root: str | Path) -> dict[str, Any] | None:
    payload = read_json(Path(root) / LINEAGE_NAME)
    return payload if payload is not None and payload.get("schema") == LINEAGE_SCHEMA else None


def registry_key(kind: str, model: str, label_variant: str | None, feature_spec: str | None) -> str:
    """One registry file per (model, labels, feature spec): the LightGBM model name already
    carries its label variant; the cross-stock experiment name does not."""
    parts = [model]
    if kind != "lgbm":
        parts.append(label_variant or "standard")
    parts.append(feature_spec or "default")
    return "__".join(parts)


def registry_path(registry_root: str | Path, kind: str, key: str) -> Path:
    return Path(registry_root) / kind / f"{key}.json"


def update_registry(
    registry_root: str | Path, launch_dir: str | Path, *, downloaded_at: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Make the downloaded launch the current one of its model in the registry.

    The registry entry lists, per fold, the launch that holds the real prediction partial and
    (for P3 folds) the launch that holds the models — the current launch for folds it trained
    or predicted, the historical launch a linked fold points at. Earlier current launches move
    to ``history``.
    """
    launch_dir = Path(launch_dir)
    lineage = read_lineage(launch_dir)
    if lineage is None:
        raise ValueError(f"{launch_dir} has no {LINEAGE_NAME}; it predates the model rule")
    kind, key = str(lineage["kind"]), str(lineage["key"])
    path = registry_path(registry_root, kind, key)
    existing = read_json(path) or {}
    history = list(existing.get("history") or [])
    previous = existing.get("current") or {}
    if previous.get("launch") and previous["launch"] != lineage["launch"]:
        history.append(
            {
                "launch": previous["launch"],
                "schedule": previous.get("schedule"),
                "data_end": previous.get("data_end"),
                "p3": previous.get("p3"),
                "superseded_by": lineage["launch"],
                "superseded_at": downloaded_at or datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
    folds = {}
    for index, entry in sorted(lineage["folds"].items(), key=lambda item: int(item[0])):
        folds[str(index)] = {
            "name": entry.get("name"),
            "predict": entry.get("predict"),
            "predictions_launch": entry.get("predictions_launch", lineage["launch"]),
            "how": entry.get("how"),
            "models": entry.get("models"),
        }
    payload = {
        "schema": REGISTRY_SCHEMA,
        "kind": kind,
        "key": key,
        "model": lineage.get("model"),
        "label_variant": lineage.get("label_variant"),
        "feature_spec": lineage.get("feature_spec"),
        "current": {
            "launch": lineage["launch"],
            "schedule": lineage.get("schedule"),
            "data_end": lineage.get("data_end"),
            "p3": lineage.get("p3"),
            "extended_from": lineage.get("extended_from"),
            "downloaded_at": downloaded_at or datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "model_folds": sorted(
            int(index) for index, entry in lineage["folds"].items() if entry.get("models")
        ),
        "folds": folds,
        "history": history,
    }
    atomic_write_json(path, payload)
    return path, payload


def lineage_payload(
    *,
    launch: str,
    kind: str,
    key: str,
    model: str,
    label_variant: str | None,
    feature_spec: str | None,
    schedule: Mapping[str, Any],
    folds: Iterable[Any],
    fold_sources: Mapping[int, Mapping[str, Any]],
    model_folds: Iterable[int],
    extended_from: str | None,
    **extra: Any,
) -> dict[str, Any]:
    """The lineage record of a finished launch. ``fold_sources`` maps a fold index to
    ``{"how": trained|linked|predicted|resumed, "predictions_launch": <launch>, ...}``."""
    periods = schedule.get("periods") or {}
    model_folds = set(int(index) for index in model_folds)
    entries = {}
    for fold in folds:
        index = int(_field(fold, "index"))
        source = dict(fold_sources.get(index, {}))
        source.setdefault("how", "trained")
        source.setdefault("predictions_launch", launch)
        entries[str(index)] = {
            "name": _field(fold, "name"),
            "training": list(training_key(fold)),
            "predict": list(prediction_key(fold)),
            **source,
            "models": f"{MODELS_DIR}/fold-{index:02d}" if index in model_folds else None,
        }
    return {
        "launch": launch,
        "kind": kind,
        "key": key,
        "model": model,
        "label_variant": label_variant,
        "feature_spec": feature_spec,
        "schedule": schedule.get("name"),
        "data_end": schedule.get("data_end"),
        "p3": list(periods["P3"]) if periods.get("P3") else None,
        "extended_from": extended_from,
        "folds": entries,
        **extra,
    }
