"""Plan (and with ``--apply``) delete every artifact on the volume built from superseded data.

Runs inside a RunPod Pod against ``/runpod-volume`` (or any directory laid out like it) with
the image's Python; only the standard library is used. It implements the 2026-10-30 decision
to rebuild every dataset and model after the Taiwan adjusted-price corrections (protocol
§13.20 supplements 2 and 3, §13.23): everything derived from the old adjusted prices goes,
the raw provider responses and the traceable experiment records stay.

What is DELETED (``--apply``):

* in every dataset namespace ``datasets/<64-hex>/``: ``raw/``, ``prepared/`` (bar stores,
  sample universes, feature and label stores), ``provider-checkpoints/`` (materialized
  adjusted bars), ``download-progress.json`` and any loose ``*.parquet`` — the adjusted prices
  and everything computed from them;
* ``diagnostics/invalid/`` entirely (quarantined results);
* in ``diagnostics/lgbm``, ``diagnostics/xs`` and ``diagnostics/features`` launches: an
  unfinished launch (no ``results.json`` / ``features-manifest.json``) entirely, and in a
  finished one its ``partial/`` resume directory — except a launch carrying
  ``resumed_into.json`` (a later launch absorbed its folds: part of that result's record,
  kept whole) and except a launch another launch LINKS INTO (2026-11-02 rule: a launch
  extended by ``--extendLaunch`` keeps its ``partial/`` because the successor's partials are
  symbolic links to it; it carries ``superseded_by.json``). A superseded launch's leftover
  ``models/`` files are deleted (the successor holds every model P3 needs);
* ``forward/<date>/shared-cache/`` and every non-JSON file of ``forward/<date>/package/``
  (bars and caches of the forward pipeline);
* model weights and cached tensors anywhere outside the protected roots: ``*.pt``, ``*.pth``,
  ``*.safetensors``, ``*.bin``, ``*.ckpt``, ``*.npy``, ``*.npz``, ``*.joblib``, ``*.pkl``,
  ``*.pickle``, LightGBM ``*.lgb`` / ``model*.txt`` (GPU checkpoints under ``savedModel/``,
  baseline caches, cross-stock checkpoints) — EXCEPT the ``models/`` directory of a finished,
  not superseded walk-forward launch (``diagnostics/lgbm``, ``diagnostics/xs``): since
  2026-11-02 those are the saved P3-fold models of the current launch and part of the result.

What is KEPT (never touched):

* ``datasets/<namespace>/api-cache/`` — the raw provider responses, identical for every
  dataset revision and expensive to re-download — plus ``download-manifest.json`` and every
  other small JSON record of the namespace;
* ``datasets/tw-chip-data/`` and any namespace whose name is not a 64-hex digest;
* ``results.json``, manifests, ``predictions.parquet``, ``targets.parquet``, metrics, CSV,
  JSON and logs of every finished launch and diagnostic — the traceable experiment record;
* ``lifecycle/``, ``logs/``, ``evaluations/`` (markers that point at deleted paths are
  listed at the end so the next ``configure`` / ``cpu prepare`` can be judged).

Without ``--apply`` the script only prints the plan and writes it to
``diagnostics/volume-reset/<timestamp>/plan.json`` (sizes included). ``--apply`` needs
``--confirm RESET``; every target is checked to be a real path inside the volume (no
symlinks) before the first deletion, and the executed plan is written next to the plan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

NAMESPACE = re.compile(r"^[0-9a-f]{64}$")
LAUNCH_ROOTS = ("diagnostics/lgbm", "diagnostics/xs", "diagnostics/features")
FINISHED_MARKERS = ("results.json", "features-manifest.json")
RESUMED_INTO_MARKER = "resumed_into.json"
SUPERSEDED_BY_MARKER = "superseded_by.json"
MODELS_DIR = "models"
WALK_FORWARD_ROOTS = ("diagnostics/lgbm", "diagnostics/xs")
PROTECTED_DIRS = ("lifecycle", "logs", "evaluations")
WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".joblib",
    ".pkl",
    ".pickle",
    ".lgb",
)
WEIGHT_NAME = re.compile(r"^model.*\.txt$")
NAMESPACE_BULK = ("raw", "prepared", "provider-checkpoints")


def _size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            file = Path(root) / name
            if not file.is_symlink():
                with contextlib.suppress(OSError):
                    total += file.stat().st_size
    return total


def _entry(volume: Path, path: Path, group: str, reason: str) -> dict:
    return {
        "path": str(path.relative_to(volume)),
        "kind": "dir" if path.is_dir() and not path.is_symlink() else "file",
        "group": group,
        "reason": reason,
        "bytes": _size(path),
    }


def _is_weight(path: Path) -> bool:
    return path.suffix.lower() in WEIGHT_SUFFIXES or WEIGHT_NAME.match(path.name) is not None


def plan_namespaces(volume: Path) -> list[dict]:
    entries: list[dict] = []
    datasets = volume / "datasets"
    if not datasets.is_dir():
        return entries
    for namespace in sorted(datasets.iterdir()):
        if namespace.is_symlink() or not namespace.is_dir() or not NAMESPACE.match(namespace.name):
            continue  # tw-chip-data and anything hand-made stay
        for name in NAMESPACE_BULK:
            target = namespace / name
            if target.is_dir() and not target.is_symlink():
                entries.append(
                    _entry(volume, target, "dataset", f"{name}: adjusted prices or derived stores")
                )
        for child in sorted(namespace.iterdir()):
            if child.is_symlink():
                continue
            if child.name == "download-progress.json" and child.is_file():
                entries.append(
                    _entry(volume, child, "dataset", "resume state of a superseded download")
                )
            elif child.is_file() and child.suffix == ".parquet":
                entries.append(_entry(volume, child, "dataset", "loose bars"))
    return entries


def linked_launches(volume: Path) -> set[Path]:
    """Launch directories that another launch's ``partial/`` links into (2026-11-02 rule:
    an extended launch's folds are symbolic links to the launch that trained them)."""
    targets: set[Path] = set()
    for relative in WALK_FORWARD_ROOTS:
        root = volume / relative
        if not root.is_dir():
            continue
        for link in root.glob("launch-*/partial/*"):
            if not link.is_symlink():
                continue
            with contextlib.suppress(OSError):
                targets.add(Path(os.path.realpath(link)).parent.parent)
    return targets


def is_current_walk_forward_launch(volume: Path, path: Path) -> bool:
    """Whether ``path`` lies in a finished, not superseded launch under diagnostics/lgbm or
    diagnostics/xs: its ``models/`` holds the saved P3-fold models and is never swept."""
    try:
        parts = path.relative_to(volume).parts
    except ValueError:
        return False
    if len(parts) < 3 or "/".join(parts[:2]) not in WALK_FORWARD_ROOTS:
        return False
    launch = volume / parts[0] / parts[1] / parts[2]
    return (
        parts[2].startswith("launch-")
        and (launch / "results.json").is_file()
        and not (launch / SUPERSEDED_BY_MARKER).is_file()
    )


def plan_launches(volume: Path) -> list[dict]:
    entries: list[dict] = []
    linked = linked_launches(volume)
    for relative in LAUNCH_ROOTS:
        root = volume / relative
        if not root.is_dir():
            continue
        for launch in sorted(root.glob("launch-*")):
            if launch.is_symlink() or not launch.is_dir():
                continue
            finished = any((launch / marker).is_file() for marker in FINISHED_MARKERS)
            if (launch / RESUMED_INTO_MARKER).is_file():
                # A later launch absorbed its folds: it is part of that result's record
                # (per-fold records, logs) and is kept whole.
                continue
            if (launch / SUPERSEDED_BY_MARKER).is_file() or launch.resolve() in linked:
                # A later launch carries this one's folds through links into its partial/:
                # the launch stays (results, partial); only leftover model files go.
                models = launch / MODELS_DIR
                if models.is_dir() and not models.is_symlink():
                    entries.append(
                        _entry(volume, models, "weights", "model files of a superseded launch")
                    )
                continue
            if not finished:
                entries.append(
                    _entry(volume, launch, "launch", "unfinished launch: nothing traceable")
                )
                continue
            partial = launch / "partial"
            if partial.is_dir() and not partial.is_symlink():
                entries.append(
                    _entry(volume, partial, "launch", "resume directory of a finished launch")
                )
    invalid = volume / "diagnostics" / "invalid"
    if invalid.is_dir() and not invalid.is_symlink():
        entries.append(_entry(volume, invalid, "launch", "quarantined results"))
    return entries


def plan_forward(volume: Path) -> list[dict]:
    entries: list[dict] = []
    forward = volume / "forward"
    if not forward.is_dir():
        return entries
    for dated in sorted(forward.iterdir()):
        if dated.is_symlink() or not dated.is_dir():
            continue
        cache = dated / "shared-cache"
        if cache.is_dir() and not cache.is_symlink():
            entries.append(_entry(volume, cache, "forward", "forward-pipeline cache"))
        package = dated / "package"
        if package.is_dir() and not package.is_symlink():
            for root, dirs, files in os.walk(package):
                dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
                for name in files:
                    file = Path(root) / name
                    if file.suffix != ".json" and not file.is_symlink():
                        entries.append(_entry(volume, file, "forward", "forward package data"))
    return entries


def is_carried_launch_partial(volume: Path, path: Path, linked: set[Path]) -> bool:
    """Whether ``path`` is the ``partial/`` of a launch a later launch carries on (stamped
    ``superseded_by.json`` or linked into): its .npz files are that successor's folds."""
    if path.name != "partial":
        return False
    launch = path.parent
    try:
        parts = launch.relative_to(volume).parts
    except ValueError:
        return False
    if len(parts) != 3 or "/".join(parts[:2]) not in WALK_FORWARD_ROOTS:
        return False
    return (launch / SUPERSEDED_BY_MARKER).is_file() or launch.resolve() in linked


def plan_weights(volume: Path, planned: set[str]) -> list[dict]:
    """Weight and tensor files anywhere, except inside protected roots or already planned dirs."""

    entries: list[dict] = []
    linked = linked_launches(volume)
    skip_prefixes = tuple(f"{p}/" for p in planned) + tuple(f"{d}/" for d in PROTECTED_DIRS)
    for root, dirs, files in os.walk(volume):
        current = Path(root)
        relative = str(current.relative_to(volume))
        rel_prefix = "" if relative == "." else relative + "/"
        if (
            rel_prefix.startswith(skip_prefixes)
            or "/api-cache/" in f"/{rel_prefix}"
            or rel_prefix.startswith("datasets/tw-chip-data/")
            # A resume source a later launch absorbed: its partial/ (.npz) is that result's
            # per-fold record, kept whole.
            or (current / RESUMED_INTO_MARKER).is_file()
            # The saved P3-fold models of a current walk-forward launch (2026-11-02 rule).
            or (current.name == MODELS_DIR and is_current_walk_forward_launch(volume, current))
            # The partial/ a later launch links into (its folds live there).
            or is_carried_launch_partial(volume, current, linked)
        ):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not (current / d).is_symlink()]
        for name in files:
            file = current / name
            if file.is_symlink() or not _is_weight(file):
                continue
            entries.append(_entry(volume, file, "weights", "model weights or cached tensors"))
    return entries


def build_plan(volume: Path) -> list[dict]:
    entries = plan_namespaces(volume) + plan_launches(volume) + plan_forward(volume)
    planned_dirs = {e["path"] for e in entries if e["kind"] == "dir"}
    entries += plan_weights(volume, planned_dirs)
    # Drop anything nested inside a planned directory.
    dirs = sorted(planned_dirs)
    kept: list[dict] = []
    for entry in entries:
        nested = any(entry["path"] != d and entry["path"].startswith(d + "/") for d in dirs)
        if not nested:
            kept.append(entry)
    return kept


def dangling_lifecycle_markers(volume: Path, deleted: list[dict]) -> list[str]:
    """Lifecycle JSON files that mention a deleted path (informational)."""

    gone = {e["path"] for e in deleted}
    markers: list[str] = []
    lifecycle = volume / "lifecycle"
    if not lifecycle.is_dir():
        return markers
    for file in sorted(lifecycle.rglob("*.json")):
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(path in text for path in gone):
            markers.append(str(file.relative_to(volume)))
    return markers


def _check_target(volume: Path, entry: dict) -> Path:
    path = volume / entry["path"]
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink: {path}")
    resolved = path.resolve(strict=True)
    if volume.resolve() not in resolved.parents:
        raise RuntimeError(f"target escapes the volume: {path}")
    if not path.exists():
        raise RuntimeError(f"target vanished: {path}")
    return path


def apply_plan(volume: Path, entries: list[dict]) -> list[dict]:
    targets = [(_check_target(volume, entry), entry) for entry in entries]
    done: list[dict] = []
    for path, entry in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        done.append(entry)
    return done


def _format(entries: list[dict]) -> str:
    by_group: dict[str, list[dict]] = {}
    for entry in entries:
        by_group.setdefault(entry["group"], []).append(entry)
    lines: list[str] = []
    total = 0
    for group, items in by_group.items():
        size = sum(e["bytes"] for e in items)
        total += size
        lines.append(f"[{group}] {len(items)} targets, {size / 1e9:.2f} GB")
        for e in items[:40]:
            lines.append(
                f"  {e['kind']:4} {e['bytes'] / 1e6:10.1f} MB  {e['path']}  ({e['reason']})"
            )
        if len(items) > 40:
            lines.append(f"  ... {len(items) - 40} more")
    lines.append(f"TOTAL {len(entries)} targets, {total / 1e9:.2f} GB")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--volume", type=Path, default=Path("/runpod-volume"))
    parser.add_argument("--apply", action="store_true", help="delete the planned targets")
    parser.add_argument("--confirm", default="", help="must be RESET together with --apply")
    parser.add_argument(
        "--plan-root",
        type=Path,
        help="where to write plan.json (default diagnostics/volume-reset/<ts>/ on the volume)",
    )
    args = parser.parse_args(argv)
    volume = args.volume.resolve()
    if not volume.is_dir():
        print(f"volume not found: {volume}", file=sys.stderr)
        return 2
    entries = build_plan(volume)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    plan_root = args.plan_root or (volume / "diagnostics" / "volume-reset" / stamp)
    plan_root.mkdir(parents=True, exist_ok=True)
    (plan_root / "plan.json").write_text(
        json.dumps({"volume": str(volume), "created_at": stamp, "targets": entries}, indent=2),
        encoding="utf-8",
    )
    print(_format(entries))
    print(f"plan written to {plan_root / 'plan.json'}")
    if not args.apply:
        print("dry run: nothing deleted (add --apply --confirm RESET)")
        return 0
    if args.confirm != "RESET":
        print("--apply needs --confirm RESET", file=sys.stderr)
        return 2
    done = apply_plan(volume, entries)
    finished_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (plan_root / "applied.json").write_text(
        json.dumps({"deleted": done, "finished_at": finished_at}, indent=2), encoding="utf-8"
    )
    markers = dangling_lifecycle_markers(volume, done)
    print(f"deleted {len(done)} targets")
    if markers:
        print(
            "lifecycle markers that still mention deleted paths "
            "(re-run configure / cpu prepare before trusting them):"
        )
        for marker in markers:
            print(f"  {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
