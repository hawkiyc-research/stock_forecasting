"""Find (and with ``--apply`` quarantine) wrong or dead research results on the volume.

Runs inside a RunPod Pod (or against any directory laid out like the volume) with the image's
Python; only the standard library is used. It inspects the research layer only:

* sample-universe indexes ``datasets/<dataset>/prepared/sample-universes/<key>/`` (built by
  the first ``cpu-features`` on a dataset from the data-cleaning policy);
* feature stores ``datasets/<dataset>/prepared/features/<identity>/`` and rank-label stores
  ``datasets/<dataset>/prepared/rank-labels/<identity>/`` (built by ``cpu-features``);
* feature-build launches ``diagnostics/features/launch-*/``;
* LightGBM launches ``diagnostics/lgbm/launch-*/`` and cross-stock launches
  ``diagnostics/xs/launch-*/``.

It never touches dataset preparation (``datasets/*/raw``, provider checkpoints, download
progress), GPU training runs (``savedModel``) or anything already under ``diagnostics/invalid/``.

A directory is INVALID (moved by ``--apply``) only with positive evidence that it is wrong or
that nothing can ever continue from it:

* a sample-universe index whose recorded data-cleaning policy is not the project's current
  ``configs/data_cleaning.json`` (for example one built before a closure such as 2026-07-10
  was listed) or whose bar-store manifest digest is not the dataset's current one: nothing
  reads it again, because every reader recomputes the key from the current policy; and every
  store built on such a universe (``identity.sample_universe``), together with the launches
  trained on those stores;
* a store whose manifest says ``ready`` but whose chip point-in-time mapping matched no symbol
  (``chip_coverage.symbols_with_margin`` or ``symbols_with_institutional`` is 0, the pre-fix
  unit bug: every chip column NaN), or whose manifest is not ``ready`` (the builders only ever
  write ``ready``);
* a store directory without its manifest — an interrupted build. Store builds are not resumed
  (a rerun of the same identity rebuilds the directory in place), so such a directory is dead
  weight; it is kept only while a feature-build job is in flight (``lifecycle/research/
  features.json`` state ``preparing``), because that job may be the one writing it;
* a feature-build launch without ``features-manifest.json``, kept only while the lifecycle
  marker names it as in flight;
* a finished launch (``results.json`` present) trained on an invalid store, or whose mean
  validation Rank IC is exactly 0 (constant predictions; a real fit never scores exactly 0);
* an unfinished launch (no ``results.json``) that the job's resume scan can no longer reach:
  the LightGBM and cross-stock jobs resume from the NEWEST launch that has ``partial/`` and no
  ``results.json``, so that one launch is a resume source (kept, whether it stopped by timeout,
  restart or crash) and every older unfinished launch is superseded. An unfinished launch
  without ``partial/`` has nothing to resume and is kept only while its lifecycle marker names
  it as in flight.

``--apply`` moves each INVALID directory to ``diagnostics/invalid/<same relative path>`` so
the resume scans and the download scripts no longer see it; ``--apply --delete`` removes them
instead, which is the only way to give the space back to the volume. Without ``--apply`` the
script only prints what it would move. All targets are checked before the first move; an
existing target aborts the run without moving anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

CHIP_SPECS = {"chip_tw_v1", "alpha_chip_tw_v1"}
STORE_KINDS = {
    "features": ("feature-store", "features.json"),
    "rank-labels": ("rank-label-store", "rank-labels.json"),
}
LAUNCH_KINDS = {
    "lgbm": ("lgbm-launch", "lifecycle/research/lgbm.json"),
    "xs": ("xs-launch", "lifecycle/research/xs-train.json"),
}
FEATURES_LIFECYCLE = "lifecycle/research/features.json"
FEATURES_LAUNCH_MANIFEST = "features-manifest.json"
UNIVERSE_MANIFEST = "universe.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_chip_identity(identity: dict | None) -> bool:
    if not isinstance(identity, dict):
        return False
    spec = identity.get("spec") if isinstance(identity.get("spec"), dict) else {}
    return "chip_data" in identity or spec.get("name") in CHIP_SPECS


def _lifecycle(volume: Path, key: str) -> tuple[str, str]:
    """(state, launch_id) of a research lifecycle marker; ("", "") when absent."""
    payload = _read_json(volume / key) or {}
    return str(payload.get("state") or ""), str(payload.get("launch_id") or "")


def _entry(kind: str, path: Path, *, spec=None, state=None, detail="", reason=None) -> dict:
    return {
        "kind": kind,
        "path": path,
        "spec": spec,
        "state": state,
        "detail": detail,
        "invalid": reason is not None,
        "reason": reason,
    }


def current_policy(project_root: Path = PROJECT_ROOT) -> dict | None:
    """The project's data-cleaning policy (None when it cannot be loaded here)."""
    try:
        tools = runpy.run_path(str(project_root / "src/stock_forecasting/data_policy.py"))
        return tools["load_data_policy"](project_root)
    except Exception:  # the audit must still run without the policy
        return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def universes(volume: Path, policy: dict | None) -> tuple[list[dict], dict[str, set[str]]]:
    """Every sample-universe index with its verdict, and per dataset the keys still current.

    A universe is superseded when its recorded policy differs from the current one or its
    bar-store manifest digest is not the dataset's: every reader recomputes the key from the
    current policy and bar store, so nothing reads a superseded index again.
    """
    features_state, _ = _lifecycle(volume, FEATURES_LIFECYCLE)
    out: list[dict] = []
    current: dict[str, set[str]] = {}
    for root in sorted(volume.glob("datasets/*/prepared/sample-universes/*")):
        if not root.is_dir():
            continue
        dataset = root.parents[2].name
        current.setdefault(dataset, set())
        payload = _read_json(root / UNIVERSE_MANIFEST)
        reason = None
        detail = ""
        if payload is None:
            state = "building" if features_state == "preparing" else "interrupted"
            if features_state != "preparing":
                reason = f"interrupted index build (no {UNIVERSE_MANIFEST})"
            detail = "no manifest"
        else:
            state = str(payload.get("state") or "")
            identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
            bar_manifest = root.parents[1] / "bar-store" / "bar-store.json"
            bar_digest = _sha256_file(bar_manifest)
            if policy is not None and identity.get("policy") != policy:
                reason = "built under a superseded data-cleaning policy"
            elif bar_digest is not None and identity.get("source_manifest_sha256") != bar_digest:
                reason = "built from a superseded bar store"
            elif state != "ready":
                reason = f"manifest state is {state!r}, not ready"
            audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
            detail = " ".join(
                f"{split}={entry.get('accepted')}" for split, entry in sorted(audit.items())
                if isinstance(entry, dict)
            )
        if reason is None:
            current[dataset].add(root.name)
        out.append(_entry("sample-universe", root, state=state, detail=detail, reason=reason))
    return out, current


def stores(volume: Path, current_universes: dict[str, set[str]] | None = None) -> list[dict]:
    """Every feature and rank-label store with its state, coverage and verdict.

    ``current_universes`` (from ``universes``) maps a dataset to the sample-universe keys that
    are still current; a store built on any other key is superseded.
    """
    features_state, _ = _lifecycle(volume, FEATURES_LIFECYCLE)
    out = []
    for layout, (kind, manifest_name) in STORE_KINDS.items():
        for root in sorted(volume.glob(f"datasets/*/prepared/{layout}/*")):
            if not root.is_dir():
                continue
            payload = _read_json(root / manifest_name)
            if payload is None:
                if (root / manifest_name).exists():
                    reason = f"{manifest_name} is unreadable"
                elif features_state == "preparing":
                    reason = None  # a feature-build job is in flight and may be writing it
                else:
                    reason = f"interrupted build (no {manifest_name}); nothing resumes from it"
                out.append(
                    _entry(
                        kind,
                        root,
                        state="building" if reason is None else "interrupted",
                        detail="no manifest",
                        reason=reason,
                    )
                )
                continue
            identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
            spec = identity.get("spec") if isinstance(identity.get("spec"), dict) else {}
            state = payload.get("state")
            reason = None
            detail = ""
            dataset = root.parents[2].name
            known = (current_universes or {}).get(dataset)
            universe = identity.get("sample_universe")
            if state != "ready":
                reason = f"manifest state is {state!r}, not ready"
            elif known is not None and universe is not None and universe not in known:
                reason = "built on a superseded sample universe"
                detail = f"universe={str(universe)[:12]}"
            elif _is_chip_identity(identity):
                coverage = payload.get("chip_coverage") or {}
                margin = int(coverage.get("symbols_with_margin", 0) or 0)
                institutional = int(coverage.get("symbols_with_institutional", 0) or 0)
                detail = f"margin={margin} institutional={institutional}"
                if margin == 0 or institutional == 0:
                    reason = "no symbol matched the chip tables (all chip columns NaN)"
            out.append(
                _entry(
                    kind,
                    root,
                    spec=spec.get("name") or identity.get("variant"),
                    state=state,
                    detail=detail,
                    reason=reason,
                )
            )
    return out


def feature_launches(volume: Path) -> list[dict]:
    """Every feature-build launch; one without its manifest is dead unless in flight."""
    state, launch_id = _lifecycle(volume, FEATURES_LIFECYCLE)
    out = []
    for launch in sorted(volume.glob("diagnostics/features/launch-*")):
        if not launch.is_dir():
            continue
        if (launch / FEATURES_LAUNCH_MANIFEST).is_file():
            out.append(_entry("features-launch", launch, state="finished"))
        elif state == "preparing" and launch.name == launch_id:
            out.append(_entry("features-launch", launch, state="running"))
        else:
            out.append(
                _entry(
                    "features-launch",
                    launch,
                    state="unfinished",
                    reason=f"no {FEATURES_LAUNCH_MANIFEST} and not in flight",
                )
            )
    return out


def launches(volume: Path, invalid_store_roots: set[Path]) -> list[dict]:
    """Every LightGBM and cross-stock launch with its verdict."""
    invalid_names = {p.name for p in invalid_store_roots}
    out = []
    for layout, (kind, lifecycle_key) in LAUNCH_KINDS.items():
        state, running_id = _lifecycle(volume, lifecycle_key)
        roots = sorted(p for p in volume.glob(f"diagnostics/{layout}/launch-*") if p.is_dir())
        # The job's resume scan: the newest launch with partial/ and without results.json.
        resume_source = None
        for root in roots:
            if (root / "partial").is_dir() and not (root / "results.json").is_file():
                resume_source = root
        for root in roots:
            manifest = _read_json(root / "walk-forward-manifest.json") or {}
            results = _read_json(root / "results.json")
            splits = manifest.get("splits", {}) if isinstance(manifest.get("splits"), dict) else {}
            feature_roots, spec_names = [], set()
            for split in splits.values():
                features = split.get("features", {}) if isinstance(split, dict) else {}
                if features.get("root"):
                    feature_roots.append(Path(features["root"]))
                identity = features.get("identity") or {}
                spec = identity.get("spec") if isinstance(identity.get("spec"), dict) else {}
                if spec.get("name"):
                    spec_names.add(str(spec["name"]))
            fits = (results or {}).get("fits", {}) if isinstance(results, dict) else {}
            rank_ic = fits.get("mean_validation_rank_ic") if isinstance(fits, dict) else None
            reason = None
            if any(root_.name in invalid_names for root_ in feature_roots):
                launch_state = "finished" if results is not None else "unfinished"
                reason = "trained on a wrong or superseded feature store"
            elif results is not None:
                launch_state = "finished"
                if rank_ic is not None and float(rank_ic) == 0.0:
                    reason = "mean validation Rank IC is exactly 0 (constant predictions)"
            elif root == resume_source:
                launch_state = "resume-source"
            elif state == "preparing" and root.name == running_id:
                launch_state = "running"
            elif (root / "partial").is_dir():
                launch_state = "superseded"
                reason = "unfinished launch superseded by a newer unfinished launch"
            else:
                launch_state = "unfinished"
                reason = "no partial/ to resume and not in flight"
            out.append(
                _entry(
                    kind,
                    root,
                    spec=", ".join(sorted(spec_names)) or None,
                    state=launch_state,
                    detail=f"mean_validation_rank_ic={rank_ic}",
                    reason=reason,
                )
            )
    return out


def quarantine(
    volume: Path, entries: list[dict], apply: bool, *, delete: bool = False
) -> list[tuple[Path, Path]]:
    """Move every invalid entry under ``diagnostics/invalid/`` (or delete it with ``delete``);
    returns (source, target) pairs."""
    moves = []
    for entry in entries:
        if entry["invalid"]:
            source = Path(entry["path"])
            moves.append((source, volume / "diagnostics" / "invalid" / source.relative_to(volume)))
    if apply and delete:
        for source, _target in moves:
            shutil.rmtree(source)
    elif apply:
        existing = [target for _, target in moves if target.exists()]
        if existing:
            raise FileExistsError(f"quarantine target already exists: {existing[0]}")
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    return moves


def _print_table(entries: list[dict]) -> None:
    for entry in entries:
        verdict = "INVALID" if entry["invalid"] else "ok"
        print(
            f"{verdict:8} {entry['kind']:16} {entry['spec'] or '-':18} {entry['state'] or '-':14} "
            f"{entry['detail']}  {entry['path']}"
            + (f"  <- {entry['reason']}" if entry["reason"] else "")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument(
        "--volume", type=Path, default=Path("/runpod-volume"), help="network volume root"
    )
    parser.add_argument(
        "--apply", action="store_true", help="move the wrong directories (default: only list)"
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="with --apply: delete the wrong directories instead of moving them (frees space)",
    )
    args = parser.parse_args(argv)
    if args.delete and not args.apply:
        parser.error("--delete needs --apply")
    volume = args.volume.resolve()
    if not volume.is_dir():
        print(f"volume root is not a directory: {volume}", file=sys.stderr)
        return 2
    universe_entries, current = universes(volume, current_policy())
    store_entries = stores(volume, current)
    invalid_roots = {Path(s["path"]) for s in store_entries if s["invalid"]}
    entries = (
        universe_entries + store_entries + feature_launches(volume)
        + launches(volume, invalid_roots)
    )
    if not entries:
        print("No stores or research launches found")
        return 0
    _print_table(entries)
    moves = quarantine(volume, entries, args.apply, delete=args.delete)
    if not moves:
        print("Nothing to quarantine")
        return 0
    for source, target in moves:
        if args.apply and args.delete:
            print(f"deleted {source}")
        else:
            print(("moved " if args.apply else "would move ") + f"{source} -> {target}")
    if not args.apply:
        print("Re-run with --apply to move them, or --apply --delete to remove them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
