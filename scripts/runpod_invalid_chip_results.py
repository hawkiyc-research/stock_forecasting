"""Find (and with ``--apply`` quarantine) chip-feature results on the volume that are wrong.

Runs inside a RunPod Pod (or against any directory laid out like the volume) with the image's
Python; only the standard library is used. Two kinds of results are inspected:

* chip feature stores ``datasets/<dataset>/prepared/features/<identity>/features.json`` whose
  identity carries ``chip_data`` — a store is wrong when its point-in-time mapping matched no
  symbol at all (``chip_coverage.symbols_with_margin`` or ``symbols_with_institutional`` is 0),
  which is what the pre-fix unit bug produced (every column NaN);
* LightGBM launches ``diagnostics/lgbm/launch-*/`` trained on a chip spec — a launch is wrong when
  any of its splits read a wrong store, or when its ``results.json`` reports a mean validation
  Rank IC of exactly 0 (constant predictions; a real fit never scores exactly 0 on 588 fits).

Nothing is deleted: ``--apply`` moves each wrong directory to ``diagnostics/invalid/<same relative
path>`` so the LightGBM job's resume scan and ``download-lgbm`` no longer see it. Without
``--apply`` the script only prints what it would move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHIP_SPECS = {"chip_tw_v1", "alpha_chip_tw_v1"}


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


def chip_stores(volume: Path) -> list[dict]:
    """Every chip feature store with its state, coverage and verdict."""
    out = []
    for manifest in sorted(volume.glob("datasets/*/prepared/features/*/features.json")):
        payload = _read_json(manifest)
        if payload is None or not _is_chip_identity(payload.get("identity")):
            continue
        coverage = payload.get("chip_coverage") or {}
        margin = int(coverage.get("symbols_with_margin", 0) or 0)
        institutional = int(coverage.get("symbols_with_institutional", 0) or 0)
        spec = payload["identity"].get("spec", {})
        reason = None
        if payload.get("state") == "ready" and (margin == 0 or institutional == 0):
            reason = "no symbol matched the chip tables (all chip columns NaN)"
        out.append(
            {
                "kind": "feature-store",
                "path": manifest.parent,
                "spec": spec.get("name"),
                "state": payload.get("state"),
                "symbols_with_margin": margin,
                "symbols_with_institutional": institutional,
                "invalid": reason is not None,
                "reason": reason,
            }
        )
    return out


def chip_launches(volume: Path, invalid_store_roots: set[Path]) -> list[dict]:
    """Every LightGBM launch trained on a chip spec with its verdict."""
    out = []
    for launch in sorted(volume.glob("diagnostics/lgbm/launch-*")):
        if not launch.is_dir():
            continue
        manifest = _read_json(launch / "walk-forward-manifest.json")
        results = _read_json(launch / "results.json")
        splits = (manifest or {}).get("splits", {})
        feature_roots = []
        chip = False
        for split in splits.values():
            features = split.get("features", {}) if isinstance(split, dict) else {}
            root = features.get("root")
            if root:
                feature_roots.append(Path(root))
            chip = chip or _is_chip_identity(features.get("identity"))
        if not chip and manifest is not None:
            continue
        if manifest is None:
            # An interrupted launch has no manifest yet: report it, decide nothing.
            out.append(
                {
                    "kind": "lgbm-launch",
                    "path": launch,
                    "spec": None,
                    "state": "incomplete",
                    "mean_validation_rank_ic": None,
                    "invalid": False,
                    "reason": None,
                }
            )
            continue
        fits = (results or {}).get("fits", {})
        rank_ic = fits.get("mean_validation_rank_ic")
        reason = None
        if any(root.name in {p.name for p in invalid_store_roots} for root in feature_roots):
            reason = "trained on a wrong chip feature store"
        elif results is not None and rank_ic is not None and float(rank_ic) == 0.0:
            reason = "mean validation Rank IC is exactly 0 (constant predictions)"
        spec_names = {
            (split.get("features", {}).get("identity", {}).get("spec", {}) or {}).get("name")
            for split in splits.values()
            if isinstance(split, dict)
        }
        out.append(
            {
                "kind": "lgbm-launch",
                "path": launch,
                "spec": ", ".join(sorted(str(s) for s in spec_names if s)),
                "state": "finished" if results is not None else "incomplete",
                "mean_validation_rank_ic": rank_ic,
                "invalid": reason is not None,
                "reason": reason,
            }
        )
    return out


def quarantine(volume: Path, entries: list[dict], apply: bool) -> list[tuple[Path, Path]]:
    """Move every invalid entry under ``diagnostics/invalid/``; returns (source, target) pairs."""
    moves = []
    for entry in entries:
        if not entry["invalid"]:
            continue
        source = Path(entry["path"])
        target = volume / "diagnostics" / "invalid" / source.relative_to(volume)
        moves.append((source, target))
        if apply:
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    return moves


def _print_table(entries: list[dict]) -> None:
    for entry in entries:
        verdict = "INVALID" if entry["invalid"] else "ok"
        detail = (
            f"margin={entry['symbols_with_margin']} "
            f"institutional={entry['symbols_with_institutional']}"
            if entry["kind"] == "feature-store"
            else f"mean_validation_rank_ic={entry['mean_validation_rank_ic']}"
        )
        print(
            f"{verdict:8} {entry['kind']:14} {entry['spec'] or '-':18} {entry['state'] or '-':10} "
            f"{detail}  {entry['path']}"
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
    args = parser.parse_args(argv)
    volume = args.volume.resolve()
    if not volume.is_dir():
        print(f"volume root is not a directory: {volume}", file=sys.stderr)
        return 2
    stores = chip_stores(volume)
    invalid_roots = {Path(s["path"]) for s in stores if s["invalid"]}
    launches = chip_launches(volume, invalid_roots)
    entries = stores + launches
    if not entries:
        print("No chip feature stores or chip LightGBM launches found")
        return 0
    _print_table(entries)
    moves = quarantine(volume, entries, args.apply)
    if not moves:
        print("Nothing to quarantine")
        return 0
    for source, target in moves:
        print(("moved " if args.apply else "would move ") + f"{source} -> {target}")
    if not args.apply:
        print("Re-run with --apply to move them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
