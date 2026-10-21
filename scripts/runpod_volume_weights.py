#!/usr/bin/env python3
"""Inventory or delete experiment weights on the RunPod network volume (cleanup side task 2).

``inventory`` lists the saved-model checkpoints, baseline model files and per-sample prediction
exports with their sizes and writes the exact object keys to a JSON file. ``delete`` removes
only the keys of one category named in such an inventory, one object per wrapper call, after
the user has read the list. Records (metadata, metrics, manifests, leaderboards, evaluation
arrays, reports) are never listed for deletion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/runpod_s3_project.sh"
INVENTORY_KIND = "runpod-volume-weight-inventory"
PREFIXES = (
    "savedModel/",
    "baselines/",
    "diagnostics/signal-predictions/",
    "diagnostics/forward-predictions/",
)
CHECKPOINT_WEIGHT = re.compile(
    r"^savedModel/(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,119})/checkpoint-[0-9]+/"
    r"[A-Za-z0-9._-]+\.(safetensors|pt|pth|bin)$"
)
BASELINE_WEIGHT = re.compile(
    r"^baselines/(?P<run>baseline-[0-9a-f]{64})/jobs/[A-Za-z0-9._/-]+"
    r"\.(pkl|pt|pth|joblib|safetensors|bin)$"
)
EXPORT = re.compile(
    r"^diagnostics/(?:signal|forward)-predictions/(?P<run>[A-Za-z0-9._-]+)/"
    r"[A-Za-z0-9._/-]+/predictions\.npy$"
)
CATEGORIES = ("weights", "exports")
MAX_WORKERS = 4


def classify(key: str) -> tuple[str, str] | None:
    """(category, owner) of a deletable object, or None for records that stay."""
    for pattern, category in ((CHECKPOINT_WEIGHT, "weights"), (BASELINE_WEIGHT, "weights"),
                              (EXPORT, "exports")):
        match = pattern.match(key)
        if match:
            return category, match.group("run")
    return None


def owner_of(key: str) -> str:
    parts = key.split("/")
    if key.startswith("diagnostics/"):
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def run_wrapper(arguments: list[str], *, allow_delete: bool = False) -> bytes:
    env = {k: v for k, v in os.environ.items()}
    if allow_delete:
        env["RUNPOD_S3_ALLOW_DELETE"] = "1"
    result = subprocess.run(
        ["bash", str(WRAPPER), *arguments], env=env, capture_output=True, check=False,
        timeout=int(os.environ.get("RUNPOD_FEATURE_TRANSFER_TIMEOUT", 1800)),
    )
    if result.returncode != 0:
        # Do not echo transport output that could contain credential values.
        raise RuntimeError(f"RunPod S3 wrapper failed (exit {result.returncode}): {arguments[:2]}")
    return result.stdout


def list_objects(volume_id: str, prefix: str) -> list[dict]:
    objects, token = [], None
    while True:
        arguments = ["s3api", "list-objects-v2", "--bucket", volume_id, "--prefix", prefix,
                     "--max-keys", "1000", "--output", "json"]
        if token:
            arguments += ["--continuation-token", token]
        payload = json.loads(run_wrapper(arguments).decode("utf-8") or "{}")
        for item in payload.get("Contents", []) or []:
            objects.append({"key": str(item["Key"]), "bytes": int(item["Size"])})
        token = payload.get("NextContinuationToken")
        if not payload.get("IsTruncated") or not token:
            return objects


def build_inventory(volume_id: str) -> dict:
    owners: dict[str, dict] = {}
    deletable = {"weights": [], "exports": []}
    for prefix in PREFIXES:
        for item in list_objects(volume_id, prefix):
            owner = owners.setdefault(owner_of(item["key"]), {
                "weights_bytes": 0, "weights_files": 0, "exports_bytes": 0, "exports_files": 0,
                "records_bytes": 0, "records_files": 0,
            })
            kind = classify(item["key"])
            if kind is None:
                owner["records_bytes"] += item["bytes"]
                owner["records_files"] += 1
                continue
            category, _ = kind
            owner[f"{category}_bytes"] += item["bytes"]
            owner[f"{category}_files"] += 1
            deletable[category].append(item)
    return {
        "kind": INVENTORY_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "volume_id": volume_id,
        "owners": dict(sorted(owners.items())),
        "deletable": deletable,
    }


def gigabytes(value: int) -> str:
    return f"{value / 1024**3:8.2f} GB"


def print_inventory(inventory: dict) -> None:
    print(f"RunPod volume {inventory['volume_id']}")
    print(f"{'owner':<72} {'weights':>12} {'exports':>12} {'records (kept)':>16}")
    for owner, row in inventory["owners"].items():
        print(f"{owner:<72} {gigabytes(row['weights_bytes']):>12} "
              f"{gigabytes(row['exports_bytes']):>12} {gigabytes(row['records_bytes']):>16}")
    for category in CATEGORIES:
        items = inventory["deletable"][category]
        total = sum(item["bytes"] for item in items)
        print(f"{category}: {len(items)} files, {gigabytes(total).strip()}")


def delete_category(inventory: dict, category: str, *, apply: bool, workers: int) -> int:
    volume_id = inventory["volume_id"]
    items = [item for item in inventory["deletable"][category] if classify(item["key"])]
    total = sum(item["bytes"] for item in items)
    print(f"{'Deleting' if apply else 'Would delete'} {len(items)} {category} files, "
          f"{gigabytes(total).strip()}, from {volume_id}")
    if not apply:
        return 0

    def remove(item: dict) -> str:
        run_wrapper(["s3", "rm", f"s3://{volume_id}/{item['key']}"], allow_delete=True)
        return item["key"]

    failures = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="volume-delete") as pool:
        pending = []
        for item in items:
            pending.append(pool.submit(remove, item))
            if len(pending) >= 2 * workers:
                failures += _collect(pending.pop(0))
        for future in pending:
            failures += _collect(future)
    print(f"Deleted {len(items) - failures} of {len(items)} {category} files; failures: {failures}")
    return 1 if failures else 0


def _collect(future) -> int:
    try:
        print(f"  deleted {future.result()}")
        return 0
    except Exception as error:
        print(f"  FAILED: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory", help="List weights, exports and records.")
    listing.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/audit"
        / f"volume-weight-inventory-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
    )
    deletion = commands.add_parser("delete", help="Delete one category of an inventory.")
    deletion.add_argument("--inventory", type=Path, required=True)
    deletion.add_argument("--category", choices=CATEGORIES, required=True)
    deletion.add_argument("--apply", action="store_true", help="Without it, only count.")
    deletion.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id):
        print("RUNPOD_NETWORK_VOLUME_ID must be loaded from .env (use runpod_workflow.sh)",
              file=sys.stderr)
        return 2
    if args.command == "inventory":
        inventory = build_inventory(volume_id)
        print_inventory(inventory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"Inventory written to {args.output}")
        return 0
    if not 1 <= args.workers <= MAX_WORKERS:
        print(f"--workers must be 1-{MAX_WORKERS}", file=sys.stderr)
        return 2
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("kind") != INVENTORY_KIND or inventory.get("volume_id") != volume_id:
        print("The inventory does not belong to the configured volume", file=sys.stderr)
        return 2
    return delete_category(inventory, args.category, apply=args.apply, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
