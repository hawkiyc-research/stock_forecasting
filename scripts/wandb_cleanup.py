#!/usr/bin/env python3
"""List or delete every W&B run of the project (cleanup side task 1).

``list`` writes an inventory of the runs and artifact collections of one project; ``delete``
removes exactly the runs and collections named in such an inventory, so the user approves the
list before anything is removed. The API key is read from the project .env by this program
only and is never printed or exported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "stock_forecasting"
INVENTORY_KIND = "wandb-inventory"


def dotenv_value(path: Path, key: str) -> str:
    """Read one simple project dotenv value without executing the file."""
    matches: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
            matches.append(line.split("=", 1)[1])
    if len(matches) != 1:
        return ""
    value = matches[0]
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    return value


def default_env_file() -> Path:
    return Path(os.environ.get("RUNPOD_ENV_FILE") or ROOT / ".env")


def connect(env_file: Path, entity: str | None, project: str | None):
    """Return (api, entity, project); the key stays inside this process."""
    if not env_file.is_file():
        raise SystemExit(f"Project .env is missing: {env_file}")
    key = dotenv_value(env_file, "WANDB_API_KEY")
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    try:
        # Without a .env key the SDK falls back to its own login (~/.netrc from `wandb login`).
        api = wandb.Api(api_key=key or None, timeout=120)
    except Exception as error:
        raise SystemExit(
            "W&B login failed: add WANDB_API_KEY to the project .env or run `wandb login` "
            f"({type(error).__name__})"
        ) from None
    entity = entity or dotenv_value(env_file, "WANDB_ENTITY") or api.default_entity
    project = project or dotenv_value(env_file, "WANDB_PROJECT") or DEFAULT_PROJECT
    if not entity:
        raise SystemExit("WANDB_ENTITY is not set and the API key has no default entity")
    return api, str(entity), str(project)


def list_inventory(api, entity: str, project: str) -> dict:
    path = f"{entity}/{project}"
    runs = []
    for run in api.runs(path, per_page=200):
        runs.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "group": getattr(run, "group", None),
                "tags": list(run.tags or []),
                "created_at": str(getattr(run, "created_at", "")),
                "url": run.url,
            }
        )
    collections = []
    for artifact_type in api.artifact_types(project=path):
        for collection in artifact_type.collections():
            versions = 0
            with_versions = getattr(collection, "artifacts", None)
            if callable(with_versions):
                try:
                    versions = sum(1 for _ in with_versions())
                except Exception:
                    versions = -1
            collections.append(
                {"type": artifact_type.name, "name": collection.name, "versions": versions}
            )
    return {
        "kind": INVENTORY_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "entity": entity,
        "project": project,
        "runs": runs,
        "artifact_collections": collections,
    }


def print_inventory(inventory: dict) -> None:
    print(f"W&B project {inventory['entity']}/{inventory['project']}")
    print(f"Runs: {len(inventory['runs'])}")
    for run in inventory["runs"]:
        print(f"  {run['id']:<12} {run['state']:<9} {run['created_at'][:19]:<19} {run['name']}")
    print(f"Artifact collections: {len(inventory['artifact_collections'])}")
    for collection in inventory["artifact_collections"]:
        print(f"  {collection['type']:<12} {collection['name']} "
              f"({collection['versions']} versions)")


def delete_inventory(api, inventory: dict, *, apply: bool) -> int:
    path = f"{inventory['entity']}/{inventory['project']}"
    runs, collections = inventory["runs"], inventory["artifact_collections"]
    print(f"{'Deleting' if apply else 'Would delete'} {len(runs)} runs and "
          f"{len(collections)} artifact collections of {path}")
    if not apply:
        return 0
    failures = 0
    for run in runs:
        try:
            api.run(f"{path}/{run['id']}").delete(delete_artifacts=True)
            print(f"  deleted run {run['id']} ({run['name']})")
        except Exception as error:
            failures += 1
            print(f"  FAILED run {run['id']}: {type(error).__name__}: {error}", file=sys.stderr)
    for artifact_type in api.artifact_types(project=path):
        for collection in artifact_type.collections():
            if not any(c["name"] == collection.name and c["type"] == artifact_type.name
                       for c in collections):
                continue
            try:
                collection.delete()
                print(f"  deleted collection {artifact_type.name}/{collection.name}")
            except Exception as error:
                failures += 1
                print(f"  FAILED collection {collection.name}: "
                      f"{type(error).__name__}: {error}", file=sys.stderr)
    remaining = sum(1 for _ in api.runs(path, per_page=200))
    print(f"Runs remaining in {path}: {remaining}; failures: {failures}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--env-file", type=Path, default=default_env_file())
    parser.add_argument("--entity")
    parser.add_argument("--project")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Write the inventory and print it.")
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    listing.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/audit" / f"wandb-inventory-{stamp}.json",
    )
    deletion = commands.add_parser("delete", help="Delete the runs named in an inventory.")
    deletion.add_argument("--inventory", type=Path, required=True)
    deletion.add_argument("--apply", action="store_true", help="Without it, only count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api, entity, project = connect(args.env_file, args.entity, args.project)
    if args.command == "list":
        inventory = list_inventory(api, entity, project)
        print_inventory(inventory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
        print(f"Inventory written to {args.output}")
        return 0
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if (
        inventory.get("kind") != INVENTORY_KIND
        or inventory.get("entity") != entity
        or inventory.get("project") != project
    ):
        raise SystemExit("The inventory does not belong to the selected W&B project")
    return delete_inventory(api, inventory, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
