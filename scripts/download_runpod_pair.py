#!/usr/bin/env python3
"""Download both independent models without duplicating their checkpoint files."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download_pair(reader, pair, *, script, resume=False, checkpoint_scope="all", workers=2):
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Paired result download workers must be 1 or 2")
    pair = reader.validate_pair(pair, pair["run_id"])
    query = runpy.run_path(str(reader.root / "scripts/runpod_run_status.py"))
    inspect = query["inspect_pair"]
    status = inspect(reader, pair["run_id"], pair)
    if status["state"] != "complete":
        states = ", ".join(f"{task}={item['state']}" for task, item in status["children"].items())
        raise ValueError(f"Paired run has not completed both final test evaluations: {states}")
    target = reader.root / "artifacts/runpod" / pair["run_id"]
    if target.exists() and not resume:
        raise ValueError("Download target already exists; use --resume to fill or refresh known files")
    target.mkdir(parents=True, exist_ok=True)

    def download(child):
        command = ["bash", str(script), "--checkpointScope", checkpoint_scope]
        if resume:
            command.append("--resume")
        command.append(child["run_id"])
        result = subprocess.run(command, check=False)
        return child["run_id"], result.returncode

    # Two finite I/O jobs; each retains the existing streaming downloader and
    # independently validated artifacts. No checkpoint data is loaded in Python.
    with ThreadPoolExecutor(max_workers=query["worker_count"](reader, workers)) as pool:
        outcomes = list(pool.map(download, pair["children"].values()))
    failed = [run_id for run_id, code in outcomes if code]
    if failed:
        raise ValueError("Paired download incomplete; retry with --resume. Failed models: " + ", ".join(failed))
    result = {
        "schema_version": 1,
        "kind": "paired-download",
        "run_id": pair["run_id"],
        "experiment": pair["experiment"],
        "seed": pair["seed"],
        "state": "complete",
        "checkpoint_scope": checkpoint_scope,
        "children": {
            task: {**child, "relative_path": "../" + child["run_id"]}
            for task, child in pair["children"].items()
        },
    }
    for name, payload in (("pair.json", pair), ("download-result.json", result)):
        temporary = target / (name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target / name)
    print(f"Downloaded paired RunPod result: run_id={pair['run_id']} target={target}")
    for task, child in result["children"].items():
        print(f"  {task}: {child['run_id']} ({child['relative_path']})")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-json", required=True)
    parser.add_argument("--download-script", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-scope", choices=("all", "best"), default="all")
    parser.add_argument("--workers", type=int, default=os.environ.get("RUNPOD_RESULT_DOWNLOAD_WORKERS", "2"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    records = runpy.run_path(str(root / "scripts/runpod_runs.py"))
    download_pair(
        records["RunRecords"](), json.loads(args.pair_json), script=args.download_script,
        resume=args.resume, checkpoint_scope=args.checkpoint_scope, workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(f"Paired result download failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
