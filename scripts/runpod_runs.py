#!/usr/bin/env python3
"""Read run-scoped control records without depending on a global latest writer."""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}"


def validate_run(value):
    if not isinstance(value, str) or not re.fullmatch(RUN, value) or "--" in value:
        raise ValueError("Invalid run ID")
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Run completion time must be a timestamp")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Run completion time must include a timezone")
    return stamp


class RunRecords:
    def __init__(self, root=ROOT, bucket=None):
        self.root = Path(root)
        self.bucket = bucket or os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.bucket):
            raise ValueError("Network volume ID must be loaded from .env")

    def call(self, *arguments, optional=False):
        result = subprocess.run(
            ["bash", str(self.root / "scripts/runpod_s3_project.sh"), *arguments],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            if optional and any(code in result.stderr for code in ("(404)", "(NoSuchKey)")):
                return None
            raise ValueError(
                "Run record request failed; no run was guessed: " + result.stderr.strip()
            )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("Run record must be a JSON object")
        return payload

    def read(self, key, *, optional=False):
        return self.call(
            "s3", "cp", f"s3://{self.bucket}/{key}", "-", "--only-show-errors", optional=optional
        )

    def keys(self):
        token = None
        seen = set()
        for _ in range(256):
            args = [
                "s3api",
                "list-objects-v2",
                "--bucket",
                self.bucket,
                "--prefix",
                "lifecycle/runs/",
                "--max-keys",
                "1000",
                "--no-paginate",
                "--output",
                "json",
            ]
            if token:
                args += ["--continuation-token", token]
            page = self.call(*args)
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if re.fullmatch(
                    r"lifecycle/runs/"
                    + RUN
                    + r"/(training|validation|training-completed|wandb)\.json",
                    key,
                ):
                    validate_run(key.split("/")[2])
                    yield key
            if not page.get("IsTruncated", False):
                return
            token = page.get("NextContinuationToken")
            if not token or token in seen:
                raise ValueError("Invalid run listing continuation")
            seen.add(token)
        raise ValueError("Run record listing exceeded its safety limit")

    def lifecycle(self, run_id, kind):
        validate_run(run_id)
        result = self.read(f"lifecycle/runs/{run_id}/{kind}.json", optional=True)
        if result is None:
            legacy = self.read(f"lifecycle/stage1/{kind}.json", optional=True)
            result = legacy if legacy and legacy.get("wandb_run_id") == run_id else None
        if result is not None and (
            result.get("wandb_run_id") != run_id or result.get("kind") != f"stage1-{kind}"
        ):
            raise ValueError("Lifecycle ownership mismatch")
        return result or {}

    def records(self):
        # Metadata only: at most two outstanding requests and 64 records per batch.
        batch = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for key in self.keys():
                batch.append(key)
                if len(batch) == 64:
                    yield from zip(batch, pool.map(self.read, batch), strict=True)
                    batch = []
            if batch:
                yield from zip(batch, pool.map(self.read, batch), strict=True)

    def choose(self, purpose):
        candidates = {}
        completed = set()

        def nominate(run_id, value):
            stamp = timestamp(value)
            candidates[run_id] = max(stamp, candidates.get(run_id, stamp))

        for key, record in self.records():
            run_id = key.split("/")[2]
            if key.endswith("/training-completed.json"):
                if (
                    record.get("run_id") != run_id
                    or record.get("kind") != "stage1-training-completion"
                    or record.get("state") != "ready"
                    or record.get("training_completed") is not True
                ):
                    raise ValueError("Invalid training completion record or ownership")
                completed.add(run_id)
                if purpose == "completed":
                    nominate(run_id, record.get("completed_at"))
            elif key.endswith("/training.json"):
                if record.get("wandb_run_id") != run_id or record.get("kind") != "stage1-training":
                    raise ValueError("Training lifecycle ownership mismatch")
                if (
                    purpose == "resume"
                    and record.get("state") in {"failed", "timed_out", "waiting_for_resume"}
                    and record.get("training_completed") is not True
                ):
                    nominate(run_id, record.get("generated_at"))
                if (
                    purpose == "results"
                    and record.get("state") == "ready"
                    and record.get("training_completed") is True
                ):
                    nominate(run_id, record.get("generated_at"))
            elif key.endswith("/validation.json") and purpose == "results":
                if (
                    record.get("wandb_run_id") != run_id
                    or record.get("kind") != "stage1-validation"
                ):
                    raise ValueError("Validation lifecycle ownership mismatch")
                if record.get("state") == "ready" and record.get("validation_completed") is True:
                    nominate(run_id, record.get("generated_at"))
        legacy = self.read("lifecycle/stage1/training.json", optional=True)
        if legacy and legacy.get("wandb_run_id"):
            run_id = validate_run(legacy.get("wandb_run_id"))
            if legacy.get("kind") != "stage1-training":
                raise ValueError("Invalid historical training lifecycle")
            if purpose == "completed":
                eligible = legacy.get("training_completed") is True
            elif purpose == "results":
                eligible = (
                    legacy.get("state") == "ready" and legacy.get("training_completed") is True
                )
            else:
                eligible = legacy.get("state") in {
                    "failed",
                    "timed_out",
                    "waiting_for_resume",
                } and not legacy.get("training_completed")
            if eligible and run_id not in candidates:
                nominate(run_id, legacy.get("generated_at"))
        if purpose == "resume":
            candidates = {k: v for k, v in candidates.items() if k not in completed}
            if len(candidates) > 1:
                raise ValueError(
                    "Multiple interrupted runs; specify RUN_ID: " + ", ".join(sorted(candidates))
                )
        if not candidates:
            raise ValueError(
                "No eligible run found; specify RUN_ID if selecting a historical artifact"
            )
        return max(candidates, key=lambda run: (candidates[run], run))

    def selection(self, run_id):
        validate_run(run_id)
        payload = self.read(f"lifecycle/runs/{run_id}/selection.json", optional=True)
        if payload is None:
            manifest = self.read(f"savedModel/{run_id}/run-manifest.json")
            provenance = manifest["selection_provenance"]
            selection_id = provenance["selection_id"]
            if not re.fullmatch(r"selection-[0-9a-f]{16}", selection_id):
                raise ValueError("Invalid historical selection identity")
            payload = self.read(f"lifecycle/selections/{selection_id}.json")
            if payload["selection_sha256"] != provenance["selection_sha256"]:
                raise ValueError("Historical selection provenance mismatch")
        selection = runpy.run_path(str(self.root / "scripts/runpod_selection.py"))
        payload = selection["_validate_selection"](payload, project_root=self.root)
        return selection["_store_selection"](self.root, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("latest", "lifecycle", "selection", "status"))
    parser.add_argument("--run-id")
    parser.add_argument("--kind", choices=("training", "validation"), default="training")
    parser.add_argument(
        "--purpose", choices=("completed", "results", "resume"), default="completed"
    )
    args = parser.parse_args(argv)
    reader = RunRecords()
    if args.command == "latest":
        print(reader.choose(args.purpose))
    elif args.command == "lifecycle":
        print(json.dumps(reader.lifecycle(args.run_id, args.kind)))
    elif args.command == "selection":
        print(reader.selection(args.run_id))
    else:
        for key, payload in reader.records():
            if key.endswith("/training-completed.json"):
                continue
            detail = ""
            if key.endswith("/wandb.json"):
                detail = " ".join(
                    f"{name}={value.get('state', '?')}"
                    for name, value in payload.get("components", {}).items()
                )
            print(
                f"{key.split('/')[2]} {Path(key).stem}: {payload.get('state', '?')} "
                f"pod={payload.get('pod_id', '?')} at={payload.get('generated_at', '?')} {detail}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"Run discovery failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
