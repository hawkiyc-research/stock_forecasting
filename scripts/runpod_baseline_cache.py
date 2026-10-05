#!/usr/bin/env python3
"""Check complete baseline artifacts on the control host before creating a paid Pod."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _has_duplicate_arrays(wrapper, bucket, prefix, relative_paths):
    """Inspect names only; cleanup progress must not gate model/data reuse."""
    expected = {prefix + path for path in relative_paths}
    continuation = None
    seen_tokens = set()
    # Sequential pages depend on the previous token. Bound both response size
    # and pagination; the artifact suite is small and no array is downloaded.
    for _page in range(32):
        command = [
            "bash",
            wrapper,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix + "jobs/",
            "--max-keys",
            "1000",
            "--no-paginate",
            "--output",
            "json",
        ]
        if continuation is not None:
            command.extend(["--continuation-token", continuation])
        response = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if response.returncode:
            raise RuntimeError("Unable to inspect baseline cleanup; no Pod will be created")
        page = json.loads(response.stdout)
        contents = page.get("Contents", [])
        if not isinstance(contents, list) or len(contents) > 1000:
            raise ValueError("Invalid baseline storage listing")
        if any(item["Key"] in expected for item in contents):
            return True
        if page.get("IsTruncated") is False:
            return False
        continuation = page.get("NextContinuationToken")
        if not isinstance(continuation, str) or not continuation or continuation in seen_tokens:
            raise ValueError("Incomplete baseline storage listing; no Pod will be created")
        seen_tokens.add(continuation)
    raise ValueError("Baseline storage listing exceeds the bounded suite limit")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "require", "identity"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    selection_tools = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    _, selection = selection_tools["_resolve_selection_path"](
        project, os.environ.get("RUNPOD_SELECTION_FILE"), validate_local_config=False
    )
    contract_tools = runpy.run_path(str(project / "src/stock_forecasting/baseline_contract.py"))
    identity = contract_tools["baseline_contract"](project, selection)
    contract_tools["validate_local_configuration"](
        project, selection, identity["contract"]["parameters"]
    )
    if args.command == "identity":
        print(json.dumps(identity))
        return 0
    if os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_TEST_MODE") != "1":
        raise ValueError("The baseline cache gate must run on the local control host")
    bucket = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket):
        raise ValueError("Network volume ID must be loaded from the project .env")
    wrapper = str(project / "scripts/runpod_s3_project.sh")
    policy_tools = runpy.run_path(str(project / "src/stock_forecasting/data_policy.py"))
    evaluation_id = policy_tools["evaluation_dataset_id"](
        selection, identity["contract"]["data_cleaning"]
    )
    dataset_ids = {
        "train": selection["dataset_request_sha256"],
        "validation": evaluation_id,
        "test": evaluation_id,
    }

    def read_json(key):
        response = subprocess.run(
            ["bash", wrapper, "s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"],
            capture_output=True,
            timeout=180,
        )
        if response.returncode:
            raise ValueError(
                f"Required prepared evaluation data is unavailable: {key}; no Pod was created"
            )
        return response.stdout

    # Resolve the common evaluation source BEFORE a paid baseline/train Pod,
    # including when no matching baseline has been built yet.
    manifests = {}
    for dataset_id in sorted(set(dataset_ids.values())):
        raw = read_json(f"datasets/{dataset_id}/prepared/bar-store/bar-store.json")
        manifest = json.loads(raw)
        if manifest.get("state") != "ready":
            raise ValueError("Required prepared data is not ready; no Pod was created")
        manifests[dataset_id] = raw
    prefix = "baselines/" + identity["baseline_id"] + "/"
    command = [
        "bash",
        wrapper,
        "s3",
        "cp",
        f"s3://{bucket}/{prefix}complete.json",
        "-",
        "--only-show-errors",
    ]
    response = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if response.returncode:
        if not any(code in response.stderr for code in ("NoSuchKey", "404", "Not Found")):
            raise RuntimeError(
                "Unable to verify the baseline cache; no Pod will be created. "
                "Check S3 credentials/connectivity."
            )
        if args.command == "require":
            raise ValueError(
                "Matching full-data baselines are missing. "
                "Run: bash scripts/runpod_workflow.sh baseline"
            )
        print(json.dumps({"complete": False, **identity}))
        return 0
    payload = json.loads(response.stdout)
    contract_tools["validate_complete"](payload, identity)
    if payload.get("evaluation_data") is None:
        if args.command == "require":
            raise ValueError(
                "Baseline results need storage finalization, not retraining. "
                "Run: bash scripts/runpod_workflow.sh baseline"
            )
        print(json.dumps({"complete": False, "storage_finalization_pending": True, **identity}))
        return 0
    contract_tools["validate_complete"](payload, identity, require_shared=True)
    # Verify the active immutable manifest locally as well, before any paid Pod.
    import hashlib

    sources = payload.get("data_identity", {}).get("split_sources", {})
    for split, dataset_id in dataset_ids.items():
        source = sources.get(split, {})
        if hashlib.sha256(manifests[dataset_id]).hexdigest() != source.get("manifest_sha256"):
            raise ValueError(f"Saved baseline {split} source differs from the prepared snapshot")
        key = source.get("sample_universe", "")
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Saved baseline has no valid cleaned sample universe")
        universe = json.loads(
            read_json(f"datasets/{dataset_id}/prepared/sample-universes/{key}/universe.json")
        )
        if (
            universe.get("state") != "ready"
            or contract_tools["digest"](universe["identity"]) != key
            or universe["identity"]["policy"] != identity["contract"]["data_cleaning"]
            or universe["identity"]["source_manifest_sha256"] != source["manifest_sha256"]
            or universe["split_counts"][split] != payload["sample_counts"][split]
            or source.get("samples") != payload["sample_counts"][split]
        ):
            raise ValueError(
                f"Saved baseline {split} does not cover the complete cleaned population"
            )

    def check_artifact(item):
        relative, metadata = item
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe baseline artifact path")
        result = subprocess.run(
            [
                "bash",
                wrapper,
                "s3api",
                "head-object",
                "--bucket",
                bucket,
                "--key",
                prefix + relative,
                "--query",
                "ContentLength",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode or result.stdout.strip() != str(metadata["bytes"]):
            raise ValueError(f"Baseline artifact is missing or truncated: {relative}")

    # The manifest contains a bounded model suite, not one future per data sample.
    workers = max(
        1,
        min(int(os.environ.get("RUNPOD_BASELINE_PREFLIGHT_WORKERS", "4")), os.cpu_count() or 1, 8),
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check_artifact, payload["artifacts"].items()))
    if args.command == "check" and _has_duplicate_arrays(
        wrapper, bucket, prefix, contract_tools["duplicate_evaluation_paths"](payload)
    ):
        # Published shared arrays are already usable by training/testing. Only
        # the baseline workflow needs to finish an interrupted storage cleanup.
        print(json.dumps({"complete": False, "storage_finalization_pending": True, **identity}))
        return 0
    print(json.dumps({"complete": True, **identity}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Baseline preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
