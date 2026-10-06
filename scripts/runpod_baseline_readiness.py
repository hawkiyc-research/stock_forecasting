#!/usr/bin/env python3
"""Read-only baseline admission for prepared sources and current cleaning indexes."""

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

DATA_GATE = runpy.run_path(str(Path(__file__).with_name("runpod_dataset_readiness.py")))
# Preserve the shared prepared-data validator for existing control-plane callers.
globals().update({key: value for key, value in DATA_GATE.items() if not key.startswith("__")})
READINESS_REUSE_SECONDS = 600


class BaselineArtifactReader(DATA_GATE["ArtifactReader"]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}

    def read(self, key):
        # Only bounded metadata is memoized, never bars, arrays or model weights.
        if key not in self.metadata:
            self.metadata[key] = super().read(key)
        return self.metadata[key]

    def inventory(self, prefix):
        """Observe revisions with paginated LISTs instead of one HEAD per artifact."""
        DATA_GATE["relative_path"](prefix.rstrip("/"))
        items, cursor, seen = [], None, set()
        # Pagination is sequential because each page supplies the next token.
        for _ in range(16):
            args = ["s3api", "list-objects-v2", "--bucket", self.bucket,
                    "--prefix", prefix, "--max-keys", "1000", "--no-paginate",
                    "--output", "json"]
            if cursor:
                args.extend(("--continuation-token", cursor))
            page = json.loads(self.s3(*args))
            if not isinstance(page, dict) or type(page.get("IsTruncated")) is not bool:
                raise ValueError("Invalid readiness artifact inventory")
            contents = page.get("Contents", [])
            if not isinstance(contents, list) or len(contents) > 1000:
                raise ValueError("Invalid readiness artifact inventory entries")
            for item in contents:
                if (not isinstance(item, dict) or not isinstance(item.get("Key"), str)
                        or not item["Key"].startswith(prefix)
                        or type(item.get("Size")) is not int or item["Size"] < 0
                        or not isinstance(item.get("ETag"), str)
                        or not isinstance(item.get("LastModified"), str)):
                    raise ValueError("Readiness inventory lacks artifact revision metadata")
                items.append({name: item[name] for name in ("Key", "Size", "ETag", "LastModified")})
            if not page["IsTruncated"]:
                if len({item["Key"] for item in items}) != len(items):
                    raise ValueError("Readiness inventory repeats an artifact")
                return sorted(items, key=lambda item: item["Key"])
            cursor = page.get("NextContinuationToken")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise ValueError("Readiness artifact inventory pagination is incomplete")
            seen.add(cursor)
        raise ValueError("Readiness artifact inventory exceeds the bounded page limit")

    def optional_read(self, key):
        """Only a confirmed missing object means pending, never a transport failure."""
        DATA_GATE["relative_path"](key)
        if self.volume is not None:
            path = self.volume / key
            if not path.exists() and not path.is_symlink():
                return None
        else:
            listing = json.loads(
                self.s3(
                    "s3api",
                    "list-objects-v2",
                    "--bucket",
                    self.bucket,
                    "--prefix",
                    key,
                    "--max-keys",
                    "1",
                    "--no-paginate",
                    "--output",
                    "json",
                )
            )
            if not isinstance(listing, dict):
                raise ValueError("Invalid cleaned-index metadata listing")
            contents = listing.get("Contents", [])
            if (
                not isinstance(contents, list)
                or len(contents) > 1
                or any(
                    not isinstance(item, dict) or not isinstance(item.get("Key"), str)
                    for item in contents
                )
                or type(listing.get("IsTruncated")) is not bool
                or (not contents and listing["IsTruncated"])
            ):
                raise ValueError("Invalid cleaned-index metadata listing")
            if not any(item.get("Key") == key for item in contents):
                return None
        return self.read(key)


def cleaning_identity(project, bar, manifest_sha256, policy):
    """Resolve the runtime cache key without importing an ML environment locally."""
    tree = ast.parse((project / "src/stock_forecasting/data/sample_universe.py").read_text())
    algorithms = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ALGORITHM" for target in node.targets
        )
    ]
    if len(algorithms) != 1 or not isinstance(algorithms[0], str) or not algorithms[0]:
        raise ValueError("The runtime cleaning algorithm identifier is unavailable")
    # Production admission already requires this fixed split contract. Keep the
    # cache's exact timestamp representation; never count the old prepared ranges.
    fixed = bar["identity"]["fixed_split"]
    train_end, validation_end, test_end = (
        fixed[name] + "T00:00:00Z" for name in ("train_end", "validation_end", "test_end")
    )
    return {
        "algorithm": algorithms[0],
        "source_manifest_sha256": manifest_sha256,
        "policy": policy,
        "window_size": bar["identity"]["window_size"],
        "splits": [
            ["train", "1900-01-01T00:00:00Z", train_end],
            ["validation", train_end, validation_end],
            ["test", validation_end, test_end],
        ],
    }


def verify_cleaning(project, selected, prepared, policy, reader, *, splits):
    prefix = f"datasets/{selected['dataset_request_sha256']}/prepared/"
    bar_bytes = reader.read(prefix + "bar-store/bar-store.json")
    manifest_sha256 = hashlib.sha256(bar_bytes).hexdigest()
    if manifest_sha256 != prepared["bar_store_manifest_sha256"]:
        raise ValueError("Prepared source changed during baseline readiness")
    expected = cleaning_identity(project, json.loads(bar_bytes), manifest_sha256, policy)
    key = DATA_GATE["digest"](expected)
    root = prefix + f"sample-universes/{key}/"
    raw = reader.optional_read(root + "universe.json")
    result = {
        "sample_universe": key,
        "policy_sha256": DATA_GATE["digest"](policy),
        "state": "pending_build",
        "eligible_sample_counts": None,
    }
    if raw is None:
        return result
    payload = json.loads(raw)
    if payload.get("identity") != expected or payload.get("state") != "ready":
        raise ValueError("Current cleaned sample index has an inconsistent policy/source/state")
    counts = payload.get("split_counts", {})
    if set(counts) != {"train", "validation", "test"} or any(
        type(value) is not int or value < 0 for value in counts.values()
    ):
        raise ValueError("Current cleaned sample index has invalid split counts")
    for split in splits:
        if counts[split] < 1:
            raise ValueError(f"No eligible {split} windows under the current cleaning rules")
        if payload.get("audit", {}).get(split, {}).get("accepted") != counts[split]:
            raise ValueError(f"Cleaned {split} count differs from its accepted-window audit")
    checksum = payload.get("ranges_sha256", "")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Current cleaned sample index has no valid ranges checksum")
    ranges = root + "cutoff-ranges.parquet"
    if reader.size(ranges) < 1:
        raise ValueError("Current cleaned sample range index is empty")
    if (
        reader.volume is not None
        and DATA_GATE["READINESS"]["_sha256"](reader.path(ranges)) != checksum
    ):
        raise ValueError("Current cleaned sample range index checksum mismatch")
    result.update(
        {
            "state": "ready",
            "eligible_sample_counts": {split: counts[split] for split in splits},
            "verification": prepared["verification"],
        }
    )
    return result


def baseline_sources(project, selected):
    policy_tools = runpy.run_path(str(project / "src/stock_forecasting/data_policy.py"))
    policy = policy_tools["load_data_policy"](project)
    evaluation = copy.deepcopy(selected)
    evaluation["dataset_request"] = policy_tools["evaluation_request"](selected, policy)
    evaluation["dataset_request_sha256"] = policy_tools["evaluation_dataset_id"](selected, policy)
    selections = {
        selected["dataset_request_sha256"]: selected,
        evaluation["dataset_request_sha256"]: evaluation,
    }
    split_sources = {
        "train": selected["dataset_request_sha256"],
        "validation": evaluation["dataset_request_sha256"],
        "test": evaluation["dataset_request_sha256"],
    }
    return policy, selections, split_sources


def verify_baseline_data(project, selected, reader, *, workers):
    policy, selections, split_sources = baseline_sources(project, selected)

    def check(item):
        dataset_id, source = item
        prepared = DATA_GATE["verify_data"](
            project, source, reader, workers=max(1, workers // len(selections))
        )
        splits = [split for split, value in split_sources.items() if value == dataset_id]
        cleaned = verify_cleaning(project, source, prepared, policy, reader, splits=splits)
        return dataset_id, {
            "date_range": prepared["date_range"],
            "splits": splits,
            "prepared": prepared,
            "cleaning": cleaned,
        }

    # Divide the existing I/O budget between at most two sources, not per split.
    sources = dict(
        DATA_GATE["bounded_map"](check, selections.items(), min(workers, len(selections)))
    )
    pending = any(source["cleaning"]["state"] != "ready" for source in sources.values())
    eligible = {
        split: (sources[key]["cleaning"]["eligible_sample_counts"] or {}).get(split)
        for split, key in split_sources.items()
    }
    return {
        "state": "ready_for_baseline_build",
        "scope": "baseline_build_inputs_not_completed_baseline_results",
        "cleaning_state": "pending_build" if pending else "ready",
        "eligible_sample_counts": eligible,
        "split_sources": split_sources,
        "sources": sources,
        "next_action": (
            "Build current-rule sample indexes from existing bars before fitting any baseline; "
            "CPU prepare and market-data downloads are not required."
            if pending
            else "Use the verified current-rule sample indexes for baseline building."
        ),
    }


def readiness_snapshot(project, selected, reader, *, workers):
    policy, selections, _ = baseline_sources(project, selected)

    def snapshot(dataset_id):
        prefix = f"datasets/{dataset_id}/"
        metadata = {name: hashlib.sha256(reader.read(prefix + name)).hexdigest() for name in (
            "dataset-manifest.json", "download-manifest.json", "prepared/bar-store/bar-store.json",
        )}
        bar_name = "prepared/bar-store/bar-store.json"
        identity = cleaning_identity(
            project, json.loads(reader.read(prefix + bar_name)), metadata[bar_name], policy,
        )
        return dataset_id, {
            "metadata": metadata,
            "bars": reader.inventory(prefix + "prepared/bar-store/"),
            "cleaned": reader.inventory(
                prefix + "prepared/sample-universes/" + DATA_GATE["digest"](identity) + "/"
            ),
        }

    return dict(DATA_GATE["bounded_map"](
        snapshot, sorted(selections), min(workers, len(selections))
    ))


def verify_with_receipt(project, selected, reader, *, workers, release_digest,
                        cache_root=None, now=None):
    """Reuse recent successful checks only after observing unchanged source revisions."""
    if reader.volume is not None or not release_digest:
        return verify_baseline_data(project, selected, reader, workers=workers), False
    if not re.fullmatch(r"[0-9a-f]{64}", release_digest):
        raise ValueError("Invalid verified code release for readiness reuse")
    started = time.time() if now is None else now
    policy, _, _ = baseline_sources(project, selected)
    context = {"schema": 1, "volume": reader.bucket, "release": release_digest,
               "dataset_request": selected["dataset_request"], "policy": policy}
    root = cache_root if cache_root is not None else project / ".runpod/readiness"
    path = root / ("baseline-" + DATA_GATE["digest"](context) + ".json")
    # Remote errors must propagate; a cached success never substitutes for a
    # failed freshness check. A deleted/replaced shard changes this snapshot.
    snapshot = readiness_snapshot(project, selected, reader, workers=workers)
    try:
        receipt = json.loads(path.read_text()) if path.stat().st_size <= 16 * 1024**2 else {}
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
        receipt = {}
    checked_at = receipt.get("checked_at") if isinstance(receipt, dict) else None
    if (type(checked_at) in (int, float) and 0 <= started - checked_at < READINESS_REUSE_SECONDS
            and receipt.get("context") == context and receipt.get("snapshot") == snapshot
            and isinstance(receipt.get("result"), dict)
            and receipt["result"].get("cleaning_state") in {"ready", "pending_build"}
            and receipt["result"].get("state") == "ready_for_baseline_build"):
        return receipt["result"], True
    result = verify_baseline_data(project, selected, reader, workers=workers)
    DATA_GATE["READINESS"]["_atomic_json"](path, {
        "context": context, "snapshot": snapshot, "checked_at": started, "result": result,
    })
    return result, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DATA_GATE["ROOT"])
    parser.add_argument("--selection", default=os.environ.get("RUNPOD_SELECTION_FILE"))
    parser.add_argument("--network-volume-root", type=Path)
    parser.add_argument("--verified-code-release", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    _, selected = DATA_GATE["SELECTION"]["_resolve_selection_path"](
        project, args.selection, validate_local_config=False
    )
    reader = BaselineArtifactReader(
        project, volume=args.network_volume_root, bucket=os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )
    result, reused = verify_with_receipt(
        project, selected, reader, workers=DATA_GATE["worker_count"](),
        release_digest=args.verified_code_release,
    )
    if reused:
        print("Baseline readiness reused (verified within 10 minutes; code, selection, "
              "cleaning policy and remote artifact revisions unchanged).")
        print(f"Current-rule cleaning: {result['cleaning_state']}; "
              "baseline completion and live Pod admission are checked separately.")
        return 0
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["cleaning_state"] == "pending_build":
        print(
            "Baseline inputs available; current-rule cleaned windows are NOT verified yet. "
            "The baseline workflow must build the pending indexes before training."
        )
    else:
        print("Baseline build inputs verified with current-rule cleaned sample counts.")
    print("This input check does not certify that baseline training/results are complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Baseline data readiness failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
