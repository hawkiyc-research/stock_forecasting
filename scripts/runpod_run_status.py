#!/usr/bin/env python3
"""Display read-only run inventories using each run's own persisted records."""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import islice
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

STATES = (
    "complete",
    "trained",
    "training",
    "evaluating",
    "incomplete",
    "not_started",
    "unknown",
    "error",
)
RUN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}"


def stamp(value):
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("Recorded timestamps must be strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Recorded timestamps must include a timezone")
    return parsed


def pages(reader, prefix, *, delimiter=False):
    token, seen = None, set()
    for _ in range(256):
        args = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            reader.bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--no-paginate",
            "--output",
            "json",
        ]
        if delimiter:
            args += ["--delimiter", "/"]
        if token:
            args += ["--continuation-token", token]
        page = reader.call(*args)
        yield page
        if not page.get("IsTruncated", False):
            return
        token = page.get("NextContinuationToken")
        if not token or token in seen:
            raise ValueError("Invalid run inventory continuation")
        seen.add(token)
    raise ValueError("Run inventory exceeded its page limit")


def discover(reader):
    """List only metadata and run prefixes, never checkpoint or dataset contents."""
    runs = {}
    for page in pages(reader, "lifecycle/runs/"):
        for item in page.get("Contents", []):
            match = re.fullmatch(
                r"lifecycle/runs/("
                + RUN_PATTERN
                + r")/(selection|training|validation|training-completed|wandb)\.json",
                item["Key"],
            )
            if match:
                run_id = match[1]
                modified = item.get("LastModified") or ""
                runs[run_id] = max(runs.get(run_id, ""), modified)
    for page in pages(reader, "savedModel/", delimiter=True):
        for item in page.get("CommonPrefixes", []):
            match = re.fullmatch(r"savedModel/(" + RUN_PATTERN + r")/", item["Prefix"])
            if match:
                runs.setdefault(match[1], "")
    if len(runs) > 10000:
        raise ValueError("Run inventory exceeded 10000 runs; narrow the stored inventory")

    def order(run_id):
        match = re.match(r"run-(\d{8}T\d{6}Z)-", run_id)
        allocated = (
            datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) if match else None
        )
        return (
            allocated or stamp(runs[run_id]) or datetime.min.replace(tzinfo=UTC),
            run_id,
        )

    return sorted((run for run in runs if "--" not in run), key=order, reverse=True)


def owned(record, run_id, field, kind=None):
    if record and (record.get(field) != run_id or (kind and record.get("kind") != kind)):
        raise ValueError(f"Record ownership/kind mismatch for {run_id}")
    return record or {}


def configuration(selection, manifest):
    provenance = manifest.get("selection_provenance") or {}
    stage = selection.get("stage") or {}
    request = selection.get("dataset_request") or provenance.get("requested_dataset") or {}
    preparation = request.get("preparation") or {}
    universe = request.get("universe") or {}
    contract = manifest.get("training_resume_contract") or {}
    model = contract.get("model") or {}
    return {
        "source": "run_selection" if selection else "run_manifest" if provenance else "unrecorded",
        "selection_id": selection.get("selection_id") or provenance.get("selection_id"),
        "experiment": stage.get("experiment"),
        "stage": stage.get("name") or provenance.get("selected_stage"),
        "config_path": stage.get("config_path") or provenance.get("stage_config_path"),
        "feature_mode": stage.get("feature_mode"),
        "data_profile": request.get("profile"),
        "dataset_revision": request.get("revision"),
        "date_range": request.get("date_range") or {},
        "h_start": preparation.get("h_start"),
        "window_size": preparation.get("window_size"),
        "fixed_split": preparation.get("fixed_split"),
        "universe": universe,
        "runtime": selection.get("runtime") or {},
        "lora": model.get("lora"),
        "unfreeze_last_blocks": model.get("unfreeze_last_blocks"),
    }


def inspect_run(reader, run_id):
    if not re.fullmatch(RUN_PATTERN, run_id) or "--" in run_id:
        raise ValueError("Invalid run ID")
    prefix = f"lifecycle/runs/{run_id}"
    manifest = owned(
        reader.read(f"savedModel/{run_id}/run-manifest.json", optional=True), run_id, "run_id"
    )
    selection = reader.read(f"{prefix}/selection.json", optional=True) or {}
    training = reader.lifecycle(run_id, "training")
    evaluation = reader.lifecycle(run_id, "validation")
    completion = owned(
        reader.read(f"{prefix}/training-completed.json", optional=True),
        run_id,
        "run_id",
        "stage1-training-completion",
    )
    result = owned(
        reader.read(f"savedModel/{run_id}/completion-result/training-result.json", optional=True),
        run_id,
        "run_id",
        "training-completion-result",
    )
    report = owned(
        reader.read(f"evaluations/{run_id}/validation-benchmark.json", optional=True),
        run_id,
        "run_id",
    )
    wandb = owned(reader.read(f"{prefix}/wandb.json", optional=True), run_id, "run_id")
    if not any((manifest, selection, training, evaluation, completion, result, report, wandb)):
        raise ValueError(f"Run not found: {run_id}")
    provenance = manifest.get("selection_provenance") or {}
    if not selection and provenance.get("selection_id"):
        selection_id = provenance["selection_id"]
        if not re.fullmatch(r"selection-[0-9a-f]{16}", selection_id):
            raise ValueError("Invalid recorded selection ID")
        selection = reader.read(f"lifecycle/selections/{selection_id}.json", optional=True) or {}
    if (
        selection
        and provenance.get("selection_id")
        and selection.get("selection_id") != provenance["selection_id"]
    ):
        raise ValueError("Run selection does not match its manifest provenance")
    # Historical inspection never activates a selection or compares it with today's YAML.
    configured = configuration(selection, manifest)
    training_done = bool(manifest) and (
        (completion.get("state") == "ready" and completion.get("training_completed") is True)
        or bool(result)
        or (training.get("state") == "ready" and training.get("training_completed") is True)
    )
    evaluation_done = report.get("state") == "ready" and bool(report.get("completed_at"))
    # A newly started evaluation must not be masked by an older successful report.
    newer_evaluation = (
        evaluation.get("state") not in (None, "ready")
        and evaluation.get("generated_at")
        and (
            not report.get("completed_at")
            or stamp(evaluation["generated_at"]) > stamp(report["completed_at"])
        )
    )
    if newer_evaluation:
        evaluation_done = False
    training_state = "complete" if training_done else training.get("state", "unknown")
    evaluation_state = (
        "complete"
        if evaluation_done
        else evaluation.get("state", report.get("state", "not_started"))
    )
    if evaluation_state == "ready" and not evaluation_done:
        evaluation_state = "missing_result"
    if training_state == "ready" and not training_done:
        training_state = "missing_result"
    failures = {"failed", "timed_out", "waiting_for_resume", "interrupted", "missing_result"}
    active = {
        "running",
        "preparing",
        "finalizing",
        "validating",
        "training",
        "evaluating",
        "stopping",
    }
    if training_done and evaluation_done:
        state = "complete"
    elif evaluation_state in failures or (not training_done and training_state in failures):
        state = "incomplete"
    elif evaluation_state in active:
        state = "evaluating"
    elif training_done:
        state = "trained"
    elif training_state in active:
        state = "training"
    else:
        state = "not_started" if selection and not manifest and not training else "unknown"
    activity = [
        payload.get(key)
        for payload, key in (
            (manifest, "created_at"),
            (training, "generated_at"),
            (evaluation, "generated_at"),
            (completion, "completed_at"),
            (result, "created_at"),
            (report, "completed_at"),
            (wandb, "generated_at"),
        )
    ]
    latest = max((value for value in activity if value), key=stamp, default=None)
    return {
        "run_id": run_id,
        "state": state,
        "configuration": configured,
        "initialized_at": manifest.get("created_at"),
        "last_activity_at": latest,
        "training": {
            "state": training_state,
            "lifecycle_state": training.get("state"),
            "completed_at": (
                completion.get("completed_at")
                or result.get("created_at")
                or training.get("generated_at")
            )
            if training_done
            else None,
            "pod_id": training.get("pod_id"),
            "launch_id": training.get("launch_id"),
            "max_runtime_seconds": training.get("max_runtime_seconds"),
        },
        "evaluation": {
            "state": evaluation_state,
            "lifecycle_state": evaluation.get("state"),
            "split": report.get("evaluation_split") or report.get("selection_split"),
            "started_at": report.get("started_at"),
            "completed_at": report.get("completed_at") if evaluation_done else None,
            "pod_id": evaluation.get("pod_id"),
            "launch_id": evaluation.get("launch_id"),
        },
        "wandb": {
            "state": wandb.get("state"),
            "components": {
                name: value.get("state")
                for name, value in (wandb.get("components") or {}).items()
                if isinstance(value, dict)
            },
        },
    }


def worker_count(reader, requested):
    resources = runpy.run_path(str(reader.root / "src/stock_forecasting/runtime_resources.py"))
    try:
        memory = resources["detect_available_memory"]().available_bytes
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=10)
        page_size = int(re.search(r"page size of ([0-9]+) bytes", stats)[1])
        memory = page_size * sum(
            int(re.search(rf"Pages {name}:\s+([0-9]+)", stats)[1]) for name in ("free", "inactive")
        )
    # Each concurrent run has one AWS CLI process, with a 256 MiB allowance.
    workers = max(
        1,
        min(
            requested,
            8,
            resources["detect_visible_cpu_count"](),
            max(0, memory - 512 * 1024**2) // (256 * 1024**2),
        ),
    )
    if workers < requested:
        print(
            f"Run query workers limited to {workers}: CPU/memory/API cap; available_bytes={memory}",
            file=sys.stderr,
        )
    return workers


def inventory(reader, *, limit, offset, experiment, state, workers):
    runs = discover(reader)
    selected, errors, matched, scanned = [], [], 0, 0

    def query(run_id):
        try:
            return inspect_run(reader, run_id)
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            return {"run_id": run_id, "state": "error", "error": str(error), "configuration": {}}

    iterator = iter(runs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(iterator, workers)):
            # At most one request per worker; no unbounded future queue or metric retention.
            for item in pool.map(query, batch):
                scanned += 1
                if item["state"] == "error":
                    errors.append({"run_id": item["run_id"], "error": item["error"]})
                if experiment and item["configuration"].get("experiment") != experiment:
                    continue
                if state and item["state"] != state:
                    continue
                matched += 1
                if offset < matched <= offset + limit:
                    selected.append(item)
            if matched >= offset + limit:
                break
    return {
        "runs": selected,
        "discovered": len(runs),
        "scanned": scanned,
        "offset": offset,
        "next_offset": offset + len(selected)
        if scanned < len(runs) or matched > offset + limit
        else None,
        "errors": errors,
        "order": "run_allocation_newest_first",
    }


def value(item):
    return "unknown" if item is None or item == "" else str(item)


def print_run(item, zone, *, detail=False):
    def time_text(raw):
        return (
            stamp(raw).astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z%z") if raw else "unrecorded"
        )

    config = item["configuration"]
    print(
        f"{item['run_id']}  {item['state'].upper()}  experiment={value(config.get('experiment'))}"
    )
    if item.get("error"):
        print(f"  Error: {item['error']}")
        return
    train, evaluation = item["training"], item["evaluation"]
    print(
        f"  Training: {train['state'].upper()} | Final evaluation: {evaluation['state'].upper()} "
        f"(split={value(evaluation['split'])})"
    )
    print(
        f"  Initialized: {time_text(item['initialized_at'])} | "
        f"Last activity: {time_text(item['last_activity_at'])}"
    )
    dates = config.get("date_range") or {}
    universe = config.get("universe") or {}
    print(
        f"  Configure: stage={value(config.get('stage'))} "
        f"data-profile={value(config.get('data_profile'))} "
        f"dataset-revision={value(config.get('dataset_revision'))}"
    )
    print(
        f"    start={value(dates.get('start_inclusive'))} "
        f"end(exclusive)={value(dates.get('end_exclusive'))} "
        f"h-start={value(config.get('h_start'))} feature-mode={value(config.get('feature_mode'))}"
    )
    limit = universe.get("symbol_limit", "unknown")
    print(
        f"    universe={value(universe.get('mode'))} "
        f"symbol-limit={'none' if limit is None else limit} "
        f"include-delisted-us={value(universe.get('include_delisted_us'))}"
    )
    print(
        f"    stocks={','.join(universe.get('us_stocks') or []) or '-'} "
        f"etfs={','.join(universe.get('us_etfs') or []) or '-'}"
    )
    print(f"    config={value(config.get('config_path'))}")
    print(f"  Pods: training={value(train['pod_id'])} evaluation={value(evaluation['pod_id'])}")
    if detail:
        print(f"  Training completed: {time_text(train['completed_at'])}")
        print(
            f"  Evaluation started: {time_text(evaluation['started_at'])} | "
            f"Completed: {time_text(evaluation['completed_at'])}"
        )
        print(
            f"  Recorded lifecycle: training={value(train['lifecycle_state'])} "
            f"evaluation={value(evaluation['lifecycle_state'])}"
        )
        for name, phase in (("Training", train), ("Evaluation", evaluation)):
            print(f"  {name} Pod: {value(phase['pod_id'])} | Launch: {value(phase['launch_id'])}")
        print(
            f"  Max runtime: {value(train['max_runtime_seconds'])} seconds | "
            f"Config source: {config['source']}"
        )
        print(
            f"  Window: {value(config['window_size'])} | "
            f"Fixed split: {json.dumps(config['fixed_split'], sort_keys=True)}"
        )
        print(f"  LoRA: {json.dumps(config['lora'], sort_keys=True)}")
        print(f"  Unfrozen backbone blocks: {value(config['unfreeze_last_blocks'])}")
    components = item["wandb"]["components"]
    print(
        "  W&B: "
        + value(item["wandb"]["state"])
        + " "
        + " ".join(f"{k}={v}" for k, v in components.items())
    )


def main(reader, command, argv):
    parser = argparse.ArgumentParser(description=__doc__)
    if command == "status":
        parser.add_argument("run_id", nargs="?")
        parser.add_argument("--run-id", dest="explicit_run_id")
    parser.add_argument("--output", choices=("table", "json"), default="table")
    parser.add_argument("--timezone", default="Asia/Taipei")
    if command == "list":
        parser.add_argument("--experiment")
        parser.add_argument("--state", choices=STATES)
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        zone = ZoneInfo(args.timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"Unknown timezone: {args.timezone}") from error
    if command == "status" and (args.run_id is not None or args.explicit_run_id is not None):
        if args.run_id is not None and args.explicit_run_id is not None:
            parser.error("Specify the run ID once")
        run_id = args.run_id if args.run_id is not None else args.explicit_run_id
        item = inspect_run(reader, run_id)
        if args.output == "json":
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
        else:
            print_run(item, zone, detail=True)
        return 0 if item["state"] == "complete" else 1
    if command == "status":
        args.limit, args.offset, args.experiment, args.state, args.workers = 20, 0, None, None, 2
    if not 1 <= args.limit <= 200 or args.offset < 0 or not 1 <= args.workers <= 8:
        parser.error("--limit must be 1..200, --offset >= 0, --workers 1..8")
    result = inventory(
        reader,
        limit=args.limit,
        offset=args.offset,
        experiment=args.experiment,
        state=args.state,
        workers=worker_count(reader, args.workers),
    )
    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Runs: {len(result['runs'])} shown; {result['discovered']} discovered; "
            f"timezone={args.timezone}"
        )
        print(
            "Newest allocation first. COMPLETE = training + final evaluation; W&B sync is separate."
        )
        for item in result["runs"]:
            print_run(item, zone)
        if result["next_offset"] is not None:
            print(
                "More runs may match: repeat runs with the same filters "
                f"and --offset {result['next_offset']}"
            )
        for error in result["errors"]:
            print(f"Run query error: {error['run_id']}: {error['error']}", file=sys.stderr)
    return 2 if result["errors"] else 0
