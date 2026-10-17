#!/usr/bin/env python3
"""Compare frozen probability calibrators on time-separated validation data.

This CPU diagnostic uses only the standard library and memory-mapped prediction
artifacts. It never imports a model, evaluates test, changes checkpoints or runs
remote commands. Run with python3 -I; see docs/SIGNAL_DIAGNOSTICS.md.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import math  # noqa: E402
import struct  # noqa: E402

from stock_forecasting.runtime_resources import (  # noqa: E402
    detect_available_memory,
    detect_visible_cpu_count,
)
from stock_forecasting.signal_diagnostic_io import (  # noqa: E402
    NpyReader,
    atomic_json,
    inspect_membership,
    read_json,
    sha256,
    temporal_guard,
)
from stock_forecasting.signal_diagnostic_math import (  # noqa: E402
    ProbabilityMoments,
    distribution,
    fit_calibrator,
    label,
)

HORIZONS = 14
METHODS = ("raw", "fixed_train_prior", "temperature", "temperature_bias")
LOWER = struct.unpack("<f", struct.pack("<f", math.log1p(-0.01)))[0]
UPPER = struct.unpack("<f", struct.pack("<f", math.log1p(0.01)))[0]


def worker_plan(requested, fit_rows, memory_budget_mb):
    """Respect visible CPUs, cgroups and a quarter of available system memory."""
    if not 1 <= requested <= 8 or not 64 <= memory_budget_mb <= 4096:
        raise ValueError("workers must be 1-8; memory budget must be 64-4096 MiB")
    try:
        available = detect_available_memory().available_bytes
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        )
        page = re.search(r"page size of (\d+) bytes", result.stdout)
        pages = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)\.", result.stdout)
        if page is None or not pages:
            raise RuntimeError("Cannot establish a safe diagnostic memory budget") from None
        available = int(page[1]) * sum(map(int, pages))
    budget = min(memory_budget_mb * 1024**2, available // 4)
    per_worker = 80 * 1024**2 + fit_rows * HORIZONS * 200
    workers = min(requested, max(1, detect_visible_cpu_count() - 1), budget // per_worker)
    if workers < 1:
        raise MemoryError("Insufficient headroom for one bounded diagnostic worker")
    return {
        "workers": workers,
        "budget_bytes": budget,
        "estimated_bytes_per_worker": per_worker,
        "available_bytes": available,
        "max_pending_jobs": workers,
    }


def fit_indices(metadata, end, limit, seed):
    """Reservoir-sample cutoffs, keeping at most limit ordinals across all models."""
    rng, chosen, seen = random.Random(seed), [], 0
    for i in range(metadata.shape[0]):
        if metadata.text(i, "date")[:10] > end:
            continue
        seen += 1
        if len(chosen) < limit:
            chosen.append(i)
        else:
            replacement = rng.randrange(seen)
            if replacement < limit:
                chosen[replacement] = i
    if seen < 16:
        raise ValueError("Calibration segment has fewer than sixteen cutoff samples")
    return sorted(chosen), seen


def validate_prior(audit):
    if (
        audit.get("source") != "full_train"
        or audit.get("class_order") != ["down", "neutral", "up"]
        or audit.get("neutral_return_threshold") != 0.01
    ):
        raise ValueError("Require an unchanged full-train fixed ±1% class-label audit")
    priors = {}
    for market in ("US", "TWSE", "TPEX"):
        priors[market] = {}
        for h in range(1, 15):
            counts = audit["counts"][market][str(h)]
            if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
                raise ValueError("Every market/horizon train class needs positive support")
            priors[market][h] = [n / sum(counts) for n in counts]
    for h in range(1, 15):
        if sum(sum(audit["counts"][m][str(h)]) for m in priors) != audit["samples"]:
            raise ValueError("Train class counts disagree with the full population")
    return priors


def manifest_file(root, manifest, relative):
    path = root / relative
    record = manifest["artifacts"].get(relative)
    if not isinstance(record, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))):
        raise ValueError("Missing immutable artifact checksum: " + relative)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record.get("bytes"):
        raise ValueError("Artifact size/path mismatch: " + relative)
    return path, record["sha256"]


def exported_model(value, membership):
    name, separator, raw = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError("Use --model LABEL=EXACT_EXPORT_DIRECTORY")
    root = Path(raw).expanduser().resolve()
    report, status = read_json(root / "export.json"), read_json(root / "status.json")
    if (
        status.get("state") != "complete"
        or report.get("kind") != "signal-prediction-export"
        or report.get("prediction_task") != "absolute_direction"
        or report.get("horizons") != list(range(1, 15))
        or report.get("checkpoint", {}).get("selection") != "validation_best"
        or status.get("checkpoint") != report.get("checkpoint", {}).get("path")
    ):
        raise ValueError("Calibration requires a completed validation-selected direction export")
    split = report["splits"].get("validation", {})
    if split.get("sample_membership") != membership:
        raise ValueError("Export and baseline validation memberships differ")
    digest = split.get("predictions_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise ValueError("Export has no valid prediction checksum")
    return {
        "name": name,
        "predictions": str(root / "validation/predictions.npy"),
        "sha256": digest,
        "export_manifest": str(root / "export.json"),
        "checkpoint": report["checkpoint"],
        "manifest_sha256": sha256(root / "export.json"),
    }


def calibrate_job(job):
    """One process maps one model; no raw dataset or model is copied to workers."""
    with ExitStack() as stack:
        meta = stack.enter_context(NpyReader(job["metadata"]))
        targets = stack.enter_context(NpyReader(job["targets"], (14,)))
        prediction = stack.enter_context(NpyReader(job["predictions"], (14, 3)))
        if not meta.shape[0] == targets.shape[0] == prediction.shape[0]:
            raise ValueError("Prediction/label/metadata counts differ")
        points = []
        for i in job["fit_indices"]:
            y, z = targets.row(i), prediction.row(i)
            points.extend(
                (z[h * 3 : h * 3 + 3], label(y[h], LOWER, UPPER)) for h in range(HORIZONS)
            )
        parameters = {
            "temperature": fit_calibrator(points, max_iterations=job["max_iterations"]),
            "temperature_bias": fit_calibrator(
                points,
                with_bias=True,
                penalty=job["bias_penalty"],
                max_iterations=job["max_iterations"],
            ),
        }
        del points
        cells = {method: defaultdict(ProbabilityMoments) for method in METHODS}
        months = {method: defaultdict(ProbabilityMoments) for method in METHODS}
        samples = 0
        for i in range(meta.shape[0]):
            day = meta.text(i, "date")[:10]
            if day < job["score_start"]:
                continue
            market, y, logits = meta.text(i, "market"), targets.row(i), prediction.row(i)
            samples += 1
            for h in range(HORIZONS):
                target = label(y[h], LOWER, UPPER)
                z = logits[h * 3 : h * 3 + 3]
                prior = job["priors"][market][h + 1]
                variants = {
                    "raw": distribution(z),
                    "fixed_train_prior": (prior, [math.log(p) for p in prior]),
                }
                for method, fitted in parameters.items():
                    variants[method] = distribution(
                        z, fitted["inverse_temperature"], fitted["bias"]
                    )
                for method, (p, logp) in variants.items():
                    cells[method][f"{market}/{h + 1}d"].add(p, target, logp[target])
                    months[method][f"{day[:7]}/{h + 1}d"].add(p, target, logp[target])
        if not samples:
            raise ValueError("Empty later validation scoring segment")
        results = {}
        for method in METHODS:
            by_cell = {key: value.result() for key, value in sorted(cells[method].items())}
            by_horizon = {}
            by_market = {}
            for key, value in cells[method].items():
                market, horizon = key.split("/")
                by_horizon.setdefault(horizon, ProbabilityMoments()).merge(value)
                by_market.setdefault(market, []).append(value.result())
            per_horizon = {key: value.result() for key, value in by_horizon.items()}
            metric_keys = (
                "cross_entropy",
                "brier_score",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "expected_calibration_error",
            )
            aggregate = {
                key: sum(row[key] for row in per_horizon.values()) / HORIZONS for key in metric_keys
            }
            market_macro = {
                key: sum(
                    sum(row[key] for row in values) / HORIZONS for values in by_market.values()
                )
                / len(by_market)
                for key in metric_keys
            }
            results[method] = {
                **aggregate,
                "per_horizon": per_horizon,
                "market_macro": market_macro,
                "market_horizon": by_cell,
                "month_horizon": {
                    key: value.result() for key, value in sorted(months[method].items())
                },
            }
        return {
            "name": job["name"],
            "score_samples": samples,
            "parameters": parameters,
            "methods": results,
        }


def run_jobs(jobs, workers, timeout, output_directory=None):
    """Bound submission and terminate all workers on failure, timeout or cancellation."""
    output, pending, iterator = [], [], iter(jobs)
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                job = next(iterator, None)
                if job is None:
                    exhausted = True
                else:
                    pending.append(pool.apply_async(calibrate_job, (job,)))
            if pending:
                result = pending.pop(0).get(timeout=timeout)
                if output_directory is not None:
                    path = output_directory / f"model-{result['name']}.json"
                    atomic_json(path, result)
                    output.append(
                        {
                            "name": result["name"],
                            "score_samples": result["score_samples"],
                            "parameters": result["parameters"],
                            "details_file": path.name,
                            "details_sha256": sha256(path),
                            "methods": {
                                method: {
                                    key: value
                                    for key, value in metrics.items()
                                    if key
                                    in (
                                        "cross_entropy",
                                        "brier_score",
                                        "accuracy",
                                        "balanced_accuracy",
                                        "macro_f1",
                                        "expected_calibration_error",
                                        "market_macro",
                                    )
                                }
                                for method, metrics in result["methods"].items()
                            },
                        }
                    )
                else:
                    output.append(result)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Exact direction baseline cache directory containing complete.json",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="LABEL=exact completed export directory; up to 16, plus shared GRU/GBDT",
    )
    p.add_argument("--calibration-end", required=True, help="Last fitting cutoff date, YYYY-MM-DD")
    p.add_argument(
        "--score-start",
        required=True,
        help="First later validation cutoff; at least 14 observed market dates after fitting",
    )
    p.add_argument("--output", type=Path, required=True, help="New immutable diagnostic directory")
    p.add_argument("--fit-rows", type=int, default=2048)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SIGNAL_DIAGNOSTIC_WORKERS", "2"))
    )
    p.add_argument("--memory-budget-mb", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=80)
    p.add_argument("--bias-penalty", type=float, default=0.1)
    p.add_argument("--job-timeout", type=int, default=3600)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        not 16 <= args.fit_rows <= 16384
        or not 1 <= args.max_iterations <= 500
        or not 1 <= args.job_timeout <= 21600
        or len(args.model) > 16
        or not math.isfinite(args.bias_penalty)
        or args.bias_penalty <= 0
    ):
        raise ValueError("Invalid fitting, timeout or model bounds")
    root = args.baseline_root.expanduser().resolve()
    manifest = read_json(root / "complete.json")
    if manifest.get("state") != "complete":
        raise ValueError("Require a complete immutable baseline manifest")
    priors = validate_prior(manifest["class_label_audit"])
    membership = manifest["validation_membership"]
    records = [
        manifest_file(root, manifest, "inputs/validation/" + name)
        for name in ("metadata.npy", "absolute_targets.npy")
    ]
    models = []
    for name in ("gru-42", "gbdt-42"):
        path, digest = manifest_file(root, manifest, f"jobs/{name}/validation/predictions.npy")
        models.append({"name": name, "predictions": str(path), "sha256": digest})
    models.extend(exported_model(value, membership) for value in args.model)
    if len({m["name"] for m in models}) != len(models) or len(
        {m["predictions"] for m in models}
    ) != len(models):
        raise ValueError("Model labels and input paths must be distinct")
    checks = records + [(Path(m["predictions"]), m["sha256"]) for m in models]
    # A small thread pool bounds simultaneous transport/checksum buffers.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for (path, expected), actual in zip(
            checks, pool.map(sha256, [p for p, _ in checks]), strict=True
        ):
            if actual != expected:
                raise ValueError("Artifact checksum mismatch: " + str(path))
    with NpyReader(records[0][0]) as metadata:
        digest, calendars = inspect_membership(metadata)
        if (
            digest != membership["ordered_symbol_dates_sha256"]
            or metadata.shape[0] != membership["samples"]
        ):
            raise ValueError("Metadata is not the declared validation membership")
        gap = temporal_guard(calendars, args.calibration_end, args.score_start)
        if len({day[:7] for days in calendars.values() for day in days}) > 12:
            raise ValueError("Bound this validation diagnostic to at most twelve months")
        indices, eligible = fit_indices(metadata, args.calibration_end, args.fit_rows, args.seed)
    resources = worker_plan(args.workers, args.fit_rows, args.memory_budget_mb)
    if resources["workers"] == 1:
        print(
            "Diagnostic uses one process because the requested CPU/memory budget permits only one.",
            file=sys.stderr,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": "1.0",
        "kind": "signal-validation-calibration-protocol",
        "split": "validation",
        "test_opened": False,
        "membership": membership,
        "baseline_manifest": str(root / "complete.json"),
        "baseline_manifest_sha256": sha256(root / "complete.json"),
        "models": models,
        "methods": list(METHODS),
        "calibration_end": args.calibration_end,
        "score_start": args.score_start,
        "conservative_label_end_guard": gap,
        "fit_indices": indices,
        "fit_eligible_rows": eligible,
        "fit_policy": "uniform early-validation cutoffs; equal fourteen horizons",
        "prior_source": "unweighted full cleaned train, never fitted on validation/test",
        "report_month_limit": 12,
        "seed": args.seed,
        "bias_penalty": args.bias_penalty,
        "max_iterations": args.max_iterations,
        "resources": resources,
        "selection_caveat": (
            "Existing checkpoints were selected using full validation; "
            "this temporal diagnostic is not an independent confirmation set."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__),
                ROOT / "src/stock_forecasting/signal_diagnostic_io.py",
                ROOT / "src/stock_forecasting/signal_diagnostic_math.py",
            )
        },
    }
    atomic_json(args.output / "protocol.json", protocol)
    jobs = [
        dict(
            m,
            metadata=str(records[0][0]),
            targets=str(records[1][0]),
            fit_indices=indices,
            priors=priors,
            score_start=args.score_start,
            max_iterations=args.max_iterations,
            bias_penalty=args.bias_penalty,
        )
        for m in models
    ]
    try:
        results = run_jobs(jobs, resources["workers"], args.job_timeout, args.output)
        atomic_json(
            args.output / "comparison.json",
            {
                "protocol_sha256": sha256(args.output / "protocol.json"),
                "results": results,
                "automatic_winner": None,
                "weights_updated": False,
            },
        )
        atomic_json(args.output / "status.json", {"state": "complete"})
    except BaseException as error:
        atomic_json(
            args.output / "status.json", {"state": "failed", "error_type": type(error).__name__}
        )
        raise
    print(json.dumps({"state": "complete", "output": str(args.output), "models": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
