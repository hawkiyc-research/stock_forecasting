"""Local steps of the independent forward-inference pipeline (protocol 13.8 supplement).

    fetch           official Taiwan bars from the fetch start to --end (exclusive)
    build           sealed input package: private bar store and cleaned sample index
    labels          realized labels of the overlap and forward populations
    compare-inputs  data check 4: overlap inputs against the shared evaluation snapshot

Everything is written below ARTIFACTS/forward/<end>/. GPU inference runs on RunPod with
``export-signal-predictions --forward-package``; no training dataset is created or read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

# Each cleaning or labeling process holds a few symbols' bars plus pandas overhead.
WORKER_MEMORY_BYTES = 1024**3
MAX_WORKERS = 16
INPUT_TOLERANCE = 1e-6
INPUT_MIN_FRACTION = 0.99


def default_workers() -> int:
    """Bound build processes by visible CPUs (one kept free) and half of free memory."""
    configured = os.environ.get("FORWARD_PIPELINE_WORKERS")
    if configured:
        workers = int(configured)
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"FORWARD_PIPELINE_WORKERS must be 1-{MAX_WORKERS}")
        return workers
    by_memory = detect_available_memory().available_bytes // 2 // WORKER_MEMORY_BYTES
    return max(1, min(MAX_WORKERS, detect_visible_cpu_count() - 1, by_memory))


def _root(args: argparse.Namespace) -> Path:
    return args.artifacts / "forward" / args.end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("fetch", "Fetch official Taiwan bars"),
        ("build", "Build the sealed forward input package"),
        ("labels", "Write overlap and forward labels"),
        ("compare-inputs", "Data check 4 against the shared evaluation snapshot"),
    ):
        command = commands.add_parser(name, help=text, allow_abbrev=False)
        command.add_argument(
            "--end", required=True, metavar="YYYY-MM-DD",
            help="Exclusive end date; must equal data.end_exclusive in backtest/config/stage3.json",
        )
        command.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        if name == "fetch":
            from stock_forecasting.forward.fetch import (
                DEFAULT_REQUESTS_PER_SECOND,
                DEFAULT_START,
            )

            command.add_argument("--start", default=DEFAULT_START, metavar="YYYY-MM-DD")
            command.add_argument(
                "--qps", type=float, default=DEFAULT_REQUESTS_PER_SECOND,
                help="Requests per second per exchange (default: CPU preparation pacing)",
            )
            command.add_argument(
                "--workers", type=int, default=2, help="Concurrent requests per exchange"
            )
            command.add_argument(
                "--shared-cache", type=Path, metavar="VOLUME/datasets/NAMESPACE/api-cache",
                help=(
                    "Provider cache on the network volume; every other datasets/*/api-cache "
                    "is read first (default: a private cache under the fetch directory)"
                ),
            )
        if name == "build":
            command.add_argument("--workers", type=int, help="Build processes (default: auto)")
        if name == "compare-inputs":
            command.add_argument(
                "--reference", type=Path, required=True,
                help="artifacts/diagnostics/evaluation-data/<dataset>/prepared",
            )
            command.add_argument("--tolerance", type=float, default=INPUT_TOLERANCE)
            command.add_argument("--min-fraction", type=float, default=INPUT_MIN_FRACTION)
    return parser


def _reference_test(prepared: Path):
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset

    universes = sorted(path for path in (prepared / "sample-universes").iterdir() if path.is_dir())
    if len(universes) != 1:
        raise SystemExit(f"Expected one sample universe under {prepared / 'sample-universes'}")
    return LazyFinancialWindowDataset(
        prepared / "bar-store",
        split="test",
        window_size=128,
        h_start=1,
        sample_index_root=universes[0],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root(args)
    if args.command == "fetch":
        from stock_forecasting.data.providers.http import TpexRelayTransport
        from stock_forecasting.forward.fetch import fetch_taiwan

        # Same relay settings as the CPU preparation; both must be set together.
        relay_url = os.environ.get("TPEX_PROXY_URL") or None
        relay_token = os.environ.get("TPEX_PROXY_TOKEN") or None
        if (relay_url is None) != (relay_token is None):
            raise SystemExit("TPEX_PROXY_URL and TPEX_PROXY_TOKEN must be configured together")
        manifest = fetch_taiwan(
            output_root=root / "fetch",
            start=args.start,
            end_exclusive=args.end,
            requests_per_second=args.qps,
            workers=args.workers,
            cache_root=args.shared_cache,
            shared_cache=args.shared_cache is not None,
            tpex_transport=(
                TpexRelayTransport(origin=relay_url, token=relay_token)
                if relay_url is not None
                else None
            ),
        )
        print(json.dumps({key: manifest[key] for key in ("raw", "calendar")}, indent=1))
        if any(manifest["calendar"].values()):
            print(
                "WARNING: official sessions differ from the cleaning calendar. build applies "
                "differences from 2026-06-01 to the forward package only and refuses earlier "
                "ones. Report this before building the package.",
                file=sys.stderr,
            )
        return 0
    if args.command == "build":
        from stock_forecasting.forward.fetch import load_fetch
        from stock_forecasting.forward.package import build_package, calendar_additions_from_fetch

        fetch = load_fetch(root / "fetch")
        if fetch["end_exclusive"] != args.end:
            raise SystemExit("The fetch end date differs from --end")
        try:
            additions = calendar_additions_from_fetch(fetch)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        payload = build_package(
            root / "fetch" / fetch["raw"]["relative_path"],
            root / "package",
            end_exclusive=args.end,
            workers=args.workers or default_workers(),
            fetch_manifest=fetch,
            calendar_additions=additions,
        )
        print(json.dumps(
            {
                "package": str(root / "package"),
                "package_sha256": payload["package_sha256"],
                "calendar_additions": payload["calendar_additions"],
                "populations": {
                    name: values["samples"] for name, values in payload["populations"].items()
                },
            },
            indent=1,
        ))
        return 0
    if args.command == "labels":
        from stock_forecasting.forward.labels import write_population_labels
        from stock_forecasting.forward.package import POPULATION_SLOTS, open_population

        # labels/inputs/<population>/ mirrors <baseline>/inputs/<split>/ for the backtest.
        for population in POPULATION_SLOTS:
            dataset = open_population(root / "package", population, h_start=1)
            summary = write_population_labels(
                dataset, root / "labels" / "inputs" / population, population=population
            )
            print(json.dumps({population: summary["sample_membership"]}, indent=1))
        return 0
    from stock_forecasting.data.manifest import atomic_write_json
    from stock_forecasting.forward.compare import compare_inputs
    from stock_forecasting.forward.package import load_package, open_population

    package = load_package(root / "package")
    result = compare_inputs(
        _reference_test(args.reference),
        open_population(root / "package", "overlap", h_start=1, verify_files=False),
        tolerance=args.tolerance,
    )
    result["min_fraction"] = args.min_fraction
    result["package_sha256"] = package["package_sha256"]
    result["passed"] = bool(
        result["within_tolerance_fraction"] is not None
        and result["within_tolerance_fraction"] >= args.min_fraction
    )
    output = root / "checks" / "input-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
