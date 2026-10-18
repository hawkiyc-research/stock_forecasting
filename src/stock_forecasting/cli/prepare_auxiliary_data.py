"""Fetch auxiliary Taiwan research data (institutional flows, margin, TAIFEX, US series).

Runs on a RunPod CPU Pod. Responses are cached under the raw cache root, so a
rerun only requests what is missing. The tables are research inputs only and are
never part of the training dataset identity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from stock_forecasting.data.auxiliary import (
    DEFAULT_EODHD_SERIES,
    SOURCES,
    AuxiliaryOptions,
    prepare_auxiliary_data,
)


def _default_output_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "/runpod-volume/data")) / "auxiliary"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--start",
        default=os.environ.get("STAGE1_DATA_START"),
        help="Inclusive first date (YYYY-MM-DD); defaults to STAGE1_DATA_START.",
    )
    parser.add_argument(
        "--end",
        default=os.environ.get("STAGE1_DATA_END"),
        help="Exclusive end date (YYYY-MM-DD); defaults to STAGE1_DATA_END.",
    )
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument(
        "--raw-cache-root",
        type=Path,
        help="Response cache; defaults to OUTPUT_ROOT/raw-cache.",
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument(
        "--eodhd-series",
        nargs="*",
        default=list(DEFAULT_EODHD_SERIES),
        help="EODHD EOD symbols to try; entitlement failures are recorded, not fatal.",
    )
    parser.add_argument(
        "--taiwan-qps",
        type=float,
        default=float(os.environ.get("RUNPOD_CPU_TAIWAN_QPS") or 0.5),
    )
    parser.add_argument("--taifex-qps", type=float, default=0.5)
    parser.add_argument("--eodhd-qps", type=float, default=2.0)
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=float(os.environ.get("RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS") or 60),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.start or not args.end:
        raise SystemExit("Pass --start and --end, or set STAGE1_DATA_START and STAGE1_DATA_END")
    options = AuxiliaryOptions(
        start=args.start,
        end=args.end,
        output_root=args.output_root,
        raw_cache_root=args.raw_cache_root or args.output_root / "raw-cache",
        sources=tuple(dict.fromkeys(args.sources)),
        eodhd_series=tuple(dict.fromkeys(args.eodhd_series)),
        taiwan_requests_per_second=args.taiwan_qps,
        taifex_requests_per_second=args.taifex_qps,
        eodhd_requests_per_second=args.eodhd_qps,
        max_backoff_seconds=args.max_backoff_seconds,
        tpex_proxy_url=os.environ.get("TPEX_PROXY_URL") or None,
        tpex_proxy_token=os.environ.get("TPEX_PROXY_TOKEN") or None,
        eodhd_api_token=os.environ.get("EODHD_API_TOKEN") or None,
    )
    manifest = prepare_auxiliary_data(options)
    summary = {name: source["status"] for name, source in manifest["sources"].items()}
    print(json.dumps({"output_root": str(args.output_root), "sources": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
