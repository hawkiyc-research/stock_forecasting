"""Forward input packages: a private bar store and cleaned sample index.

The package reuses ``build_symbol_bar_store`` unchanged and builds the sample index with
``forward.universe`` (``ensure_sample_universe`` plus forward-only session changes).
Those builders know only train/validation/test, so the package fixes the boundaries at
2025-12-01, 2026-06-01 and the end date: the ``overlap`` population (the original test
definition) occupies the validation slot and the ``forward`` population the test slot.
Every caller addresses populations by name; the slot mapping stays inside this module.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.bar_store import DEFAULT_MAX_HORIZON, build_symbol_bar_store
from stock_forecasting.data.content_identity import bar_store_materialization_digest
from stock_forecasting.data.manifest import (
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from stock_forecasting.data_policy import load_data_policy
from stock_forecasting.forward.universe import ensure_forward_sample_universe

PACKAGE_SCHEMA_VERSION = "1.0"
PACKAGE_KIND = "forward-input-package"
PACKAGE_FILE = "forward-package.json"
OVERLAP_START = "2025-12-01"
FORWARD_START = "2026-06-01"
WINDOW_SIZE = 128
# Population name -> slot of the reused fixed-split builders.
POPULATION_SLOTS = {"overlap": "validation", "forward": "test"}


def _date(value: str, name: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a YYYY-MM-DD date") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be a YYYY-MM-DD date")
    return value


def population_bounds(end_exclusive: str) -> dict[str, dict[str, str]]:
    """Signal-date windows; labels must also end before each window's upper bound."""
    _date(end_exclusive, "end_exclusive")
    if end_exclusive <= FORWARD_START:
        raise ValueError(f"end_exclusive must be after {FORWARD_START}")
    return {
        "overlap": {"start_inclusive": OVERLAP_START, "end_exclusive": FORWARD_START},
        "forward": {"start_inclusive": FORWARD_START, "end_exclusive": end_exclusive},
    }


def _package_files(root: Path) -> dict[str, str]:
    """SHA-256 of every durable package file; build scratch and locks are excluded."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or relative.as_posix() == PACKAGE_FILE
            or ".work" in relative.parts
            or "parts" in relative.parts
            or path.name.endswith(".lock")
        ):
            continue
        files[relative.as_posix()] = sha256_file(path)
    return files


def calendar_additions_from_fetch(fetch_manifest: dict[str, Any]) -> dict[str, Any]:
    """Official sessions that differ from the pinned calendar, allowed only from 2026-06-01."""
    differences = fetch_manifest["calendar"]
    closed = list(differences["calendar_sessions_without_trading"])
    opened = list(differences["trading_days_missing_from_calendar"])
    early = sorted(day for day in closed + opened if day < FORWARD_START)
    if early:
        raise ValueError(
            f"Official sessions differ from the pinned calendar before {FORWARD_START}: "
            f"{', '.join(early)}; the protocol allows additions only from {FORWARD_START}"
        )
    if not closed and not opened:
        return {}
    return {load_data_policy()["calendars"]["TWSE"]: {"open": opened, "closed": closed}}


def build_package(
    raw_path: str | Path,
    output_root: str | Path,
    *,
    end_exclusive: str,
    workers: int = 1,
    fetch_manifest: dict[str, Any] | None = None,
    calendar_additions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the store and sample index for both populations, then seal the package."""
    bounds = population_bounds(end_exclusive)
    for changes in (calendar_additions or {}).values():
        if any(day < FORWARD_START for days in changes.values() for day in days):
            raise ValueError(f"Calendar additions are allowed only from {FORWARD_START}")
    raw = Path(raw_path).resolve(strict=True)
    root = Path(output_root).resolve(strict=False)
    if (root / PACKAGE_FILE).exists():
        raise FileExistsError(f"Forward package already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as parquet

    rows = parquet.ParquetFile(raw).metadata.num_rows
    store = root / "bar-store"
    result = build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=raw.parent, row_count=rows)}
        },
        window_size=WINDOW_SIZE,
        max_horizon=DEFAULT_MAX_HORIZON,
        fixed_split={
            "train_end": OVERLAP_START,
            "validation_end": FORWARD_START,
            "test_end": end_exclusive,
        },
        workers=workers,
    )
    universe = ensure_forward_sample_universe(
        store, workers=workers, calendar_additions=calendar_additions
    )
    marker = json.loads((universe / "universe.json").read_text())
    populations = {
        name: {
            **bounds[name],
            "slot": slot,
            "samples": int(marker["split_counts"][slot]),
            "audit": marker["audit"].get(slot, {}),
        }
        for name, slot in POPULATION_SLOTS.items()
    }
    payload = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": PACKAGE_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "end_exclusive": end_exclusive,
        "window_size": WINDOW_SIZE,
        "max_horizon": DEFAULT_MAX_HORIZON,
        "populations": populations,
        "raw": {"sha256": sha256_file(raw), "rows": rows},
        "fetch_manifest": fetch_manifest,
        "bar_store": {
            "relative_path": store.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(result.bar_store_manifest_path),
            "materialization_digest": bar_store_materialization_digest(),
        },
        "sample_universe": {
            "relative_path": universe.relative_to(root).as_posix(),
            "identity_sha256": canonical_json_sha256(marker["identity"]),
            "ranges_sha256": marker["ranges_sha256"],
        },
        "data_cleaning": load_data_policy(),
        # Forward-only exchange announcements (protocol 13.8 amendment, item 1).
        "calendar_additions": calendar_additions or {},
        "files": _package_files(root),
    }
    payload["package_sha256"] = canonical_json_sha256(payload["files"])
    atomic_write_json(root / PACKAGE_FILE, payload)
    return payload


def load_package(package_root: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Read a sealed package; optionally re-hash every file before it is used."""
    root = Path(package_root).resolve(strict=True)
    payload = json.loads((root / PACKAGE_FILE).read_text())
    if (
        payload.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or payload.get("kind") != PACKAGE_KIND
    ):
        raise ValueError(f"Not a forward input package: {root}")
    if set(payload.get("populations", {})) != set(POPULATION_SLOTS) or any(
        payload["populations"][name].get("slot") != slot
        for name, slot in POPULATION_SLOTS.items()
    ):
        raise ValueError("Forward package populations differ from the fixed slot mapping")
    if payload.get("package_sha256") != canonical_json_sha256(payload.get("files", {})):
        raise ValueError("Forward package file list does not match its digest")
    if verify_files and _package_files(root) != payload["files"]:
        raise ValueError("Forward package files changed after the package was sealed")
    return payload


def open_population(
    package_root: str | Path,
    population: str,
    *,
    h_start: int,
    symbol_cache_size: int = 32,
    verify_files: bool = True,
):
    """Open one population with the unchanged lazy dataset used by validation/test."""
    from stock_forecasting.data.dataset import LazyFinancialWindowDataset

    if population not in POPULATION_SLOTS:
        raise ValueError(f"Population must be one of {sorted(POPULATION_SLOTS)}")
    root = Path(package_root).resolve(strict=True)
    payload = load_package(root, verify_files=verify_files)
    return LazyFinancialWindowDataset(
        root / payload["bar_store"]["relative_path"],
        split=POPULATION_SLOTS[population],
        window_size=payload["window_size"],
        h_start=h_start,
        symbol_cache_size=symbol_cache_size,
        sample_index_root=root / payload["sample_universe"]["relative_path"],
    )
