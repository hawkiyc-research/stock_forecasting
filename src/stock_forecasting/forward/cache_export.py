"""Export a local forward fetch cache for the shared provider cache on the network volume.

A cache filled before month scopes existed keys benchmark months by request only. Copied
to the volume, such an entry gets a new file time and could later pass as a complete
month, so a month entry is exported only when its file proves it was fetched after the
month ended: then under both the complete-month key (forward fetches) and its original
key (the dataset preparation). Unfinished or unproven months are left out. Every other
response (daily quotes, corporate actions) keeps its key. The month requests are replayed
through the providers against the local cache only; a missing response fails instead of
fetching.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.data.providers.cache_scope import (
    TAIWAN_MONTH_ENDPOINTS,
    TaiwanMonthScope,
    taiwan_month_scope,
)
from stock_forecasting.data.providers.http import CachedJsonClient
from stock_forecasting.forward.fetch import DATASET_PROFILE, PROVIDERS, load_fetch

EXPORT_KIND = "forward-shared-cache-export"
EXPORT_SCHEMA_VERSION = "1.0"
MANIFEST_FILE = "shared-cache.json"
CACHE_ENTRY = re.compile(r"[a-z_]+/[0-9a-f]{2}/[0-9a-f]{64}\.json")


class _OfflineSession:
    def get(self, endpoint: str, **_kwargs: Any) -> requests.Response:
        raise RuntimeError(f"Response is missing from the local cache: {endpoint}")


class _CacheReplayClient:
    """Serve provider requests from one local cache only and record month requests."""

    def __init__(self, provider: str, cache_root: Path, scope: TaiwanMonthScope) -> None:
        def client(cache_scope: TaiwanMonthScope | None) -> CachedJsonClient:
            return CachedJsonClient(
                provider=provider,
                raw_cache_root=cache_root,
                max_requests_per_second=1000.0,
                max_attempts=1,
                session=_OfflineSession(),
                cache_scope=cache_scope,
            )

        self.unscoped = client(None)
        self.scoped = client(scope)
        self.months: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, endpoint: str, *, params: Mapping[str, Any]) -> tuple[Any, Any]:
        if endpoint in TAIWAN_MONTH_ENDPOINTS:
            self.months.append((endpoint, dict(params)))
        legacy, _identity = self.unscoped._identity(endpoint=endpoint, params=params)
        if self.unscoped._load_cached_response(legacy) is not None:
            return self.unscoped.get_json(endpoint, params=params)
        return self.scoped.get_json(endpoint, params=params)


def _month_targets(
    cache_root: Path, fetch: dict[str, Any], end_exclusive: str
) -> tuple[dict[str, str], set[str]]:
    """Complete-month keys for unscoped month entries, and the unfinished ones to leave out."""
    scope = taiwan_month_scope(end_exclusive)
    rekeyed: dict[str, str] = {}
    skipped: set[str] = set()
    for name, provider_class in PROVIDERS.items():
        replay = _CacheReplayClient(name, cache_root, scope)
        provider = provider_class(replay)
        for month in fetch["providers"][name]["benchmark_months"]:
            provider.fetch_benchmark_month(month=month, dataset_profile=DATASET_PROFILE)
        for endpoint, params in replay.months:
            legacy, _identity = replay.unscoped._identity(endpoint=endpoint, params=params)
            relative = f"{name}/{legacy[:2]}/{legacy}.json"
            if not (cache_root / relative).is_file():
                # Already cached under its scope key; copied unchanged.
                continue
            written_at = datetime.fromtimestamp((cache_root / relative).stat().st_mtime, tz=UTC)
            if scope.reuses_unscoped(endpoint, params, written_at):
                scoped, _identity = replay.scoped._identity(endpoint=endpoint, params=params)
                rekeyed[relative] = f"{name}/{scoped[:2]}/{scoped}.json"
            else:
                skipped.add(relative)
    return rekeyed, skipped


def export_shared_cache(
    fetch_root: str | Path, output_root: str | Path, *, end_exclusive: str
) -> dict[str, Any]:
    """Write a re-keyed copy of the fetch cache and ``shared-cache.json`` under output_root."""
    fetch_root = Path(fetch_root)
    fetch = load_fetch(fetch_root)
    if fetch["end_exclusive"] != end_exclusive:
        raise ValueError("The fetch end date differs from --end")
    cache_root = Path((fetch.get("cache") or {}).get("root") or fetch_root / "api-cache")
    output = Path(output_root)
    if (output / MANIFEST_FILE).exists():
        raise FileExistsError(f"Shared-cache export already exists: {output}")
    rekeyed, skipped = _month_targets(cache_root, fetch, end_exclusive)
    files: dict[str, str] = {}
    for path in sorted(cache_root.rglob("*.json")):
        relative = path.relative_to(cache_root).as_posix()
        if CACHE_ENTRY.fullmatch(relative) is None or path.is_symlink() or relative in skipped:
            continue
        for target_relative in (relative, rekeyed.get(relative)):
            if target_relative is None:
                continue
            target = output / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                # Same filesystem: a hard link avoids a second copy of the cache.
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
            files[target_relative] = sha256_file(target)
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "kind": EXPORT_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "end_exclusive": end_exclusive,
        "source_cache": str(cache_root),
        "rekeyed_complete_months": len(rekeyed),
        "skipped_unfinished_months": sorted(skipped),
        "files": files,
    }
    atomic_write_json(output / MANIFEST_FILE, manifest)
    return manifest
