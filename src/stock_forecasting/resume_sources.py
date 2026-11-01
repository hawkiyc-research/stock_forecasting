"""Resume sources of the walk-forward trainers (LightGBM and cross-stock).

A Pod restart loses the job but not the network volume, so each trainer writes every finished
fold to ``<output root>/partial/`` and a later launch reuses them. The later launch writes its
own ``results.json``; the source keeps its ``partial/`` and never gets one, so without a stamp
every later Pod job would offer it again as an unfinished launch. ``stamp_resume_source`` writes
``resumed_into.json`` into a source once a launch has absorbed folds from it; the Pod jobs skip
stamped launches and the volume reset keeps them (they hold part of a finished result's record).

Jobs may pass several ``--resume-root`` candidates (every unfinished, unstamped launch on the
volume); the trainer decides by content which folds belong to the current run.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.manifest import atomic_write_json

RESUMED_INTO_NAME = "resumed_into.json"


def resume_candidates(roots: Path | str | Iterable[Path | str] | None) -> list[Path]:
    """Usable resume sources, newest launch name first: existing directories with a
    ``partial/`` that no earlier launch has absorbed yet; duplicates dropped."""
    if roots is None:
        return []
    if isinstance(roots, (str, Path)):
        roots = [roots]
    seen: set[Path] = set()
    candidates = []
    for root in roots:
        path = Path(root)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not (path / "partial").is_dir() or (path / RESUMED_INTO_NAME).is_file():
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda path: path.name, reverse=True)


def stamp_resume_source(
    source: Path, *, resumed_into: Path, keys: list[Any], **details: Any
) -> dict[str, Any]:
    """Record in ``source`` which launch absorbed which of its folds; returns the payload."""
    payload = {
        "resumed_into": Path(resumed_into).name,
        "keys": keys,
        **details,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(Path(source) / RESUMED_INTO_NAME, payload)
    return payload


def resume_summary(contributions: dict[Path, list[Any]]) -> list[dict[str, Any]] | None:
    """The ``resume`` entry of a results file: which source supplied which folds."""
    if not contributions:
        return None
    return [
        {"root": source.name, "keys": keys}
        for source, keys in sorted(contributions.items(), key=lambda item: item[0].name)
    ]
