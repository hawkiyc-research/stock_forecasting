"""The Pod-side audit finds wrong stores, dead half-builds and unreachable launches, and keeps
every resume source."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_invalid_chip_results.py"
spec = importlib.util.spec_from_file_location("runpod_invalid_chip_results", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

DATASET = "ds"


def _store(
    volume: Path, digest: str, *, chip: bool, margin: int, state="ready", layout="features",
    universe: str | None = None,
):
    root = volume / "datasets" / DATASET / "prepared" / layout / digest
    root.mkdir(parents=True)
    identity = {"spec": {"name": "chip_tw_v1" if chip else "alpha_tw_v1"}}
    if universe is not None:
        identity["sample_universe"] = universe
    payload = {"state": state, "identity": identity}
    if chip:
        identity["chip_data"] = {"manifests": {}}
        payload["chip_coverage"] = {
            "symbols_with_margin": margin,
            "symbols_with_institutional": margin,
        }
    manifest = "features.json" if layout == "features" else "rank-labels.json"
    (root / manifest).write_text(json.dumps(payload))
    (root / "bucket-000.parquet").write_bytes(b"x")
    return root


def _bar_store(volume: Path) -> str:
    manifest = volume / "datasets" / DATASET / "prepared" / "bar-store" / "bar-store.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"kind": "symbol-oriented-ohlcv-bar-store"}))
    return module._sha256_file(manifest)


def _universe(volume: Path, key: str, policy: dict, bar_digest: str, state="ready") -> Path:
    root = volume / "datasets" / DATASET / "prepared" / "sample-universes" / key
    (root / "parts").mkdir(parents=True)
    (root / "parts" / "0000.parquet").write_bytes(b"x")
    payload = {
        "state": state,
        "identity": {"policy": policy, "source_manifest_sha256": bar_digest, "window_size": 128},
        "audit": {"test": {"accepted": 10}},
    }
    (root / "universe.json").write_text(json.dumps(payload))
    return root


def _half_build(volume: Path, digest: str, layout="features") -> Path:
    """What an interrupted store build leaves behind: lock and work files, no manifest."""
    root = volume / "datasets" / DATASET / "prepared" / layout / digest
    (root / "work").mkdir(parents=True)
    (root / "build.lock").write_bytes(b"")
    (root / "work" / "features.npy").write_bytes(b"x" * 16)
    return root


def _launch(
    volume: Path,
    name: str,
    store: Path | None,
    rank_ic: float | None,
    *,
    chip=True,
    layout="lgbm",
    partial=True,
):
    root = volume / "diagnostics" / layout / name
    root.mkdir(parents=True)
    if partial:
        (root / "partial").mkdir()
    if store is not None:
        identity = {"spec": {"name": "chip_tw_v1" if chip else "alpha_tw_v1"}}
        if chip:
            identity["chip_data"] = {}
        manifest = {
            "splits": {
                split: {"features": {"root": str(store), "identity": identity}}
                for split in ("train", "validation", "test")
            }
        }
        (root / "walk-forward-manifest.json").write_text(json.dumps(manifest))
    if rank_ic is not None:
        (root / "results.json").write_text(
            json.dumps({"fits": {"count": 588, "mean_validation_rank_ic": rank_ic}})
        )
    return root


def _features_launch(volume: Path, name: str, finished: bool) -> Path:
    root = volume / "diagnostics" / "features" / name
    root.mkdir(parents=True)
    if finished:
        (root / "features-manifest.json").write_text("{}")
    return root


def _lifecycle(volume: Path, key: str, state: str, launch_id: str) -> None:
    path = volume / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": state, "launch_id": launch_id, "pod_id": "pod"}))


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    return tmp_path / "volume"


def _invalid_paths(volume: Path) -> set[Path]:
    universe_entries, current = module.universes(volume, module.current_policy())
    store_entries = module.stores(volume, current)
    invalid_roots = {Path(s["path"]) for s in store_entries if s["invalid"]}
    entries = (
        universe_entries + store_entries + module.feature_launches(volume)
        + module.launches(volume, invalid_roots)
    )
    return {Path(e["path"]) for e in entries if e["invalid"]}


def test_superseded_universes_and_everything_built_on_them(volume: Path, capsys) -> None:
    policy = module.current_policy()
    assert policy is not None and "2026-07-10" in policy["calendar_overrides"]["XTAI"]["closed"]
    bar = _bar_store(volume)
    old_policy = json.loads(json.dumps(policy))
    old_policy["calendar_overrides"]["XTAI"]["closed"].remove("2026-07-10")
    stale = _universe(volume, "u-old", old_policy, bar)
    fresh = _universe(volume, "u-new", policy, bar)
    other_bar = _universe(volume, "u-other", policy, "0" * 64)
    old_store = _store(volume, "f-old", chip=False, margin=0, universe="u-old")
    old_labels = _store(volume, "l-old", chip=False, margin=0, layout="rank-labels",
                        universe="u-old")
    new_store = _store(volume, "f-new", chip=False, margin=0, universe="u-new")
    legacy = _store(volume, "f-legacy", chip=False, margin=0)  # no universe recorded: kept
    trained_on_old = _launch(volume, "launch-1", old_store, 0.05, chip=False)
    trained_on_new = _launch(volume, "launch-2", new_store, 0.05, chip=False)

    invalid = _invalid_paths(volume)
    assert {stale, other_bar, old_store, old_labels, trained_on_old} <= invalid
    assert not ({fresh, new_store, legacy, trained_on_new} & invalid)

    assert module.main(["--volume", str(volume)]) == 0
    out = capsys.readouterr().out
    assert "superseded data-cleaning policy" in out and "superseded bar store" in out
    assert "superseded sample universe" in out and "would move" in out

    assert module.main(["--volume", str(volume), "--apply", "--delete"]) == 0
    assert not stale.exists() and not old_store.exists() and not trained_on_old.exists()
    assert fresh.exists() and new_store.exists() and legacy.exists()
    assert not (volume / "diagnostics" / "invalid").exists()
    assert "deleted" in capsys.readouterr().out


def test_delete_needs_apply(volume: Path) -> None:
    volume.mkdir(parents=True)
    with pytest.raises(SystemExit):
        module.main(["--volume", str(volume), "--delete"])


def test_wrong_stores_and_the_launches_built_on_them(volume: Path, capsys) -> None:
    broken = _store(volume, "broken", chip=True, margin=0)
    good = _store(volume, "good", chip=True, margin=2444)
    alpha = _store(volume, "alpha", chip=False, margin=0)  # F0's store has no chip coverage
    foreign = _store(volume, "foreign", chip=True, margin=5, state="building")
    garbage_1 = _launch(volume, "launch-1", broken, 0.0)
    garbage_2 = _launch(volume, "launch-2", good, 0.0)  # resumed garbage folds: IC exactly 0
    valid = _launch(volume, "launch-3", good, 0.041)
    f0 = _launch(volume, "launch-0", alpha, 0.05, chip=False)

    assert module.main(["--volume", str(volume)]) == 0
    out = capsys.readouterr().out
    assert "would move" in out and "Re-run with --apply" in out
    assert broken.exists() and garbage_1.exists()  # a dry run moves nothing

    assert module.main(["--volume", str(volume), "--apply"]) == 0
    invalid_root = volume / "diagnostics" / "invalid"
    for gone, relative in (
        (broken, "datasets/ds/prepared/features/broken/features.json"),
        (foreign, "datasets/ds/prepared/features/foreign/features.json"),
        (garbage_1, "diagnostics/lgbm/launch-1/results.json"),
        (garbage_2, "diagnostics/lgbm/launch-2/partial"),
    ):
        assert not gone.exists(), gone
        assert (invalid_root / relative).exists(), relative
    for kept in (good, alpha, valid, f0):
        assert kept.exists(), kept


def test_interrupted_store_builds_are_dead_unless_a_build_is_in_flight(volume: Path) -> None:
    orphan = _half_build(volume, "orphan")
    label_orphan = _half_build(volume, "label-orphan", layout="rank-labels")
    good = _store(volume, "good", chip=True, margin=10)
    labels = _store(volume, "labels", chip=False, margin=0, layout="rank-labels")
    assert _invalid_paths(volume) == {orphan, label_orphan}

    # A feature-build job in flight may be the writer of a manifest-less directory: keep it.
    _lifecycle(volume, "lifecycle/research/features.json", "preparing", "launch-now")
    assert _invalid_paths(volume) == set()
    _lifecycle(volume, "lifecycle/research/features.json", "ready", "launch-now")
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert not orphan.exists() and not label_orphan.exists()
    assert good.exists() and labels.exists()
    assert (volume / "diagnostics/invalid/datasets/ds/prepared/features/orphan/build.lock").exists()


def test_feature_build_launches_without_manifest(volume: Path) -> None:
    finished = _features_launch(volume, "launch-1", True)
    dead = _features_launch(volume, "launch-2", False)
    running = _features_launch(volume, "launch-3", False)
    _lifecycle(volume, "lifecycle/research/features.json", "preparing", "launch-3")
    assert _invalid_paths(volume) == {dead}
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert finished.exists() and running.exists() and not dead.exists()


@pytest.mark.parametrize("layout", ["lgbm", "xs"])
def test_unfinished_launches_keep_only_the_resume_source(volume: Path, layout: str) -> None:
    good = _store(volume, "good", chip=True, margin=10)
    # Timed out twice, then one finished launch: the job resumes from the newest unfinished
    # launch (launch-2) even though launch-3 is newer, so launch-1 is unreachable.
    timed_out_old = _launch(volume, "launch-1", good, None, layout=layout)
    timed_out_new = _launch(volume, "launch-2", good, None, layout=layout)
    finished = _launch(volume, "launch-3", good, 0.03, layout=layout)
    # Died before writing any fold and not in flight: nothing to resume.
    empty = _launch(volume, "launch-4", None, None, layout=layout, partial=False)
    # Just started on this Pod (no partial yet): in flight per the lifecycle marker.
    starting = _launch(volume, "launch-5", None, None, layout=layout, partial=False)
    key = {"lgbm": "lifecycle/research/lgbm.json", "xs": "lifecycle/research/xs-train.json"}[layout]
    _lifecycle(volume, key, "preparing", "launch-5")

    assert _invalid_paths(volume) == {timed_out_old, empty}
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert not timed_out_old.exists() and not empty.exists()
    for kept in (timed_out_new, finished, starting):
        assert kept.exists(), kept


def test_a_single_timed_out_launch_is_never_moved(volume: Path, capsys) -> None:
    good = _store(volume, "good", chip=True, margin=10)
    timed_out = _launch(volume, "launch-1", good, None)
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert timed_out.exists()
    out = capsys.readouterr().out
    assert "resume-source" in out and "Nothing to quarantine" in out


def test_nothing_to_do_on_a_clean_volume(volume: Path, capsys) -> None:
    _store(volume, "good", chip=True, margin=10)
    _launch(volume, "launch-3", volume / "datasets/ds/prepared/features/good", 0.03)
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert "Nothing to quarantine" in capsys.readouterr().out
    assert not (volume / "diagnostics" / "invalid").exists()


def test_refuses_to_move_anything_when_a_target_exists(volume: Path) -> None:
    broken = _store(volume, "broken", chip=True, margin=0)
    orphan = _half_build(volume, "orphan")
    (volume / "diagnostics/invalid/datasets/ds/prepared/features/orphan").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.main(["--volume", str(volume), "--apply"])
    assert broken.exists() and orphan.exists()  # checked before the first move


def test_missing_volume_is_an_error(tmp_path: Path) -> None:
    assert module.main(["--volume", str(tmp_path / "nope")]) == 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
