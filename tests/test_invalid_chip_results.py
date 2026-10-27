"""The Pod-side audit of chip results finds the all-NaN store and the launches built on it."""

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


def _store(volume: Path, dataset: str, digest: str, *, chip: bool, margin: int, state="ready"):
    root = volume / "datasets" / dataset / "prepared" / "features" / digest
    root.mkdir(parents=True)
    identity = {"spec": {"name": "chip_tw_v1" if chip else "alpha_tw_v1"}}
    payload = {"state": state, "identity": identity}
    if chip:
        identity["chip_data"] = {"manifests": {}}
        payload["chip_coverage"] = {
            "symbols_with_margin": margin,
            "symbols_with_institutional": margin,
        }
    (root / "features.json").write_text(json.dumps(payload))
    (root / "bucket-000.parquet").write_bytes(b"x")
    return root


def _launch(volume: Path, name: str, store: Path | None, rank_ic: float | None, chip=True):
    root = volume / "diagnostics" / "lgbm" / name
    (root / "partial").mkdir(parents=True)
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


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    return tmp_path / "volume"


def test_finds_only_the_wrong_store_and_the_launches_built_on_it(volume: Path, capsys) -> None:
    broken = _store(volume, "ds", "broken", chip=True, margin=0)
    good = _store(volume, "ds", "good", chip=True, margin=2444)
    _store(volume, "ds", "alpha", chip=False, margin=0)  # F0's store has no chip coverage
    building = _store(volume, "ds", "pending", chip=True, margin=0, state="building")
    garbage_1 = _launch(volume, "launch-1", broken, 0.0)
    garbage_2 = _launch(volume, "launch-2", good, 0.0)  # resumed garbage folds: IC exactly 0
    valid = _launch(volume, "launch-3", good, 0.041)
    alpha_store = volume / "datasets/ds/prepared/features/alpha"
    f0 = _launch(volume, "launch-0", alpha_store, 0.05, chip=False)
    interrupted = _launch(volume, "launch-4", None, None)

    assert module.main(["--volume", str(volume)]) == 0
    out = capsys.readouterr().out
    assert "would move" in out and "Re-run with --apply" in out
    assert broken.exists() and garbage_1.exists()  # a dry run moves nothing

    assert module.main(["--volume", str(volume), "--apply"]) == 0
    invalid_root = volume / "diagnostics" / "invalid"
    assert not broken.exists()
    assert (invalid_root / "datasets/ds/prepared/features/broken/features.json").is_file()
    assert not garbage_1.exists() and not garbage_2.exists()
    assert (invalid_root / "diagnostics/lgbm/launch-1/results.json").is_file()
    assert (invalid_root / "diagnostics/lgbm/launch-2/partial").is_dir()
    for kept in (good, building, valid, f0, interrupted):
        assert kept.exists(), kept
    out = capsys.readouterr().out
    assert str(f0) not in out  # non-chip launches are not even listed
    assert "incomplete" in out  # the interrupted chip launch is reported, not moved


def test_nothing_to_do_on_a_clean_volume(volume: Path, capsys) -> None:
    _store(volume, "ds", "good", chip=True, margin=10)
    _launch(volume, "launch-3", volume / "datasets/ds/prepared/features/good", 0.03)
    assert module.main(["--volume", str(volume), "--apply"]) == 0
    assert "Nothing to quarantine" in capsys.readouterr().out
    assert not (volume / "diagnostics" / "invalid").exists()


def test_refuses_to_overwrite_an_existing_quarantine_entry(volume: Path) -> None:
    broken = _store(volume, "ds", "broken", chip=True, margin=0)
    target = volume / "diagnostics/invalid/datasets/ds/prepared/features/broken"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.main(["--volume", str(volume), "--apply"])
    assert broken.exists()


def test_missing_volume_is_an_error(tmp_path: Path) -> None:
    assert module.main(["--volume", str(tmp_path / "nope")]) == 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
