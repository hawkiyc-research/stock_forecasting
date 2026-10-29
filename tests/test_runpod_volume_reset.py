"""The volume reset plan removes superseded data and weights and keeps the traceable record."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESET = runpy.run_path(str(ROOT / "scripts/runpod_volume_reset.py"))
NS = "a" * 64


def _write(path: Path, payload: bytes = b"x" * 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _volume(tmp_path: Path) -> Path:
    volume = tmp_path / "runpod-volume"
    ns = volume / "datasets" / NS
    _write(ns / "raw" / "market.parquet", b"r" * 1000)
    _write(ns / "prepared" / "features" / "id1" / "store.parquet", b"f" * 500)
    _write(ns / "provider-checkpoints" / "twse_official" / "bars.parquet", b"c" * 300)
    _write(ns / "api-cache" / "twse" / "abc.json", b"{}")
    _write(ns / "download-manifest.json", b"{}")
    _write(ns / "download-progress.json", b"{}")
    _write(volume / "datasets" / "tw-chip-data" / "margin" / "2026.csv", b"m" * 100)
    finished = volume / "diagnostics" / "lgbm" / "launch-20261028T000000Z-1"
    _write(finished / "results.json", b"{}")
    _write(finished / "predictions.parquet", b"p" * 200)
    _write(finished / "partial" / "fold-01.parquet", b"q" * 200)
    unfinished = volume / "diagnostics" / "lgbm" / "launch-20261029T000000Z-2"
    _write(unfinished / "partial" / "fold-01.parquet", b"q" * 200)
    xs = volume / "diagnostics" / "xs" / "launch-20261023T000000Z-3"
    _write(xs / "results.json", b"{}")
    _write(xs / "checkpoints" / "fold-01.pt", b"w" * 400)
    _write(volume / "diagnostics" / "invalid" / "datasets" / "x" / "store.parquet", b"i" * 50)
    checkpoint = volume / "savedModel" / "run-1" / "checkpoint-000100"
    _write(checkpoint / "adapter.safetensors", b"a" * 900)
    _write(checkpoint / "optimizer.pt", b"o" * 900)
    _write(checkpoint / "trainer_state.json", b"{}")
    _write(volume / "forward" / "2026-10-20" / "shared-cache" / "bars.npz", b"n" * 100)
    _write(volume / "forward" / "2026-10-20" / "package" / "bars.parquet", b"b" * 100)
    _write(volume / "forward" / "2026-10-20" / "package" / "manifest.json", b"{}")
    marker = json.dumps({"dataset": f"datasets/{NS}/raw"}).encode()
    _write(volume / "lifecycle" / "stage1" / "dataset.json", marker)
    _write(volume / "evaluations" / "run-1" / "metrics.json", b"{}")
    _write(volume / "logs" / "run-1" / "train.log", b"log")
    return volume


def test_plan_targets_superseded_data_and_weights_only(tmp_path: Path) -> None:
    volume = _volume(tmp_path)
    plan = RESET["build_plan"](volume)
    paths = {entry["path"] for entry in plan}

    assert f"datasets/{NS}/raw" in paths
    assert f"datasets/{NS}/prepared" in paths
    assert f"datasets/{NS}/provider-checkpoints" in paths
    assert f"datasets/{NS}/download-progress.json" in paths
    assert "diagnostics/lgbm/launch-20261028T000000Z-1/partial" in paths
    assert "diagnostics/lgbm/launch-20261029T000000Z-2" in paths
    assert "diagnostics/xs/launch-20261023T000000Z-3/checkpoints/fold-01.pt" in paths
    assert "diagnostics/invalid" in paths
    assert "savedModel/run-1/checkpoint-000100/adapter.safetensors" in paths
    assert "savedModel/run-1/checkpoint-000100/optimizer.pt" in paths
    assert "forward/2026-10-20/shared-cache" in paths
    assert "forward/2026-10-20/package/bars.parquet" in paths
    # Kept: api cache, manifests, chip data, finished results, lifecycle, evaluations, logs.
    for kept in (
        f"datasets/{NS}/api-cache",
        f"datasets/{NS}/download-manifest.json",
        "datasets/tw-chip-data",
        "diagnostics/lgbm/launch-20261028T000000Z-1/results.json",
        "diagnostics/lgbm/launch-20261028T000000Z-1/predictions.parquet",
        "diagnostics/xs/launch-20261023T000000Z-3/results.json",
        "savedModel/run-1/checkpoint-000100/trainer_state.json",
        "forward/2026-10-20/package/manifest.json",
        "lifecycle/stage1/dataset.json",
        "evaluations/run-1/metrics.json",
        "logs/run-1/train.log",
    ):
        assert not any(p == kept or kept.startswith(p + "/") for p in paths), kept
    assert all(entry["bytes"] > 0 for entry in plan)


def test_dry_run_deletes_nothing_and_apply_needs_confirmation(tmp_path: Path) -> None:
    volume = _volume(tmp_path)
    before = sorted(str(p.relative_to(volume)) for p in volume.rglob("*"))
    plan_root = tmp_path / "plan"

    assert RESET["main"](["--volume", str(volume), "--plan-root", str(plan_root)]) == 0
    assert sorted(str(p.relative_to(volume)) for p in volume.rglob("*")) == before
    assert (plan_root / "plan.json").is_file()
    assert RESET["main"](["--volume", str(volume), "--plan-root", str(plan_root), "--apply"]) == 2
    assert sorted(str(p.relative_to(volume)) for p in volume.rglob("*")) == before


def test_apply_removes_targets_and_reports_dangling_markers(tmp_path: Path, capsys) -> None:
    volume = _volume(tmp_path)
    plan_root = tmp_path / "plan"

    code = RESET["main"](
        ["--volume", str(volume), "--plan-root", str(plan_root), "--apply", "--confirm", "RESET"]
    )

    assert code == 0
    assert not (volume / "datasets" / NS / "raw").exists()
    assert not (volume / "datasets" / NS / "prepared").exists()
    assert (volume / "datasets" / NS / "api-cache" / "twse" / "abc.json").is_file()
    assert (volume / "datasets" / NS / "download-manifest.json").is_file()
    assert (volume / "datasets" / "tw-chip-data" / "margin" / "2026.csv").is_file()
    finished = volume / "diagnostics" / "lgbm" / "launch-20261028T000000Z-1"
    assert (finished / "results.json").is_file()
    assert not (finished / "partial").exists()
    assert not (volume / "diagnostics" / "lgbm" / "launch-20261029T000000Z-2").exists()
    assert not (volume / "diagnostics" / "invalid").exists()
    checkpoint = volume / "savedModel" / "run-1" / "checkpoint-000100"
    assert not (checkpoint / "adapter.safetensors").exists()
    assert (checkpoint / "trainer_state.json").is_file()
    assert (volume / "forward" / "2026-10-20" / "package" / "manifest.json").is_file()
    assert not (volume / "forward" / "2026-10-20" / "package" / "bars.parquet").exists()
    applied = json.loads((plan_root / "applied.json").read_text(encoding="utf-8"))
    planned = json.loads((plan_root / "plan.json").read_text(encoding="utf-8"))["targets"]
    assert len(applied["deleted"]) == len(planned)
    assert "lifecycle/stage1/dataset.json" in capsys.readouterr().out
