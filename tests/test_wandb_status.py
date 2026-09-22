"""Durable delivery-state contracts for online and offline W&B runs."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from stock_forecasting.tracking import TrackingRun, collect_selection_provenance
from stock_forecasting.wandb_status import read_wandb_status, update_wandb_status


class _FailingLogBackend:
    def __init__(self, directory: Path) -> None:
        self.dir = str(directory / "files")
        self.summary: dict[str, Any] = {}

    def log(self, payload: dict[str, Any]) -> None:
        raise RuntimeError(
            f"simulated upload failure at step {payload.get('trainer/global_step')}"
        )

    def finish(self, *, exit_code: int = 0) -> None:
        assert exit_code == 1


def _configure_selection_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    different_dataset: bool = False,
) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    helper = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    args = helper["build_parser"]().parse_args([
        "create", "--project-root", str(project), "--stage", "stage2",
        "--data-profile", "us_tw_eodhd", "--start", "2021-01-01", "--end", "2026-06-01",
        "--universe", "all",
    ])
    selected = helper["_build_selection"](args, project)
    selection_file = tmp_path / (selected["selection_id"] + ".json")
    selection_file.write_text(json.dumps(selected), encoding="utf-8")
    request = selected["dataset_request"]
    marker = tmp_path / "datasets" / selected["dataset_request_sha256"] / "dataset-manifest.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "kind": "ohlcv-bar-store-dataset", "state": "ready",
                "dataset_profile": request["profile"],
                "selected_datasets": request["selected_datasets"],
                "date_range": {
                    **request["date_range"],
                    "start_inclusive": "2016-01-01" if different_dataset else "2021-01-01",
                },
                "storage_preparation_spec": helper["dataset_request_identity_payload"](
                    request
                )["storage_preparation"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNPOD_POD_ID", "test-pod")
    monkeypatch.setenv("NETWORK_VOLUME_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNPOD_VOLUME_ROOT", str(tmp_path))
    for key, value in helper["_selection_exports"](selection_file, selected).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RUNPOD_REMOTE_SELECTION_PATH", str(selection_file))
    # An unrelated last CPU preparation is not the active dataset's provenance.
    legacy = tmp_path / "lifecycle/stage1/dataset.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"state":"ready","stage_config_sha256":"obsolete"}')
    return selected


def test_tracking_accepts_training_only_selection_revisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = _configure_selection_provenance(monkeypatch, tmp_path)

    provenance = collect_selection_provenance()

    assert provenance["selection_id"] == selected["selection_id"]
    assert provenance["selection_sha256"] == selected["selection_sha256"]
    assert provenance["stage_config_sha256"] == selected["stage"]["config_sha256"]
    assert provenance["dataset_request_sha256"] == selected["dataset_request_sha256"]
    assert provenance["path"].endswith("/dataset-manifest.json")


def test_tracking_rejects_a_different_dataset_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_selection_provenance(
        monkeypatch,
        tmp_path,
        different_dataset=True,
    )

    with pytest.raises(ValueError, match="date_range"):
        collect_selection_provenance()


def test_component_states_merge_into_one_workflow_status(tmp_path: Path) -> None:
    transaction = tmp_path / "wandb" / "offline-run"
    transaction.mkdir(parents=True)
    status_path = update_wandb_status(
        run_id="run-test",
        component="training",
        state="offline_pending",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=transaction,
    )
    update_wandb_status(
        run_id="run-test",
        component="validation",
        state="online_finished",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )
    assert read_wandb_status(status_path)["state"] == "offline_pending"

    update_wandb_status(
        run_id="run-test",
        component="training",
        state="synced",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=transaction,
    )
    assert read_wandb_status(status_path)["state"] == "ready"


def test_online_log_failure_remains_pending_after_finish(tmp_path: Path) -> None:
    run_directory = tmp_path / "savedModel" / "run-test"
    transaction = tmp_path / "wandb" / "run-test"
    run_directory.mkdir(parents=True)
    transaction.mkdir(parents=True)
    backend = _FailingLogBackend(transaction)
    tracking = TrackingRun(
        id="run-test",
        name="test",
        directory=run_directory,
        backend=backend,
        mode="online",
        tracking_enabled=True,
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tracking.log({"train/loss": 1.0}, step=1)
    tracking.finish(exit_code=1)

    payload = read_wandb_status(
        tmp_path / "lifecycle" / "runs" / "run-test" / "wandb.json"
    )
    assert payload["state"] == "offline_pending"
    assert payload["components"]["training"]["state"] == "offline_pending"
    metric = json.loads((run_directory / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["step"] == 1
    assert metric["train/loss"] == 1.0


def test_offline_transaction_write_failure_is_not_marked_recoverable(tmp_path: Path) -> None:
    run_directory = tmp_path / "savedModel" / "run-test"
    transaction = tmp_path / "wandb" / "offline-run"
    run_directory.mkdir(parents=True)
    transaction.mkdir(parents=True)
    tracking = TrackingRun(
        id="run-test",
        name="test",
        directory=run_directory,
        backend=_FailingLogBackend(transaction),
        mode="offline",
        tracking_enabled=True,
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tracking.log({"train/loss": 1.0}, step=1)
    tracking.finish(exit_code=1)

    payload = read_wandb_status(
        tmp_path / "lifecycle" / "runs" / "run-test" / "wandb.json"
    )
    assert payload["state"] == "failed"
    assert payload["components"]["training"]["state"] == "failed"


def test_resume_preserves_every_pending_transaction(tmp_path: Path) -> None:
    first = tmp_path / "wandb" / "offline-first"
    second = tmp_path / "wandb" / "online-resume"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    status_path = update_wandb_status(
        run_id="run-test",
        component="training",
        state="offline_pending",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=first,
    )
    update_wandb_status(
        run_id="run-test",
        component="training",
        state="online_running",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=second,
    )
    update_wandb_status(
        run_id="run-test",
        component="training",
        state="online_finished",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=second,
    )

    payload = read_wandb_status(status_path)
    training = payload["components"]["training"]
    assert training["state"] == "offline_pending"
    assert {
        transaction["directory"]: transaction["state"]
        for transaction in training["transactions"]
    } == {
        str(first): "offline_pending",
        str(second): "online_finished",
    }

    update_wandb_status(
        run_id="run-test",
        component="training",
        state="synced",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=first,
    )
    assert read_wandb_status(status_path)["state"] == "ready"
