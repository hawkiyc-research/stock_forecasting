"""Unit tests for the post-wake RunPod recovery decision boundary."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "scripts" / "recover_runpod_after_wake.py"
SPEC = importlib.util.spec_from_file_location("runpod_recovery", RECOVERY_PATH)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def _pod(role: str = "gpu-train", *, stage: str = "stage2") -> dict[str, object]:
    return {
        "id": "recovery-pod",
        "name": "stock-forecasting-train",
        "runtimeStatus": "running",
        "networkVolumeId": "volume-id",
        "uptimeSeconds": 3600,
        "env": {
            "RUNPOD_ROLE": role,
            "RUNPOD_STAGE": stage,
            "NETWORK_VOLUME_ROOT": "/runpod-volume",
            "PROJECT_ROOT": "/runpod-volume/stock_forecasting",
            "WANDB_RUN_ID": "run-recovery",
        },
    }


def test_project_filter_requires_the_same_volume_and_project_role() -> None:
    pod = _pod()

    assert RECOVERY.project_pod(pod, "volume-id") is True
    assert RECOVERY.project_pod(pod, "other-volume") is False
    pod["env"]["RUNPOD_ROLE"] = "unrelated"
    assert RECOVERY.project_pod(pod, "volume-id") is False


def test_stage2_cpu_pod_maps_to_mixed_finalization_marker() -> None:
    kind, key = RECOVERY.expected_lifecycle(_pod("cpu-prep"))
    assert (kind, key) == (
        "stage1-mixed-finalization",
        "lifecycle/stage1/mixed-finalization.json",
    )


def test_guard_deadline_reads_new_and_legacy_guard_records(tmp_path: Path) -> None:
    armed_at = datetime.now(UTC) - timedelta(seconds=30)
    ready = {
        "state": "armed",
        "pod_id": "recovery-pod",
        "delay_seconds": 600,
        "armed_at": armed_at.isoformat(),
    }
    (tmp_path / "recovery-pod.ready.json").write_text(json.dumps(ready), encoding="utf-8")
    deadline, delay = RECOVERY.guard_deadline("recovery-pod", tmp_path)
    assert delay == 600
    assert deadline <= int(datetime.now(UTC).timestamp()) + 570

    (tmp_path / "recovery-pod.ready.json").write_text(
        json.dumps({"state": "armed", "armed_at": armed_at.isoformat()}),
        encoding="utf-8",
    )
    (tmp_path / "recovery-pod.log").write_text(
        "guard armed for Pod recovery-pod; terminate after 120 seconds\n",
        encoding="utf-8",
    )
    _, legacy_delay = RECOVERY.guard_deadline("recovery-pod", tmp_path)
    assert legacy_delay == 120


def test_guard_permission_error_is_not_reported_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "recovery-pod.pid").write_text("1234\n", encoding="utf-8")
    (tmp_path / "recovery-pod.ready.json").write_text(
        json.dumps({"state": "armed", "pod_id": "recovery-pod", "pid": 1234}),
        encoding="utf-8",
    )

    def deny_probe(pid: int, signal: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(RECOVERY.os, "kill", deny_probe)

    assert RECOVERY.guard_alive("recovery-pod", tmp_path) is True


def test_guard_from_previous_boot_is_reported_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "recovery-pod.pid").write_text("1234\n", encoding="utf-8")
    (tmp_path / "recovery-pod.ready.json").write_text(
        json.dumps(
            {
                "state": "armed",
                "pod_id": "recovery-pod",
                "pid": 1234,
                "host_boot_id": "darwin-old-boot",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        RECOVERY, "host_boot_identity", lambda: ("darwin-current-boot", 200)
    )

    assert RECOVERY.guard_alive("recovery-pod", tmp_path) is False


def test_legacy_guard_armed_before_current_boot_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "recovery-pod.pid").write_text("1234\n", encoding="utf-8")
    (tmp_path / "recovery-pod.ready.json").write_text(
        json.dumps(
            {
                "state": "armed",
                "pod_id": "recovery-pod",
                "pid": 1234,
                "armed_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    current_boot = int(datetime(2026, 1, 2, tzinfo=UTC).timestamp())
    monkeypatch.setattr(
        RECOVERY, "host_boot_identity", lambda: ("darwin-current-boot", current_boot)
    )

    assert RECOVERY.guard_alive("recovery-pod", tmp_path) is False


def test_rearm_disables_emergency_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(RECOVERY.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(RECOVERY.os, "access", lambda path, mode: True)
    monkeypatch.setattr(
        RECOVERY, "host_boot_identity", lambda: ("darwin-current-boot", 200)
    )

    def run_guard(*args: object, **kwargs: object) -> object:
        captured["env"] = kwargs["env"]
        return RECOVERY.subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(RECOVERY.subprocess, "run", run_guard)

    RECOVERY.rearm_guard(_pod(), 300, tmp_path)

    command_env = captured["env"]
    assert isinstance(command_env, dict)
    assert command_env["RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE"] == "0"


def test_rearm_preserves_training_checkpoint_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    pod = _pod()
    pod["env"]["RUNPOD_CONFIG"] = "configs/stage1_kronos_base_lora.yaml"
    pod["env"]["RUNPOD_DATASET_REQUEST_SHA256"] = "0" * 64
    ready = {
        "armed_at": (datetime.now(UTC) - timedelta(seconds=100)).isoformat(),
        "delay_seconds": 300,
        "soft_limit_seconds": 80,
        "checkpoint_not_before": "2026-01-01T00:00:00Z",
    }
    (tmp_path / "recovery-pod.ready.json").write_text(json.dumps(ready), encoding="utf-8")
    monkeypatch.setattr(RECOVERY.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(RECOVERY.os, "access", lambda path, mode: True)
    monkeypatch.setattr(RECOVERY, "host_boot_identity", lambda: ("darwin-current-boot", 200))

    def run_guard(*args: object, **kwargs: object) -> object:
        captured["env"] = kwargs["env"]
        return RECOVERY.subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(RECOVERY.subprocess, "run", run_guard)
    RECOVERY.rearm_guard(pod, 200, tmp_path)
    command_env = captured["env"]
    assert isinstance(command_env, dict)
    assert command_env["RUNPOD_GUARD_SOFT_LIMIT_SECONDS"] == "1"
    assert command_env["RUNPOD_GUARD_DATASET_REQUEST_SHA256"] == "0" * 64
    assert command_env["RUNPOD_GUARD_CHECKPOINT_NOT_BEFORE"] == "2026-01-01T00:00:00Z"
    assert command_env["RUNPOD_GUARD_CHECKPOINT_CONFIG"] == str(
        ROOT / "configs/stage1_kronos_base_lora.yaml"
    )


def test_expired_section_safe_guard_is_rearmed_without_pod_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = _pod()
    pod["env"]["RUNPOD_SECTION_SAFE_STOP"] = "1"
    (tmp_path / "recovery-pod.ready.json").write_text(json.dumps({
        "state": "armed",
        "pod_id": "recovery-pod",
        "delay_seconds": 30,
        "soft_limit_seconds": 10,
        "armed_at": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
    }), encoding="utf-8")
    monkeypatch.setattr(RECOVERY, "list_lifecycle_keys", lambda volume_id: set())
    monkeypatch.setattr(RECOVERY, "pod_records", lambda volume_id, pod_id: [pod])
    monkeypatch.setattr(RECOVERY, "lifecycle_state", lambda *args: ("active", "training"))
    monkeypatch.setattr(RECOVERY, "guard_alive", lambda pod_id, guard_dir: False)

    observation = RECOVERY.observe("volume-id", None, tmp_path, object())[0]
    assert observation["action"] == "keep"
    assert observation["rearm_remaining"] == 2


def test_guard_pid_must_still_belong_to_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "recovery-pod.pid").write_text("1234\n", encoding="utf-8")
    (tmp_path / "recovery-pod.ready.json").write_text(
        json.dumps(
            {
                "state": "armed",
                "pod_id": "recovery-pod",
                "pid": 1234,
                "host_boot_id": "darwin-current-boot",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        RECOVERY, "host_boot_identity", lambda: ("darwin-current-boot", 200)
    )
    monkeypatch.setattr(RECOVERY.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(
        RECOVERY.subprocess,
        "run",
        lambda *args, **kwargs: RECOVERY.subprocess.CompletedProcess(
            args=[], returncode=0, stdout="unrelated-process", stderr=""
        ),
    )

    assert RECOVERY.guard_alive("recovery-pod", tmp_path) is False


def test_unknown_lifecycle_state_is_not_a_termination_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Readiness:
        @staticmethod
        def _guard_lifecycle_state(*args: object, **kwargs: object) -> str:
            raise ValueError("marker cannot be verified")

    monkeypatch.setattr(RECOVERY, "load_marker", lambda *args: {"state": "unverifiable"})
    pod = _pod()
    state, evidence = RECOVERY.lifecycle_state(
        pod,
        "volume-id",
        {"lifecycle/stage1/training.json"},
        Readiness(),
    )

    assert state == "unknown"
    assert "invalid_marker" in evidence


def test_active_and_terminal_lifecycle_states_are_distinct() -> None:
    class Readiness:
        state = "preparing"

        @classmethod
        def _guard_lifecycle_state(cls, *args: object, **kwargs: object) -> str:
            return cls.state

    pod = _pod()
    marker_key = "lifecycle/stage1/training.json"
    original_loader = RECOVERY.load_marker
    try:
        RECOVERY.load_marker = lambda *args: {"state": "preparing"}  # type: ignore[assignment]
        active, _ = RECOVERY.lifecycle_state(
            pod, "volume-id", {marker_key}, Readiness()
        )
        Readiness.state = "ready"
        RECOVERY.load_marker = lambda *args: {"state": "ready"}  # type: ignore[assignment]
        terminal, _ = RECOVERY.lifecycle_state(
            pod, "volume-id", {marker_key}, Readiness()
        )
    finally:
        RECOVERY.load_marker = original_loader

    assert active == "active"
    assert terminal == "terminal"


@pytest.mark.parametrize("runtime", ("initializing", "unknown"))
def test_non_running_or_indeterminate_runtime_is_kept(runtime: str) -> None:
    pod = _pod()
    pod["runtimeStatus"] = runtime
    assert runtime in RECOVERY.ACTIVE_RUNTIME_STATES
