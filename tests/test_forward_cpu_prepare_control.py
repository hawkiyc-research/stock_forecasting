"""Dependency-free checks of the CPU preparation's forward mode, its lifecycle and transfers."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import test_probe_guard
import test_probe_tmux

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))
KIND = "forward-cpu-preparation"
KEY = "lifecycle/forward/cpu-preparation.json"
LAUNCH = "launch-20261020T000000Z-1"
END = "2026-10-20"


def write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o700)


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def canonical_sha256(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ForwardLifecycleTests(unittest.TestCase):
    def test_key_maps_to_the_forward_kind_in_every_allowlist(self) -> None:
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; runpod_guard_lifecycle_kind "$2"', "kind",
             str(ROOT / "scripts/lib/runpod_paths.sh"), KEY],
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual((result.returncode, result.stdout.strip()), (0, KIND), result.stderr)
        self.assertIn(KIND, READINESS["GUARD_LIFECYCLE_KINDS"])
        guard = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
        self.assertIn(f'expected_lifecycle_kind="{KIND}"', guard)
        launched = subprocess.run(
            ["bash", str(ROOT / "scripts/launch_runpod_guard.sh"), "pod-forward", "60", KEY,
             "relative.log"],
            env={"PATH": os.defpath, "HOME": "/nonexistent"},
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual(launched.returncode, 2)
        self.assertIn("Guard log path must be absolute", launched.stderr)

    def test_write_state_uses_only_the_forward_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()

            def write(state: str, output: Path, *extra: str):
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/runpod_readiness.py"), "write-state",
                     "--output", str(output), "--network-volume-root", str(volume),
                     "--kind", KIND, "--state", state, "--launch-id", LAUNCH, *extra],
                    env={**os.environ, "RUNPOD_POD_ID": "pod-forward"},
                    capture_output=True, text=True, check=False, timeout=30,
                )

            # The training dataset's CPU-preparation marker is never written in forward mode.
            self.assertNotEqual(
                write("preparing", volume / "lifecycle/stage1/cpu-preparation.json").returncode, 0
            )
            # Acquisition pauses exist only for dataset preparation.
            self.assertNotEqual(
                write("waiting_for_provider", volume / KEY, "--exit-code", "75").returncode, 0
            )
            for state, extra in (("preparing", ()), ("failed", ("--exit-code", "1")),
                                 ("ready", ())):
                with self.subTest(state=state):
                    result = write(state, volume / KEY, *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    guard = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/runpod_readiness.py"),
                         "guard-lifecycle-state", "--expected-kind", KIND,
                         "--expected-pod-id", "pod-forward"],
                        input=(volume / KEY).read_text(encoding="utf-8"),
                        capture_output=True, text=True, check=False, timeout=30,
                    )
                    self.assertEqual(guard.stdout.strip(), state, guard.stderr)

    def test_recovery_and_environment_import_know_the_forward_mode(self) -> None:
        pod = {
            "id": "pod-forward", "networkVolumeId": "volume-id",
            "env": {"RUNPOD_ROLE": "cpu-prep", "RUNPOD_STAGE": "stage2",
                    "RUNPOD_CPU_PREPARE_MODE": "forward",
                    "NETWORK_VOLUME_ROOT": "/runpod-volume",
                    "PROJECT_ROOT": "/runpod-volume/stock_forecasting"},
        }
        self.assertTrue(RECOVERY["project_pod"](pod, "volume-id"))
        self.assertEqual(RECOVERY["expected_lifecycle"](pod), (KIND, KEY))
        pod["env"]["RUNPOD_CPU_PREPARE_MODE"] = "dataset"
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod)[1], "lifecycle/stage1/mixed-finalization.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "environ"
            names = {
                "RUNPOD_POD_ID": "pod-forward", "RUNPOD_ROLE": "cpu-prep",
                "RUNPOD_CPU_PREPARE_MODE": "forward", "RUNPOD_FORWARD_END": END,
            }
            source.write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in names.items()))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/runpod_reexec_with_pid1_env.py"),
                 "--source", str(source), "--", sys.executable, "-c",
                 "import json, os; print(json.dumps(dict(os.environ)))"],
                env={"PATH": os.defpath, "RUNPOD_TEST_MODE": "1"},
                capture_output=True, text=True, check=False, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        imported = json.loads(result.stdout)
        self.assertEqual(
            (imported["RUNPOD_CPU_PREPARE_MODE"], imported["RUNPOD_FORWARD_END"]), ("forward", END)
        )


class ForwardGuardTests(unittest.TestCase):
    def harness(self):
        # Reuse the probe guard harness: real guard, fake S3 reads and Pod deletion.
        return test_probe_guard.ProbeGuardTests.harness(self)

    def publish(self, harness, pod_id: str, state: str) -> None:
        marker = harness.volume / KEY
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "schema_version": 1, "kind": KIND, "state": state, "pod_id": pod_id,
            "launch_id": LAUNCH, "generated_at": "2026-10-20T00:00:00+00:00",
        }), encoding="utf-8")

    def run_guard(self, harness, seconds: int) -> subprocess.CompletedProcess:
        return test_probe_guard.ProbeGuardTests.run_guard(
            self, harness, seconds=seconds,
            overrides={"RUNPOD_GUARD_LIFECYCLE_KEY": KEY, "RUNPOD_GUARD_RUN_ID": ""},
        )

    def test_guard_terminates_when_the_forward_lifecycle_finishes(self) -> None:
        for state in ("ready", "failed", "timed_out"):
            with self.subTest(state=state):
                harness = self.harness()
                self.publish(harness, "probe-fixture", state)
                result = self.run_guard(harness, seconds=8)
                log = harness.read("local-guard.log")
                self.assertEqual(result.returncode, 0, result.stderr + log)
                self.assertIn(f"termination triggered by lifecycle-{state}", log)

    def test_guard_ignores_running_or_foreign_forward_markers(self) -> None:
        for pod_id, state in (("probe-fixture", "preparing"), ("another-pod", "ready")):
            with self.subTest(pod_id=pod_id, state=state):
                harness = self.harness()
                self.publish(harness, pod_id, state)
                result = self.run_guard(harness, seconds=2)
                log = harness.read("local-guard.log")
                self.assertEqual(result.returncode, 0, result.stderr + log)
                self.assertIn("termination triggered by hard-limit", log)


class ForwardTmuxTests(unittest.TestCase):
    def harness(self) -> test_probe_tmux.ProbeHarness:
        harness = test_probe_tmux.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        harness.environment["RUNPOD_CPU_PREPARE_MODE"] = "forward"
        # The forward job is a separate script; the frozen dataset job must not run.
        source = (harness.scripts / "runpod_cpu_prepare.sh").read_text(encoding="utf-8")
        forward = harness.scripts / "runpod_cpu_prepare_forward.sh"
        forward.write_text(
            source.replace("set -eu\n", 'set -eu\n: > "${HARNESS_ROOT}/forward-job"\n', 1),
            encoding="utf-8",
        )
        forward.chmod(0o700)
        return harness

    def test_runner_finalizes_only_the_forward_marker(self) -> None:
        for exit_code, state in (("0", "succeeded"), ("1", "failed"), ("75", "failed")):
            with self.subTest(exit_code=exit_code):
                harness = self.harness()
                harness.environment["HARNESS_JOB_EXIT"] = exit_code
                launched = harness.command("runpod_tmux_launch.sh", "cpu-prepare")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-cpu-prepare", launched.stdout)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, int(exit_code), runner.stderr)
                self.assertTrue((harness.root / "forward-job").exists())
                # Exit 75 is no resumable acquisition pause in forward mode.
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.read("worker-env").splitlines()[0], "cpu-prep")
                self.assertEqual(harness.read("shutdown-called").strip(), "cpu-prep")
                calls_path = harness.root / "lifecycle-calls"
                calls = calls_path.read_text(encoding="utf-8") if calls_path.exists() else ""
                self.assertNotIn("stage1/cpu-preparation.json", calls)
                self.assertNotIn("resumable-cpu-preparation-lifecycle", calls)
                if exit_code == "0":
                    self.assertEqual(calls, "")
                elif exit_code == "1":
                    self.assertIn(str(harness.volume / KEY), calls)
                    self.assertIn(f"--kind {KIND} --state failed", calls)

    def test_dataset_mode_keeps_the_dataset_marker(self) -> None:
        harness = self.harness()
        harness.environment["RUNPOD_CPU_PREPARE_MODE"] = "dataset"
        harness.environment["HARNESS_JOB_EXIT"] = "1"
        self.assertEqual(harness.command("runpod_tmux_launch.sh", "cpu-prepare").returncode, 0)
        harness.run_worker()
        self.assertFalse((harness.root / "forward-job").exists())
        calls = harness.read("lifecycle-calls")
        self.assertIn("lifecycle/stage1/cpu-preparation.json", calls)
        self.assertNotIn(KEY, calls)

    def test_unknown_mode_starts_nothing(self) -> None:
        harness = self.harness()
        harness.environment["RUNPOD_CPU_PREPARE_MODE"] = "other"
        result = harness.command("runpod_tmux_launch.sh", "cpu-prepare")
        self.assertEqual(result.returncode, 2)
        self.assertIn("dataset or forward", result.stderr)
        self.assertEqual(harness.runners(), [])


class ForwardJobTests(unittest.TestCase):
    def run_job(self, script: str, **overrides: str) -> subprocess.CompletedProcess:
        base = {
            "PATH": os.defpath, "RUNPOD_TEST_MODE": "1", "NETWORK_VOLUME_ROOT": "/runpod-volume",
            "RUNPOD_LAUNCH_ID": LAUNCH, "RUNPOD_ROLE": "cpu-prep", "RUNPOD_FORWARD_END": END,
            "RUNPOD_CPU_PREPARE_MODE": "forward", "RUNPOD_PYTHON_BIN": sys.executable,
        }
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / script)], env={**base, **overrides},
            capture_output=True, text=True, check=False, timeout=10,
        )

    def test_forward_job_refuses_another_role_date_launch_or_missing_relay(self) -> None:
        for overrides, message in (
            ({}, "TPEX_PROXY_TOKEN"),
            ({"RUNPOD_ROLE": "cpu-research"}, "cpu-prep role"),
            ({"RUNPOD_FORWARD_END": "2026-10"}, "RUNPOD_FORWARD_END"),
            ({"RUNPOD_LAUNCH_ID": "../x"}, "RUNPOD_LAUNCH_ID"),
            ({"TPEX_PROXY_TOKEN": "t" * 40, "TPEX_PROXY_URL": "https://example.com"},
             "TPEX_PROXY_URL"),
        ):
            with self.subTest(message=message):
                result = self.run_job("runpod_cpu_prepare_forward.sh", **overrides)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_forward_job_order_and_shared_cache(self) -> None:
        job = (ROOT / "scripts/runpod_cpu_prepare_forward.sh").read_text(encoding="utf-8")
        quarantine = job.index('runpod_readiness.py" quarantine-stale-code')
        self.assertLess(job.index("write_forward_state preparing\n"), quarantine)
        self.assertLess(quarantine, job.index('runpod_readiness.py" check-code'))
        self.assertIn('--shared-cache "${SHARED_CACHE_ROOT}"', job)
        self.assertIn("/datasets/forward-taiwan/api-cache", job)
        upload = (ROOT / "scripts/upload_runpod_forward_cache.py").read_text(encoding="utf-8")
        self.assertIn('REMOTE_NAMESPACE = "datasets/forward-taiwan/"', upload)
        self.assertNotIn("DATA_ROOT", job)


class ForwardControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-forward-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        for folder in ("scripts", "src", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pyproject.toml", self.project / "pyproject.toml")
        self.env_file = self.project / ".env"
        self.volume = self.root / "volume"
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "python3").symlink_to(sys.executable)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "FIN_TS_", "WANDB_", "TPEX_", "AWS_", "STAGE1_"))
            and key not in {"PROJECT_ROOT", "DATA_ROOT", "EODHD_API_TOKEN"}
        }
        self.environment.update({
            "PATH": f"{binary}:{os.defpath}", "RUNPOD_ENV_FILE": str(self.env_file),
            "RUNPOD_TEST_MODE": "1", "RUNPOD_TEST_READINESS_READY": "1",
            "HARNESS_VOLUME": str(self.volume), "RUNPOD_PROBE_DOWNLOAD_WORKERS": "1",
            "RUNPOD_FORWARD_UPLOAD_WORKERS": "2",
        })
        # A fake volume: s3 cp in either direction and an unpaginated object listing.
        write_script(self.project / "scripts/runpod_s3_project.sh", r"""
            if [[ "$1 $2" == "s3api list-objects-v2" ]]; then
                prefix="$6"
                exec python3 -c '
import json, os, sys
root, prefix = sys.argv[1], sys.argv[2]
items = []
for folder, _dirs, files in os.walk(root):
    for name in files:
        path = os.path.join(folder, name)
        key = os.path.relpath(path, root)
        if key.startswith(prefix):
            items.append({"Key": key, "Size": os.path.getsize(path)})
print(json.dumps({"Contents": sorted(items, key=lambda item: item["Key"]), "IsTruncated": False}))
' "${HARNESS_VOLUME}" "${prefix}"
            fi
            [[ "$1 $2" == "s3 cp" && "$5" == --only-show-errors ]] || exit 93
            if [[ "$3" == s3://fixture-volume/* ]]; then
                source="${HARNESS_VOLUME}/${3#s3://fixture-volume/}"
                [[ -f "${source}" ]] || exit 1
                if [[ "$4" == - ]]; then cat "${source}"; else cp "${source}" "$4"; fi
            elif [[ "$4" == s3://fixture-volume/* ]]; then
                target="${HARNESS_VOLUME}/${4#s3://fixture-volume/}"
                mkdir -p "$(dirname "${target}")"
                cp "$3" "${target}"
                printf '%s\n' "${4#s3://fixture-volume/}" >> "${HARNESS_VOLUME}.uploads"
            else
                exit 94
            fi
        """)
        write_script(self.project / "scripts/runpodctl_project.sh", "exit 93\n")
        arguments = SELECTION["build_parser"]().parse_args([
            "create", "--project-root", str(self.project), "--stage", "stage2",
            "--data-profile", "us_tw_eodhd", "--start", "2016-01-01",
            "--end", "2026-06-01", "--universe", "all",
        ])
        self.selection = SELECTION["_build_selection"](arguments, self.project)
        SELECTION["_activate_selection"](self.project, self.selection)

    def command(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["/bin/bash", str(self.project / "scripts/runpod_workflow.sh"), *arguments],
            env=self.environment, cwd=self.project,
            capture_output=True, text=True, check=False, timeout=120,
        )

    def relay_env(self) -> None:
        write_env(self.env_file, (
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "TPEX_PROXY_URL=https://relay-fixture.a.run.app\n"
            "RUNPOD_TPEX_PROXY_SECRET_NAME=tpex_relay_token\n"
            "RUNPOD_S3_ACCESS_KEY_ID=fixture\nRUNPOD_S3_SECRET_ACCESS_KEY=fixture\n"
            "RUNPOD_S3_REGION=eu-ro-1\n"
        ))

    def test_forward_pod_prepares_only_the_forward_package(self) -> None:
        self.relay_env()
        pointer = SELECTION["_active_pointer_path"](self.project)
        before = pointer.read_bytes()
        result = self.command("cpu", "prepare", "--forward", END, "--cpuNumber", "8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(pointer.read_bytes(), before)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        self.assertEqual(command[command.index("--name") + 1],
                         "stock-forecasting-cpu-prepare-forward")
        env = json.loads(command[command.index("--env") + 1])
        self.assertEqual(env["RUNPOD_ROLE"], "cpu-prep")
        self.assertEqual((env["RUNPOD_CPU_PREPARE_MODE"], env["RUNPOD_FORWARD_END"]),
                         ("forward", END))
        self.assertEqual(env["DATA_ROOT"], "")
        self.assertEqual(env["RUNPOD_CPU_MAX_RUNTIME_SECONDS"], "10800")
        self.assertEqual(env["TPEX_PROXY_TOKEN"], "{{ RUNPOD_SECRET_tpex_relay_token }}")
        self.assertEqual(
            (env["HF_TOKEN"], env["EODHD_API_TOKEN"], env["RUNPOD_CPU_MAX_API_CALLS"],
             env["RUNPOD_CPU_EODHD_QPS"]),
            ("", "", "", ""),
        )

    def test_forward_refuses_dataset_options_bad_dates_and_a_missing_relay(self) -> None:
        self.relay_env()
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
        for arguments, message in (
            (("--forward", END, "--max-api-calls", "5"), "--max-api-calls"),
            (("--forward", END, "--interactive"), "--interactive"),
            (("--forward", "2026-02-30"), "YYYY-MM-DD"),
            (("--forward", tomorrow), "later than today"),
        ):
            with self.subTest(arguments=arguments):
                result = self.command("cpu", "prepare", *arguments)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("cpu", "prepare", "--forward", END)
        self.assertEqual(result.returncode, 2)
        self.assertIn("TPEx relay secret name is missing", result.stderr)

    def write_cache_export(self, files: dict[str, bytes]) -> Path:
        export = self.project / "artifacts/forward" / END / "shared-cache"
        for relative, body in files.items():
            (export / relative).parent.mkdir(parents=True, exist_ok=True)
            (export / relative).write_bytes(body)
        (export / "shared-cache.json").write_text(json.dumps({
            "kind": "forward-shared-cache-export", "end_exclusive": END,
            "rekeyed_complete_months": 1, "skipped_unfinished_months": ["x"],
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        }))
        return export

    def test_cache_upload_adds_new_entries_and_never_overwrites(self) -> None:
        self.relay_env()
        entry = "twse_official/ab/" + "ab" * 32 + ".json"
        kept = "tpex_official/cd/" + "cd" * 32 + ".json"
        remote = self.volume / "datasets/forward-taiwan/api-cache"
        (remote / kept).parent.mkdir(parents=True)
        (remote / kept).write_bytes(b'{"kept":1}')
        self.write_cache_export({entry: b'{"new":1}', kept: b'{"kept":1}'})
        result = self.command("upload-forward-cache", END)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((remote / entry).read_bytes(), b'{"new":1}')
        uploads = Path(f"{self.volume}.uploads").read_text().splitlines()
        self.assertEqual(sorted(uploads), sorted([
            f"datasets/forward-taiwan/api-cache/{entry}",
            f"datasets/forward-taiwan/exports/{END}.json",
        ]))
        # A different body under an existing key stops the upload before any transfer.
        (remote / kept).write_bytes(b'{"other":12}')
        result = self.command("upload-forward-cache", END)
        self.assertEqual(result.returncode, 1)
        self.assertIn("differ from the volume", result.stderr)
        self.assertEqual((remote / kept).read_bytes(), b'{"other":12}')

    def write_remote_package(self) -> dict:
        package = self.volume / "forward" / END / "package"
        files = {"bar-store/a.parquet": b"bars", "sample-universes/u/universe.json": b"{}"}
        for relative, body in files.items():
            (package / relative).parent.mkdir(parents=True, exist_ok=True)
            (package / relative).write_bytes(body)
        listing = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
        payload = {"kind": "forward-input-package", "end_exclusive": END, "files": listing,
                   "package_sha256": canonical_sha256(listing)}
        (package / "forward-package.json").write_text(json.dumps(payload))
        return payload

    def test_package_download_verifies_every_file(self) -> None:
        self.relay_env()
        payload = self.write_remote_package()
        result = self.command("download-forward-package", END)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        local = self.project / "artifacts/forward" / END / "package"
        self.assertEqual((local / "bar-store/a.parquet").read_bytes(), b"bars")
        self.assertIn(payload["package_sha256"], result.stdout)
        again = self.command("download-forward-package", END)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("already downloaded", again.stdout)

    def test_package_download_refuses_a_changed_file(self) -> None:
        self.relay_env()
        self.write_remote_package()
        (self.volume / "forward" / END / "package/bar-store/a.parquet").write_bytes(b"other")
        result = self.command("download-forward-package", END)
        self.assertEqual(result.returncode, 1)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse((self.project / "artifacts/forward" / END / "package").exists())


if __name__ == "__main__":
    unittest.main()
