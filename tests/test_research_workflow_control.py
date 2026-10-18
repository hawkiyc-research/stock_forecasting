"""Dependency-free checks of the CPU research workflow, its lifecycle and its guard."""

from __future__ import annotations

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
from pathlib import Path

import test_probe_guard
import test_probe_tmux

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))
KIND = "research-tw-information"
KEY = "lifecycle/research/tw-information.json"
LAUNCH = "launch-20261018T000000Z-1"


def write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o700)


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


class ResearchLifecycleTests(unittest.TestCase):
    def test_key_maps_to_the_research_kind_in_every_allowlist(self) -> None:
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; runpod_guard_lifecycle_kind "$2"', "kind",
             str(ROOT / "scripts/lib/runpod_paths.sh"), KEY],
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual((result.returncode, result.stdout.strip()), (0, KIND), result.stderr)
        self.assertIn(KIND, READINESS["GUARD_LIFECYCLE_KINDS"])
        guard = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
        self.assertIn(f'expected_lifecycle_kind="{KIND}"', guard)
        # A relative log path fails only after the launcher accepted the lifecycle kind.
        launched = subprocess.run(
            ["bash", str(ROOT / "scripts/launch_runpod_guard.sh"), "pod-research", "60", KEY,
             "relative.log"],
            env={"PATH": os.defpath, "HOME": "/nonexistent"},
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual(launched.returncode, 2)
        self.assertIn("Guard log path must be absolute", launched.stderr)

    def test_write_state_and_guard_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            marker = volume / KEY

            def write(state: str, output: Path = marker, *extra: str):
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/runpod_readiness.py"), "write-state",
                     "--output", str(output), "--network-volume-root", str(volume),
                     "--kind", KIND, "--state", state, "--launch-id", LAUNCH, *extra],
                    env={**os.environ, "RUNPOD_POD_ID": "pod-research"},
                    capture_output=True, text=True, check=False, timeout=30,
                )

            def guard_state(pod_id: str) -> subprocess.CompletedProcess:
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/runpod_readiness.py"),
                     "guard-lifecycle-state", "--expected-kind", KIND,
                     "--expected-pod-id", pod_id],
                    input=marker.read_text(encoding="utf-8"),
                    capture_output=True, text=True, check=False, timeout=30,
                )

            other = volume / "lifecycle/stage1/cpu-preparation.json"
            self.assertNotEqual(write("preparing", other).returncode, 0)
            for state, extra in (("preparing", ()), ("failed", ("--exit-code", "1")),
                                 ("ready", ())):
                with self.subTest(state=state):
                    result = write(state, marker, *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(guard_state("pod-research").stdout.strip(), state)
            self.assertNotEqual(guard_state("another-pod").returncode, 0)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual((payload["kind"], payload["launch_id"]), (KIND, LAUNCH))

    def test_recovery_and_environment_import_know_the_research_role(self) -> None:
        pod = {
            "id": "pod-research", "networkVolumeId": "volume-id",
            "env": {"RUNPOD_ROLE": "cpu-research", "RUNPOD_STAGE": "stage2",
                    "NETWORK_VOLUME_ROOT": "/runpod-volume",
                    "PROJECT_ROOT": "/runpod-volume/stock_forecasting"},
        }
        self.assertTrue(RECOVERY["project_pod"](pod, "volume-id"))
        self.assertEqual(RECOVERY["expected_lifecycle"](pod), (KIND, KEY))
        pod["env"]["RUNPOD_ROLE"] = "cpu-prep"
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod)[1], "lifecycle/stage1/mixed-finalization.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "environ"
            names = {
                "RUNPOD_POD_ID": "pod-research", "RUNPOD_ROLE": "cpu-research",
                "EODHD_API_TOKEN": "eodhd-fixture", "TPEX_PROXY_TOKEN": "relay-fixture",
                "HF_TOKEN": "hf-fixture", "WANDB_API_KEY": "wandb-fixture",
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
        self.assertEqual(imported["EODHD_API_TOKEN"], "eodhd-fixture")
        self.assertEqual(imported["TPEX_PROXY_TOKEN"], "relay-fixture")
        self.assertNotIn("HF_TOKEN", imported)
        self.assertNotIn("WANDB_API_KEY", imported)

    def test_research_selection_config_and_sync_agree(self) -> None:
        config = json.loads(
            (ROOT / "configs/research/tw_information_test.json").read_text(encoding="utf-8")
        )
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
        experiment = "a-absolute-direction64"
        self.assertIn(f"research_experiment={experiment}\n", workflow)
        self.assertEqual(
            SELECTION["EXPERIMENT_CONFIGS"][experiment]["config_path"],
            config["experiment_config"],
        )
        sync = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text(encoding="utf-8")
        self.assertIn('append_manifest_file "configs/research/tw_information_test.json"', sync)


class ResearchGuardTests(unittest.TestCase):
    def harness(self):
        # Reuse the probe guard harness: real guard, fake S3 reads and Pod deletion.
        return test_probe_guard.ProbeGuardTests.harness(self)

    def publish(self, harness, pod_id: str, state: str) -> None:
        marker = harness.volume / KEY
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "schema_version": 1, "kind": KIND, "state": state, "pod_id": pod_id,
            "launch_id": LAUNCH, "generated_at": "2026-10-18T00:00:00+00:00",
        }), encoding="utf-8")

    def run_guard(self, harness, seconds: int) -> subprocess.CompletedProcess:
        return test_probe_guard.ProbeGuardTests.run_guard(
            self, harness, seconds=seconds,
            overrides={"RUNPOD_GUARD_LIFECYCLE_KEY": KEY, "RUNPOD_GUARD_RUN_ID": ""},
        )

    def test_guard_terminates_when_the_research_lifecycle_finishes(self) -> None:
        for state in ("ready", "failed", "timed_out"):
            with self.subTest(state=state):
                harness = self.harness()
                self.publish(harness, "probe-fixture", state)
                result = self.run_guard(harness, seconds=8)
                log = harness.read("local-guard.log")
                self.assertEqual(result.returncode, 0, result.stderr + log)
                self.assertIn(f"termination triggered by lifecycle-{state}", log)
                self.assertEqual(harness.read("local-delete-calls").strip(),
                                 "pod delete probe-fixture")

    def test_guard_ignores_running_or_foreign_research_markers(self) -> None:
        for pod_id, state in (("probe-fixture", "preparing"), ("another-pod", "ready")):
            with self.subTest(pod_id=pod_id, state=state):
                harness = self.harness()
                self.publish(harness, pod_id, state)
                result = self.run_guard(harness, seconds=2)
                log = harness.read("local-guard.log")
                self.assertEqual(result.returncode, 0, result.stderr + log)
                self.assertIn("termination triggered by hard-limit", log)


class ResearchTmuxTests(unittest.TestCase):
    def harness(self) -> test_probe_tmux.ProbeHarness:
        harness = test_probe_tmux.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        source = (harness.scripts / "runpod_cpu_prepare.sh").read_text(encoding="utf-8")
        (harness.scripts / "runpod_cpu_research.sh").write_text(source, encoding="utf-8")
        (harness.scripts / "runpod_cpu_research.sh").chmod(0o700)
        return harness

    def test_runner_self_terminates_and_finalizes_failures_only(self) -> None:
        for exit_code, state in (("0", "succeeded"), ("1", "failed")):
            with self.subTest(exit_code=exit_code):
                harness = self.harness()
                harness.environment["HARNESS_JOB_EXIT"] = exit_code
                launched = harness.command("runpod_tmux_launch.sh", "research-tw-information")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-cpu-research", launched.stdout)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, int(exit_code), runner.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.read("worker-env").splitlines()[0], "cpu-research")
                self.assertEqual(harness.read("shutdown-called").strip(), "cpu-research")
                self.assertFalse((harness.root / "lease-calls").exists())
                calls_path = harness.root / "lifecycle-calls"
                if exit_code == "0":
                    # The job script publishes ready itself; the runner adds nothing.
                    self.assertFalse(calls_path.exists())
                else:
                    calls = calls_path.read_text(encoding="utf-8")
                    self.assertIn(str(harness.volume / KEY), calls)
                    self.assertIn(f"--kind {KIND} --state failed", calls)

    def test_research_job_refuses_another_role_or_dataset(self) -> None:
        script = ROOT / "scripts/runpod_cpu_research.sh"
        base = {
            "PATH": os.defpath, "RUNPOD_TEST_MODE": "1", "NETWORK_VOLUME_ROOT": "/runpod-volume",
            "RUNPOD_LAUNCH_ID": LAUNCH, "STAGE1_DATA_START": "2021-01-01",
            "STAGE1_DATA_END": "2026-06-01",
            "DATA_ROOT": "/runpod-volume/datasets/" + "a" * 64,
        }
        for overrides, message in (
            ({"RUNPOD_ROLE": "cpu-prep"}, "cpu-research role"),
            ({"RUNPOD_ROLE": "cpu-research", "DATA_ROOT": "/runpod-volume/data"},
             "experiment dataset"),
            ({"RUNPOD_ROLE": "cpu-research", "RUNPOD_LAUNCH_ID": "../x"}, "RUNPOD_LAUNCH_ID"),
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    ["bash", str(script)], env={**base, **overrides},
                    capture_output=True, text=True, check=False, timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


class ResearchControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-research-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        for folder in ("scripts", "src", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pyproject.toml", self.project / "pyproject.toml")
        self.env_file = self.project / ".env"
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
            "HARNESS_VOLUME": str(self.root / "volume"),
        })
        # Fake S3 reads come from a local directory; writes and other actions fail closed.
        write_script(self.project / "scripts/runpod_s3_project.sh", r"""
            [[ "$1 $2" == "s3 cp" && "$3" == s3://fixture-volume/* \
                && "$5" == --only-show-errors ]] || exit 93
            source="${HARNESS_VOLUME}/${3#s3://fixture-volume/}"
            [[ -f "${source}" ]] || exit 1
            if [[ "$4" == - ]]; then cat "${source}"; else cp "${source}" "$4"; fi
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
            capture_output=True, text=True, check=False, timeout=60,
        )

    def test_research_pod_uses_the_experiment_dataset_without_changing_selection(self) -> None:
        write_env(self.env_file, (
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "TPEX_PROXY_URL=https://relay-fixture.a.run.app\n"
            "RUNPOD_TPEX_PROXY_SECRET_NAME=tpex_relay_token\n"
        ))
        pointer = SELECTION["_active_pointer_path"](self.project)
        before = pointer.read_bytes()
        result = self.command("research", "tw-information", "--maxRuntime", "2h",
                              "--cpuNumber", "8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(pointer.read_bytes(), before)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        self.assertEqual(command[command.index("--name") + 1], "stock-forecasting-cpu-research")
        self.assertEqual(command[command.index("--vcpu-count") + 1], "8")
        env = json.loads(command[command.index("--env") + 1])
        experiment = SELECTION["_with_experiment"](
            self.selection, "a-absolute-direction64", self.project
        )
        self.assertEqual(env["RUNPOD_ROLE"], "cpu-research")
        self.assertEqual(env["RUNPOD_CONFIG"], "configs/experiments/a_absolute_direction64.yaml")
        self.assertEqual(env["DATA_ROOT"],
                         "/runpod-volume/datasets/" + experiment["dataset_request_sha256"])
        self.assertEqual((env["STAGE1_DATA_START"], env["STAGE1_DATA_END"]),
                         ("2021-01-01", "2026-06-01"))
        self.assertEqual(env["RUNPOD_CPU_MAX_RUNTIME_SECONDS"], "7200")
        self.assertEqual(env["TPEX_PROXY_URL"], "https://relay-fixture.a.run.app")
        self.assertEqual(env["TPEX_PROXY_TOKEN"], "{{ RUNPOD_SECRET_tpex_relay_token }}")
        self.assertEqual(env["EODHD_API_TOKEN"], "{{ RUNPOD_SECRET_eodhd_api_token }}")
        self.assertEqual((env["HF_TOKEN"], env["RUNPOD_CPU_MAX_API_CALLS"]), ("", ""))

    def test_research_pod_requires_the_relay(self) -> None:
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("research", "tw-information")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("TPEx relay secret name is missing", result.stderr)
        for arguments in (("research",), ("research", "tw-information", "--unknown", "1")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.command(*arguments).returncode, 2)

    def download(self, *arguments: str) -> subprocess.CompletedProcess:
        write_env(self.env_file, (
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_ACCESS_KEY_ID=fixture\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture\nRUNPOD_S3_REGION=eu-ro-1\n"
        ))
        return self.command("download-research", *arguments)

    def publish(self, state: str, files: tuple[str, ...]) -> None:
        volume = self.root / "volume"
        marker = volume / KEY
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"kind": KIND, "state": state, "launch_id": LAUNCH}))
        for relative in files:
            path = volume / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)

    def test_download_saves_the_latest_report_and_logs(self) -> None:
        logs = f"logs/tmux/stock-forecasting-cpu-research/{LAUNCH}/"
        report = f"diagnostics/tw-information-test/{LAUNCH}/"
        self.publish("ready", (
            logs + "combined.log", logs + "status.json", report + "report.json",
            report + "summary.md", report + "auxiliary-manifest.json",
        ))
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        local = self.project / "artifacts/diagnostics/tw-information-test" / LAUNCH
        for relative in ("report.json", "summary.md", "auxiliary-manifest.json",
                         "logs/combined.log", "logs/status.json", "lifecycle.json"):
            self.assertTrue((local / relative).is_file(), relative)
        self.assertIn("summary.md", result.stdout)

    def test_download_reports_running_failed_and_invalid_requests(self) -> None:
        logs = f"logs/tmux/stock-forecasting-cpu-research/{LAUNCH}/"
        self.publish("preparing", ())
        self.assertEqual(self.download().returncode, 3)
        self.publish("failed", (logs + "combined.log", logs + "status.json"))
        result = self.download()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        local = self.project / "artifacts/diagnostics/tw-information-test" / LAUNCH
        self.assertTrue((local / "logs/combined.log").is_file())
        self.assertEqual(self.download("../outside").returncode, 2)
        self.assertEqual(self.download(LAUNCH).returncode, 4)


if __name__ == "__main__":
    unittest.main()
