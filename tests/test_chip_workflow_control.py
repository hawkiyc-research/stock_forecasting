"""Dependency-free checks of the tw_chip_data Pod workflow (protocol 13.20): its lifecycle
kind, the cpu-chip launcher, the chain after a dataset preparation, the chip Pod and the job's
refusals."""

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

import test_probe_tmux

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))
KIND = "research-chip-data"
KEY = "lifecycle/research/chip-data.json"
LAUNCH = "launch-20261028T000000Z-1"


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o700)


class ChipLifecycleTests(unittest.TestCase):
    def test_key_maps_to_the_chip_kind_in_every_allowlist(self) -> None:
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; runpod_guard_lifecycle_kind "$2"', "kind",
             str(ROOT / "scripts/lib/runpod_paths.sh"), KEY],
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual((result.returncode, result.stdout.strip()), (0, KIND), result.stderr)
        self.assertIn(KIND, READINESS["GUARD_LIFECYCLE_KINDS"])
        terminate = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
        self.assertIn(f'expected_lifecycle_kind="{KIND}"', terminate)
        guard = (ROOT / "scripts/launch_runpod_guard.sh").read_text(encoding="utf-8")
        self.assertIn(KIND, guard)
        status = (ROOT / "scripts/show_runpod_status.sh").read_text(encoding="utf-8")
        self.assertIn(f"summarize_marker {KEY} chip_data", status)
        launcher = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
        self.assertIn("SESSION_NAME=stock-forecasting-cpu-chip", launcher)
        self.assertIn(f"FAILURE_LIFECYCLE_KIND={KIND}", launcher)
        self.assertIn("CHAIN_CHIP_UPDATE=1", launcher)
        create = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_CPU_TMUX_WORKFLOW=cpu-chip", create)
        self.assertIn('"RUNPOD_CHIP_DATA_PREPARE":"%s"', create)
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
        self.assertIn("--skipChip|--skip-chip)", workflow)
        self.assertIn("export RUNPOD_CPU_WORKFLOW=chip", workflow)
        job = (ROOT / "scripts/runpod_cpu_chip.sh").read_text(encoding="utf-8")
        self.assertIn(f"--kind {KIND}", job)
        self.assertIn('TWCHIP_RELAY_ORIGIN="${TPEX_PROXY_URL%/}"', job)
        self.assertIn("setup_runpod_chip_environment.sh", job)

    def test_write_state_and_guard_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            marker = volume / KEY

            def write(state: str, *extra: str):
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/runpod_readiness.py"), "write-state",
                     "--output", str(marker), "--network-volume-root", str(volume),
                     "--kind", KIND, "--state", state, "--launch-id", LAUNCH, *extra],
                    env={**os.environ, "RUNPOD_POD_ID": "pod-chip"},
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

            for state, extra in (("preparing", ()), ("failed", ("--exit-code", "3")),
                                 ("ready", ())):
                with self.subTest(state=state):
                    result = write(state, *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(guard_state("pod-chip").stdout.strip(), state)
            self.assertNotEqual(guard_state("another-pod").returncode, 0)

    def test_recovery_distinguishes_the_chip_update_from_a_dataset_preparation(self) -> None:
        pod = {
            "id": "pod-chip", "networkVolumeId": "volume-id",
            "env": {"RUNPOD_ROLE": "cpu-prep", "RUNPOD_STAGE": "stage1",
                    "RUNPOD_CPU_WORKFLOW": "chip",
                    "NETWORK_VOLUME_ROOT": "/runpod-volume",
                    "PROJECT_ROOT": "/runpod-volume/stock_forecasting"},
        }
        self.assertTrue(RECOVERY["project_pod"](pod, "volume-id"))
        self.assertEqual(RECOVERY["expected_lifecycle"](pod), (KIND, KEY))
        pod["env"]["RUNPOD_CPU_WORKFLOW"] = "prepare"
        pod["env"]["RUNPOD_CHIP_DATA_PREPARE"] = "1"
        # A preparation Pod that chains the chip update still owns the preparation marker.
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod),
            ("stage1-cpu-preparation", "lifecycle/stage1/cpu-preparation.json"),
        )


class ChipTmuxTests(unittest.TestCase):
    def harness(self) -> test_probe_tmux.ProbeHarness:
        harness = test_probe_tmux.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        source = (harness.scripts / "runpod_cpu_prepare.sh").read_text(encoding="utf-8")
        (harness.scripts / "runpod_cpu_chip.sh").write_text(source, encoding="utf-8")
        (harness.scripts / "runpod_cpu_chip.sh").chmod(0o700)
        return harness

    def test_cpu_chip_runner_self_terminates_and_finalizes_failures_only(self) -> None:
        for exit_code, state in (("0", "succeeded"), ("1", "failed")):
            with self.subTest(exit_code=exit_code):
                harness = self.harness()
                harness.environment["HARNESS_JOB_EXIT"] = exit_code
                launched = harness.command("runpod_tmux_launch.sh", "cpu-chip")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-cpu-chip", launched.stdout)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, int(exit_code), runner.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.read("worker-env").splitlines()[0], "cpu-prep")
                self.assertEqual(harness.read("shutdown-called").strip(), "cpu-prep")
                calls_path = harness.root / "lifecycle-calls"
                if exit_code == "0":
                    self.assertFalse(calls_path.exists())
                else:
                    calls = calls_path.read_text(encoding="utf-8")
                    self.assertIn(str(harness.volume / KEY), calls)
                    self.assertIn(f"--kind {KIND} --state failed", calls)

    def test_dataset_preparation_chains_the_chip_update_only_on_success(self) -> None:
        for prepare_exit, chained, chip_flag in (("0", True, "1"), ("1", False, "1"),
                                                 ("0", False, "0")):
            with self.subTest(prepare_exit=prepare_exit, chip_flag=chip_flag):
                harness = self.harness()
                write_script(harness.scripts / "runpod_cpu_chip.sh", r"""
                    printf '%s\n' "${RUNPOD_LAUNCH_ID}" "${RUNPOD_ROLE}" \
                        > "${HARNESS_ROOT}/chip-ran"
                    exit 0
                """)
                harness.environment["HARNESS_JOB_EXIT"] = prepare_exit
                harness.environment["RUNPOD_CHIP_DATA_PREPARE"] = chip_flag
                # The chain needs more than 15 minutes of runtime left after the preparation.
                harness.environment["RUNPOD_CPU_MAX_RUNTIME_SECONDS"] = "7200"
                launched = harness.command("runpod_tmux_launch.sh", "cpu-prepare")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, int(prepare_exit), runner.stderr)
                # The preparation's own status and shutdown are unchanged by the chain.
                self.assertEqual(
                    harness.status()["state"], "succeeded" if prepare_exit == "0" else "failed"
                )
                self.assertEqual(harness.read("shutdown-called").strip(), "cpu-prep")
                ran = harness.root / "chip-ran"
                self.assertEqual(ran.exists(), chained, runner.stdout + runner.stderr)
                if chained:
                    launch_id, role = ran.read_text(encoding="utf-8").split()
                    self.assertTrue(launch_id.endswith("-chip"), launch_id)
                    self.assertEqual(role, "cpu-prep")

    def test_short_remaining_runtime_skips_the_chain_with_a_notice(self) -> None:
        harness = self.harness()
        write_script(harness.scripts / "runpod_cpu_chip.sh", r"""
            printf 'ran\n' > "${HARNESS_ROOT}/chip-ran"
        """)
        harness.environment["RUNPOD_CHIP_DATA_PREPARE"] = "1"
        harness.environment["RUNPOD_CPU_MAX_RUNTIME_SECONDS"] = "600"
        self.assertEqual(harness.command("runpod_tmux_launch.sh", "cpu-prepare").returncode, 0)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertFalse((harness.root / "chip-ran").exists())
        self.assertIn("Skipping the chip-data update", runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "succeeded")

    def test_chip_job_refuses_wrong_role_missing_source_and_bad_relay(self) -> None:
        script = ROOT / "scripts/runpod_cpu_chip.sh"
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            project = volume / "tw_chip_data"
            (project / "src/tw_chip_data").mkdir(parents=True)
            (project / "configs").mkdir()
            base = {
                "PATH": os.defpath, "RUNPOD_TEST_MODE": "1",
                "NETWORK_VOLUME_ROOT": str(volume), "RUNPOD_LAUNCH_ID": LAUNCH,
                "RUNPOD_ROLE": "cpu-prep",
                "TPEX_PROXY_URL": "https://relay-fixture.a.run.app",
                "TPEX_PROXY_TOKEN": "fixture-token-that-is-longer-than-thirty-two-characters",
            }
            result = subprocess.run(
                ["bash", str(script)], env={**base, "RUNPOD_ROLE": "cpu-research"},
                capture_output=True, text=True, check=False, timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("cpu-prep role", result.stderr)
            result = subprocess.run(
                ["bash", str(script)], env=base,
                capture_output=True, text=True, check=False, timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("tw_chip_data source is missing", result.stderr)
            for name in ("src/tw_chip_data/cli.py", "configs/sources.yaml",
                         "requirements-runpod.txt"):
                (project / name).write_text("# fixture\n", encoding="utf-8")
            for overrides, message in (
                ({"TPEX_PROXY_URL": "https://example.com"}, "approved run.app origin"),
                ({"TPEX_PROXY_TOKEN": ""}, "TPEX_PROXY_TOKEN"),
                ({"RUNPOD_CHIP_REFETCH_DAYS": "x"}, "RUNPOD_CHIP_REFETCH_DAYS"),
                ({"STAGE1_DATA_END": "2026/10/23"}, "STAGE1_DATA_END"),
            ):
                with self.subTest(message=message):
                    result = subprocess.run(
                        ["bash", str(script)], env={**base, **overrides},
                        capture_output=True, text=True, check=False, timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)


class ChipControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-chip-")
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
        })
        write_script(self.project / "scripts/runpod_s3_project.sh", "exit 93\n")
        write_script(self.project / "scripts/runpodctl_project.sh", "exit 93\n")
        arguments = SELECTION["build_parser"]().parse_args([
            "create", "--project-root", str(self.project), "--stage", "stage1",
            "--data-profile", "tw_only", "--start", "2016-01-01",
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

    @staticmethod
    def pod_env(result: subprocess.CompletedProcess) -> tuple[list[str], dict]:
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        return command, json.loads(command[command.index("--env") + 1])

    def test_chip_pod_is_a_relay_carrying_preparation_pod(self) -> None:
        write_env(self.env_file, (
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "TPEX_PROXY_URL=https://relay-fixture.a.run.app\n"
            "RUNPOD_TPEX_PROXY_SECRET_NAME=tpex_relay_token\n"
        ))
        result = self.command("chip", "--maxRuntime", "2h", "--cpuNumber", "4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command, env = self.pod_env(result)
        self.assertEqual(command[command.index("--name") + 1], "stock-forecasting-cpu-chip")
        self.assertEqual(command[command.index("--vcpu-count") + 1], "4")
        self.assertEqual(env["RUNPOD_ROLE"], "cpu-prep")
        self.assertEqual(env["RUNPOD_CPU_WORKFLOW"], "chip")
        self.assertEqual(env["RUNPOD_CHIP_DATA_PREPARE"], "0")
        self.assertEqual(env["RUNPOD_CPU_MAX_RUNTIME_SECONDS"], "7200")
        self.assertEqual(env["TPEX_PROXY_URL"], "https://relay-fixture.a.run.app")
        self.assertEqual(env["TPEX_PROXY_TOKEN"], "{{ RUNPOD_SECRET_tpex_relay_token }}")
        self.assertEqual((env["STAGE1_DATA_END"], env["RUNPOD_CPU_MAX_API_CALLS"]),
                         ("2026-06-01", ""))
        for arguments in (("chip", "--unknown", "1"), ("chip", "--cpuNumber", "3")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.command(*arguments).returncode, 2)

    def test_chip_pod_requires_the_relay_and_other_pods_never_chain(self) -> None:
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("chip")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("TPEx relay secret name is missing", result.stderr)
        features = self.command("features", "--cpuNumber", "16")
        self.assertEqual(features.returncode, 0, features.stdout + features.stderr)
        _command, env = self.pod_env(features)
        self.assertEqual((env["RUNPOD_CPU_WORKFLOW"], env["RUNPOD_CHIP_DATA_PREPARE"]),
                         ("features", "0"))


if __name__ == "__main__":
    unittest.main()
