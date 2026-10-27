"""Dependency-free checks of the cpu-lgbm workflow, its lifecycle marker and its download."""

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
DOWNLOAD = runpy.run_path(str(ROOT / "scripts/download_runpod_lgbm.py"))
KIND = "research-lgbm"
KEY = "lifecycle/research/lgbm.json"
LAUNCH = "launch-20261023T000000Z-1"
DATASET = "a" * 64


def write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o700)


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


class LgbmLifecycleTests(unittest.TestCase):
    def test_key_maps_to_the_lgbm_kind_in_every_allowlist(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; runpod_guard_lifecycle_kind "$2"',
                "kind",
                str(ROOT / "scripts/lib/runpod_paths.sh"),
                KEY,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual((result.returncode, result.stdout.strip()), (0, KIND), result.stderr)
        self.assertIn(KIND, READINESS["GUARD_LIFECYCLE_KINDS"])
        guard = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
        self.assertIn(f'expected_lifecycle_kind="{KIND}"', guard)
        self.assertIn("|research-lgbm|", guard)
        launcher = (ROOT / "scripts/launch_runpod_guard.sh").read_text(encoding="utf-8")
        self.assertIn("|research-lgbm|", launcher)
        status = (ROOT / "scripts/show_runpod_status.sh").read_text(encoding="utf-8")
        self.assertIn(f"summarize_marker {KEY} lgbm", status)

    def test_write_state_and_guard_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary).resolve()
            marker = volume / KEY

            def write(state: str, *extra: str):
                return subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/runpod_readiness.py"),
                        "write-state",
                        "--output",
                        str(marker),
                        "--network-volume-root",
                        str(volume),
                        "--kind",
                        KIND,
                        "--state",
                        state,
                        "--launch-id",
                        LAUNCH,
                        *extra,
                    ],
                    env={**os.environ, "RUNPOD_POD_ID": "pod-lgbm"},
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )

            def guard_state(pod_id: str) -> subprocess.CompletedProcess:
                return subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/runpod_readiness.py"),
                        "guard-lifecycle-state",
                        "--expected-kind",
                        KIND,
                        "--expected-pod-id",
                        pod_id,
                    ],
                    input=marker.read_text(encoding="utf-8"),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )

            for state, extra in (
                ("preparing", ()),
                ("failed", ("--exit-code", "1")),
                ("ready", ()),
            ):
                with self.subTest(state=state):
                    result = write(state, *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(guard_state("pod-lgbm").stdout.strip(), state)
            self.assertNotEqual(guard_state("another-pod").returncode, 0)

    def test_recovery_distinguishes_the_baseline_from_the_feature_build(self) -> None:
        pod = {
            "id": "pod-lgbm",
            "networkVolumeId": "volume-id",
            "env": {
                "RUNPOD_ROLE": "cpu-research",
                "RUNPOD_STAGE": "stage1",
                "RUNPOD_CPU_WORKFLOW": "lgbm",
                "NETWORK_VOLUME_ROOT": "/runpod-volume",
                "PROJECT_ROOT": "/runpod-volume/stock_forecasting",
            },
        }
        self.assertTrue(RECOVERY["project_pod"](pod, "volume-id"))
        self.assertEqual(RECOVERY["expected_lifecycle"](pod), (KIND, KEY))
        pod["env"]["RUNPOD_CPU_WORKFLOW"] = "features"
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod),
            ("research-features", "lifecycle/research/features.json"),
        )

    def test_launcher_creator_and_console_entry_know_the_workflow(self) -> None:
        launcher = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
        self.assertIn("SESSION_NAME=stock-forecasting-cpu-lgbm", launcher)
        self.assertIn("FAILURE_LIFECYCLE_KIND=research-lgbm", launcher)
        create = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_CPU_TMUX_WORKFLOW=cpu-lgbm", create)
        scripts = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        entry = (
            'stock-forecasting-walk-forward-lgbm = "stock_forecasting.cli.walk_forward_lgbm:main"'
        )
        self.assertIn(entry, scripts)
        self.assertIn('"lightgbm>=', scripts)
        workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
        self.assertIn("    lgbm)\n", workflow)
        self.assertIn("    download-lgbm)\n", workflow)


class LgbmTmuxTests(unittest.TestCase):
    def harness(self) -> test_probe_tmux.ProbeHarness:
        harness = test_probe_tmux.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        source = (harness.scripts / "runpod_cpu_prepare.sh").read_text(encoding="utf-8")
        (harness.scripts / "runpod_cpu_lgbm.sh").write_text(source, encoding="utf-8")
        (harness.scripts / "runpod_cpu_lgbm.sh").chmod(0o700)
        return harness

    def test_runner_self_terminates_and_finalizes_failures_only(self) -> None:
        for exit_code, state in (("0", "succeeded"), ("1", "failed")):
            with self.subTest(exit_code=exit_code):
                harness = self.harness()
                harness.environment["HARNESS_JOB_EXIT"] = exit_code
                launched = harness.command("runpod_tmux_launch.sh", "cpu-lgbm")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-cpu-lgbm", launched.stdout)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, int(exit_code), runner.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.read("worker-env").splitlines()[0], "cpu-research")
                self.assertEqual(harness.read("shutdown-called").strip(), "cpu-research")
                self.assertFalse((harness.root / "lease-calls").exists())
                calls_path = harness.root / "lifecycle-calls"
                if exit_code == "0":
                    self.assertFalse(calls_path.exists())
                else:
                    calls = calls_path.read_text(encoding="utf-8")
                    self.assertIn(str(harness.volume / KEY), calls)
                    self.assertIn(f"--kind {KIND} --state failed", calls)

    def test_job_quarantines_stale_sources_then_tests_then_trains(self) -> None:
        job = (ROOT / "scripts/runpod_cpu_lgbm.sh").read_text(encoding="utf-8")
        quarantine = job.index('runpod_readiness.py" quarantine-stale-code')
        self.assertLess(job.index("write_features_state preparing\n"), quarantine)
        self.assertLess(quarantine, job.index('runpod_readiness.py" check-code'))
        self.assertLess(job.index("check-code"), job.index("tests/test_walk_forward_lgbm.py"))
        self.assertLess(
            job.index("tests/test_walk_forward_lgbm.py"),
            job.index("stock-forecasting-walk-forward-lgbm"),
        )
        self.assertIn("--kind research-lgbm", job)
        self.assertIn("diagnostics/lgbm/${LAUNCH_ID}", job)
        self.assertIn("configs/walk_forward/tw_quarterly_v1.json", job)

    def test_job_resumes_only_from_an_unfinished_launch(self) -> None:
        """A launch that wrote results.json is a result, never a resume source: the newest
        unfinished launch wins even when finished launches are newer."""
        job = (ROOT / "scripts/runpod_cpu_lgbm.sh").read_text(encoding="utf-8")
        start = job.index('RESUME_ROOT=""\n')
        end = job.index("done\n", start) + len("done\n")
        loop = job[start:end]
        self.assertIn('! -f "$(dirname "${resume_candidate}")/results.json"', loop)
        with tempfile.TemporaryDirectory() as tmp:
            lgbm = Path(tmp, "diagnostics", "lgbm")
            for name, finished in (
                ("launch-20261026T215000Z-1", True),
                ("launch-20261027T010000Z-2", False),
                ("launch-20261027T065055Z-3", True),
            ):
                (lgbm / name / "partial").mkdir(parents=True)
                if finished:
                    (lgbm / name / "results.json").write_text("{}", encoding="utf-8")
            (lgbm / "launch-20261027T070000Z-4").mkdir()  # no partial/ at all
            script = f'NETWORK_VOLUME_ROOT={shlex.quote(tmp)}\n{loop}printf "%s" "$RESUME_ROOT"\n'
            chosen = subprocess.run(
                ["bash", "-c", script], check=True, capture_output=True, text=True
            ).stdout
            self.assertEqual(Path(chosen), lgbm / "launch-20261027T010000Z-2")
            for name in ("launch-20261027T010000Z-2",):
                (lgbm / name / "results.json").write_text("{}", encoding="utf-8")
            chosen = subprocess.run(
                ["bash", "-c", script], check=True, capture_output=True, text=True
            ).stdout
            self.assertEqual(chosen, "")  # every launch finished: a fresh run, no resume

    def test_job_refuses_another_role_dataset_or_config(self) -> None:
        script = ROOT / "scripts/runpod_cpu_lgbm.sh"
        base = {
            "PATH": os.defpath,
            "RUNPOD_TEST_MODE": "1",
            "NETWORK_VOLUME_ROOT": "/runpod-volume",
            "RUNPOD_LAUNCH_ID": LAUNCH,
            "DATA_ROOT": "/runpod-volume/datasets/" + DATASET,
        }
        for overrides, message in (
            ({"RUNPOD_ROLE": "cpu-prep"}, "cpu-research role"),
            (
                {"RUNPOD_ROLE": "cpu-research", "DATA_ROOT": "/runpod-volume/data"},
                "experiment dataset",
            ),
            ({"RUNPOD_ROLE": "cpu-research", "RUNPOD_LAUNCH_ID": "../x"}, "RUNPOD_LAUNCH_ID"),
            (
                {"RUNPOD_ROLE": "cpu-research", "FEATURE_EXPERIMENT_CONFIG": "../x.yaml"},
                "configs/experiments",
            ),
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    ["bash", str(script)],
                    env={**base, **overrides},
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)


class LgbmControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-lgbm-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        for folder in ("scripts", "src", "configs"):
            shutil.copytree(
                ROOT / folder,
                self.project / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(ROOT / "pyproject.toml", self.project / "pyproject.toml")
        self.env_file = self.project / ".env"
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "python3").symlink_to(sys.executable)
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("RUNPOD_", "FIN_TS_", "WANDB_", "TPEX_", "AWS_", "STAGE1_"))
            and key not in {"PROJECT_ROOT", "DATA_ROOT", "EODHD_API_TOKEN"}
        }
        self.volume = self.root / "volume"
        self.environment.update(
            {
                "PATH": f"{binary}:{os.defpath}",
                "RUNPOD_ENV_FILE": str(self.env_file),
                "RUNPOD_TEST_MODE": "1",
                "RUNPOD_TEST_READINESS_READY": "1",
                "HARNESS_VOLUME": str(self.volume),
            }
        )
        write_script(
            self.project / "scripts/runpod_s3_project.sh",
            r"""
            [[ "$1 $2" == "s3 cp" && "$3" == s3://fixture-volume/* \
                && "$5" == --only-show-errors ]] || exit 93
            source="${HARNESS_VOLUME}/${3#s3://fixture-volume/}"
            [[ -f "${source}" ]] || exit 1
            if [[ "$4" == - ]]; then cat "${source}"; else cp "${source}" "$4"; fi
        """,
        )
        write_script(self.project / "scripts/runpodctl_project.sh", "exit 93\n")
        arguments = SELECTION["build_parser"]().parse_args(
            [
                "create",
                "--project-root",
                str(self.project),
                "--stage",
                "stage1",
                "--data-profile",
                "tw_only",
                "--start",
                "2016-01-01",
                "--end",
                "2026-06-01",
                "--universe",
                "all",
            ]
        )
        self.selection = SELECTION["_build_selection"](arguments, self.project)
        SELECTION["_activate_selection"](self.project, self.selection)

    def command(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["/bin/bash", str(self.project / "scripts/runpod_workflow.sh"), *arguments],
            env=self.environment,
            cwd=self.project,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def test_lgbm_pod_defaults_to_24h_and_carries_no_secrets(self) -> None:
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("lgbm", "--cpuNumber", "16")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        self.assertEqual(command[command.index("--name") + 1], "stock-forecasting-cpu-lgbm")
        env = json.loads(command[command.index("--env") + 1])
        self.assertEqual(env["RUNPOD_ROLE"], "cpu-research")
        self.assertEqual(env["RUNPOD_CPU_WORKFLOW"], "lgbm")
        self.assertEqual(
            env["DATA_ROOT"], "/runpod-volume/datasets/" + self.selection["dataset_request_sha256"]
        )
        self.assertEqual(env["RUNPOD_CPU_MAX_RUNTIME_SECONDS"], "86400")
        for secret in ("HF_TOKEN", "EODHD_API_TOKEN", "TPEX_PROXY_TOKEN", "TPEX_PROXY_URL"):
            self.assertEqual(env[secret], "", secret)
        for arguments in (("lgbm", "--unknown", "1"), ("lgbm", "--cpuNumber", "3")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.command(*arguments).returncode, 2)

    def test_lgbm_pod_carries_the_requested_walk_forward_schedule(self) -> None:
        """--schedule (protocol 13.20) reaches the Pod as RUNPOD_WALK_FORWARD_SCHEDULE; the job
        script defaults to tw_quarterly_v1 and validates the override."""
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("lgbm", "--schedule", "configs/walk_forward/tw_quarterly_v2.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        env = json.loads(command[command.index("--env") + 1])
        self.assertEqual(
            env["RUNPOD_WALK_FORWARD_SCHEDULE"], "configs/walk_forward/tw_quarterly_v2.json"
        )
        plain = self.command("lgbm")
        line = next(line for line in plain.stdout.splitlines() if "--name" in line)
        env = json.loads(shlex.split(line.removeprefix("DRY RUN:"))[
            shlex.split(line.removeprefix("DRY RUN:")).index("--env") + 1
        ])
        self.assertEqual(env["RUNPOD_WALK_FORWARD_SCHEDULE"], "")
        for arguments in (
            ("lgbm", "--schedule", "configs/walk_forward/missing.json"),
            ("lgbm", "--schedule", "../etc/passwd"),
            ("features", "--schedule", "configs/walk_forward/tw_quarterly_v2.json"),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.command(*arguments).returncode, 2)
        job = (ROOT / "scripts/runpod_cpu_lgbm.sh").read_text(encoding="utf-8")
        self.assertIn(
            'WALK_FORWARD_SCHEDULE="${RUNPOD_WALK_FORWARD_SCHEDULE:-configs/walk_forward/'
            'tw_quarterly_v1.json}"',
            job,
        )
        self.assertIn("RUNPOD_WALK_FORWARD_SCHEDULE must name a file under", job)

    def test_download_reports_running_missing_and_verified_results(self) -> None:
        write_env(
            self.env_file,
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_ACCESS_KEY_ID=fixture\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture\nRUNPOD_S3_REGION=eu-ro-1\n",
        )
        marker = self.volume / KEY
        marker.parent.mkdir(parents=True)
        payload = {"kind": KIND, "state": "preparing", "launch_id": LAUNCH}
        marker.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.command("download-lgbm").returncode, 3)
        payload["state"] = "ready"
        marker.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.command("download-lgbm").returncode, 4)
        remote = self.volume / "diagnostics/lgbm" / LAUNCH
        remote.mkdir(parents=True)
        results = {
            "model": "t-lgbm-alpha",
            "backend": "lightgbm",
            "fits": {"count": 2},
            "cells": [
                {
                    "model": "t-lgbm-alpha",
                    "cell": "short",
                    "period": "P1+P2",
                    "rank_ic": 0.05,
                    "ci95_low": 0.01,
                    "ci95_high": 0.09,
                    "rank_icir": 0.4,
                    "days": 10,
                }
            ],
        }
        (remote / "results.json").write_text(json.dumps(results), encoding="utf-8")
        (remote / "predictions.parquet").write_bytes(b"parquet")
        files = {
            name: {"sha256": __import__("hashlib").sha256((remote / name).read_bytes()).hexdigest()}
            for name in ("results.json", "predictions.parquet")
        }
        (remote / "walk-forward-manifest.json").write_text(
            json.dumps({"kind": "walk-forward-lgbm", "files": files}), encoding="utf-8"
        )
        result = self.command("download-lgbm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("short: rank_ic=0.0500", result.stdout)
        local = self.project / "artifacts/lgbm" / LAUNCH
        self.assertEqual((local / "predictions.parquet").read_bytes(), b"parquet")
        (remote / "predictions.parquet").write_bytes(b"tampered")
        shutil.rmtree(local)
        self.assertNotEqual(self.command("download-lgbm").returncode, 0)
        self.assertEqual(self.command("download-lgbm", "bad-id").returncode, 2)
        self.assertEqual(DOWNLOAD["summary_lines"](results)[1].split(":")[0], "short")


if __name__ == "__main__":
    unittest.main()
