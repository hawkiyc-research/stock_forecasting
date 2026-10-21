"""Dependency-free checks of the CPU feature-build workflow, its lifecycle and its download."""

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
from pathlib import Path

import test_probe_tmux

ROOT = Path(__file__).resolve().parents[1]
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
RECOVERY = runpy.run_path(str(ROOT / "scripts/recover_runpod_after_wake.py"))
KIND = "research-features"
KEY = "lifecycle/research/features.json"
LAUNCH = "launch-20261022T000000Z-1"
DATASET = "a" * 64


def write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o700)


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FeatureLifecycleTests(unittest.TestCase):
    def test_key_maps_to_the_features_kind_in_every_allowlist(self) -> None:
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; runpod_guard_lifecycle_kind "$2"', "kind",
             str(ROOT / "scripts/lib/runpod_paths.sh"), KEY],
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual((result.returncode, result.stdout.strip()), (0, KIND), result.stderr)
        self.assertIn(KIND, READINESS["GUARD_LIFECYCLE_KINDS"])
        guard = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
        self.assertIn(f'expected_lifecycle_kind="{KIND}"', guard)
        status = (ROOT / "scripts/show_runpod_status.sh").read_text(encoding="utf-8")
        self.assertIn(f"summarize_marker {KEY} features", status)
        launched = subprocess.run(
            ["bash", str(ROOT / "scripts/launch_runpod_guard.sh"), "pod-features", "60", KEY,
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
                    env={**os.environ, "RUNPOD_POD_ID": "pod-features"},
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

            other = volume / "lifecycle/research/tw-information.json"
            self.assertNotEqual(write("preparing", other).returncode, 0)
            for state, extra in (("preparing", ()), ("failed", ("--exit-code", "1")),
                                 ("ready", ())):
                with self.subTest(state=state):
                    result = write(state, marker, *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(guard_state("pod-features").stdout.strip(), state)
            self.assertNotEqual(guard_state("another-pod").returncode, 0)

    def test_recovery_distinguishes_the_feature_build_from_the_information_test(self) -> None:
        pod = {
            "id": "pod-features", "networkVolumeId": "volume-id",
            "env": {"RUNPOD_ROLE": "cpu-research", "RUNPOD_STAGE": "stage1",
                    "RUNPOD_CPU_WORKFLOW": "features",
                    "NETWORK_VOLUME_ROOT": "/runpod-volume",
                    "PROJECT_ROOT": "/runpod-volume/stock_forecasting"},
        }
        self.assertTrue(RECOVERY["project_pod"](pod, "volume-id"))
        self.assertEqual(RECOVERY["expected_lifecycle"](pod), (KIND, KEY))
        pod["env"]["RUNPOD_CPU_WORKFLOW"] = "research"
        self.assertEqual(
            RECOVERY["expected_lifecycle"](pod),
            ("research-tw-information", "lifecycle/research/tw-information.json"),
        )

    def test_sync_and_launcher_know_the_feature_workflow(self) -> None:
        sync = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text(encoding="utf-8")
        self.assertIn('append_manifest_file "configs/features/alpha_tw_v1.json"', sync)
        self.assertIn('append_manifest_file "configs/walk_forward/tw_quarterly_v1.json"', sync)
        launcher = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
        self.assertIn("SESSION_NAME=stock-forecasting-cpu-features", launcher)
        self.assertIn("FAILURE_LIFECYCLE_KIND=research-features", launcher)
        create = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
        self.assertIn("RUNPOD_CPU_TMUX_WORKFLOW=cpu-features", create)
        self.assertIn('"RUNPOD_CPU_WORKFLOW":"%s"', create)
        scripts = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        entry = 'stock-forecasting-build-features = "stock_forecasting.cli.build_features:main"'
        self.assertIn(entry, scripts)


class FeatureTmuxTests(unittest.TestCase):
    def harness(self) -> test_probe_tmux.ProbeHarness:
        harness = test_probe_tmux.ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        source = (harness.scripts / "runpod_cpu_prepare.sh").read_text(encoding="utf-8")
        (harness.scripts / "runpod_cpu_features.sh").write_text(source, encoding="utf-8")
        (harness.scripts / "runpod_cpu_features.sh").chmod(0o700)
        return harness

    def test_runner_self_terminates_and_finalizes_failures_only(self) -> None:
        for exit_code, state in (("0", "succeeded"), ("1", "failed")):
            with self.subTest(exit_code=exit_code):
                harness = self.harness()
                harness.environment["HARNESS_JOB_EXIT"] = exit_code
                launched = harness.command("runpod_tmux_launch.sh", "cpu-features")
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertIn("stock-forecasting-cpu-features", launched.stdout)
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

    def test_feature_job_quarantines_stale_sources_before_checking_code(self) -> None:
        job = (ROOT / "scripts/runpod_cpu_features.sh").read_text(encoding="utf-8")
        quarantine = job.index("runpod_readiness.py\" quarantine-stale-code")
        self.assertLess(job.index("write_features_state preparing\n"), quarantine)
        self.assertLess(quarantine, job.index("runpod_readiness.py\" check-code"))
        self.assertLess(job.index("check-code"), job.index("stock-forecasting-build-features"))

    def test_feature_job_refuses_another_role_dataset_or_config(self) -> None:
        script = ROOT / "scripts/runpod_cpu_features.sh"
        base = {
            "PATH": os.defpath, "RUNPOD_TEST_MODE": "1", "NETWORK_VOLUME_ROOT": "/runpod-volume",
            "RUNPOD_LAUNCH_ID": LAUNCH, "DATA_ROOT": "/runpod-volume/datasets/" + DATASET,
        }
        for overrides, message in (
            ({"RUNPOD_ROLE": "cpu-prep"}, "cpu-research role"),
            ({"RUNPOD_ROLE": "cpu-research", "DATA_ROOT": "/runpod-volume/data"},
             "experiment dataset"),
            ({"RUNPOD_ROLE": "cpu-research", "RUNPOD_LAUNCH_ID": "../x"}, "RUNPOD_LAUNCH_ID"),
            ({"RUNPOD_ROLE": "cpu-research", "FEATURE_EXPERIMENT_CONFIG": "../x.yaml"},
             "configs/experiments"),
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    ["bash", str(script)], env={**base, **overrides},
                    capture_output=True, text=True, check=False, timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)


class FeatureControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stock-forecasting-features-")
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
        self.volume = self.root / "volume"
        self.environment.update({
            "PATH": f"{binary}:{os.defpath}", "RUNPOD_ENV_FILE": str(self.env_file),
            "RUNPOD_TEST_MODE": "1", "RUNPOD_TEST_READINESS_READY": "1",
            "HARNESS_VOLUME": str(self.volume),
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

    def test_features_pod_uses_the_active_selection_without_secrets(self) -> None:
        write_env(self.env_file, "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
        result = self.command("features", "--maxRuntime", "2h", "--cpuNumber", "16")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if "--name" in line)
        command = shlex.split(line.removeprefix("DRY RUN:"))
        self.assertEqual(command[command.index("--name") + 1], "stock-forecasting-cpu-features")
        self.assertEqual(command[command.index("--vcpu-count") + 1], "16")
        env = json.loads(command[command.index("--env") + 1])
        self.assertEqual(env["RUNPOD_ROLE"], "cpu-research")
        self.assertEqual(env["RUNPOD_CPU_WORKFLOW"], "features")
        self.assertEqual(env["DATA_ROOT"],
                         "/runpod-volume/datasets/" + self.selection["dataset_request_sha256"])
        self.assertEqual(env["FIN_TS_DATASET_PROFILE"], "tw_only")
        self.assertEqual((env["STAGE1_DATA_START"], env["STAGE1_DATA_END"]),
                         ("2016-01-01", "2026-06-01"))
        self.assertEqual(env["RUNPOD_CPU_MAX_RUNTIME_SECONDS"], "7200")
        for secret in ("HF_TOKEN", "EODHD_API_TOKEN", "TPEX_PROXY_TOKEN", "TPEX_PROXY_URL"):
            self.assertEqual(env[secret], "", secret)
        self.assertEqual((env["RUNPOD_CPU_MAX_API_CALLS"], env["RUNPOD_CPU_EODHD_QPS"]), ("", ""))
        for arguments in (("features", "--unknown", "1"), ("features", "--cpuNumber", "3")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.command(*arguments).returncode, 2)

    def download(self, *arguments: str) -> subprocess.CompletedProcess:
        write_env(self.env_file, (
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\nRUNPOD_S3_ACCESS_KEY_ID=fixture\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture\nRUNPOD_S3_REGION=eu-ro-1\n"
        ))
        return self.command("download-features", *arguments)

    def write_volume_file(self, relative: str, data: bytes) -> str:
        path = self.volume / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return sha256(data)

    def publish_store(self, kind: str, split: str) -> dict:
        manifest_name, index_name, index_field = {
            "rank_labels": ("rank-labels.json", "label-index.parquet", "label_index_sha256"),
            "features": ("features.json", "feature-index.parquet", "feature_index_sha256"),
        }[kind]
        folder = "rank-labels" if kind == "rank_labels" else "features"
        identity = sha256(f"{kind}-{split}".encode())
        remote = f"datasets/{DATASET}/prepared/{folder}/{identity}"
        files = {}
        for bucket in range(2):
            relative = f"shards/bucket-{bucket:04d}/shard.parquet"
            digest = self.write_volume_file(f"{remote}/{relative}", f"{split}{bucket}".encode())
            files[relative] = {"bytes": 1, "sha256": digest}
        index_digest = self.write_volume_file(f"{remote}/{index_name}", b"index")
        payload = json.dumps({
            "state": "ready", "identity": {"split": split}, index_field: index_digest,
            "files": files,
        }).encode()
        manifest_digest = self.write_volume_file(f"{remote}/{manifest_name}", payload)
        return {"root": "/runpod-volume/" + remote, "volume_relative": remote,
                "manifest_sha256": manifest_digest}

    def publish(self, state: str, *, with_manifest: bool = True) -> None:
        self.write_volume_file(KEY, json.dumps(
            {"kind": KIND, "state": state, "launch_id": LAUNCH}
        ).encode())
        logs = f"logs/tmux/stock-forecasting-cpu-features/{LAUNCH}/"
        self.write_volume_file(logs + "combined.log", b"log")
        self.write_volume_file(logs + "status.json", b"{}")
        if with_manifest:
            manifest = {
                "schema_version": "1.0", "kind": "feature-library-build",
                "splits": {
                    split: {
                        "samples": 10, "rank_labels": self.publish_store("rank_labels", split),
                        "features": self.publish_store("features", split),
                    }
                    for split in ("train", "validation", "test")
                },
            }
            self.write_volume_file(
                f"diagnostics/features/{LAUNCH}/features-manifest.json",
                json.dumps(manifest).encode(),
            )

    def test_download_saves_verified_stores_and_logs(self) -> None:
        self.publish("ready")
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launch = self.project / "artifacts/features/launches" / LAUNCH
        for relative in ("features-manifest.json", "logs/combined.log", "logs/status.json",
                         "lifecycle.json"):
            self.assertTrue((launch / relative).is_file(), relative)
        dataset = self.project / "artifacts/features" / DATASET
        stores = sorted(path.relative_to(dataset).as_posix() for path in dataset.rglob("*.parquet"))
        self.assertEqual(len(stores), 3 * 2 * 3)
        self.assertTrue(any(name.startswith("rank-labels/") for name in stores))
        self.assertTrue(any(name.startswith("features/") for name in stores))
        self.assertIn("Verified 12 shard files", result.stdout)
        # Verified files are kept on a rerun; a corrupted shard on the volume is refused.
        shard = next(
            self.volume.glob(f"datasets/{DATASET}/prepared/features/*/shards/bucket-0001/*")
        )
        shard.write_bytes(b"corrupted")
        self.assertEqual(self.download().returncode, 0)
        local = next((self.project / "artifacts/features").glob(
            f"{DATASET}/features/*/shards/bucket-0001/shard.parquet"
        ))
        local.unlink()
        self.assertNotEqual(self.download().returncode, 0)

    def test_download_reports_running_incomplete_and_invalid_requests(self) -> None:
        self.publish("preparing", with_manifest=False)
        self.assertEqual(self.download().returncode, 3)
        self.publish("failed", with_manifest=False)
        result = self.download()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertTrue(
            (self.project / "artifacts/features/launches" / LAUNCH / "logs/combined.log").is_file()
        )
        self.assertEqual(self.download("../outside").returncode, 2)


if __name__ == "__main__":
    unittest.main()
