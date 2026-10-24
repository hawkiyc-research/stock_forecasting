"""Dependency-free shell integration tests; no real tmux, model, or cloud calls."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# GNU timeout without --foreground: when the job dies from a signal before the deadline,
# it reports any core dump, disables its own core and re-raises that signal on itself.
GNU_TIMEOUT_EMULATOR = r"""
import contextlib
import os
import resource
import signal
import sys

if sys.argv[1:3] != ["--signal=TERM", "--kill-after=60"]:
    sys.exit(94)
os.setpgid(0, 0)
child = os.fork()
if child == 0:
    os.execvp(sys.argv[4], sys.argv[4:])
_, status = os.waitpid(child, 0)
if os.WIFEXITED(status):
    sys.exit(os.WEXITSTATUS(status))
job_signal = os.WTERMSIG(status)
if os.WCOREDUMP(status):
    print("timeout: the monitored command dumped core", file=sys.stderr, flush=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with contextlib.suppress(OSError, ValueError):
    signal.signal(job_signal, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, [job_signal])
os.kill(os.getpid(), job_signal)
sys.exit(128 + job_signal)
"""

CRASHING_JOB = r"""
import os
import signal
import sys

runner, cut_runner = sys.argv[1], sys.argv[2] == "1"
if cut_runner:
    # End the runner file inside its post-job section, as a short volume read would.
    content = open(runner, "rb").read()
    os.truncate(runner, content.index(b"publish_status() {") + len(b"publish_status() {"))
# SIGKILL leaves no macOS crash report; GNU timeout re-raises it like the SIGBUS seen.
os.kill(os.getpid(), signal.SIGKILL)
"""


class ProbeHarness:
    """Execute the real launcher and generated runner with bounded local stubs."""

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="probe-tmux-")
        self.root = Path(self.temporary.name).resolve()
        self.volume = self.root / "volume"
        self.project = self.volume / "stock_forecasting"
        self.scripts = self.project / "scripts"
        self.bin = self.root / "bin"
        (self.scripts / "lib").mkdir(parents=True)
        self.bin.mkdir()
        for name in (
            "runpod_tmux_launch.sh",
            "runpod_workflow.sh",
            "runpod_probe_lifecycle.py",
            "lib/runpod_paths.sh",
            "lib/runpod_cli.sh",
        ):
            shutil.copyfile(ROOT / "scripts" / name, self.scripts / name)
        self.environment = {
            "PATH": f"{self.bin}{os.pathsep}{os.defpath}",
            "LC_ALL": "C",
            "HARNESS_ROOT": str(self.root),
            "RUNPOD_SSH_ENV_IMPORTED": "1",
            "RUNPOD_POD_ID": "probe-fixture",
            "WANDB_RUN_ID": "run-pod-owner",
            "NETWORK_VOLUME_ROOT": str(self.volume),
            "RUNPOD_VOLUME_ROOT": str(self.volume),
            "PROJECT_ROOT": str(self.project),
            "LOG_ROOT": str(self.volume / "logs"),
            "RUNPOD_PYTHON_BIN": str(self.bin / "readiness-stub"),
            "MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_CPU_MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_CPU_FINALIZE_MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_VALIDATION_MAX_RUNTIME_SECONDS": "120",
        }
        self.write_script(
            self.scripts / "verify_runpod_mounted_readiness.sh",
            r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/mount-checks"
            exit "${HARNESS_MOUNT_EXIT:-0}"
            """,
        )
        self.write_script(self.scripts / "ensure_runpod_tmux.sh", "exit 0\n")
        self.write_script(
            self.bin / "tmux",
            r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/tmux-calls"
            [[ "$1" == -L ]] || exit 91
            socket="$2"
            shift 2
            case "$1" in
                has-session)
                    [[ "${HARNESS_DUPLICATE:-0}" == 1 \
                        || -f "${HARNESS_ROOT}/session-${socket}" ]]
                    ;;
                new-session)
                    [[ "$2" == -d ]] || exit 92
                    : > "${HARNESS_ROOT}/session-${socket}"
                    ;;
                set-option|send-keys) exit 0 ;;
                *) exit 93 ;;
            esac
            """,
        )
        self.write_script(
            self.bin / "timeout",
            r"""
            printf '%s\n' "$1" "$2" "$3" > "${HARNESS_ROOT}/timeout-options"
            [[ "$1" == --signal=TERM && "$2" == --kill-after=60 ]] || exit 94
            shift 3
            if [[ "${HARNESS_TIMEOUT:-0}" == 1 ]]; then exit 124; fi
            exec "$@"
            """,
        )
        self.write_script(
            self.bin / "flock",
            r"""
            [[ "$1" == -n && "$2" == 9 ]] || exit 95
            [[ -e /dev/fd/9 ]] || exit 96
            printf 'acquire\n' >> "${HARNESS_ROOT}/lease-calls"
            exit "${HARNESS_LEASE_EXIT:-0}"
            """,
        )
        self.write_script(
            self.bin / "readiness-stub",
            'if [[ "$1" == */runpod_probe_lifecycle.py ]]; then exec '
            + shlex.quote(sys.executable)
            + ' "$@"; fi\n'
            + r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/lifecycle-calls"
            if [[ "$2" == training-phase-completed ]]; then exit 1; fi
            exit 0
            """,
        )
        self.write_script(
            self.bin / "sleep",
            r"""
            if [[ "$1" == 60 ]]; then
                [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 && -e /dev/fd/9 ]] || exit 99
                printf 'lease-held\n' > "${HARNESS_ROOT}/waiting-for-local-guard"
                exit 1
            fi
            exec /bin/sleep "$@"
            """,
        )
        for name in (
            "runpod_probe_scales.sh",
            "runpod_cpu_prepare.sh",
            "runpod_cpu_finalize.sh",
            "runpod_entrypoint.sh",
            "runpod_validation.sh",
        ):
            self.write_script(
                self.scripts / name,
                r"""
                : > "${HARNESS_ROOT}/worker-args"
                if [[ $# -gt 0 ]]; then
                    printf '%s\0' "$@" > "${HARNESS_ROOT}/worker-args"
                fi
                printf '%s\n' "${RUNPOD_ROLE}" "${RUNPOD_SCALE_PROBE_TMUX_WORKER:-0}" \
                    "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" > "${HARNESS_ROOT}/worker-env"
                if [[ "${RUNPOD_ROLE}" == gpu-probe ]]; then
                    [[ -e /dev/fd/9 ]] || exit 97
                fi
                printf 'fixture worker finished\n'
                exit "${HARNESS_JOB_EXIT:-0}"
                """,
            )
        self.write_script(
            self.scripts / "runpod_self_terminate.sh",
            r"""
            # Verify terminal status and lease ownership before the shutdown request.
            status="${RUNPOD_SHUTDOWN_DIR%/pod-shutdown}/status.json"
            [[ -s "${status}" ]] || exit 98
            if [[ "${RUNPOD_ROLE}" == gpu-probe ]]; then
                [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 \
                    && -e /dev/fd/9 ]] || exit 99
            fi
            cp "${status}" "${HARNESS_ROOT}/status-at-shutdown.json"
            printf '%s\n' "${RUNPOD_ROLE}" > "${HARNESS_ROOT}/shutdown-called"
            mkdir -p "${RUNPOD_SHUTDOWN_DIR}"
            printf '{"fixture":true}\n' > "${RUNPOD_SHUTDOWN_MARKER}"
            exit "${HARNESS_SHUTDOWN_EXIT:-0}"
            """,
        )

    @staticmethod
    def write_script(path: Path, body: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o700)

    def command(self, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.scripts / name), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def runners(self) -> list[Path]:
        return sorted((self.volume / "logs").rglob("runner.sh"))

    def run_worker(self) -> subprocess.CompletedProcess[str]:
        runners = self.runners()
        if len(runners) != 1:
            raise AssertionError(f"Expected one generated runner, found {len(runners)}")
        # Execute the generated script, not a second model or a real tmux server.
        return self.command(str(runners[0]))

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def status(self) -> dict:
        return json.loads((self.runners()[0].parent / "status.json").read_text(encoding="utf-8"))

    def diagnostic_marker(self) -> Path:
        return self.volume / "lifecycle/diagnostics/representation-scales/probe-fixture.json"


class ProbeTmuxTests(unittest.TestCase):
    def harness(self) -> ProbeHarness:
        harness = ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        return harness

    def assert_no_training_markers(self, harness: ProbeHarness) -> None:
        self.assertFalse((harness.root / "lifecycle-calls").exists())
        self.assertEqual(list((harness.volume / "lifecycle/stage1").rglob("*.json")), [])

    def test_scoped_training_and_validation_keep_runner_and_finalization_bound(self) -> None:
        for workflow, phase, role in (
            ("stage1-train", "training", "gpu-train"),
            ("stage1-validate", "validation", "gpu-validation"),
        ):
            with self.subTest(workflow=workflow):
                harness = self.harness()
                harness.environment["RUNPOD_SCOPED_LIFECYCLE"] = "1"
                harness.write_script(
                    harness.bin / "flock",
                    r"""
                    [[ "$*" == "-n 8" || "$*" == "-s -n 9" ]] || exit 95
                    printf '%s\n' "$*" >> "${HARNESS_ROOT}/lease-calls"
                """,
                )
                result = harness.command("runpod_tmux_launch.sh", workflow)
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
                calls = harness.read("lifecycle-calls")
                self.assertIn(f"/lifecycle/runs/run-pod-owner/{phase}.json", calls)
                self.assertNotIn("/lifecycle/stage1/", calls)
                self.assertEqual(harness.read("lease-calls").splitlines(), ["-n 8", "-s -n 9"])
                self.assertEqual(harness.read("worker-env").splitlines(), [role, "0", "1"])
                self.assertEqual(harness.status()["state"], "succeeded")

    def test_legacy_workflow_alias_delegates_once_with_unchanged_arguments(self) -> None:
        for arguments in ((), ("--checkpoint", "run-selected", "--batch-size", "8")):
            with self.subTest(arguments=arguments):
                harness = self.harness()
                shutil.copyfile(
                    ROOT / "scripts/runpod_probe_scales.sh",
                    harness.scripts / "runpod_probe_scales.sh",
                )
                harness.write_script(
                    harness.scripts / "runpod_tmux_launch.sh",
                    'printf \'%s\\0\' "$@" > "${HARNESS_ROOT}/launcher-args"\n',
                )
                result = harness.command("runpod_workflow.sh", "probe-scales", *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = (harness.root / "launcher-args").read_bytes().split(b"\0")[:-1]
                self.assertEqual(actual, [item.encode() for item in ("probe-scales", *arguments)])
                self.assertFalse((harness.root / "lease-calls").exists())
                self.assertFalse((harness.root / "shutdown-called").exists())

    def test_launcher_help_needs_no_pod_tmux_or_python(self) -> None:
        harness = self.harness()
        shutil.copyfile(
            ROOT / "scripts/runpod_probe_scales.sh", harness.scripts / "runpod_probe_scales.sh"
        )
        harness.environment.pop("RUNPOD_POD_ID")
        harness.environment.pop("RUNPOD_SSH_ENV_IMPORTED")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local guard terminates the Pod", result.stdout)
        self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales", result.stdout)
        self.assertNotIn("runpod_workflow.sh probe-scales", result.stdout)
        legacy_help = harness.command("runpod_workflow.sh", "probe-scales", "--help")
        self.assertEqual(legacy_help.returncode, 0, legacy_help.stderr)
        self.assertEqual(legacy_help.stdout, result.stdout)
        workflow_help = harness.command("runpod_workflow.sh", "--help")
        self.assertEqual(workflow_help.returncode, 0, workflow_help.stderr)
        self.assertNotIn("probe-scales", workflow_help.stdout + workflow_help.stderr)
        self.assertFalse((harness.root / "mount-checks").exists())
        self.assertFalse((harness.root / "tmux-calls").exists())

    def test_readme_only_documents_tmux_for_remote_probe_execution(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        # The probe chapter sits one level deeper since the Kronos line moved to the appendix.
        for heading in (
            "#### 歷史尺度表徵診斷",
            "#### Historical scale representation diagnostics",
        ):
            with self.subTest(heading=heading):
                section = readme.split(heading, 1)[1].split("\n#### ", 1)[0]
                language = "zh" if "歷史" in heading else "en"
                self.assertIn(f"#single-model-{language}", section)
                advanced = readme.split(f'<a id="single-model-{language}"></a>', 1)[1].split(
                    "\n## ", 1
                )[0]
                self.assertIn(
                    "bash scripts/runpod_workflow.sh train --experiment a-alpha-point64 "
                    "--seed 42 --maxRuntime 2h",
                    advanced,
                )
                self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales\n", section)
                self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales --help", section)
                self.assertNotIn("runpod_workflow.sh probe-scales", section)
        diagnostic_source = (ROOT / "src/stock_forecasting/cli/probe_scales.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Use bash scripts/runpod_tmux_launch.sh probe-scales", diagnostic_source)
        self.assertNotIn("runpod_workflow.sh probe-scales", diagnostic_source)

    def test_readme_documents_local_downloads_with_matching_bilingual_commands(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        commands_by_language = []
        for heading in (
            "##### 在本機下載診斷結果",
            "##### Download diagnostic results locally",
        ):
            with self.subTest(heading=heading):
                section = readme.split(heading, 1)[1].split("\n##### ", 1)[0]
                commands = [part.split("```", 1)[0] for part in section.split("```bash\n")[1:]]
                commands_by_language.append(commands)
                self.assertEqual(
                    commands,
                    [
                        "bash scripts/runpod_workflow.sh download-probes\n",
                        'bash scripts/runpod_workflow.sh download-probes "<RUN_ID>"\n',
                    ],
                )
                for forbidden in (
                    "runpod_s3_project.sh",
                    "PROBE_VOLUME_ID",
                    "PROBE_CHECKPOINT",
                    "PROBE_ID",
                    "PROBE_LAUNCH_ID",
                    "s3://",
                    "python3 -m json.tool",
                ):
                    self.assertNotIn(forbidden, section)
                self.assertIn("state: complete", section)
                for command in commands:
                    self.assertNotIn("runpod_tmux_launch.sh", command)
                    self.assertNotIn("runpod_workflow.sh train", command)
                    self.assertNotIn("source .env", command)
                    self.assertNotIn("--resume", command)
                    self.assertNotIn("--delete", command)
                    syntax = subprocess.run(
                        ["bash", "-n"], input=command, capture_output=True, text=True, timeout=5
                    )
                    self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertEqual(*commands_by_language)

    def test_success_publishes_signal_then_waits_for_local_guard_while_owning_lease(self) -> None:
        harness = self.harness()
        # An inherited hint must not bypass real lease acquisition in the runner.
        harness.environment["RUNPOD_GPU_WORKFLOW_LEASE_HELD"] = "1"
        arguments = ("--checkpoint", "run-selected", "--batch-size", "8")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Detached tmux session started: stock-forecasting-probe-scales", result.stdout
        )
        self.assertIn(
            "new-session -d -s stock-forecasting-probe-scales", harness.read("tmux-calls")
        )
        self.assertIn("remain-on-exit on", harness.read("tmux-calls"))
        self.assertFalse((harness.root / "worker-args").exists())
        self.assertFalse((harness.root / "lease-calls").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assertFalse((harness.runners()[0].parent / "status.json").exists())
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        self.assertEqual(harness.read("lease-calls"), "acquire\n")
        self.assertEqual(harness.read("worker-env").splitlines(), ["gpu-probe", "1", "1"])
        self.assertEqual(harness.status()["state"], "succeeded")
        self.assertEqual(harness.status()["exit_code"], 0)
        self.assertEqual(harness.status()["finalization_exit_code"], 0)
        marker = json.loads(harness.diagnostic_marker().read_text())
        self.assertEqual(marker["state"], "succeeded")
        self.assertEqual(marker["owner_run_id"], "run-pod-owner")
        self.assertEqual(marker["pod_id"], "probe-fixture")
        self.assertEqual(marker["status_path"], str(harness.runners()[0].parent / "status.json"))
        self.assertEqual(harness.read("waiting-for-local-guard"), "lease-held\n")
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assertEqual(harness.read("timeout-options").splitlines()[-1], "120s")
        self.assertIn(
            "fixture worker finished",
            Path(harness.status()["log_path"]).read_text(),
            runner.stdout + runner.stderr,
        )
        self.assertFalse((harness.runners()[0].parent / "pod-shutdown/shutdown.json").exists())
        self.assert_no_training_markers(harness)

    def test_probe_arguments_are_quoted_not_executed_as_shell_code(self) -> None:
        harness = self.harness()
        sentinel = harness.root / "injected-command"
        arguments = ("--seed", f"x y; touch {sentinel}; $(touch {sentinel})")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertEqual(
            (harness.root / "worker-args").read_bytes().split(b"\0")[:-1],
            [item.encode() for item in arguments],
        )
        self.assertFalse(sentinel.exists())

    def test_failure_and_timeout_publish_signals_for_local_termination(self) -> None:
        for environment, state, exit_code in (
            ({"HARNESS_JOB_EXIT": "7"}, "failed", 7),
            ({"HARNESS_TIMEOUT": "1"}, "timed_out", 124),
        ):
            with self.subTest(state=state):
                harness = self.harness()
                harness.environment.update(environment)
                result = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, exit_code, runner.stdout + runner.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.status()["exit_code"], exit_code)
                marker = json.loads(harness.diagnostic_marker().read_text())
                self.assertEqual(marker["state"], state)
                self.assertEqual(marker["exit_code"], exit_code)
                self.assertTrue((harness.root / "waiting-for-local-guard").is_file())
                self.assertFalse((harness.root / "shutdown-called").exists())
                self.assert_no_training_markers(harness)

    def test_signal_killed_job_publishes_failure_even_if_runner_file_is_cut(self) -> None:
        # On 2026-10-19 the runner exited 2 after a crash, with no status.json or probe
        # state, because bash had not yet parsed its post-job tail on the volume.
        python = shlex.quote(sys.executable)
        killed_exit_code = 128 + signal.SIGKILL
        for workflow, job_script in (
            ("probe-scales", "runpod_probe_scales.sh"),
            ("export-signal-predictions", "runpod_export_signal_predictions.sh"),
        ):
            for cut_runner in ("0", "1"):
                with self.subTest(workflow=workflow, cut_runner=cut_runner):
                    harness = self.harness()
                    emulator = harness.bin / "gnu-timeout.py"
                    emulator.write_text(GNU_TIMEOUT_EMULATOR, encoding="utf-8")
                    harness.write_script(
                        harness.bin / "timeout",
                        f'exec {python} {shlex.quote(str(emulator))} "$@"\n',
                    )
                    job = harness.bin / "crashing-job.py"
                    job.write_text(CRASHING_JOB, encoding="utf-8")
                    harness.write_script(
                        harness.scripts / job_script,
                        f"""
                        printf 'fixture worker crashing\\n'
                        runner="$(dirname "${{RUNPOD_TMUX_LOG_FILE}}")/runner.sh"
                        exec {python} {shlex.quote(str(job))} "${{runner}}" {cut_runner}
                        """,
                    )
                    result = harness.command("runpod_tmux_launch.sh", workflow)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    runner = harness.run_worker()
                    self.assertEqual(
                        runner.returncode, killed_exit_code, runner.stdout + runner.stderr
                    )
                    self.assertEqual(harness.status()["state"], "failed")
                    self.assertEqual(harness.status()["exit_code"], killed_exit_code)
                    marker = json.loads(harness.diagnostic_marker().read_text())
                    self.assertEqual(marker["state"], "failed")
                    self.assertEqual(marker["exit_code"], killed_exit_code)
                    self.assertTrue((harness.root / "waiting-for-local-guard").is_file())
                    self.assertFalse((harness.root / "shutdown-called").exists())

    def test_wait_loop_retries_publication_until_volume_recovers(self) -> None:
        # A volume I/O error at job end fails every finalization attempt within seconds;
        # the guard learns the terminal state only if the waiting runner retries it.
        for workflow, job_script in (
            ("probe-scales", "runpod_probe_scales.sh"),
            ("export-signal-predictions", "runpod_export_signal_predictions.sh"),
        ):
            with self.subTest(workflow=workflow):
                harness = self.harness()
                harness.environment["RUNPOD_TEST_MODE"] = "1"
                # Blocker files replace the job directory and diagnostic lifecycle root.
                harness.write_script(
                    harness.scripts / job_script,
                    r"""
                    job_dir="$(dirname "${RUNPOD_TMUX_LOG_FILE}")"
                    printf '%s\n' "${job_dir}" > "${HARNESS_ROOT}/job-dir"
                    for path in "${job_dir}" "${NETWORK_VOLUME_ROOT}/lifecycle/diagnostics"; do
                        mv "${path}" "${path}.offline"
                        printf 'fixture-volume-error\n' > "${path}"
                    done
                    printf 'fixture worker failed\n'
                    exit 7
                    """,
                )
                harness.write_script(
                    harness.bin / "sleep",
                    r"""
                    if [[ "$1" != 60 ]]; then exec /bin/sleep "$@"; fi
                    [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 && -e /dev/fd/9 ]] || exit 99
                    waits="${HARNESS_ROOT}/guard-waits"
                    count=$(( $(cat "${waits}" 2>/dev/null || printf 0) + 1 ))
                    printf '%s\n' "${count}" > "${waits}"
                    # The first retry still fails; the volume recovers before the second.
                    if [[ ${count} -eq 2 ]]; then
                        job_dir="$(cat "${HARNESS_ROOT}/job-dir")"
                        for path in "${job_dir}" "${NETWORK_VOLUME_ROOT}/lifecycle/diagnostics"; do
                            rm "${path}"
                            mv "${path}.offline" "${path}"
                        done
                    fi
                    if [[ ${count} -ge 4 ]]; then exit 1; fi
                    """,
                )
                result = harness.command("runpod_tmux_launch.sh", workflow)
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                output = runner.stdout + runner.stderr
                self.assertEqual(runner.returncode, 7, output)
                self.assertEqual(harness.read("guard-waits"), "4\n", output)
                self.assertIn("Diagnostic terminal signal could not be saved", output)
                self.assertEqual(harness.status()["state"], "failed")
                self.assertEqual(harness.status()["exit_code"], 7)
                self.assertEqual(harness.status()["finalization_exit_code"], 74)
                marker = json.loads(harness.diagnostic_marker().read_text())
                self.assertEqual(marker["state"], "failed")
                self.assertEqual(marker["exit_code"], 7)
                self.assertEqual(
                    marker["status_path"], str(harness.runners()[0].parent / "status.json")
                )
                # Each publication is retried until it succeeds once, then never again.
                self.assertEqual(output.count("Published failed workflow status on retry"), 1)
                self.assertEqual(
                    output.count("Published failed diagnostic terminal signal on retry"), 1
                )
                self.assertFalse((harness.root / "shutdown-called").exists())

    def test_contended_lease_preserves_pod_without_running_probe(self) -> None:
        harness = self.harness()
        harness.environment.update(
            {"HARNESS_LEASE_EXIT": "1", "RUNPOD_GPU_WORKFLOW_LEASE_HELD": "1"}
        )
        result = harness.command("runpod_tmux_launch.sh", "probe-scales")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 75, runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "failed")
        self.assertEqual(harness.status()["exit_code"], 75)
        self.assertIn("preserving the Pod", runner.stdout)
        self.assertFalse((harness.root / "worker-args").exists())
        self.assertFalse((harness.root / "timeout-options").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assertFalse(harness.diagnostic_marker().exists())
        self.assert_no_training_markers(harness)

    def test_duplicate_session_or_rejected_preflight_never_terminates_pod(self) -> None:
        for environment, exit_code in (
            ({"HARNESS_DUPLICATE": "1"}, 3),
            ({"HARNESS_MOUNT_EXIT": "2"}, 2),
            ({"MAX_RUNTIME_SECONDS": "invalid"}, 2),
        ):
            with self.subTest(environment=environment):
                harness = self.harness()
                harness.environment.update(environment)
                # This records even an early call, unlike the normal shutdown stub.
                harness.write_script(
                    harness.scripts / "runpod_self_terminate.sh",
                    ': > "${HARNESS_ROOT}/shutdown-called"\n',
                )
                result = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                self.assertFalse((harness.root / "shutdown-called").exists())
                self.assertFalse((harness.root / "worker-args").exists())
                self.assertEqual(harness.runners(), [])
                self.assert_no_training_markers(harness)

    def test_probe_does_not_invoke_pod_side_termination(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_SHUTDOWN_EXIT"] = "4"
        result = harness.command("runpod_tmux_launch.sh", "probe-scales")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "succeeded")
        self.assertIn("awaiting Pod termination by the local guard", runner.stdout)
        self.assertNotIn("self-termination", runner.stdout + runner.stderr)
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assert_no_training_markers(harness)

    def test_missing_or_conflicting_owner_identity_rejects_launch(self) -> None:
        for updates in ({"WANDB_RUN_ID": ""}, {"VALIDATION_RUN_ID": "another-run"}):
            with self.subTest(updates=updates):
                harness = self.harness()
                harness.environment.update(updates)
                result = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((harness.root / "shutdown-called").exists())
                self.assertFalse(harness.diagnostic_marker().exists())
                self.assertEqual(harness.runners(), [])

    def test_failed_running_signal_prevents_model_execution_and_waits_for_guard(self) -> None:
        harness = self.harness()
        blocker = harness.volume / "lifecycle/diagnostics"
        blocker.parent.mkdir(parents=True)
        blocker.write_text("fixture-not-a-directory", encoding="utf-8")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 74, runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "failed")
        self.assertFalse((harness.root / "worker-args").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assertTrue((harness.root / "waiting-for-local-guard").is_file())

    def test_existing_workflows_keep_timeout_roles_and_lifecycle_finalization(self) -> None:
        for workflow, role, duration in (
            ("cpu-prepare", "cpu-prep", "120s"),
            ("cpu-finalize", "cpu-prep", "120s"),
            ("stage1-train", "gpu-train", "420s"),
            ("stage1-validate", "gpu-validation", "420s"),
        ):
            with self.subTest(workflow=workflow):
                harness = self.harness()
                harness.environment.update({"WANDB_RUN_ID": "run-legacy", "HARNESS_JOB_EXIT": "7"})
                result = harness.command("runpod_tmux_launch.sh", workflow)
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, 7, runner.stdout + runner.stderr)
                expected_lease = "1" if role in {"gpu-train", "gpu-validation"} else "0"
                self.assertEqual(
                    harness.read("worker-env").splitlines(), [role, "0", expected_lease]
                )
                self.assertEqual(harness.read("timeout-options").splitlines()[-1], duration)
                self.assertEqual(harness.read("shutdown-called"), f"{role}\n")
                self.assertEqual(harness.status()["state"], "failed")
                self.assertIn("write-state", harness.read("lifecycle-calls"))
                self.assertEqual((harness.root / "lease-calls").exists(), expected_lease == "1")

    def test_section_safe_gpu_workflows_do_not_use_outer_timeout(self) -> None:
        for workflow in ("stage1-train", "stage1-validate"):
            with self.subTest(workflow=workflow):
                harness = self.harness()
                harness.environment.update(
                    {
                        "WANDB_RUN_ID": "run-section",
                        "RUNPOD_SECTION_SAFE_STOP": "1",
                        "HARNESS_TIMEOUT": "1",
                        "HARNESS_JOB_EXIT": "7",
                    }
                )
                result = harness.command("runpod_tmux_launch.sh", workflow)
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, 7, runner.stdout + runner.stderr)
                self.assertFalse((harness.root / "timeout-options").exists())
                self.assertTrue((harness.root / "worker-args").exists())

    def test_existing_workflows_do_not_accept_probe_arguments(self) -> None:
        harness = self.harness()
        harness.environment["RUNPOD_TEST_MODE"] = "1"
        for workflow in ("cpu-prepare", "cpu-finalize", "stage1-train", "stage1-validate"):
            with self.subTest(workflow=workflow):
                result = harness.command("runpod_tmux_launch.sh", workflow, "--batch-size", "8")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Usage:", result.stderr)
        result = harness.command("runpod_tmux_launch.sh")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Usage:", result.stderr)
        self.assertEqual(harness.runners(), [])


if __name__ == "__main__":
    unittest.main()
