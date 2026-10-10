"""Read-only, bounded NVIDIA telemetry transport for the local RunPod monitor."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

MAX_OUTPUT_BYTES = 1024 * 1024
MAX_GPUS = 32
REASONS = {
    "gpu_idle": ("gpu_idle", "idle"),
    "applications_clocks_setting": ("applications_clocks_setting",),
    "sw_power_cap": ("sw_power_cap", "sw_power_capping"),
    "hw_slowdown": ("hw_slowdown",),
    "sync_boost": ("sync_boost", "sync_boost_group"),
    "sw_thermal_slowdown": ("sw_thermal_slowdown", "sw_therm_slowdown"),
    "hw_thermal_slowdown": ("hw_thermal_slowdown", "hw_therm_slowdown"),
    "hw_power_brake_slowdown": ("hw_power_brake_slowdown", "hw_power_braking", "hw_power_brake"),
    "display_clock_setting": ("display_clock_setting", "display_clocks_setting"),
}


def bounded_call(command, *, timeout=15, env=None, stop=None):
    """Spool bounded-size replies, reap the whole owned group on timeout/cancel."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if stop is not None and stop():
                    raise InterruptedError("GPU monitor stopped")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{Path(command[0]).name} query timed out")
                if (
                    os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size
                    > MAX_OUTPUT_BYTES
                ):
                    raise ValueError("Telemetry reply exceeds the size limit")
                time.sleep(0.1)
            if (
                os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size
                > MAX_OUTPUT_BYTES
            ):
                raise ValueError("Telemetry reply exceeds the size limit")
            output.seek(0)
            errors.seek(0)
            result = output.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
            detail = errors.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
            if process.returncode:
                # Never expose API response bodies or remote environment variables.
                safe = re.sub(r"[\x00-\x1f\x7f]", " ", detail)[-500:]
                raise RuntimeError(f"{Path(command[0]).name} exited {process.returncode}: {safe}")
            return result
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def number(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:MHz|W|C|%|us|usec|microseconds)?\s*", value
    )
    return float(match[1]) if match else None


def parse_nvidia_xml(raw):
    """Accept both NVIDIA event/throttle XML names and keep unsupported data unknown."""
    if len(raw.encode("utf-8")) > MAX_OUTPUT_BYTES or "<!ENTITY" in raw.upper():
        raise ValueError("Unsafe or oversized NVIDIA XML")
    root = ET.fromstring(raw)
    if root.tag != "nvidia_smi_log":
        raise ValueError("Not an NVIDIA telemetry document")
    devices = root.findall("gpu")
    if not 1 <= len(devices) <= MAX_GPUS:
        raise ValueError("No accessible GPU, or too many devices")
    result = []
    for gpu in devices:

        def read(*paths, device=gpu):
            for path in paths:
                value = number(device.findtext(path))
                if value is not None:
                    return value
            return None

        reasons, counters = {}, {}
        for group in gpu:
            if (
                "clocks_event_reasons" not in group.tag
                and "clocks_throttle_reasons" not in group.tag
            ):
                continue
            for item in group:
                for name, aliases in REASONS.items():
                    if not any(
                        item.tag == alias or item.tag.endswith("_" + alias) for alias in aliases
                    ):
                        continue
                    if "counter" in group.tag:
                        counters[name] = number(item.text)
                    else:
                        text = (item.text or "").strip().lower()
                        reasons[name] = {"active": True, "not active": False}.get(text)
        uuid = gpu.findtext("uuid")
        if not uuid or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
            raise ValueError("GPU UUID is missing or invalid")
        result.append(
            {
                "uuid": uuid,
                "name": (gpu.findtext("product_name") or "unknown")[:120],
                "pstate": gpu.findtext("performance_state"),
                "sm_mhz": read("clocks/sm_clock", "current_clocks/sm_clock"),
                "graphics_mhz": read("clocks/graphics_clock", "current_clocks/graphics_clock"),
                "memory_mhz": read("clocks/mem_clock", "current_clocks/mem_clock"),
                "max_sm_mhz": read("max_clocks/sm_clock"),
                "application_sm_mhz": read("applications_clocks/graphics_clock"),
                "utilization_pct": read("utilization/gpu_util"),
                "temperature_c": read("temperature/gpu_temp"),
                "power_w": read(
                    "gpu_power_readings/power_draw",
                    "gpu_power_readings/instant_power_draw",
                    "power_readings/power_draw",
                ),
                "power_limit_w": read(
                    "gpu_power_readings/current_power_limit",
                    "gpu_power_readings/power_limit",
                    "power_readings/power_limit",
                ),
                "default_power_limit_w": read(
                    "gpu_power_readings/default_power_limit", "power_readings/default_power_limit"
                ),
                "reasons": reasons,
                "counters_us": counters,
            }
        )
    if len({item["uuid"] for item in result}) != len(result):
        raise ValueError("Duplicate GPU UUIDs")
    return result


def endpoint(pod, pod_id):
    if pod.get("id") != pod_id:
        raise ValueError("RunPod endpoint identity mismatch")
    if pod.get("desiredStatus") != "RUNNING":
        raise RuntimeError("Pod is not running")
    ssh = pod.get("ssh") or {}
    address = ssh.get("direct")
    proxy = False
    if not address:
        ports = [
            item
            for item in (pod.get("runtime") or {}).get("ports", [])
            if item.get("private", item.get("privatePort")) == 22
            and item.get("type") == "tcp"
            and item.get("isIpPublic", True)
        ]
        if len(ports) == 1:
            item = ports[0]
            address = {
                "host": item.get("ip"),
                "port": item.get("public", item.get("publicPort")),
                "username": "root",
            }
        else:
            address, proxy = ssh.get("proxy"), True
    if not isinstance(address, dict):
        raise RuntimeError("Pod SSH is not ready")
    host, port, user = address.get("host"), address.get("port"), address.get("username")
    if proxy:
        if host != "ssh.runpod.io":
            raise ValueError("Invalid RunPod SSH proxy")
    else:
        host = str(ipaddress.ip_address(host))
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", user):
        raise ValueError("Invalid SSH username")
    return host, port, user


class TelemetrySSH:
    """One monitor-owned connection; never extend or close the agent's SSH sockets."""

    def __init__(self, project, pod_id, key, *, timeout=15, stop=None):
        self.project, self.pod_id, self.key = Path(project), pod_id, Path(key)
        self.timeout, self.stop = timeout, stop
        self.directory = tempfile.TemporaryDirectory(prefix="rpg-", dir="/tmp")
        self.base = None
        self.name = pod_id
        self.last_endpoint = 0.0

    def connect(self):
        if not self.key.is_file():
            raise RuntimeError(
                "GPU monitor SSH identity is missing; set RUNPOD_GPU_MONITOR_SSH_KEY"
            )
        pod = json.loads(
            bounded_call(
                [
                    "bash",
                    str(self.project / "scripts/runpodctl_project.sh"),
                    "pod",
                    "get",
                    self.pod_id,
                    "--output",
                    "json",
                ],
                timeout=self.timeout,
                stop=self.stop,
            )
        )
        host, port, user = endpoint(pod, self.pod_id)
        self.name = str(pod.get("name") or self.pod_id)[:120]
        self.base = [
            "ssh",
            "-F",
            "/dev/null",
            "-S",
            self.directory.name + "/control.sock",
            "-p",
            str(port),
            "-i",
            str(self.key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=" + self.directory.name + "/known_hosts",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=90",
            user + "@" + host,
        ]
        self.last_endpoint = time.monotonic()

    def sample(self):
        if self.base is None:
            self.connect()
        try:
            # This is the only remote command. No training or GPU control commands.
            return parse_nvidia_xml(
                bounded_call(
                    [*self.base, "nvidia-smi -q -x"],
                    timeout=self.timeout,
                    stop=self.stop,
                )
            )
        except (RuntimeError, TimeoutError, ValueError, ET.ParseError):
            # Refresh a stale endpoint at most once per minute, not on every poll.
            # A boot-time proxy may return a PTY error with exit status zero.
            # Invalid telemetry must also allow switching to the ready direct port.
            if time.monotonic() - self.last_endpoint >= 60:
                self.disconnect()
            raise

    def disconnect(self):
        if self.base:
            with suppress(OSError, RuntimeError, TimeoutError):
                bounded_call([*self.base[:-1], "-O", "exit", self.base[-1]], timeout=3)
        self.base = None

    def close(self):
        self.disconnect()
        self.directory.cleanup()
