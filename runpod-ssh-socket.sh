#!/usr/bin/env bash

# Local-only agent access helper; the root upload allowlist excludes this file.
set -Eeuo pipefail
set +x
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 - "${PROJECT_ROOT}" "$@" <<'PY'
"""Create a bounded-lived SSH master without changing the training workflow."""

import argparse
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path


def command(arguments, timeout=60):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Command failed: " + arguments[0])
    return result.stdout


def private_keys():
    # Keep priority and authentication attempts sequential; each owns one master.
    directory = Path.home() / ".ssh"
    preferred = directory / "id_ed25519_runpod"
    if preferred.is_file():
        yield preferred
    if not directory.is_dir():
        return
    def walk_error(error):
        raise error

    for current, directories, filenames in os.walk(directory, followlinks=False, onerror=walk_error):
        directories.sort()
        for name in sorted(filenames):
            path = Path(current) / name
            if path == preferred or not path.is_file():
                continue
            with path.open("rb") as stream:
                header = stream.read(128)
            if re.match(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----", header):
                yield path


def write_json(path, payload):
    # Publish metadata atomically, including across concurrent invocations.
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", dir=str(path.parent))
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def connect_pod(root, pod, ssh, ttl):
    wrapper = root / "scripts/runpodctl_project.sh"
    # Refresh the selected endpoint immediately before authentication.
    refreshed = json.loads(command(["bash", str(wrapper), "pod", "get", pod["id"], "--output", "json"]))
    if refreshed.get("id") != pod["id"] or refreshed.get("desiredStatus") != "RUNNING":
        raise RuntimeError("Selected Pod identity changed; rerun the helper")
    ssh_details = refreshed.get("ssh") or {}
    direct = ssh_details.get("direct") if isinstance(ssh_details, dict) else None
    endpoints = []
    connection_type = "direct"
    if isinstance(direct, dict) and direct.get("host") and direct.get("port"):
        endpoints.append((direct.get("host"), direct.get("port"), direct.get("username")))
    else:
        runtime = refreshed.get("runtime") or {}
        ports = runtime.get("ports", []) if isinstance(runtime, dict) else []
        for port in ports:
            # REST v2 uses private/public; accept older compatibility fields too.
            if (isinstance(port, dict) and port.get("private", port.get("privatePort")) == 22
                    and port.get("type") == "tcp" and port.get("isIpPublic", True) is True):
                endpoints.append((port.get("ip"), port.get("public", port.get("publicPort")), "root"))
    if not endpoints:
        proxy = ssh_details.get("proxy") if isinstance(ssh_details, dict) else None
        if isinstance(proxy, dict):
            endpoints.append((proxy.get("host"), proxy.get("port"), proxy.get("username")))
            connection_type = "proxy"
    if len(endpoints) != 1:
        raise RuntimeError("Selected Pod has no unique SSH endpoint; check its Connect tab")
    host, port, username = endpoints[0]
    if connection_type == "proxy":
        if host != "ssh.runpod.io":
            raise ValueError("RunPod SSH proxy host is invalid")
    else:
        host = str(ipaddress.ip_address(host))
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("RunPod SSH public port is invalid")
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", username):
        raise ValueError("RunPod SSH username is invalid")
    destination = username + "@" + host
    # A short per-session path avoids macOS's Unix-domain socket path limit.
    session = Path(tempfile.mkdtemp(prefix="rps-", dir="/tmp")).resolve()
    socket = session / "control.sock"
    base = [ssh, "-F", "/dev/null", "-S", str(socket), "-p", str(port)]
    master_options = [
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
        "-o", "ControlPersist=no",
    ]
    established = False
    watcher = None
    try:
        for key in private_keys():
            print("Pod {}: trying SSH identity {}".format(pod["id"], key), flush=True)
            with (session / "authentication.log").open("a") as log:
                try:
                    result = subprocess.run(
                        base + master_options + ["-i", str(key), "-M", "-N", "-f", destination],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    log.write("Authentication timed out for " + str(key) + "\n")
                    subprocess.run(base + ["-O", "stop", destination], capture_output=True, timeout=15)
                    continue
            if result.returncode == 0:
                established = True
                identity = key
                break
        if not established:
            raise RuntimeError("No SSH identity succeeded; see " + str(session / "authentication.log")
                               + ". Unlock encrypted keys with ssh-add before retrying.")
        command(base + ["-O", "check", destination], timeout=10)
        expires = time.time() + ttl
        # Use wall time and bounded polling so host sleep does not extend the TTL.
        expiry_code = '''import subprocess, sys, time
deadline = float(sys.argv[1])
while time.time() < deadline:
    time.sleep(min(30, max(0, deadline - time.time())))
result = subprocess.run(sys.argv[2:], timeout=15)
raise SystemExit(result.returncode)
'''
        with (session / "expiry.log").open("a") as log:
            watcher = subprocess.Popen(
                [sys.executable, "-c", expiry_code, str(expires)] + base + ["-O", "stop", destination],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                close_fds=True,
            )
        # ProxyCommand=false forbids a fresh authenticated connection if the socket expires.
        reuse = base + ["-o", "BatchMode=yes", "-o", "ControlMaster=no",
                        "-o", "ProxyCommand=false", destination]
        metadata = {
            "project_root": str(root), "pod_id": pod["id"], "host": host, "port": port,
            "connection_type": connection_type,
            "destination": destination, "identity_file": str(identity),
            "control_path": str(socket), "ttl_seconds": ttl,
            "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
            "expires_at_epoch": expires, "expiry_pid": watcher.pid, "ssh_command": reuse,
            "check_command": base + ["-O", "check", destination],
            "close_command": base + ["-O", "stop", destination],
        }
        write_json(session / "session.json", metadata)
        return metadata, watcher
    except BaseException:
        if watcher is not None:
            watcher.terminate()
            watcher.wait(timeout=15)
        if established:
            subprocess.run(base + ["-O", "stop", destination], capture_output=True, timeout=15)
        raise


def available_memory_bytes():
    # Budget recoverable host memory, capped by container limits when present.
    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"(?m)^MemAvailable:\s+(\d+) kB$", meminfo.read_text())
        if match:
            available = int(match[1]) * 1024
    elif sys.platform == "darwin":
        statistics = command(["/usr/bin/vm_stat"], timeout=5)
        page_size = re.search(r"page size of (\d+) bytes", statistics)
        if page_size:
            pages = sum(int(value) for value in re.findall(
                r"(?m)^Pages (?:free|inactive|speculative):\s+(\d+)\.", statistics
            ))
            available = pages * int(page_size[1])
    if available is None:
        raise RuntimeError("Unable to estimate available memory for SSH connection planning")
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit, usage = Path(limit_path), Path(usage_path)
        if limit.is_file() and usage.is_file():
            value = limit.read_text().strip()
            if value.isdigit():
                available = min(available, max(0, int(value) - int(usage.read_text())))
    return available


def worker_plan(count):
    # Reserve 512 MiB and allow 64 MiB each for retained sessions and active work.
    per_connection = 64 * 1024 * 1024
    reserve = 512 * 1024 * 1024
    available = available_memory_bytes()
    capacity = max(0, (available - reserve) // per_connection - count)
    if capacity < 1:
        raise RuntimeError("Insufficient available memory for {} SSH sessions plus working memory; "
                           "use --pod-id to reduce the connection scope".format(count))
    value = os.environ.get("RUNPOD_SSH_SOCKET_WORKERS", "")
    if value and not re.fullmatch(r"[1-9][0-9]{0,3}", value):
        raise ValueError("RUNPOD_SSH_SOCKET_WORKERS must be a positive integer of at most four digits")
    # Limit account API/SSH load even on a machine with many CPU cores.
    service_limit = 4
    requested = int(value) if value else service_limit
    workers = min(count, os.cpu_count() or 1, capacity, requested, service_limit)
    print("SSH workers: {} (Pods: {}, available memory: {} MiB, reserve: 512 MiB, "
          "session/worker allowance: 64 MiB each)".format(workers, count, available // (1024 * 1024)),
          flush=True)
    return workers


def close_connection(metadata, watcher):
    watcher.terminate()
    watcher.wait(timeout=15)
    command(metadata["close_command"], timeout=15)


def main():
    root = Path(sys.argv[1])
    parser = argparse.ArgumentParser(
        prog="bash runpod-ssh-socket.sh",
        description="Create local SSH sockets only when you want an agent to operate RunPod. "
                    "Without --pod-id, connect to all accessible running Pods.",
        epilog="Try ~/.ssh/id_ed25519_runpod first, then other private keys under ~/.ssh. "
               "Unlock encrypted keys with ssh-add before connecting.",
    )
    parser.add_argument("socket_ttl", nargs="?", default="4h", metavar="SOCKET_TTL",
                        help="socket TTL (default: 4h; examples: 30m, 8h, 1d)")
    parser.add_argument("--pod-id", metavar="POD_ID", help="connect only to this running Pod")
    args = parser.parse_args(sys.argv[2:])
    match = re.fullmatch(r"([1-9][0-9]{0,8})([mhd])", args.socket_ttl)
    if match is None:
        parser.error("Socket TTL must be a positive duration such as 30m, 4h, or 1d")
    if args.pod_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id):
        parser.error("Pod ID contains invalid characters")
    ttl = int(match[1]) * {"m": 60, "h": 3600, "d": 86400}[match[2]]
    ssh = shutil.which("ssh")
    if ssh is None:
        raise RuntimeError("OpenSSH is required on the local control machine")
    wrapper = root / "scripts/runpodctl_project.sh"
    pods = json.loads(command(["bash", str(wrapper), "pod", "list", "--output", "json"]))
    if not isinstance(pods, list):
        raise ValueError("RunPod Pod list must be an array")
    candidates = {}
    for pod in pods:
        if (isinstance(pod, dict) and pod.get("desiredStatus") == "RUNNING"
                and re.fullmatch(r"[A-Za-z0-9_-]+", str(pod.get("id", "")))
                and (args.pod_id is None or pod["id"] == args.pod_id)):
            candidates[pod["id"]] = pod
    if not candidates:
        raise RuntimeError("Requested Pod is not running or accessible to this API key" if args.pod_id
                           else "No running Pod is accessible to the configured RunPod API key")
    workers = worker_plan(len(candidates))
    state = root / ".runpod/ssh-socket"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    connections, failures, watchers = {}, {}, {}
    result_lock = threading.Lock()

    def connect_and_record(pod):
        metadata, watcher = connect_pod(root, pod, ssh, ttl)
        with result_lock:
            connections[pod["id"]] = metadata
            watchers[pod["id"]] = watcher

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    iterator = iter(candidates.values())
    try:
        # Never submit more than workers futures; replenish only after completion.
        for _ in range(workers):
            pod = next(iterator)
            pending[pool.submit(connect_and_record, pod)] = pod["id"]
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pod_id = pending.pop(future)
                try:
                    future.result()
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    failures[pod_id] = str(error)
                    print("Pod {}: FAILED: {}".format(pod_id, error), file=sys.stderr, flush=True)
                pod = next(iterator, None)
                if pod is not None:
                    pending[pool.submit(connect_and_record, pod)] = pod["id"]
        pool.shutdown(wait=True)
        index = {"schema_version": 2, "project_root": str(root), "requested_pod_id": args.pod_id,
                 "connections": dict(sorted(connections.items())), "failures": dict(sorted(failures.items()))}
        write_json(state / "latest.json", index)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        for pod_id, metadata in connections.items():
            try:
                close_connection(metadata, watchers[pod_id])
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print("Pod {}: cleanup failed: {}".format(pod_id, error), file=sys.stderr)
        raise
    for pod_id, metadata in sorted(connections.items()):
        print("\nPod: " + pod_id)
        print("SSH type: " + metadata["connection_type"])
        print("Socket: " + metadata["control_path"])
        print("Expires (UTC): " + metadata["expires_at"])
        print("Agent SSH: " + shlex.join(metadata["ssh_command"]))
        print("Check: " + shlex.join(metadata["check_command"]))
        print("Close new connections: " + shlex.join(metadata["close_command"]))
    print("\nMetadata: " + str(state / "latest.json"))
    print("Connected: {}; failed: {}".format(len(connections), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("runpod-ssh-socket: " + str(error), file=sys.stderr)
        raise SystemExit(1)
PY
