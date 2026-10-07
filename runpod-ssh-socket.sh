#!/usr/bin/env bash

# Local-only agent access helper; the root upload allowlist excludes this file.
set -Eeuo pipefail
set +x
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash runpod-ssh-socket.sh [SOCKET_TTL]" >&2
    exit 2
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash runpod-ssh-socket.sh [SOCKET_TTL]

Create a local shared SSH connection only when you want an agent to operate RunPod.
The only parameter is the socket TTL: default 4h; examples: 30m, 8h, 1d.
Connect to any running Pod accessible to the configured RunPod API key.
Select the Pod interactively when multiple Pods are running.
Try ~/.ssh/id_ed25519_runpod first, then other private keys under ~/.ssh.
Encrypted keys must already be unlocked in ssh-agent (ssh-add).
HELP
    exit 0
fi

exec python3 - "${PROJECT_ROOT}" "${1:-4h}" <<'PY'
"""Create a bounded-lived SSH master without changing the training workflow."""

import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
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


def main():
    root, duration = Path(sys.argv[1]), sys.argv[2]
    match = re.fullmatch(r"([1-9][0-9]{0,8})([mhd])", duration)
    if match is None:
        raise ValueError("Socket TTL must be a positive duration such as 30m, 4h, or 1d")
    ttl = int(match[1]) * {"m": 60, "h": 3600, "d": 86400}[match[2]]
    ssh = shutil.which("ssh")
    if ssh is None:
        raise RuntimeError("OpenSSH is required on the local control machine")
    wrapper = root / "scripts/runpodctl_project.sh"
    pods = json.loads(command(["bash", str(wrapper), "pod", "list", "--output", "json"]))
    if not isinstance(pods, list):
        raise ValueError("RunPod Pod list must be an array")
    candidates = []
    for pod in pods:
        if not isinstance(pod, dict):
            continue
        if (pod.get("desiredStatus") == "RUNNING"
                and re.fullmatch(r"[A-Za-z0-9_-]+", str(pod.get("id", "")))):
            candidates.append(pod)
    candidates.sort(key=lambda pod: pod["id"])
    if not candidates:
        raise RuntimeError("No running Pod is accessible to the configured RunPod API key")
    if len(candidates) == 1:
        pod = candidates[0]
    else:
        for index, pod in enumerate(candidates, 1):
            name = json.dumps(str(pod.get("name", "")), ensure_ascii=False)
            print("{}: {} {}".format(index, pod["id"], name), flush=True)
        # Python source occupies stdin, so read selection from the controlling terminal.
        with open("/dev/tty", "r+") as terminal:
            terminal.write("Select Pod [1-{}]: ".format(len(candidates)))
            terminal.flush()
            selection = terminal.readline().strip()
        if not selection.isdigit() or not 1 <= int(selection) <= len(candidates):
            raise ValueError("Invalid Pod selection")
        pod = candidates[int(selection) - 1]
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
            print("Trying SSH identity: " + str(key), flush=True)
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
        state = root / ".runpod/ssh-socket"
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_json(session / "session.json", metadata)
        write_json(state / "latest.json", metadata)
        print("Pod: " + pod["id"])
        print("SSH type: " + connection_type)
        print("Socket: " + str(socket))
        print("Expires (UTC): " + metadata["expires_at"])
        print("Metadata: " + str(state / "latest.json"))
        print("Agent SSH: " + shlex.join(reuse))
        print("Check: " + shlex.join(metadata["check_command"]))
        print("Close new connections: " + shlex.join(metadata["close_command"]))
    except BaseException:
        if watcher is not None:
            watcher.terminate()
            watcher.wait(timeout=15)
        if established:
            subprocess.run(base + ["-O", "stop", destination], capture_output=True, timeout=15)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("runpod-ssh-socket: " + str(error), file=sys.stderr)
        raise SystemExit(1)
PY
