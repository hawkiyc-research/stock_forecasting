#!/usr/bin/env bash
# Shared consumers must never mutate a Python runtime another run is using.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Data dependency setup requires a RunPod workflow" >&2
    exit 2
fi
data_dependencies_ready() {
    "${PROJECT_ROOT}/.venv/bin/python" - "${PROJECT_ROOT}/configs/data_cleaning.json" <<'PY'
import json
import sys
from importlib.metadata import PackageNotFoundError, version

expected = json.load(open(sys.argv[1]))["calendar_version"]
try:
    valid = version("exchange_calendars") == expected
except PackageNotFoundError:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}
if data_dependencies_ready; then
    exit 0
fi
echo "Updating the project-owned cloud dependencies; prepared market data is unchanged."
if [[ "${RUNPOD_GPU_WORKFLOW_LEASE_MODE:-exclusive}" == shared ]]; then
    # Upgrading a flock is not atomic. Release the read lease, acquire the writer
    # lease with a bounded wait, then recheck: another starter may have updated it.
    flock -u 9
    runtime_deadline=$((SECONDS + 120))
    while ! flock -n 9; do
        # A completed updater may already be training under a shared lease.
        # Recheck under a read lease instead of waiting for that entire run.
        if flock -s -n 9; then
            if data_dependencies_ready; then exit 0; fi
            flock -u 9
        fi
        if [[ ${SECONDS} -ge ${runtime_deadline} ]]; then
            echo "Runtime update requires all training readers to finish; no environment was changed" >&2
            exit 75
        fi
        sleep 2
    done
    if data_dependencies_ready; then
        flock -s 9
        exit 0
    fi
fi
RUNPOD_ROLE=gpu-train bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ "${RUNPOD_GPU_WORKFLOW_LEASE_MODE:-exclusive}" == shared ]]; then flock -s 9; fi
