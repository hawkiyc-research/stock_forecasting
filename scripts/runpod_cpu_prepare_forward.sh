#!/usr/bin/env bash

# Fetch official Taiwan data and seal the forward input package on a CPU preparation Pod.
# runpod_tmux_launch.sh cpu-prepare starts it on a Pod created with --forward; no training
# dataset, readiness marker or selection is read or written.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
RUNPOD_ROLE="${RUNPOD_ROLE:-cpu-prep}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
FORWARD_END="${RUNPOD_FORWARD_END:-}"
FORWARD_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/cpu-prep/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for CPU preparation; the guard checks its Pod ID.
FORWARD_MARKER="${LIFECYCLE_ROOT}/forward/cpu-preparation.json"
FORWARD_ROOT="${NETWORK_VOLUME_ROOT}/forward/${FORWARD_END}"
FETCH_ROOT="${FORWARD_ROOT}/fetch"
PACKAGE_ROOT="${FORWARD_ROOT}/package"
QUARANTINE_ROOT="${FORWARD_ROOT}/quarantine"
# Read with every datasets/*/api-cache first; later dataset preparations read it in turn.
# upload_runpod_forward_cache.py writes the same namespace.
SHARED_CACHE_ROOT="${NETWORK_VOLUME_ROOT}/datasets/forward-taiwan/api-cache"
RUNPOD_CPU_TAIWAN_QPS="${RUNPOD_CPU_TAIWAN_QPS:-0.5}"
RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS="${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS:-60}"
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Forward preparation must run inside a RunPod CPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" ]]; then
    echo "Forward preparation requires the cpu-prep role" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
if [[ ! "${LAUNCH_ID}" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    echo "RUNPOD_LAUNCH_ID contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${FORWARD_END}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "RUNPOD_FORWARD_END must use YYYY-MM-DD" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ \
    || "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^0([.]0+)?$ \
    || ! "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Taiwan QPS and provider backoff settings are invalid" >&2
    exit 2
fi
for path_name in \
    PROJECT_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN FORWARD_LOG CODE_MARKER \
    FORWARD_MARKER FORWARD_ROOT SHARED_CACHE_ROOT; do
    path_value="${!path_name}"
    case "${path_value}" in
        /workspace|/workspace/*)
            echo "${path_name} must never use /workspace" >&2
            exit 2
            ;;
    esac
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "RunPod image Python is unavailable: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi
# The CPU preparation reaches TPEx only through the Cloud Run relay.
if [[ -z "${TPEX_PROXY_TOKEN:-}" || "${TPEX_PROXY_TOKEN}" == *'{{ RUNPOD_SECRET_'* ]]; then
    echo "TPEX_PROXY_TOKEN RunPod Secret is missing or was not resolved" >&2
    exit 2
fi
if [[ ! "${TPEX_PROXY_URL:-}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
    echo "TPEX_PROXY_URL is missing or is not an approved run.app origin" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

mkdir -p "$(dirname "${FORWARD_LOG}")" "$(dirname "${FORWARD_MARKER}")" "${FORWARD_ROOT}"
FORWARD_SUCCEEDED=0

write_forward_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${FORWARD_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind forward-cpu-preparation
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${FORWARD_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_forward_preparation() {
    local forward_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${FORWARD_SUCCEEDED} -ne 1 ]]; then
        local state=failed
        if [[ ${forward_exit_code} -eq 124 ]]; then
            state=timed_out
        fi
        write_forward_state "${state}" "${forward_exit_code}" || true
    fi
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${forward_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_forward_preparation EXIT

# Move an unfinished step aside without deleting it, so the step starts clean.
quarantine_forward_path() {
    local source="$1" name="$2"
    local destination="${QUARANTINE_ROOT}/${name}-${LAUNCH_ID}"
    runpod_validate_path_in_root \
        "${destination}" "${FORWARD_ROOT}" destination FORWARD_ROOT
    mkdir -p "${QUARANTINE_ROOT}"
    mv "${source}" "${destination}"
    printf 'Moved the unfinished %s aside: %s\n' "${name}" "${destination}" >&2
}

write_forward_state preparing
# As in the other CPU workflows: move sources outside this release (for example
# files another worktree synced) to lifecycle/quarantine/ before checking the code.
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" quarantine-stale-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"

bash "${SCRIPT_DIR}/bootstrap_network_volume.sh"
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}"

printf '[%s] forward pipeline tests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run pytest -q \
    tests/test_provider_cache_scope.py tests/test_forward_fetch.py tests/test_forward_package.py

if [[ -f "${PACKAGE_ROOT}/forward-package.json" ]]; then
    # A sealed package, built here or uploaded from a local build, is verified and kept.
    printf '[%s] verifying the sealed package\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${POETRY_BIN}" run python -c '
import sys
from stock_forecasting.forward.package import load_package
payload = load_package(sys.argv[1])
if payload["end_exclusive"] != sys.argv[2]:
    raise SystemExit("The sealed package end date differs from RUNPOD_FORWARD_END")
print("Sealed forward package verified:", payload["package_sha256"])
' "${PACKAGE_ROOT}" "${FORWARD_END}"
else
    if [[ -f "${FETCH_ROOT}/fetch.json" ]]; then
        "${POETRY_BIN}" run python -c '
import sys
from stock_forecasting.forward.fetch import load_fetch
if load_fetch(sys.argv[1])["end_exclusive"] != sys.argv[2]:
    raise SystemExit("The fetch end date differs from RUNPOD_FORWARD_END")
print("Reusing the verified forward fetch")
' "${FETCH_ROOT}" "${FORWARD_END}"
    else
        if [[ -e "${FETCH_ROOT}" ]]; then
            quarantine_forward_path "${FETCH_ROOT}" fetch
        fi
        bash "${SCRIPT_DIR}/warm_tpex_cloud_run_relay.sh"
        printf '[%s] fetch through %s (shared cache %s)\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FORWARD_END}" "${SHARED_CACHE_ROOT}"
        "${POETRY_BIN}" run python -m stock_forecasting.cli.forward_pipeline fetch \
            --end "${FORWARD_END}" \
            --artifacts "${NETWORK_VOLUME_ROOT}" \
            --shared-cache "${SHARED_CACHE_ROOT}" \
            --qps "${RUNPOD_CPU_TAIWAN_QPS}" \
            --max-backoff-seconds "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}"
    fi
    if [[ -e "${PACKAGE_ROOT}" ]]; then
        quarantine_forward_path "${PACKAGE_ROOT}" package
    fi
    printf '[%s] build the sealed package\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${POETRY_BIN}" run python -m stock_forecasting.cli.forward_pipeline build \
        --end "${FORWARD_END}" \
        --artifacts "${NETWORK_VOLUME_ROOT}"
fi

# Refuse to report a package if source changed during the run.
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"
write_forward_state ready
FORWARD_SUCCEEDED=1
printf 'Forward input package ready: %s\n' "${PACKAGE_ROOT}"
