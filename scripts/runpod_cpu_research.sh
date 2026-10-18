#!/usr/bin/env bash

# Fetch auxiliary Taiwan data and run the Phase 1 information test on a CPU Pod.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
DATA_ROOT="${DATA_ROOT:-}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
RUNPOD_ROLE="${RUNPOD_ROLE:-cpu-research}"
RESEARCH_CONFIG="configs/research/tw_information_test.json"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
RESEARCH_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/cpu-research/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for CPU preparation; the guard checks its Pod ID.
RESEARCH_MARKER="${LIFECYCLE_ROOT}/research/tw-information.json"
AUXILIARY_ROOT="${DATA_ROOT}/auxiliary"
OUTPUT_ROOT="${NETWORK_VOLUME_ROOT}/diagnostics/tw-information-test/${LAUNCH_ID}"
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "The information test must run inside a RunPod CPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-research" ]]; then
    echo "The information test requires the CPU-only cpu-research role" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
if [[ ! "${LAUNCH_ID}" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    echo "RUNPOD_LAUNCH_ID contains unsupported characters" >&2
    exit 2
fi
if [[ "${DATA_ROOT%/*}" != "${NETWORK_VOLUME_ROOT}/datasets" \
    || ! "${DATA_ROOT##*/}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "DATA_ROOT must be the experiment dataset selected before the CPU Pod" >&2
    exit 2
fi
if [[ ! "${STAGE1_DATA_START:-}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ \
    || ! "${STAGE1_DATA_END:-}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "STAGE1_DATA_START and STAGE1_DATA_END must come from the experiment selection" >&2
    exit 2
fi
for path_name in \
    PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN RESEARCH_LOG \
    CODE_MARKER RESEARCH_MARKER AUXILIARY_ROOT OUTPUT_ROOT; do
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
if [[ ! -f "${PROJECT_ROOT}/${RESEARCH_CONFIG}" || -L "${PROJECT_ROOT}/${RESEARCH_CONFIG}" ]]; then
    echo "Research config is missing or is a symlink: ${RESEARCH_CONFIG}" >&2
    exit 2
fi
for secret_name in EODHD_API_TOKEN TPEX_PROXY_TOKEN; do
    secret_value="${!secret_name:-}"
    if [[ -z "${secret_value}" || "${secret_value}" == *'{{ RUNPOD_SECRET_'* ]]; then
        echo "${secret_name} RunPod Secret is missing or was not resolved" >&2
        exit 2
    fi
done
if [[ ! "${TPEX_PROXY_URL:-}" =~ ^https://[a-z0-9.-]+\.run\.app/?$ ]]; then
    echo "TPEX_PROXY_URL must be the deployed Cloud Run relay origin" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

mkdir -p "$(dirname "${RESEARCH_LOG}")" "$(dirname "${RESEARCH_MARKER}")" \
    "$(dirname "${OUTPUT_ROOT}")"
RESEARCH_SUCCEEDED=0

write_research_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${RESEARCH_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind research-tw-information
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${RESEARCH_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_cpu_research() {
    local research_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${RESEARCH_SUCCEEDED} -ne 1 ]]; then
        write_research_state failed "${research_exit_code}" || true
    fi
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${research_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_cpu_research EXIT

write_research_state preparing
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"
if [[ ! -f "${DATA_ROOT}/dataset-manifest.json" ]]; then
    echo "The experiment dataset is not prepared: ${DATA_ROOT}" >&2
    exit 2
fi

bash "${SCRIPT_DIR}/bootstrap_network_volume.sh"
# The research job installs the same CPU environment as data preparation.
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1

printf '[%s] research tests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run pytest -q \
    tests/test_auxiliary_data.py tests/test_tw_information.py tests/test_research_control.py

printf '[%s] auxiliary data: %s to %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STAGE1_DATA_START}" "${STAGE1_DATA_END}"
# Each provider host is requested sequentially at its own QPS; cached responses are reused.
"${POETRY_BIN}" run python -m stock_forecasting.cli.prepare_auxiliary_data \
    --output-root "${AUXILIARY_ROOT}" \
    --taiwan-qps "${RUNPOD_CPU_TAIWAN_QPS:-0.5}" \
    --eodhd-qps "${RUNPOD_CPU_EODHD_QPS:-2}" \
    --max-backoff-seconds "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS:-60}"

printf '[%s] information test\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run python -m stock_forecasting.cli.tw_information_test \
    --config "${RESEARCH_CONFIG}" \
    --auxiliary-root "${AUXILIARY_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
cp "${AUXILIARY_ROOT}/auxiliary-manifest.json" "${OUTPUT_ROOT}/auxiliary-manifest.json"

write_research_state ready
RESEARCH_SUCCEEDED=1
printf 'Information test complete: %s\n' "${OUTPUT_ROOT}"
