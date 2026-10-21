#!/usr/bin/env bash

# Build rank labels and the feature library for the selected dataset on a CPU Pod.
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
FEATURE_SPEC="configs/features/alpha_tw_v1.json"
FEATURE_EXPERIMENT_CONFIG="${FEATURE_EXPERIMENT_CONFIG:-configs/experiments/a_rank64.yaml}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
FEATURES_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/cpu-features/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for the research job; the guard checks its Pod ID.
FEATURES_MARKER="${LIFECYCLE_ROOT}/research/features.json"
OUTPUT_ROOT="${NETWORK_VOLUME_ROOT}/diagnostics/features/${LAUNCH_ID}"
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE FEATURE_EXPERIMENT_CONFIG

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "The feature build must run inside a RunPod CPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-research" ]]; then
    echo "The feature build requires the CPU-only cpu-research role" >&2
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
if [[ ! "${FEATURE_EXPERIMENT_CONFIG}" =~ ^configs/experiments/[a-z0-9_]+\.yaml$ ]]; then
    echo "FEATURE_EXPERIMENT_CONFIG must name a file under configs/experiments/" >&2
    exit 2
fi
for path_name in \
    PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN FEATURES_LOG \
    CODE_MARKER FEATURES_MARKER OUTPUT_ROOT; do
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
for relative in "${FEATURE_SPEC}" "${FEATURE_EXPERIMENT_CONFIG}"; do
    if [[ ! -f "${PROJECT_ROOT}/${relative}" || -L "${PROJECT_ROOT}/${relative}" ]]; then
        echo "Required config is missing or is a symlink: ${relative}" >&2
        exit 2
    fi
done
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

mkdir -p "$(dirname "${FEATURES_LOG}")" "$(dirname "${FEATURES_MARKER}")" \
    "$(dirname "${OUTPUT_ROOT}")"
FEATURES_SUCCEEDED=0

write_features_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${FEATURES_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind research-features
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${FEATURES_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_cpu_features() {
    local features_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${FEATURES_SUCCEEDED} -ne 1 ]]; then
        write_features_state failed "${features_exit_code}" || true
    fi
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${features_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_cpu_features EXIT

write_features_state preparing
# As in the other CPU workflows: move sources outside this release (for example
# files another worktree synced) to lifecycle/quarantine/ before checking the code.
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" quarantine-stale-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"
if [[ ! -f "${DATA_ROOT}/dataset-manifest.json" ]]; then
    echo "The experiment dataset is not prepared: ${DATA_ROOT}" >&2
    exit 2
fi

bash "${SCRIPT_DIR}/bootstrap_network_volume.sh"
# The feature job installs the same CPU environment as data preparation.
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1

printf '[%s] feature tests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run pytest -q \
    tests/test_rank_labels.py tests/test_feature_library.py tests/test_walk_forward.py

printf '[%s] rank labels and feature library: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATA_ROOT}"
"${POETRY_BIN}" run stock-forecasting-build-features \
    --experiment-config "${FEATURE_EXPERIMENT_CONFIG}" \
    --spec "${FEATURE_SPEC}" \
    --output-root "${OUTPUT_ROOT}"

write_features_state ready
FEATURES_SUCCEEDED=1
printf 'Feature build complete: %s\n' "${OUTPUT_ROOT}"
