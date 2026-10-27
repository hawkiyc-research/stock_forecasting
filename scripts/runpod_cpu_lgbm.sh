#!/usr/bin/env bash

# Train the B0 LightGBM walk-forward baseline (protocol 13.13) on a CPU Pod.
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
# Protocol 13.18: the feature spec may be a chip spec under configs/features/, whose store
# identity needs the tw_chip_data curated layer on the volume (RUNPOD_CHIP_DATA_ROOT).
FEATURE_SPEC="${RUNPOD_FEATURE_SPEC:-configs/features/alpha_tw_v1.json}"
CHIP_DATA_ROOT="${RUNPOD_CHIP_DATA_ROOT:-}"
FEATURE_EXPERIMENT_CONFIG="${FEATURE_EXPERIMENT_CONFIG:-configs/experiments/a_rank64.yaml}"
# Protocol 13.16: an optional residualized label spec under configs/labels/.
LABEL_VARIANT="${RUNPOD_LABEL_VARIANT:-}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
FEATURES_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/cpu-lgbm/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for the research job; the guard checks its Pod ID.
FEATURES_MARKER="${LIFECYCLE_ROOT}/research/lgbm.json"
OUTPUT_ROOT="${NETWORK_VOLUME_ROOT}/diagnostics/lgbm/${LAUNCH_ID}"
WALK_FORWARD_SCHEDULE="configs/walk_forward/tw_quarterly_v1.json"
LGBM_SCRATCH_ROOT="${NETWORK_VOLUME_ROOT}/tmp/lgbm-${LAUNCH_ID}"
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE FEATURE_EXPERIMENT_CONFIG

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "The LightGBM baseline must run inside a RunPod CPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-research" ]]; then
    echo "The LightGBM baseline requires the CPU-only cpu-research role" >&2
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
if [[ -n "${LABEL_VARIANT}" && ! "${LABEL_VARIANT}" =~ ^configs/labels/[a-z0-9_]+\.json$ ]]; then
    echo "RUNPOD_LABEL_VARIANT must name a file under configs/labels/" >&2
    exit 2
fi
if [[ ! "${FEATURE_SPEC}" =~ ^configs/features/[a-z0-9_]+\.json$ ]]; then
    echo "RUNPOD_FEATURE_SPEC must name a file under configs/features/" >&2
    exit 2
fi
if [[ -n "${CHIP_DATA_ROOT}" ]]; then
    runpod_validate_path_in_root \
        "${CHIP_DATA_ROOT}" "${NETWORK_VOLUME_ROOT}" CHIP_DATA_ROOT NETWORK_VOLUME_ROOT
    if [[ ! -d "${CHIP_DATA_ROOT}/curated" ]]; then
        echo "RUNPOD_CHIP_DATA_ROOT has no curated/ directory: ${CHIP_DATA_ROOT}" >&2
        exit 2
    fi
    export TW_CHIP_DATA_ROOT="${CHIP_DATA_ROOT}"
fi
for path_name in \
    PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN FEATURES_LOG \
    CODE_MARKER FEATURES_MARKER OUTPUT_ROOT LGBM_SCRATCH_ROOT; do
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
for relative in "${FEATURE_SPEC}" "${FEATURE_EXPERIMENT_CONFIG}" "${WALK_FORWARD_SCHEDULE}" \
    ${LABEL_VARIANT:+"${LABEL_VARIANT}"}; do
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
        --kind research-lgbm
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${FEATURES_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_cpu_lgbm() {
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
trap finish_cpu_lgbm EXIT

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
# The baseline installs the same CPU environment as data preparation (LightGBM is locked there).
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1

printf '[%s] walk-forward tests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run pytest -q \
    tests/test_walk_forward.py tests/test_walk_forward_lgbm.py

printf '[%s] LightGBM walk-forward baseline: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATA_ROOT}"
mkdir -p "${LGBM_SCRATCH_ROOT}"
# A container restart loses the job but not the volume: reuse the newest UNFINISHED earlier
# launch's folds (the trainer verifies label variant and feature spec before using them). A
# launch that already wrote results.json is a result, never a resume source: resuming from one
# would re-emit its predictions under a new launch id instead of training anything.
RESUME_ROOT=""
for resume_candidate in "${NETWORK_VOLUME_ROOT}/diagnostics/lgbm"/launch-*/partial; do
    # The glob stays literal when nothing matches; only real directories qualify.
    if [[ -d "${resume_candidate}" && ! -f "$(dirname "${resume_candidate}")/results.json" ]]; then
        RESUME_ROOT="$(dirname "${resume_candidate}")"
    fi
done
LGBM_ARGUMENTS=(
    --experiment-config "${FEATURE_EXPERIMENT_CONFIG}"
    --spec "${FEATURE_SPEC}"
    --schedule "${WALK_FORWARD_SCHEDULE}"
    --output-root "${OUTPUT_ROOT}"
    --scratch "${LGBM_SCRATCH_ROOT}"
)
if [[ -n "${RESUME_ROOT}" && -d "${RESUME_ROOT}" ]]; then
    printf 'Resuming finished folds from %s\n' "${RESUME_ROOT}"
    LGBM_ARGUMENTS+=(--resume-root "${RESUME_ROOT}")
fi
if [[ -n "${FIN_TS_LGBM_WORKERS:-}" ]]; then
    LGBM_ARGUMENTS+=(--workers "${FIN_TS_LGBM_WORKERS}")
fi
if [[ -n "${LABEL_VARIANT}" ]]; then
    # The trainer only resumes partial folds saved under the same label variant.
    printf 'Residualized label variant: %s\n' "${LABEL_VARIANT}"
    LGBM_ARGUMENTS+=(--label-variant "${LABEL_VARIANT}")
fi
if [[ -n "${CHIP_DATA_ROOT}" ]]; then
    printf 'Feature spec %s with chip data root %s\n' "${FEATURE_SPEC}" "${CHIP_DATA_ROOT}"
    LGBM_ARGUMENTS+=(--chip-data-root "${CHIP_DATA_ROOT}")
fi
"${POETRY_BIN}" run stock-forecasting-walk-forward-lgbm "${LGBM_ARGUMENTS[@]}"
rm -rf "${LGBM_SCRATCH_ROOT}"

write_features_state ready
FEATURES_SUCCEEDED=1
printf 'LightGBM baseline complete: %s\n' "${OUTPUT_ROOT}"
