#!/usr/bin/env bash

# Train a cross-stock transformer experiment (protocol 13.13 D0/A1-A4/D1) on a GPU Pod.
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
RUNPOD_ROLE="${RUNPOD_ROLE:-gpu-train}"
FEATURE_SPEC="configs/features/alpha_tw_v1.json"
FEATURE_EXPERIMENT_CONFIG="${FEATURE_EXPERIMENT_CONFIG:-configs/experiments/a_rank64.yaml}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
FEATURES_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/xs-train/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for the research job; the guard checks its Pod ID.
FEATURES_MARKER="${LIFECYCLE_ROOT}/research/xs-train.json"
BASELINE_MARKER="${LIFECYCLE_ROOT}/research/lgbm.json"
OUTPUT_ROOT="${NETWORK_VOLUME_ROOT}/diagnostics/xs/${LAUNCH_ID}"
WALK_FORWARD_SCHEDULE="configs/walk_forward/tw_quarterly_v1.json"
XS_EXPERIMENT="${RUNPOD_XS_EXPERIMENT:-t-xs-scratch}"
XS_SEEDS="${RUNPOD_XS_SEEDS:-42 43 44}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_DISABLE_XET=1
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE FEATURE_EXPERIMENT_CONFIG

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Cross-stock training must run inside a RunPod GPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "gpu-train" ]]; then
    echo "Cross-stock training requires the gpu-train role" >&2
    exit 2
fi
if [[ ! "${XS_EXPERIMENT}" =~ ^t-xs-[a-z0-9-]+$ ]]; then
    echo "RUNPOD_XS_EXPERIMENT must name a t-xs-* experiment" >&2
    exit 2
fi
if [[ ! "${XS_SEEDS}" =~ ^[0-9]+( [0-9]+)*$ ]]; then
    echo "RUNPOD_XS_SEEDS must be space-separated non-negative integers" >&2
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
    CODE_MARKER FEATURES_MARKER BASELINE_MARKER OUTPUT_ROOT CACHE_ROOT; do
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
for relative in "${FEATURE_SPEC}" "${FEATURE_EXPERIMENT_CONFIG}" "${WALK_FORWARD_SCHEDULE}"; do
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
        --kind research-xs-train
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${FEATURES_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_gpu_xs() {
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
trap finish_gpu_xs EXIT

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
# The GPU environment keeps the image PyTorch and verifies a CUDA device.
RUNPOD_ROLE=gpu-train bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1

printf '[%s] cross-stock tests\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${POETRY_BIN}" run pytest -q \
    tests/test_walk_forward.py tests/test_cross_stock.py tests/test_encoders.py
bash "${SCRIPT_DIR}/ensure_runpod_data_dependencies.sh"
# The paired H-A comparison needs the finished B0 run recorded on this volume.
BASELINE_ROOT=""
if [[ -f "${BASELINE_MARKER}" ]]; then
    BASELINE_ROOT="$("${RUNPOD_PYTHON_BIN}" - "${BASELINE_MARKER}" "${NETWORK_VOLUME_ROOT}" <<'PY'
import json, sys
marker = json.load(open(sys.argv[1]))
launch = str(marker.get("launch_id", ""))
if marker.get("kind") == "research-lgbm" and marker.get("state") == "ready" and launch.startswith("launch-"):
    print(f"{sys.argv[2]}/diagnostics/lgbm/{launch}")
PY
)"
fi
if [[ -z "${BASELINE_ROOT}" || ! -f "${BASELINE_ROOT}/daily_rank_ic.parquet" ]]; then
    echo "The B0 LightGBM baseline is not ready on this volume; run cpu-lgbm first" >&2
    exit 2
fi

printf '[%s] cross-stock experiment %s seeds %s: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${XS_EXPERIMENT}" "${XS_SEEDS}" "${DATA_ROOT}"
mkdir -p "${HF_HOME}"
# A container restart loses the job but not the volume: reuse the newest earlier launch's
# finished folds of this dataset (the trainer verifies they match before using them).
RESUME_ROOT=""
for resume_candidate in "${NETWORK_VOLUME_ROOT}/diagnostics/xs"/launch-*/partial; do
    # The glob stays literal when nothing matches; only real directories qualify.
    if [[ -d "${resume_candidate}" ]]; then
        RESUME_ROOT="$(dirname "${resume_candidate}")"
    fi
done
read -r -a XS_SEED_LIST <<< "${XS_SEEDS}"
XS_ARGUMENTS=(
    --experiment-config "${FEATURE_EXPERIMENT_CONFIG}"
    --spec "${FEATURE_SPEC}"
    --schedule "${WALK_FORWARD_SCHEDULE}"
    --experiment "${XS_EXPERIMENT}"
    --seeds "${XS_SEED_LIST[@]}"
    --output-root "${OUTPUT_ROOT}"
    --baseline-root "${BASELINE_ROOT}"
)
if [[ -n "${RESUME_ROOT}" && -d "${RESUME_ROOT}" ]]; then
    printf 'Resuming finished folds from %s\n' "${RESUME_ROOT}"
    XS_ARGUMENTS+=(--resume-root "${RESUME_ROOT}")
fi
if [[ -n "${FIN_TS_XS_CACHE_WORKERS:-}" ]]; then
    XS_ARGUMENTS+=(--workers "${FIN_TS_XS_CACHE_WORKERS}")
fi
"${POETRY_BIN}" run stock-forecasting-train-cross-stock "${XS_ARGUMENTS[@]}"

write_features_state ready
FEATURES_SUCCEEDED=1
printf 'Cross-stock experiment complete: %s\n' "${OUTPUT_ROOT}"
