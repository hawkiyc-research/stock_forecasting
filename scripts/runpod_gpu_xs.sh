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
OUTPUT_ROOT="${NETWORK_VOLUME_ROOT}/diagnostics/xs/${LAUNCH_ID}"
WALK_FORWARD_SCHEDULE="configs/walk_forward/tw_quarterly_v1.json"
XS_EXPERIMENT="${RUNPOD_XS_EXPERIMENT:-t-xs-scratch}"
XS_SEEDS="${RUNPOD_XS_SEEDS:-42 43 44}"
# Protocol 13.16: train on a residualized label variant and pair against the B0 run of the
# same variant; empty = the standard labels.
LABEL_VARIANT="${RUNPOD_LABEL_VARIANT:-}"
# Optional explicit B0 launch to pair against (launch-...); default = newest finished B0 of
# the same label variant on this volume.
XS_BASELINE_LAUNCH="${RUNPOD_XS_BASELINE_LAUNCH:-}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_DISABLE_XET=1
# torch.compile artifacts (Inductor FX graphs, Triton kernels, autotune results) live on
# the volume instead of the container's /tmp, so a later Pod with the same GPU model and
# torch build skips the cold compile; cache keys include the device name, so different
# GPUs never share entries. CUDA-graph capture itself is per process and stays.
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torch-compile/inductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/torch-compile/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
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
if [[ -n "${LABEL_VARIANT}" && ! "${LABEL_VARIANT}" =~ ^configs/labels/[a-z0-9_]+\.json$ ]]; then
    echo "RUNPOD_LABEL_VARIANT must name a file under configs/labels/" >&2
    exit 2
fi
if [[ -n "${XS_BASELINE_LAUNCH}" && ! "${XS_BASELINE_LAUNCH}" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    echo "RUNPOD_XS_BASELINE_LAUNCH must be a launch-... ID" >&2
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
    CODE_MARKER FEATURES_MARKER OUTPUT_ROOT CACHE_ROOT; do
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
    tests/test_walk_forward.py tests/test_cross_stock.py tests/test_encoders.py \
    tests/test_verify_resume_rules.py
bash "${SCRIPT_DIR}/ensure_runpod_data_dependencies.sh"
# The paired H-A comparison needs a finished B0 run of the same label variant on this
# volume: the newest diagnostics/lgbm/launch-*/results.json whose label_variant matches (the
# lgbm marker only remembers the latest launch, which may be another variant's).
BASELINE_ROOT="$("${RUNPOD_PYTHON_BIN}" - "${NETWORK_VOLUME_ROOT}" "${PROJECT_ROOT}" \
    "${LABEL_VARIANT}" "${XS_BASELINE_LAUNCH}" <<'PY'
import json, sys
from pathlib import Path
volume, project, spec_path, explicit = sys.argv[1:5]
wanted = json.load(open(Path(project) / spec_path))["name"] if spec_path else None
root = Path(volume) / "diagnostics" / "lgbm"
candidates = [root / explicit] if explicit else sorted(root.glob("launch-*"), reverse=True)
for launch in candidates:
    results = launch / "results.json"
    if not results.is_file() or not (launch / "daily_rank_ic.parquet").is_file():
        continue
    payload = json.load(open(results))
    if (payload.get("label_variant") or None) != wanted:
        continue
    print(launch)
    break
PY
)"
if [[ -z "${BASELINE_ROOT}" ]]; then
    echo "No finished B0 LightGBM run of label variant '${LABEL_VARIANT:-standard}' is on this volume; run cpu-lgbm first" >&2
    exit 2
fi
printf 'Paired baseline: %s\n' "${BASELINE_ROOT}"

printf '[%s] cross-stock experiment %s seeds %s labels %s: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${XS_EXPERIMENT}" "${XS_SEEDS}" \
    "${LABEL_VARIANT:-standard}" "${DATA_ROOT}"
mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
# A container restart loses the job but not the volume: every UNFINISHED earlier launch is
# offered to the trainer, which reuses, per fold, the newest one whose folds belong to this
# run (same experiment / labels / feature spec / schedule / dataset) and stamps every source it
# used with resumed_into.json. A launch that wrote results.json is a result, never a resume
# source: resuming from one would re-emit its predictions under a new launch id instead of
# training anything. A stamped launch was absorbed by a later one (its partial/ never gains a
# results.json) and is skipped too.
RESUME_ROOTS=()
for resume_candidate in "${NETWORK_VOLUME_ROOT}/diagnostics/xs"/launch-*/partial; do
    # The glob stays literal when nothing matches; only real directories qualify.
    if [[ -d "${resume_candidate}" && ! -f "$(dirname "${resume_candidate}")/results.json" \
        && ! -f "$(dirname "${resume_candidate}")/resumed_into.json" ]]; then
        RESUME_ROOTS+=("$(dirname "${resume_candidate}")")
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
for resume_root in ${RESUME_ROOTS[@]+"${RESUME_ROOTS[@]}"}; do
    printf 'Resume candidate (used only for folds of this run): %s\n' "${resume_root}"
    XS_ARGUMENTS+=(--resume-root "${resume_root}")
done
if [[ -n "${FIN_TS_XS_CACHE_WORKERS:-}" ]]; then
    XS_ARGUMENTS+=(--workers "${FIN_TS_XS_CACHE_WORKERS}")
fi
if [[ -n "${LABEL_VARIANT}" ]]; then
    # The trainer only resumes partial folds saved under the same label variant.
    printf 'Residualized label variant: %s\n' "${LABEL_VARIANT}"
    XS_ARGUMENTS+=(--label-variant "${LABEL_VARIANT}")
fi
"${POETRY_BIN}" run stock-forecasting-train-cross-stock "${XS_ARGUMENTS[@]}"

write_features_state ready
FEATURES_SUCCEEDED=1
printf 'Cross-stock experiment complete: %s\n' "${OUTPUT_ROOT}"
