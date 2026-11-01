#!/usr/bin/env bash

# Create a short-lived CPU preparation Pod after uploaded source passes its readiness gate.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
if [[ "${RUNPOD_TEST_MODE:-0}" != 1 && "${RUNPOD_CREATE_DRY_RUN:-0}" != 1 \
    && "${RUNPOD_LAUNCH_CONTROL_LOCK_HELD:-0}" != 1 ]]; then
    exec python3 "${SCRIPT_DIR}/runpod_concurrency.py" create --kind cpu
fi
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"
runpod_load_active_selection "${LOCAL_PROJECT_ROOT}"

RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUNPOD_VOLUME_MOUNT_PATH}/stock_forecasting}"
# research runs the Phase 1 information test and features builds the rank labels and the
# feature library of the selected dataset; neither prepares or finalizes a dataset.
RUNPOD_CPU_WORKFLOW="${RUNPOD_CPU_WORKFLOW:-prepare}"
case "${RUNPOD_CPU_WORKFLOW}" in
    prepare|research|features|lgbm|chip) ;;
    *)
        echo "RUNPOD_CPU_WORKFLOW must be prepare, research, features, lgbm or chip" >&2
        exit 2
        ;;
esac
# prepare Pods of a Taiwan profile update the tw_chip_data layers after the dataset
# (protocol 13.20); 0 skips it (runpod_workflow.sh cpu prepare --skipChip).
RUNPOD_CHIP_DATA_PREPARE="${RUNPOD_CHIP_DATA_PREPARE:-0}"
if [[ ! "${RUNPOD_CHIP_DATA_PREPARE}" =~ ^[01]$ ]]; then
    echo "RUNPOD_CHIP_DATA_PREPARE must be 0 or 1" >&2
    exit 2
fi
if [[ "${RUNPOD_CHIP_DATA_PREPARE}" == 1 && "${RUNPOD_CPU_WORKFLOW}" != "prepare" ]]; then
    echo "RUNPOD_CHIP_DATA_PREPARE applies to the prepare workflow only" >&2
    exit 2
fi
# prepare builds the selected training dataset, or in forward mode a forward input
# package (protocol 13.8) from official Taiwan data; forward never reads or writes a
# training dataset, its readiness or the active selection.
RUNPOD_CPU_PREPARE_MODE="${RUNPOD_CPU_PREPARE_MODE:-dataset}"
RUNPOD_FORWARD_END="${RUNPOD_FORWARD_END:-}"
case "${RUNPOD_CPU_PREPARE_MODE}" in
    dataset) RUNPOD_FORWARD_END="" ;;
    forward)
        if [[ "${RUNPOD_CPU_WORKFLOW}" != "prepare" ]]; then
            echo "Forward mode belongs to the prepare workflow" >&2
            exit 2
        fi
        if [[ ! "${RUNPOD_FORWARD_END}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
            echo "RUNPOD_FORWARD_END must use YYYY-MM-DD" >&2
            exit 2
        fi
        ;;
    *)
        echo "RUNPOD_CPU_PREPARE_MODE must be dataset or forward" >&2
        exit 2
        ;;
esac
RUNPOD_POD_NAME="${RUNPOD_CPU_POD_NAME:-stock-forecasting-cpu-prepare}"
if [[ "${RUNPOD_CPU_WORKFLOW}" == "research" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_CPU_RESEARCH_POD_NAME:-stock-forecasting-cpu-research}"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "features" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_CPU_FEATURES_POD_NAME:-stock-forecasting-cpu-features}"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "lgbm" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_CPU_LGBM_POD_NAME:-stock-forecasting-cpu-lgbm}"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "chip" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_CPU_CHIP_POD_NAME:-stock-forecasting-cpu-chip}"
elif [[ "${RUNPOD_CPU_PREPARE_MODE}" == "forward" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_CPU_FORWARD_POD_NAME:-stock-forecasting-cpu-prepare-forward}"
fi
runpod_validate_project_resource_name "${RUNPOD_POD_NAME}" RUNPOD_CPU_POD_NAME
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_HF_SECRET_NAME="${RUNPOD_HF_SECRET_NAME:-huggingface_token}"
RUNPOD_EODHD_SECRET_NAME="${RUNPOD_EODHD_SECRET_NAME:-eodhd_api_token}"
RUNPOD_TPEX_PROXY_SECRET_NAME="${RUNPOD_TPEX_PROXY_SECRET_NAME:-}"
TPEX_PROXY_URL="${TPEX_PROXY_URL:-}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
RUNPOD_DATACENTER_ID="${RUNPOD_DATACENTER_ID:-EU-RO-1}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CPU_CONTAINER_DISK_GB:-}"
RUNPOD_API_BASE_URL="${RUNPOD_API_BASE_URL:-https://api.runpod.io/v2}"
RUNPOD_CPU_FLAVOR_ID="${RUNPOD_CPU_FLAVOR_ID:-cpu3g}"
RUNPOD_CPU_VCPU_COUNT="${RUNPOD_CPU_VCPU_COUNT:-8}"
RUNPOD_CPU_MAX_RUNTIME_SECONDS="${RUNPOD_CPU_MAX_RUNTIME_SECONDS:-21600}"
RUNPOD_CPU_PREPARE_RESERVE_SECONDS="${RUNPOD_CPU_PREPARE_RESERVE_SECONDS:-}"
RUNPOD_CPU_MAX_API_CALLS="${RUNPOD_CPU_MAX_API_CALLS:-}"
RUNPOD_CPU_EODHD_QPS="${RUNPOD_CPU_EODHD_QPS:-}"
RUNPOD_CPU_TAIWAN_QPS="${RUNPOD_CPU_TAIWAN_QPS:-}"
RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS="${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS:-60}"
RUNPOD_CPU_HARD_LIMIT_SECONDS="${RUNPOD_CPU_HARD_LIMIT_SECONDS:-25200}"
# Optional protocol 13.16 residualized label spec (features and lgbm workflows only).
RUNPOD_LABEL_VARIANT="${RUNPOD_LABEL_VARIANT:-}"
if [[ -n "${RUNPOD_LABEL_VARIANT}" ]]; then
    if [[ ! "${RUNPOD_LABEL_VARIANT}" =~ ^configs/labels/[a-z0-9_]+\.json$ ]]; then
        echo "RUNPOD_LABEL_VARIANT must name a file under configs/labels/" >&2
        exit 2
    fi
    if [[ "${RUNPOD_CPU_WORKFLOW}" != "features" && "${RUNPOD_CPU_WORKFLOW}" != "lgbm" ]]; then
        echo "RUNPOD_LABEL_VARIANT applies to the features and lgbm workflows only" >&2
        exit 2
    fi
fi
# Optional protocol 13.18 feature spec and chip data root (features and lgbm workflows only).
RUNPOD_FEATURE_SPEC="${RUNPOD_FEATURE_SPEC:-}"
RUNPOD_CHIP_DATA_ROOT="${RUNPOD_CHIP_DATA_ROOT:-}"
if [[ -n "${RUNPOD_FEATURE_SPEC}" || -n "${RUNPOD_CHIP_DATA_ROOT}" ]]; then
    if [[ "${RUNPOD_CPU_WORKFLOW}" != "features" && "${RUNPOD_CPU_WORKFLOW}" != "lgbm" ]]; then
        echo "RUNPOD_FEATURE_SPEC and RUNPOD_CHIP_DATA_ROOT apply to features and lgbm only" >&2
        exit 2
    fi
fi
if [[ -n "${RUNPOD_FEATURE_SPEC}" && ! "${RUNPOD_FEATURE_SPEC}" =~ ^configs/features/[a-z0-9_]+\.json$ ]]; then
    echo "RUNPOD_FEATURE_SPEC must name a file under configs/features/" >&2
    exit 2
fi
if [[ -n "${RUNPOD_CHIP_DATA_ROOT}" && ! "${RUNPOD_CHIP_DATA_ROOT}" =~ ^/runpod-volume/[A-Za-z0-9._/-]+$ ]]; then
    echo "RUNPOD_CHIP_DATA_ROOT must be an absolute path under /runpod-volume" >&2
    exit 2
fi
# Optional protocol 13.20 walk-forward schedule (lgbm workflow only).
RUNPOD_WALK_FORWARD_SCHEDULE="${RUNPOD_WALK_FORWARD_SCHEDULE:-}"
if [[ -n "${RUNPOD_WALK_FORWARD_SCHEDULE}" ]]; then
    if [[ "${RUNPOD_CPU_WORKFLOW}" != "lgbm" ]]; then
        echo "RUNPOD_WALK_FORWARD_SCHEDULE applies to the lgbm workflow only" >&2
        exit 2
    fi
    if [[ ! "${RUNPOD_WALK_FORWARD_SCHEDULE}" =~ ^configs/walk_forward/[a-z0-9_]+\.json$ ]]; then
        echo "RUNPOD_WALK_FORWARD_SCHEDULE must name a file under configs/walk_forward/" >&2
        exit 2
    fi
fi
# Optional 2026-11-02 rule: the finished lgbm launch to carry on (lgbm workflow only).
RUNPOD_LGBM_EXTEND_LAUNCH="${RUNPOD_LGBM_EXTEND_LAUNCH:-}"
if [[ -n "${RUNPOD_LGBM_EXTEND_LAUNCH}" ]]; then
    if [[ "${RUNPOD_CPU_WORKFLOW}" != "lgbm" ]]; then
        echo "RUNPOD_LGBM_EXTEND_LAUNCH applies to the lgbm workflow only" >&2
        exit 2
    fi
    if [[ ! "${RUNPOD_LGBM_EXTEND_LAUNCH}" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
        echo "RUNPOD_LGBM_EXTEND_LAUNCH must be a launch-... ID" >&2
        exit 2
    fi
fi
RUNPOD_GUARD_LOG_DIR="${RUNPOD_GUARD_LOG_DIR:-${HOME:-/tmp}/.local/state/runpod-guards}"
RUNPOD_GUARD_LAUNCHER="${SCRIPT_DIR}/launch_runpod_guard.sh"
RUNPOD_CPU_ROLE=cpu-prep
RUNPOD_CPU_TMUX_WORKFLOW=cpu-prepare
RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/stage1/cpu-preparation.json
RUNPOD_CPU_POD_LABEL="CPU preparation"
if [[ "${RUNPOD_CPU_WORKFLOW}" == "research" ]]; then
    RUNPOD_CPU_ROLE=cpu-research
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-research
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/research/tw-information.json
    RUNPOD_CPU_POD_LABEL="CPU research"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "features" ]]; then
    # Same CPU-only role as research (no model or W&B access); its own lifecycle marker.
    RUNPOD_CPU_ROLE=cpu-research
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-features
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/research/features.json
    RUNPOD_CPU_POD_LABEL="CPU features"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "lgbm" ]]; then
    # The LightGBM walk-forward baseline (protocol 13.13): same role, its own marker.
    RUNPOD_CPU_ROLE=cpu-research
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-lgbm
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/research/lgbm.json
    RUNPOD_CPU_POD_LABEL="CPU LightGBM baseline"
elif [[ "${RUNPOD_CPU_WORKFLOW}" == "chip" ]]; then
    # The tw_chip_data update (protocol 13.20): the cpu-prep role carries the relay secret.
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-chip
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/research/chip-data.json
    RUNPOD_CPU_POD_LABEL="CPU chip data"
elif [[ "${RUNPOD_CPU_PREPARE_MODE}" == "forward" ]]; then
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/forward/cpu-preparation.json
    RUNPOD_CPU_POD_LABEL="CPU forward preparation"
elif [[ "${RUNPOD_STAGE}" == "stage2" ]]; then
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-finalize
    RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/stage1/mixed-finalization.json
fi
# Taiwan acquisition (stage 1 preparation, forward preparation or research) reaches TPEx
# through the relay.
REQUIRE_TPEX_RELAY=0
if [[ "${RUNPOD_CPU_WORKFLOW}" == "research" || "${RUNPOD_CPU_WORKFLOW}" == "chip" \
    || "${RUNPOD_CPU_PREPARE_MODE}" == "forward" \
    || ( "${RUNPOD_CPU_WORKFLOW}" == "prepare" && "${RUNPOD_STAGE}" == "stage1" \
        && ( "${FIN_TS_DATASET_PROFILE}" == "tw_only" \
            || "${FIN_TS_DATASET_PROFILE}" == "us_tw_eodhd" \
            || "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ) ) ]]; then
    REQUIRE_TPEX_RELAY=1
fi
# The chained chip update only makes sense for Taiwan data; other profiles never run it.
if [[ "${RUNPOD_CHIP_DATA_PREPARE}" == 1 && ( "${RUNPOD_CPU_PREPARE_MODE}" != "dataset" \
    || ( "${FIN_TS_DATASET_PROFILE}" != "tw_only" \
        && "${FIN_TS_DATASET_PROFILE}" != "us_tw_eodhd" \
        && "${FIN_TS_DATASET_PROFILE}" != "us_tw_massive" ) ) ]]; then
    RUNPOD_CHIP_DATA_PREPARE=0
fi

if [[ -n "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "create_runpod_cpu_pod.sh must run outside the RunPod Pod" >&2
    exit 2
fi
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required and must use safe characters" >&2
    exit 2
fi
runpod_validate_absolute_path "${RUNPOD_VOLUME_MOUNT_PATH}" RUNPOD_VOLUME_MOUNT_PATH
case "${RUNPOD_VOLUME_MOUNT_PATH}" in
    /workspace|/workspace/*)
        echo "Network volumes must not be mounted under /workspace" >&2
        exit 2
        ;;
esac
runpod_validate_path_in_root \
    "${PROJECT_ROOT}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    PROJECT_ROOT RUNPOD_VOLUME_MOUNT_PATH
if [[ "${PROJECT_ROOT}" == "${RUNPOD_VOLUME_MOUNT_PATH}" ]]; then
    echo "PROJECT_ROOT must be a child of RUNPOD_VOLUME_MOUNT_PATH" >&2
    exit 2
fi
if [[ ! "${RUNPOD_VOLUME_MOUNT_PATH}" =~ ^/[A-Za-z0-9._/-]+$ \
    || ! "${PROJECT_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "Volume and project paths contain unsupported characters" >&2
    exit 2
fi
if [[ ! "${RUNPOD_DATACENTER_ID}" =~ ^[A-Z0-9]+(-[A-Z0-9]+)+$ ]]; then
    echo "RUNPOD_DATACENTER_ID has an invalid format" >&2
    exit 2
fi
if [[ "${RUNPOD_API_BASE_URL%/}" != "https://api.runpod.io/v2" ]]; then
    echo "RUNPOD_API_BASE_URL must be https://api.runpod.io/v2" >&2
    exit 2
fi
case "${RUNPOD_CPU_FLAVOR_ID}" in
    cpu3c|cpu3g|cpu3m|cpu5c|cpu5g|cpu5m) ;;
    *)
        echo "RUNPOD_CPU_FLAVOR_ID must be one of: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m" >&2
        exit 2
        ;;
esac
case "${RUNPOD_CPU_VCPU_COUNT}" in
    2|4|8|16|32) ;;
    *)
        echo "RUNPOD_CPU_VCPU_COUNT must be one of: 2, 4, 8, 16, 32" >&2
        exit 2
        ;;
esac
case "${RUNPOD_CPU_FLAVOR_ID}" in
    cpu3c|cpu3g|cpu3m) CONTAINER_DISK_GB_PER_VCPU=10 ;;
    cpu5c|cpu5g|cpu5m) CONTAINER_DISK_GB_PER_VCPU=15 ;;
esac
MAX_CONTAINER_DISK_GB=$((RUNPOD_CPU_VCPU_COUNT * CONTAINER_DISK_GB_PER_VCPU))
if [[ -z "${RUNPOD_CONTAINER_DISK_GB}" ]]; then
    RUNPOD_CONTAINER_DISK_GB=30
    if [[ "${RUNPOD_CONTAINER_DISK_GB}" -gt "${MAX_CONTAINER_DISK_GB}" ]]; then
        RUNPOD_CONTAINER_DISK_GB="${MAX_CONTAINER_DISK_GB}"
    fi
elif [[ ! "${RUNPOD_CONTAINER_DISK_GB}" =~ ^[1-9][0-9]*$ \
    || "${RUNPOD_CONTAINER_DISK_GB}" -gt "${MAX_CONTAINER_DISK_GB}" ]]; then
    printf 'RUNPOD_CPU_CONTAINER_DISK_GB must be an integer from 1 through %d for %s with %s vCPUs\n' \
        "${MAX_CONTAINER_DISK_GB}" "${RUNPOD_CPU_FLAVOR_ID}" \
        "${RUNPOD_CPU_VCPU_COUNT}" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ \
    || ! "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU runtime limits must be positive integers" >&2
    exit 2
fi
if [[ "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" -le "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" ]]; then
    echo "RUNPOD_CPU_HARD_LIMIT_SECONDS must exceed RUNPOD_CPU_MAX_RUNTIME_SECONDS" >&2
    exit 2
fi
if [[ ! "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ -n "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" ]]; then
    if [[ ! "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" =~ ^[1-9][0-9]*$ \
        || "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" -ge "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" ]]; then
        echo "RUNPOD_CPU_PREPARE_RESERVE_SECONDS must be positive and shorter than max runtime" >&2
        exit 2
    fi
fi
if [[ ! "${RUNPOD_IMAGE}" =~ ^[A-Za-z0-9._/:+-]+$ ]]; then
    echo "RunPod image contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
    || "${RUNPOD_CONFIG}" == /* \
    || "/${RUNPOD_CONFIG}/" == *"/../"* \
    || "/${RUNPOD_CONFIG}/" == *"/./"* \
    || "${RUNPOD_CONFIG}" == *"//"* ]]; then
    echo "RUNPOD_CONFIG must be a safe path relative to PROJECT_ROOT" >&2
    exit 2
fi
LOCAL_CONFIG_PATH="${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}"
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" && ! -r "${LOCAL_CONFIG_PATH}" ]]; then
    echo "RUNPOD_CONFIG is not readable in the uploaded local project: ${LOCAL_CONFIG_PATH}" >&2
    exit 2
fi
if [[ ! "${RUNPOD_HF_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ \
    || ! "${RUNPOD_EODHD_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RunPod secret names contain unsupported characters" >&2
    exit 2
fi
if [[ ! "${FIN_TS_DATASET_PROFILE}" =~ ^(tw_only|us_only_eodhd|us_tw_eodhd|us_tw_massive)$ ]]; then
    echo "FIN_TS_DATASET_PROFILE is unsupported" >&2
    exit 2
fi
if [[ ! "${FIN_TS_H_START}" =~ ^[1-3]$ ]]; then
    echo "FIN_TS_H_START must be 1, 2, or 3" >&2
    exit 2
fi
if [[ "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ]]; then
    echo "The Massive provider interface is reserved but not implemented" >&2
    exit 2
fi
if [[ ${REQUIRE_TPEX_RELAY} -eq 1 ]]; then
    if [[ ! "${RUNPOD_TPEX_PROXY_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "RunPod TPEx relay secret name is missing or invalid; deploy the Cloud Run relay first" >&2
        exit 2
    fi
    if [[ ! "${TPEX_PROXY_URL}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
        echo "TPEX_PROXY_URL is missing or is not an approved run.app origin" >&2
        exit 2
    fi
fi
for symbol_list in "${STAGE1_US_SYMBOLS}" "${STAGE1_US_ETF_SYMBOLS}"; do
    if [[ -n "${symbol_list}" \
        && ! "${symbol_list}" =~ ^[A-Za-z0-9.^_-]+([[:space:]]+[A-Za-z0-9.^_-]+)*$ ]]; then
        echo "Optional US symbol lists contain unsupported characters" >&2
        exit 2
    fi
done
if [[ ! "${STAGE1_DATA_START}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ \
    || ! "${STAGE1_DATA_END}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ \
    || ! "${STAGE1_TEST_END:-2026-06-01}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Stage 1 dates must use YYYY-MM-DD" >&2
    exit 2
fi
STAGE1_TEST_END="${STAGE1_TEST_END:-2026-06-01}"
if [[ -n "${STAGE1_SYMBOL_LIMIT}" && ! "${STAGE1_SYMBOL_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STAGE1_SYMBOL_LIMIT must be empty or a positive integer" >&2
    exit 2
fi
# Forward mode fetches only TWSE/TPEx, so it has no EODHD budget or rate; the feature
# build and the LightGBM baseline fetch nothing at all; the chip update has its own
# per-host spacing inside tw_chip_data.
if [[ "${RUNPOD_CPU_PREPARE_MODE}" == "forward" || "${RUNPOD_CPU_WORKFLOW}" == "features" \
    || "${RUNPOD_CPU_WORKFLOW}" == "lgbm" || "${RUNPOD_CPU_WORKFLOW}" == "chip" ]]; then
    RUNPOD_CPU_MAX_API_CALLS=""
    RUNPOD_CPU_EODHD_QPS=""
    RUNPOD_CPU_PREPARE_RESERVE_SECONDS=""
fi
if [[ "${RUNPOD_CPU_WORKFLOW}" == "features" || "${RUNPOD_CPU_WORKFLOW}" == "lgbm" \
    || "${RUNPOD_CPU_WORKFLOW}" == "chip" ]]; then
    RUNPOD_CPU_TAIWAN_QPS=""
elif [[ ( "${RUNPOD_CPU_WORKFLOW}" == "prepare" && "${RUNPOD_CPU_PREPARE_MODE}" == "dataset" \
        && ! "${RUNPOD_CPU_MAX_API_CALLS}" =~ ^[1-9][0-9]*$ ) \
    || ( "${RUNPOD_CPU_PREPARE_MODE}" == "dataset" \
        && ( ! "${RUNPOD_CPU_EODHD_QPS}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ \
            || "${RUNPOD_CPU_EODHD_QPS}" =~ ^0([.]0+)?$ ) ) \
    || ! "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ \
    || "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^0([.]0+)?$ ]]; then
    echo "CPU acquisition budget and QPS settings are invalid" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

if [[ "${RUNPOD_TEST_READINESS_READY:-0}" == "1" ]]; then
    if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
        echo "RUNPOD_TEST_READINESS_READY is allowed only in test mode" >&2
        exit 2
    fi
else
    bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" --code-only
fi

REMOTE_SELECTION_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}"
DATA_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}/datasets/${RUNPOD_DATASET_REQUEST_SHA256}"
if [[ "${RUNPOD_CPU_PREPARE_MODE}" == "forward" ]]; then
    # No training dataset or selection is read; nothing is uploaded for the selection.
    DATA_ROOT=""
fi
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" && "${RUNPOD_CPU_PREPARE_MODE}" == "dataset" ]]; then
    bash "${SCRIPT_DIR}/runpod_s3_project.sh" s3 cp \
        "${RUNPOD_SELECTION_FILE}" \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" \
        --only-show-errors
    LOCAL_SELECTION_FILE_SHA256="$(python3 -c \
        'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "${RUNPOD_SELECTION_FILE}")"
    REMOTE_SELECTION_FILE_SHA256="$(bash "${SCRIPT_DIR}/runpod_s3_project.sh" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" - \
        --only-show-errors \
        | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
    if [[ "${REMOTE_SELECTION_FILE_SHA256}" != "${LOCAL_SELECTION_FILE_SHA256}" ]]; then
        echo "Uploaded training selection failed its S3 integrity check" >&2
        exit 3
    fi
fi

HF_TOKEN_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_HF_SECRET_NAME} }}"
if [[ "${RUNPOD_CPU_WORKFLOW}" != "prepare" || "${RUNPOD_CPU_PREPARE_MODE}" == "forward" ]]; then
    HF_TOKEN_REFERENCE=""
fi
EODHD_TOKEN_REFERENCE=""
if [[ ( "${FIN_TS_DATASET_PROFILE}" == *eodhd* && "${RUNPOD_CPU_WORKFLOW}" == "prepare" \
        && "${RUNPOD_CPU_PREPARE_MODE}" == "dataset" ) \
    || "${RUNPOD_CPU_WORKFLOW}" == "research" ]]; then
    EODHD_TOKEN_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_EODHD_SECRET_NAME} }}"
fi
TPEX_PROXY_TOKEN_REFERENCE=""
TPEX_PROXY_POD_URL=""
if [[ ${REQUIRE_TPEX_RELAY} -eq 1 ]]; then
    TPEX_PROXY_TOKEN_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_TPEX_PROXY_SECRET_NAME} }}"
    TPEX_PROXY_POD_URL="${TPEX_PROXY_URL}"
fi
POD_ENV_JSON="$(printf \
    '{"NETWORK_VOLUME_ROOT":"%s","RUNPOD_VOLUME_ROOT":"%s","RUNPOD_EXPECTED_VOLUME_ID":"%s","RUNPOD_REQUESTED_CPU_COUNT":"%s","PROJECT_ROOT":"%s","DATA_ROOT":"%s","RUNPOD_CONFIG":"%s","RUNPOD_STAGE":"%s","RUNPOD_STAGE_CONFIG_SHA256":"%s","RUNPOD_SELECTION_ID":"%s","RUNPOD_SELECTION_SHA256":"%s","RUNPOD_DATASET_REQUEST_SHA256":"%s","RUNPOD_DATASET_REVISION":"%s","RUNPOD_REMOTE_SELECTION_PATH":"%s","RUNPOD_ROLE":"%s","RUNPOD_CPU_MAX_RUNTIME_SECONDS":"%s","RUNPOD_CPU_PREPARE_RESERVE_SECONDS":"%s","RUNPOD_CPU_MAX_API_CALLS":"%s","RUNPOD_CPU_EODHD_QPS":"%s","RUNPOD_CPU_TAIWAN_QPS":"%s","RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS":"%s","RUNPOD_SHUTDOWN_ACTION":"terminate","RUNPOD_IMAGE":"%s","RUNPOD_EXPECTED_TORCH_VERSION":"%s","RUNPOD_EXPECTED_CUDA_PREFIX":"%s","RUNPOD_EXPECTED_UBUNTU_VERSION":"%s","HF_TOKEN":"%s","EODHD_API_TOKEN":"%s","TPEX_PROXY_URL":"%s","TPEX_PROXY_TOKEN":"%s","FIN_TS_DATASET_PROFILE":"%s","FIN_TS_H_START":"%s","FIN_TS_FEATURE_MODE":"%s","STAGE1_US_SYMBOLS":"%s","STAGE1_US_ETF_SYMBOLS":"%s","STAGE1_SYMBOL_LIMIT":"%s","STAGE1_DATA_START":"%s","STAGE1_DATA_END":"%s","STAGE1_TEST_END":"%s","RUNPOD_CPU_PREPARE_MODE":"%s","RUNPOD_FORWARD_END":"%s","RUNPOD_CPU_WORKFLOW":"%s","RUNPOD_LABEL_VARIANT":"%s","RUNPOD_FEATURE_SPEC":"%s","RUNPOD_CHIP_DATA_ROOT":"%s","RUNPOD_WALK_FORWARD_SCHEDULE":"%s","RUNPOD_CHIP_DATA_PREPARE":"%s","RUNPOD_LGBM_EXTEND_LAUNCH":"%s"}' \
    "${RUNPOD_VOLUME_MOUNT_PATH}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    "${RUNPOD_NETWORK_VOLUME_ID}" "${RUNPOD_CPU_VCPU_COUNT}" \
    "${PROJECT_ROOT}" "${DATA_ROOT}" "${RUNPOD_CONFIG}" "${RUNPOD_STAGE}" \
    "${RUNPOD_STAGE_CONFIG_SHA256}" "${RUNPOD_SELECTION_ID}" \
    "${RUNPOD_SELECTION_SHA256}" "${RUNPOD_DATASET_REQUEST_SHA256}" \
    "${RUNPOD_DATASET_REVISION}" "${REMOTE_SELECTION_MOUNT_PATH}" \
    "${RUNPOD_CPU_ROLE}" "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" \
    "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" "${RUNPOD_CPU_MAX_API_CALLS}" \
    "${RUNPOD_CPU_EODHD_QPS}" "${RUNPOD_CPU_TAIWAN_QPS}" \
    "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}" \
    "${RUNPOD_IMAGE}" "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    "${RUNPOD_EXPECTED_CUDA_PREFIX}" "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    "${HF_TOKEN_REFERENCE}" "${EODHD_TOKEN_REFERENCE}" \
    "${TPEX_PROXY_POD_URL}" "${TPEX_PROXY_TOKEN_REFERENCE}" "${FIN_TS_DATASET_PROFILE}" \
    "${FIN_TS_H_START}" "${FIN_TS_FEATURE_MODE}" \
    "${STAGE1_US_SYMBOLS}" "${STAGE1_US_ETF_SYMBOLS}" "${STAGE1_SYMBOL_LIMIT}" \
    "${STAGE1_DATA_START}" "${STAGE1_DATA_END}" "${STAGE1_TEST_END}" \
    "${RUNPOD_CPU_PREPARE_MODE}" "${RUNPOD_FORWARD_END}" "${RUNPOD_CPU_WORKFLOW}" \
    "${RUNPOD_LABEL_VARIANT}" "${RUNPOD_FEATURE_SPEC}" "${RUNPOD_CHIP_DATA_ROOT}" \
    "${RUNPOD_WALK_FORWARD_SCHEDULE}" "${RUNPOD_CHIP_DATA_PREPARE}" \
    "${RUNPOD_LGBM_EXTEND_LAUNCH}")"

POD_CREATE_JSON="$(
    POD_ENV_JSON="${POD_ENV_JSON}" \
    RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE}" \
    RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB}" \
    RUNPOD_CPU_FLAVOR_ID="${RUNPOD_CPU_FLAVOR_ID}" \
    RUNPOD_CPU_VCPU_COUNT="${RUNPOD_CPU_VCPU_COUNT}" \
    RUNPOD_DATACENTER_ID="${RUNPOD_DATACENTER_ID}" \
    RUNPOD_IMAGE="${RUNPOD_IMAGE}" \
    RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID}" \
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME}" \
    RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH}" \
    python3 -c '
import json
import os

payload = {
    "cloudType": os.environ["RUNPOD_CLOUD_TYPE"],
    "computeType": "CPU",
    "containerDiskInGb": int(os.environ["RUNPOD_CONTAINER_DISK_GB"]),
    "cpuFlavorIds": [os.environ["RUNPOD_CPU_FLAVOR_ID"]],
    "cpuFlavorPriority": "custom",
    "dataCenterIds": [os.environ["RUNPOD_DATACENTER_ID"]],
    "env": json.loads(os.environ["POD_ENV_JSON"]),
    "imageName": os.environ["RUNPOD_IMAGE"],
    "name": os.environ["RUNPOD_POD_NAME"],
    "networkVolumeId": os.environ["RUNPOD_NETWORK_VOLUME_ID"],
    "vcpuCount": int(os.environ["RUNPOD_CPU_VCPU_COUNT"]),
    "volumeMountPath": os.environ["RUNPOD_VOLUME_MOUNT_PATH"],
}
print(json.dumps(payload, separators=(",", ":")))
'
)"

if [[ "${RUNPOD_CREATE_DRY_RUN:-0}" == "1" || "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
    printf 'DRY RUN: POST %s/pods --name %q --compute-type cpu --cloud-type %q --image-name %q --cpu-flavor-id %q --vcpu-count %q --data-center-ids %q --container-disk-in-gb %q --network-volume-id %q --volume-mount-path %q --env %q\n' \
        "${RUNPOD_API_BASE_URL%/}" "${RUNPOD_POD_NAME}" \
        "${RUNPOD_CLOUD_TYPE}" "${RUNPOD_IMAGE}" \
        "${RUNPOD_CPU_FLAVOR_ID}" "${RUNPOD_CPU_VCPU_COUNT}" \
        "${RUNPOD_DATACENTER_ID}" "${RUNPOD_CONTAINER_DISK_GB}" \
        "${RUNPOD_NETWORK_VOLUME_ID}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
        "${POD_ENV_JSON}"
    exit 0
fi
runpod_load_project_env_key "${LOCAL_PROJECT_ROOT}" RUNPOD_API_KEY required
if [[ ! -r "${RUNPOD_GUARD_LAUNCHER}" ]]; then
    echo "External guard launcher is unavailable: ${RUNPOD_GUARD_LAUNCHER}" >&2
    exit 127
fi

python3 "${SCRIPT_DIR}/runpod_concurrency.py" admit --mode exclusive
CREATE_BODY="$(printf '%s' "${POD_CREATE_JSON}" \
    | python3 "${SCRIPT_DIR}/runpod_rest_v2_control.py" pod create-cpu)"
POD_ID="$(printf '%s' "${CREATE_BODY}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin).get("id", ""))')"
if [[ ! "${POD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Unable to parse a safe CPU Pod ID; no external guard was armed" >&2
    exit 3
fi

mkdir -p "${RUNPOD_GUARD_LOG_DIR}"
GUARD_LOG="${RUNPOD_GUARD_LOG_DIR}/${POD_ID}.log"
GUARD_PID="$(RUNPOD_GUARD_VOLUME_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}" \
    bash "${RUNPOD_GUARD_LAUNCHER}" \
    "${POD_ID}" "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" \
    "${RUNPOD_CPU_GUARD_LIFECYCLE_KEY}" \
    "${GUARD_LOG}")"

printf 'Created %s Pod: %s\n' "${RUNPOD_CPU_POD_LABEL}" "${POD_ID}"
printf 'External guard PID: %s\n' "${GUARD_PID}"
printf 'External guard log: %s\n' "${GUARD_LOG}"
printf 'External guard readiness: %s\n' "${GUARD_LOG%.log}.ready.json"
printf 'After SSH login, run: bash scripts/runpod_tmux_launch.sh %s\n' \
    "${RUNPOD_CPU_TMUX_WORKFLOW}"
