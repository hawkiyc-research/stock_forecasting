#!/usr/bin/env bash

# Verify source and data readiness through the network-volume S3 API before renting compute.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
CODE_MARKER_KEY="lifecycle/stage1/code.json"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"

if [[ $# -ne 1 ]]; then
    echo "Usage: verify_runpod_stage_readiness.sh --code-only|--gpu|--baseline" >&2
    exit 2
fi
case "$1" in
    --code-only) MODE=code-only ;;
    --gpu) MODE=gpu ;;
    --baseline) MODE=baseline ;;
    *)
        echo "Usage: verify_runpod_stage_readiness.sh --code-only|--gpu|--baseline" >&2
        exit 2
        ;;
esac
if [[ "${MODE}" == "baseline" ]]; then
    runpod_load_active_selection "${LOCAL_PROJECT_ROOT}" baseline
else
    runpod_load_active_selection "${LOCAL_PROJECT_ROOT}"
fi

if [[ ! "${RUNPOD_NETWORK_VOLUME_ID:-}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required" >&2
    exit 2
fi
if [[ ! -f "${S3_WRAPPER}" || ! -r "${S3_WRAPPER}" || ! -f "${READINESS_HELPER}" ]]; then
    echo "RunPod S3 or readiness helper is unavailable" >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for readiness verification" >&2
    exit 127
fi
if [[ "${MODE}" != "baseline" && ( ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
    || "${RUNPOD_CONFIG}" == /* \
    || "/${RUNPOD_CONFIG}/" == *"/../"* \
    || "/${RUNPOD_CONFIG}/" == *"/./"* \
    || "${RUNPOD_CONFIG}" == *"//"* \
    || ! -f "${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}" ) ]]; then
    echo "RUNPOD_CONFIG must be an existing safe path relative to the local project" >&2
    exit 2
fi

CODE_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CODE_MARKER_KEY}" - \
    --only-show-errors)"
printf '%s\n' "${CODE_JSON}" \
    | python3 "${READINESS_HELPER}" check-code \
        --marker - \
        --project-root "${LOCAL_PROJECT_ROOT}"

if [[ "${MODE}" == "code-only" ]]; then
    printf 'CPU preparation gate passed: uploaded code is ready.\n'
    exit 0
fi

if [[ "${MODE}" == "baseline" ]]; then
    REMOTE_SELECTION_JSON="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" - \
        --only-show-errors)"
    printf '%s\n' "${REMOTE_SELECTION_JSON}" \
        | python3 "${SCRIPT_DIR}/runpod_selection.py" verify-selection-copy --data-only \
            --project-root "${LOCAL_PROJECT_ROOT}" --selection "${RUNPOD_SELECTION_FILE}" --candidate -
    python3 "${SCRIPT_DIR}/runpod_baseline_readiness.py" --project-root "${LOCAL_PROJECT_ROOT}" \
        --selection "${RUNPOD_SELECTION_FILE}"
    printf 'Baseline gate passed: selected full dataset is ready; no main-model cache is required.\n'
    exit 0
fi

REMOTE_SELECTION_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" - \
    --only-show-errors)"
printf '%s\n' "${REMOTE_SELECTION_JSON}" \
    | python3 "${SCRIPT_DIR}/runpod_selection.py" verify-selection-copy \
        --project-root "${LOCAL_PROJECT_ROOT}" \
        --selection "${RUNPOD_SELECTION_FILE}" \
        --candidate -
python3 "${SCRIPT_DIR}/runpod_training_readiness.py" \
    --project-root "${LOCAL_PROJECT_ROOT}" --selection "${RUNPOD_SELECTION_FILE}"

printf '%s\n' \
    'GPU artifact gate passed: code, dataset, and offline model cache are ready; checking GPU workflow availability next.'
