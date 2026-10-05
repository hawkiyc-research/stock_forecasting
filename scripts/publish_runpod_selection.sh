#!/usr/bin/env bash

# Publish only the immutable selection. Source files and prepared data are untouched.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
source "${SCRIPT_DIR}/lib/runpod_selection.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
runpod_load_active_selection "${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" --code-only
bash "${SCRIPT_DIR}/runpod_s3_project.sh" s3 cp \
    "${RUNPOD_SELECTION_FILE}" \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" \
    --only-show-errors
