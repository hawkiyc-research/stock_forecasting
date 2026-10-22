#!/usr/bin/env bash

# Download the results of one finished xs-train job (cross-stock experiment).
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-xs [LAUNCH_ID]"
    echo "Omit LAUNCH_ID for the latest finished job recorded in lifecycle/research/xs-train.json."
    echo "Results are verified by SHA-256 and saved under artifacts/xs/<launch-id>/."
}

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 ]] || [[ $# -eq 1 && ! "$1" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    usage >&2
    exit 2
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
export RUNPOD_NETWORK_VOLUME_ID
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
exec python3 "${SCRIPT_DIR}/download_runpod_lgbm.py" --project-root "${PROJECT_ROOT}" \
    --workflow xs "$@"
