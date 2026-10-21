#!/usr/bin/env bash

# Download the rank-label and feature stores of one finished cpu-features job.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-features [LAUNCH_ID]"
    echo "Omit LAUNCH_ID for the latest finished build recorded in lifecycle/research/features.json."
    echo "Stores are verified by SHA-256 and saved under artifacts/features/<dataset>/<store>/<identity>/;"
    echo "the manifest and logs go to artifacts/features/launches/<launch-id>/."
    echo "RUNPOD_FEATURE_DOWNLOAD_WORKERS=1..4 bounds parallel transfers (default 2)."
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
exec python3 "${SCRIPT_DIR}/download_runpod_features.py" --project-root "${PROJECT_ROOT}" "$@"
