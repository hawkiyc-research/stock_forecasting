#!/usr/bin/env bash

# Add artifacts/forward/END/shared-cache to the network volume's shared provider cache.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh upload-forward-cache YYYY-MM-DD"
    echo "Uploads artifacts/forward/YYYY-MM-DD/shared-cache (written by forward_pipeline"
    echo "export-cache) to /runpod-volume/datasets/forward-taiwan/api-cache after re-hashing"
    echo "every file. Entries already on the volume are kept; a differing entry stops the upload."
}

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    usage >&2
    exit 2
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
exec python3 "${SCRIPT_DIR}/upload_runpod_forward_cache.py" "$@"
