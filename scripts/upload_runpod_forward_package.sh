#!/usr/bin/env bash

# Upload the sealed forward input package artifacts/forward/END/package to the network volume.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh upload-forward-package YYYY-MM-DD"
    echo "Uploads artifacts/forward/YYYY-MM-DD/package (built by forward_pipeline build) to"
    echo "/runpod-volume/forward/YYYY-MM-DD/package after re-hashing every file. An identical"
    echo "package already on the volume is reported as uploaded; a different one is refused."
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
exec python3 "${SCRIPT_DIR}/upload_runpod_forward_package.py" "$@"
