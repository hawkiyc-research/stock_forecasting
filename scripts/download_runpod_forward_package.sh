#!/usr/bin/env bash

# Download the sealed forward input package /runpod-volume/forward/END/package.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-forward-package YYYY-MM-DD"
    echo "Downloads /runpod-volume/forward/YYYY-MM-DD/package (built by cpu prepare --forward)"
    echo "to artifacts/forward/YYYY-MM-DD/package, checking every file's SHA-256. An identical"
    echo "local package is reported as downloaded; a different one is never replaced."
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
exec python3 "${SCRIPT_DIR}/download_runpod_forward_package.py" "$@"
