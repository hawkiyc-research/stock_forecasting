#!/usr/bin/env bash

# Download the latest completed signal-prediction export of each named child run.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-signal-predictions RUN_ID [RUN_ID ...]"
    echo "       bash scripts/runpod_workflow.sh download-signal-predictions --baseline-arrays"
    echo "Pass 1-16 signal child run IDs (the -alpha/-direction runs, not a parent ID)."
    echo "Each run's latest completed export is verified by SHA-256 and saved under"
    echo "artifacts/diagnostics/signal-predictions/<run>/<checkpoint>/<export>/."
    echo "--baseline-arrays fetches the current baselines' evaluation arrays (about 1.3 GB),"
    echo "verified against their manifests, into artifacts/diagnostics/signal-decomposition/."
    echo "Volume, credentials and remote paths are resolved automatically."
}

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 || $# -gt 16 ]]; then
    usage >&2
    exit 2
fi
if [[ "$1" != "--baseline-arrays" || $# -ne 1 ]]; then
    for run_id in "$@"; do
        if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ || "${run_id}" == *--* ]]; then
            usage >&2
            exit 2
        fi
    done
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
exec python3 "${SCRIPT_DIR}/download_runpod_signal_predictions.py" "$@"
