#!/usr/bin/env bash

# Download the latest completed signal-prediction export of each named checkpoint.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-signal-predictions SELECTOR [SELECTOR ...]"
    echo "       bash scripts/runpod_workflow.sh download-signal-predictions --baseline-arrays"
    echo "       bash scripts/runpod_workflow.sh download-signal-predictions --evaluation-data [--list]"
    echo "Pass 1-16 selectors: a signal child run ID (the -alpha/-direction runs, not a parent"
    echo "ID), or RUN_ID/checkpoint-NNNNNN when that run has exports of several checkpoints."
    echo "Each selected latest completed export is verified by SHA-256 and saved under"
    echo "artifacts/diagnostics/signal-predictions/<run>/<checkpoint>/<export>/."
    echo "--baseline-arrays fetches the current baselines' evaluation arrays (about 1.3 GB),"
    echo "verified against their manifests, into artifacts/diagnostics/signal-decomposition/."
    echo "--evaluation-data fetches the baselines' shared validation/test bar store and its"
    echo "cleaned sample index for local inference into artifacts/diagnostics/evaluation-data/;"
    echo "every file is verified by SHA-256 and verified files are kept on a rerun. --list only"
    echo "prints the dataset, the file count and the size."
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
if [[ "$1" == "--evaluation-data" ]]; then
    if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--list" ) ]]; then
        usage >&2
        exit 2
    fi
elif [[ "$1" != "--baseline-arrays" || $# -ne 1 ]]; then
    for selector in "$@"; do
        if [[ ! "${selector}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}(/checkpoint-[0-9]{6,})?$ \
            || "${selector}" == *--* ]]; then
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
