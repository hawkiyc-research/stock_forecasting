#!/usr/bin/env bash

# Inventory or delete experiment weights on the RunPod network volume (cleanup side task 2).
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh volume-weights inventory [--output JSON]"
    echo "       bash scripts/runpod_workflow.sh volume-weights delete --inventory JSON --category weights|exports [--apply] [--workers N]"
    echo "inventory lists checkpoint weights, baseline model files and per-sample prediction"
    echo "exports with sizes and writes their exact keys to artifacts/audit/volume-weight-inventory-*.json."
    echo "delete removes only one category of that inventory; without --apply it only counts."
    echo "Records (metrics, manifests, leaderboards, evaluation arrays, reports) are never deleted."
}

if [[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --help || "$1" == -h ) ) ]]; then
    usage
    [[ $# -eq 1 ]] && exit 0
    exit 2
fi
case "$1" in
    inventory|delete) ;;
    *)
        usage >&2
        exit 2
        ;;
esac

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
export RUNPOD_NETWORK_VOLUME_ID
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
exec python3 "${SCRIPT_DIR}/runpod_volume_weights.py" "$@"
