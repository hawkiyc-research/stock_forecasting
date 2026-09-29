#!/usr/bin/env bash

# Use REST API v2 for project control; retain one GraphQL-only GPU deadline path.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -eq 0 ]]; then
    echo "Usage: runpodctl_project.sh RESOURCE ACTION [ARGUMENTS...]" >&2
    exit 2
fi

runpod_load_runpodctl_env "${LOCAL_PROJECT_ROOT}"

API_KEY_VALUE="${RUNPOD_API_KEY}"
PATH_VALUE="${PATH:-/usr/local/bin:/usr/bin:/bin}"
HOME_VALUE="${HOME:-/tmp}"
TMPDIR_VALUE="${TMPDIR:-/tmp}"

runpod_clear_exported_environment
export PATH="${PATH_VALUE}"
export HOME="${HOME_VALUE}"
export TMPDIR="${TMPDIR_VALUE}"
export RUNPOD_API_KEY="${API_KEY_VALUE}"

if [[ "$1" == pod && "${2:-}" == create ]]; then
    # REST v2 and v1 cannot set the provider-side terminateAfter deadline.
    has_deadline=0
    for argument in "$@"; do
        if [[ "${argument}" == --terminate-after || "${argument}" == --terminate-after=* ]]; then
            has_deadline=1
        fi
    done
    if [[ "${has_deadline}" != 1 ]]; then
        echo "GPU Pod creation requires the provider-side --terminate-after deadline" >&2
        exit 2
    fi
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "runpodctl is required only for GPU Pod creation with terminate-after" >&2
        exit 127
    fi
    if ! runpodctl pod create --help 2>/dev/null | grep -- '--terminate-after' >/dev/null; then
        echo "Installed runpodctl does not support the required --terminate-after safety deadline" >&2
        exit 127
    fi
    exec "$(command -v runpodctl)" "$@"
fi

exec python3 "${SCRIPT_DIR}/runpod_rest_v2_control.py" "$@"
