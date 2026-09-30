#!/usr/bin/env bash

# Use REST API v2 for all project RunPod control operations.
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

exec python3 "${SCRIPT_DIR}/runpod_rest_v2_control.py" "$@"
