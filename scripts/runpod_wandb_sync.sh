#!/usr/bin/env bash

# Sync preserved offline or incomplete W&B transactions from the network volume.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID1_ENV_HELPER="${SCRIPT_DIR}/runpod_reexec_with_pid1_env.py"
PID1_IMPORT_PYTHON="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
if [[ "${RUNPOD_SSH_ENV_IMPORTED:-0}" != "1" ]]; then
    exec "${PID1_IMPORT_PYTHON}" "${PID1_ENV_HELPER}" -- bash "${BASH_SOURCE[0]}" "$@"
fi

terminate_sync_pod() {
    local sync_exit_code=$?
    trap - EXIT
    bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    exit "${sync_exit_code}"
}

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ "${RUNPOD_ROLE:-}" != "gpu-train" && "${RUNPOD_ROLE:-}" != "gpu-validation" ]]; then
    echo "W&B sync must run in a GPU Pod that has the W&B RunPod Secret" >&2
    exit 2
fi
if [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "Persistent project Python is unavailable: ${PROJECT_PYTHON}" >&2
    exit 127
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
if [[ "${RUNPOD_SCOPED_LIFECYCLE:-0}" == 1 ]]; then
    # Never sync another experiment's still-open transactions on a shared volume.
    if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "${WANDB_RUN_ID:-}" ) ]]; then
        echo "W&B sync is restricted to this Pod's run ID" >&2
        exit 2
    fi
    set -- "${WANDB_RUN_ID:?A run-scoped Pod requires WANDB_RUN_ID}"
    source "${SCRIPT_DIR}/lib/runpod_paths.sh"
    unset RUNPOD_GPU_WORKFLOW_LEASE_HELD
    runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
fi
# Only an admitted sync task owns this Pod. Invalid arguments or a busy training
# lease must never terminate the experiment that is already using it.
trap terminate_sync_pod EXIT
set +e
"${PROJECT_PYTHON}" -m stock_forecasting.cli.sync_wandb "$@"
SYNC_EXIT_CODE=$?
set -e
exit "${SYNC_EXIT_CODE}"
