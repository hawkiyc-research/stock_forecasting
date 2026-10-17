#!/usr/bin/env bash

# Launch a detached signal-prediction export, or run its internal tmux worker.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

usage() {
    cat <<'EOF'
Usage (inside an existing CUDA RunPod Pod):
  bash scripts/runpod_tmux_launch.sh export-signal-predictions --checkpoint RUN_ID [OPTIONS]

Options:
  --checkpoint RUN_ID       Signal child run ID (alpha or direction); its validation-selected
                            best checkpoint is exported. Required; there is no latest default.
  --splits SPLIT [SPLIT]    validation and/or test (default: validation test)
  --confirm-test            Required whenever test is exported

The resolved checkpoint config is used, never the active stage selection.
Results: NETWORK_VOLUME_ROOT/diagnostics/signal-predictions/<run>/<checkpoint>/export-*/
  <split>/predictions.npy   float32 alpha points [N,14] or direction logits [N,14,3]
  <split>/split.json        Shape, SHA-256, membership digest and recomputed metrics
  export.json, status.json, export.log
This command starts detached tmux session stock-forecasting-export-predictions; SSH may
disconnect afterward. It does not create a Pod. It shares the diagnostic lifecycle of
probe-scales: the local guard terminates the Pod after success/failure/timeout.
Logs: NETWORK_VOLUME_ROOT/logs/tmux/stock-forecasting-export-predictions/<launch-id>/
No training, W&B run, checkpoint, baseline or validation lifecycle marker is changed.
EOF
}

for argument in "$@"; do
    if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
        usage
        exit 0
    fi
done

# SSH sessions may omit the Pod environment. Reuse the existing allowlisted
# importer, without executing Python on the local macOS control machine.
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" \
    && "${RUNPOD_SSH_ENV_IMPORTED:-0}" != "1" \
    && -r /proc/1/environ ]]; then
    if [[ ! -x /usr/local/bin/python ]]; then
        echo "RunPod PID 1 environment importer is unavailable" >&2
        exit 127
    fi
    exec /usr/local/bin/python "${SCRIPT_DIR}/runpod_reexec_with_pid1_env.py" -- \
        bash "${BASH_SOURCE[0]}" "$@"
fi

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "Prediction export must run inside an existing CUDA RunPod Pod; no local model execution" >&2
    exit 2
fi

# Inspect the selector without changing argument order; argparse validates the rest.
checkpoint_selector=""
checkpoint_count=0
read_checkpoint_selector() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint|--checkpoint=*)
                checkpoint_count=$((checkpoint_count + 1))
                if [[ "$1" == --checkpoint=* ]]; then
                    checkpoint_selector="${1#--checkpoint=}"
                else
                    [[ $# -ge 2 ]] || { echo "--checkpoint requires a RUN_ID" >&2; exit 2; }
                    checkpoint_selector="$2"
                    shift
                fi
                ;;
        esac
        shift
    done
}
read_checkpoint_selector "$@"
if [[ ${checkpoint_count} -ne 1 ]]; then
    echo "Pass --checkpoint RUN_ID exactly once" >&2
    exit 2
fi
if [[ ! "${checkpoint_selector}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
    || "${checkpoint_selector}" == *--* ]]; then
    echo "--checkpoint must be a safe RUN_ID, not a path; use --help" >&2
    exit 2
fi

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${NETWORK_VOLUME_ROOT}/savedModel}"
case "${NETWORK_VOLUME_ROOT}" in
    /|/workspace|/workspace/*)
        echo "Prediction export requires a persistent network volume, not / or /workspace" >&2
        exit 2
        ;;
esac
for path_name in PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
if [[ "${SAVED_MODEL_ROOT}" != "${NETWORK_VOLUME_ROOT}/savedModel" ]]; then
    echo "SAVED_MODEL_ROOT must equal NETWORK_VOLUME_ROOT/savedModel" >&2
    exit 2
fi
export NETWORK_VOLUME_ROOT PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"

# Verify the exact mount and share the existing exclusive GPU lease, but do not run
# training/validation readiness transitions or rewrite their lifecycle markers.
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
# The launcher marks every gpu-probe diagnostic worker with this shared flag.
if [[ "${RUNPOD_SCALE_PROBE_TMUX_WORKER:-0}" != "1" ]]; then
    exec bash "${SCRIPT_DIR}/runpod_tmux_launch.sh" export-signal-predictions "$@"
fi
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
bash "${SCRIPT_DIR}/ensure_runpod_data_dependencies.sh"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    echo "Persistent project .venv is unavailable; use the existing cloud setup workflow" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
exec "${PROJECT_ROOT}/.venv/bin/python" -m stock_forecasting.cli.export_signal_predictions "$@"
