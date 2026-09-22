#!/usr/bin/env bash
# Authorized bounded verification only: no production training or market-data calls.
set -Eeuo pipefail
umask 077
main() {
# Parse the full body before executing; never hot-edit a running shell workflow.
REGRESSION_ONLY=0
if [[ "${1:-}" == "--regression-only" && $# -eq 1 ]]; then
    REGRESSION_ONLY=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: runpod_verify_full_workflow.sh [--regression-only]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
[[ "${RUNPOD_ROLE:-}" == "gpu-baseline" ]] || exit 2
[[ "${MAX_RUNTIME_SECONDS:?}" -le 2700 ]] || { echo "Verification requires a workload limit of at most 45 minutes" >&2; exit 2; }
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh"
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="${NETWORK_VOLUME_ROOT}/cache/huggingface"
OUTPUT="${NETWORK_VOLUME_ROOT}/diagnostics/full-workflow/${WANDB_RUN_ID:?}"
mkdir -p "${OUTPUT}"
# Short, Pod-local scratch avoids AF_UNIX path limits and repeated fixture-copy
# traffic on the network volume. Persistent evidence stays in OUTPUT.
export TMPDIR="$(mktemp -d /tmp/stock-forecasting-qa.XXXXXXXX)"
# No completion manifest is written under baselines/. This is not a baseline build.
if [[ "${REGRESSION_ONLY}" -eq 1 ]]; then
    set +e
    timeout --kill-after=15s 120 .venv/bin/ruff check . >"${OUTPUT}/ruff.log" 2>&1
    lint_exit=$?
    test_exit=125
    if [[ "${lint_exit}" -eq 0 ]]; then
        # SIGINT lets pytest finalize JUnit and the durable journal on timeout.
        timeout --signal=INT --kill-after=30s 2400 \
            .venv/bin/python scripts/verify_cloud_regression.py --output "${OUTPUT}" \
            >"${OUTPUT}/pytest.log" 2>&1
        test_exit=$?
    fi
    set -e
    printf '{"mode":"regression-only","pytest":%s,"ruff":%s}\n' \
        "${test_exit}" "${lint_exit}" >"${OUTPUT}/regression-status.json"
    tail -n 75 "${OUTPUT}/pytest.log" 2>/dev/null || true
    tail -n 20 "${OUTPUT}/ruff.log"
    printf 'Regression complete; awaiting control-host review within the existing hard deadline.\n'
    while sleep 10; do :; done
fi
set +e
timeout --kill-after=15s 1100 .venv/bin/python scripts/verify_baseline_throughput.py \
    --output "${OUTPUT}/baseline-throughput" >"${OUTPUT}/baseline-throughput.log" 2>&1
throughput_exit=$?
timeout --kill-after=15s 1200 .venv/bin/python -m pytest tests \
    -o faulthandler_timeout=120 --junitxml="${OUTPUT}/pytest.xml" >"${OUTPUT}/pytest.log" 2>&1
test_exit=$?
timeout --kill-after=15s 120 .venv/bin/ruff check . >"${OUTPUT}/ruff.log" 2>&1
lint_exit=$?
timeout --kill-after=15s 420 .venv/bin/python scripts/verify_kronos_full_workflow.py \
    >"${OUTPUT}/kronos-smoke.log" 2>&1
kronos_exit=$?
timeout --kill-after=15s 300 .venv/bin/python scripts/verify_full_evaluation_capacity.py \
    --output "${OUTPUT}/capacity" >"${OUTPUT}/capacity.log" 2>&1
capacity_exit=$?
timeout --kill-after=15s 420 .venv/bin/python scripts/verify_lazy_evaluation.py \
    >"${OUTPUT}/lazy-evaluation.log" 2>&1
coverage_exit=$?
set -e
printf '{"pytest":%s,"ruff":%s,"kronos":%s,"capacity":%s,"coverage":%s,"baseline_throughput":%s}\n' \
    "${test_exit}" "${lint_exit}" "${kronos_exit}" "${capacity_exit}" "${coverage_exit}" "${throughput_exit}" \
    >"${OUTPUT}/acceptance-status.json"
tail -n 65 "${OUTPUT}/pytest.log"
tail -n 35 "${OUTPUT}/ruff.log"
tail -n 30 "${OUTPUT}/kronos-smoke.log"
tail -n 30 "${OUTPUT}/capacity.log"
tail -n 20 "${OUTPUT}/lazy-evaluation.log"
tail -n 10 "${OUTPUT}/baseline-throughput.log"
printf 'Bounded verification: pytest=%s ruff=%s output=%s\n' "${test_exit}" "${lint_exit}" "${OUTPUT}"
# Keep the authorized debugging window bounded by tmux timeout and the local
# hard-limit guard. The control host publishes the terminal lifecycle after QA.
printf 'Awaiting control-host QA completion; the hard deadline remains armed.\n'
while sleep 10; do :; done
}

main "$@"
