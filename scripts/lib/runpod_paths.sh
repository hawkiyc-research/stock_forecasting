#!/usr/bin/env bash

# Validate untrusted RunPod volume paths lexically before any filesystem access.

runpod_validate_absolute_path() {
    local path_value="${1:-}"
    local path_label="${2:-PATH}"

    if [[ -z "${path_value}" || "${path_value}" != /* ]]; then
        echo "${path_label} must be an absolute path" >&2
        return 2
    fi
    if [[ ! "${path_value}" =~ ^/[A-Za-z0-9._/-]*$ ]]; then
        echo "${path_label} contains unsupported characters" >&2
        return 2
    fi
    if [[ "${path_value}" == *"//"* ]]; then
        echo "${path_label} must not contain //" >&2
        return 2
    fi
    if [[ "${path_value}/" == *"/./"* ]]; then
        echo "${path_label} must not contain a dot path component" >&2
        return 2
    fi
    if [[ "${path_value}/" == *"/../"* ]]; then
        echo "${path_label} must not contain a parent path component" >&2
        return 2
    fi
    if [[ "${path_value}" != "/" && "${path_value}" == */ ]]; then
        echo "${path_label} must not end with a slash" >&2
        return 2
    fi
}

runpod_validate_path_in_root() {
    local path_value="${1:-}"
    local root_value="${2:-}"
    local path_label="${3:-PATH}"
    local root_label="${4:-NETWORK_VOLUME_ROOT}"

    runpod_validate_absolute_path "${root_value}" "${root_label}" || return
    runpod_validate_absolute_path "${path_value}" "${path_label}" || return

    if [[ "${root_value}" == "/" ]]; then
        return 0
    fi
    case "${path_value}" in
        "${root_value}"|"${root_value}"/*) return 0 ;;
        *)
            echo "${path_label} must be inside ${root_label} (${root_value})" >&2
            return 2
            ;;
    esac
}

runpod_gpu_lifecycle_key() {
    local phase="$1" run_id="${2:-${WANDB_RUN_ID:-}}"
    case "${phase}" in training|validation|baseline) ;; *) return 2 ;; esac
    if [[ "${RUNPOD_SCOPED_LIFECYCLE:-0}" == 1 && "${phase}" != baseline ]]; then
        if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ || "${run_id}" == *--* ]]; then
            echo "Run-scoped lifecycle requires a canonical run ID" >&2
            return 2
        fi
        printf 'lifecycle/runs/%s/%s.json\n' "${run_id}" "${phase}"
    else
        printf 'lifecycle/stage1/%s.json\n' "${phase}"
    fi
}

runpod_guard_lifecycle_kind() {
    local key="$1" run_id="${2:-}"
    case "${key}" in
        lifecycle/stage1/cpu-preparation.json) echo stage1-cpu-preparation ;;
        lifecycle/stage1/mixed-finalization.json) echo stage1-mixed-finalization ;;
        lifecycle/stage1/training.json) echo stage1-training ;;
        lifecycle/stage1/validation.json) echo stage1-validation ;;
        lifecycle/stage1/baseline.json) echo stage1-baseline ;;
        "lifecycle/runs/${run_id}/training.json"|"lifecycle/runs/${run_id}/validation.json")
            [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ && "${run_id}" != *--* ]] || return 2
            echo "stage1-$(basename "${key}" .json)"
            ;;
        "") echo "" ;;
        *) echo "Unsupported lifecycle marker key" >&2; return 2 ;;
    esac
}

runpod_acquire_gpu_workflow_lease() {
    local volume_root="${1:-}"
    local lease_root lease_path

    if [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == "1" ]]; then
        return 0
    fi
    runpod_validate_absolute_path "${volume_root}" NETWORK_VOLUME_ROOT || return
    lease_root="${volume_root}/lifecycle/stage1"
    lease_path="${lease_root}/gpu-workflow.lock"
    runpod_validate_path_in_root \
        "${lease_path}" "${volume_root}" GPU_WORKFLOW_LEASE NETWORK_VOLUME_ROOT || return
    if ! command -v flock >/dev/null 2>&1; then
        if [[ "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
            export RUNPOD_GPU_WORKFLOW_LEASE_HELD=1
            return 0
        fi
        echo "flock is required for the shared GPU workflow lease" >&2
        return 127
    fi
    mkdir -p "${lease_root}"
    exec 9>"${lease_path}"
    if [[ "${RUNPOD_SCOPED_LIFECYCLE:-0}" == 1 \
        && ( "${RUNPOD_ROLE:-}" == gpu-train || "${RUNPOD_ROLE:-}" == gpu-validation ) ]]; then
        local key
        key="$(runpod_gpu_lifecycle_key training)" || return
        mkdir -p "${volume_root}/$(dirname "${key}")"
        exec 8>"${volume_root}/$(dirname "${key}")/gpu-workflow.lock"
        if ! flock -n 8; then
            echo "This run already has a training or validation owner" >&2
            return 75
        fi
        if ! flock -s -n 9; then
            echo "An exclusive volume workflow is active" >&2
            return 75
        fi
        export RUNPOD_GPU_WORKFLOW_LEASE_MODE=shared
    elif ! flock -n 9; then
        echo "Another training or validation workflow already holds the GPU workflow lease" >&2
        return 75
    fi
    export RUNPOD_GPU_WORKFLOW_LEASE_HELD=1
}
