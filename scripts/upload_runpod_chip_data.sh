#!/usr/bin/env bash

# Upload the tw_chip_data curated layer and its manifests to the RunPod network volume
# (protocol 13.18): the chip feature specs read them on the CPU Pods.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
REMOTE_DIR="${RUNPOD_REMOTE_CHIP_DATA_DIR:-datasets/tw-chip-data}"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
# shellcheck source=lib/runpod_s3_retry.sh
source "${SCRIPT_DIR}/lib/runpod_s3_retry.sh"
runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh upload-chip-data <tw_chip_data>/data [--apply]" >&2
    echo "Uploads curated/<dataset>/year=*/{part-0.parquet,_SUCCESS.json} and manifests/*.json" >&2
    echo "to s3://<volume>/${REMOTE_DIR}/ (/runpod-volume/${REMOTE_DIR} on the Pod)." >&2
    echo "Without --apply it only lists what would be uploaded." >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi
LOCAL_DATA_ROOT="$1"
MODE=dry-run
if [[ $# -eq 2 ]]; then
    case "$2" in
        --apply) MODE=apply ;;
        --dry-run) MODE=dry-run ;;
        *) usage; exit 2 ;;
    esac
fi
if [[ ! -d "${LOCAL_DATA_ROOT}/curated" || ! -d "${LOCAL_DATA_ROOT}/manifests" ]]; then
    echo "${LOCAL_DATA_ROOT} must contain curated/ and manifests/" >&2
    exit 2
fi
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID contains invalid characters" >&2
    exit 2
fi
if [[ ! "${REMOTE_DIR}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
    echo "RUNPOD_REMOTE_CHIP_DATA_DIR contains invalid characters" >&2
    exit 2
fi

# Only finished partitions (those with a _SUCCESS.json marker) and the dataset manifests.
RELATIVE_PATHS=()
while IFS= read -r marker; do
    folder="$(dirname "${marker}")"
    if [[ -f "${folder}/part-0.parquet" ]]; then
        RELATIVE_PATHS+=("${folder#"${LOCAL_DATA_ROOT}/"}/part-0.parquet")
        RELATIVE_PATHS+=("${folder#"${LOCAL_DATA_ROOT}/"}/_SUCCESS.json")
    fi
done < <(find "${LOCAL_DATA_ROOT}/curated" -name _SUCCESS.json -type f | sort)
while IFS= read -r manifest; do
    RELATIVE_PATHS+=("${manifest#"${LOCAL_DATA_ROOT}/"}")
done < <(find "${LOCAL_DATA_ROOT}/manifests" -maxdepth 1 -name '*.json' -type f | sort)
if [[ ${#RELATIVE_PATHS[@]} -eq 0 ]]; then
    echo "No finished partitions found under ${LOCAL_DATA_ROOT}/curated" >&2
    exit 2
fi
TOTAL_BYTES=0
for relative in "${RELATIVE_PATHS[@]}"; do
    size="$(wc -c < "${LOCAL_DATA_ROOT}/${relative}" | tr -d ' ')"
    TOTAL_BYTES=$((TOTAL_BYTES + size))
done
printf 'Chip data upload: %d files, %d bytes -> s3://%s/%s/\n' \
    "${#RELATIVE_PATHS[@]}" "${TOTAL_BYTES}" "${RUNPOD_NETWORK_VOLUME_ID}" "${REMOTE_DIR}"
if [[ "${MODE}" == dry-run ]]; then
    printf 'Dry-run only; use --apply to upload.\n'
    for relative in "${RELATIVE_PATHS[@]}"; do
        printf '  %s\n' "${relative}"
    done
    exit 0
fi

runpod_s3_retry_command "${S3_WRAPPER}" "verify network volume access" \
    s3api head-bucket --bucket "${RUNPOD_NETWORK_VOLUME_ID}" >/dev/null
index=0
for relative in "${RELATIVE_PATHS[@]}"; do
    index=$((index + 1))
    printf '[%d/%d] Uploading %s\n' "${index}" "${#RELATIVE_PATHS[@]}" "${relative}"
    runpod_s3_retry_command "${S3_WRAPPER}" "upload ${relative}" s3 cp \
        "${LOCAL_DATA_ROOT}/${relative}" \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${REMOTE_DIR}/${relative}" \
        --only-show-errors
done
for relative in "${RELATIVE_PATHS[@]}"; do
    local_size="$(wc -c < "${LOCAL_DATA_ROOT}/${relative}" | tr -d ' ')"
    remote_size="$(runpod_s3_retry_capture \
        "${S3_WRAPPER}" "verify uploaded size for ${relative}" \
        s3api head-object \
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
        --key "${REMOTE_DIR}/${relative}" \
        --query ContentLength \
        --output text)"
    if [[ "${remote_size}" != "${local_size}" ]]; then
        echo "Remote size verification failed for: ${relative}" >&2
        exit 4
    fi
done
printf 'Chip data upload complete: /runpod-volume/%s\n' "${REMOTE_DIR}"
