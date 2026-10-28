#!/usr/bin/env bash

# Upload the tw_chip_data curated layer and its manifests to the RunPod network volume
# (protocol 13.18): the chip feature specs read them on the CPU Pods. With --include-raw the
# raw layer travels once as a single tar archive (the Pod's chip job extracts it and then only
# fetches new sessions); with --include-source the tw_chip_data code, source registry and
# pinned requirements land in /runpod-volume/tw_chip_data for that job (protocol 13.20).
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
REMOTE_DIR="${RUNPOD_REMOTE_CHIP_DATA_DIR:-datasets/tw-chip-data}"
REMOTE_SOURCE_DIR="${RUNPOD_REMOTE_CHIP_SOURCE_DIR:-tw_chip_data}"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
# shellcheck source=lib/runpod_s3_retry.sh
source "${SCRIPT_DIR}/lib/runpod_s3_retry.sh"
runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh upload-chip-data <tw_chip_data>/data [--include-raw] [--include-source] [--apply]" >&2
    echo "Uploads curated/<dataset>/year=*/{part-0.parquet,_SUCCESS.json} and manifests/*.json" >&2
    echo "to s3://<volume>/${REMOTE_DIR}/ (/runpod-volume/${REMOTE_DIR} on the Pod)." >&2
    echo "--include-raw adds raw/ as one archive (${REMOTE_DIR}/raw-archive/raw.tar);" >&2
    echo "--include-source adds the tw_chip_data code to ${REMOTE_SOURCE_DIR}/." >&2
    echo "Without --apply it only lists what would be uploaded." >&2
}
if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi
LOCAL_DATA_ROOT="$1"
shift
MODE=dry-run
INCLUDE_RAW=0
INCLUDE_SOURCE=0
for option in "$@"; do
    case "${option}" in
        --apply) MODE=apply ;;
        --dry-run) MODE=dry-run ;;
        --include-raw) INCLUDE_RAW=1 ;;
        --include-source) INCLUDE_SOURCE=1 ;;
        *) usage; exit 2 ;;
    esac
done
if [[ ! -d "${LOCAL_DATA_ROOT}/curated" || ! -d "${LOCAL_DATA_ROOT}/manifests" ]]; then
    echo "${LOCAL_DATA_ROOT} must contain curated/ and manifests/" >&2
    exit 2
fi
LOCAL_PROJECT_DIR="$(cd "${LOCAL_DATA_ROOT}/.." && pwd)"
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID contains invalid characters" >&2
    exit 2
fi
if [[ ! "${REMOTE_DIR}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
    echo "RUNPOD_REMOTE_CHIP_DATA_DIR contains invalid characters" >&2
    exit 2
fi
if [[ ! "${REMOTE_SOURCE_DIR}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
    echo "RUNPOD_REMOTE_CHIP_SOURCE_DIR contains invalid characters" >&2
    exit 2
fi
if [[ ${INCLUDE_RAW} -eq 1 && ! -d "${LOCAL_DATA_ROOT}/raw" ]]; then
    echo "${LOCAL_DATA_ROOT}/raw is missing; nothing to archive" >&2
    exit 2
fi
# The source upload needs the files the Pod job reads: the package, the registry and the
# requirements exported from uv.lock (uv export --no-dev --no-hashes --no-emit-project).
SOURCE_FILES=()
if [[ ${INCLUDE_SOURCE} -eq 1 ]]; then
    for required in src/tw_chip_data/cli.py configs/sources.yaml requirements-runpod.txt \
        pyproject.toml uv.lock; do
        if [[ ! -f "${LOCAL_PROJECT_DIR}/${required}" ]]; then
            echo "tw_chip_data source file is missing: ${LOCAL_PROJECT_DIR}/${required}" >&2
            exit 2
        fi
    done
    while IFS= read -r source_file; do
        SOURCE_FILES+=("${source_file#"${LOCAL_PROJECT_DIR}/"}")
    done < <(find "${LOCAL_PROJECT_DIR}/src/tw_chip_data" -name '*.py' -type f | sort)
    SOURCE_FILES+=(configs/sources.yaml requirements-runpod.txt pyproject.toml uv.lock)
    for extra in README.md DECISION_SUMMARY.md docs/INTEGRATION.md; do
        if [[ -f "${LOCAL_PROJECT_DIR}/${extra}" ]]; then
            SOURCE_FILES+=("${extra}")
        fi
    done
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
if [[ ${INCLUDE_RAW} -eq 1 ]]; then
    raw_kib="$(du -sk "${LOCAL_DATA_ROOT}/raw" | cut -f1)"
    printf 'Raw archive: raw/ (%s KiB on disk) -> s3://%s/%s/raw-archive/raw.tar\n' \
        "${raw_kib}" "${RUNPOD_NETWORK_VOLUME_ID}" "${REMOTE_DIR}"
fi
if [[ ${INCLUDE_SOURCE} -eq 1 ]]; then
    printf 'Source: %d files from %s -> s3://%s/%s/\n' \
        "${#SOURCE_FILES[@]}" "${LOCAL_PROJECT_DIR}" "${RUNPOD_NETWORK_VOLUME_ID}" \
        "${REMOTE_SOURCE_DIR}"
fi
if [[ "${MODE}" == dry-run ]]; then
    printf 'Dry-run only; use --apply to upload.\n'
    for relative in "${RELATIVE_PATHS[@]}"; do
        printf '  %s\n' "${relative}"
    done
    for relative in "${SOURCE_FILES[@]}"; do
        printf '  [source] %s\n' "${relative}"
    done
    exit 0
fi

runpod_s3_retry_command "${S3_WRAPPER}" "verify network volume access" \
    s3api head-bucket --bucket "${RUNPOD_NETWORK_VOLUME_ID}" >/dev/null
upload_and_verify() {
    local local_path="$1" key="$2"
    runpod_s3_retry_command "${S3_WRAPPER}" "upload ${key}" s3 cp \
        "${local_path}" "s3://${RUNPOD_NETWORK_VOLUME_ID}/${key}" --only-show-errors
    local local_size remote_size
    local_size="$(wc -c < "${local_path}" | tr -d ' ')"
    remote_size="$(runpod_s3_retry_capture \
        "${S3_WRAPPER}" "verify uploaded size for ${key}" \
        s3api head-object --bucket "${RUNPOD_NETWORK_VOLUME_ID}" --key "${key}" \
        --query ContentLength --output text)"
    if [[ "${remote_size}" != "${local_size}" ]]; then
        echo "Remote size verification failed for: ${key}" >&2
        exit 4
    fi
}
index=0
for relative in "${RELATIVE_PATHS[@]}"; do
    index=$((index + 1))
    printf '[%d/%d] Uploading %s\n' "${index}" "${#RELATIVE_PATHS[@]}" "${relative}"
    upload_and_verify "${LOCAL_DATA_ROOT}/${relative}" "${REMOTE_DIR}/${relative}"
done
if [[ ${INCLUDE_SOURCE} -eq 1 ]]; then
    index=0
    for relative in "${SOURCE_FILES[@]}"; do
        index=$((index + 1))
        printf '[source %d/%d] Uploading %s\n' "${index}" "${#SOURCE_FILES[@]}" "${relative}"
        upload_and_verify "${LOCAL_PROJECT_DIR}/${relative}" "${REMOTE_SOURCE_DIR}/${relative}"
    done
fi
if [[ ${INCLUDE_RAW} -eq 1 ]]; then
    ARCHIVE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/chip-raw-archive.XXXXXX")"
    trap 'rm -f "${ARCHIVE_DIR}/raw.tar" "${ARCHIVE_DIR}/raw.tar.sha256"; rmdir "${ARCHIVE_DIR}" 2>/dev/null || true' EXIT
    printf 'Archiving %s/raw ...\n' "${LOCAL_DATA_ROOT}"
    tar -cf "${ARCHIVE_DIR}/raw.tar" -C "${LOCAL_DATA_ROOT}" raw
    (cd "${ARCHIVE_DIR}" && shasum -a 256 raw.tar > raw.tar.sha256)
    printf 'Uploading the raw archive (%s bytes)\n' \
        "$(wc -c < "${ARCHIVE_DIR}/raw.tar" | tr -d ' ')"
    upload_and_verify "${ARCHIVE_DIR}/raw.tar" "${REMOTE_DIR}/raw-archive/raw.tar"
    upload_and_verify "${ARCHIVE_DIR}/raw.tar.sha256" "${REMOTE_DIR}/raw-archive/raw.tar.sha256"
fi
printf 'Chip data upload complete: /runpod-volume/%s\n' "${REMOTE_DIR}"
