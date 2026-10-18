#!/usr/bin/env bash

# Download one information-test report and its tmux log through the local control plane.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
LIFECYCLE_KEY="lifecycle/research/tw-information.json"
REMOTE_ROOT="diagnostics/tw-information-test"
SESSION_NAME="stock-forecasting-cpu-research"
LOCAL_ROOT="${PROJECT_ROOT}/artifacts/diagnostics/tw-information-test"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-research [LAUNCH_ID]"
    echo "Omit LAUNCH_ID for the latest finished run recorded in ${LIFECYCLE_KEY}."
    echo "Files are saved to artifacts/diagnostics/tw-information-test/<launch-id>/."
}

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 ]] || [[ $# -eq 1 && ! "$1" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    usage >&2
    exit 2
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

s3_get() {
    bash "${S3_WRAPPER}" s3 cp "s3://${RUNPOD_NETWORK_VOLUME_ID}/$1" "$2" --only-show-errors
}

LAUNCH_ID="${1:-}"
LIFECYCLE_JSON=""
if [[ -z "${LAUNCH_ID}" ]]; then
    LIFECYCLE_JSON="$(s3_get "${LIFECYCLE_KEY}" -)"
    LIFECYCLE_FIELDS="$(printf '%s' "${LIFECYCLE_JSON}" | python3 -c '
import json
import re
import sys

payload = json.load(sys.stdin)
launch_id = str(payload.get("launch_id", ""))
state = str(payload.get("state", ""))
if payload.get("kind") != "research-tw-information" \
        or re.fullmatch(r"launch-[A-Za-z0-9_-]{1,150}", launch_id) is None:
    raise SystemExit("The research lifecycle marker is invalid")
print(launch_id, state)
')"
    LAUNCH_ID="${LIFECYCLE_FIELDS%% *}"
    LIFECYCLE_STATE="${LIFECYCLE_FIELDS#* }"
    if [[ "${LIFECYCLE_STATE}" == "preparing" ]]; then
        echo "Research launch ${LAUNCH_ID} is still running; try again after it finishes" >&2
        exit 3
    fi
    printf 'Latest research launch: %s (state=%s)\n' "${LAUNCH_ID}" "${LIFECYCLE_STATE}"
fi

LOCAL_DIR="${LOCAL_ROOT}/${LAUNCH_ID}"
mkdir -p "${LOCAL_DIR}/logs"
if [[ -n "${LIFECYCLE_JSON}" ]]; then
    printf '%s\n' "${LIFECYCLE_JSON}" > "${LOCAL_DIR}/lifecycle.json"
fi
# Five small files: sequential requests cost less than a bounded worker pool here.
# Logs and status exist for every launch; the report exists only after the test finished.
for name in combined.log status.json; do
    s3_get "logs/tmux/${SESSION_NAME}/${LAUNCH_ID}/${name}" "${LOCAL_DIR}/logs/${name}" \
        || printf 'Not found on the volume: %s\n' "${name}" >&2
done
REPORT_DOWNLOADED=1
for name in report.json summary.md auxiliary-manifest.json; do
    if ! s3_get "${REMOTE_ROOT}/${LAUNCH_ID}/${name}" "${LOCAL_DIR}/${name}"; then
        printf 'Not found on the volume: %s\n' "${name}" >&2
        REPORT_DOWNLOADED=0
    fi
done

printf 'Saved to: %s\n' "${LOCAL_DIR}"
if [[ ${REPORT_DOWNLOADED} -ne 1 ]]; then
    echo "The report is incomplete; read logs/combined.log for the failure" >&2
    exit 4
fi
printf 'Summary: %s\n' "${LOCAL_DIR}/summary.md"
