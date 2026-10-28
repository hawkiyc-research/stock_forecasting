#!/usr/bin/env bash

# Update the tw_chip_data layers on the network volume from the official sources
# (protocols 13.18 and 13.20): fetch the sessions since the last curated date, re-curate every
# dataset, derive the calendar, eligibility and event tables, write the manifests and run the
# validation suite. TPEx is the only source that refuses requests from outside Taiwan (RunPod
# probe of 2026-10-28), so www.tpex.org.tw goes through the Cloud Run relay the dataset
# preparation already uses; every other host is fetched directly. Runs on its own cpu-prep Pod
# (workflow `chip`) or chained after the dataset preparation by runpod_tmux_launch.sh.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
RUNPOD_ROLE="${RUNPOD_ROLE:-cpu-prep}"
CHIP_PROJECT_ROOT="${RUNPOD_CHIP_PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/tw_chip_data}"
CHIP_DATA_ROOT="${RUNPOD_CHIP_DATA_ROOT:-${NETWORK_VOLUME_ROOT}/datasets/tw-chip-data}"
CHIP_TOOLS_ROOT="${RUNPOD_CHIP_TOOLS_ROOT:-${NETWORK_VOLUME_ROOT}/tools/tw-chip-data}"
# Sessions before the last curated date are fetched again so late corrections (the T+2
# day-trade finals, next-day margin adjustments) and recent no_data placeholders are refreshed.
CHIP_REFETCH_DAYS="${RUNPOD_CHIP_REFETCH_DAYS:-10}"
# The exclusive end of the market data the selection asked for; the chip layers stop the day
# before it so both pipelines cover the same sessions. Empty: yesterday in Taipei.
CHIP_END_EXCLUSIVE="${STAGE1_DATA_END:-}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
CHIP_LOG="${RUNPOD_TMUX_LOG_FILE:-${LOG_ROOT}/cpu-chip/${LAUNCH_ID}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
# One marker per volume, as for the other research jobs; the guard checks its Pod ID.
CHIP_MARKER="${LIFECYCLE_ROOT}/research/chip-data.json"
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "The chip-data update must run inside a RunPod CPU Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" ]]; then
    echo "The chip-data update requires the cpu-prep role (it carries the TPEx relay secret)" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
if [[ ! "${LAUNCH_ID}" =~ ^launch-[A-Za-z0-9_-]{1,150}$ ]]; then
    echo "RUNPOD_LAUNCH_ID contains unsupported characters" >&2
    exit 2
fi
for variable in CHIP_PROJECT_ROOT CHIP_DATA_ROOT CHIP_TOOLS_ROOT; do
    runpod_validate_path_in_root "${!variable}" "${NETWORK_VOLUME_ROOT}" \
        "${variable}" NETWORK_VOLUME_ROOT
done
if [[ ! "${CHIP_REFETCH_DAYS}" =~ ^[0-9]{1,3}$ ]]; then
    echo "RUNPOD_CHIP_REFETCH_DAYS must be an integer number of days" >&2
    exit 2
fi
if [[ -n "${CHIP_END_EXCLUSIVE}" && ! "${CHIP_END_EXCLUSIVE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "STAGE1_DATA_END must use YYYY-MM-DD" >&2
    exit 2
fi
for required in "src/tw_chip_data/cli.py" "configs/sources.yaml" "requirements-runpod.txt"; do
    if [[ ! -f "${CHIP_PROJECT_ROOT}/${required}" ]]; then
        echo "tw_chip_data source is missing on the volume: ${CHIP_PROJECT_ROOT}/${required}" >&2
        echo "Upload it with: bash scripts/runpod_workflow.sh upload-chip-data <tw_chip_data>/data --include-source --apply" >&2
        exit 2
    fi
done
# The relay settings arrive as the Pod's TPEx acquisition environment (cpu-prep Pods only).
if [[ ! "${TPEX_PROXY_URL:-}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
    echo "TPEX_PROXY_URL is missing or is not an approved run.app origin; TPEx needs the relay" >&2
    exit 2
fi
if [[ -z "${TPEX_PROXY_TOKEN:-}" ]]; then
    echo "TPEX_PROXY_TOKEN RunPod Secret is missing or was not resolved" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

mkdir -p "$(dirname "${CHIP_LOG}")" "$(dirname "${CHIP_MARKER}")" "${CHIP_DATA_ROOT}"
CHIP_SUCCEEDED=0

write_chip_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${CHIP_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind research-chip-data
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${CHIP_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_cpu_chip() {
    local chip_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${CHIP_SUCCEEDED} -ne 1 ]]; then
        write_chip_state failed "${chip_exit_code}" || true
    fi
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${chip_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_cpu_chip EXIT

write_chip_state preparing
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" quarantine-stale-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"

bash "${SCRIPT_DIR}/setup_runpod_chip_environment.sh"
CHIP_PYTHON="$(cat "${CHIP_TOOLS_ROOT}/python-path")"
runpod_validate_path_in_root "${CHIP_PYTHON}" "${CHIP_TOOLS_ROOT}" CHIP_PYTHON CHIP_TOOLS_ROOT

# A one-time raw archive (upload-chip-data --include-raw) seeds the raw layer so the first run
# fetches only the new sessions instead of 2012 onwards at one request every three seconds.
RAW_ARCHIVE="${CHIP_DATA_ROOT}/raw-archive/raw.tar"
if [[ ! -d "${CHIP_DATA_ROOT}/raw" && -f "${RAW_ARCHIVE}" ]]; then
    printf '[%s] extracting the raw archive %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RAW_ARCHIVE}"
    if [[ -f "${RAW_ARCHIVE}.sha256" ]]; then
        expected="$(cut -d' ' -f1 "${RAW_ARCHIVE}.sha256")"
        actual="$(sha256sum "${RAW_ARCHIVE}" | cut -d' ' -f1)"
        if [[ "${expected}" != "${actual}" ]]; then
            echo "Raw archive digest mismatch: ${RAW_ARCHIVE}" >&2
            exit 4
        fi
    fi
    tar -xf "${RAW_ARCHIVE}" -C "${CHIP_DATA_ROOT}"
fi

export TWCHIP_ROOT="${CHIP_PROJECT_ROOT}"
export TWCHIP_DATA_DIR="${CHIP_DATA_ROOT}"
export TWCHIP_CONFIG="${CHIP_PROJECT_ROOT}/configs/sources.yaml"
export TWCHIP_RELAY_ORIGIN="${TPEX_PROXY_URL%/}"
export TWCHIP_RELAY_TOKEN="${TPEX_PROXY_TOKEN}"
export TWCHIP_RELAY_HOSTS="${RUNPOD_CHIP_RELAY_HOSTS:-www.tpex.org.tw}"
export PYTHONPATH="${CHIP_PROJECT_ROOT}/src"
export PYTHONUNBUFFERED=1
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
mkdir -p "${TMPDIR}"
cd "${CHIP_PROJECT_ROOT}"

chip() {
    "${CHIP_PYTHON}" -c 'import sys; from tw_chip_data.cli import main; sys.exit(main(sys.argv[1:]))' "$@"
}

# Fetch window: from the earliest "last curated date" across the daily datasets minus the
# refetch margin (the project start when nothing is curated yet) to the day before the data end.
read -r FETCH_START FETCH_END < <("${CHIP_PYTHON}" - "${CHIP_DATA_ROOT}" "${CHIP_REFETCH_DAYS}" "${CHIP_END_EXCLUSIVE}" <<'PY'
import datetime as dt, json, sys
from pathlib import Path
from zoneinfo import ZoneInfo
from tw_chip_data.config import load_registry

data_root, refetch_days, end_exclusive = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
registry = load_registry()
daily = {
    name for name, info in registry.datasets.items()
    if any(registry.tables[t].kind == "daily" for t in info["tables"] if t in registry.tables)
}
last = []
for name in sorted(daily):
    manifest = data_root / "manifests" / f"{name}.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8")).get("date_max")
        if value:
            last.append(dt.date.fromisoformat(str(value)[:10]))
start = (min(last) - dt.timedelta(days=refetch_days)) if last else registry.project_start
if end_exclusive:
    end = dt.date.fromisoformat(end_exclusive) - dt.timedelta(days=1)
else:
    end = dt.datetime.now(ZoneInfo("Asia/Taipei")).date() - dt.timedelta(days=1)
if start > end:
    start = end
print(start.isoformat(), end.isoformat())
PY
)
printf '[%s] tw_chip_data fetch %s -> %s (relay %s for %s)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FETCH_START}" "${FETCH_END}" \
    "${TWCHIP_RELAY_ORIGIN}" "${TWCHIP_RELAY_HOSTS}"
chip -v fetch --table all --start "${FETCH_START}" --end "${FETCH_END}"

DATASETS="$("${CHIP_PYTHON}" -c 'from tw_chip_data.cli import DATASETS; print(" ".join(DATASETS))')"
for dataset in ${DATASETS}; do
    printf '[%s] curate %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${dataset}"
    chip curate --dataset "${dataset}"
done
printf '[%s] calendar, derived datasets and event tables\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
chip calendar
chip derive
chip events
for dataset in ${DATASETS}; do
    chip manifest --dataset "${dataset}"
done
printf '[%s] validation suite\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
chip validate --dataset all
chip status

write_chip_state ready
CHIP_SUCCEEDED=1
printf 'Chip data update complete: %s (fetched %s -> %s)\n' "${CHIP_DATA_ROOT}" "${FETCH_START}" "${FETCH_END}"
