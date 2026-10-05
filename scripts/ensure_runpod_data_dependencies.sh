#!/usr/bin/env bash
# Called only after the owning GPU workflow has acquired its exclusive lease.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Data dependency setup requires a RunPod workflow" >&2
    exit 2
fi
if "${PROJECT_ROOT}/.venv/bin/python" - "${PROJECT_ROOT}/configs/data_cleaning.json" <<'PY'
import json
import sys
from importlib.metadata import PackageNotFoundError, version

expected = json.load(open(sys.argv[1]))["calendar_version"]
try:
    valid = version("exchange_calendars") == expected
except PackageNotFoundError:
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
    exit 0
fi
echo "Updating the project-owned cloud dependencies; prepared market data is unchanged."
RUNPOD_ROLE=gpu-train bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
