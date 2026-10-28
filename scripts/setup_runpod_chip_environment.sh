#!/usr/bin/env bash

# Create (or reuse) the tw_chip_data virtual environment on the network volume.
# tw_chip_data locks pandas 3 and pyarrow 26, which the project's Poetry environment (pandas 2)
# cannot host, so it gets its own venv keyed by the digest of its pinned requirements file.
# The chosen interpreter path is written to <tools root>/python-path for the chip job.
set -Eeuo pipefail
umask 077

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
CHIP_PROJECT_ROOT="${RUNPOD_CHIP_PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/tw_chip_data}"
CHIP_TOOLS_ROOT="${RUNPOD_CHIP_TOOLS_ROOT:-${NETWORK_VOLUME_ROOT}/tools/tw-chip-data}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
REQUIREMENTS="${CHIP_PROJECT_ROOT}/requirements-runpod.txt"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${NETWORK_VOLUME_ROOT}/cache/pip}"

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "tw_chip_data requirements are missing: ${REQUIREMENTS} (upload-chip-data --include-source)" >&2
    exit 2
fi
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "Python interpreter is unavailable: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi
digest="$(sha256sum "${REQUIREMENTS}" | cut -c1-16)"
VENV="${CHIP_TOOLS_ROOT}/venv-${digest}"
mkdir -p "${CHIP_TOOLS_ROOT}" "${PIP_CACHE_DIR}"
if [[ ! -f "${VENV}/.ready" ]]; then
    printf '[%s] installing tw_chip_data environment %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${VENV}"
    "${RUNPOD_PYTHON_BIN}" -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install --no-input --disable-pip-version-check --upgrade pip
    "${VENV}/bin/python" -m pip install --no-input --disable-pip-version-check \
        --requirement "${REQUIREMENTS}"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${VENV}/.ready"
fi
"${VENV}/bin/python" -c "import pandas, pyarrow, requests, yaml" \
    || { echo "tw_chip_data environment is incomplete: ${VENV}" >&2; exit 3; }
printf '%s\n' "${VENV}/bin/python" > "${CHIP_TOOLS_ROOT}/python-path"
printf 'tw_chip_data environment ready: %s\n' "${VENV}/bin/python"
