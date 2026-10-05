#!/usr/bin/env python3
"""Run bounded remote checkpoint verification for a local Pod guard."""

import argparse
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-wrapper", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-request-sha256", required=True)
    parser.add_argument("--created-after", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.timeout_seconds < 1 or arguments.timeout_seconds > 300:
        parser.error("--timeout-seconds must be between 1 and 300")
    preflight = Path(__file__).with_name("runpod_remote_checkpoint_preflight.py")
    command = [
        sys.executable,
        str(preflight),
        "--s3-wrapper", str(arguments.s3_wrapper),
        "--bucket", arguments.bucket,
        "--run-id", arguments.run_id,
        "--config", str(arguments.config),
        "--dataset-request-sha256", arguments.dataset_request_sha256,
        "--created-after", arguments.created_after,
        "--selection-policy", "latest",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        output, errors = process.communicate(timeout=arguments.timeout_seconds)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        print("Remote checkpoint verification timed out", file=sys.stderr)
        return 3
    if process.returncode:
        print(errors.strip()[-1000:] or "Remote checkpoint verification failed", file=sys.stderr)
        return 3
    print(output.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
