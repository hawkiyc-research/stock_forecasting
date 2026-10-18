"""Dependency-free checks that the local relay verifier matches the deployed relay source."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify_tpex_cloud_run_response.py"
RELAY = (ROOT / "cloudrun/tpex-relay/src/relay.mjs").read_text(encoding="utf-8")


def relay_version() -> str:
    match = re.search(r'^const RELAY_VERSION = "([0-9]+)";$', RELAY, re.MULTILINE)
    assert match is not None
    return match.group(1)


def verify(payload: dict, *, status: int = 200, mode: str = "warmup"):
    body = json.dumps(payload).encode() + f"\n__FIN_TS_HTTP_STATUS__:{status}\n".encode()
    return subprocess.run(
        [sys.executable, str(VERIFIER), "--mode", mode, "--label", "fixture",
         "--expected-region", "asia-east1"],
        input=body, capture_output=True, check=False, timeout=10,
    )


class RelayVerifierTests(unittest.TestCase):
    def test_warmup_accepts_exactly_the_source_relay_version(self) -> None:
        current = relay_version()
        accepted = verify({"ready": True, "version": current, "region": "asia-east1"})
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        stale = verify({"ready": True, "version": str(int(current) - 1), "region": "asia-east1"})
        self.assertEqual(stale.returncode, 2, stale.stdout)
        other_region = verify({"ready": True, "version": current, "region": "us-central1"})
        self.assertEqual(other_region.returncode, 2, other_region.stdout)

    def test_official_routes_require_a_table(self) -> None:
        table = {"fields": ["a"], "data": [[1]]}
        self.assertEqual(verify(table, mode="official").returncode, 0)
        self.assertEqual(verify({"tables": []}, mode="official").returncode, 0)
        self.assertEqual(verify({"stat": "ok"}, mode="official").returncode, 2)
        self.assertEqual(verify({"error": "x"}, status=503, mode="official").returncode, 75)


class DotenvWriterTests(unittest.TestCase):
    WRITERS = (
        "configure_runpod_credentials.sh", "configure_tpex_cloud_run_relay.sh",
        "deploy_runpod_network_volume.sh", "deploy_tpex_cloud_run_relay.sh",
    )

    def test_writers_update_the_dotenv_that_readers_resolve(self) -> None:
        for name in self.WRITERS:
            line = next(
                line for line in (ROOT / "scripts" / name).read_text().splitlines()
                if line.startswith("ENV_FILE=")
            )
            for override in ("/elsewhere/shared.env", ""):
                with self.subTest(script=name, override=override):
                    result = subprocess.run(
                        ["bash", "-c",
                         f'source "$1"; PROJECT_ROOT=/project; {line}; '
                         'printf "%s\\n%s" "$ENV_FILE" "$(runpod_project_env_file /project)"',
                         "writer", str(ROOT / "scripts/lib/runpod_project_env.sh")],
                        env={"PATH": "/usr/bin:/bin", "RUNPOD_ENV_FILE": override},
                        capture_output=True, text=True, check=False, timeout=10,
                    )
                    written, read = result.stdout.splitlines()
                    self.assertEqual(written, read)
                    self.assertEqual(written, override or "/project/.env")


if __name__ == "__main__":
    unittest.main()
