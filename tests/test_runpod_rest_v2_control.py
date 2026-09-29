"""Verify Runpod REST v2 control request and compatibility contracts."""

from __future__ import annotations

import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


CONTROL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/runpod_rest_v2_control.py")
)


class RestV2ControlTests(unittest.TestCase):
    def test_fallback_shutdown_uses_v2_action_and_delete(self) -> None:
        for action, method, suffix in (
            ("stop", "POST", "/action"),
            ("terminate", "DELETE", ""),
        ):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bin_dir = root / "bin"
                bin_dir.mkdir()
                capture = root / "curl-arguments"
                curl = bin_dir / "curl"
                curl.write_text(
                    "#!/usr/bin/env bash\n"
                    "printf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n"
                    "cat >/dev/null\n"
                    "if [[ \"${RUNPOD_SHUTDOWN_ACTION}\" == terminate ]]; then "
                    "printf '204'; else printf '200'; fi\n",
                    encoding="utf-8",
                )
                curl.chmod(0o755)
                volume = root / "volume"
                volume.mkdir()
                result = subprocess.run(
                    ["bash", str(Path(__file__).resolve().parents[1]
                                 / "scripts/stop_runpod_pod.sh")],
                    env={
                        **os.environ,
                        "PATH": f"{bin_dir}:{os.environ['PATH']}",
                        "CAPTURE_PATH": str(capture),
                        "NETWORK_VOLUME_ROOT": str(volume),
                        "RUNPOD_API_KEY": "test-only",
                        "RUNPOD_POD_ID": "pod_fixture",
                        "RUNPOD_SHUTDOWN_ACTION": action,
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = capture.read_text(encoding="utf-8").splitlines()
                self.assertEqual(arguments[arguments.index("--request") + 1], method)
                self.assertEqual(
                    arguments[arguments.index("--url") + 1],
                    f"https://api.runpod.io/v2/pods/pod_fixture{suffix}",
                )
                if action == "stop":
                    self.assertEqual(arguments[arguments.index("--data") + 1],
                                     '{"action":"stop"}')
                marker = json.loads((volume / "logs/bootstrap/shutdown.json").read_text())
                self.assertTrue(marker["success"])

    def test_pod_list_uses_bounded_cursor_and_preserves_volume_identity(self) -> None:
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if "cursor=" not in path:
                return {
                    "pods": [{
                        "id": "pod_one", "status": "RUNNING", "runtime": {"uptime": 12},
                        "mounts": {"network": [
                            {"volumeId": "volume_one", "path": "/runpod-volume"}
                        ]},
                    }],
                    "pagination": {"hasNextPage": True, "nextCursor": "page/2"},
                }
            return {
                "pods": [{"id": "pod_two", "status": "EXITED", "runtime": None}],
                "pagination": {"hasNextPage": False, "nextCursor": None},
            }

        with patch.dict(CONTROL["_pod_list"].__globals__, {"_request": fake_request}):
            pods = CONTROL["_pod_list"]()
        self.assertEqual([pod["id"] for pod in pods], ["pod_one", "pod_two"])
        self.assertEqual(pods[0]["runtimeStatus"], "running")
        self.assertEqual(pods[0]["networkVolumeId"], "volume_one")
        self.assertEqual(pods[0]["uptimeSeconds"], 12)
        self.assertEqual(calls[1][1], "/pods?limit=50&cursor=page%2F2")

    def test_cpu_create_translates_existing_request_to_v2(self) -> None:
        captured = {}

        def fake_request(method, path, **kwargs):
            captured.update(method=method, path=path, **kwargs)
            return {"id": "cpu_pod", "status": "PROVISIONING", "runtime": None}

        source = {
            "name": "cpu-prepare", "imageName": "runpod/pytorch:example",
            "cloudType": "SECURE", "containerDiskInGb": 30,
            "cpuFlavorIds": ["cpu3g"], "vcpuCount": 8,
            "dataCenterIds": ["EU-RO-1"], "env": {"RUNPOD_ROLE": "cpu-prepare"},
            "networkVolumeId": "volume_one", "volumeMountPath": "/runpod-volume",
        }
        with patch.dict(CONTROL["_create_cpu_pod"].__globals__, {"_request": fake_request}), \
                patch.object(sys, "stdin", io.StringIO(json.dumps(source))):
            created = CONTROL["_create_cpu_pod"]()
        self.assertEqual(captured["path"], "/pods")
        self.assertEqual(captured["expected_status"], 201)
        self.assertEqual(captured["body"]["cpu"], {"id": "cpu3g", "vcpuCount": 8})
        self.assertEqual(captured["body"]["mounts"], {
            "network": [{"volumeId": "volume_one", "path": "/runpod-volume"}]
        })
        self.assertEqual(captured["body"]["ports"], ["22/tcp"])
        self.assertTrue(captured["body"]["startSsh"])
        self.assertEqual(created["id"], "cpu_pod")

    def test_cpu_v1_only_count_is_rejected_by_v2_adapter(self) -> None:
        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "cpuFlavorIds": ["cpu3g"], "vcpuCount": 3, "dataCenterIds": ["EU-RO-1"],
            "networkVolumeId": "volume_one", "volumeMountPath": "/runpod-volume",
        }))):
            with self.assertRaises(CONTROL["ApiError"]) as caught:
                CONTROL["_create_cpu_pod"]()
        self.assertEqual(caught.exception.code, "usage_error")

    def test_http_404_keeps_not_found_code_for_orphan_reconciliation(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.runpod.io/v2/pods/missing", 404, "Not Found", {},
            io.BytesIO(b'{"title":"Pod not found"}'),
        )
        with patch.dict("os.environ", {"RUNPOD_API_KEY": "test-only"}), \
                patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(CONTROL["ApiError"]) as caught:
                CONTROL["_request"]("GET", "/pods/missing")
        self.assertEqual((caught.exception.status, caught.exception.code), (404, "not_found"))


if __name__ == "__main__":
    unittest.main()
