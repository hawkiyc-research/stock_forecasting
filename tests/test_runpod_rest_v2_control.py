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
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

CONTROL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/runpod_rest_v2_control.py")
)


class ErrorResponseTests(unittest.TestCase):
    def fail_request(self, raw, *, status=400, body=None, path="/pods", method="POST"):
        stream = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode())
        error = urllib.error.HTTPError("https://api.runpod.io/v2/pods", status, "Error", {}, stream)
        with (
            patch.dict(os.environ, {"RUNPOD_API_KEY": "fixture-private-api-key"}),
            patch("urllib.request.urlopen", side_effect=error) as request,
            self.assertRaises(CONTROL["ApiError"]) as caught,
        ):
            CONTROL["_request"](method, path, body=body)
        request.assert_called_once()
        self.assertTrue(stream.closed)
        return caught.exception.payload()

    def test_entire_json_body_including_unknown_fields_and_whitespace_is_preserved(self):
        raw = '{\n "title": "Bad Request", "detail": "Placement rejected",\n' \
              ' "errors": [{"field":"gpu.id","reason":"fixture"}], "new_field": [1,2]\n}'
        result = self.fail_request(raw)
        self.assertEqual(result["response_body"], raw)
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["path"], "/pods")
        self.assertIn("POST /pods", result["error"])
        self.assertFalse(result["response_body_truncated"])

    def test_error_body_is_not_parsed_or_summarized(self):
        with patch("json.loads", side_effect=AssertionError("Do not parse the error body")):
            result = self.fail_request('{"arbitrary_provider_format": "reason"}')
        self.assertIn("arbitrary_provider_format", result["response_body"])

    def test_plain_text_html_empty_and_malformed_json_are_preserved(self):
        for raw in ("No matching capacity\nTry again later",
                    "<html>upstream failed</html>", "", "{bad"):
            with self.subTest(raw=raw):
                self.assertEqual(self.fail_request(raw)["response_body"], raw)

    def test_error_larger_than_old_eight_kilobyte_limit_is_preserved(self):
        raw = "x" * 10000 + "\nThe actual reason at the end."
        self.assertEqual(self.fail_request(raw)["response_body"], raw)

    def test_http_status_is_authoritative_for_reconciliation_not_response_fields(self):
        for status, code in ((400, "bad_request"), (401, "unauthorized"), (403, "forbidden"),
                             (404, "not_found"), (409, "conflict"), (422, "bad_request"),
                             (429, "rate_limited"), (500, "api_error")):
            with self.subTest(status=status):
                result = self.fail_request('{"status":404,"code":"not_found"}', status=status)
                self.assertEqual((result["status"], result["code"]), (status, code))

    def test_request_credentials_environment_values_and_encoded_values_are_redacted(self):
        body = {"env": {"HF_TOKEN": "fixture-hf-private", "APP_CONFIG": "opaque-env-private"},
                "registry": {"password": 'fixture-quote-"-private'}}
        raw = json.dumps({"title": "fixture-private-api-key", "detail": body,
                          "unknown": "fixture-quote-%22-private"})
        result = self.fail_request(raw, body=body)
        output = json.dumps(result)
        for secret in ("fixture-private-api-key", "fixture-hf-private", "opaque-env-private",
                       "fixture-quote-", "%22-private"):
            self.assertNotIn(secret, output)
        self.assertIn("[REDACTED]", output)

    def test_credentials_returned_by_server_are_redacted_without_known_request_value(self):
        raw = '{"detail":"Bearer unseen-bearer", "access_token":"unseen-token",' \
              '"password":"unseen-pass", "GPU":"5090"}'
        output = self.fail_request(raw)["response_body"]
        self.assertNotIn("unseen-", output)
        self.assertIn('"GPU":"5090"', output)

    def test_request_query_values_are_not_in_diagnostic_endpoint(self):
        result = self.fail_request("Not available", path="/pods?cursor=opaque-cursor", method="GET")
        self.assertEqual(result["path"], "/pods")
        self.assertNotIn("opaque-cursor", json.dumps(result))

    def test_control_characters_are_escaped_in_cli_json(self):
        output = json.dumps(self.fail_request("problem\n\x1b[2J\x00failure"))
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x00", output)
        self.assertIn("\\u001b", output)

    def test_bounded_read_reports_truncation_and_drops_partial_final_line(self):
        raw = "Complete line\n" + "x" * CONTROL["MAX_ERROR_BYTES"]
        result = self.fail_request(raw)
        self.assertTrue(result["response_body_truncated"])
        self.assertEqual(result["response_body"], "Complete line")

    def test_non_utf8_error_body_is_losslessly_escaped(self):
        result = self.fail_request(b"failure:\xff")
        self.assertEqual(result["response_body"], "failure:\\xff")

    def test_cli_stderr_stays_one_json_document_and_exits_nonzero(self):
        raw = b'{"detail":"fixture-provider-reason","extra":{"keep":true}}'
        error = urllib.error.HTTPError(
            "https://api.runpod.io/v2/pods", 400, "Error", {}, io.BytesIO(raw)
        )
        stderr, stdout = io.StringIO(), io.StringIO()
        with (
            patch.dict(os.environ, {"RUNPOD_API_KEY": "fixture-private-api-key"}),
            patch.object(sys, "argv", ["control", "pod", "list"]),
            patch("urllib.request.urlopen", side_effect=error) as request,
            redirect_stderr(stderr), redirect_stdout(stdout),
            self.assertRaises(SystemExit) as caught,
        ):
            runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "scripts/runpod_rest_v2_control.py"),
                run_name="__main__",
            )
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        result = json.loads(stderr.getvalue())
        self.assertEqual(result["response_body"], raw.decode())
        self.assertEqual(result["method"], "GET")
        request.assert_called_once()


class GpuCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gpus = [
            {
                "id": "NVIDIA GeForce RTX 5090",
                "name": "RTX 5090",
                "memory": 32,
                "price": {"secure": 0.99, "community": 0.69, "serverless": 3.0},
                "availability": "LOW",
                "dataCenters": [
                    {"id": "EU-RO-1", "availability": "NONE"},
                    {"id": "EUR-IS-1", "availability": "LOW"},
                ],
            },
            {
                "id": "NVIDIA RTX PRO 4500 Blackwell Server Edition",
                "name": "RTX PRO 4500 SE",
                "memory": 32,
                "price": {"secure": None, "community": 0.0},
                "availability": "MEDIUM",
                "dataCenters": [
                    {"id": "EU-RO-1", "availability": "LOW"},
                ],
            },
        ]

    def invoke(self, *arguments, response=None):
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            return {"gpus": self.gpus} if response is None else response

        output = io.StringIO()
        with (
            patch.dict(CONTROL["main"].__globals__, {"_request": fake_request}),
            patch.object(sys, "argv", ["control", "gpu", "list", *arguments]),
            redirect_stdout(output),
        ):
            self.assertEqual(CONTROL["main"](), 0)
        self.assertEqual(calls, [("GET", "/catalog/gpus?include=AVAILABILITY&product=POD", {})])
        return output.getvalue()

    def test_default_table_shows_prices_ids_and_data_centers(self) -> None:
        output = self.invoke()
        for text in (
            "GPU catalog: 2 matches",
            "USD/hour",
            "Stock scope: global",
            "VRAM/GB",
            "SECURE",
            "COMMUNITY",
            "0.99",
            "0.69",
            "gpuId: NVIDIA GeForce RTX 5090",
            "EU-RO-1:NONE",
            "EUR-IS-1:LOW",
            "stock is not a capacity reservation",
        ):
            self.assertIn(text, output)
        self.assertNotIn('"serverless"', output)
        self.assertIn("--", output)
        self.assertIn("0.00", output)

    def test_json_preserves_complete_provider_entries(self) -> None:
        self.assertEqual(json.loads(self.invoke("--output", "json")), self.gpus)

    def test_search_matches_display_name_and_full_id_case_insensitively(self) -> None:
        for query in ("5090", "nvidia geforce", "rTx 5090"):
            with self.subTest(query=query):
                output = self.invoke("--search", query)
                self.assertIn("RTX 5090", output)
                self.assertNotIn("4500", output)

    def test_data_center_stock_does_not_use_global_stock(self) -> None:
        output = self.invoke("--search", "5090", "--data-center", "eu-ro-1")
        self.assertIn("Stock scope: eu-ro-1", output)
        self.assertIn("NONE", output)
        self.assertNotIn("LOW", output)
        self.assertNotIn("EUR-IS-1", output)

    def test_filters_combine_and_json_keeps_original_entry(self) -> None:
        output = self.invoke("--search", "4500", "--data-center", "EU-RO-1", "--output", "json")
        self.assertEqual(json.loads(output), [self.gpus[1]])
        output = self.invoke("--data-center", "EUR-IS-1", "--output", "json")
        self.assertEqual(json.loads(output), [self.gpus[0]])

    def test_empty_and_no_match_catalogs_are_clear(self) -> None:
        for arguments in (("--search", "no-such-gpu"), ("--data-center", "no-such-center")):
            with self.subTest(arguments=arguments):
                self.assertIn("No matching GPUs.", self.invoke(*arguments))
                self.assertEqual(json.loads(self.invoke(*arguments, "--output", "json")), [])
        self.assertIn("No matching GPUs.", self.invoke(response={"gpus": []}))

    def test_missing_optional_fields_and_unknown_stock_do_not_fail(self) -> None:
        output = self.invoke(response={"gpus": [{"id": "fixture", "name": "Fixture"}]})
        self.assertIn("gpuId: fixture", output)
        self.assertIn("-- (not reported)", output)

    def test_malformed_entries_fail_before_table_or_json_is_printed(self) -> None:
        for entry in (
            None,
            {},
            {"id": "fixture", "name": ""},
            {"id": "bad\nidentifier", "name": "Fixture"},
        ):
            for output_format in ("table", "json"):
                with (
                    self.subTest(entry=entry, output=output_format),
                    self.assertRaises(CONTROL["ApiError"]),
                ):
                    self.invoke("--output", output_format, response={"gpus": [self.gpus[0], entry]})

    def test_long_ids_remain_complete_and_locations_wrap(self) -> None:
        self.gpus[0]["id"] = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        self.gpus[0]["dataCenters"] *= 10
        output = self.invoke("--search", "6000")
        self.assertIn("gpuId: " + self.gpus[0]["id"], output)
        self.assertTrue(all(len(line) <= 96 for line in output.splitlines()))

    def test_pod_query_output_remains_json(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(
                CONTROL["main"].__globals__,
                {
                    "_request": lambda *args: {
                        "id": "pod_fixture",
                        "status": "RUNNING",
                        "runtime": {"uptime": 10},
                    }
                },
            ),
            patch.object(sys, "argv", ["control", "pod", "get", "pod_fixture"]),
            redirect_stdout(output),
        ):
            self.assertEqual(CONTROL["main"](), 0)
        self.assertEqual(json.loads(output.getvalue())["id"], "pod_fixture")


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
                    'printf \'%s\\n\' "$@" > "${CAPTURE_PATH}"\n'
                    "cat >/dev/null\n"
                    'if [[ "${RUNPOD_SHUTDOWN_ACTION}" == terminate ]]; then '
                    "printf '204'; else printf '200'; fi\n",
                    encoding="utf-8",
                )
                curl.chmod(0o755)
                volume = root / "volume"
                volume.mkdir()
                result = subprocess.run(
                    [
                        "bash",
                        str(Path(__file__).resolve().parents[1] / "scripts/stop_runpod_pod.sh"),
                    ],
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
                    self.assertEqual(arguments[arguments.index("--data") + 1], '{"action":"stop"}')
                marker = json.loads((volume / "logs/bootstrap/shutdown.json").read_text())
                self.assertTrue(marker["success"])

    def test_pod_list_uses_bounded_cursor_and_preserves_volume_identity(self) -> None:
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if "cursor=" not in path:
                return {
                    "pods": [
                        {
                            "id": "pod_one",
                            "status": "RUNNING",
                            "runtime": {"uptime": 12},
                            "mounts": {
                                "network": [{"volumeId": "volume_one", "path": "/runpod-volume"}]
                            },
                        }
                    ],
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
            "name": "cpu-prepare",
            "imageName": "runpod/pytorch:example",
            "cloudType": "SECURE",
            "containerDiskInGb": 30,
            "cpuFlavorIds": ["cpu3g"],
            "vcpuCount": 8,
            "dataCenterIds": ["EU-RO-1"],
            "env": {"RUNPOD_ROLE": "cpu-prepare"},
            "networkVolumeId": "volume_one",
            "volumeMountPath": "/runpod-volume",
        }
        with (
            patch.dict(CONTROL["_create_cpu_pod"].__globals__, {"_request": fake_request}),
            patch.object(sys, "stdin", io.StringIO(json.dumps(source))),
        ):
            created = CONTROL["_create_cpu_pod"]()
        self.assertEqual(captured["path"], "/pods")
        self.assertEqual(captured["expected_status"], 201)
        self.assertEqual(captured["body"]["cpu"], {"id": "cpu3g", "vcpuCount": 8})
        self.assertEqual(
            captured["body"]["mounts"],
            {"network": [{"volumeId": "volume_one", "path": "/runpod-volume"}]},
        )
        self.assertEqual(captured["body"]["ports"], ["22/tcp"])
        self.assertTrue(captured["body"]["startSsh"])
        self.assertEqual(created["id"], "cpu_pod")

    def test_gpu_create_uses_rest_v2_with_cuda_floor_and_network_volume(self) -> None:
        captured = {}

        def fake_request(method, path, **kwargs):
            captured.update(method=method, path=path, **kwargs)
            return {"id": "gpu_pod", "status": "PROVISIONING", "runtime": None}

        source = {
            "name": "stock-forecasting-train",
            "imageName": "runpod/pytorch:example",
            "cloudType": "SECURE",
            "containerDiskInGb": 50,
            "gpuId": "NVIDIA GeForce RTX 5090",
            "gpuCount": 1,
            "minCudaVersion": "12.8",
            "dataCenterIds": ["EU-RO-1"],
            "env": {"RUNPOD_ROLE": "gpu-train"},
            "networkVolumeId": "volume_one",
            "volumeMountPath": "/runpod-volume",
        }
        with (
            patch.dict(CONTROL["_create_gpu_pod"].__globals__, {"_request": fake_request}),
            patch.object(sys, "stdin", io.StringIO(json.dumps(source))),
        ):
            created = CONTROL["_create_gpu_pod"]()
        self.assertEqual((captured["method"], captured["path"]), ("POST", "/pods"))
        self.assertEqual(captured["expected_status"], 201)
        self.assertEqual(
            captured["body"]["gpu"],
            {
                "id": "NVIDIA GeForce RTX 5090",
                "count": 1,
                "minCudaVersion": "12.8",
            },
        )
        self.assertEqual(
            captured["body"]["mounts"],
            {"network": [{"volumeId": "volume_one", "path": "/runpod-volume"}]},
        )
        self.assertTrue(captured["body"]["startSsh"])
        self.assertEqual(captured["body"]["ports"], ["22/tcp"])
        self.assertEqual(created["id"], "gpu_pod")

    def test_unsupported_cpu_counts_are_rejected_by_v2_adapter(self) -> None:
        for count in (1, 3, 6, 33):
            with (
                self.subTest(count=count),
                patch.object(
                    sys,
                    "stdin",
                    io.StringIO(
                        json.dumps(
                            {
                                "cpuFlavorIds": ["cpu3g"],
                                "vcpuCount": count,
                                "dataCenterIds": ["EU-RO-1"],
                                "networkVolumeId": "volume_one",
                                "volumeMountPath": "/runpod-volume",
                            }
                        )
                    ),
                ),
                self.assertRaises(CONTROL["ApiError"]) as caught,
            ):
                CONTROL["_create_cpu_pod"]()
            self.assertEqual(caught.exception.code, "usage_error")

    def test_http_404_keeps_not_found_code_for_orphan_reconciliation(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.runpod.io/v2/pods/missing",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"title":"Pod not found"}'),
        )
        with (
            patch.dict("os.environ", {"RUNPOD_API_KEY": "test-only"}),
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaises(CONTROL["ApiError"]) as caught,
        ):
            CONTROL["_request"]("GET", "/pods/missing")
        self.assertEqual((caught.exception.status, caught.exception.code), (404, "not_found"))


if __name__ == "__main__":
    unittest.main()
