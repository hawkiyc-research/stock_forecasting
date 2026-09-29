"""Check the RunPod Secret REST requests without contacting RunPod."""

from __future__ import annotations

import importlib.util
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/create_runpod_tpex_proxy_secret.py"
SPEC = importlib.util.spec_from_file_location("runpod_rest_secret", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
secret_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(secret_helper)


class Response:
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class RunpodRestSecretTests(unittest.TestCase):
    def test_preflight_is_read_only_and_uses_bearer_auth(self) -> None:
        def respond(request: object, timeout: int) -> Response:
            self.assertEqual(timeout, 30)
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(
                request.full_url,
                "https://api.runpod.io/v2/account/secrets?name=tpex_relay_token_preflight",
            )
            self.assertEqual(request.get_header("Authorization"), "Bearer sample-key")
            self.assertIsNone(request.data)
            return Response(200, {"secrets": []})

        with patch.object(secret_helper.urllib.request, "urlopen", side_effect=respond):
            secret_helper._check_access("sample-key")

    def test_create_secret_keeps_name_reference_and_sends_value_in_body(self) -> None:
        shared_secret = "token-value-with-at-least-thirty-two-characters"

        def respond(request: object, timeout: int) -> Response:
            self.assertEqual(timeout, 30)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, "https://api.runpod.io/v2/account/secrets")
            self.assertEqual(request.get_header("Authorization"), "Bearer sample-key")
            body = json.loads(request.data)
            self.assertEqual(body["value"], shared_secret)
            self.assertTrue(body["name"].startswith("tpex_relay_token_"))
            return Response(201, {"id": "secret-id", "name": body["name"]})

        with patch.dict(secret_helper.os.environ, {"TPEX_PROXY_SHARED_SECRET": shared_secret}):
            with patch.object(secret_helper.urllib.request, "urlopen", side_effect=respond):
                name = secret_helper._create_secret("sample-key")
        self.assertTrue(name.startswith("tpex_relay_token_"))

    def test_http_error_redacts_api_key_and_secret(self) -> None:
        shared_secret = 'token " value with at least thirty-two characters'

        def reject(request: object, timeout: int) -> None:
            self.assertEqual(timeout, 30)
            self.assertNotIn("sample-key", request.full_url)
            error = {"title": "Conflict", "status": 409, "detail": shared_secret}
            raise urllib.error.HTTPError(
                request.full_url,
                409,
                "Conflict",
                {},
                io.BytesIO(json.dumps(error).encode("utf-8")),
            )

        with patch.dict(secret_helper.os.environ, {"TPEX_PROXY_SHARED_SECRET": shared_secret}):
            with patch.object(secret_helper.urllib.request, "urlopen", side_effect=reject):
                with self.assertRaisesRegex(RuntimeError, "HTTP 409") as caught:
                    secret_helper._create_secret("sample-key")
        self.assertNotIn(shared_secret, str(caught.exception))
        self.assertNotIn("sample-key", str(caught.exception))

    def test_unexpected_success_status_is_rejected(self) -> None:
        with patch.object(
            secret_helper.urllib.request,
            "urlopen",
            return_value=Response(200, {"id": "secret-id", "name": "name"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected HTTP 200"):
                secret_helper._rest_request(
                    api_key="sample-key",
                    method="POST",
                    path="/account/secrets",
                    expected_status=201,
                    body={"name": "name", "value": "sample-value"},
                )


if __name__ == "__main__":
    unittest.main()
