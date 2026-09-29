#!/usr/bin/env python3
"""Create a non-destructive, uniquely named RunPod Secret for the TPEx relay."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime

RUNPOD_REST_V2_ENDPOINT = "https://api.runpod.io/v2"
RUNPOD_REST_USER_AGENT = "stock-forecasting-runpod-control/0.1"


def _redacted_error(raw: bytes, *secrets_to_hide: str) -> str:
    text = raw[:65536].decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "RunPod returned a non-JSON error response"

    def safe_detail(value: object) -> str:
        rendered = str(value)
        for secret in secrets_to_hide:
            if secret:
                rendered = rendered.replace(secret, "<redacted>")
        return " ".join(rendered.split())[:500]

    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list):
        messages = [
            safe_detail(item.get("message", "unknown error"))
            if isinstance(item, dict)
            else safe_detail(item)
            for item in errors
            if isinstance(item, (dict, str))
        ]
        if messages:
            return "; ".join(messages)[:1000]
    gateway_fields = []
    for name in (
        "title", "status", "error_code", "error_name", "error_category", "detail"
    ):
        value = payload.get(name) if isinstance(payload, dict) else None
        if isinstance(value, (str, int, float, bool)):
            gateway_fields.append(f"{name}={safe_detail(value)}")
    if gateway_fields:
        return "; ".join(gateway_fields)
    return "RunPod rejected the REST API v2 request"


def _required_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY is required")
    return api_key


def _rest_request(
    *,
    api_key: str,
    method: str,
    path: str,
    expected_status: int,
    body: dict[str, str] | None = None,
    additional_redactions: tuple[str, ...] = (),
) -> dict[str, object]:
    request = urllib.request.Request(
        RUNPOD_REST_V2_ENDPOINT + path,
        data=(
            json.dumps(body, separators=(",", ":")).encode("utf-8")
            if body is not None
            else None
        ),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": RUNPOD_REST_USER_AGENT,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
        finally:
            error.close()
        raise RuntimeError(
            f"RunPod REST API v2 request failed with HTTP {error.code}: "
            f"{_redacted_error(raw, api_key, *additional_redactions)}"
        ) from None
    except urllib.error.URLError as error:
        reason = str(error.reason)
        for value in (api_key, *additional_redactions):
            if value:
                reason = reason.replace(value, "<redacted>")
        raise RuntimeError(f"RunPod REST API v2 could not be reached: {reason}") from None
    if status != expected_status:
        raise RuntimeError(
            f"RunPod REST API v2 returned unexpected HTTP {status}: "
            f"{_redacted_error(raw, api_key, *additional_redactions)}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("RunPod REST API v2 returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("RunPod REST API v2 returned an invalid response object")
    return payload


def _check_access(api_key: str) -> None:
    payload = _rest_request(
        api_key=api_key,
        method="GET",
        path="/account/secrets?name=tpex_relay_token_preflight",
        expected_status=200,
    )
    if not isinstance(payload.get("secrets"), list):
        raise RuntimeError("RunPod REST API v2 returned an invalid secrets list")


def _create_secret(api_key: str) -> str:
    shared_secret = os.environ.get("TPEX_PROXY_SHARED_SECRET", "")
    if len(shared_secret) < 32 or len(shared_secret) > 512:
        raise ValueError("TPEX_PROXY_SHARED_SECRET has an invalid length")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    secret_name = f"tpex_relay_token_{stamp}_{secrets.token_hex(3)}"
    payload = _rest_request(
        api_key=api_key,
        method="POST",
        path="/account/secrets",
        expected_status=201,
        body={"name": secret_name, "value": shared_secret},
        additional_redactions=(shared_secret,),
    )
    if (
        payload.get("name") != secret_name
        or not isinstance(payload.get("id"), str)
        or not payload["id"]
    ):
        raise RuntimeError("RunPod REST API v2 returned invalid Secret metadata")
    return secret_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Verify read-only RunPod REST API v2 access without creating a Secret.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    api_key = _required_api_key()
    if arguments.check_access:
        _check_access(api_key)
        print("RunPod REST API v2 access preflight passed.")
        return 0
    print(_create_secret(api_key))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"RunPod TPEx relay Secret setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
