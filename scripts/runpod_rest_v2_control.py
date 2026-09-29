#!/usr/bin/env python3
"""Run the project's Runpod control operations through REST API v2."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.runpod.io/v2"
USER_AGENT = "stock-forecasting-runpod-control/0.1"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "bad_request",
    429: "rate_limited",
}


class ApiError(RuntimeError):
    def __init__(self, status: int | None, code: str, message: str) -> None:
        self.status = status
        self.code = code
        super().__init__(message)


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key or "\n" in key or "\r" in key:
        raise ApiError(None, "no_credentials", "A valid RUNPOD_API_KEY is required")
    return key


def _resource_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ApiError(None, "usage_error", "Runpod resource ID is invalid")
    return value


def _request(
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
    expected_status: int = 200,
) -> dict[str, object] | None:
    key = _api_key()
    request = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as error:
        try:
            raw = error.read(8192)
        finally:
            error.close()
        message = f"Runpod REST v2 returned HTTP {error.code}"
        try:
            problem = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            problem = None
        if isinstance(problem, dict) and isinstance(problem.get("title"), str):
            message += f": {problem['title'][:120]}"
        raise ApiError(error.code, STATUS_CODES.get(error.code, "api_error"), message) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = error.reason if isinstance(error, urllib.error.URLError) else type(error).__name__
        raise ApiError(None, "network_error", f"Runpod REST v2 is unavailable: {reason}") from None
    if status != expected_status:
        raise ApiError(status, "api_error", f"Runpod REST v2 returned unexpected HTTP {status}")
    if expected_status == 204:
        return None
    if len(raw) > 2_000_000:
        raise ApiError(status, "api_error", "Runpod REST v2 response is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApiError(status, "api_error", "Runpod REST v2 returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ApiError(status, "api_error", "Runpod REST v2 returned an invalid object")
    return payload


def _pod_compat(pod: dict[str, object]) -> dict[str, object]:
    pod_id = pod.get("id")
    status = pod.get("status")
    if not isinstance(pod_id, str) or SAFE_ID.fullmatch(pod_id) is None:
        raise ApiError(None, "api_error", "Runpod REST v2 Pod has no safe ID")
    if status not in {"PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED"}:
        raise ApiError(None, "api_error", "Runpod REST v2 Pod has an unknown status")
    runtime = pod.get("runtime")
    runtime_status = {
        "PROVISIONING": "initializing",
        "STARTING": "initializing",
        "RUNNING": "running" if isinstance(runtime, dict) else "unknown",
        "EXITED": "stopped",
        "ERROR": "unknown",
        "TERMINATED": "terminated",
    }[status]
    mapped = dict(pod)
    mapped["desiredStatus"] = status
    mapped["runtimeStatus"] = runtime_status
    mounts = pod.get("mounts")
    if isinstance(mounts, dict):
        network = mounts.get("network")
        if isinstance(network, list) and len(network) == 1 and isinstance(network[0], dict):
            volume_id = network[0].get("volumeId")
            if isinstance(volume_id, str):
                mapped["networkVolumeId"] = volume_id
                mapped["networkVolume"] = {"id": volume_id}
    if isinstance(runtime, dict) and isinstance(runtime.get("uptime"), int):
        mapped["uptimeSeconds"] = runtime["uptime"]
    return mapped


def _pod_list() -> list[dict[str, object]]:
    pods: list[dict[str, object]] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(20):
        path = "/pods?limit=50"
        if cursor is not None:
            path += "&cursor=" + urllib.parse.quote(cursor, safe="")
        response = _request("GET", path)
        if not isinstance(response, dict) or not isinstance(response.get("pods"), list):
            raise ApiError(None, "api_error", "Runpod REST v2 Pod list is invalid")
        for pod in response["pods"]:
            if not isinstance(pod, dict):
                raise ApiError(None, "api_error", "Runpod REST v2 Pod entry is invalid")
            pods.append(_pod_compat(pod))
        pagination = response.get("pagination")
        if not isinstance(pagination, dict) or not isinstance(pagination.get("hasNextPage"), bool):
            raise ApiError(None, "api_error", "Runpod REST v2 Pod pagination is invalid")
        if not pagination["hasNextPage"]:
            return pods
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise ApiError(None, "api_error", "Runpod REST v2 Pod cursor is invalid")
        seen.add(cursor)
    raise ApiError(None, "api_error", "Runpod REST v2 Pod list exceeds the 1000-Pod safety limit")


def _create_cpu_pod() -> dict[str, object]:
    try:
        source = json.load(sys.stdin)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApiError(None, "usage_error", "CPU Pod request is invalid JSON") from error
    if not isinstance(source, dict):
        raise ApiError(None, "usage_error", "CPU Pod request must be an object")
    flavor = source.get("cpuFlavorIds")
    datacenters = source.get("dataCenterIds")
    if not isinstance(flavor, list) or len(flavor) != 1 or not isinstance(flavor[0], str):
        raise ApiError(None, "usage_error", "CPU Pod flavor is invalid")
    if not isinstance(datacenters, list) or not datacenters or not all(
        isinstance(dc, str) and dc for dc in datacenters
    ):
        raise ApiError(None, "usage_error", "CPU Pod data centers are invalid")
    vcpu_count = source.get("vcpuCount")
    if type(vcpu_count) is not int or vcpu_count < 2 or vcpu_count & (vcpu_count - 1):
        raise ApiError(None, "usage_error", "REST v2 CPU Pod requires a power-of-two vCPU count")
    volume_id = source.get("networkVolumeId")
    mount_path = source.get("volumeMountPath")
    if not isinstance(volume_id, str) or SAFE_ID.fullmatch(volume_id) is None:
        raise ApiError(None, "usage_error", "CPU Pod network volume ID is invalid")
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        raise ApiError(None, "usage_error", "CPU Pod network mount is invalid")
    if not isinstance(source.get("name"), str) or not isinstance(source.get("imageName"), str):
        raise ApiError(None, "usage_error", "CPU Pod name or image is invalid")
    if not isinstance(source.get("env"), dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in source["env"].items()
    ):
        raise ApiError(None, "usage_error", "CPU Pod environment is invalid")
    body: dict[str, object] = {
        "name": source.get("name"),
        "image": source.get("imageName"),
        "cloud": source.get("cloudType"),
        "disk": source.get("containerDiskInGb"),
        "cpu": {"id": flavor[0], "vcpuCount": vcpu_count},
        "dataCenterIds": datacenters,
        "env": source.get("env"),
        "mounts": {"network": [{"volumeId": volume_id, "path": mount_path}]},
        "ports": ["22/tcp"],
        "startSsh": True,
    }
    response = _request("POST", "/pods", body=body, expected_status=201)
    if not isinstance(response, dict):
        raise ApiError(None, "api_error", "Runpod REST v2 returned no created Pod")
    return _pod_compat(response)


def _print(value: object) -> None:
    print(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    resources = parser.add_subparsers(dest="resource", required=True)
    pod = resources.add_parser("pod")
    pod_actions = pod.add_subparsers(dest="action", required=True)
    listed = pod_actions.add_parser("list")
    listed.add_argument("--all", action="store_true")
    listed.add_argument("--output", choices=("json",), default="json")
    got = pod_actions.add_parser("get")
    got.add_argument("id")
    got.add_argument("--include-network-volume", action="store_true")
    got.add_argument("--include-machine", action="store_true")
    got.add_argument("--output", choices=("json",), default="json")
    for action in ("delete", "start", "stop", "restart"):
        pod_actions.add_parser(action).add_argument("id")
    pod_actions.add_parser("create-cpu")
    volume = resources.add_parser("network-volume")
    volume_actions = volume.add_subparsers(dest="action", required=True)
    create = volume_actions.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--size", type=int, required=True)
    create.add_argument("--data-center-id", required=True)
    create.add_argument("--output", choices=("json",), default="json")
    gpu = resources.add_parser("gpu")
    gpu_actions = gpu.add_subparsers(dest="action", required=True)
    gpu_actions.add_parser("list")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.resource == "pod":
        if args.action == "list":
            pods = _pod_list()
            if not args.all:
                pods = [
                    pod for pod in pods
                    if pod["runtimeStatus"] not in {"stopped", "terminated"}
                ]
            _print(pods)
        elif args.action == "get":
            result = _request("GET", f"/pods/{_resource_id(args.id)}")
            if not isinstance(result, dict):
                raise ApiError(None, "api_error", "Runpod REST v2 returned no Pod")
            _print(_pod_compat(result))
        elif args.action == "delete":
            _request("DELETE", f"/pods/{_resource_id(args.id)}", expected_status=204)
            _print({"id": args.id, "deleted": True})
        elif args.action == "create-cpu":
            _print(_create_cpu_pod())
        else:
            result = _request(
                "POST",
                f"/pods/{_resource_id(args.id)}/action",
                body={"action": args.action},
            )
            if not isinstance(result, dict):
                raise ApiError(None, "api_error", "Runpod REST v2 returned no Pod")
            _print(_pod_compat(result))
    elif args.resource == "network-volume":
        if args.size < 10 or args.size > 4096:
            raise ApiError(None, "usage_error", "Network volume size must be 10-4096 GB")
        result = _request(
            "POST",
            "/network-volumes",
            body={"name": args.name, "size": args.size, "dataCenter": args.data_center_id},
            expected_status=201,
        )
        _print(result)
    elif args.resource == "gpu":
        result = _request("GET", "/catalog/gpus?include=AVAILABILITY&product=POD")
        if not isinstance(result, dict) or not isinstance(result.get("gpus"), list):
            raise ApiError(None, "api_error", "Runpod REST v2 GPU list is invalid")
        _print(result["gpus"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as error:
        payload: dict[str, object] = {"error": str(error), "code": error.code}
        if error.status is not None:
            payload["status"] = error.status
        print(json.dumps(payload, separators=(",", ":")), file=sys.stderr)
        raise SystemExit(1) from None
