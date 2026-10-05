#!/usr/bin/env python3
"""Run the project's Runpod control operations through REST API v2."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
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
    if type(vcpu_count) is not int or vcpu_count not in {2, 4, 8, 16, 32}:
        raise ApiError(None, "usage_error", "CPU Pod requires 2, 4, 8, 16, or 32 vCPUs")
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


def _create_gpu_pod() -> dict[str, object]:
    try:
        source = json.load(sys.stdin)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApiError(None, "usage_error", "GPU Pod request is invalid JSON") from error
    if not isinstance(source, dict):
        raise ApiError(None, "usage_error", "GPU Pod request must be an object")
    gpu_id = source.get("gpuId")
    gpu_count = source.get("gpuCount")
    volume_id = source.get("networkVolumeId")
    mount_path = source.get("volumeMountPath")
    env = source.get("env")
    datacenters = source.get("dataCenterIds")
    disk = source.get("containerDiskInGb")
    min_cuda = source.get("minCudaVersion")
    if not isinstance(gpu_id, str) or not gpu_id or any(char in gpu_id for char in "\r\n"):
        raise ApiError(None, "usage_error", "GPU type ID is invalid")
    if type(gpu_count) is not int or gpu_count < 1:
        raise ApiError(None, "usage_error", "GPU count is invalid")
    if not isinstance(volume_id, str) or SAFE_ID.fullmatch(volume_id) is None:
        raise ApiError(None, "usage_error", "GPU Pod network volume ID is invalid")
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        raise ApiError(None, "usage_error", "GPU Pod network mount is invalid")
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise ApiError(None, "usage_error", "GPU Pod environment is invalid")
    if not isinstance(source.get("name"), str) or not isinstance(source.get("imageName"), str):
        raise ApiError(None, "usage_error", "GPU Pod name or image is invalid")
    if source.get("cloudType") not in {"SECURE", "COMMUNITY"}:
        raise ApiError(None, "usage_error", "GPU Pod cloud type is invalid")
    if type(disk) is not int or disk < 1:
        raise ApiError(None, "usage_error", "GPU Pod disk size is invalid")
    if not isinstance(datacenters, list) or not datacenters or not all(
        isinstance(dc, str) and dc for dc in datacenters
    ):
        raise ApiError(None, "usage_error", "GPU Pod data centers are invalid")
    if not isinstance(min_cuda, str) or re.fullmatch(r"[0-9]+\.[0-9]+", min_cuda) is None:
        raise ApiError(None, "usage_error", "GPU Pod CUDA floor is invalid")
    body: dict[str, object] = {
        "name": source.get("name"),
        "image": source.get("imageName"),
        "cloud": source.get("cloudType"),
        "disk": disk,
        "gpu": {
            "id": gpu_id,
            "count": gpu_count,
            "minCudaVersion": min_cuda,
        },
        "dataCenterIds": datacenters,
        "env": env,
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


def _gpu_entries(value: list[object]) -> list[dict[str, object]]:
    # Validate before rendering so malformed entries cannot produce partial tables.
    for entry in value:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            or not all(character.isprintable() for character in entry[key])
            for key in ("id", "name")
        ):
            raise ApiError(None, "api_error", "Runpod REST v2 GPU entry is invalid")
    return value


def _gpu_data_centers(gpu: dict[str, object]) -> list[dict[str, object]]:
    centers = gpu.get("dataCenters")
    if not isinstance(centers, list):
        return []
    return [center for center in centers if isinstance(center, dict)
            and isinstance(center.get("id"), str)]


def _gpu_number(value: object, *, price: bool = False) -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "--"
    return f"{value:.2f}" if price else f"{value:g}"


def _gpu_text(value: object) -> str:
    # Keep provider text from injecting terminal controls into human-readable output.
    return "".join(character if character.isprintable() else "?"
                   for character in str(value)) if value is not None else "--"


def _print_gpu_table(gpus: list[dict[str, object]], data_center: str | None) -> None:
    print(f"GPU catalog: {len(gpus)} matches | prices: USD/hour")
    print(f"Stock scope: {data_center}" if data_center else "Stock scope: global catalog")
    if not gpus:
        print("No matching GPUs.")
        return
    row = "{:<32} {:>7} {:>10} {:>12}  {}"
    print(row.format("GPU", "VRAM/GB", "SECURE", "COMMUNITY", "STOCK"))
    print("-" * 80)
    for gpu in gpus:
        price = gpu.get("price")
        price = price if isinstance(price, dict) else {}
        centers = _gpu_data_centers(gpu)
        if data_center:
            centers = [center for center in centers
                       if center["id"].casefold() == data_center.casefold()]
        stock = centers[0].get("availability") if data_center else gpu.get("availability")
        name = textwrap.shorten(gpu["name"], width=32, placeholder="...")
        print(row.format(name, _gpu_number(gpu.get("memory")),
                         _gpu_number(price.get("secure"), price=True),
                         _gpu_number(price.get("community"), price=True),
                         _gpu_text(stock)))
        # Never truncate the identifier users need to pass to --gpuId.
        print(f"  gpuId: {gpu['id']}")
        locations = ", ".join(
            f"{_gpu_text(center['id'])}:{_gpu_text(center.get('availability'))}"
            for center in centers
        ) or "-- (not reported)"
        print(textwrap.fill("Data centers: " + locations, width=96,
                            initial_indent="  ", subsequent_indent="    ",
                            break_long_words=False, break_on_hyphens=False))
        print()
    print("Use the full gpuId above for --gpuId; stock is not a capacity reservation.")


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
    pod_actions.add_parser("create-gpu")
    volume = resources.add_parser("network-volume")
    volume_actions = volume.add_subparsers(dest="action", required=True)
    create = volume_actions.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--size", type=int, required=True)
    create.add_argument("--data-center-id", required=True)
    create.add_argument("--output", choices=("json",), default="json")
    gpu = resources.add_parser("gpu")
    gpu_actions = gpu.add_subparsers(dest="action", required=True)
    gpu_list = gpu_actions.add_parser("list", help="Show GPU models, prices, and stock")
    gpu_list.add_argument("--output", choices=("table", "json"), default="table",
                          help="Output format (default: table; use json for scripts)")
    gpu_list.add_argument("--search", help="Case-insensitive GPU name or gpuId substring")
    gpu_list.add_argument("--data-center", help="Only show GPUs listed in this data center")
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
        elif args.action == "create-gpu":
            _print(_create_gpu_pod())
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
        gpus = _gpu_entries(result["gpus"])
        if args.search:
            term = args.search.casefold()
            gpus = [gpu for gpu in gpus
                    if term in gpu["name"].casefold() or term in gpu["id"].casefold()]
        if args.data_center:
            gpus = [gpu for gpu in gpus if any(
                center["id"].casefold() == args.data_center.casefold()
                for center in _gpu_data_centers(gpu)
            )]
        if args.output == "json":
            _print(gpus)
        else:
            _print_gpu_table(gpus, args.data_center)
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
