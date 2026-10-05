"""Dependency-free numerical data policy shared by baseline and model workflows."""

from __future__ import annotations

import copy
import json
import math
import runpy
from datetime import date
from pathlib import Path


def load_data_policy(project: Path | None = None) -> dict:
    project = project or Path(__file__).resolve().parents[2]
    policy = json.loads((project / "configs/data_cleaning.json").read_text())
    expected = {
        "schema_version",
        "calendar_version",
        "calendars",
        "require_positive_asset_volume",
        "liquidity_lookback_sessions",
        "minimum_median_daily_turnover",
        "evaluation_source_start",
        "calendar_overrides",
    }
    if set(policy) != expected or policy["schema_version"] != 1:
        raise ValueError("Unsupported data-cleaning policy")
    if policy["calendar_version"] != "4.13.2" or policy["calendars"] != {
        "US": "XNYS",
        "TWSE": "XTAI",
        "TPEX": "XTAI",
    }:
        raise ValueError("Data cleaning requires the pinned offline exchange calendars")
    if policy["require_positive_asset_volume"] is not True:
        raise ValueError("Complete input/output windows require positive asset volume")
    overrides = policy["calendar_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - set(policy["calendars"].values()):
        raise ValueError("Calendar overrides must refer to configured exchange calendars")
    for values in overrides.values():
        if set(values) != {"open", "closed"}:
            raise ValueError("Calendar overrides require open and closed dates")
        for days in values.values():
            if not isinstance(days, list) or days != sorted(set(days)):
                raise ValueError("Calendar override dates must be sorted and unique")
            for day in days:
                date.fromisoformat(day)
        if set(values["open"]) & set(values["closed"]):
            raise ValueError("A calendar date cannot be both open and closed")
    lookback = policy["liquidity_lookback_sessions"]
    if type(lookback) is not int or not 1 <= lookback <= 128:
        raise ValueError("Liquidity lookback must be between 1 and 128 market sessions")
    thresholds = policy["minimum_median_daily_turnover"]
    if set(thresholds) != {"USD", "TWD"} or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in thresholds.values()
    ):
        raise ValueError("Liquidity thresholds must be finite positive USD/TWD amounts")
    date.fromisoformat(policy["evaluation_source_start"])
    return policy


def evaluation_request(selection: dict, policy: dict) -> dict:
    """Pin both evaluation splits to one prepared snapshot, independent of train history."""
    request = copy.deepcopy(selection["dataset_request"])
    request["date_range"]["start_inclusive"] = policy["evaluation_source_start"]
    if request["date_range"]["start_inclusive"] >= request["date_range"]["end_exclusive"]:
        raise ValueError("Shared evaluation history must start before the dataset end")
    return request


def evaluation_dataset_id(selection: dict, policy: dict) -> str:
    # Use the same dependency-free identity definition as configure; never stamp
    # a historical SHA into a manifest or bind data to the model architecture.
    import hashlib

    identity = runpy.run_path(str(Path(__file__).with_name("dataset_identity.py")))
    core = identity["dataset_request_identity_payload"](evaluation_request(selection, policy))
    return hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
