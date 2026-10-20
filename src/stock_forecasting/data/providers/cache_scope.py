"""Cache scopes for provider responses that keep changing until their period ends.

The provider cache is keyed by request content and shared by every dataset namespace on
the volume, so a response is reused by later, longer datasets. A Taiwan benchmark-month
response requested before the month ends holds only the sessions so far; reusing it
later would silently drop the rest of the month. Month requests therefore get an extra
identity field: complete months share one key, while an incomplete month is keyed to the
date the data was cut off and is never reused by a later end date. Every other request
keeps its historical key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

# Benchmark-month endpoints and their month parameter format (taiwan.py providers).
TAIWAN_MONTH_ENDPOINTS = {
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST": "%Y%m%d",
    "https://www.twse.com.tw/indicesReport/MFI94U": "%Y%m%d",
    "https://www.tpex.org.tw/www/zh-tw/indexInfo/inx": "%Y/%m/%d",
    "https://www.tpex.org.tw/www/zh-tw/indexInfo/ROE": "%Y/%m/%d",
}

CacheScope = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None]


def taiwan_month_scope(exclusive_end: str, *, today: date | None = None) -> CacheScope:
    """Scope month responses by whether the month had ended when the data was cut off."""
    cutoff = min(date.fromisoformat(exclusive_end), today or datetime.now(UTC).date())

    def scope(endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        pattern = TAIWAN_MONTH_ENDPOINTS.get(endpoint)
        if pattern is None:
            return None
        first = datetime.strptime(str(params["date"]), pattern).date().replace(day=1)
        following = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following <= cutoff:
            return {"month": "complete"}
        return {"month": "partial", "cutoff_exclusive": cutoff.isoformat()}

    return scope
