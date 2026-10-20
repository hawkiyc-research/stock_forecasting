"""Cache scopes for provider responses that keep changing until their period ends.

The provider cache is keyed by request content and shared by every dataset namespace on
the volume, so a response is reused by later, longer datasets. A Taiwan benchmark-month
response requested before the month ends holds only the sessions so far; reusing it
later would silently drop the rest of the month. Month requests therefore get an extra
identity field: complete months share one key, while an incomplete month is keyed to the
date the data was cut off and is never reused by a later end date. Every other request
keeps its historical key.

Month entries cached before scopes existed carry the historical key. Such an entry is
reused for a complete month only when its file was written after the month ended (the
same file time the client already reports as a cache hit's request time), so history
already on the volume is not downloaded again while an entry written mid-month never
passes as complete. Unscoped month entries must therefore never be copied: a copy gets a
new file time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

# Benchmark-month endpoints and their month parameter format (taiwan.py providers).
TAIWAN_MONTH_ENDPOINTS = {
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST": "%Y%m%d",
    "https://www.twse.com.tw/indicesReport/MFI94U": "%Y%m%d",
    "https://www.tpex.org.tw/www/zh-tw/indexInfo/inx": "%Y/%m/%d",
    "https://www.tpex.org.tw/www/zh-tw/indexInfo/ROE": "%Y/%m/%d",
}


class CacheScope(Protocol):
    def __call__(self, endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Extra identity fields for this request, or None to keep its historical key."""

    def reuses_unscoped(
        self, endpoint: str, params: Mapping[str, Any], written_at: datetime
    ) -> bool:
        """Whether an entry cached under the historical key at ``written_at`` is valid."""


def month_following(endpoint: str, params: Mapping[str, Any]) -> date | None:
    """First day after the requested benchmark month, or None for other requests."""
    pattern = TAIWAN_MONTH_ENDPOINTS.get(endpoint)
    if pattern is None:
        return None
    first = datetime.strptime(str(params["date"]), pattern).date().replace(day=1)
    return (first.replace(day=28) + timedelta(days=4)).replace(day=1)


@dataclass(frozen=True)
class TaiwanMonthScope:
    cutoff: date

    def __call__(self, endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        following = month_following(endpoint, params)
        if following is None:
            return None
        if following <= self.cutoff:
            return {"month": "complete"}
        return {"month": "partial", "cutoff_exclusive": self.cutoff.isoformat()}

    def reuses_unscoped(
        self, endpoint: str, params: Mapping[str, Any], written_at: datetime
    ) -> bool:
        following = month_following(endpoint, params)
        return (
            following is not None
            and following <= self.cutoff
            and following <= written_at.astimezone(UTC).date()
        )


def taiwan_month_scope(exclusive_end: str, *, today: date | None = None) -> TaiwanMonthScope:
    """Scope month responses by whether the month had ended when the data was cut off."""
    return TaiwanMonthScope(
        cutoff=min(date.fromisoformat(exclusive_end), today or datetime.now(UTC).date())
    )
