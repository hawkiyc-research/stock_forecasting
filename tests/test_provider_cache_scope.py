"""Unfinished Taiwan benchmark months must never be reused as complete months."""

from __future__ import annotations

import inspect
import json
import os
from datetime import UTC, date, datetime
from typing import Any

import requests

from stock_forecasting.data.providers.cache_scope import (
    TAIWAN_MONTH_ENDPOINTS,
    taiwan_month_scope,
)
from stock_forecasting.data.providers.http import CachedJsonClient
from stock_forecasting.data.providers.taiwan import TPExProvider, TWSEProvider

TWSE_MONTH = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
TWSE_DAILY = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
TODAY = date(2026, 10, 20)


class _Session:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        self.calls.append(endpoint)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.payload).encode()
        response.headers = {}
        return response


def _client(root, session, end=None, today=TODAY):
    return CachedJsonClient(
        provider="twse_official",
        raw_cache_root=root,
        max_requests_per_second=10.0,
        session=session,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
        cache_scope=None if end is None else taiwan_month_scope(end, today=today),
    )


def test_month_scope_separates_complete_and_unfinished_months():
    scope = taiwan_month_scope("2026-10-20", today=TODAY)
    assert scope(TWSE_MONTH, {"date": "20260901"}) == {"month": "complete"}
    assert scope(TWSE_MONTH, {"date": "20261001"}) == {
        "month": "partial", "cutoff_exclusive": "2026-10-20"
    }
    tpex = "https://www.tpex.org.tw/www/zh-tw/indexInfo/inx"
    assert scope(tpex, {"date": "2026/05/01"}) == {"month": "complete"}
    assert scope(TWSE_DAILY, {"date": "20261001"}) is None
    # A month boundary end makes the previous month complete; a future end never does.
    assert taiwan_month_scope("2026-06-01", today=TODAY)(
        TWSE_MONTH, {"date": "20260501"}
    ) == {"month": "complete"}
    assert taiwan_month_scope("2026-12-01", today=TODAY)(
        TWSE_MONTH, {"date": "20261101"}
    )["month"] == "partial"


def test_only_month_requests_get_a_new_key(tmp_path):
    plain = _client(tmp_path, _Session({}))
    scoped = _client(tmp_path, _Session({}), end="2026-10-20")
    daily = {"response": "json", "date": "20261019", "type": "ALLBUT0999"}
    assert plain._identity(endpoint=TWSE_DAILY, params=daily) == scoped._identity(
        endpoint=TWSE_DAILY, params=daily
    )
    month = {"response": "json", "date": "20260901"}
    assert plain._identity(endpoint=TWSE_MONTH, params=month)[0] != scoped._identity(
        endpoint=TWSE_MONTH, params=month
    )[0]


def test_an_unfinished_month_is_refetched_by_a_later_end_date(tmp_path):
    october = {"response": "json", "date": "20261001"}
    september = {"response": "json", "date": "20260901"}
    early = _Session({"data": ["through 2026-10-19"]})
    first = _client(tmp_path, early, end="2026-10-20")
    first.get_json(TWSE_MONTH, params=october)
    first.get_json(TWSE_MONTH, params=september)
    # The same cut-off reuses both responses.
    again = _Session({"data": ["unused"]})
    _client(tmp_path, again, end="2026-10-20").get_json(TWSE_MONTH, params=october)
    assert again.calls == []
    # After October ends, the partial response is never read; September is.
    later = _Session({"data": ["full October"]})
    client = _client(tmp_path, later, end="2026-11-01", today=date(2026, 11, 2))
    payload, record = client.get_json(TWSE_MONTH, params=october)
    assert payload == {"data": ["full October"]} and not record.cache_hit
    _payload, cached = client.get_json(TWSE_MONTH, params=september)
    assert cached.cache_hit and len(later.calls) == 1


def _written(path, day):
    stamp = datetime(day.year, day.month, day.day, 12, tzinfo=UTC).timestamp()
    os.utime(path, (stamp, stamp))


def test_history_cached_before_scopes_is_reused_only_when_written_after_month_end(tmp_path):
    legacy_root = tmp_path / "datasets" / "training" / "api-cache"
    september = {"response": "json", "date": "20260901"}
    august = {"response": "json", "date": "20260801"}
    october = {"response": "json", "date": "20261001"}
    legacy = _client(legacy_root, _Session({"data": ["legacy"]}))
    for params, written in (
        (september, date(2026, 10, 2)),
        (august, date(2026, 8, 20)),
        (october, date(2026, 10, 19)),
    ):
        legacy.get_json(TWSE_MONTH, params=params)
        sha, _identity = legacy._identity(endpoint=TWSE_MONTH, params=params)
        _written(legacy._cache_path(sha), written)
    session = _Session({"data": ["fresh"]})
    client = CachedJsonClient(
        provider="twse_official",
        raw_cache_root=tmp_path / "datasets" / "forward" / "api-cache",
        read_cache_roots=(legacy_root,),
        max_requests_per_second=10.0,
        session=session,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
        cache_scope=taiwan_month_scope("2026-10-20", today=TODAY),
    )
    # Written after September ended: complete history, reused from the other namespace.
    payload, record = client.get_json(TWSE_MONTH, params=september)
    assert payload == {"data": ["legacy"]} and record.cache_hit and session.calls == []
    assert record.request_sha256 == legacy._identity(endpoint=TWSE_MONTH, params=september)[0]
    # Written mid-August, or an unfinished month: fetched again under the scoped key.
    assert client.get_json(TWSE_MONTH, params=august)[0] == {"data": ["fresh"]}
    assert client.get_json(TWSE_MONTH, params=october)[0] == {"data": ["fresh"]}
    assert len(session.calls) == 2


def test_a_fetched_complete_month_is_also_stored_for_readers_without_a_scope(tmp_path):
    september = {"response": "json", "date": "20260901"}
    october = {"response": "json", "date": "20261001"}
    fetched = _Session({"data": ["fetched"]})
    _client(tmp_path, fetched, end="2026-10-20").get_json(TWSE_MONTH, params=september)
    _client(tmp_path, fetched, end="2026-10-20").get_json(TWSE_MONTH, params=october)
    assert len(fetched.calls) == 2
    # The dataset preparation (no scope) reuses the complete month, never the partial one.
    plain = _Session({"data": ["plain"]})
    payload, record = _client(tmp_path, plain).get_json(TWSE_MONTH, params=september)
    assert payload == {"data": ["fetched"]} and record.cache_hit
    payload, record = _client(tmp_path, plain).get_json(TWSE_MONTH, params=october)
    assert payload == {"data": ["plain"]} and not record.cache_hit


def test_scope_covers_every_provider_benchmark_month_endpoint():
    source = inspect.getsource(TWSEProvider.fetch_benchmark_month) + inspect.getsource(
        TPExProvider.fetch_benchmark_month
    )
    assert all(endpoint in source for endpoint in TAIWAN_MONTH_ENDPOINTS)
    assert source.count("self.client.get_json(") == len(TAIWAN_MONTH_ENDPOINTS)
