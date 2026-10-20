"""Exporting a forward fetch cache for the shared volume cache (no network)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
import requests

from stock_forecasting.data.manifest import sha256_file
from stock_forecasting.data.providers.base import ProviderFetch
from stock_forecasting.data.providers.cache_scope import taiwan_month_scope
from stock_forecasting.data.providers.http import CachedJsonClient
from stock_forecasting.forward import cache_export
from stock_forecasting.forward import fetch as forward_fetch

END = "2026-10-20"
PRICE = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
RETURN = "https://www.twse.com.tw/indicesReport/MFI94U"
DAILY = "https://www.twse.com.tw/exchangeReport/MI_INDEX"


class _Session:
    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"endpoint": endpoint, **kwargs["params"]}).encode()
        response.headers = {}
        return response


class _FakeTWSE:
    def __init__(self, client):
        self.client = client

    def fetch_benchmark_month(self, *, month, dataset_profile):
        params = {"response": "json", "date": month.replace("-", "") + "01"}
        self.client.get_json(PRICE, params=params)
        self.client.get_json(RETURN, params=params)
        return ProviderFetch(frame=pd.DataFrame(), requests=(), metadata={})


def _write(cache, endpoint, date, written):
    client = CachedJsonClient(
        provider="twse_official", raw_cache_root=cache, max_requests_per_second=100.0,
        session=_Session(), sleeper=lambda _seconds: None,
    )
    params = {"response": "json", "date": date}
    client.get_json(endpoint, params=params)
    sha, _identity = client._identity(endpoint=endpoint, params=params)
    relative = f"twse_official/{sha[:2]}/{sha}.json"
    stamp = datetime.fromisoformat(written).replace(tzinfo=UTC).timestamp()
    os.utime(cache / relative, (stamp, stamp))
    return relative


def _scoped(endpoint, date):
    client = CachedJsonClient(
        provider="twse_official", raw_cache_root=".", max_requests_per_second=1.0,
        cache_scope=taiwan_month_scope(END),
    )
    sha, _identity = client._identity(endpoint=endpoint, params={"response": "json", "date": date})
    return f"twse_official/{sha[:2]}/{sha}.json"


@pytest.fixture
def fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_export, "PROVIDERS", {"twse_official": _FakeTWSE})
    root = tmp_path / "fetch"
    cache = root / "api-cache"
    entries = {
        "september_price": _write(cache, PRICE, "20260901", "2026-10-02T01:00:00"),
        # Written before September ended: nothing proves it complete.
        "september_return": _write(cache, RETURN, "20260901", "2026-09-20T01:00:00"),
        "october_price": _write(cache, PRICE, "20261001", "2026-10-19T12:00:00"),
        "october_return": _write(cache, RETURN, "20261001", "2026-10-19T12:00:00"),
        "daily": _write(cache, DAILY, "20261019", "2026-10-19T12:00:00"),
    }
    raw = root / "raw" / "market.parquet"
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"symbol": ["TAIEX.TW"]}).to_parquet(raw, index=False)
    (root / "fetch.json").write_text(json.dumps({
        "schema_version": forward_fetch.FETCH_SCHEMA_VERSION,
        "kind": forward_fetch.FETCH_KIND,
        "end_exclusive": END,
        "providers": {"twse_official": {"benchmark_months": ["2026-09", "2026-10"]}},
        "raw": {"relative_path": "raw/market.parquet", "sha256": sha256_file(raw)},
    }))
    return root, entries


def test_export_rekeys_complete_months_and_leaves_out_the_rest(fetched, tmp_path):
    root, entries = fetched
    output = tmp_path / "shared-cache"
    manifest = cache_export.export_shared_cache(root, output, end_exclusive=END)
    cache = root / "api-cache"
    rekeyed = _scoped(PRICE, "20260901")
    complete = (cache / entries["september_price"]).read_bytes()
    # A proven complete month under both keys: forward fetches and the dataset preparation.
    assert (output / rekeyed).read_bytes() == complete
    assert (output / entries["september_price"]).read_bytes() == complete
    assert (output / entries["daily"]).read_bytes() == (cache / entries["daily"]).read_bytes()
    assert set(manifest["files"]) == {rekeyed, entries["september_price"], entries["daily"]}
    assert manifest["rekeyed_complete_months"] == 1
    assert manifest["skipped_unfinished_months"] == sorted(
        entries[name] for name in ("september_return", "october_price", "october_return")
    )
    # Unfinished or unproven months are never exported under any key.
    for name in ("september_return", "october_price", "october_return"):
        assert not (output / entries[name]).exists()
    with pytest.raises(FileExistsError):
        cache_export.export_shared_cache(root, output, end_exclusive=END)


def test_export_never_fetches_a_missing_month(fetched, tmp_path):
    root, entries = fetched
    os.replace(root / "api-cache" / entries["october_return"], tmp_path / "moved.json")
    with pytest.raises(RuntimeError, match="missing from the local cache"):
        cache_export.export_shared_cache(root, tmp_path / "out", end_exclusive=END)
    with pytest.raises(ValueError, match="end date"):
        cache_export.export_shared_cache(root, tmp_path / "out", end_exclusive="2026-10-21")
