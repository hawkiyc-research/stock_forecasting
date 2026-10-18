"""TAIFEX daily futures institutional positions and TXO put/call ratios.

The official OpenAPI only serves recent sessions, so history comes from the
public download forms (POST, Big5 CSV). Columns are located by header name and a
changed header fails loudly. Form fields follow the official download pages and
are confirmed by the first live run; failures are recorded per source.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pandas as pd
import requests

from .base import ProviderFetch
from .http import CachedJsonClient
from .taiwan import _gregorian_date, _number

FUTURES_INSTITUTIONAL_ENDPOINT = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
PUT_CALL_RATIO_ENDPOINT = "https://www.taifex.com.tw/cht/3/pcRatioDown"
CSV_ENCODINGS = ("cp950", "utf-8-sig")

FUTURES_HEADERS = {
    "timestamp": "日期",
    "commodity": "商品名稱",
    "investor": "身份別",
    "net_trade_contracts": "多空交易口數淨額",
    "long_open_interest": "多方未平倉口數",
    "short_open_interest": "空方未平倉口數",
    "net_open_interest": "多空未平倉口數淨額",
}
PUT_CALL_HEADERS = {
    "timestamp": "日期",
    "put_volume": "賣權成交量",
    "call_volume": "買權成交量",
    "volume_ratio_percent": "買賣權成交量比率%",
    "put_open_interest": "賣權未平倉量",
    "call_open_interest": "買權未平倉量",
    "open_interest_ratio_percent": "買賣權未平倉量比率%",
}
INVESTORS = {"外資": "foreign", "外資及陸資": "foreign", "投信": "trust", "自營商": "dealer"}


def _decoded_csv(content: bytes) -> tuple[str, str] | None:
    for encoding in CSV_ENCODINGS:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        if "," in first_line and "日期" in first_line:
            return text, encoding
    return None


def csv_as_json_response(response: requests.Response) -> requests.Response:
    """Wrap a CSV download as JSON so the shared cache and retry logic apply.

    A successful response that is not a CSV table (for example an HTML notice)
    becomes an invalid JSON body, which the client retries and never caches.
    """
    if response.status_code != 200:
        return response
    decoded = _decoded_csv(response.content)
    wrapped = requests.Response()
    wrapped.status_code = response.status_code
    wrapped.headers.update({"Content-Type": "application/json"})
    wrapped.url = response.url
    if decoded is None:
        wrapped._content = b"not a TAIFEX CSV table"
    else:
        text, encoding = decoded
        wrapped._content = json.dumps(
            {"format": "csv", "encoding": encoding, "text": text}, ensure_ascii=False
        ).encode("utf-8")
    return wrapped


class CachedFormCsvClient(CachedJsonClient):
    """POST public download forms and cache their CSV text as JSON documents."""

    def _send(self, endpoint: str, params: Any) -> requests.Response:
        session = self._session()
        arguments = {
            "data": dict(params),
            "timeout": self.timeout_seconds,
            "headers": dict(self.headers),
        }
        if self.session is None:
            return csv_as_json_response(session.post(endpoint, **arguments))
        with self._provided_session_lock:
            return csv_as_json_response(session.post(endpoint, **arguments))


def _rows(payload: Any, required: dict[str, str], label: str) -> tuple[dict[str, int], list]:
    if not isinstance(payload, dict) or payload.get("format") != "csv":
        raise ValueError(f"{label} payload is not a cached CSV document")
    reader = list(csv.reader(io.StringIO(payload["text"])))
    if not reader:
        raise ValueError(f"{label} CSV is empty")
    header = [cell.strip() for cell in reader[0]]
    missing = [name for name in required.values() if name not in header]
    if missing:
        raise ValueError(f"{label} CSV header changed; missing {missing}")
    indexes = {key: header.index(name) for key, name in required.items()}
    return indexes, [row for row in reader[1:] if len(row) > max(indexes.values())]


def parse_futures_institutional(payload: Any, *, commodity_prefix: str) -> pd.DataFrame:
    indexes, rows = _rows(payload, FUTURES_HEADERS, "TAIFEX futures institutional")
    records = []
    for row in rows:
        commodity = row[indexes["commodity"]].strip()
        investor = INVESTORS.get(row[indexes["investor"]].strip())
        if not commodity.startswith(commodity_prefix) or investor is None:
            continue
        values = {
            key: _number(row[index])
            for key, index in indexes.items()
            if key not in {"timestamp", "commodity", "investor"}
        }
        if any(value is None for value in values.values()):
            continue
        records.append({
            "timestamp": _gregorian_date(row[indexes["timestamp"]]),
            "commodity": commodity,
            "investor": investor,
            **{key: int(value) for key, value in values.items()},
        })
    return pd.DataFrame(
        records,
        columns=["timestamp", "commodity", "investor", *list(FUTURES_HEADERS)[3:]],
    )


def parse_put_call_ratio(payload: Any) -> pd.DataFrame:
    indexes, rows = _rows(payload, PUT_CALL_HEADERS, "TAIFEX put/call ratio")
    records = []
    for row in rows:
        values = {key: _number(row[index]) for key, index in indexes.items() if key != "timestamp"}
        if any(value is None for value in values.values()):
            continue
        records.append({"timestamp": _gregorian_date(row[indexes["timestamp"]]), **values})
    return pd.DataFrame(records, columns=list(PUT_CALL_HEADERS))


class TAIFEXSource:
    """Fetch one calendar month of a TAIFEX download per request."""

    def __init__(self, client: CachedFormCsvClient) -> None:
        self.client = client

    @staticmethod
    def _range(start: str, end: str) -> dict[str, str]:
        return {"queryStartDate": start.replace("-", "/"), "queryEndDate": end.replace("-", "/")}

    def fetch_futures_institutional(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            FUTURES_INSTITUTIONAL_ENDPOINT,
            params={**self._range(start, end), "commodityId": "TXF"},
        )
        frame = parse_futures_institutional(payload, commodity_prefix="臺股期貨")
        return ProviderFetch(frame=frame, requests=(request,), metadata={"start": start})

    def fetch_put_call_ratio(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            PUT_CALL_RATIO_ENDPOINT, params=self._range(start, end)
        )
        return ProviderFetch(
            frame=parse_put_call_ratio(payload), requests=(request,), metadata={"start": start}
        )
