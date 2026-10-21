"""Official TWSE/TPEx daily institutional-investor trades and margin balances.

These are auxiliary research inputs, not training instruments: each row keys a
listed code by session date, and every parser checks accounting identities so a
silent column-layout change fails instead of mislabelling investor groups.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ProviderFetch
from .http import CachedJsonClient
from .taiwan import _asset_type, _clean_field, _is_no_data_payload, _number, _tables

# Share of rows that may break an accounting identity before the layout is
# treated as changed. Official tables occasionally carry corrected rows.
MAX_IDENTITY_VIOLATION_SHARE = 0.01
MARGIN_IDENTITY_MIN_SHARE = 0.95

FLOW_COLUMNS = (
    "foreign_net_shares",
    "foreign_dealer_net_shares",
    "trust_net_shares",
    "dealer_net_shares",
    "institutional_net_shares",
)
MARGIN_COLUMNS = (
    "margin_buy_lots",
    "margin_sell_lots",
    "margin_redeem_lots",
    "margin_previous_lots",
    "margin_balance_lots",
    "margin_limit_lots",
    "short_sell_lots",
    "short_buy_lots",
    "short_redeem_lots",
    "short_previous_lots",
    "short_balance_lots",
    "short_limit_lots",
    "offset_lots",
)

TWSE_T86_FIELDS = {
    "symbol": {"證券代號"},
    "foreign_net_shares": {"外陸資買賣超股數(不含外資自營商)"},
    "foreign_dealer_net_shares": {"外資自營商買賣超股數"},
    "trust_net_shares": {"投信買賣超股數"},
    "dealer_net_shares": {"自營商買賣超股數"},
    "institutional_net_shares": {"三大法人買賣超股數"},
}
TWSE_MARGIN_FIELDS = (
    "代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
    "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記",
)
TPEX_MARGIN_FIELDS = {
    "symbol": {"代號"},
    "margin_previous_lots": {"前資餘額(張)"},
    "margin_buy_lots": {"資買"},
    "margin_sell_lots": {"資賣"},
    "margin_redeem_lots": {"現償"},
    "margin_balance_lots": {"資餘額"},
    "margin_limit_lots": {"資限額"},
    "short_previous_lots": {"前券餘額(張)"},
    "short_sell_lots": {"券賣"},
    "short_buy_lots": {"券買"},
    "short_redeem_lots": {"券償"},
    "short_balance_lots": {"券餘額"},
    "short_limit_lots": {"券限額"},
    "offset_lots": {"資券相抵(張)"},
}
# TPEx repeats buy/sell/net names without group labels. The order is fixed by the
# official page and verified below through the three accounting identities.
TPEX_INSTITUTIONAL_GROUPS = (
    "foreign",
    "foreign_dealer",
    "foreign_total",
    "trust",
    "dealer_own",
    "dealer_hedge",
    "dealer_total",
)


def _table_with(payload: Any, required: set[str]) -> tuple[list[str], list[list[Any]]]:
    for table in _tables(payload):
        fields, data = table.get("fields"), table.get("data")
        if not isinstance(fields, list) or not isinstance(data, list):
            continue
        normalized = [_clean_field(field) for field in fields]
        if required.issubset(normalized):
            return normalized, [row for row in data if isinstance(row, list)]
    raise ValueError("Official Taiwan response has no compatible flow table")


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number != number:
        return None
    return round(number)


def _check_identity(violations: int, rows: int, label: str, limit: float) -> None:
    if rows and violations / rows > limit:
        raise ValueError(f"{label} accounting identity failed for {violations}/{rows} rows")


def _supported(code: str, market: str) -> bool:
    return _asset_type(code, market=market) is not None


def parse_twse_t86(payload: Any, *, date: str) -> pd.DataFrame:
    fields, data = _table_with(payload, set().union(*TWSE_T86_FIELDS.values()))
    indexes = {
        name: next(index for index, field in enumerate(fields) if field in aliases)
        for name, aliases in TWSE_T86_FIELDS.items()
    }
    rows, violations = [], 0
    for raw in data:
        if len(raw) <= max(indexes.values()):
            continue
        code = str(raw[indexes["symbol"]]).strip().upper()
        if not _supported(code, "TWSE"):
            continue
        values = {name: _integer(raw[index]) for name, index in indexes.items() if name != "symbol"}
        if any(value is None for value in values.values()):
            continue
        parts = sum(values[name] for name in FLOW_COLUMNS[:-1])
        violations += parts != values["institutional_net_shares"]
        rows.append({"timestamp": date, "symbol": f"{code}.TW", "market": "TWSE", **values})
    _check_identity(violations, len(rows), "TWSE T86", MAX_IDENTITY_VIOLATION_SHARE)
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "market", *FLOW_COLUMNS])


def parse_tpex_institutional(payload: Any, *, date: str) -> pd.DataFrame:
    fields, data = _table_with(payload, {"代號", "三大法人買賣超股數合計"})
    expected_width = 2 + 3 * len(TPEX_INSTITUTIONAL_GROUPS) + 1
    if len(fields) != expected_width or fields[-1] != "三大法人買賣超股數合計":
        raise ValueError("TPEx institutional table layout changed")
    rows, violations = [], 0
    for raw in data:
        if len(raw) < expected_width:
            continue
        code = str(raw[0]).strip().upper()
        if not _supported(code, "TPEX"):
            continue
        net = {
            group: _integer(raw[2 + 3 * position + 2])
            for position, group in enumerate(TPEX_INSTITUTIONAL_GROUPS)
        }
        total = _integer(raw[-1])
        if total is None or any(value is None for value in net.values()):
            continue
        violations += (
            net["foreign_total"] != net["foreign"] + net["foreign_dealer"]
            or net["dealer_total"] != net["dealer_own"] + net["dealer_hedge"]
            or total != net["foreign_total"] + net["trust"] + net["dealer_total"]
        )
        rows.append({
            "timestamp": date,
            "symbol": f"{code}.TWO",
            "market": "TPEX",
            "foreign_net_shares": net["foreign"],
            "foreign_dealer_net_shares": net["foreign_dealer"],
            "trust_net_shares": net["trust"],
            "dealer_net_shares": net["dealer_total"],
            "institutional_net_shares": total,
        })
    _check_identity(violations, len(rows), "TPEx institutional", MAX_IDENTITY_VIOLATION_SHARE)
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "market", *FLOW_COLUMNS])


def _margin_identity(values: dict[str, int]) -> bool:
    margin = (
        values["margin_previous_lots"] + values["margin_buy_lots"]
        - values["margin_sell_lots"] - values["margin_redeem_lots"]
    )
    short = (
        values["short_previous_lots"] + values["short_sell_lots"]
        - values["short_buy_lots"] - values["short_redeem_lots"]
    )
    return margin == values["margin_balance_lots"] and short == values["short_balance_lots"]


def _margin_frame(rows: list[dict[str, Any]], satisfied: int, label: str) -> pd.DataFrame:
    if rows and satisfied / len(rows) < MARGIN_IDENTITY_MIN_SHARE:
        raise ValueError(f"{label} balance identity holds for only {satisfied}/{len(rows)} rows")
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "market", *MARGIN_COLUMNS])


def parse_twse_margin(payload: Any, *, date: str) -> pd.DataFrame:
    fields, data = _table_with(payload, {"代號", "次一營業日限額", "資券互抵"})
    if tuple(fields[: len(TWSE_MARGIN_FIELDS)]) != TWSE_MARGIN_FIELDS:
        raise ValueError("TWSE margin table layout changed")
    # Positions follow TWSE_MARGIN_FIELDS: margin group, then short group.
    positions = {
        "margin_buy_lots": 2, "margin_sell_lots": 3, "margin_redeem_lots": 4,
        "margin_previous_lots": 5, "margin_balance_lots": 6, "margin_limit_lots": 7,
        "short_buy_lots": 8, "short_sell_lots": 9, "short_redeem_lots": 10,
        "short_previous_lots": 11, "short_balance_lots": 12, "short_limit_lots": 13,
        "offset_lots": 14,
    }
    rows, satisfied = [], 0
    for raw in data:
        if len(raw) < len(TWSE_MARGIN_FIELDS) - 1:
            continue
        code = str(raw[0]).strip().upper()
        if not _supported(code, "TWSE"):
            continue
        values = {name: _integer(raw[index]) for name, index in positions.items()}
        if any(value is None for value in values.values()):
            continue
        satisfied += _margin_identity(values)
        rows.append({"timestamp": date, "symbol": f"{code}.TW", "market": "TWSE", **values})
    return _margin_frame(rows, satisfied, "TWSE margin")


def parse_tpex_margin(payload: Any, *, date: str) -> pd.DataFrame:
    fields, data = _table_with(payload, set().union(*TPEX_MARGIN_FIELDS.values()))
    indexes = {
        name: next(index for index, field in enumerate(fields) if field in aliases)
        for name, aliases in TPEX_MARGIN_FIELDS.items()
    }
    rows, satisfied = [], 0
    for raw in data:
        if len(raw) <= max(indexes.values()):
            continue
        code = str(raw[indexes["symbol"]]).strip().upper()
        if not _supported(code, "TPEX"):
            continue
        values = {name: _integer(raw[index]) for name, index in indexes.items() if name != "symbol"}
        if any(value is None for value in values.values()):
            continue
        satisfied += _margin_identity(values)
        rows.append({"timestamp": date, "symbol": f"{code}.TWO", "market": "TPEX", **values})
    return _margin_frame(rows, satisfied, "TPEx margin")


class _FlowSource:
    """One official endpoint fetched per session date through a shared client."""

    name: str
    endpoint: str
    columns: tuple[str, ...]

    def __init__(self, client: CachedJsonClient) -> None:
        self.client = client

    def _params(self, date: str) -> dict[str, str]:
        raise NotImplementedError

    def _parse(self, payload: Any, date: str) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_date(self, date: str) -> ProviderFetch:
        payload, request = self.client.get_json(self.endpoint, params=self._params(date))
        try:
            frame = self._parse(payload, date)
        except ValueError:
            if not _is_no_data_payload(payload):
                raise
            frame = pd.DataFrame(columns=["timestamp", "symbol", "market", *self.columns])
        return ProviderFetch(frame=frame, requests=(request,), metadata={"date": date})


class TWSEInstitutionalSource(_FlowSource):
    name = "twse_t86"
    endpoint = "https://www.twse.com.tw/rwd/zh/fund/T86"
    columns = FLOW_COLUMNS

    def _params(self, date: str) -> dict[str, str]:
        return {"date": date.replace("-", ""), "selectType": "ALLBUT0999", "response": "json"}

    def _parse(self, payload: Any, date: str) -> pd.DataFrame:
        return parse_twse_t86(payload, date=date)


class TWSEMarginSource(_FlowSource):
    name = "twse_margin"
    endpoint = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    columns = MARGIN_COLUMNS

    def _params(self, date: str) -> dict[str, str]:
        return {"date": date.replace("-", ""), "selectType": "ALL", "response": "json"}

    def _parse(self, payload: Any, date: str) -> pd.DataFrame:
        return parse_twse_margin(payload, date=date)


class TPExInstitutionalSource(_FlowSource):
    name = "tpex_institutional"
    endpoint = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
    columns = FLOW_COLUMNS

    def _params(self, date: str) -> dict[str, str]:
        return {"type": "Daily", "sect": "EW", "date": date.replace("-", "/"), "response": "json"}

    def _parse(self, payload: Any, date: str) -> pd.DataFrame:
        return parse_tpex_institutional(payload, date=date)


class TPExMarginSource(_FlowSource):
    name = "tpex_margin"
    endpoint = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
    columns = MARGIN_COLUMNS

    def _params(self, date: str) -> dict[str, str]:
        return {"date": date.replace("-", "/"), "response": "json"}

    def _parse(self, payload: Any, date: str) -> pd.DataFrame:
        return parse_tpex_margin(payload, date=date)
