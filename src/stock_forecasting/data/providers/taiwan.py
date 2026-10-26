"""Official TWSE and TPEx market-wide daily quote providers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from stock_forecasting.data.adjustments import apply_cumulative_adjustments
from stock_forecasting.data.benchmarks import is_allowlisted_taiwan_equity_etf
from stock_forecasting.data.schema import normalize_ohlcv_frame

from .base import Instrument, ProviderFetch, RequestRecord
from .http import CachedJsonClient

_ETF_CODE = re.compile(r"^00[0-9A-Z]{2,6}$")
_STOCK_CODE = re.compile(r"^[1-9][0-9]{3}$")
_DEPOSITARY_RECEIPT_CODE = re.compile(r"^91(?:[0-9]{2}|[0-9]{4})$")
_PREFERRED_STOCK_CODE = re.compile(r"^[1-9][0-9]{3}[A-E]$")
_RIGHTS_CERTIFICATE_CODE = re.compile(r"^[1-9][0-9]{3}[L-Z]$")
_SIX_DIGIT_NUMERIC_SECURITY_CODE = re.compile(r"^[0-9]{6}$")


def _clean_field(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).replace("\u3000", "")


def _number(value: Any) -> float | None:
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "---", "N/A", "nan", "None"}:
        return None
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def _gregorian_date(value: Any) -> str:
    """Normalize Gregorian or ROC dates returned by Taiwan exchanges."""

    text = str(value).strip().replace("年", "/").replace("月", "/").replace("日", "")
    compact = re.sub(r"[^0-9]", "", text)
    parts = [part for part in re.split(r"[^0-9]+", text) if part]
    if len(parts) == 3:
        year = int(parts[0]) + (1911 if len(parts[0]) <= 3 else 0)
        return f"{year:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    if len(compact) == 6:
        roc_year = int(compact[:2])
        if roc_year < 1:
            raise ValueError(f"Unsupported Taiwan date value: {value!r}")
        return f"{roc_year + 1911:04d}-{compact[2:4]}-{compact[4:6]}"
    if len(compact) == 7:
        return f"{int(compact[:3]) + 1911:04d}-{compact[3:5]}-{compact[5:7]}"
    if len(compact) == 8:
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    raise ValueError(f"Unsupported Taiwan date value: {value!r}")


def _field_index(fields: list[Any], aliases: set[str]) -> int:
    normalized = [_clean_field(field) for field in fields]
    try:
        return next(index for index, field in enumerate(normalized) if field in aliases)
    except StopIteration as error:
        raise ValueError("Official Taiwan response is missing a required field") from error


def _first_table(payload: Any) -> tuple[list[Any], list[list[Any]]]:
    for table in _tables(payload):
        fields = table.get("fields")
        data = table.get("data")
        if isinstance(fields, list) and isinstance(data, list):
            return fields, [row for row in data if isinstance(row, list)]
    raise ValueError("Official Taiwan response has no data table")


def _is_no_data_payload(payload: Any) -> bool:
    """Recognize official empty-result responses without hiding schema drift."""

    if not isinstance(payload, dict):
        return False
    status = str(payload.get("stat", payload.get("status", "")))
    tables = payload.get("tables")
    data = payload.get("data")
    return (
        "沒有符合條件" in status
        or "無相關資料" in status
        or "很抱歉" in status
        or "no data" in status.lower()
        or tables == []
        or data == []
    )


def _index_frame(
    price_payload: Any,
    return_payload: Any,
    *,
    price_aliases: dict[str, set[str]],
    return_date_aliases: set[str],
    return_value_aliases: set[str],
    symbol: str,
    provider: str,
    market: str,
    source_symbol: str,
    dataset_profile: str,
) -> pd.DataFrame:
    price_fields, price_rows = _first_table(price_payload)
    price_indexes = {
        field: _field_index(price_fields, aliases) for field, aliases in price_aliases.items()
    }
    return_fields, return_rows = _first_table(return_payload)
    return_date_index = _field_index(return_fields, return_date_aliases)
    return_value_index = _field_index(return_fields, return_value_aliases)
    total_return_by_date = {
        _gregorian_date(row[return_date_index]): _number(row[return_value_index])
        for row in return_rows
        if len(row) > max(return_date_index, return_value_index)
    }
    rows: list[dict[str, Any]] = []
    for raw in price_rows:
        if len(raw) <= max(price_indexes.values()):
            continue
        trading_date = _gregorian_date(raw[price_indexes["timestamp"]])
        prices = {
            field: _number(raw[price_indexes[field]]) for field in ("open", "high", "low", "close")
        }
        total_return = total_return_by_date.get(trading_date)
        if total_return is None or any(value is None for value in prices.values()):
            continue
        rows.append(
            {
                "timestamp": trading_date,
                "symbol": symbol,
                "asset_type": "index",
                **prices,
                "volume": 0.0,
                "adjusted_close": total_return,
                "split_adjusted_volume": 0.0,
                "adjustment_source": "official_total_return_index",
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": source_symbol,
                "is_active": True,
                "dataset_profile": dataset_profile,
            }
        )
    if not rows:
        return pd.DataFrame()
    return normalize_ohlcv_frame(pd.DataFrame(rows))


def _tables(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Taiwan official endpoint did not return a JSON object")
    tables = payload.get("tables")
    output = (
        [table for table in tables if isinstance(table, dict)] if isinstance(tables, list) else []
    )
    for key, fields in payload.items():
        if not key.startswith("fields") or not isinstance(fields, list):
            continue
        suffix = key.removeprefix("fields")
        data = payload.get(f"data{suffix}")
        if isinstance(data, list):
            output.append({"fields": fields, "data": data})
    return output


def _find_quote_table(
    payload: Any,
    aliases: dict[str, set[str]],
) -> tuple[list[str], list[list[Any]]]:
    required = set(aliases)
    for table in _tables(payload):
        fields = table.get("fields")
        data = table.get("data")
        if not isinstance(fields, list) or not isinstance(data, list):
            continue
        normalized = [_clean_field(field) for field in fields]
        mapped = {
            canonical: next(
                (index for index, field in enumerate(normalized) if field in names),
                None,
            )
            for canonical, names in aliases.items()
        }
        if set(key for key, index in mapped.items() if index is not None) != required:
            continue
        return normalized, [row for row in data if isinstance(row, list)]
    raise ValueError("Official Taiwan response has no compatible OHLCV quote table")


def _asset_type(code: str, *, market: str) -> str | None:
    if _DEPOSITARY_RECEIPT_CODE.fullmatch(code):
        return "stock"
    if _ETF_CODE.fullmatch(code):
        suffix = "TW" if market == "TWSE" else "TWO"
        return (
            "etf"
            if is_allowlisted_taiwan_equity_etf(
                symbol=f"{code}.{suffix}",
                market=market,
            )
            else None
        )
    if _STOCK_CODE.fullmatch(code):
        return "stock"
    return None


def _unsupported_action_type(code: str) -> str:
    if _ETF_CODE.fullmatch(code):
        return "non_allowlisted_etf_code"
    if _PREFERRED_STOCK_CODE.fullmatch(code):
        return "preferred_stock_code"
    if _RIGHTS_CERTIFICATE_CODE.fullmatch(code):
        return "rights_certificate_code"
    if _SIX_DIGIT_NUMERIC_SECURITY_CODE.fullmatch(code):
        return "six_digit_numeric_security_code"
    return "unsupported_symbol_code"


_TPEX_DETAIL_SHARE_RATE = re.compile(r"每壹仟股換發新股票[^0-9]*([0-9]+(?:\.[0-9]+)?)")

# TPEx par-value changes (變更股票面額). The TPEx website publishes these only as
# per-company announcements, not as a queryable table, so the resumption dates
# below come from the price-discontinuity scan of the official daily quotes
# (first session after the 5-7 session share-exchange halt) and the par values
# from the TPEx company basic data (configs/reference/tw_company_reference_*.csv).
# The reference price on the first session equals the last close scaled by
# new par / old par, so price_factor = new / old and share_multiplier = old / new.
# Code, first session of the new shares, old par (TWD), new par (TWD).
_TPEX_PAR_VALUE_CHANGES: tuple[tuple[str, str, float, float], ...] = (
    ("6548", "2019-09-09", 10.0, 1.0),
    ("6613", "2022-08-29", 10.0, 5.0),
    ("6548", "2022-09-05", 1.0, 0.4),
    ("5536", "2022-09-19", 10.0, 5.0),
    ("3093", "2022-12-12", 10.0, 2.5),
    ("6763", "2024-09-09", 10.0, 1.0),
    ("8932", "2024-09-09", 10.0, 5.0),
    ("5278", "2024-12-09", 10.0, 1.0),
    ("5314", "2025-03-31", 10.0, 0.5),
    ("8932", "2026-03-09", 5.0, 2.5),
    ("8937", "2026-04-13", 10.0, 2.5),
    ("3086", "2026-04-20", 10.0, 1.0),
)


def _tpex_par_value_action_rows(*, start: str, end: str) -> list[dict[str, Any]]:
    """Return the curated TPEx par-value changes inside [start, end] as action rows."""

    rows: list[dict[str, Any]] = []
    for code, resumption_date, old_par, new_par in _TPEX_PAR_VALUE_CHANGES:
        if resumption_date < start or resumption_date > end:
            continue
        if _asset_type(code, market="TPEX") is None:
            continue
        rows.append(
            {
                "timestamp": resumption_date,
                "symbol": f"{code}.TWO",
                "price_factor": new_par / old_par,
                "share_multiplier": old_par / new_par,
                "source": "tpex_par_value_reference",
            }
        )
    return rows


def _tpex_detail_number(detail: Any, pattern: re.Pattern[str]) -> float | None:
    """Read one numeric field out of the HTML detail cell of a TPEx resumption row."""

    match = pattern.search(str(detail))
    return _number(match.group(1)) if match else None


def _resumption_action_rows(
    payload: Any,
    *,
    market: str,
    suffix: str,
    source: str,
    date_aliases: set[str],
    symbol_aliases: set[str],
    prior_close_aliases: set[str],
    reference_price_aliases: set[str],
    share_rate_policy: str,
    detail_aliases: set[str] | None = None,
    kind_aliases: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse one official "resume trading at a reference price" table.

    Capital reductions (cash refunds and loss-covering reductions), par-value
    changes and ETF splits halt a security for several sessions and resume it
    at an official reference price. The resumption date is the action date and
    ``price_factor`` is reference / last close before the halt, exactly like
    the ex-dividend tables, so ``apply_cumulative_adjustments`` multiplies every
    earlier ``adjusted_close`` by it on top of the dividend factors.

    ``share_rate_policy`` chooses the share multiplier (new shares per old share):

    * ``"price_ratio"``: the official formula is reference = prior / rate, so
      rate = prior / reference. Used for par-value changes, ETF splits and
      loss-covering reductions.
    * ``"tpex_detail"``: rate = 每壹仟股換發新股票 / 1000 from the detail cell;
      rows without that field fall back to the price ratio and say so in
      ``source``.

    * ``"twse_reduction"``: loss-covering rows (彌補虧損) follow the price-ratio
      formula; cash-refund rows (退還股款) cannot separate the refund from the
      share rate without a detail page, so they use the price ratio and record
      ``_share_rate_from_price`` in ``source``. The share multiplier only
      scales ``split_adjusted_volume``, never prices.
    """

    fields, data = _first_table(payload)
    indexes = {
        "timestamp": _field_index(fields, date_aliases),
        "symbol": _field_index(fields, symbol_aliases),
        "prior_close": _field_index(fields, prior_close_aliases),
        "reference_price": _field_index(fields, reference_price_aliases),
    }
    detail_index = (
        _field_index(fields, detail_aliases) if detail_aliases is not None else None
    )
    kind_index = _field_index(fields, kind_aliases) if kind_aliases is not None else None
    rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "actions": 0,
        "dropped_rows": 0,
        "pending_rows": 0,
        "share_rate_from_price": 0,
        "skipped_unsupported_action_rows": 0,
        "skipped_unsupported_action_types": Counter(),
    }
    for raw in data:
        required_length = max(
            index
            for index in (*indexes.values(), detail_index, kind_index)
            if index is not None
        )
        if len(raw) <= required_length:
            counts["dropped_rows"] += 1
            continue
        symbol_code = str(raw[indexes["symbol"]]).strip().upper()
        if _asset_type(symbol_code, market=market) is None:
            counts["skipped_unsupported_action_rows"] += 1
            counts["skipped_unsupported_action_types"][
                _unsupported_action_type(symbol_code)
            ] += 1
            continue
        prior_close = _number(raw[indexes["prior_close"]])
        reference_price = _number(raw[indexes["reference_price"]])
        if prior_close is None or reference_price is None:
            # Announced resumptions list the prices only once they are known.
            counts["pending_rows"] += 1
            continue
        if prior_close <= 0.0 or reference_price <= 0.0:
            counts["dropped_rows"] += 1
            continue
        price_factor = reference_price / prior_close
        if not np.isfinite(price_factor) or price_factor <= 0.0:
            counts["dropped_rows"] += 1
            continue
        row_source = source
        share_multiplier = prior_close / reference_price
        if share_rate_policy == "tpex_detail":
            detail = raw[detail_index] if detail_index is not None else ""
            share_rate = _tpex_detail_number(detail, _TPEX_DETAIL_SHARE_RATE)
            if share_rate is not None and share_rate > 0.0:
                share_multiplier = share_rate / 1000.0
            else:
                counts["share_rate_from_price"] += 1
                row_source = f"{source}_share_rate_from_price"
        elif share_rate_policy == "twse_reduction":
            kind = str(raw[kind_index]) if kind_index is not None else ""
            if "退還" in kind:
                counts["share_rate_from_price"] += 1
                row_source = f"{source}_share_rate_from_price"
        elif share_rate_policy != "price_ratio":
            raise ValueError(f"Unsupported share-rate policy: {share_rate_policy}")
        rows.append(
            {
                "timestamp": _gregorian_date(raw[indexes["timestamp"]]),
                "symbol": f"{symbol_code}.{suffix}",
                "price_factor": price_factor,
                "share_multiplier": share_multiplier,
                "source": row_source,
            }
        )
    counts["actions"] = len(rows)
    return rows, counts


def _parse_quotes(
    payload: Any,
    *,
    aliases: dict[str, set[str]],
    date: str,
    provider: str,
    market: str,
    suffix: str,
    dataset_profile: str,
) -> tuple[pd.DataFrame, int]:
    fields, data = _find_quote_table(payload, aliases)
    indexes = {
        canonical: next(index for index, field in enumerate(fields) if field in names)
        for canonical, names in aliases.items()
    }
    rows: list[dict[str, Any]] = []
    dropped = 0
    for row in data:
        if len(row) <= max(indexes.values()):
            dropped += 1
            continue
        code = str(row[indexes["symbol"]]).strip().upper()
        asset_type = _asset_type(code, market=market)
        prices = {
            field: _number(row[indexes[field]])
            for field in ("open", "high", "low", "close")
        }
        volume = _number(row[indexes["volume"]])
        if (
            asset_type is None
            or volume is None
            or volume < 0.0
            or any(value is None or value <= 0.0 for value in prices.values())
        ):
            dropped += 1
            continue
        rows.append(
            {
                "timestamp": date,
                "symbol": f"{code}.{suffix}",
                "asset_type": asset_type,
                **prices,
                "volume": volume,
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": code,
                # The daily quote endpoint does not prove current listing status.
                "is_active": pd.NA,
                "dataset_profile": dataset_profile,
            }
        )
    if not rows:
        return pd.DataFrame(), dropped
    return normalize_ohlcv_frame(pd.DataFrame(rows)), dropped


class _TaiwanMarketProvider:
    name: str
    market: str
    suffix: str
    endpoint: str
    action_adjustment_source: str
    aliases: dict[str, set[str]]

    def __init__(self, client: CachedJsonClient) -> None:
        self.client = client

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
        del include_delisted
        return [], ()

    def _params(self, date: str) -> dict[str, Any]:
        raise NotImplementedError

    def fetch_date(self, *, date: str, dataset_profile: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            self.endpoint,
            params=self._params(date),
        )
        try:
            frame, dropped = _parse_quotes(
                payload,
                aliases=self.aliases,
                date=date,
                provider=self.name,
                market=self.market,
                suffix=self.suffix,
                dataset_profile=dataset_profile,
            )
        except ValueError:
            if _is_no_data_payload(payload):
                frame, dropped = pd.DataFrame(), 0
            else:
                raise
        return ProviderFetch(
            frame=frame,
            requests=(request,),
            metadata={"dropped_rows": dropped, "date": date},
        )

    def fetch_adjusted_date(
        self,
        *,
        date: str,
        dataset_profile: str,
        action_frame: pd.DataFrame,
    ) -> tuple[ProviderFetch, pd.DataFrame]:
        """Fetch one session and apply the provider's durable action factors."""

        fetched = self.fetch_date(
            date=date,
            dataset_profile=dataset_profile,
        )
        if fetched.frame.empty:
            return fetched, fetched.frame
        adjusted = apply_cumulative_adjustments(fetched.frame, action_frame)
        adjusted["adjustment_source"] = self.action_adjustment_source
        return fetched, adjusted

    def fetch_instrument(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
        dataset_profile: str,
    ) -> ProviderFetch:
        del instrument, start, end, dataset_profile
        raise NotImplementedError("Taiwan providers fetch market-wide data by trading date")


class TWSEProvider(_TaiwanMarketProvider):
    name = "twse_official"
    market = "TWSE"
    suffix = "TW"
    endpoint = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    # Resume-trading reference prices: capital reductions (TWTAUU), par-value
    # changes (TWTB8U) and ETF splits / reverse splits (TWTCAU).
    capital_reduction_endpoint = "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU"
    par_value_change_endpoint = "https://www.twse.com.tw/rwd/zh/change/TWTB8U"
    etf_split_endpoint = "https://www.twse.com.tw/rwd/zh/split/TWTCAU"
    action_adjustment_source = "twse_twt49u+twtauu+twtb8u+twtcau"
    aliases: ClassVar[dict[str, set[str]]] = {
        "symbol": {"證券代號"},
        "open": {"開盤價"},
        "high": {"最高價"},
        "low": {"最低價"},
        "close": {"收盤價"},
        "volume": {"成交股數"},
    }

    def _params(self, date: str) -> dict[str, Any]:
        return {
            "response": "json",
            "date": date.replace("-", ""),
            "type": "ALLBUT0999",
        }

    def fetch_resumption_actions(self, *, start: str, end: str) -> ProviderFetch:
        """Fetch capital reductions, par-value changes and ETF splits as action factors."""

        params = {
            "response": "json",
            "startDate": start.replace("-", ""),
            "endDate": end.replace("-", ""),
        }
        specs = (
            (
                self.capital_reduction_endpoint,
                "twse_twtauu",
                {"股票代號"},
                "twse_reduction",
                {"減資原因"},
                "capital_reduction_actions",
            ),
            (
                self.par_value_change_endpoint,
                "twse_twtb8u",
                {"股票代號"},
                "price_ratio",
                None,
                "par_value_change_actions",
            ),
            (
                self.etf_split_endpoint,
                "twse_twtcau",
                {"ETF代號"},
                "price_ratio",
                None,
                "etf_split_actions",
            ),
        )
        rows: list[dict[str, Any]] = []
        requests: list[RequestRecord] = []
        metadata: dict[str, Any] = {
            "dropped_rows": 0,
            "pending_rows": 0,
            "share_rate_from_price": 0,
            "skipped_unsupported_action_rows": 0,
            "skipped_unsupported_action_types": Counter(),
        }
        for endpoint, source, symbol_aliases, policy, kind_aliases, key in specs:
            payload, request = self.client.get_json(endpoint, params=params)
            requests.append(request)
            if _is_no_data_payload(payload):
                metadata[key] = 0
                continue
            parsed, counts = _resumption_action_rows(
                payload,
                market=self.market,
                suffix=self.suffix,
                source=source,
                date_aliases={"恢復買賣日期"},
                symbol_aliases=symbol_aliases,
                prior_close_aliases={"停止買賣前收盤價格"},
                reference_price_aliases={"恢復買賣參考價"},
                share_rate_policy=policy,
                kind_aliases=kind_aliases,
            )
            rows.extend(parsed)
            metadata[key] = counts["actions"]
            for name in (
                "dropped_rows",
                "pending_rows",
                "share_rate_from_price",
                "skipped_unsupported_action_rows",
            ):
                metadata[name] += counts[name]
            metadata["skipped_unsupported_action_types"].update(
                counts["skipped_unsupported_action_types"]
            )
        metadata["skipped_unsupported_action_types"] = dict(
            sorted(metadata["skipped_unsupported_action_types"].items())
        )
        metadata["corporate_actions"] = len(rows)
        return ProviderFetch(
            frame=pd.DataFrame(rows),
            requests=tuple(requests),
            metadata=metadata,
        )

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            "https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
            params={
                "response": "json",
                "startDate": start.replace("-", ""),
                "endDate": end.replace("-", ""),
            },
        )
        fields, data = _first_table(payload)
        indexes = {
            "timestamp": _field_index(fields, {"資料日期"}),
            "symbol": _field_index(fields, {"股票代號"}),
            "prior_close": _field_index(fields, {"除權息前收盤價"}),
            "reference_price": _field_index(fields, {"除權息參考價"}),
            "kind": _field_index(fields, {"權/息"}),
        }
        rows: list[dict[str, Any]] = []
        requests: list[RequestRecord] = [request]
        dropped = 0
        missing_share_multiplier_details = 0
        skipped_unsupported_action_rows = 0
        skipped_unsupported_action_types: Counter[str] = Counter()
        for raw in data:
            if len(raw) <= max(indexes.values()):
                dropped += 1
                continue
            symbol_code = str(raw[indexes["symbol"]]).strip().upper()
            if _asset_type(symbol_code, market=self.market) is None:
                skipped_unsupported_action_rows += 1
                skipped_unsupported_action_types[_unsupported_action_type(symbol_code)] += 1
                continue
            prior_close = _number(raw[indexes["prior_close"]])
            reference_price = _number(raw[indexes["reference_price"]])
            if prior_close is None or reference_price is None or prior_close <= 0.0:
                dropped += 1
                continue
            effective_date = _gregorian_date(raw[indexes["timestamp"]])
            share_multiplier = 1.0
            action_source = "twse_twt49u"
            if "權" in str(raw[indexes["kind"]]):
                detail, detail_request = self.client.get_json(
                    "https://www.twse.com.tw/rwd/zh/exRight/TWT49UDetail",
                    params={
                        "response": "json",
                        "STK_NO": symbol_code,
                        "T1": effective_date.replace("-", ""),
                    },
                )
                requests.append(detail_request)
                if _is_no_data_payload(detail):
                    missing_share_multiplier_details += 1
                    action_source = "twse_twt49u_price_only_missing_detail"
                else:
                    detail_fields, detail_rows = _first_table(detail)
                    try:
                        free_share_index = _field_index(
                            detail_fields,
                            {"A.按普通股股東持股比例每千股無償配股"},
                        )
                    except ValueError:
                        free_share_index = None
                    if (
                        free_share_index is not None
                        and detail_rows
                        and len(detail_rows[0]) > free_share_index
                    ):
                        free_shares = _number(
                            re.sub(
                                r"[^0-9.+-]",
                                "",
                                str(detail_rows[0][free_share_index]),
                            )
                        )
                        if free_shares is not None and free_shares >= 0.0:
                            share_multiplier += free_shares / 1000.0
                        else:
                            missing_share_multiplier_details += 1
                            action_source = "twse_twt49u_price_only_missing_detail"
                    else:
                        missing_share_multiplier_details += 1
                        action_source = "twse_twt49u_price_only_missing_detail"
            price_factor = reference_price / prior_close
            if not np.isfinite(price_factor) or price_factor <= 0.0:
                dropped += 1
                continue
            rows.append(
                {
                    "timestamp": effective_date,
                    "symbol": f"{symbol_code}.TW",
                    "price_factor": price_factor,
                    "share_multiplier": share_multiplier,
                    "source": action_source,
                }
            )
        resumption = self.fetch_resumption_actions(start=start, end=end)
        requests.extend(resumption.requests)
        skipped_unsupported_action_types.update(
            resumption.metadata["skipped_unsupported_action_types"]
        )
        frame = pd.concat(
            [pd.DataFrame(rows), resumption.frame],
            ignore_index=True,
        )
        return ProviderFetch(
            frame=frame,
            requests=tuple(requests),
            metadata={
                "corporate_actions": len(frame),
                "ex_dividend_actions": len(rows),
                "capital_reduction_actions": resumption.metadata["capital_reduction_actions"],
                "par_value_change_actions": resumption.metadata["par_value_change_actions"],
                "etf_split_actions": resumption.metadata["etf_split_actions"],
                "pending_resumption_rows": resumption.metadata["pending_rows"],
                "resumption_share_rate_from_price": resumption.metadata[
                    "share_rate_from_price"
                ],
                "dropped_rows": dropped + resumption.metadata["dropped_rows"],
                "missing_share_multiplier_details": missing_share_multiplier_details,
                "skipped_unsupported_action_rows": (
                    skipped_unsupported_action_rows
                    + resumption.metadata["skipped_unsupported_action_rows"]
                ),
                "skipped_unsupported_action_types": dict(
                    sorted(skipped_unsupported_action_types.items())
                ),
            },
        )

    def fetch_benchmark_month(self, *, month: str, dataset_profile: str) -> ProviderFetch:
        date_value = month.replace("-", "")[:6] + "01"
        price_payload, price_request = self.client.get_json(
            "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
            params={"response": "json", "date": date_value},
        )
        return_payload, return_request = self.client.get_json(
            "https://www.twse.com.tw/indicesReport/MFI94U",
            params={"response": "json", "date": date_value},
        )
        frame = _index_frame(
            price_payload,
            return_payload,
            price_aliases={
                "timestamp": {"日期", "日日期"},
                "open": {"開盤指數"},
                "high": {"最高指數"},
                "low": {"最低指數"},
                "close": {"收盤指數"},
            },
            return_date_aliases={"日期", "日日期"},
            return_value_aliases={"發行量加權股價報酬指數"},
            symbol="TAIEX.TW",
            provider=self.name,
            market=self.market,
            source_symbol="TAIEX",
            dataset_profile=dataset_profile,
        )
        return ProviderFetch(
            frame=frame,
            requests=(price_request, return_request),
            metadata={"month": month, "benchmark": "TAIEX.TW"},
        )


class TPExProvider(_TaiwanMarketProvider):
    name = "tpex_official"
    market = "TPEX"
    suffix = "TWO"
    endpoint = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
    # Resume-trading reference prices after capital reductions; the detail
    # cell carries 每壹仟股換發新股票 (share rate) and 每股退還股款 (cash refund).
    capital_reduction_endpoint = "https://www.tpex.org.tw/www/zh-tw/bulletin/revivt"
    action_adjustment_source = "tpex_exdailyq+revivt"
    aliases: ClassVar[dict[str, set[str]]] = {
        "symbol": {"代號", "證券代號"},
        "open": {"開盤", "開盤價"},
        "high": {"最高", "最高價"},
        "low": {"最低", "最低價"},
        "close": {"收盤", "收盤價"},
        "volume": {"成交股數", "成交量"},
    }

    def _params(self, date: str) -> dict[str, Any]:
        return {
            "date": date.replace("-", "/"),
            "id": "",
            "response": "json",
        }

    def fetch_resumption_actions(self, *, start: str, end: str) -> ProviderFetch:
        """Fetch TPEx capital reductions as action factors."""

        payload, request = self.client.get_json(
            self.capital_reduction_endpoint,
            params={
                "startDate": start.replace("-", "/"),
                "endDate": end.replace("-", "/"),
                "response": "json",
            },
        )
        if _is_no_data_payload(payload):
            rows: list[dict[str, Any]] = []
            counts: dict[str, Any] = {
                "actions": 0,
                "dropped_rows": 0,
                "pending_rows": 0,
                "share_rate_from_price": 0,
                "skipped_unsupported_action_rows": 0,
                "skipped_unsupported_action_types": Counter(),
            }
        else:
            rows, counts = _resumption_action_rows(
                payload,
                market=self.market,
                suffix=self.suffix,
                source="tpex_revivt",
                date_aliases={"恢復買賣日期"},
                symbol_aliases={"股票代號", "代號"},
                prior_close_aliases={"最後交易日之收盤價格"},
                reference_price_aliases={"減資恢復買賣開始日參考價格"},
                share_rate_policy="tpex_detail",
                detail_aliases={"詳細資料"},
            )
        par_value_rows = _tpex_par_value_action_rows(start=start, end=end)
        rows.extend(par_value_rows)
        return ProviderFetch(
            frame=pd.DataFrame(rows),
            requests=(request,),
            metadata={
                "corporate_actions": len(rows),
                "capital_reduction_actions": counts["actions"],
                "par_value_change_actions": len(par_value_rows),
                "dropped_rows": counts["dropped_rows"],
                "pending_rows": counts["pending_rows"],
                "share_rate_from_price": counts["share_rate_from_price"],
                "skipped_unsupported_action_rows": counts["skipped_unsupported_action_rows"],
                "skipped_unsupported_action_types": dict(
                    sorted(counts["skipped_unsupported_action_types"].items())
                ),
            },
        )

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ",
            params={
                "startDate": start.replace("-", "/"),
                "endDate": end.replace("-", "/"),
                "response": "json",
            },
        )
        fields, data = _first_table(payload)
        indexes = {
            "timestamp": _field_index(fields, {"除權息日期"}),
            "symbol": _field_index(fields, {"代號"}),
            "prior_close": _field_index(fields, {"除權息前收盤價"}),
            "reference_price": _field_index(fields, {"除權息參考價"}),
            "free_shares": _field_index(fields, {"每仟股無償配股"}),
        }
        rows: list[dict[str, Any]] = []
        dropped = 0
        skipped_unsupported_action_rows = 0
        skipped_unsupported_action_types: Counter[str] = Counter()
        for raw in data:
            if len(raw) <= max(indexes.values()):
                dropped += 1
                continue
            symbol_code = str(raw[indexes["symbol"]]).strip().upper()
            if _asset_type(symbol_code, market=self.market) is None:
                skipped_unsupported_action_rows += 1
                skipped_unsupported_action_types[_unsupported_action_type(symbol_code)] += 1
                continue
            prior_close = _number(raw[indexes["prior_close"]])
            reference_price = _number(raw[indexes["reference_price"]])
            free_shares = _number(raw[indexes["free_shares"]])
            if (
                prior_close is None
                or reference_price is None
                or free_shares is None
                or prior_close <= 0.0
                or free_shares < 0.0
            ):
                dropped += 1
                continue
            price_factor = reference_price / prior_close
            if not np.isfinite(price_factor) or price_factor <= 0.0:
                dropped += 1
                continue
            rows.append(
                {
                    "timestamp": _gregorian_date(raw[indexes["timestamp"]]),
                    "symbol": f"{symbol_code}.TWO",
                    "price_factor": price_factor,
                    "share_multiplier": 1.0 + free_shares / 1000.0,
                    "source": "tpex_exdailyq",
                }
            )
        resumption = self.fetch_resumption_actions(start=start, end=end)
        skipped_unsupported_action_types.update(
            resumption.metadata["skipped_unsupported_action_types"]
        )
        frame = pd.concat(
            [pd.DataFrame(rows), resumption.frame],
            ignore_index=True,
        )
        return ProviderFetch(
            frame=frame,
            requests=(request, *resumption.requests),
            metadata={
                "corporate_actions": len(frame),
                "ex_dividend_actions": len(rows),
                "capital_reduction_actions": resumption.metadata["capital_reduction_actions"],
                "par_value_change_actions": resumption.metadata["par_value_change_actions"],
                "pending_resumption_rows": resumption.metadata["pending_rows"],
                "resumption_share_rate_from_price": resumption.metadata[
                    "share_rate_from_price"
                ],
                "dropped_rows": dropped + resumption.metadata["dropped_rows"],
                "skipped_unsupported_action_rows": (
                    skipped_unsupported_action_rows
                    + resumption.metadata["skipped_unsupported_action_rows"]
                ),
                "skipped_unsupported_action_types": dict(
                    sorted(skipped_unsupported_action_types.items())
                ),
            },
        )

    def fetch_benchmark_month(self, *, month: str, dataset_profile: str) -> ProviderFetch:
        query_date = month[:7].replace("-", "/") + "/01"
        price_payload, price_request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/indexInfo/inx",
            params={"date": query_date, "response": "json"},
        )
        return_payload, return_request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/indexInfo/ROE",
            params={"date": query_date, "response": "json"},
        )
        frame = _index_frame(
            price_payload,
            return_payload,
            price_aliases={
                "timestamp": {"日期"},
                "open": {"開市"},
                "high": {"最高"},
                "low": {"最低"},
                "close": {"收市"},
            },
            return_date_aliases={"日期"},
            return_value_aliases={"櫃買報酬指數(基期:94/12/30)"},
            symbol="TPEX.TWO",
            provider=self.name,
            market=self.market,
            source_symbol="TPEX",
            dataset_profile=dataset_profile,
        )
        return ProviderFetch(
            frame=frame,
            requests=(price_request, return_request),
            metadata={"month": month, "benchmark": "TPEX.TWO"},
        )
