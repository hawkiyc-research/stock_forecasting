"""Cloud tests for auxiliary Taiwan flow, margin, TAIFEX and EODHD research data."""

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from stock_forecasting.data import auxiliary
from stock_forecasting.data.providers.base import RequestRecord
from stock_forecasting.data.providers.taifex import (
    CachedFormCsvClient,
    csv_as_json_response,
    parse_futures_institutional,
    parse_put_call_ratio,
)
from stock_forecasting.data.providers.taiwan_flows import (
    TWSE_MARGIN_FIELDS,
    parse_tpex_institutional,
    parse_tpex_margin,
    parse_twse_margin,
    parse_twse_t86,
)

T86_FIELDS = [
    "證券代號",
    "證券名稱",
    "外陸資買進股數(不含外資自營商)",
    "外陸資賣出股數(不含外資自營商)",
    "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數",
    "外資自營商賣出股數",
    "外資自營商買賣超股數",
    "投信買進股數",
    "投信賣出股數",
    "投信買賣超股數",
    "自營商買賣超股數",
    "自營商買進股數(自行買賣)",
    "自營商賣出股數(自行買賣)",
    "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)",
    "自營商賣出股數(避險)",
    "自營商買賣超股數(避險)",
    "三大法人買賣超股數",
]
TPEX_MARGIN = [
    "代號",
    "名稱",
    "前資餘額(張)",
    "資買",
    "資賣",
    "現償",
    "資餘額",
    "資屬證金",
    "資使用率(%)",
    "資限額",
    "前券餘額(張)",
    "券賣",
    "券買",
    "券償",
    "券餘額",
    "券屬證金",
    "券使用率(%)",
    "券限額",
    "資券相抵(張)",
    "備註",
]


def t86_row(code, foreign, trust, dealer, total=None):
    total = foreign + trust + dealer if total is None else total
    return [
        code,
        "名稱",
        "0",
        "0",
        f"{foreign:,}",
        "0",
        "0",
        "0",
        "0",
        "0",
        f"{trust:,}",
        f"{dealer:,}",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        f"{total:,}",
    ]


def test_twse_t86_maps_named_columns_and_checks_the_total():
    payload = {
        "stat": "OK",
        "fields": T86_FIELDS,
        "data": [
            t86_row("2317", 46_741_199, 1_070_960, 16_141_524),
            t86_row("030001", 5, 0, 0),  # Warrant codes are not research instruments.
        ],
    }
    frame = parse_twse_t86(payload, date="2021-01-04")
    assert frame["symbol"].tolist() == ["2317.TW"]
    assert frame.loc[0, "institutional_net_shares"] == 63_953_683
    assert frame.loc[0, "trust_net_shares"] == 1_070_960
    broken = {"stat": "OK", "fields": T86_FIELDS, "data": [t86_row("2317", 1, 1, 1, total=9)]}
    with pytest.raises(ValueError, match="identity"):
        parse_twse_t86(broken, date="2021-01-04")


def test_twse_margin_requires_the_exact_layout_and_balance_identity():
    fields = list(TWSE_MARGIN_FIELDS)
    row = [
        "2330",
        "台積電",
        "66",
        "28",
        "0",
        "487",
        "525",
        "1000",
        "19",
        "0",
        "0",
        "205",
        "186",
        "1000",
        "0",
        "",
    ]
    payload = {"tables": [{"fields": ["項目"], "data": []}, {"fields": fields, "data": [row]}]}
    frame = parse_twse_margin(payload, date="2021-01-04")
    assert frame.loc[0, "margin_balance_lots"] == 525
    assert frame.loc[0, "short_buy_lots"] == 19 and frame.loc[0, "short_balance_lots"] == 186
    reordered = {
        "tables": [{"fields": [*fields[:2], fields[3], fields[2], *fields[4:]], "data": [row]}]
    }
    with pytest.raises(ValueError):
        parse_twse_margin(reordered, date="2021-01-04")


def tpex_row(code, groups, total):
    values = []
    for net in groups:
        values += ["0", "0", f"{net:,}"]
    return [code, "名稱", *values, f"{total:,}"]


def test_tpex_institutional_positions_are_verified_by_three_identities():
    fields = [
        "代號",
        "名稱",
        *(["買進股數", "賣出股數", "買賣超股數"] * 7),
        "三大法人買賣超股數合計",
    ]
    good = tpex_row("6488", [100, 5, 105, 20, 7, 3, 10], 135)
    frame = parse_tpex_institutional(
        {"tables": [{"fields": fields, "data": [good]}]}, date="2021-01-04"
    )
    assert frame.loc[0, "symbol"] == "6488.TWO"
    assert (frame.loc[0, "foreign_net_shares"], frame.loc[0, "dealer_net_shares"]) == (100, 10)
    shifted = tpex_row("6488", [100, 5, 20, 105, 7, 3, 10], 135)
    with pytest.raises(ValueError, match="identity"):
        parse_tpex_institutional(
            {"tables": [{"fields": fields, "data": [shifted]}]}, date="2021-01-04"
        )
    with pytest.raises(ValueError, match="layout"):
        parse_tpex_institutional(
            {"tables": [{"fields": fields[:-2] + fields[-1:], "data": []}]}, date="2021-01-04"
        )


def test_tpex_margin_maps_named_columns():
    row = [
        "6488",
        "環球晶",
        "178",
        "10",
        "4",
        "1",
        "183",
        "0",
        "0.2",
        "1000",
        "20",
        "6",
        "2",
        "0",
        "24",
        "0",
        "0.1",
        "1000",
        "1",
        "",
    ]
    frame = parse_tpex_margin(
        {"tables": [{"fields": TPEX_MARGIN, "data": [row]}]}, date="2021-01-04"
    )
    assert frame.loc[0, "margin_balance_lots"] == 183
    assert frame.loc[0, "short_balance_lots"] == 24


FUTURES_CSV = (
    "日期,商品名稱,身份別,多方交易口數,多方交易契約金額(千元),空方交易口數,空方交易契約金額(千元),"
    "多空交易口數淨額,多空交易契約金額淨額(千元),多方未平倉口數,多方未平倉契約金額(千元),"
    "空方未平倉口數,空方未平倉契約金額(千元),多空未平倉口數淨額,多空未平倉契約金額淨額(千元)\n"
    "2021/01/04,臺股期貨,自營商,1,1,1,1,0,0,10,1,12,1,-2,0\n"
    "2021/01/04,臺股期貨,外資及陸資,5,5,3,3,2,2,40,4,30,3,10,1\n"
    "2021/01/04,電子期貨,外資及陸資,5,5,3,3,2,2,40,4,30,3,10,1\n"
)


def csv_response(text, encoding="cp950", status=200):
    response = requests.Response()
    response.status_code = status
    response._content = text.encode(encoding)
    return response


def test_taifex_csv_is_wrapped_parsed_by_header_and_html_is_rejected():
    wrapped = csv_as_json_response(csv_response(FUTURES_CSV))
    payload = json.loads(wrapped.content)
    assert payload["encoding"] == "cp950"
    frame = parse_futures_institutional(payload, commodity_prefix="臺股期貨")
    assert frame["investor"].tolist() == ["dealer", "foreign"]
    assert frame.loc[1, "net_open_interest"] == 10
    html = csv_as_json_response(csv_response("<html>查詢錯誤</html>"))
    with pytest.raises(json.JSONDecodeError):
        json.loads(html.content)
    with pytest.raises(ValueError, match="header changed"):
        parse_futures_institutional(
            {"format": "csv", "text": "日期,商品名稱\n2021/01/04,臺股期貨\n"},
            commodity_prefix="臺股期貨",
        )
    ratio = parse_put_call_ratio(
        {
            "format": "csv",
            "text": (
                "日期,賣權成交量,買權成交量,買賣權成交量比率%,賣權未平倉量,買權未平倉量,買賣權未平倉量比率%\n"
                "2021/01/04,100,80,125.00,500,400,125.00\n"
            ),
        }
    )
    assert ratio.loc[0, "timestamp"] == "2021-01-04"
    assert ratio.loc[0, "open_interest_ratio_percent"] == 125.0


class _PostSession:
    def __init__(self):
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs["data"]))
        return csv_response(FUTURES_CSV)


def test_form_client_posts_once_and_reuses_the_cache(tmp_path):
    session = _PostSession()
    client = CachedFormCsvClient(
        provider="taifex_auxiliary",
        raw_cache_root=tmp_path,
        max_requests_per_second=100.0,
        session=session,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )
    params = {"queryStartDate": "2021/01/01", "queryEndDate": "2021/01/31", "commodityId": "TXF"}
    first, record = client.get_json(
        "https://www.taifex.com.tw/cht/3/futContractsDateDown", params=params
    )
    second, cached = client.get_json(
        "https://www.taifex.com.tw/cht/3/futContractsDateDown", params=params
    )
    assert len(session.calls) == 1 and session.calls[0][1] == params
    assert first == second and not record.cache_hit and cached.cache_hit


class _FakeClient:
    """Serve fixture payloads per endpoint without any network access."""

    def __init__(self, payloads, failing=()):
        self.payloads, self.failing = payloads, failing

    def get_json(self, endpoint, *, params):
        if any(name in endpoint for name in self.failing):
            raise ValueError("layout changed")
        record = RequestRecord("fake", "0" * 64, "x.json", "0" * 64, 1, False, "now")
        return self.payloads(endpoint, params), record


def test_prepare_auxiliary_data_writes_tables_and_isolates_failures(tmp_path):
    t86 = {"stat": "OK", "fields": T86_FIELDS, "data": [t86_row("2317", 10, 5, 1)]}
    holiday = {"stat": "很抱歉, 沒有符合條件的資料!"}

    def payloads(endpoint, params):
        if "T86" in endpoint:
            return holiday if params["date"] == "20210105" else t86
        if "taifex" in endpoint:
            return json.loads(csv_as_json_response(csv_response(FUTURES_CSV)).content)
        raise AssertionError(endpoint)

    options = auxiliary.AuxiliaryOptions(
        start="2021-01-04",
        end="2021-01-06",
        output_root=tmp_path / "aux",
        raw_cache_root=tmp_path / "cache",
        sources=("twse_t86", "twse_margin", "taifex_futures", "eodhd_series"),
    )
    manifest = auxiliary.prepare_auxiliary_data(
        options,
        json_client_factory=lambda **_: _FakeClient(payloads, failing=("MI_MARGN",)),
        form_client_factory=lambda **_: _FakeClient(payloads),
        session_dates=lambda start, end: ["2021-01-04", "2021-01-05"],
    )
    sources = manifest["sources"]
    assert sources["twse_t86"]["status"] == "complete"
    assert sources["twse_t86"]["empty_sessions"] == ["2021-01-05"]
    assert sources["twse_margin"]["status"] == "failed"
    assert sources["taifex_futures"]["status"] == "complete"
    assert sources["eodhd_series"]["status"] == "failed"  # No token configured.
    table = pd.read_parquet(tmp_path / "aux" / "twse_t86.parquet")
    assert table["symbol"].tolist() == ["2317.TW"]
    saved = json.loads((tmp_path / "aux" / "auxiliary-manifest.json").read_text())
    assert saved["sources"]["twse_t86"]["publication"] == {"local_time": "16:30", "day_offset": 0}
    assert not Path(tmp_path / "aux" / "twse_margin.parquet").exists()


def test_options_reject_unknown_sources_and_half_relay_credentials(tmp_path):
    common = {
        "start": "2021-01-04",
        "end": "2021-02-01",
        "output_root": tmp_path,
        "raw_cache_root": tmp_path,
    }
    with pytest.raises(ValueError, match="Unknown"):
        auxiliary.AuxiliaryOptions(**common, sources=("twse_t86", "other"))
    with pytest.raises(ValueError, match="together"):
        auxiliary.AuxiliaryOptions(**common, tpex_proxy_url="https://relay.a.run.app")
    assert auxiliary._month_ranges("2021-01-15", "2021-03-01") == [
        ("2021-01-15", "2021-01-31"),
        ("2021-02-01", "2021-02-28"),
    ]


def test_table_writer_keeps_a_stable_numeric_schema(tmp_path):
    writer = auxiliary._TableWriter(tmp_path / "flows.parquet")
    # The first session has a value column without any value; later sessions have integers.
    writer.write(
        pd.DataFrame({"timestamp": ["2021-01-04"], "symbol": ["2330.TW"], "offset_lots": [None]})
    )
    writer.write(
        pd.DataFrame({"timestamp": ["2021-01-05"], "symbol": ["2330.TW"], "offset_lots": [12]})
    )
    record = writer.close()
    table = pd.read_parquet(tmp_path / "flows.parquet")
    assert record["rows"] == 2
    assert str(table["offset_lots"].dtype) == "float64"
    assert table["offset_lots"].isna().tolist() == [True, False]
    assert table["offset_lots"].iloc[1] == 12.0
    assert table["symbol"].tolist() == ["2330.TW", "2330.TW"]
