"""TAIFEX parsing and stock-futures eligibility (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from twbacktest import taifex

HEADER = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,"
          "未沖銷契約數,最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,"
          "價差對單式委託成交量")


def _csv(rows):
    lines = [HEADER] + [
        f"{d},{c},{e}  ,10,11,9,10,0,0%,{v},10,5,10,10,12,8,,{s},,"
        for d, c, e, v, s in rows
    ]
    return ("\r\n".join(lines) + "\r\n").encode("cp950")


def _tx_csv(rows):
    """TX rows with explicit open and settlement: (date, expiry, open, settlement, session)."""
    lines = [HEADER] + [
        f"{d},TX,{e}  ,{o},{o + 10},{o - 10},{o + 5},0,0%,1000,{st},5000,{o},{o},1,1,,{s},,"
        for d, e, o, st, s in rows
    ]
    return ("\r\n".join(lines) + "\r\n").encode("cp950")


def _listing():
    def row(prefix, stock, short, board, shares):
        cells = [prefix, f"{short}公司", stock, short, "● 是股票期貨標的", "", "",
                 "◎" if board == "twse" else "", "◎" if board == "tpex" else "", "", "",
                 shares, "8:45~13:45", "-"]
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

    return ("<table><tr><th>代碼</th></tr>" + row("CD", "2330", "台積電", "twse", "2,000")
            + row("QF", "2330", "台積電", "twse", "100")
            + row("OZ", "6488", "環球晶", "tpex", "2,000") + "</table>")


def test_chunk_ranges_cover_the_interval():
    assert taifex.chunk_ranges("2025-05-01", "2025-05-17", 7) == [
        (date(2025, 5, 1), date(2025, 5, 7)), (date(2025, 5, 8), date(2025, 5, 14)),
        (date(2025, 5, 15), date(2025, 5, 16)),
    ]
    with pytest.raises(ValueError):
        taifex.chunk_ranges("2025-07-01", "2025-07-01", 7)


def test_parse_futures_keeps_columns_despite_the_trailing_comma():
    frame = taifex.parse_futures(_csv([("2025/06/02", "CDF", "202506", "5521", "一般"),
                                       ("2025/06/02", "CDF", "202506", "12", "盤後")]))
    assert frame["contract"].tolist() == ["CDF", "CDF"]
    assert frame["expiry"].tolist() == ["202506", "202506"]
    assert frame["volume"].tolist() == [5521, 12]
    assert frame["session"].tolist() == ["regular", "after_hours"]
    assert frame["date"].iloc[0] == "2025-06-02"
    with pytest.raises(ValueError):
        taifex.parse_futures("<html>error</html>".encode("cp950"))


def test_parse_underlyings_maps_codes_to_symbols():
    listing = taifex.parse_underlyings(_listing())
    assert listing.set_index("code")["symbol"].to_dict() == {
        "CDF": "2330.TW", "QFF": "2330.TW", "OZF": "6488.TWO"}
    assert listing.set_index("code")["shares"].to_dict() == {"CDF": 2000, "QFF": 100,
                                                              "OZF": 2000}


def test_eligibility_uses_the_near_month_regular_session_and_trailing_volume():
    rows = []
    for day, volume in (("2025/06/02", 300), ("2025/06/03", 0), ("2025/06/04", 0)):
        rows += [(day, "CDF", "202506", volume, "一般"), (day, "CDF", "202507", 999, "一般"),
                 (day, "CDF", "202506", 999, "盤後"), (day, "QFF", "202506", 2000, "一般")]
    rows += [("2025/06/02", "OZF", "202506", 10, "一般"),
             ("2025/06/02", "XXF", "202506", 1, "一般"),
             ("2025/06/02", "OZF", "202506/202507", 99, "一般")]
    futures = taifex.parse_futures(_csv(rows))
    daily, audit = taifex.eligibility(futures, taifex.parse_underlyings(_listing()),
                                      min_volume=150, lookback=2)
    table = daily.set_index(["date", "symbol"])
    # 300 standard + 2000 small (= 100 standard) on the first day, then only the small contract.
    assert table.loc[("2025-06-02", "2330.TW"), "standard_volume"] == 400
    assert table.loc[("2025-06-03", "2330.TW"), "mean_volume"] == 250
    assert bool(table.loc[("2025-06-03", "2330.TW"), "liquid"])
    assert not bool(table.loc[("2025-06-04", "2330.TW"), "liquid"])
    assert bool(table.loc[("2025-06-02", "2330.TW"), "has_small"])
    assert not bool(table.loc[("2025-06-02", "6488.TWO"), "liquid"])
    assert ("2025-06-03", "6488.TWO") not in table.index
    assert audit["unmatched_three_letter_codes"] == ["XXF"]
    assert isinstance(daily, pd.DataFrame)


def test_parse_futures_keeps_high_low_and_settlement():
    frame = taifex.parse_futures(_tx_csv([("2025/06/02", "202506", 21000, 21005, "一般")]))
    assert frame["high"].iloc[0] == 21010
    assert frame["low"].iloc[0] == 20990
    assert frame["settlement"].iloc[0] == 21005


def test_third_wednesday():
    assert taifex.third_wednesday(2025, 6) == date(2025, 6, 18)
    assert taifex.third_wednesday(2026, 1) == date(2026, 1, 21)
    assert taifex.third_wednesday(2025, 10) == date(2025, 10, 15)


def test_continuous_front_month_rolls_on_the_third_wednesday():
    # June 2025 expires on Wednesday 2025-06-18; the position rolls at that session's open.
    rows = [
        ("2025/06/16", "202506", 21000, 21010, "一般"),
        ("2025/06/16", "202507", 21100, 21110, "一般"),
        ("2025/06/17", "202506", 21200, 21210, "一般"),
        ("2025/06/17", "202507", 21300, 21310, "一般"),
        ("2025/06/18", "202506", 21400, 21410, "一般"),
        ("2025/06/18", "202507", 21500, 21510, "一般"),
        ("2025/06/19", "202507", 21600, 21610, "一般"),
        ("2025/06/19", "202507", 21650, 0, "盤後"),
        ("2025/06/19", "202507W4", 21600, 21610, "一般"),
    ]
    series = taifex.continuous_front_month(taifex.parse_futures(_tx_csv(rows)), "TX")
    assert series["expiry"].tolist() == ["202506", "202506", "202507", "202507"]
    assert series["roll"].tolist() == [False, False, True, False]
    # Into 06-17: June open 21000 -> 21200. Into 06-18 (roll): June open 21200 -> 21400,
    # then the July contract is held from its 21500 open. Into 06-19: July 21500 -> 21600.
    growth = series["front_growth"].to_numpy()
    assert np.isnan(growth[0])
    assert growth[1] == pytest.approx(np.log(21200 / 21000))
    assert growth[2] == pytest.approx(np.log(21400 / 21200))
    assert growth[3] == pytest.approx(np.log(21600 / 21500))
    assert series["adjusted_open"].iloc[0] == 21000
    assert series["adjusted_open"].iloc[3] == pytest.approx(
        21000 * np.exp(np.nansum(growth)))
    assert series["settlement"].tolist() == [21010, 21210, 21510, 21610]


def test_download_contract_names_and_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_post(form):
        calls.append(form)
        return _tx_csv([(form["queryStartDate"], "202506", 21000, 21005, "一般")])

    monkeypatch.setattr(taifex, "_post", fake_post)
    monkeypatch.setattr(taifex.time, "sleep", lambda seconds: None)
    manifest = taifex.download_contract(tmp_path, "TX", "2025-06-01", "2025-07-15", pause=0)
    assert [form["commodity_id"] for form in calls] == ["TX", "TX"]
    assert sorted(manifest["files"]) == ["futures-TX-20250601-20250628.csv",
                                         "futures-TX-20250629-20250714.csv"]
    frame = taifex.load_contract(tmp_path, "TX")
    assert set(frame["contract"]) == {"TX"}
    with pytest.raises(ValueError):
        taifex.download_contract(tmp_path, "all", "2025-06-01", "2025-07-15", pause=0)
