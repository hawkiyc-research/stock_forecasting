"""The chip-source probe renders the registry templates, classifies answers like the chip
pipeline and routes a host to the relay as soon as one of its tables is blocked."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_probe_chip_sources.py"
spec = importlib.util.spec_from_file_location("runpod_probe_chip_sources", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_render_fills_every_template_the_registry_uses() -> None:
    day = dt.date(2026, 10, 20)
    url, params = module.render(
        "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_0.html", {}, day
    )
    assert url == "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_115_10_0.html" and params == {}
    _, params = module.render(
        "https://www.tpex.org.tw/www/zh-tw/bulletin/attention",
        {"startDate": "{start_slash}", "endDate": "{end_slash}", "response": "json"},
        day,
    )
    assert params == {"startDate": "2026/10/01", "endDate": "2026/10/20", "response": "json"}
    _, params = module.render("https://x/T86", {"date": "{ymd}", "s": "{start_ymd}"}, day)
    assert params == {"date": "20261020", "s": "20261001"}


def test_classify_matches_the_chip_pipeline_verdicts() -> None:
    ok_twse = json.dumps({"stat": "OK", "data": [[1], [2]], "total": 2}).encode()
    ok_tpex = json.dumps({"stat": "ok", "tables": [{"data": [1]}]}).encode()
    no_data = json.dumps({"stat": "查詢日期大於可查詢最大日期", "total": 0}).encode()
    assert module.classify("json", 200, ok_twse) == ("ok", "stat=OK rows=2")
    assert module.classify("json", 200, ok_tpex)[0] == "ok"
    assert module.classify("json", 200, no_data)[0] == "no_data"
    assert module.classify("json", 200, b"[{}, {}]") == ("ok", "JSON array, 2 rows")
    assert module.classify("json", 403, b"") == ("blocked", "HTTP 403")
    assert module.classify("json", 429, b"") == ("blocked", "HTTP 429")
    assert module.classify("json", 500, b"") == ("http_error", "HTTP 500")
    assert module.classify("json", 200, b"<html><body>challenge</body></html>")[0] == "blocked"
    assert module.classify("html", 200, "公司代號 公司名稱".encode("big5"))[0] == "ok"
    assert module.classify("html", 200, "查無資料".encode("big5"))[0] == "no_data"
    assert module.classify("html", 200, b"<html>please verify</html>")[0] == "blocked"
    csv = "﻿資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%\n20261022,0050,1,0,0,0\n"
    assert module.classify("csv", 200, csv.encode("utf-8"))[0] == "ok"
    assert module.classify("csv", 200, b"<html>blocked</html>")[0] == "blocked"


def test_probe_routes_a_host_to_the_relay_when_any_table_is_blocked() -> None:
    def fetcher(url: str, params: dict[str, str]) -> tuple[int, bytes, str]:
        if "tpex.org.tw" in url:
            return 403, b"", ""
        if "tdcc" in url:
            return 0, b"", "TimeoutError: timed out"
        if url.endswith(".html"):
            return 200, "公司代號".encode("big5"), ""
        return 200, json.dumps({"stat": "OK", "data": [1]}).encode(), ""

    rows = module.probe(dt.date(2026, 10, 20), interval=0.0, fetcher=fetcher)
    hosts = module.summarize(rows)
    assert hosts["www.tpex.org.tw"]["route"] == "relay"
    assert hosts["smart.tdcc.com.tw"]["route"] == "relay"
    assert hosts["www.twse.com.tw"]["route"] == "direct"
    assert hosts["mopsov.twse.com.tw"]["route"] == "direct"
    assert "control_twse_mi_index_direct_ok" not in hosts  # controls are reported, not routed
    controls = {r["table"]: r["verdict"] for r in rows if r["table"].startswith("control_")}
    assert controls == {
        "control_twse_mi_index_direct_ok": "ok",
        "control_tpex_daily_quotes_needs_relay": "blocked",
    }
    assert all(set(r) >= {"table", "host", "status", "verdict", "detail", "seconds"} for r in rows)
