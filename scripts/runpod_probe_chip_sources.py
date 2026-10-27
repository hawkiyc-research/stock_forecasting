"""Probe every tw_chip_data source from a RunPod Pod: which hosts answer a non-Taiwan IP?

Runs with the Pod image's Python (standard library only). It sends ONE request per chip table
with the same URL, parameters, User-Agent and Accept header that ``tw_chip_data`` uses, plus
two controls whose behaviour from RunPod is already known (TWSE ``MI_INDEX`` is fetched directly
by ``cpu prepare``; TPEx ``dailyQuotes`` needs the Cloud Run relay), and classifies each answer:

* ``ok``        2xx with the expected body (JSON with rows or a TWSE/TPEx status, the MOPS
                HTML table, the TDCC CSV header, the ISIN HTML table);
* ``no_data``   2xx but the exchange says there is nothing for that date (closure day, not yet
                published) — the host is reachable;
* ``blocked``   403 / 429 / 451, a connection reset, a timeout, or a 2xx HTML page that is not
                the expected table (a challenge or an interstitial);
* ``http_error`` any other status.

Hosts are probed one after another with the chip pipeline's 3-second spacing (about two
minutes in total: too small a job for a thread per host). A host with any ``blocked`` table
needs the relay; a host with only ``ok`` / ``no_data`` answers can be fetched directly.
The report is printed and written as JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "tw-chip-data/0.1 (research; low-frequency; contact via repository)"
ACCEPT = "application/json,*/*"
TIMEOUT_S = 60.0
INTERVAL_S = 3.0
BLOCKED_STATUSES = {403, 429, 451}
NO_DATA_MARKERS = ("沒有符合條件", "查詢日期小於", "查詢日期大於", "很抱歉", "查無資料")

# (table, kind, url, params, format); templates follow tw_chip_data/configs/sources.yaml.
TABLES: tuple[tuple[str, str, str, dict[str, str], str], ...] = (
    ("twse_notice", "monthly", "https://www.twse.com.tw/rwd/zh/announcement/notice",
     {"startDate": "{start_ymd}", "endDate": "{end_ymd}", "response": "json"}, "json"),
    ("twse_punish", "monthly", "https://www.twse.com.tw/rwd/zh/announcement/punish",
     {"startDate": "{start_ymd}", "endDate": "{end_ymd}", "response": "json"}, "json"),
    ("tpex_attention", "monthly", "https://www.tpex.org.tw/www/zh-tw/bulletin/attention",
     {"startDate": "{start_slash}", "endDate": "{end_slash}", "response": "json"}, "json"),
    ("tpex_disposal", "monthly", "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal",
     {"startDate": "{start_slash}", "endDate": "{end_slash}", "response": "json"}, "json"),
    ("twse_t86", "daily", "https://www.twse.com.tw/rwd/zh/fund/T86",
     {"date": "{ymd}", "selectType": "ALLBUT0999", "response": "json"}, "json"),
    ("tpex_insti", "daily", "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
     {"type": "Daily", "sect": "EW", "date": "{slash}", "response": "json"}, "json"),
    ("twse_margn", "daily", "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
     {"date": "{ymd}", "selectType": "ALL", "response": "json"}, "json"),
    ("tpex_margin", "daily", "https://www.tpex.org.tw/www/zh-tw/margin/balance",
     {"date": "{slash}", "response": "json"}, "json"),
    ("twse_twt93u", "daily", "https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U",
     {"date": "{ymd}", "response": "json"}, "json"),
    ("tpex_sbl", "daily", "https://www.tpex.org.tw/www/zh-tw/margin/sbl",
     {"date": "{slash}", "response": "json"}, "json"),
    ("twse_twtb4u", "daily", "https://www.twse.com.tw/exchangeReport/TWTB4U",
     {"response": "json", "date": "{ymd}", "selectType": "All"}, "json"),
    ("tpex_daytrade", "daily", "https://www.tpex.org.tw/www/zh-tw/intraday/stat",
     {"type": "Daily", "date": "{slash}", "response": "json"}, "json"),
    ("twse_qfiis", "daily", "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS",
     {"date": "{ymd}", "selectType": "ALLBUT0999", "response": "json"}, "json"),
    ("twse_twt92u", "daily", "https://www.twse.com.tw/rwd/zh/marginTrading/TWT92U",
     {"date": "{ymd}", "response": "json"}, "json"),
    ("tpex_mark", "daily", "https://www.tpex.org.tw/www/zh-tw/margin/mark",
     {"date": "{slash}", "response": "json"}, "json"),
    ("twse_bfi84u", "snapshot", "https://www.twse.com.tw/rwd/zh/marginTrading/BFI84U",
     {"response": "json"}, "json"),
    ("tdcc_dispersion", "snapshot", "https://smart.tdcc.com.tw/opendata/getOD.ashx",
     {"id": "1-5"}, "csv"),
    ("mops_revenue_sii", "monthly",
     "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_0.html", {}, "html"),
    ("mops_revenue_otc", "monthly",
     "https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_{roc_year}_{month}_0.html", {}, "html"),
    ("twse_company_ref", "snapshot", "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
     {}, "json"),
    ("tpex_company_ref", "snapshot", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
     {}, "json"),
    ("isin_twse", "snapshot", "https://isin.twse.com.tw/isin/C_public.jsp", {"strMode": "2"},
     "html"),
    ("twse_holiday_schedule", "snapshot",
     "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule", {}, "json"),
)
# Controls with a known answer from RunPod: the dataset pipeline fetches MI_INDEX directly and
# needs the relay for TPEx dailyQuotes.
CONTROLS: tuple[tuple[str, str, str, dict[str, str], str], ...] = (
    ("control_twse_mi_index_direct_ok", "daily",
     "https://www.twse.com.tw/exchangeReport/MI_INDEX",
     {"response": "json", "date": "{ymd}", "type": "ALLBUT0999"}, "json"),
    ("control_tpex_daily_quotes_needs_relay", "daily",
     "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
     {"date": "{slash}", "id": "", "response": "json"}, "json"),
)


def render(url: str, params: dict[str, str], day: dt.date) -> tuple[str, dict[str, str]]:
    """Fill the sources.yaml templates for one calendar day (monthly tables use its month)."""
    first = day.replace(day=1)
    values = {
        "ymd": day.strftime("%Y%m%d"),
        "slash": day.strftime("%Y/%m/%d"),
        "start_ymd": first.strftime("%Y%m%d"),
        "end_ymd": day.strftime("%Y%m%d"),
        "start_slash": first.strftime("%Y/%m/%d"),
        "end_slash": day.strftime("%Y/%m/%d"),
        "roc_year": str(day.year - 1911),
        "month": str(day.month),
    }
    return url.format(**values), {k: v.format(**values) for k, v in params.items()}


def classify(fmt: str, status: int, body: bytes) -> tuple[str, str]:
    """(verdict, detail) for one answer; never raises on odd bodies."""
    if status in BLOCKED_STATUSES:
        return "blocked", f"HTTP {status}"
    if status < 200 or status >= 300:
        return "http_error", f"HTTP {status}"
    head = body[:4000]
    if fmt == "json":
        try:
            payload = json.loads(body.decode("utf-8-sig", errors="replace"))
        except ValueError:
            text = head.decode("utf-8", errors="replace")
            if "<html" in text.lower():
                return "blocked", "2xx but an HTML page instead of JSON"
            return "http_error", "2xx but not JSON"
        if isinstance(payload, list):
            return "ok", f"JSON array, {len(payload)} rows"
        if isinstance(payload, dict):
            stat = str(payload.get("stat", ""))
            rows = payload.get("data") or payload.get("tables") or payload.get("aaData")
            count = len(rows) if isinstance(rows, list) else 0
            if stat and stat.upper() != "OK" and any(m in stat for m in NO_DATA_MARKERS):
                return "no_data", stat
            if stat.upper() == "OK" or count:
                return "ok", f"stat={stat or '-'} rows={count}"
            return "no_data", f"stat={stat or '-'} rows=0"
        return "http_error", "2xx but an unexpected JSON type"
    text = body.decode("big5", errors="replace") if fmt == "html" else body.decode(
        "utf-8-sig", errors="replace"
    )
    if fmt == "html":
        if "公司代號" in text or "有價證券代號" in text:
            return "ok", f"HTML table, {len(body)} bytes"
        if any(m in text for m in NO_DATA_MARKERS) or "Not Found" in text:
            return "no_data", "placeholder page"
        return "blocked", f"2xx but not the expected HTML table: {_snippet(text)}"
    first_line = text.splitlines()[0] if text.strip() else ""
    if "證券代號" in first_line or "資料日期" in first_line or first_line.count(",") >= 3:
        return "ok", f"CSV, {len(text.splitlines())} lines"
    return "blocked", f"2xx but not the expected CSV: {_snippet(text)}"


def _snippet(text: str, limit: int = 80) -> str:
    return " ".join(text.split())[:limit]


def fetch(url: str, params: dict[str, str]) -> tuple[int, bytes, str]:
    """(status, body, error) for one GET; network failures come back as status 0."""
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": ACCEPT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, response.read(), ""
    except urllib.error.HTTPError as error:
        return error.code, error.read() if error.fp else b"", ""
    except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as error:
        return 0, b"", f"{type(error).__name__}: {error}"


def probe(day: dt.date, *, interval: float, fetcher=fetch, tables=TABLES + CONTROLS) -> list[dict]:
    rows = []
    last_by_host: dict[str, float] = {}
    for table, kind, template, params, fmt in tables:
        url, rendered = render(template, params, day)
        host = urllib.parse.urlsplit(url).hostname or ""
        wait = interval - (time.monotonic() - last_by_host.get(host, -interval))
        if wait > 0:
            time.sleep(wait)
        started = time.monotonic()
        status, body, error = fetcher(url, rendered)
        last_by_host[host] = time.monotonic()
        if error:
            verdict, detail = "blocked", error
        else:
            verdict, detail = classify(fmt, status, body)
        rows.append(
            {
                "table": table,
                "kind": kind,
                "host": host,
                "url": url,
                "params": rendered,
                "status": status,
                "verdict": verdict,
                "detail": detail,
                "bytes": len(body),
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        print(f"{verdict:10} {host:22} {table:36} HTTP {status:<3} {detail}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    hosts: dict[str, dict] = {}
    for row in rows:
        if row["table"].startswith("control_"):
            continue
        entry = hosts.setdefault(
            row["host"], {"ok": 0, "no_data": 0, "blocked": 0, "http_error": 0}
        )
        entry[row["verdict"]] += 1
    for entry in hosts.values():
        entry["route"] = "relay" if entry["blocked"] else "direct"
    return hosts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date(2026, 10, 20),
                        help="trading day to request (default 2026-10-20)")
    parser.add_argument("--interval", type=float, default=INTERVAL_S,
                        help="seconds between requests to the same host")
    parser.add_argument("--output", type=Path,
                        default=Path("/runpod-volume/diagnostics/chip-source-probe"),
                        help="directory that receives probe-<timestamp>.json")
    args = parser.parse_args(argv)
    rows = probe(args.date, interval=args.interval)
    hosts = summarize(rows)
    print("\nRoute per host (any blocked table => relay):")
    for host, entry in sorted(hosts.items()):
        print(f"  {entry['route']:6} {host:22} ok={entry['ok']} no_data={entry['no_data']} "
              f"blocked={entry['blocked']} http_error={entry['http_error']}")
    controls = {r["table"]: r["verdict"] for r in rows if r["table"].startswith("control_")}
    print(f"Controls: {controls}")
    if controls.get("control_twse_mi_index_direct_ok") == "blocked":
        print("WARNING: the known-good TWSE control is blocked too; this Pod may have no egress "
              "at all, do not read the table above as geo-blocking", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    report = args.output / f"probe-{stamp}.json"
    report.write_text(json.dumps({"date": args.date.isoformat(), "rows": rows, "hosts": hosts,
                                  "controls": controls}, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
