#!/usr/bin/env python3
"""Fetch the TWSE and TPEX company basic-data snapshots into one reference CSV (protocol 13.16).

Sources (official OpenAPIs, current snapshot only):
  TWSE  https://openapi.twse.com.tw/v1/opendata/t187ap03_L      (listed companies)
  TPEX  https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O   (OTC companies)

Output: configs/reference/tw_company_reference_<YYYYMMDD>.csv with columns
symbol, market, industry_code, issued_shares, par_value_twd, snapshot_date, plus a JSON
sidecar with the source URLs, record counts and the CSV's sha256. The snapshot date comes from
the records' ROC-calendar date field. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
USER_AGENT = "stock-forecasting reference snapshot (protocol 13.16)"
COLUMNS = ("symbol", "market", "industry_code", "issued_shares", "par_value_twd", "snapshot_date")


def fetch_json(url: str, timeout: int = 120) -> list[dict]:
    """One request through curl with bounded retries: Python's HTTP client is cut off
    mid-transfer by these servers (IncompleteRead, seen 2026-10-23); curl is not."""
    command = ["curl", "--silent", "--show-error", "--fail", "--max-time", str(timeout),
               "--retry", "2", "--retry-delay", "5", "--user-agent", USER_AGENT,
               "--header", "Accept: application/json", url]
    result = subprocess.run(command, capture_output=True, timeout=timeout * 3 + 30, check=False)
    if result.returncode:
        detail = result.stderr.decode()[:200]
        raise RuntimeError(f"curl exit {result.returncode} for {url}: {detail}")
    payload = json.loads(result.stdout.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{url} did not return a non-empty JSON array")
    return payload


def roc_to_iso(value: str) -> str:
    """ROC calendar YYYMMDD (e.g. 1151022) to ISO 2026-10-22."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 7:
        raise ValueError(f"Unexpected ROC date {value!r}")
    return f"{int(digits[:3]) + 1911:04d}-{digits[3:5]}-{digits[5:7]}"


def par_value(text: str) -> str:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text or "")
    return match.group(1) if match else ""


def rows_from_twse(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        code = (record.get("公司代號") or "").strip()
        if not code:
            continue
        out.append({
            "symbol": f"{code}.TW", "market": "TWSE",
            "industry_code": (record.get("產業別") or "").strip(),
            "issued_shares": re.sub(r"\D", "", record.get("已發行普通股數或TDR原股發行股數") or ""),
            "par_value_twd": par_value(record.get("普通股每股面額", "")),
            "snapshot_date": roc_to_iso(record.get("出表日期", "")),
        })
    return out


def rows_from_tpex(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        code = (record.get("SecuritiesCompanyCode") or "").strip()
        if not code:
            continue
        out.append({
            "symbol": f"{code}.TWO", "market": "TPEX",
            "industry_code": (record.get("SecuritiesIndustryCode") or "").strip(),
            "issued_shares": re.sub(r"\D", "", record.get("IssueShares") or ""),
            "par_value_twd": par_value(record.get("ParValueOfCommonStock", "")),
            "snapshot_date": roc_to_iso(record.get("Date", "")),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/reference"))
    args = parser.parse_args(argv)
    twse, tpex = fetch_json(TWSE_URL), fetch_json(TPEX_URL)
    rows = rows_from_twse(twse) + rows_from_tpex(tpex)
    dates = sorted({row["snapshot_date"] for row in rows})
    snapshot = dates[-1]
    rows.sort(key=lambda row: (row["market"], row["symbol"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"tw_company_reference_{snapshot.replace('-', '')}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "sources": {"TWSE": TWSE_URL, "TPEX": TPEX_URL},
        "snapshot_dates": dates,
        "records": {"TWSE": len(twse), "TPEX": len(tpex), "rows": len(rows)},
        "columns": list(COLUMNS),
        "sha256": digest,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"path": str(path), **sidecar}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
