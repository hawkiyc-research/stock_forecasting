#!/usr/bin/env python3
"""Fetch the official Taiwan resume-trading reference prices into one reference CSV.

Sources (the same endpoints and parsers the data pipeline uses through
``TWSEProvider.fetch_resumption_actions`` / ``TPExProvider.fetch_resumption_actions``):
  TWSE  https://www.twse.com.tw/rwd/zh/reducation/TWTAUU   capital reductions (減資恢復買賣參考價)
  TWSE  https://www.twse.com.tw/rwd/zh/change/TWTB8U       par-value changes (變更股票面額)
  TWSE  https://www.twse.com.tw/rwd/zh/split/TWTCAU        ETF splits / reverse splits
  TPEx  https://www.tpex.org.tw/www/zh-tw/bulletin/revivt  capital reductions (減資恢復買賣)

Output: configs/reference/tw_capital_actions_<YYYYMMDD>.csv with columns
timestamp, symbol, market, price_factor, share_multiplier, source, plus a JSON sidecar with
the source URLs, the providers' parse counts, the CSV's sha256 and the fetch time. The CSV is
documentation and the input of scripts/detect_tw_price_discontinuities.py; the data pipeline
fetches the tables itself during CPU prepare. Needs the stock_forecasting package (pandas).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_forecasting.data.providers.base import RequestRecord  # noqa: E402
from stock_forecasting.data.providers.taiwan import TPExProvider, TWSEProvider  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)
COLUMNS = ("timestamp", "symbol", "market", "price_factor", "share_multiplier", "source")


class CurlJsonClient:
    """Minimal ``get_json`` through curl: no cache, bounded retries, browser headers."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self.urls: list[str] = []

    def get_json(self, endpoint: str, *, params: dict) -> tuple[object, RequestRecord]:
        url = endpoint + "?" + urllib.parse.urlencode(params)
        referer = "https://www.tpex.org.tw/" if "tpex.org.tw" in endpoint else "https://www.twse.com.tw/"
        command = [
            "curl", "--silent", "--show-error", "--fail", "--max-time", str(self.timeout),
            "--retry", "2", "--retry-delay", "5", "--user-agent", USER_AGENT,
            "--header", "Accept: application/json,text/plain,*/*",
            "--header", f"Referer: {referer}", url,
        ]
        result = subprocess.run(
            command, capture_output=True, timeout=self.timeout * 3 + 30, check=False
        )
        if result.returncode:
            raise RuntimeError(f"curl exit {result.returncode} for {url}: {result.stderr[:200]!r}")
        self.urls.append(url)
        payload = json.loads(result.stdout.decode("utf-8"))
        record = RequestRecord(
            provider="reference-snapshot",
            request_sha256=hashlib.sha256(url.encode("utf-8")).hexdigest(),
            cache_relative_path="",
            response_sha256=hashlib.sha256(result.stdout).hexdigest(),
            response_size_bytes=len(result.stdout),
            cache_hit=False,
            requested_at=datetime.now(UTC).isoformat(),
        )
        return payload, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2016-01-01", help="first resumption date, ISO")
    parser.add_argument("--end", default=datetime.now(UTC).strftime("%Y-12-31"),
                        help="last resumption date, ISO (default: end of this year)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default configs/reference/tw_capital_actions_<today>.csv)")
    args = parser.parse_args()
    today = datetime.now(UTC).strftime("%Y%m%d")
    output = args.output or ROOT / "configs" / "reference" / f"tw_capital_actions_{today}.csv"

    client = CurlJsonClient()
    frames = []
    counts = {}
    for provider in (TWSEProvider(client), TPExProvider(client)):
        fetched = provider.fetch_resumption_actions(start=args.start, end=args.end)
        frame = fetched.frame.copy()
        frame["market"] = provider.market
        frames.append(frame)
        counts[provider.name] = fetched.metadata
    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    table["timestamp"] = pd.to_datetime(table["timestamp"]).dt.strftime("%Y-%m-%d")
    table = table.sort_values(["timestamp", "symbol"], kind="stable").loc[:, list(COLUMNS)]
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.10g", lineterminator="\n")
    sidecar = {
        "sources": client.urls,
        "range": {"start": args.start, "end": args.end},
        "counts": counts,
        "rows": len(table),
        "rows_by_source": {k: int(v) for k, v in table.groupby("source").size().items()},
        "columns": list(COLUMNS),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(table)} actions -> {output}")
    for name, metadata in counts.items():
        print(f"  {name}: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
