"""TAIFEX stock-futures eligibility for the capturability analysis (protocol section 13.9).

Two public sources are used: the daily futures trading data (download form, one week per
request, Big5 CSV) and the current stock futures underlying list (HTML). A stock futures
contract code is the list's product prefix plus "F" (prefix CD -> contract CDF, stock
2330); small contracts (100 shares) have their own prefix. TAIFEX is an external public
service, so requests are sequential and paced, and every response is kept on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import subprocess
import time
import urllib.parse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DOWNLOAD_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
LIST_URL = "https://www.taifex.com.tw/cht/2/stockLists"
USER_AGENT = "Mozilla/5.0 (stock-forecasting capturability analysis)"
TIMEOUT_SECONDS = 120
# A month in one response is cut off mid-transfer; a week is about 1 MB.
CHUNK_DAYS = 7
ATTEMPTS = 3
STANDARD_SHARES = 2000
FUTURES_COLUMNS = {
    "交易日期": "date",
    "契約": "contract",
    "到期月份(週別)": "expiry",
    "開盤價": "open",
    "收盤價": "close",
    "成交量": "volume",
    "未沖銷契約數": "open_interest",
    "交易時段": "session",
}
LIST_COLUMNS = ("prefix", "name", "stock", "short", "is_futures", "is_option", "is_weekly",
                "twse", "tpex", "twse_etf", "tpex_etf", "shares", "hours", "after_hours")
BOARD_SUFFIX = {"twse": ".TW", "tpex": ".TWO", "twse_etf": ".TW", "tpex_etf": ".TWO"}
BACKTEST_ROOT = Path(__file__).resolve().parents[1]


def chunk_ranges(start: str, end_exclusive: str, days: int) -> list[tuple[date, date]]:
    """Consecutive inclusive query ranges of at most ``days`` calendar days."""
    first, stop = date.fromisoformat(start), date.fromisoformat(end_exclusive)
    if first >= stop or days < 1:
        raise ValueError("start must precede end_exclusive and days must be positive")
    ranges = []
    while first < stop:
        following = min(first + timedelta(days=days), stop)
        ranges.append((first, following - timedelta(days=1)))
        first = following
    return ranges


def _curl(arguments: list[str]) -> bytes:
    """One request through curl (bounded retries). Python's HTTP client is cut off
    mid-transfer on some of these responses (seen 2026-10-20); curl is not."""
    command = ["curl", "--silent", "--show-error", "--fail", "--max-time", str(TIMEOUT_SECONDS),
               "--retry", str(ATTEMPTS - 1), "--retry-delay", "5", "--user-agent", USER_AGENT,
               *arguments]
    result = subprocess.run(command, capture_output=True, timeout=TIMEOUT_SECONDS * ATTEMPTS + 30,
                            check=False)
    if result.returncode:
        raise TransferError(f"curl exit {result.returncode}: {result.stderr.decode()[:200]}")
    return result.stdout


class TransferError(RuntimeError):
    """A TAIFEX request failed after its retries."""


def _post(form: dict[str, str]) -> bytes:
    return _curl(["--request", "POST", "--data", urllib.parse.urlencode(form), DOWNLOAD_URL])


def _get(url: str) -> bytes:
    return _curl([url])


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(content)
    temporary.replace(path)


def _fetch_range(root: Path, first: date, last: date, *, pause: float) -> dict[str, str]:
    """Download one inclusive date range unless it is on disk; returns its file digest."""
    path = root / "raw" / f"futures-{first:%Y%m%d}-{last:%Y%m%d}.csv"
    if not path.is_file():
        content = _post({
            "down_type": "1", "commodity_id": "all", "commodity_id2": "",
            "queryStartDate": f"{first:%Y/%m/%d}", "queryEndDate": f"{last:%Y/%m/%d}",
        })
        parse_futures(content)  # refuse to keep an error page; holidays return the header only
        _write_once(path, content)
        time.sleep(pause)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def download(root: Path, start: str, end_exclusive: str, *, pause: float) -> dict[str, Any]:
    """Fetch missing weeks and today's underlying list; existing files are kept."""
    files = {}
    for first, last in chunk_ranges(start, end_exclusive, CHUNK_DAYS):
        week = root / "raw" / f"futures-{first:%Y%m%d}-{last:%Y%m%d}.csv"
        if week.is_file():
            files[week.name] = hashlib.sha256(week.read_bytes()).hexdigest()
            continue
        try:
            files.update(_fetch_range(root, first, last, pause=pause))
        except (TransferError, ValueError):
            # A week that keeps failing is fetched one day at a time.
            day = first
            while day <= last:
                files.update(_fetch_range(root, day, day, pause=pause))
                day += timedelta(days=1)
    listing = root / "raw" / f"stock-lists-{datetime.now(UTC):%Y%m%d}.html"
    if not listing.is_file():
        content = _get(LIST_URL)
        parse_underlyings(content.decode("utf-8"))
        _write_once(listing, content)
    files[listing.name] = hashlib.sha256(listing.read_bytes()).hexdigest()
    manifest = {"source": {"futures": DOWNLOAD_URL, "list": LIST_URL}, "start": start,
                "end_exclusive": end_exclusive, "files": files,
                "downloaded_at": datetime.now(UTC).isoformat()}
    (root / "download.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def parse_futures(content: bytes) -> pd.DataFrame:
    """Daily rows of every futures contract; rows end with a comma, so no index column."""
    text = content.decode("cp950")
    if not text.lstrip().startswith("交易日期"):
        raise ValueError("TAIFEX futures download did not return the expected CSV")
    frame = pd.read_csv(io.StringIO(text), dtype=str, index_col=False)
    frame.columns = [column.strip() for column in frame.columns]
    missing = set(FUTURES_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"TAIFEX futures CSV lacks columns: {sorted(missing)}")
    frame = frame[list(FUTURES_COLUMNS)].rename(columns=FUTURES_COLUMNS)
    for column in ("contract", "expiry", "session"):
        frame[column] = frame[column].str.strip()
    frame["date"] = pd.to_datetime(frame["date"].str.strip(), format="%Y/%m/%d").dt.strftime(
        "%Y-%m-%d")
    for column in ("open", "close", "volume", "open_interest"):
        frame[column] = pd.to_numeric(frame[column].str.strip().replace("-", None),
                                      errors="coerce")
    frame["session"] = frame["session"].map({"一般": "regular", "盤後": "after_hours"})
    return frame


def parse_underlyings(text: str) -> pd.DataFrame:
    """Rows of the underlying list with the contract code, stock symbol and contract size."""
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip()
                 for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)]
        if len(cells) == len(LIST_COLUMNS):
            rows.append(cells)
    if not rows:
        raise ValueError("TAIFEX underlying list has no rows")
    frame = pd.DataFrame(rows, columns=LIST_COLUMNS)
    frame = frame[frame["is_futures"].str.startswith("●")].copy()
    boards = frame[list(BOARD_SUFFIX)].apply(lambda row: [b for b in BOARD_SUFFIX if row[b]],
                                              axis=1)
    if (boards.map(len) != 1).any():
        raise ValueError("An underlying has no single board")
    frame["board"] = boards.map(lambda values: values[0])
    frame["symbol"] = frame["stock"] + frame["board"].map(BOARD_SUFFIX)
    frame["shares"] = pd.to_numeric(frame["shares"].str.replace(",", ""), errors="raise")
    frame["code"] = frame["prefix"] + "F"
    return frame[["code", "prefix", "stock", "symbol", "short", "board", "shares"]].reset_index(
        drop=True)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((root / "download.json").read_text())
    futures, listing = [], None
    for name, digest in sorted(manifest["files"].items()):
        path = root / "raw" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"TAIFEX file changed after download: {path}")
        if name.startswith("futures-"):
            futures.append(parse_futures(path.read_bytes()))
        else:
            listing = parse_underlyings(path.read_text(encoding="utf-8"))
    if listing is None or not futures:
        raise ValueError("TAIFEX download is incomplete")
    return pd.concat(futures, ignore_index=True), listing


def eligibility(futures: pd.DataFrame, listing: pd.DataFrame, *, min_volume: float,
                lookback: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per (date, symbol): a listed regular-session contract and near-month liquidity.

    Near-month volume counts the nearest monthly expiry of each contract (weeklies and
    spreads excluded) in standard-contract equivalents; a stock is liquid when its mean over
    the last ``lookback`` sessions up to the signal date reaches ``min_volume``.
    """
    regular = futures[(futures["session"] == "regular") & futures["expiry"].str.fullmatch(r"\d{6}")]
    stock = regular[regular["contract"].isin(set(listing["code"]))]
    near = stock.loc[stock.groupby(["date", "contract"])["expiry"].transform("min")
                     == stock["expiry"]]
    near = near.merge(listing[["code", "symbol", "shares"]], left_on="contract", right_on="code")
    near["standard_volume"] = near["volume"].fillna(0) * near["shares"] / STANDARD_SHARES
    daily = (near.groupby(["date", "symbol"])
             .agg(standard_volume=("standard_volume", "sum"),
                  has_standard=("shares", lambda s: bool((s == STANDARD_SHARES).any())),
                  has_small=("shares", lambda s: bool((s < STANDARD_SHARES).any())))
             .reset_index())
    dates = sorted(daily["date"].unique())
    wide = daily.pivot(index="date", columns="symbol", values="standard_volume").reindex(dates)
    rolling = wide.fillna(0).rolling(lookback, min_periods=1).mean()
    daily = daily.merge(rolling.stack().rename("mean_volume").reset_index(),
                        on=["date", "symbol"], how="left")
    daily["liquid"] = daily["mean_volume"] >= min_volume
    codes = set(regular["contract"].unique())
    unmatched = sorted(code for code in codes if re.fullmatch(r"[A-Z0-9]{2}F", code)
                       and code not in set(listing["code"]))
    audit = {"dates": len(dates), "listed_codes": len(set(stock["contract"])),
             "list_codes": int(listing["code"].nunique()),
             "unmatched_three_letter_codes": unmatched,
             "symbols_per_date": float(daily.groupby("date")["symbol"].nunique().mean())}
    return daily, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download",))
    parser.add_argument("--config", type=Path,
                        default=BACKTEST_ROOT / "config" / "capturability.json")
    parser.add_argument("--artifacts-root", type=Path, default=BACKTEST_ROOT.parent / "artifacts")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    settings = config["taifex"]
    root = args.artifacts_root / config["inputs"]["taifex"]
    manifest = download(root, settings["start"], settings["end_exclusive"],
                        pause=float(settings["pause_seconds"]))
    futures, listing = load(root)
    _daily, audit = eligibility(
        futures, listing,
        min_volume=float(config["segments"]["liquid_futures_min_volume"]),
        lookback=int(config["segments"]["liquid_futures_lookback_sessions"]),
    )
    print(json.dumps({"files": len(manifest["files"]), **audit}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
