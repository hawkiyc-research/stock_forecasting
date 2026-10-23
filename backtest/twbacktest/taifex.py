"""TAIFEX daily futures data: stock-futures eligibility (protocol section 13.9) and the
continuous front-month index futures series used by the strategy layer (section 13.14).

Two public sources are used: the daily futures trading data (download form, one week per
request for every contract or four weeks for a single contract, Big5 CSV) and the current
stock futures underlying list (HTML). A stock futures contract code is the list's product
prefix plus "F" (prefix CD -> contract CDF, stock 2330); small contracts (100 shares) have
their own prefix. TAIFEX is an external public service, so requests are sequential and paced,
and every response is kept on disk.
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

import numpy as np
import pandas as pd

DOWNLOAD_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
LIST_URL = "https://www.taifex.com.tw/cht/2/stockLists"
USER_AGENT = "Mozilla/5.0 (stock-forecasting capturability analysis)"
TIMEOUT_SECONDS = 120
# A month of every contract in one response is cut off mid-transfer; a week is about 1 MB.
CHUNK_DAYS = 7
# One contract's rows are small enough for four weeks per request.
CONTRACT_CHUNK_DAYS = 28
ATTEMPTS = 3
STANDARD_SHARES = 2000
FUTURES_COLUMNS = {
    "交易日期": "date",
    "契約": "contract",
    "到期月份(週別)": "expiry",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "成交量": "volume",
    "結算價": "settlement",
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


def _raw_name(commodity: str, first: date, last: date) -> str:
    prefix = "futures" if commodity == "all" else f"futures-{commodity}"
    return f"{prefix}-{first:%Y%m%d}-{last:%Y%m%d}.csv"


def _fetch_range(
    root: Path, first: date, last: date, *, pause: float, commodity: str = "all"
) -> dict[str, str]:
    """Download one inclusive date range unless it is on disk; returns its file digest."""
    path = root / "raw" / _raw_name(commodity, first, last)
    if not path.is_file():
        content = _post({
            "down_type": "1", "commodity_id": commodity, "commodity_id2": "",
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


def download_contract(
    root: Path, commodity: str, start: str, end_exclusive: str, *, pause: float
) -> dict[str, Any]:
    """Fetch one contract's daily rows (e.g. TX) in four-week chunks; existing files are kept."""
    if commodity == "all" or not re.fullmatch(r"[A-Z0-9]{2,3}", commodity):
        raise ValueError("download_contract needs one TAIFEX commodity code such as TX")
    files = {}
    for first, last in chunk_ranges(start, end_exclusive, CONTRACT_CHUNK_DAYS):
        chunk = root / "raw" / _raw_name(commodity, first, last)
        if chunk.is_file():
            files[chunk.name] = hashlib.sha256(chunk.read_bytes()).hexdigest()
            continue
        try:
            files.update(_fetch_range(root, first, last, pause=pause, commodity=commodity))
        except (TransferError, ValueError):
            for week_first, week_last in chunk_ranges(
                first.isoformat(), (last + timedelta(days=1)).isoformat(), CHUNK_DAYS
            ):
                files.update(_fetch_range(root, week_first, week_last, pause=pause,
                                          commodity=commodity))
    manifest = {"source": DOWNLOAD_URL, "commodity": commodity, "start": start,
                "end_exclusive": end_exclusive, "files": files,
                "downloaded_at": datetime.now(UTC).isoformat()}
    (root / f"download-{commodity}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_contract(root: Path, commodity: str) -> pd.DataFrame:
    """Rows of one downloaded contract, after checking every file against its digest."""
    manifest = json.loads((root / f"download-{commodity}.json").read_text())
    frames = []
    for name, digest in sorted(manifest["files"].items()):
        path = root / "raw" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"TAIFEX file changed after download: {path}")
        frames.append(parse_futures(path.read_bytes()))
    if not frames:
        raise ValueError(f"TAIFEX download of {commodity} is empty")
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["contract"] == commodity].reset_index(drop=True)


def third_wednesday(year: int, month: int) -> date:
    """Last trading day of a TAIFEX monthly contract: the third Wednesday of its month."""
    first = date(year, month, 1)
    return first + timedelta(days=(2 - first.weekday()) % 7 + 14)


def continuous_front_month(futures: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """One row per session: the front monthly contract and its ratio-adjusted open series.

    Regular-session rows of monthly expiries only (weeklies and spreads excluded). A contract
    is the front month up to the session before its third Wednesday; on the third Wednesday
    the position is already in the next month (rolled at that open). ``adjusted_open`` chains
    the opens across rolls by ratio so that log differences are the return of a position
    that is always in the front month; ``roll`` marks the sessions whose open starts a new
    contract. ``settlement`` is the daily mark of the contract held after that open, and
    ``settlement_growth`` is the log return of the position between consecutive daily
    settlements (on a roll session: the old contract from its previous settlement to that
    open, then the new contract from that open to its settlement). TAIFEX computes the
    daily settlement price from trades just before 13:30, so it is the futures mark that
    lines up with the stock market's close.
    """
    rows = futures[(futures["contract"] == commodity) & (futures["session"] == "regular")
                   & futures["expiry"].str.fullmatch(r"\d{6}")].copy()
    if rows.empty:
        raise ValueError(f"No regular-session monthly rows for {commodity}")
    rows["last_day"] = [
        third_wednesday(int(e[:4]), int(e[4:])).isoformat() for e in rows["expiry"]
    ]
    rows = rows[rows["open"].notna() & (rows["open"] > 0)]
    # The expiring contract still trades on its last day (closed at that open on a roll).
    rows_by_key = rows.set_index(["date", "expiry"])["open"]
    settle_by_key = rows.set_index(["date", "expiry"])["settlement"]
    rows = rows[rows["date"] < rows["last_day"]]
    front = rows.loc[rows.groupby("date")["expiry"].idxmin()].sort_values("date")
    front = front.reset_index(drop=True)
    roll = front["expiry"].ne(front["expiry"].shift()).to_numpy()
    roll[0] = False
    # On a roll session the old contract is closed at this open and the new one opened, so
    # the position's growth into this session is the OLD contract's open-to-open move.
    previous_contract_open = front["open"].shift()
    old_open_today = [
        rows_by_key.get((d, e), float("nan")) if r else o
        for d, e, o, r in zip(front["date"], front["expiry"].shift(fill_value=""),
                              front["open"], roll, strict=True)
    ]
    growth = [
        float("nan") if i == 0 else (
            float(np.log(old / prev)) if np.isfinite(old) and prev > 0 else float("nan")
        )
        for i, (old, prev) in enumerate(zip(old_open_today, previous_contract_open, strict=True))
    ]
    front["front_growth"] = growth
    adjusted = np.exp(np.nancumsum(np.nan_to_num(front["front_growth"].to_numpy(), nan=0.0)))
    front["adjusted_open"] = adjusted * float(front["open"].iloc[0])
    front["roll"] = roll
    settlement_growth = [float("nan")]
    dates = front["date"].tolist()
    expiries = front["expiry"].tolist()
    for i in range(1, len(front)):
        previous_settle = settle_by_key.get((dates[i - 1], expiries[i - 1]), float("nan"))
        if roll[i]:
            old_open = rows_by_key.get((dates[i], expiries[i - 1]), float("nan"))
            new_open = float(front["open"].iloc[i])
            new_settle = float(front["settlement"].iloc[i])
            values = (previous_settle, old_open, new_open, new_settle)
            if all(np.isfinite(v) and v > 0 for v in values):
                settlement_growth.append(
                    float(np.log(old_open / previous_settle) + np.log(new_settle / new_open))
                )
            else:
                settlement_growth.append(float("nan"))
        else:
            settle = float(front["settlement"].iloc[i])
            ok = np.isfinite(previous_settle) and previous_settle > 0 and settle > 0
            settlement_growth.append(
                float(np.log(settle / previous_settle)) if ok else float("nan")
            )
    front["settlement_growth"] = settlement_growth
    return front[["date", "expiry", "open", "high", "low", "close", "settlement", "volume",
                  "open_interest", "front_growth", "adjusted_open", "settlement_growth",
                  "roll"]].reset_index(drop=True)


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
    for column in ("open", "high", "low", "close", "settlement", "volume", "open_interest"):
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
    parser.add_argument("command", choices=("download", "download-contract"))
    parser.add_argument("--config", type=Path,
                        default=BACKTEST_ROOT / "config" / "capturability.json")
    parser.add_argument("--artifacts-root", type=Path, default=BACKTEST_ROOT.parent / "artifacts")
    parser.add_argument("--commodity", default="TX", help="download-contract: TAIFEX code")
    parser.add_argument("--start", help="download-contract: first date (YYYY-MM-DD)")
    parser.add_argument("--end-exclusive", help="download-contract: end date, exclusive")
    parser.add_argument("--root", type=Path, help="download-contract: output root")
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.command == "download-contract":
        if not (args.start and args.end_exclusive and args.root):
            parser.error("download-contract needs --start, --end-exclusive and --root")
        manifest = download_contract(args.root, args.commodity, args.start, args.end_exclusive,
                                     pause=args.pause_seconds)
        series = continuous_front_month(load_contract(args.root, args.commodity), args.commodity)
        print(json.dumps({"files": len(manifest["files"]), "sessions": len(series),
                          "first": series["date"].iloc[0], "last": series["date"].iloc[-1],
                          "rolls": int(series["roll"].sum())}, ensure_ascii=False, indent=1))
        return 0
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
