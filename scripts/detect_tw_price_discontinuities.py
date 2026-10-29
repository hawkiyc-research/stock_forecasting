#!/usr/bin/env python3
"""List suspected splits and capital reductions in a raw market Parquet and verify them.

For every TWSE / TPEx stock and ETF the script computes the dividend-adjusted open
(open x adjusted_close / close) and flags a session whose adjusted open differs from the
previous session's by more than ``--threshold`` (default 25%). Such jumps come from
corporate actions that the dividend tables do not carry: capital reductions, par-value
changes and ETF splits, which halt the security for several sessions and resume it at an
official reference price. ``halt_sessions`` counts the market's trading sessions (taken from
the market's index rows) between the two bars.

With ``--actions`` (the CSV written by scripts/fetch_tw_capital_actions.py) every event is
matched to the official action on the same symbol and resumption date; ``residual_ratio`` is
the jump that would remain after the pipeline multiplies the earlier adjusted_close by the
official price factor. The script also applies every official action to the whole table and
reports how many jumps remain, so the adjustment can be validated before a Pod run.

Outputs: a CSV of events (one row per jump) and a JSON summary. Needs pandas and pyarrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX_BY_MARKET = {"TWSE": "TAIEX.TW", "TPEX": "TPEX.TWO"}
EVENT_COLUMNS = (
    "symbol", "market", "asset_type", "resumption_date", "previous_session", "halt_sessions",
    "open_ratio", "prior_open", "resumed_open", "resumed_volume", "matched_source",
    "official_price_factor", "residual_ratio",
)


def _adjusted_open(frame: pd.DataFrame, adjusted_close: str = "adjusted_close") -> pd.Series:
    return frame["open"] * frame[adjusted_close] / frame["close"]


def _jumps(frame: pd.DataFrame, threshold: float, adjusted_close: str) -> pd.DataFrame:
    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    ordered["adj_open"] = _adjusted_open(ordered, adjusted_close)
    grouped = ordered.groupby("symbol", sort=False)
    ordered["prev_adj_open"] = grouped["adj_open"].shift(1)
    ordered["previous_session"] = grouped["timestamp"].shift(1)
    ordered["open_ratio"] = ordered["adj_open"] / ordered["prev_adj_open"]
    mask = (ordered["open_ratio"] > 1.0 + threshold) | (ordered["open_ratio"] < 1.0 - threshold)
    return ordered.loc[mask].copy()


def _halt_sessions(events: pd.DataFrame, sessions_by_market: dict[str, np.ndarray]) -> np.ndarray:
    counts = np.zeros(len(events), dtype=np.int64)
    for position, (market, previous, current) in enumerate(
        zip(events["market"], events["previous_session"], events["timestamp"], strict=True)
    ):
        sessions = sessions_by_market.get(market)
        if sessions is None:
            counts[position] = -1
            continue
        left = np.searchsorted(sessions, np.datetime64(previous), side="right")
        right = np.searchsorted(sessions, np.datetime64(current), side="left")
        counts[position] = max(int(right - left), 0)
    return counts


def _apply_actions(frame: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Multiply adjusted_close by every official factor dated after the row (per symbol)."""

    factor = np.ones(len(frame), dtype=np.float64)
    timestamps = frame["timestamp"].to_numpy()
    symbols = frame["symbol"].to_numpy()
    for symbol, rows in actions.groupby("symbol"):
        positions = np.flatnonzero(symbols == symbol)
        if positions.size == 0:
            continue
        for action in rows.itertuples(index=False):
            before = timestamps[positions] < np.datetime64(action.timestamp)
            factor[positions[before]] *= float(action.price_factor)
    return frame["adjusted_close"] * factor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market-parquet", type=Path, required=True,
                        help="raw market.parquet of a prepared dataset")
    parser.add_argument("--actions", type=Path, default=None,
                        help="CSV from scripts/fetch_tw_capital_actions.py for verification")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="relative jump of the adjusted open that flags an event")
    parser.add_argument("--start", default="2016-01-01", help="first session to scan, ISO")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="event CSV (default configs/reference/tw_price_discontinuities_<today>.csv)",
    )
    args = parser.parse_args()
    today = datetime.now(UTC).strftime("%Y%m%d")
    output = args.output or (
        ROOT / "configs" / "reference" / f"tw_price_discontinuities_{today}.csv"
    )

    columns = ["timestamp", "symbol", "asset_type", "market", "open", "close",
               "adjusted_close", "volume"]
    market = pd.read_parquet(args.market_parquet, columns=columns)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True).dt.tz_localize(None)
    market = market[market["timestamp"] >= pd.Timestamp(args.start)]
    sessions_by_market = {
        name: np.sort(market.loc[market["symbol"] == index, "timestamp"].to_numpy())
        for name, index in INDEX_BY_MARKET.items()
    }
    securities = market[market["asset_type"] != "index"].reset_index(drop=True)

    actions = None
    if args.actions is not None:
        actions = pd.read_csv(args.actions)
        actions["timestamp"] = pd.to_datetime(actions["timestamp"])
        actions = actions.sort_values(["symbol", "timestamp"], kind="stable")
        securities["adjusted_close_official"] = _apply_actions(securities, actions)

    events = _jumps(securities, args.threshold, "adjusted_close")
    events["halt_sessions"] = _halt_sessions(events, sessions_by_market)
    events = events.rename(columns={"timestamp": "resumption_date"})
    events["prior_open"] = events["prev_adj_open"]
    events["resumed_open"] = events["adj_open"]
    events["resumed_volume"] = events["volume"]
    if actions is not None:
        merged = actions.rename(
            columns={"timestamp": "resumption_date", "price_factor": "official_price_factor",
                     "source": "matched_source"}
        )
        keys = ["symbol", "resumption_date"]
        events = events.merge(
            merged.loc[:, [*keys, "matched_source", "official_price_factor"]],
            on=keys, how="left",
        )
        events["residual_ratio"] = events["open_ratio"] / events["official_price_factor"]
        remaining = _jumps(securities, args.threshold, "adjusted_close_official")
    else:
        events["matched_source"] = ""
        events["official_price_factor"] = np.nan
        events["residual_ratio"] = np.nan
        remaining = None

    events = events.sort_values(["resumption_date", "symbol"], kind="stable")
    for name in ("resumption_date", "previous_session"):
        events[name] = events[name].dt.strftime("%Y-%m-%d")
    events = events.loc[:, list(EVENT_COLUMNS)]
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output, index=False, float_format="%.6g", lineterminator="\n")

    def _block(subset: pd.DataFrame) -> dict:
        halted = subset[subset["halt_sessions"] >= 2]
        block = {
            "events": len(subset),
            "symbols": int(subset["symbol"].nunique()),
            "events_after_halt": len(halted),
            "symbols_after_halt": int(halted["symbol"].nunique()),
        }
        if actions is not None:
            matched = subset["matched_source"].notna()
            block["matched_events"] = int(matched.sum())
            block["matched_share"] = float(matched.mean()) if len(subset) else None
            block["matched_events_after_halt"] = int(halted["matched_source"].notna().sum())
            block["matched_share_after_halt"] = (
                float(halted["matched_source"].notna().mean()) if len(halted) else None
            )
            block["unmatched_after_halt"] = halted.loc[
                halted["matched_source"].isna(), ["symbol", "resumption_date", "open_ratio"]
            ].to_dict("records")
        return block

    summary = {
        "market_parquet": str(args.market_parquet),
        "threshold": args.threshold,
        "scan_start": args.start,
        "rows_scanned": len(securities),
        "symbols_scanned": int(securities["symbol"].nunique()),
        "since_scan_start": _block(events),
        "since_2023": _block(events[events["resumption_date"] >= "2023-01-01"]),
        "etf_events": events[events["asset_type"] == "etf"].to_dict("records"),
        "actions_csv": str(args.actions) if args.actions is not None else None,
        "events_csv": str(output),
        "events_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if remaining is not None:
        remaining["halt_sessions"] = _halt_sessions(remaining, sessions_by_market)
        halted = remaining[remaining["halt_sessions"] >= 2]
        recent = remaining["timestamp"] >= "2023-01-01"
        summary["after_official_adjustment"] = {
            "events": len(remaining),
            "symbols": int(remaining["symbol"].nunique()),
            "events_after_halt": len(halted),
            "events_since_2023": int(recent.sum()),
            "events_after_halt_since_2023": int((halted["timestamp"] >= "2023-01-01").sum()),
            "remaining_after_halt": [
                {
                    "symbol": row.symbol,
                    "resumption_date": row.timestamp.strftime("%Y-%m-%d"),
                    "halt_sessions": int(row.halt_sessions),
                    "open_ratio": float(row.open_ratio),
                }
                for row in halted.sort_values("timestamp").itertuples(index=False)
            ],
        }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "etf_events"}, indent=2,
                     ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
