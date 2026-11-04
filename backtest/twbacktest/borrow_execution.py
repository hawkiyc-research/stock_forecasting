"""Execution uncertainty of the short book (protocol 13.19 supplement 4).

Being on the exchange's borrowable list is not the same as getting the borrow, and a short
sale has no market-order fallback the way a buy or a sell has. The layer therefore has four
parts, all applied to the short band book only:

1. the borrowable universe: institution shorts need the day's 借券 list (``sbl_eligible``) and
   a positive next-session lending quota; retail shorts need the 融券 list (not suspended, not
   blocked by the limit-down rule) and positive margin-short room;
1b. the observable quota: in a capital scenario a new short is capped at the quota (shares)
   times the execution open, under the same cap-and-scale rule as the participation cap;
2. borrow failure: a Bernoulli draw per candidate and day with a probability by dollar-volume
   tercile of the universe (large / mid / small), times a sensitivity multiplier, plus a daily
   recall probability on held shorts;
3. tiered borrow fees for the institution profile by the same tercile;
4. (protocol 13.23 supplement 3, sensitivities) rows under a disposition period get no new
   entry in either leg (``disposition``), and a held short whose row is flagged
   ``short_suspended`` is covered at the next open (``forced_cover``).

The per-row data comes from the ``tw_chip_data`` curated layer (``eligibility`` and ``sbl``;
the row of signal date t is published at t 23:59:59 and used at the open of t + 1). A
prediction row without a list row on its date is not borrowable (the daily list is the universe
of margin-eligible securities, so absence means the stock is outside it); a signal date with no
list at all carries the previous list date forward and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twbacktest import chip_data
from twbacktest.exposure_features import quantile_labels
from twbacktest.walkforward_data import WalkForwardInputs

TERCILES = 3
LIST_FOR = {"sbl": "sbl_eligible", "margin": "margin_eligible"}


@dataclass(frozen=True)
class BorrowData:
    """Per prediction row of one market: borrowability inputs aligned with the walk-forward rows."""

    sbl_eligible: np.ndarray  # bool: on the 借券 list and next-session quota > 0
    margin_eligible: np.ndarray  # bool: 融券 allowed and margin-short room > 0
    sbl_quota_value: np.ndarray  # NT$: next-session lending quota x execution open (NaN: unknown)
    margin_room_value: np.ndarray  # NT$: (short_limit - short_balance) x execution open
    tercile: np.ndarray  # 0 large, 1 mid, 2 small by 20-session dollar volume (-1: not ranked)
    list_matched_share: float
    quota_matched_share: float
    carried_dates: int
    digests: dict[str, str]
    short_suspended: np.ndarray | None = None  # bool: margin short selling suspended that day
    disposition: np.ndarray | None = None  # bool: the execution day is inside a disposition

    def forced_exit(self, column: str | None) -> np.ndarray | None:
        """The row flag that forces a held short out (supplement 3 rule C), or None."""
        if column is None:
            return None
        if column != "short_suspended" or self.short_suspended is None:
            raise ValueError(f"forced_cover column {column!r} is not aligned")
        return self.short_suspended

    def eligible(self, borrow_list: str) -> np.ndarray:
        return getattr(self, LIST_FOR[borrow_list])

    def value_cap(self, borrow_list: str) -> np.ndarray:
        return self.sbl_quota_value if borrow_list == "sbl" else self.margin_room_value


@dataclass(frozen=True)
class ExecutionSpec:
    """One point of the execution layer for one cost profile."""

    profile: str
    borrow_list: str  # "sbl" (institution) or "margin" (retail)
    failure_probs: tuple[float, float, float]  # by tercile, before the multiplier
    failure_multiplier: float
    recall_prob: float
    fee_tiers: tuple[float, float, float] | None  # annual borrow fee by tercile (None: profile)
    quota_cap: bool
    replicates: int
    seed: int
    primary: bool
    forced_cover: str | None = None  # eligibility column that forces a cover (rule C)

    @property
    def key(self) -> tuple[float, float]:
        return (self.failure_multiplier, self.recall_prob)

    def fail_prob(self, tercile: np.ndarray) -> np.ndarray:
        probs = np.append(np.asarray(self.failure_probs, dtype=np.float64), 0.0)
        out = probs[np.where(tercile >= 0, tercile, TERCILES)] * self.failure_multiplier
        return np.clip(out, 0.0, 1.0)

    def fee_daily(self, tercile: np.ndarray, default_annual: float) -> np.ndarray | float:
        if self.fee_tiers is None:
            return default_annual / 252.0
        tiers = np.append(np.asarray(self.fee_tiers, dtype=np.float64), default_annual)
        return tiers[np.where(tercile >= 0, tercile, TERCILES)] / 252.0

    def replicate_seeds(self) -> list[int]:
        return [self.seed + r for r in range(max(1, self.replicates))]


def execution_specs(config: dict[str, Any], profile: str) -> list[ExecutionSpec]:
    """The primary point first, then the registered sensitivities (failure x recall grid)."""
    spec = config["execution"]
    borrow_list = str(spec["borrow_list"][profile])
    probs = tuple(float(x) for x in spec["borrow_failure"][profile])
    if len(probs) != TERCILES:
        raise ValueError("borrow_failure needs one probability per tercile")
    tiers = spec.get("borrow_fee_tiers", {}).get(profile)
    tiers = tuple(float(x) for x in tiers) if tiers is not None else None
    primary_mult = float(spec["primary_failure_multiplier"])
    primary_recall = float(spec["primary_recall_probability"])
    forced_cover = (spec.get("forced_cover") or {}).get(profile)
    forced_cover = str(forced_cover) if forced_cover else None
    points = [(primary_mult, primary_recall)]
    for mult in spec["failure_multipliers"]:
        for recall in spec["recall_probabilities"]:
            if (float(mult), float(recall)) not in points:
                points.append((float(mult), float(recall)))
    return [
        ExecutionSpec(
            profile=profile, borrow_list=borrow_list, failure_probs=probs,
            failure_multiplier=mult, recall_prob=recall, fee_tiers=tiers,
            quota_cap=bool(spec.get("quota_cap", True)), replicates=int(spec["replicates"]),
            seed=int(spec["seed"]), primary=(i == 0), forced_cover=forced_cover,
        )
        for i, (mult, recall) in enumerate(points)
    ]


def _carry_dates(signal_dates: pd.Series, table_dates: np.ndarray) -> tuple[pd.Series, int]:
    """Map each signal date to itself when the table has that date, else to the latest earlier
    table date (a missing daily list is carried forward); returns the mapped dates and the
    number of distinct signal dates that were carried."""
    table_dates = np.sort(table_dates)
    values = signal_dates.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(table_dates, values, side="right") - 1
    mapped = np.where(pos >= 0, table_dates[np.maximum(pos, 0)], np.datetime64("NaT", "ns"))
    carried = np.unique(values[(mapped != values) & (pos >= 0)]).size
    return pd.Series(mapped, index=signal_dates.index), carried


def _aligned(predictions: pd.DataFrame, table: pd.DataFrame, columns: list[str]):
    dates = table["date"].unique()
    mapped, carried = _carry_dates(predictions["signal_date"], dates)
    keyed = predictions[["symbol"]].assign(signal_date=mapped.to_numpy())
    return chip_data.align_rows(keyed, table, columns), carried


def _disposition_flags(
    predictions: pd.DataFrame, inputs: WalkForwardInputs, periods: pd.DataFrame
) -> np.ndarray:
    """Per prediction row: the execution day (the next signal date) lies inside a disposition
    window announced on or before the signal date (rule D of supplement 3)."""
    dates = np.sort(predictions["signal_date"].unique()).astype("datetime64[ns]")
    following = np.append(dates[1:], dates[-1] + np.timedelta64(1, "D"))
    signal = predictions["signal_date"].to_numpy(dtype="datetime64[ns]")
    execution = following[np.searchsorted(dates, signal)]
    keyed = pd.DataFrame({
        "symbol": predictions["symbol"].to_numpy(), "signal_date": signal,
        "execution_date": execution, "row": np.arange(len(predictions)),
    })
    merged = keyed.merge(periods, left_on="symbol", right_on="prediction_symbol", how="inner")
    hit = merged[(merged["announce_date"] <= merged["signal_date"])
                 & (merged["period_start"] <= merged["execution_date"])
                 & (merged["execution_date"] <= merged["period_end"])]
    flags = np.zeros(len(predictions), dtype=bool)
    flags[hit["row"].to_numpy()] = True
    return flags


def load_borrow_data(
    root: Path,
    predictions: pd.DataFrame,
    inputs: WalkForwardInputs,
    groups: tuple[np.ndarray, ...],
    *,
    datasets: dict[str, str] | None = None,
    disposition: bool = False,
) -> BorrowData:
    """Join the eligibility and lending tables onto the prediction rows of one model.

    ``predictions`` is the frame ``build_inputs`` was built from (same row order as ``inputs``);
    ``groups`` are the universe rows per signal date, used for the dollar-volume terciles.
    With ``disposition`` the disposition windows are aligned too (rule D of supplement 3).
    """
    names = {"eligibility": "eligibility", "sbl": "sbl", "disposition": "disposition",
             **(datasets or {})}
    years = sorted({int(y) for y in predictions["signal_date"].dt.year.unique()})
    eligibility = chip_data.short_eligibility(root, years)
    lending = chip_data.borrow_quota(root, years)
    lists, carried_lists = _aligned(
        predictions, eligibility,
        ["sbl_eligible", "sbl_suspended", "margin_short_allowed", "short_suspended"])
    disposition_flags = None
    digests = {**chip_data.partition_digests(root, names["eligibility"], years),
               **chip_data.partition_digests(root, names["sbl"], years)}
    if disposition:
        period_years = sorted({years[0] - 1, *years})
        periods = chip_data.disposition_periods(root, period_years)
        disposition_flags = _disposition_flags(predictions, inputs, periods)
        digests.update(chip_data.partition_digests(root, names["disposition"], period_years))
    quota, carried_quota = _aligned(predictions, lending,
                                    ["sbl_quota_shares", "margin_short_room_shares"])
    open_ = inputs.raw_open_1 if inputs.raw_open_1 is not None else np.full(inputs.index.count,
                                                                             np.nan)
    with np.errstate(invalid="ignore"):
        sbl_value = quota["sbl_quota_shares"] * open_
        margin_value = quota["margin_short_room_shares"] * open_
    sbl_ok = lists["sbl_eligible"] & ~lists["sbl_suspended"] & (quota["sbl_quota_shares"] > 0)
    margin_ok = lists["margin_short_allowed"] & (quota["margin_short_room_shares"] > 0)
    tercile = np.full(inputs.index.count, -1, dtype=np.int64)
    volume = inputs.dollar_volume
    if volume is None:
        raise ValueError("the walk-forward inputs carry no dollar volume")
    for rows in groups:
        if rows.size == 0:
            continue
        values = np.where(np.isfinite(volume[rows]), -volume[rows], np.inf)  # largest first
        tercile[rows] = quantile_labels(values, TERCILES)
    return BorrowData(
        sbl_eligible=sbl_ok, margin_eligible=margin_ok, sbl_quota_value=sbl_value,
        margin_room_value=margin_value, tercile=tercile,
        list_matched_share=float(lists["matched"].mean()),
        quota_matched_share=float(quota["matched"].mean()),
        carried_dates=carried_lists + carried_quota,
        digests=digests,
        short_suspended=lists["short_suspended"], disposition=disposition_flags,
    )


def short_arrays(
    spec: ExecutionSpec, borrow: BorrowData, *, fractional: bool, default_fee_annual: float
) -> dict[str, Any]:
    """Keyword arguments of ``simulate_band`` for the short book plus the per-row daily fee.

    With ``forced_cover`` the flagged rows are also removed from the entry list: a lender who
    recalls at the suspension does not lend during it (the margin list excludes them already),
    otherwise the institution book re-enters the day after every forced cover.
    """
    forced_exit = borrow.forced_exit(spec.forced_cover)
    eligible = borrow.eligible(spec.borrow_list)
    if forced_exit is not None:
        eligible = eligible & ~forced_exit
    return {
        "eligible": eligible,
        "fail_prob": spec.fail_prob(borrow.tercile),
        "value_cap": None if (fractional or not spec.quota_cap)
        else borrow.value_cap(spec.borrow_list),
        "recall_prob": spec.recall_prob,
        "forced_exit": forced_exit,
        "fee_daily": spec.fee_daily(borrow.tercile, default_fee_annual),
    }


def summarize_borrow(borrow: BorrowData, groups: tuple[np.ndarray, ...]) -> dict[str, float]:
    """Coverage of the universe rows, for the report."""
    rows = np.concatenate([g for g in groups if g.size]) if groups else np.array([], dtype=int)
    if rows.size == 0:
        return {}
    out = {
        "universe_rows": int(rows.size),
        "sbl_eligible_share": float(borrow.sbl_eligible[rows].mean()),
        "margin_eligible_share": float(borrow.margin_eligible[rows].mean()),
        "sbl_quota_value_median": float(np.nanmedian(borrow.sbl_quota_value[rows])),
        "list_matched_share": borrow.list_matched_share,
        "quota_matched_share": borrow.quota_matched_share,
        "carried_dates": borrow.carried_dates,
    }
    if borrow.short_suspended is not None:
        out["short_suspended_share"] = float(borrow.short_suspended[rows].mean())
    if borrow.disposition is not None:
        out["disposition_share"] = float(borrow.disposition[rows].mean())
    return out
