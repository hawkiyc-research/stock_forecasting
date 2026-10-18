"""Taiwan trading costs per side, as fractions of the traded amount."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from twbacktest.data import MARKETS


@dataclass(frozen=True)
class Scenario:
    discount: float
    slippage_multiplier: float
    turnover: str  # "conservative": full replacement; "optimistic": kept names pay nothing

    @property
    def key(self) -> str:
        return f"discount={self.discount:g},slippage=x{self.slippage_multiplier:g},{self.turnover}"


def scenarios(costs: dict[str, Any]) -> list[Scenario]:
    return [
        Scenario(float(discount), float(multiplier), str(turnover))
        for discount in costs["discounts"]
        for multiplier in costs["slippage_multipliers"]
        for turnover in costs["turnover_modes"]
    ]


def primary_scenario(costs: dict[str, Any]) -> Scenario:
    primary = costs["primary"]
    return Scenario(
        float(primary["discount"]), float(primary["slippage_multiplier"]), primary["turnover"]
    )


def side_costs(
    costs: dict[str, Any],
    scenario: Scenario,
    market_codes: np.ndarray,
    is_etf: np.ndarray,
    *,
    same_day: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Buy and sell cost per row.

    Buy: discounted commission plus slippage. Sell: discounted commission, the
    transaction tax (the same-day rate when bought and sold in one session) and
    slippage.
    """
    commission = costs["commission_rate"] * scenario.discount
    slippage = np.array(
        [costs["slippage_per_side"][market] for market in MARKETS], dtype=np.float64
    )[market_codes] * scenario.slippage_multiplier
    taxes = costs["day_trade_tax" if same_day else "transaction_tax"]
    tax = np.where(is_etf, taxes["etf"], taxes["stock"])
    return commission + slippage, commission + tax + slippage
