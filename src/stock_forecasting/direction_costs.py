"""Ex-ante round-trip cost bands for listed cash equities; no future-price inputs."""

from __future__ import annotations

import math
from typing import Any


def direction_cost_band(
    price: float, market: str, asset_type: str, symbol: str, costs: Any
) -> dict:
    """Estimate both legs at the known cutoff price in the listing currency.

    The lower band is an adverse-move label, not the profitability of a short.
    Alpha uses the same hurdle as a minimum relative edge, not a hedged P&L.
    """
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Direction costs require a finite positive cutoff price")
    if market in ("TWSE", "TPEX"):
        quantity = costs.taiwan_shares
        notional = quantity * price
        commission = sum(
            max(notional * rate, costs.taiwan_minimum_commission)
            for rate in costs.taiwan_commission_rates
        ) / len(costs.taiwan_commission_rates)
        local_code = symbol.split(".", 1)[0]
        # TWSE recognizes both legacy four-digit and newer six-digit TDR codes.
        is_tdr = (
            local_code.isdecimal() and local_code.startswith("91") and len(local_code) in (4, 6)
        )
        tax = (
            costs.taiwan_etf_tdr_sell_tax
            if asset_type.lower() == "etf" or is_tdr
            else costs.taiwan_stock_sell_tax
        )
        regulatory = notional * tax
    elif market == "US":
        quantity = costs.us_shares
        notional = quantity * price
        ibkr = min(
            max(quantity * costs.us_ibkr_per_share, costs.us_ibkr_minimum),
            notional * costs.us_ibkr_maximum_rate,
        )
        commission = (ibkr + costs.us_schwab_commission) / 2
        # Cent rounding is conservative for small tickets; CAT is charged on both legs.
        sec = math.ceil(notional * costs.us_sec_sell_rate * 100 - 1e-12) / 100
        taf = (
            math.ceil(min(quantity * costs.us_taf_per_share, costs.us_taf_cap) * 100 - 1e-12) / 100
        )
        regulatory = sec + taf + 2 * quantity * costs.us_cat_per_share
    else:
        raise ValueError(f"No direction cost scenario for market {market!r}")
    fraction = (2 * commission + regulatory) / notional + 2 * costs.slippage_bps_per_side / 10000
    if not 0 <= fraction < 1:
        raise ValueError("Round-trip cost must be below the entire order notional")
    return {
        "lower_log_return": math.log1p(-fraction),
        "upper_log_return": math.log1p(fraction),
        "round_trip_fraction": fraction,
        "shares": quantity,
        "reference_price": price,
        "currency": "USD" if market == "US" else "TWD",
        "scenario": costs.scenario,
    }
