"""Exchange cost per cleared virtual MWh, by calendar year of the delivery date (CONTRACT.md "Positions
and money"; cost audit research/us/results/cost_capital_audit.md, folded in by the dated addendum of
6 Oct 2026, commit c3adae3).

Each 1 MW position for one hour pays, once (on the cleared day-ahead quantity; the real-time buy-back
is not charged again):
  * the Rate Schedule 1 virtual rate (RS1, OATT 6.1.2.4) and
  * the FERC annual charge passed through Rate Schedule 1 (OATT 6.1.15), per cleared virtual MWh;
  * virtual SUPPLY legs only, also the forecast-pass BPCG uplift (OATT Attachment T), at its yearly
    upper bound (the whole pot spread over load-zone virtual supply). Virtual load pays no uplift.
Stress costs (0.50 and 1.00 USD/MWh) replace the whole per-MWh cost on both sides.

Sources (details, quotes and URLs in cost_capital_audit.md, sections 1 and 2):
  RS1  2020 0.0862  "Schedule 1 Rates for 2020", nyiso.com posting
       2021 0.0757  NYISO BPWG "September 2021 Draft Budget versus Actual Results", slide 4
       2022 0.0853  "Schedule 1 Rates for 2022", nyiso.com posting
       2023 0.1066  "Schedule 1 Rates for 2023", nyiso.com posting
       2024 0.1333  "Non-Physical Schedule 1 Rates for 2024", nyiso.com posting
       2025 0.1666  posting not found; 0.1333 x 1.25, the tariff's maximum yearly rise (OATT 6.1.2.4.4),
                    matching a secondary report of the 2025 posting
       2026 0.1919  from CONTRACT.md; UNVERIFIED at a primary source (inside the tariff band)
  FERC 0.010 (2020, estimate) 0.015 0.016 0.017 0.017 (2021-2024, NYISO FERC fee postings divided by
       cleared virtual MWh) 0.020 0.020 (2025, 2026: planning value, pots not found)
  Uplift upper bound, supply only: 2020 0.007, 2021 0.004, 2022 0.003, 2023 0.026, 2024 0.017 (market
       monitor's forecast-pass BPCG totals over cleared virtual supply), 2025 0.028 (upper end of the
       audit's 0.017 to 0.028), 2026 0.030 (audit's approximate value)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RS1 = {2020: 0.0862, 2021: 0.0757, 2022: 0.0853, 2023: 0.1066, 2024: 0.1333, 2025: 0.1666, 2026: 0.1919}
FERC = {2020: 0.010, 2021: 0.015, 2022: 0.016, 2023: 0.017, 2024: 0.017, 2025: 0.020, 2026: 0.020}
UPLIFT_SUPPLY = {2020: 0.007, 2021: 0.004, 2022: 0.003, 2023: 0.026, 2024: 0.017, 2025: 0.028, 2026: 0.030}
RATES = {y: round(RS1[y] + FERC[y], 6) for y in RS1}                   # every cleared MWh, either side
SUPPLY_RATES = {y: round(RATES[y] + UPLIFT_SUPPLY[y], 6) for y in RS1}  # a virtual supply MWh
SOURCE = {"rs1": {2020: "posting", 2021: "NYISO budget deck", 2022: "posting", 2023: "posting", 2024: "posting",
                  2025: "tariff cap (0.1333 x 1.25), posting not found", 2026: "CONTRACT.md, unverified"},
          "ferc": "NYISO FERC fee postings / cleared virtual MWh; 2020 estimate; 2025-2026 planning value",
          "uplift_supply": "forecast-pass BPCG upper bound, market monitor; 2025-2026 upper estimates",
          "audit": "research/us/results/cost_capital_audit.md"}
STRESS = (0.50, 1.00)


def fee(year: int) -> float:
    """Cost of one cleared MWh of virtual load (RS1 + FERC)."""
    return RATES[int(year)]


def supply_fee(year: int) -> float:
    """Cost of one cleared MWh of virtual supply (RS1 + FERC + uplift upper bound)."""
    return SUPPLY_RATES[int(year)]


def _years(delivery_dates) -> pd.Series:
    s = pd.Series(delivery_dates).reset_index(drop=True)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("America/New_York")
    return pd.to_datetime(s).dt.year


def fee_for(delivery_dates, override: float | None = None) -> np.ndarray:
    """Per row: the load-side cost (RS1 + FERC), or the stress cost for every MWh."""
    if override is not None:
        return np.full(len(delivery_dates), float(override))
    return _years(delivery_dates).map(RATES).to_numpy(dtype=float)


def supply_fee_for(delivery_dates, override: float | None = None) -> np.ndarray:
    """Per row: the supply-side cost (RS1 + FERC + uplift), or the stress cost for every MWh."""
    if override is not None:
        return np.full(len(delivery_dates), float(override))
    return _years(delivery_dates).map(SUPPLY_RATES).to_numpy(dtype=float)


def cost_for(delivery_dates, pos, override: float | None = None) -> np.ndarray:
    """Per row cost per MWh of the position taken: supply legs (pos < 0) pay the supply cost, load legs
    the load cost, no position nothing to choose (the ledger multiplies by MWh)."""
    pos = np.asarray(pos, dtype=float)
    return np.where(pos < 0, supply_fee_for(delivery_dates, override), fee_for(delivery_dates, override))
