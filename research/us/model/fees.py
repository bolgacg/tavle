"""NYISO OATT Rate Schedule 1 charge for non-physical (virtual) transactions, USD per cleared MWh,
by calendar year of the delivery date (CONTRACT.md, "Positions and money").

Every virtual position (supply or load, 1 MW for one hour) pays this once. No posting reviewed shows
a mid-year change.

Sources (read in the posting itself on 6 Oct 2026):
  2020  0.0862  "Schedule 1 Rates for 2020": "Rate Schedule 1 for Non-Physical Transactions: Virtual
                Resources $0.0862 per cleared MWh"
                https://www.nyiso.com/documents/20142/7661617/2020-Rate-Schedule-1.pdf/6436863b-bf7c-4228-5be6-a78e7971023c
  2022  0.0853  "Schedule 1 Rates for 2022": "Virtual Resources $0.0853 per cleared MWh"
                https://www.nyiso.com/documents/20142/24362705/2022-Schedule-One-Posting-with-accounting-detail.pdf/844f8b6a-089c-4837-f453-7e0f6989bd75
  2023  0.1066  "2023 Schedule One Posting": "Virtual Resources $0.1066 per cleared MWh"
                https://www.nyiso.com/documents/20142/32669344/2023-Schedule-One-Posting-for-posting-with-detail.pdf/840e952d-8555-a853-8d46-33a05cba1911
  2024  0.1333  "Non-Physical Schedule 1 Rates for 2024": "Virtual Resources $0.1333 per cleared MWh"
                https://www.nyiso.com/documents/20142/39240890/2024-Sched-One-Posting-non-Physical-for-posting.pdf/e0e5b9a6-04b4-e0c3-bb08-ebf489192f83
  2026  0.1919  given in CONTRACT.md (orchestrator); the 2026 posting was not found in this search.

Not found (postings searched on nyiso.com, the billing and budget pages, and the 2025 draft budget deck):
  2021  uses 0.0862, the nearest known years being 2020 (0.0862) and 2022 (0.0853), equally near; the
        higher one is taken so the cost is not understated.
  2025  uses 0.1919, the nearest known years being 2024 (0.1333) and 2026 (0.1919), equally near; the
        higher one is taken. An unsourced search snippet said 0.1666 for 2025: not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATES = {2020: 0.0862, 2021: 0.0862, 2022: 0.0853, 2023: 0.1066, 2024: 0.1333, 2025: 0.1919, 2026: 0.1919}
SOURCE = {2020: "posting", 2021: "nearest known year (2020/2022 tie, higher taken)", 2022: "posting",
          2023: "posting", 2024: "posting", 2025: "nearest known year (2024/2026 tie, higher taken)",
          2026: "CONTRACT.md, posting not found"}
STRESS = (0.50, 1.00)


def fee(year: int) -> float:
    return RATES[int(year)]


def fee_for(delivery_dates, override: float | None = None) -> np.ndarray:
    """Fee per row from delivery dates (naive local dates or tz-aware hours). `override` gives the
    fee-stress cost (0.50 or 1.00) for every row instead."""
    s = pd.Series(delivery_dates)
    if override is not None:
        return np.full(len(s), float(override))
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("America/New_York")
    years = pd.to_datetime(s).dt.year
    return years.map(RATES).to_numpy(dtype=float)
