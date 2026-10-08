"""V15d: the zone-pair rule (C, strategies_v2.Pairs) driven by the V4 weather model's saved predictions
(preds/V4_B_reforecast.parquet, column pred). Same pair rule as V1_C_gbm and V1_C_deep: per pair the leg when the
predicted spread clears load + supply cost, the 5 best positive pairs of 55 on the trailing 4 out-of-sample
quarters. Scored 2013 to 2019 only, as V4 (the reforecast ends in 2019). Settings tried: 1.

    python v15_pairs.py
Row: V15d_V4_pairs -> results/v2/pos/.
"""
from __future__ import annotations

import pandas as pd

import v15_common as V
from v15_common import R, SV


def main():
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp", "gap_365d_h"])
    g = SV.attach(R.load_preds("V4_B_reforecast"), panel)
    g = g[pd.to_datetime(g["delivery_date"]) < V.REFORECAST_END].reset_index(drop=True)
    SV.idea_c("V15d_V4_pairs", g, "pred", "V15",
              "V15d: pair rule on the V4 weather (GEFS reforecast) gap model; 2013-2019 only")


if __name__ == "__main__":
    main()
