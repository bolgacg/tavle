"""V6: zone specialists for New York City (N.Y.C.) and Long Island (LONGIL), where the market monitor
reports persistent real-time premiums (SOM 2022 and 2024; papers_notes.md sections 1, 3, 4).

Each specialist is its own model trained on its zone's rows only (train_mask), with load-pocket inputs
the all-zone model does not see. Spike thresholds are Trading Electrons' NYISO ones (+5 and -30 USD).
Columns (all from prices public at 05:00 on D, PriceGrid):
    sp_li_nyc_da_d0_h        LONGIL minus N.Y.C. day-ahead, same hour on D
    sp_cong_share_d0_h       the row zone's day-ahead congestion over its day-ahead price, hour on D
    sp_up5_28d_h             share of the last 28 days with gap >= +5 at the hour, row zone
    sp_dn30_28d_h            share with gap <= -30
    sp_up5_7d_all            share of all public hours of the last 7 days with gap >= +5, row zone
    sp_gap_q90_28d_h         90th percentile of the 28-day gap at the hour (right tail)
    sp_li_nyc_gap_28d_h      trailing 28-day mean of LONGIL gap minus N.Y.C. gap at the hour
Rows: V6_NYC and V6_LI (two-sided rule, specialist model, test rows of that zone only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ideas_common as C
from v5_border import grid as _grid

NAME = "V6"
SPECIALIST_ZONES = {"V6_NYC": "N.Y.C.", "V6_LI": "LONGIL"}
UP, DN = 5.0, -30.0
COLUMNS = ["sp_li_nyc_da_d0_h", "sp_cong_share_d0_h", "sp_up5_28d_h", "sp_dn30_28d_h", "sp_up5_7d_all",
           "sp_gap_q90_28d_h", "sp_li_nyc_gap_28d_h"]
ROWS = {k: dict(features="base+V6", rule="two_sided", zone=z, model="specialist (own zone rows only)")
        for k, z in SPECIALIST_ZONES.items()}


def train_mask(panel: pd.DataFrame, row: str) -> np.ndarray:
    return (panel["zone"] == SPECIALIST_ZONES[row]).to_numpy()


def features(panel_index: pd.MultiIndex, panel: pd.DataFrame | None = None, store=None) -> pd.DataFrame:
    p = C.panel_frame(panel_index)
    g = _grid(store)
    days = sorted(p["bid_date"].unique())
    li, ny = g.nodes.index("LONGIL"), g.nodes.index("N.Y.C.")
    da0 = C.same_hour_on_D(g, "da_lbmp", "da", days)
    cg0 = C.same_hour_on_D(g, "da_congestion", "da", days)
    up = C.trailing(g, None, "rt", days, 28, transform=lambda gg, sl, D: np.where(
        np.isnan(x := C.gap_transform(gg, sl, D)), np.nan, (x >= UP).astype(float)))
    dn = C.trailing(g, None, "rt", days, 28, transform=lambda gg, sl, D: np.where(
        np.isnan(x := C.gap_transform(gg, sl, D)), np.nan, (x <= DN).astype(float)))
    up7 = C.trailing(g, None, "rt", days, 7, transform=lambda gg, sl, D: np.where(
        np.isnan(x := C.gap_transform(gg, sl, D)), np.nan, (x >= UP).astype(float)))
    up7_all = np.repeat(np.nanmean(up7, axis=1, keepdims=True), 24, axis=1)
    q90 = C.trailing(g, None, "rt", days, 28, op=lambda x, axis: np.nanquantile(x, 0.9, axis=axis),
                     transform=C.gap_transform)
    spr = C.trailing(g, None, "rt", days, 28, transform=lambda gg, sl, D: (
        lambda x: x[:, :, [li]] - x[:, :, [ny]])(C.gap_transform(gg, sl, D)))
    zda = C.to_rows(da0, days, g.nodes, p)
    out = {
        "sp_li_nyc_da_d0_h": C.to_rows(da0, days, g.nodes, p, fixed_node="LONGIL")
                             - C.to_rows(da0, days, g.nodes, p, fixed_node="N.Y.C."),
        "sp_cong_share_d0_h": C.to_rows(cg0, days, g.nodes, p) / np.where(np.abs(zda) < 1, np.nan, zda),
        "sp_up5_28d_h": C.to_rows(up, days, g.nodes, p),
        "sp_dn30_28d_h": C.to_rows(dn, days, g.nodes, p),
        "sp_up5_7d_all": C.to_rows(up7_all, days, g.nodes, p),
        "sp_gap_q90_28d_h": C.to_rows(q90, days, g.nodes, p),
        "sp_li_nyc_gap_28d_h": C.to_rows(spr, days, [g.nodes[li]], p, fixed_node="LONGIL"),
    }
    res = pd.DataFrame(out)[COLUMNS].astype(float)
    res.index = panel_index
    return res


rule = C.two_sided
