"""V5: neighbouring regions. NYISO's proxy-bus prices for PJM, New England (NPX), Ontario (O H) and
Quebec (H Q) as model inputs, and a zone-versus-border spread strategy.

Source: parquet_v2/prices_zone.parquet (zones) and prices_gen/YYYYMM.parquet rows with point_type
'external' named PJM, NPX, O H, H Q (same columns as zones:
da_lbmp, rt_lbmp, da_congestion, rt_congestion, da_published_at, rt_published_at). A value counts for
bid day D only if published by 05:00 on D (PriceGrid). Day-ahead of D is posted on D-1, so the D
day-ahead is a legal input; D+1's is not.

Columns (B = PJM, NPX, OH, HQ):
    bx_da_d0_h_B        border day-ahead price on D, same wall hour as the panel row
    bx_cong_d0_h_B      its congestion part
    bx_zspread_d0_h_B   the row's zone day-ahead minus the border day-ahead, same hour on D
    bx_gap_7d_h_B       border trailing 7-day mean gap (rt - da) at the hour
    bx_gap_28d_h_B      same, 28 days
    bx_zgap_28d_h_B     trailing 28-day mean of (zone gap - border gap) at the hour: the spread signal
Rows:
    V5_border_inputs    base features + these; v1 two-sided rule
    V5_zone_border_spread
        per (zone, border) cell: take the side of the trailing zone-minus-border gap spread
        (bx_zgap_28d_h_B) when it clears one load leg plus one supply leg; the trade is load zone +
        supply border or the reverse. Scored as two legs; the border leg is NOT a NYISO virtual (virtuals
        clear at load zones only), it stands for an import or export schedule at the proxy, so the
        row is reported as a hypothetical spread, flagged not_tradable_as_virtual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ideas_common as C

NAME = "V5"
TAG = {"PJM": "PJM", "NPX": "NPX", "O H": "OH", "H Q": "HQ"}
VALS = ["da_lbmp", "rt_lbmp", "da_congestion"]
COLUMNS = [f"bx_{k}_{TAG[b]}" for b in C.BORDERS for k in
           ("da_d0_h", "cong_d0_h", "zspread_d0_h", "gap_7d_h", "gap_28d_h", "zgap_28d_h")]
SPREAD_PAIRS = [(z, b) for z in C.ZONES for b in C.BORDERS]
ROWS = {"V5_border_inputs": dict(features="base+V5", rule="two_sided"),
        "V5_zone_border_spread": dict(features="none (trailing spread)", rule="spread",
                                      flag="not_tradable_as_virtual")}

_GRID: dict = {}


def grid(store=None) -> C.PriceGrid:
    key = id(store)
    if key not in _GRID:
        if store is not None and "prices_zone" in store:
            pz = store["prices_zone"]
        else:
            cols = ["delivery_hour"] + VALS + ["da_published_at", "rt_published_at"]
            pz = C.read_table("prices_zone", "delivery_hour", columns=["zone"] + cols)
            pz = pd.concat([pz, border_rows(cols)], ignore_index=True)
        _GRID[key] = C.PriceGrid(pz, C.ZONES + C.BORDERS, VALS)
    return _GRID[key]


def border_rows(cols) -> pd.DataFrame:
    """Border proxy rows: prices_gen/YYYYMM.parquet, point_type 'external', name in C.BORDERS (data
    agent, 7 Oct). Month files from 2024 on are never opened."""
    import pyarrow.parquet as pq
    parts = []
    for f in sorted((C.PARQUET_V2 / "prices_gen").glob("*.parquet")):
        if int(f.stem[:4]) >= C.HOLDOUT.year:
            continue
        t = pq.read_table(f, columns=["name", "point_type"] + cols,
                          filters=[("point_type", "=", "external")]).to_pandas()
        parts.append(t[t["name"].isin(C.BORDERS)].rename(columns={"name": "zone"}).drop(columns="point_type"))
    b = pd.concat(parts, ignore_index=True)
    C.assert_pre_holdout(b["delivery_hour"])
    return b


def _zgap_transform(b_idx):
    def f(g, sl, D):
        gap = C.gap_transform(g, sl, D)
        return gap - gap[:, :, [b_idx]]                 # every node minus border b
    return f


def features(panel_index: pd.MultiIndex, panel: pd.DataFrame | None = None, store=None) -> pd.DataFrame:
    p = C.panel_frame(panel_index)
    g = grid(store)
    days = sorted(p["bid_date"].unique())
    out = {}
    da0 = C.same_hour_on_D(g, "da_lbmp", "da", days)
    cg0 = C.same_hour_on_D(g, "da_congestion", "da", days)
    g7 = C.trailing(g, None, "rt", days, 7, transform=C.gap_transform)
    g28 = C.trailing(g, None, "rt", days, 28, transform=C.gap_transform)
    zone_da = C.to_rows(da0, days, g.nodes, p)
    for b in C.BORDERS:
        t = TAG[b]
        bi = g.nodes.index(b)
        out[f"bx_da_d0_h_{t}"] = C.to_rows(da0, days, g.nodes, p, fixed_node=b)
        out[f"bx_cong_d0_h_{t}"] = C.to_rows(cg0, days, g.nodes, p, fixed_node=b)
        out[f"bx_zspread_d0_h_{t}"] = zone_da - out[f"bx_da_d0_h_{t}"]
        out[f"bx_gap_7d_h_{t}"] = C.to_rows(g7, days, g.nodes, p, fixed_node=b)
        out[f"bx_gap_28d_h_{t}"] = C.to_rows(g28, days, g.nodes, p, fixed_node=b)
        zg = C.trailing(g, None, "rt", days, 28, transform=_zgap_transform(bi))
        out[f"bx_zgap_28d_h_{t}"] = C.to_rows(zg, days, g.nodes, p)
    res = pd.DataFrame(out)[COLUMNS].astype(float)
    res.index = panel_index
    return res


rule = C.two_sided


def border_labels(panel: pd.DataFrame, store=None) -> pd.DataFrame:
    """Scoring only (never a feature): each border's realised gap for the panel row's delivery hour."""
    g = grid(store)
    p = C.panel_frame(pd.MultiIndex.from_frame(panel[["bid_date", "zone", "delivery_hour"]]))
    d1 = p["bid_date"] + pd.Timedelta(days=1)
    di = g.days.get_indexer(d1)
    out = {}
    for b in C.BORDERS:
        bi = g.nodes.index(b)
        gap = g.v["rt_lbmp"][:, :, bi] - g.v["da_lbmp"][:, :, bi]
        v = np.full(len(p), np.nan)
        ok = di >= 0
        v[ok] = gap[di[ok], p["whour"].to_numpy()[ok]]
        out[TAG[b]] = v
    return pd.DataFrame(out, index=panel.index)


def spread_legs(panel: pd.DataFrame, feats: pd.DataFrame, costs, border: str) -> pd.DataFrame:
    """Zone-versus-border spread for one border: pos +1 = load zone and supply border, -1 the reverse,
    when the trailing spread clears load cost + supply cost. Returns zone legs (mw) plus the border leg
    as a separate column, for the scorer's two-leg view."""
    load_c, supply_c = (np.asarray(c, float) for c in costs)
    s = feats[f"bx_zgap_28d_h_{TAG[border]}"].to_numpy(float)
    pair = load_c + supply_c
    pos = np.where(s >= pair, 1, np.where(s <= -pair, -1, 0))
    pos = np.where(np.isnan(s), 0, pos)
    return pd.DataFrame({"delivery_hour": panel["delivery_hour"].to_numpy(), "zone": panel["zone"].to_numpy(),
                         "mw": pos.astype(float), "border": border, "border_mw": -pos.astype(float)},
                        index=panel.index)
