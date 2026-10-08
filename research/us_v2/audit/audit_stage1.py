"""Independent audit of v2 stage 1 (V2_limit_gbm, V1_C_deep structure, V3_tail_gbm sizing).

Reads only rows before 2024-01-01 (filtered at read time and asserted). Writes audit_stage1.json next to itself.
Uses rolling.py only for the fee table; prices, fills, limits, pair spreads and scoring are re-derived here.

    ~/nyiso-us/.venv/bin/python audit_stage1.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "v2" / "model"))
import rolling as R  # noqa: E402  fee table only

TZ = "America/New_York"
END = pd.Timestamp("2024-01-01", tz=TZ)
ZONES = R.ZONES
PAIRS = list(itertools.combinations(ZONES, 2))
YEARS = list(range(2013, 2024))
MULT = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)
OUT = {}
SUP, LOD = R.costs()


def chk(ts):
    assert pd.Series(ts).max() < END, "HOLDOUT BREACH"


def read(path, cols, tcol="delivery_hour"):
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(tcol).type
    t = d.to_table(filter=ds.field(tcol) < pa.scalar(END, type=typ), columns=cols).to_pandas()
    t[tcol] = t[tcol].dt.tz_convert(TZ)
    chk(t[tcol])
    return t


# ------------------------------------------------------------------ prices (scoring truth, not the panel)
px = read(HOME / "parquet_v2" / "prices_zone.parquet", ["delivery_hour", "zone", "da_lbmp", "rt_lbmp"])
px = px[px.zone.isin(ZONES)]
px["gap"] = px.rt_lbmp - px.da_lbmp
px["ddate"] = px.delivery_hour.dt.tz_localize(None).dt.normalize()
px["year"] = px.ddate.dt.year
KEY = ["delivery_hour", "zone"]


def score(pos: pd.DataFrame, label: str) -> dict:
    """pos: delivery_hour, zone, mw. Net = mw*gap - |mw|*cost(side), 2013..2023, rows with a settled gap."""
    x = px.merge(pos[KEY + ["mw"]], on=KEY, how="left")
    x = x[(x.year >= 2013) & x.gap.notna()]
    mw = x.mw.fillna(0.0).to_numpy()
    c = np.where(mw < 0, x.year.map(SUP), np.where(mw > 0, x.year.map(LOD), 0.0))
    x["pnl"] = mw * x.gap.to_numpy() - np.abs(mw) * c
    x["mwh"] = np.abs(mw)
    d = x.groupby("ddate")[["pnl", "mwh"]].sum()
    yr = d.groupby(d.index.year)["pnl"].sum().round().astype(int).to_dict()
    sh = float(d.pnl.mean() / d.pnl.std() * np.sqrt(365)) if d.pnl.std() > 0 else None
    r = {"years": yr, "total": int(round(d.pnl.sum())), "sharpe": round(sh, 2) if sh else None,
         "mwh_total": int(d.mwh.sum()), "mwh_per_day": round(float(d.mwh.mean()), 1),
         "usd_per_mwh": round(float(d.pnl.sum() / d.mwh.sum()), 3) if d.mwh.sum() else None,
         "positive_years": int(sum(v > 0 for v in yr.values()))}
    print(label, r["total"], r["sharpe"], r["usd_per_mwh"], flush=True)
    return r


def posdf(keys: pd.DataFrame, mw) -> pd.DataFrame:
    d = keys[KEY].copy()
    d["mw"] = np.nan_to_num(np.asarray(mw, float))
    return d


# ================================================================== Q1: V2 limit bids
g = read(HOME / "results" / "v2" / "preds" / "gbm.parquet",
         ["delivery_hour", "zone", "delivery_date", "mean", "q10", "q90"] + [f"dag_{k}" for k in range(9)] +
         [f"cond_{k}" for k in range(9)])
pan = read(HOME / "parquet_v2" / "features" / "panel_2010_2023.parquet",
           ["delivery_hour", "zone", "y_da_lbmp", "da_d0_h"])
g = g.merge(pan, on=KEY, how="left", validate="one_to_one")
g = g.merge(px[KEY + ["da_lbmp", "rt_lbmp"]], on=KEY, how="left", validate="one_to_one")
g["year"] = g.delivery_hour.dt.tz_localize(None).dt.year
q1 = {}
# (a) the grid levels are D's same-hour day-ahead price times the multipliers, nothing realised
lv = np.stack([g[f"dag_{k}"].to_numpy(float) for k in range(9)], 1)
exp = g.da_d0_h.to_numpy(float)[:, None] * np.array(MULT)[None, :]
ok = np.isfinite(lv) & np.isfinite(exp)
q1["levels_equal_da_d0_times_mult_max_abs_diff"] = float(np.abs(lv - exp)[ok].max())
q1["levels_equal_realised_da_share"] = float(np.mean(np.isclose(lv, g.y_da_lbmp.to_numpy(float)[:, None]).any(1)))
q1["panel_y_da_vs_prices_da_max_abs_diff"] = float(np.nanmax(np.abs(g.y_da_lbmp - g.da_lbmp)))

# (b) independent limit and side, written as an explicit scan (not the cumprod form of strategies_v2)
C = np.stack([g[f"cond_{k}"].to_numpy(float) for k in range(9)], 1)
sup = g.year.map(SUP).to_numpy(float)
lod = g.year.map(LOD).to_numpy(float)
n = len(g)
side = np.zeros(n)
limit = np.full(n, np.nan)
valid = np.isfinite(C).all(1) & np.isfinite(lv).all(1)
o = np.argsort(np.nan_to_num(lv), 1)
Ls, Cs = np.take_along_axis(lv, o, 1), np.take_along_axis(C, o, 1)
for r in np.where(valid)[0]:
    L, c = Ls[r], Cs[r]
    s_lim = None
    for k in range(8, -1, -1):              # from the highest level down while supply stays profitable
        if -c[k] - sup[r] > 0:
            s_lim = L[k]
        else:
            break
    l_lim = None
    for k in range(9):                      # from the lowest level up while load stays profitable
        if c[k] - lod[r] > 0:
            l_lim = L[k]
        else:
            break
    es, el = -c[4] - sup[r], c[4] - lod[r]
    if s_lim is not None and (l_lim is None or es >= el):
        side[r], limit[r] = -1.0, s_lim
    elif l_lim is not None:
        side[r], limit[r] = 1.0, l_lim
da = g.da_lbmp.to_numpy(float)                 # realised DA from the price file decides the fill, nothing else
fill = np.where(side < 0, da >= limit, np.where(side > 0, da <= limit, False)) & np.isfinite(da)
mine = np.where(fill, side, 0.0)
saved = pd.read_parquet(HOME / "results" / "v2" / "pos" / "V2_limit_gbm.parquet")
chk(saved.delivery_hour)
cmp = g[KEY].assign(mine=mine).merge(saved, on=KEY, how="left")
q1["positions_identical_share"] = float(np.mean(cmp.mine.to_numpy() == cmp.mw.fillna(0).to_numpy()))
q1["max_abs_mw"] = float(np.abs(saved.mw).max())
q1["mine"] = score(posdf(g, mine), "V2 mine")
q1["saved_positions"] = score(saved, "V2 saved")
sc = g.year >= 2013
q1["bids_per_day"] = round(float((side[sc] != 0).sum() / g[sc].delivery_hour.dt.date.nunique()), 1)
q1["fill_rate"] = round(float(fill[sc & (side != 0)].mean()), 3)
q1["share_supply_of_fills"] = round(float((mine[sc] < 0).sum() / (mine[sc] != 0).sum()), 3)
# where does the limit sit relative to D's price
ratio = np.round(limit / g.da_d0_h.to_numpy(float), 3)
lr = pd.Series(ratio[sc & (side != 0)])
q1["limit_multiplier_distribution_supply"] = (pd.Series(ratio[sc & (side < 0)]).value_counts(normalize=True)
                                              .round(3).head(9).to_dict())
q1["limit_multiplier_distribution_load"] = (pd.Series(ratio[sc & (side > 0)]).value_counts(normalize=True)
                                            .round(3).head(9).to_dict())
# (c) decomposition: model side always filled (no limit); fills by binding vs non-binding limit
q1["model_side_no_limit"] = score(posdf(g, side), "V2 side no limit")
d0 = g.da_d0_h.to_numpy(float)
lowest = np.where(side < 0, np.isclose(limit, np.nanmin(lv, 1)), np.isclose(limit, np.nanmax(lv, 1)))
q1["fills_with_widest_limit_(market_like)"] = score(posdf(g, np.where(lowest, mine, 0.0)), "V2 widest-limit fills")
q1["fills_with_binding_limit"] = score(posdf(g, np.where(~lowest, mine, 0.0)), "V2 binding-limit fills")
# (d) no-model price-sensitive rules: supply when DA clears >= m x D's same-hour DA; load when DA <= m x D's DA
nm = {}
for m in (1.0, 1.15, 1.3, 1.6, 2.0):
    s = np.where(np.isfinite(da) & np.isfinite(d0) & (d0 > 1) & (da >= m * d0), -1.0, 0.0)
    nm[f"supply_if_DA>={m}xD"] = score(posdf(g, s), f"static supply {m}")
for m in (0.85, 0.7, 0.5):
    s = np.where(np.isfinite(da) & np.isfinite(d0) & (d0 > 1) & (da <= m * d0), 1.0, 0.0)
    nm[f"load_if_DA<={m}xD"] = score(posdf(g, s), f"static load {m}")
s = np.where(np.isfinite(da) & (d0 > 1) & (da >= 1.15 * d0), -1.0, np.where(np.isfinite(da) & (d0 > 1) & (da <= 0.85 * d0), 1.0, 0.0))
nm["supply>=1.15xD_load<=0.85xD"] = score(posdf(g, s), "static both")
s = np.where(np.isfinite(da) & (d0 > 1) & (da >= d0), -1.0, 0.0)
s2 = np.where(np.isfinite(da) & (d0 > 1) & (da >= 0.0), -1.0, 0.0)
nm["supply_every_hour_(always_supply)"] = score(posdf(g, s2), "always supply")
# the same no-model fill rule, but only on the zone-hours and sides V2 bid on
nm["V2_sides_with_static_limit_1.0xD"] = score(posdf(g, np.where(
    (side < 0) & (da >= d0) | (side > 0) & (da <= d0), side, 0.0)), "V2 sides static limit 1.0")
q1["no_model_rules"] = nm
# (e) realised information in the side: is the side correlated with the realised DA move beyond D's price?
mv = (da - d0)
q1["mean_realised_DA_minus_D_when_side_supply"] = round(float(np.nanmean(mv[sc & (side < 0)])), 2)
q1["mean_realised_DA_minus_D_when_side_load"] = round(float(np.nanmean(mv[sc & (side > 0)])), 2)
q1["mean_realised_DA_minus_D_all"] = round(float(np.nanmean(mv[sc])), 2)
OUT["Q1_V2"] = q1

# ================================================================== Q3: V3 sizes
v3 = pd.read_parquet(HOME / "results" / "v2" / "pos" / "V3_tail_gbm.parquet")
chk(v3.delivery_hour)
v3s = v3[v3.delivery_hour.dt.tz_localize(None).dt.year >= 2013]
q3 = {"max_abs_mw": float(np.abs(v3s.mw).max()), "share_at_cap_3": round(float((np.abs(v3s.mw) >= 2.999).mean()), 3),
      "share_nonzero": round(float((v3s.mw != 0).mean()), 3),
      "mean_abs_mw_when_open": round(float(np.abs(v3s.mw[v3s.mw != 0]).mean()), 3)}
q3["saved"] = score(v3, "V3 saved")
q3["same_sides_1MW"] = score(v3.assign(mw=np.sign(v3.mw)), "V3 sides 1MW")
q3["saved_divided_by_3"] = score(v3.assign(mw=v3.mw / 3.0), "V3 /3")
OUT["Q3_V3"] = q3

# ================================================================== Q2: C deep, static pair rule
hours = px.pivot_table(index="delivery_hour", columns="zone", values="gap", aggfunc="first").reindex(columns=ZONES)
chk(hours.index)
G = hours.to_numpy(float)
H = hours.index
dd = pd.Series(H.tz_localize(None).normalize())
hod = H.hour.to_numpy()
yr = dd.dt.year.to_numpy()
pc_h = np.array([SUP[y] + LOD[y] for y in yr])
cj = json.loads((HOME / "results" / "v2" / "pos" / "V1_C_deep.json").read_text())["choices"]
cpos = pd.read_parquet(HOME / "results" / "v2" / "pos" / "V1_C_deep.parquet")
chk(cpos.delivery_hour)
CW = cpos.pivot_table(index="delivery_hour", columns="zone", values="mw", aggfunc="sum").reindex(index=H, columns=ZONES).fillna(0).to_numpy()


def spread(k):
    i, j = PAIRS[k]
    return G[:, ZONES.index(i)] - G[:, ZONES.index(j)]


SP = np.stack([spread(k) for k in range(len(PAIRS))], 1)            # hours x 55, NaN where a gap is missing


def legs_to_zones(legs):                                            # legs: hours x 55 in {-1,0,1}
    Z = np.zeros((len(H), len(ZONES)))
    for k, (i, j) in enumerate(PAIRS):
        Z[:, ZONES.index(i)] += legs[:, k]
        Z[:, ZONES.index(j)] -= legs[:, k]
    return Z


def wide_to_pos(Z):
    d = pd.DataFrame(Z, index=H, columns=ZONES).stack().rename("mw").reset_index()
    d.columns = ["delivery_hour", "zone", "mw"]
    return d


qs = [q for q in R.quarters() if q[0].year >= 2013]
pidx = {"|".join(p): k for k, p in enumerate(PAIRS)}
static_same_all = np.zeros_like(SP)      # C's pairs, static side per pair from the 3-year window, every hour
static_same_hod = np.zeros_like(SP)      # C's pairs, static side per pair x local hour, every hour
static_full = np.zeros_like(SP)          # no model at all: pairs chosen by trailing-4Q net of the static rule
static_on_c_hours = np.zeros((len(H), len(ZONES)))   # C's traded hours (zone positions) with the static side
agree = []
for q in qs:
    qs_m = ((dd >= pd.Timestamp(q[0])) & (dd <= pd.Timestamp(q[1]))).to_numpy()
    lo = pd.Timestamp(q[0]) - pd.DateOffset(years=3)
    hi = pd.Timestamp(q[0]) - pd.Timedelta(days=2)
    tr = ((dd >= lo) & (dd <= hi)).to_numpy()
    mu_all = np.nanmean(SP[tr], 0)
    mu_hod = np.full((24, len(PAIRS)), np.nan)
    for h in range(24):
        mu_hod[h] = np.nanmean(SP[tr & (hod == h)], 0)
    side_hod = np.sign(np.nan_to_num(mu_hod))[hod[qs_m]]
    chosen = [pidx[p] for p in cj[q[2]]["pairs"]]
    for k in chosen:
        static_same_all[qs_m, k] = np.sign(mu_all[k])
        static_same_hod[qs_m, k] = side_hod[:, k]
    # model-free pair choice: trailing 4 quarters (public by q start - 2 days), static per-hour side from the
    # 3-year window ending at each trailing quarter's own start (approximated by this window), net after cost
    t4 = ((dd >= pd.Timestamp(q[0]) - pd.DateOffset(months=12)) & (dd <= hi)).to_numpy()
    lo4 = pd.Timestamp(q[0]) - pd.DateOffset(years=4)
    hi4 = pd.Timestamp(q[0]) - pd.DateOffset(months=12) - pd.Timedelta(days=2)
    tr4 = ((dd >= lo4) & (dd <= hi4)).to_numpy()
    mu4 = np.full((24, len(PAIRS)), np.nan)
    for h in range(24):
        mu4[h] = np.nanmean(SP[tr4 & (hod == h)], 0)
    s4 = np.sign(np.nan_to_num(mu4))[hod[t4]]
    net4 = np.nansum(s4 * SP[t4] - np.abs(s4) * pc_h[t4][:, None], 0)
    best = [k for k in np.argsort(-net4) if net4[k] > 0][:5]
    for k in best:
        static_full[qs_m, k] = side_hod[:, k]
    # agreement: C's zone positions vs the static legs of the same chosen pairs
    Zs = legs_to_zones(np.where(qs_m[:, None], static_same_hod, 0))[qs_m]
    Zc = CW[qs_m]
    on = Zc != 0
    static_on_c_hours[np.where(qs_m)[0]] = np.where(on, np.sign(Zs) * np.abs(Zc), 0)
    agree.append({"q": q[2], "c_open_zone_hours": int(on.sum()),
                  "same_sign_as_static": round(float((np.sign(Zc[on]) == np.sign(Zs[on])).mean()), 3) if on.any() else None})


def ok_legs(L):
    return np.where(np.isfinite(SP), L, 0.0)


q2 = {}
q2["C_deep_saved"] = score(cpos, "C deep saved")
q2["static_C_pairs_side_per_pair"] = score(wide_to_pos(legs_to_zones(ok_legs(static_same_all))), "static same pairs all")
q2["static_C_pairs_side_per_pair_hour"] = score(wide_to_pos(legs_to_zones(ok_legs(static_same_hod))), "static same pairs hod")
q2["static_no_model_pairs_side_per_pair_hour"] = score(wide_to_pos(legs_to_zones(ok_legs(static_full))), "static full")
q2["C_hours_with_static_side"] = score(wide_to_pos(static_on_c_hours), "C hours static side")
Zc_s = np.sign(CW)
Zs_s = np.sign(legs_to_zones(ok_legs(static_same_hod)))
on = (Zc_s != 0) & (yr[:, None] >= 2013)
q2["share_C_zone_hours_same_sign_as_static"] = round(float((Zc_s[on] == Zs_s[on]).mean()), 3)
q2["C_pnl_where_agrees_with_static"] = score(wide_to_pos(np.where(on & (Zc_s == Zs_s), CW, 0)), "C agree")
q2["C_pnl_where_disagrees_with_static"] = score(wide_to_pos(np.where(on & (Zc_s != Zs_s), CW, 0)), "C disagree")
# persistence: same zone-hour position as the previous day
cp = cpos.copy()
cp["ddate"] = cp.delivery_hour.dt.tz_localize(None).dt.normalize()
cp["h"] = cp.delivery_hour.dt.hour
cp = cp[cp.ddate.dt.year >= 2013].sort_values(["zone", "h", "ddate"])
prev = cp.groupby(["zone", "h"]).mw.shift(1)
q2["share_zone_hours_same_position_as_day_before"] = round(float((cp.mw == prev)[cp.mw != 0].mean()), 3)
q2["share_open_zone_hours"] = round(float((cp.mw != 0).mean()), 3)
q2["per_quarter_agreement"] = agree
OUT["Q2_C"] = q2

(Path(__file__).parent / "audit_stage1.json").write_text(json.dumps(OUT, indent=1, default=str))
print("done")
