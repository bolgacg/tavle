"""Part A: why the C pair rows earned so much in 2025. Read-only.
Build mode (the held-out lock stays shut, no guard is touched). Prices are read directly from the held-out tables;
the frame below mirrors score_v2.FrameV2 field by field, positions go through score_v2.positions_to_mw, and the
money is lab.Frame.day_pnl itself with the fee table rolling.costs() (pipeline_v2/fees.py, covers 2010..2026)."""
import os, sys, json, types
sys.path.insert(0, "/home/bo/nyiso-us_heldout/tavle/research/us_v2/model")
import numpy as np, pandas as pd, pyarrow.dataset as ds
import rolling as R
import score_v2 as S

OUT = "/home/bo/analysis_20261011"
RUN = "/home/bo/nyiso-us_heldout/run"
lab = S.import_lab()
sup, lod = R.costs()
cols = ["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "da_energy", "rt_energy", "da_loss", "rt_loss",
        "da_congestion", "rt_congestion"]
px = ds.dataset(f"{RUN}/ho_data/parquet_v2/prices_zone.parquet").to_table(
    columns=cols, filter=ds.field("zone").isin(R.ZONES)).to_pandas()
h = px.copy()
h["delivery_hour"] = h["delivery_hour"].dt.tz_convert(R.TZ)
loc = h["delivery_hour"].dt.tz_localize(None)
h = h[(loc >= pd.Timestamp("2013-01-01")) & (loc < pd.Timestamp("2026-10-01"))]
h = h.sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
h["gap"] = h["rt_lbmp"] - h["da_lbmp"]
h = h[h["gap"].notna()].reset_index(drop=True)
h["ddate"] = h["delivery_hour"].dt.tz_localize(None).dt.normalize()
h["hour"] = h["delivery_hour"].dt.hour
F = types.SimpleNamespace()
F.h = h
F.days = pd.DatetimeIndex(sorted(h["ddate"].unique()))
F.ND = len(F.days)
F.DI = F.days.get_indexer(h["ddate"])
_yr = h["ddate"].dt.year
F.GAP = h["gap"].to_numpy(float)
F.SUPC = _yr.map(sup).to_numpy(float)
F.LOADC = _yr.map(lod).to_numpy(float)
F.day_pnl = lambda mw, cost=None, gap=None: lab.Frame.day_pnl(F, mw, cost, gap)
for c in ("energy", "loss", "congestion"):
    h["g_" + c] = h["rt_" + c] - h["da_" + c]
resid = (h["g_energy"] + h["g_loss"] + h["g_congestion"] - h["gap"]).abs()
resid_neg = (h["g_energy"] + h["g_loss"] - h["g_congestion"] - h["gap"]).abs()
sign = 1.0 if resid.median() <= resid_neg.median() else -1.0
print("component identity: median |E+L+C-gap|", resid.median(), " |E+L-C-gap|", resid_neg.median(), "-> sign", sign)
h["g_congestion"] *= sign

daily_all = pd.read_parquet(f"{RUN}/ho_v2/results/v2/v2_daily_all_heldout.parquet")
ROWS = ["V1_C_deep", "C_static_pairs"]
MW = {}
for n in ROWS:
    pos = pd.read_parquet(f"{RUN}/ho_v2/results/v2/pos/{n}.parquet")
    MW[n] = S.positions_to_mw(F, pos)
    d = pd.Series(F.day_pnl(MW[n]), index=F.days)
    ref = daily_all[n].reindex(F.days)
    yr = pd.DataFrame({"recomputed": d.groupby(d.index.year).sum().round(), "scorer": ref.groupby(ref.index.year).sum().round()})
    print(n, "max |daily diff|", float((d - ref).abs().max())); print(yr.to_string())

h["year"] = h["ddate"].dt.year; h["month"] = h["ddate"].dt.month
blk = pd.cut(h["hour"], [-1, 5, 9, 15, 21, 23], labels=["00-05", "06-09", "10-15", "16-21", "22-23"])
h["block"] = blk.astype(str)
cost = np.where(MW[ROWS[0]] < 0, F.SUPC, F.LOADC)
res = {}
for n in ROWS:
    mw = MW[n]
    c = np.where(mw < 0, F.SUPC, np.where(mw > 0, F.LOADC, 0.0))
    h[n + "_net"] = mw * F.GAP - np.abs(mw) * c
    h[n + "_fee"] = -np.abs(mw) * c
    for k in ("energy", "loss", "congestion"):
        h[f"{n}_{k}"] = mw * h["g_" + k].to_numpy()
    h[n + "_mw"] = mw
per = lambda y: h["year"] == y if isinstance(y, int) else h["year"].between(2013, 2023)
L = []
def P(*a): L.append(" ".join(str(x) for x in a))
for n in ROWS:
    P(f"\n## {n}")
    # components by period, per year
    rows = []
    for lab_, sel, ny in [("2013-23 avg", per("b"), 11), ("2024", per(2024), 1), ("2025", per(2025), 1), ("2026 Jan-Sep", per(2026), 1)]:
        s = h.loc[sel, [n + "_net", n + "_energy", n + "_loss", n + "_congestion", n + "_fee"]].sum() / ny
        rows.append(s.rename(lab_))
    P("components per year (USD, 1 MW): net = energy + loss + congestion + fees\n" + pd.DataFrame(rows).round().to_string())
    # by zone: net and mean signed MW
    rows = []
    for lab_, sel, ny in [("2013-23 avg", per("b"), 11), ("2024", per(2024), 1), ("2025", per(2025), 1)]:
        g = h[sel].groupby("zone")
        rows.append(pd.DataFrame({f"{lab_} net": g[n + "_net"].sum() / ny, f"{lab_} cong": g[n + "_congestion"].sum() / ny,
                                  f"{lab_} meanMW": g[n + "_mw"].mean()}))
    P("by zone (+MW = load/long DA, -MW = supply/short DA)\n" + pd.concat(rows, axis=1).round(2).to_string())
    m = h[per(2025)].groupby("month")[[n + "_net", n + "_congestion"]].sum().round()
    m["2024 net"] = h[per(2024)].groupby("month")[n + "_net"].sum().round()
    m["2013-23 avg net"] = (h[per("b")].groupby("month")[n + "_net"].sum() / 11).round()
    P("by month\n" + m.to_string())
    b = pd.DataFrame({"2025": h[per(2025)].groupby("block")[n + "_net"].sum(), "2024": h[per(2024)].groupby("block")[n + "_net"].sum(),
                      "2013-23 avg": h[per("b")].groupby("block")[n + "_net"].sum() / 11}).round()
    P("by hour block (hour beginning, local)\n" + b.to_string())
    dd = h[per(2025)].groupby("ddate")[[n + "_net", n + "_congestion"]].sum()
    top = dd.sort_values(n + "_net", ascending=False).head(10)
    # which zone gap drove each top day: CAPITL and NYC mean gap
    gz = h[per(2025)].pivot_table(index="ddate", columns="zone", values="gap", aggfunc="mean")
    top["CAPITL gap"] = gz.loc[top.index, "CAPITL"]; top["NYC gap"] = gz.loc[top.index, "N.Y.C."]; top["LONGIL gap"] = gz.loc[top.index, "LONGIL"]
    P(f"top 10 days 2025 (sum {top[n + '_net'].sum():,.0f} of {dd[n + '_net'].sum():,.0f})\n" + top.round(1).to_string())
# model vs static rule, 2025 and other periods
a, s = "V1_C_deep", "C_static_pairs"
ma, ms = h[a + "_mw"].to_numpy(), h[s + "_mw"].to_numpy()
cat = np.select([(ma == 0) & (ms == 0), np.sign(ma) == np.sign(ms), ms == 0, ma == 0],
                ["both flat", "same side", "model only", "static only"], "opposite sides")
h["cat"] = cat
P("\n## V1_C_deep minus C_static_pairs (difference earned), by overlap of zone-hour positions")
for lab_, sel, ny in [("2013-23 avg", per("b"), 11), ("2024", per(2024), 1), ("2025", per(2025), 1), ("2026", per(2026), 1)]:
    g = h[sel].groupby("cat")
    t = pd.DataFrame({"zone-hours": g.size() / ny, "model net": g[a + "_net"].sum() / ny, "static net": g[s + "_net"].sum() / ny})
    t["difference"] = t["model net"] - t["static net"]
    P(f"{lab_}\n" + t.round().to_string())
dz = h[per(2025)].groupby("zone")[[a + "_net", s + "_net"]].sum(); dz["diff"] = dz[a + "_net"] - dz[s + "_net"]
P("2025 difference by zone\n" + dz.round().to_string())
# 2025 CAPITL-NYC spread facts
for y in (2024, 2025):
    sel = per(y)
    gz = h[sel].pivot_table(index="delivery_hour", columns="zone", values=["gap", "g_congestion", "da_congestion", "rt_congestion"], aggfunc="first")
    sp = gz[("gap", "CAPITL")] - gz[("gap", "N.Y.C.")]
    spc = gz[("g_congestion", "CAPITL")] - gz[("g_congestion", "N.Y.C.")]
    P(f"{y} CAPITL minus NYC hourly gap (RT-DA): mean {sp.mean():.2f}, congestion part {spc.mean():.2f}, share>0 {float((sp>0).mean()):.2f}; "
      f"mean DA cong CAPITL {gz[('da_congestion','CAPITL')].mean():.2f} NYC {gz[('da_congestion','N.Y.C.')].mean():.2f}; "
      f"RT cong CAPITL {gz[('rt_congestion','CAPITL')].mean():.2f} NYC {gz[('rt_congestion','N.Y.C.')].mean():.2f}")
spb = []
for y in range(2013, 2027):
    sel = per(y)
    gz = h[sel].pivot_table(index="delivery_hour", columns="zone", values="gap", aggfunc="first")
    spb.append((y, round(float((gz["CAPITL"] - gz["N.Y.C."]).mean()), 2), round(float((gz["CAPITL"] - gz["LONGIL"]).mean()), 2)))
P("mean hourly (RT-DA) spread CAPITL-NYC, CAPITL-LONGIL by year: " + str(spb))
open(f"{OUT}/a_2025.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
