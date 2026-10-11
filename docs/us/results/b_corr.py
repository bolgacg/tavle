"""Part B: sleeve correlations and two proposed books on the scorers' daily net P&L at 1 MW. Read-only."""
import json
import numpy as np, pandas as pd
RUN = "/home/bo/nyiso-us_heldout/run"; OUT = "/home/bo/analysis_20261011"
L = []
def P(*a): L.append(" ".join(str(x) for x in a))
v2 = pd.read_parquet(f"{RUN}/ho_v2/results/v2/v2_daily_all_heldout.parquet")
v1 = pd.read_parquet(f"{RUN}/ho_v1/side/lab_daily_2021_202609.parquet")
v1.index = pd.to_datetime(v1.index)
i13b = pd.read_parquet(f"{RUN}/ho_v1/side/idea13_build_daily.parquet"); i13b.index = pd.to_datetime(i13b.index)
ov = i13b.index.intersection(v1.index)
P("idea13 build file vs lab_daily 2021-23: max |diff|", float((i13b.loc[ov, "idea13_portfolio"] - v1.loc[ov, "idea13_portfolio"]).abs().max()))
ov2 = v1.index.intersection(v2.index)
P("v1 always_supply vs v2 V1_always_supply: max |diff|", float((v1.loc[ov2, "always_supply"] - v2.loc[ov2, "V1_always_supply"]).abs().max()))
D = pd.DataFrame({"pairs": v2["C_static_pairs"], "V1_C_deep": v2["V1_C_deep"], "V15c": v2["V15c_wx_border_gbm"],
                  "V16": v2["V16_B_gefs_joined"]})
D["always_supply"] = v1["always_supply"].reindex(D.index)
D["idea13"] = v1["idea13_portfolio"].reindex(D.index)
D["v1_C_deep"] = v1["C_deep"].reindex(D.index)
# reconciliation with published yearly totals
P("\nyearly sums (reconcile with v2_results.json / v2_results_heldout.md / lab results)")
P(D.groupby(D.index.year).sum(min_count=1).round().to_string())
ev = ((D.index >= "2026-01-24") & (D.index <= "2026-02-02")) | ((D.index >= "2026-06-28") & (D.index <= "2026-07-05"))
build = (D.index.year >= 2013) & (D.index.year <= 2023)
ho = D.index >= "2024-01-01"
def corr_block(sel, title):
    X = D[sel]
    P(f"\n### {title}: days {sel.sum()}")
    P("Pearson (pairwise complete)\n" + X.corr("pearson").round(2).to_string())
    P("Spearman\n" + X.corr("spearman").round(2).to_string())
    # tail: for each sleeve's 20 worst days, mean P&L of others in z-units and correlation on those days
    T = {}
    for c in X.columns:
        x = X[c].dropna()
        w = x.nsmallest(20).index
        T[c] = {o: round(float((X.loc[w, o] < 0).mean()), 2) for o in X.columns}
    P("tail co-movement: share of row-sleeve's 20 worst days on which column-sleeve also lost\n" + pd.DataFrame(T).T.round(2).to_string())
    T2 = {}
    for c in X.columns:
        w = X[c].dropna().nsmallest(20).index
        T2[c] = X.loc[w].corr("pearson")[c].round(2)
    P("Pearson on each row-sleeve's 20 worst days\n" + pd.DataFrame(T2).T.to_string())
corr_block(build, "2013-2023 (v1 rows 2021-2023 only)")
corr_block(ho, "held-out 2024-01-01..2026-09-30")
corr_block(ho & ~ev, "held-out without the two event windows")
corr_block(build & ~ev, "2013-2023 (no event days in this span; same as above, check)") if False else None
# books
bj = {r["name"]: r for r in json.load(open("/home/bo/nyiso-us/results/v2/v2_results.json"))["rows"]}
sp = bj["C_static_pairs"]["sizing_view"]["scale_for_100k_drawdown"]; sv = bj["V15c_wx_border_gbm"]["sizing_view"]["scale_for_100k_drawdown"]
P(f"\nbuild-year 100k-drawdown scales: pairs {sp} MW, V15c {sv} MW")
books = {"unlevered 0.8 pairs + 0.2 V15c": (0.8 * sp, 0.2 * sv, 100_000), "levered 1.6 pairs + 0.2 V15c": (1.6 * sp, 0.2 * sv, 175_000)}
def st(d):
    cum = np.concatenate([[0.0], np.cumsum(d.to_numpy())])
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {"days": len(d), "net": round(float(d.sum())), "ret/yr on 500k %": round(100 * float(d.sum()) / 500_000 * 365 / len(d), 1),
            "Sharpe": round(float(d.mean() / d.std() * np.sqrt(365)), 2), "worst DD": round(dd), "worst day": round(float(d.min())),
            "worst day date": str(d.idxmin().date())}
for name, (wp, wv, tgt) in books.items():
    B = wp * D["pairs"] + wv * D["V15c"]
    rows = {}
    for lab_, sel in [("2013-2023", build), ("held-out", ho), ("held-out w/o events", ho & ~ev)] + \
                     [(str(y), D.index.year == y) for y in (2024, 2025, 2026)]:
        r = st(B[sel]); r["DD target held"] = r["worst DD"] >= -tgt; rows[lab_] = r
    P(f"\n### book {name}: pairs at {wp:.3f} MW, V15c at {wv:.3f} MW, DD target {tgt:,}")
    P(pd.DataFrame(rows).T.to_string())
    # each leg's contribution in held-out
    P("held-out net by leg: pairs", round(float((wp * D['pairs'][ho]).sum())), " V15c", round(float((wv * D['V15c'][ho]).sum())),
      "; V15c on event days", round(float((wv * D['V15c'][ho & ev]).sum())))
open(f"{OUT}/b_corr.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
