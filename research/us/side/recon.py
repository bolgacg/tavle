"""Reconcile the lab's storm filter, storm flip and idea 9 with the lead modeller's list (build years only).

Writes ~/nyiso-us/side/recon.json. Uses lab.py's loaders (holdout filter at load) and the lead's exported storm
score (side/lead_storm_score.parquet).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lab as L  # noqa: E402

px, lf, wx = L.load_tables()
F = L.Frame(px)
out = {}

# ---------------------------------------------------------------- idea 9 (tail spike S = 100)
tail, _ = L.load_tail_cube(F)
st9 = L.mk_spike_rule("t", "9", "", "p_tail", tail)
r = {}
for Y in L.YEARS:
    sel = F.YEAR == Y
    r[str(Y)] = {str(ps): round(float(F.day_pnl(st9.pos(F, st9.inputs, ps))[F.day_year == Y].sum())) for ps in L.PSTAR_GRID}
    m = L.choice_mask(F, Y - 1, Y)
    if Y > 2021:
        r[str(Y)]["chosen_on_prior_year_wf_predictions"] = {
            str(ps): round(float(F.day_pnl(st9.pos(F, st9.inputs, ps))[m].mean()), 2) for ps in L.PSTAR_GRID}
out["idea9_net_by_pstar"] = r

# ---------------------------------------------------------------- storm score: lab (storm_value) vs lead
s_lab, _ = L.storm_score(L.storm_inputs(F, px, lf, wx))


def storm_inputs_rule(F, px, lf, wx):
    """As lab.storm_inputs, but GFS by the pipeline's three-day rule (day2 to 21:00, day3 for 22 and 23 and
    where day2 is missing) and the load forecast as the newest public value per target hour."""
    T = pd.Series(F.t05(), index=F.bid)
    l = lf.assign(D=L._bid_of_target(lf["target_hour"]))
    l = l[l["D"].isin(F.bid)]
    l = l[l["published_at"] <= l["D"].map(T)]
    l = l.sort_values("published_at", kind="stable").drop_duplicates(["D", "target_hour"], keep="last")
    w = wx.assign(D=L._bid_of_target(wx["target_hour"]))
    w = w[w["D"].isin(F.bid)]
    w = w[w["published_at"] <= w["D"].map(T)]
    hr = w["target_hour"].dt.hour
    w = w[((w["run_lead_hours"] == 48) & (hr <= 21)) | (w["run_lead_hours"] == 72)]
    w = w.sort_values(["point", "target_hour", "run_lead_hours"], kind="stable").drop_duplicates(["point", "target_hour"])
    base = L.storm_inputs(F, px, lf, wx)
    base["peak"] = l.groupby("D")["load_forecast_mw"].max()
    base["tmin"] = w.groupby("D")["temperature_2m_c"].min()
    base["tmax"] = w.groupby("D")["temperature_2m_c"].max()
    return base


s_rule, _ = L.storm_score(storm_inputs_rule(F, px, lf, wx))
ls = pd.read_parquet(L.SIDE / "lead_storm_score.parquet")
ls["delivery_date"] = pd.to_datetime(ls["delivery_date"])
ls = ls[ls["delivery_date"] < L.END_NAIVE]
s_lead = ls.set_index("delivery_date")["storm"].reindex(F.days).to_numpy(float)
sc = np.isin(F.day_year, [2020] + L.YEARS)
both = sc & np.isfinite(s_lab) & np.isfinite(s_lead)
out["storm_score_compare_2020_2023"] = {
    "days": int(both.sum()),
    "lab_vs_lead_equal_days": int((np.abs(s_lab - s_lead) < 1e-9)[both].sum()),
    "lab_vs_lead_max_abs_diff": float(np.nanmax(np.abs(s_lab - s_lead)[both])),
    "lab_three_day_rule_vs_lead_equal_days": int((np.abs(s_rule - s_lead) < 1e-9)[both].sum()),
    "lab_three_day_rule_vs_lead_max_abs_diff": float(np.nanmax(np.abs(s_rule - s_lead)[both])),
    "flag_differs_at_0.80_lab_vs_lead": int(((s_lab > 0.8) != (s_lead > 0.8))[both].sum()),
    "flag_differs_at_0.90_lab_vs_lead": int(((s_lab > 0.9) != (s_lead > 0.9))[both].sum()),
}

# ---------------------------------------------------------------- each storm score x each choice procedure
strats = {s.name: s for s in L.mk_storm(F, s_lab)[:2]}


def run(score, name, procedure):
    st = strats[name]
    inp = {"storm": score}
    res = {}
    for Y in L.YEARS:
        if procedure == "lab (prior year to 30 Dec, total net)":
            m = L.choice_mask(F, Y - 1, Y)
        elif procedure == "storm_value (prior year to 31 Dec, total net)":
            m = F.day_year == Y - 1
        else:                     # the lead: the 365 delivery days before 1 January of Y, mean daily net
            m = (F.days >= pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=365)) & (F.days < pd.Timestamp(f"{Y}-01-01"))
        vals = {c: float(F.day_pnl(st.pos(F, inp, c))[m].sum()) for c in L.CUTS}
        c = max(L.CUTS, key=lambda q: vals[q])
        res[str(Y)] = {"cut": c, "net": round(float(F.day_pnl(st.pos(F, inp, c))[F.day_year == Y].sum())),
                       "prior_net_by_cut": {str(q): round(v) for q, v in vals.items()}}
    res["total"] = sum(res[str(Y)]["net"] for Y in L.YEARS)
    return res


grid = {}
for sname, score in (("lab score (storm_value inputs)", s_lab), ("lab score, three-day GFS rule", s_rule),
                     ("lead's score", s_lead)):
    for name in strats:
        for proc in ("storm_value (prior year to 31 Dec, total net)", "lab (prior year to 30 Dec, total net)",
                     "lead (365 days before, mean daily net)"):
            grid[f"{name} | {sname} | {proc}"] = run(score, name, proc)
out["storm_grid"] = grid
(L.SIDE / "recon.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in grid.items():
    print(k, [(v[str(Y)]["cut"], v[str(Y)]["net"]) for Y in L.YEARS], v["total"])
print(json.dumps(out["storm_score_compare_2020_2023"]))
print(json.dumps(out["idea9_net_by_pstar"]))
