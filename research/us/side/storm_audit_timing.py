"""Audit part 1: lookahead in the storm score of storm_value.py. Build years only (holdout enforced at load).

(a) runs the script's OWN per-day input block (extracted verbatim from ~/nyiso-us/storm_value.py) and this
    audit's independent code on the same holdout-filtered tables, and compares them;
(b) checks every row each input uses against the published_at rules in pipeline/README.md;
(c) compares the script's deadline (midnight + 5 h) with the wall-clock 05:00 on DST days;
(d) injects future data on chosen days and shows the score moves only when the rule is broken.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from storm_audit_lib import (BID_DAYS, END, OUT, TZ, assert_holdout, deadlines, inputs, load_lf,
                             load_prices, load_wx, storm_score, used_rows)

SCRIPT = Path.home() / "nyiso-us" / "storm_value.py"


def script_segment() -> str:
    lines = SCRIPT.read_text().splitlines()
    i = next(k for k, s in enumerate(lines) if s.strip() == "rows = []")
    j = next(k for k, s in enumerate(lines) if s.startswith("day = pd.DataFrame(rows)"))
    return "\n".join(lines[i:j + 1])


SEG = script_segment()


def script_inputs(bid_dates, px, lf, w) -> pd.DataFrame:
    """The script's own loop, unchanged, on the given (possibly injected) tables."""
    ns = {"pd": pd, "np": np, "TZ": TZ, "px": px, "bid_dates": pd.DatetimeIndex(bid_dates),
          "lf": lf[lf["zone"] == "NYISO"], "w": w[w["primary"] & w["temperature_2m_c"].notna()]}
    exec(SEG, ns)
    d = ns["day"].rename(columns={"peak_load_fc": "peak", "rt_stress_now": "rt"})
    return d[["peak", "tmin", "tmax", "rt"]]


def same(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per row: True when all four inputs agree (NaN equals NaN)."""
    A, B = a[["peak", "tmin", "tmax", "rt"]].to_numpy(float), b[["peak", "tmin", "tmax", "rt"]].to_numpy(float)
    assert len(A) == len(B) and (a.index == b.index).all()
    eq = np.isclose(A, B, rtol=0, atol=1e-9) | (np.isnan(A) & np.isnan(B))
    return pd.Series(eq.all(axis=1), index=a.index)


px, lf, wx = load_prices(), load_lf(), load_wx()
assert_holdout((px, "delivery_hour"), (lf, "target_hour"), (wx, "target_hour"))
res = {"holdout": {"prices_max": str(px.delivery_hour.max()), "lf_target_max": str(lf.target_hour.max()),
                   "wx_target_max": str(wx.target_hour.max()), "end": str(END)}}

# ---------------------------------------------------------------- (a) script loop vs independent code
scr = script_inputs(BID_DAYS, px, lf, wx)
mine_s = inputs(BID_DAYS, px, lf, wx, deadline="script")
mine_w = inputs(BID_DAYS, px, lf, wx, deadline="wallclock")
m1 = same(scr, mine_s)
m2 = same(scr, mine_w)
res["reproduction_inputs"] = {
    "days": len(BID_DAYS),
    "script_vs_mine_same_deadline_rule_days_differ": int((~m1).sum()),
    "script_vs_mine_wallclock_days_differ": int((~m2).sum()),
    "differing_days_wallclock": [str(d.date()) for d in m2.index[~m2]],
}
diff_rows = []
for d in m2.index[~m2]:
    diff_rows.append({"D": str(d.date()), **{f"script_{c}": (None if pd.isna(scr.at[d, c]) else round(float(scr.at[d, c]), 3)) for c in scr},
                      **{f"wall_{c}": (None if pd.isna(mine_w.at[d, c]) else round(float(mine_w.at[d, c]), 3)) for c in mine_w}})
res["dst_input_differences"] = diff_rows

# storm score and the decision on those days (script's chosen cuts: filter 0.8/0.9/0.8, flip same)
s_script, _ = storm_score(scr)
s_wall, _ = storm_score(mine_w)
cuts = {2021: 0.8, 2022: 0.9, 2023: 0.8}
dst_effect = []
for d in m2.index[~m2]:
    dd = d + pd.Timedelta(days=1)
    c = cuts.get(dd.year)
    dst_effect.append({"D": str(d.date()), "delivery": str(dd.date()),
                       "storm_script": None if pd.isna(s_script.get(dd)) else round(float(s_script[dd]), 4),
                       "storm_wallclock": None if pd.isna(s_wall.get(dd)) else round(float(s_wall[dd]), 4),
                       "cut_used": c,
                       "decision_flips": (None if c is None else bool((s_script.get(dd, np.nan) > c) != (s_wall.get(dd, np.nan) > c)))})
res["dst_storm_effect"] = dst_effect
# later days' ranks also see the changed value: how many delivery days change side over 2021-2023
sv, sw = s_script.reindex(pd.date_range("2021-01-01", "2023-12-31")), s_wall.reindex(pd.date_range("2021-01-01", "2023-12-31"))
side_change = sum(int((sv[sv.index.year == y] > c).ne(sw[sw.index.year == y] > c).sum()) for y, c in cuts.items())
res["dst_days_changing_side_2021_2023_incl_knock_on"] = side_change

# ---------------------------------------------------------------- (b) rule conformance of every used row
def conformance(mode):
    l, w, r = used_rows(BID_DAYS, px, lf, wx, deadline=mode)
    t_true = deadlines(BID_DAYS, "wallclock")
    out = {}
    # load forecast
    lD = l["D"]
    out["lf_rows"] = len(l)
    out["lf_published_after_true_0500"] = int((l["published_at"] > lD.map(t_true)).sum())
    out["lf_published_ne_file_written"] = int((l["published_at"] != l["file_written_at"]).sum())
    out["lf_file_named_after_D"] = int((l["issue_date"] > lD).sum())
    age = (lD - l["issue_date"]).dt.days.groupby(lD).first()
    out["lf_vintage_age_days_count"] = {str(k): int(v) for k, v in age.value_counts().sort_index().items()}
    wr = (l["issue_date"] - l["published_at"].dt.tz_localize(None).dt.normalize()).dt.days
    out["lf_issue_minus_written_date_days"] = {str(k): int(v) for k, v in wr.value_counts().sort_index().items()}
    hrs = l.groupby("D")["target_hour"].nunique()
    exp = pd.Series({D: len(pd.date_range(pd.Timestamp(D + pd.Timedelta(days=1), tz=TZ),
                                         pd.Timestamp(D + pd.Timedelta(days=2), tz=TZ), freq="h", inclusive="left"))
                     for D in hrs.index})
    out["lf_days_missing_target_hours"] = int((hrs != exp).sum())
    out["lf_bid_days_without_forecast"] = int(len(BID_DAYS) - hrs.size)
    out["lf_lead_days_count"] = {str(k): int(v) for k, v in l["lead_days"].value_counts().sort_index().items()}
    # GFS
    expw = w["target_hour"] - pd.to_timedelta(w["run_lead_hours"], unit="h") + pd.Timedelta(hours=8)
    out["gfs_rows"] = len(w)
    out["gfs_published_ne_rule"] = int((w["published_at"] != expw).sum())
    out["gfs_published_after_true_0500"] = int((w["published_at"] > w["D"].map(t_true)).sum())
    out["gfs_target_not_D_plus_1"] = int(((w["target_hour"].dt.tz_localize(None).dt.normalize() - w["D"]).dt.days != 1).sum())
    out["gfs_rows_by_lead"] = {str(k): int(v) for k, v in w["run_lead_hours"].value_counts().items()}
    out["gfs_day2_rows_after_21h_used"] = int(((w["run_lead_hours"] == 48) & (w["target_hour"].dt.hour > 21)).sum())
    out["gfs_first_bid_day"] = str(w["D"].min().date())
    out["gfs_bid_days_with_data"] = int(w["D"].nunique())
    # RT of day D
    out["rt_rows"] = len(r)
    out["rt_published_after_true_0500"] = int((r["rt_published_at"] > r["D"].map(t_true)).sum())
    out["rt_before_hour_end_plus_15"] = int((r["rt_published_at"] < r["delivery_hour"] + pd.Timedelta(minutes=75)).sum())
    out["rt_revised_rows_used"] = int(r["rt_revised"].sum())
    out["rt_not_day_D"] = int((r["ddate"] != r["D"]).sum())
    hpd = r.groupby("D")["delivery_hour"].nunique()
    out["rt_hours_per_day_count"] = {str(k): int(v) for k, v in hpd.value_counts().sort_index().items()}
    out["rt_bid_days_without_rt"] = int(len(BID_DAYS) - hpd.size)
    # why: day D's rows exist but all were published after the deadline because the file was rewritten
    dD = px[px["ddate"].isin(BID_DAYS)]
    rev_days = dD[dD["rt_revised"]].groupby("ddate").size().index
    out["rt_bid_days_whose_file_was_rewritten"] = int(len(rev_days))
    out["rt_used_value_from_file_written_after_deadline_rows"] = int((r["rt_file_written_at"] > r["D"].map(t_true)).sum())
    return out


res["conformance_wallclock"] = conformance("wallclock")
res["conformance_script_deadline"] = conformance("script")

# ---------------------------------------------------------------- (d) injection tests
rev = px[px["rt_revised"] & px["ddate"].between(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-30"))]
REV_DAY = rev["ddate"].min()
TEST_DAYS = [pd.Timestamp(x) for x in ["2021-07-27", "2022-03-12", "2022-03-13", "2022-11-05", "2022-11-06", "2022-12-23"]] + [REV_DAY]
res["corrected_test_day"] = {"D": str(REV_DAY.date()),
                             "rt_published_at_min": str(px.loc[px.ddate == REV_DAY, "rt_published_at"].min()),
                             "rt_file_written_at": str(px.loc[px.ddate == REV_DAY, "rt_file_written_at"].max())}


def local(D, h, m=0):
    return (D + pd.Timedelta(hours=h, minutes=m)).tz_localize(TZ)


def d1_hours(D):
    a = pd.Timestamp(D + pd.Timedelta(days=1), tz=TZ)
    b = pd.Timestamp(D + pd.Timedelta(days=2), tz=TZ)
    return pd.date_range(a, b, freq="h", inclusive="left")


def inj_lf(D, pub):
    hrs = d1_hours(D)
    add = pd.DataFrame({"issue_date": pd.Timestamp(D + pd.Timedelta(days=1)), "file_written_at": pub, "published_at": pub,
                        "target_hour": hrs, "lead_days": 0, "zone": "NYISO", "load_forecast_mw": 99_999})
    return px, pd.concat([lf, add], ignore_index=True), wx


def inj_wx(D, sel, temp, pub=None):
    w2 = wx.copy()
    m = sel(w2)
    w2.loc[m, "temperature_2m_c"] = temp
    if pub is not None:
        w2.loc[m, "published_at"] = pub
    assert m.sum() > 0
    return px, lf, w2


def inj_px(mask_fn, rt=None, da=None, pub=None):
    p2 = px.copy()
    m = mask_fn(p2)
    assert m.sum() > 0
    if rt is not None:
        p2.loc[m, "rt_lbmp"] = rt
    if da is not None:
        p2.loc[m, "da_lbmp"] = da
    if pub is not None:
        p2.loc[m, "rt_published_at"] = pub(p2.loc[m])
    p2["gap"] = p2["rt_lbmp"] - p2["da_lbmp"]
    return p2, lf, wx


def cases(D):
    d1, d2 = D + pd.Timedelta(days=1), D + pd.Timedelta(days=2)
    tgt = lambda w, day: w["target_hour"].dt.tz_localize(None).dt.normalize() == day
    c = [
        ("isolf file D+1 at its real write time (07:05 on D)", True, inj_lf(D, local(D, 7, 5))),
        ("isolf file D+1 mislabelled public 04:59 on D", False, inj_lf(D, local(D, 4, 59))),
        ("GFS day2 22:00-23:00 of D+1 set to -60 C, true published_at", True,
         inj_wx(D, lambda w: w["primary"] & tgt(w, d1) & (w["target_hour"].dt.hour >= 22) & (w["run_lead_hours"] == 48), -60.0)),
        ("same rows mislabelled public 04:00 on D", False,
         inj_wx(D, lambda w: w["primary"] & tgt(w, d1) & (w["target_hour"].dt.hour >= 22) & (w["run_lead_hours"] == 48), -60.0, local(D, 4))),
        ("GFS for D+2 at -60 C, mislabelled public 00:00 on D (date mapping)", True,
         inj_wx(D, lambda w: w["primary"] & tgt(w, d2), -60.0, local(D, 0))),
        ("RT hour 05:00 of D set to 9999 (public 06:15)", True,
         inj_px(lambda p: (p["ddate"] == D) & (p["delivery_hour"] == local(D, 5)), rt=9999.0)),
        ("RT hour 04:00 of D set to 9999 (public 05:15)", True,
         inj_px(lambda p: (p["ddate"] == D) & (p["delivery_hour"] == local(D, 4)), rt=9999.0)),
        ("RT hour 05:00 of D mislabelled public 04:00", False,
         inj_px(lambda p: (p["ddate"] == D) & (p["delivery_hour"] == local(D, 5)), rt=9999.0, pub=lambda s: local(D, 4))),
        ("target day D+1 own DA and RT set to 9999", True,
         inj_px(lambda p: p["ddate"] == d1, rt=9999.0, da=9999.0)),
    ]
    if D == REV_DAY:
        c += [("corrected day D: RT hour 00:00 set to 9999 (rewrite after midnight)", True,
               inj_px(lambda p: (p["ddate"] == D) & (p["delivery_hour"] == local(D, 0)), rt=9999.0)),
              ("same, relabelled end of hour + 15 min (ignoring the rewrite)", False,
               inj_px(lambda p: (p["ddate"] == D), rt=9999.0,
                      pub=lambda s: s["delivery_hour"] + pd.Timedelta(minutes=75)))]
    return c


base_script = scr
base_wall = mine_w
inj = []
for D in TEST_DAYS:
    for name, legal, (p2, l2, w2) in cases(D):
        a = script_inputs([D], p2, l2, w2)
        b = inputs([D], p2, l2, w2, deadline="wallclock")
        ch_s = not bool(same(a, base_script.loc[[D]]).iloc[0])
        ch_m = not bool(same(b, base_wall.loc[[D]]).iloc[0])
        # storm for D+1 with the injected inputs, history unchanged
        fs, fm = base_script.copy(), base_wall.copy()
        fs.loc[D] = a.loc[D]
        fm.loc[D] = b.loc[D]
        dd = D + pd.Timedelta(days=1)
        inj.append({"D": str(D.date()), "case": name, "rule_respected": legal,
                    "script_inputs_changed": ch_s, "audit_inputs_changed": ch_m,
                    "storm_script_base": round(float(s_script.get(dd, np.nan)), 4),
                    "storm_script_injected": round(float(storm_score(fs)[0].get(dd, np.nan)), 4),
                    "storm_audit_base": round(float(s_wall.get(dd, np.nan)), 4),
                    "storm_audit_injected": round(float(storm_score(fm)[0].get(dd, np.nan)), 4)})
res["injection"] = inj
bad_s = [r for r in inj if r["rule_respected"] == r["script_inputs_changed"]]
bad_m = [r for r in inj if r["rule_respected"] == r["audit_inputs_changed"]]
res["injection_summary"] = {"cases": len(inj),
                            "script_unexpected": [(r["D"], r["case"]) for r in bad_s],
                            "audit_unexpected": [(r["D"], r["case"]) for r in bad_m]}

OUT.mkdir(exist_ok=True)
json.dump(res, open(OUT / "storm_audit_timing.json", "w"), indent=1, default=str)
for k in ["holdout", "reproduction_inputs", "dst_input_differences", "dst_storm_effect",
          "dst_days_changing_side_2021_2023_incl_knock_on", "conformance_wallclock", "conformance_script_deadline",
          "corrected_test_day", "injection_summary"]:
    print(k, json.dumps(res[k], default=str))
for r in inj:
    print(r["D"], "|", r["case"], "| legal" if r["rule_respected"] else "| BROKEN", "| script changed:", r["script_inputs_changed"],
          "| audit changed:", r["audit_inputs_changed"], "|", r["storm_script_base"], "to", r["storm_script_injected"])
