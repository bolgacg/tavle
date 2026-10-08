"""Idea 10 audit, part 1: which GFS values the rule uses, when they were public, and injected lookahead.

Build years only (holdout filter at load in idea10_audit_lib and in lab.load_tables, both asserted).
Run on gene: cd ~/nyiso-us/side_audit && ../.venv/bin/python -I idea10_audit_timing.py
Writes idea10_audit_timing.json. Does not edit lab.py; imports it read-only to run its own function.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "side_audit"))
import idea10_audit_lib as A    # noqa: E402

sys.path.insert(1, str(HOME / "side"))
import lab as L                 # noqa: E402  (read-only import; main() is not run)

OUT = HOME / "side_audit" / "idea10_audit_timing.json"
H = pd.Timedelta(hours=1)


def floor6_utc(ts: pd.Series) -> pd.Series:
    u = ts.dt.tz_convert("UTC")
    return u.dt.floor("6h")


def main():
    res = {}
    w = A.load_wx()
    book = A.load_book()
    dl = A.deadline_series()

    # ---- 1. published_at in the table follows the rule on every pre-2024 row
    exp = np.where(w["run_lead_hours"] == 48, 40, np.where(w["run_lead_hours"] == 72, 64, -1))
    assert (exp > 0).all()
    viol = (w["published_at"] != w["target_hour"] - pd.to_timedelta(exp, unit="h")).sum()
    res["published_at_rule_violations"] = int(viol)

    # ---- 2. the rows idea 10 uses
    j = A.pairs(w)
    j["deadline"] = j["ddate"].map(dl)
    res["rows_used"] = int(len(j))
    res["rows_used_by_lead"] = {"48": int(j["pub48"].notna().sum()), "72": int(j["pub72"].notna().sum())}
    res["max_pub48_minus_deadline_h"] = float((j["pub48"] - j["deadline"]).max() / H)
    res["max_pub72_minus_deadline_h"] = float((j["pub72"] - j["deadline"]).max() / H)
    res["local_hours_used"] = sorted(int(h) for h in j["lhour"].unique())
    # Open-Meteo construction (Open-Meteo blog 'Weather forecasts from previous model runs'): previous_day2 is
    # built from forecast steps 48 onward of successive runs, previous_day3 from steps 72 onward; for a 6-hourly
    # model the run is the latest one initialised at or before valid time minus 48 (72) hours.
    j["init48"] = floor6_utc(j["target_hour"] - 48 * H)
    j["init72"] = floor6_utc(j["target_hour"] - 72 * H)
    j["step48"] = (j["target_hour"].dt.tz_convert("UTC") - j["init48"]) / H
    j["step72"] = (j["target_hour"].dt.tz_convert("UTC") - j["init72"]) / H
    # bound: whole GFS run on NOMADS by init + 5 h (steps 48-77 are out about 4 h after init)
    j["run48_public"] = j["init48"] + 5 * H
    j["run72_public"] = j["init72"] + 5 * H
    j["margin48_h"] = (j["deadline"].dt.tz_convert("UTC") - j["run48_public"]) / H
    j["margin72_h"] = (j["deadline"].dt.tz_convert("UTC") - j["run72_public"]) / H
    j["rule_margin48_h"] = (j["deadline"] - j["pub48"]) / H
    by_h = j.groupby("lhour").agg(n=("s", "size"), step48_min=("step48", "min"), step48_max=("step48", "max"),
                                  rule_margin48_min_h=("rule_margin48_h", "min"),
                                  run_margin48_min_h=("margin48_h", "min"), run_margin72_min_h=("margin72_h", "min"))
    res["by_local_hour"] = {int(k): {kk: (float(vv) if not isinstance(vv, (int, np.integer)) else int(vv))
                                     for kk, vv in r.items()} for k, r in by_h.iterrows()}
    res["min_margin_run48_h"] = float(j["margin48_h"].min())
    res["min_margin_run72_h"] = float(j["margin72_h"].min())
    res["day2_newest_run_used"] = "00Z on D for the evening hours of D+1; 00Z of D-1 for its first hours"

    def runs_for(day):
        x = j[j["ddate"] == pd.Timestamp(day)]
        x = x[x["point"] == "albany"].sort_values("target_hour")
        rows = []
        for _, r in x[x["lhour"].isin([0, 6, 12, 18, 21])].iterrows():
            rows.append({"valid_local": str(r["target_hour"]), "day2_run_utc": str(r["init48"]),
                         "day2_step": int(r["step48"]), "day3_run_utc": str(r["init72"]), "day3_step": int(r["step72"]),
                         "pub48_rule": str(r["pub48"]), "deadline": str(r["deadline"]),
                         "t48": round(float(r["t48"]), 2), "t72": round(float(r["t72"]), 2)})
        return {"distinct_day2_runs": int(j[j["ddate"] == pd.Timestamp(day)]["init48"].nunique()), "albany": rows}

    res["runs_example"] = {d: runs_for(d) for d in ["2022-12-23", "2022-12-24", "2022-07-20"]}

    # ---- 3. DST days and the 21:00 cutoff
    dst = {}
    for d in ["2021-03-14", "2021-03-15", "2021-11-07", "2021-11-08", "2022-03-13", "2022-03-14",
              "2022-11-06", "2022-11-07", "2023-03-12", "2023-03-13", "2023-11-05", "2023-11-06"]:
        x = j[j["ddate"] == pd.Timestamp(d)]
        raw = w[(w["ddate"] == pd.Timestamp(d)) & w["primary"] & (w["run_lead_hours"] == 48)]
        late = raw[raw["published_at"] > dl[pd.Timestamp(d)]]
        dst[d] = {"deadline": str(dl[pd.Timestamp(d)]), "pairs": int(len(x)), "hours_used": int(x["target_hour"].nunique()),
                  "last_hour_used": str(x["target_hour"].max()), "max_pub48": str(x["pub48"].max()),
                  "day2_rows_after_deadline_hours": sorted({str(t.strftime("%H:%M %Z")) for t in late["target_hour"]}),
                  "min_run_margin48_h": float(x["margin48_h"].min())}
    res["dst_days"] = dst

    # ---- 4. lab's own surprise against mine (same rows?)
    px, lf, wxl = L.load_tables()
    F = L.Frame(px)
    lrank, lday = L.weather_surprise(F, wxl)
    lday.index = lday.index + pd.Timedelta(days=1)          # lab indexes by bid day; mine by delivery day
    mine = A.score(w)
    both = lday[["surprise", "pairs"]].join(mine[["surprise", "pairs"]], rsuffix="_mine")
    res["lab_vs_mine"] = {
        "days_with_surprise_lab": int(lday["surprise"].notna().sum()), "days_with_surprise_mine": int(mine["surprise"].notna().sum()),
        "max_abs_surprise_diff": float((both["surprise"] - both["surprise_mine"]).abs().max()),
        "pairs_equal_days": int((both["pairs"].fillna(-1) == both["pairs_mine"].fillna(-1)).sum()),
        "max_abs_rank_diff": float(np.nanmax(np.abs(lrank - mine["rank"].to_numpy()))),
        "rank_nan_pattern_equal": bool((np.isnan(lrank) == np.isnan(mine["rank"].to_numpy())).all())}

    # ---- 5. injections on chosen days, through the lab's function and mine
    cuts = {2021: 0.95, 2022: 0.9, 2023: 0.8}
    days = ["2021-07-07", "2021-11-07", "2021-11-08", "2022-01-16", "2022-03-13", "2022-03-14", "2022-11-06",
            "2022-11-07", "2022-12-23", "2022-12-24", "2022-12-25", "2023-01-11", "2023-09-05"]
    inj = []
    base_rank = mine["rank"]
    for d in days:
        D1 = pd.Timestamp(d)
        i = F.days.get_loc(D1)
        T = dl[D1]
        cut = cuts[D1.year]
        dec0 = bool(base_rank[D1] > cut)
        extreme = 18.3 if dec0 else 60.0          # push the day the other way
        lab_dec0 = bool(lrank[i] > cut)
        assert lab_dec0 == dec0

        def mine_rank(wmod):
            return A.score(wmod)["rank"][D1]

        def lab_rank(wmod_lab):
            r, _ = L.weather_surprise(F, wmod_lab)
            return r[i]

        def lab_frame(wmod):
            x = wmod[wmod["primary"] & wmod["temperature_2m_c"].notna()].reset_index(drop=True)
            return x[[c for c in wxl.columns if c in x.columns]]

        rec = {"delivery_day": d, "deadline": str(T), "cut": cut, "rank": round(float(base_rank[D1]), 4)
               if np.isfinite(base_rank[D1]) else None, "decision": "load" if dec0 else "supply", "tests": {}}
        # honest: every row published after the deadline corrupted (+40 C)
        w1 = w.copy(); m = w1["published_at"] > T; w1.loc[m, "temperature_2m_c"] += 40
        # honest: a fake newer run (lead 24) for D+1, published at its real time (target - 16 h, after 05:00 on D)
        fk = w[(w["ddate"] == D1) & (w["run_lead_hours"] == 48)].copy()
        fk["run_lead_hours"] = 24; fk["temperature_2m_c"] = extreme
        fk["published_at"] = fk["target_hour"] - 16 * H
        assert (fk["published_at"] > T).all()
        w2 = pd.concat([w, fk], ignore_index=True)
        # broken: the same fake run labelled public at 05:00 on D
        fk3 = fk.copy(); fk3["published_at"] = T
        w3 = pd.concat([w, fk3], ignore_index=True)
        # broken: a future value written into the day2 rows themselves (label unchanged)
        w4 = w.copy(); m4 = (w4["ddate"] == D1) & (w4["run_lead_hours"] == 48) & (w4["lhour"] <= 21)
        w4.loc[m4, "temperature_2m_c"] = extreme
        # broken: filter opened to 05:00 on D+1 (every day2 and day3 row of D+1 plus the fake run)
        for name, wm, broken in [("late_rows_corrupted", w1, False), ("newer_run_true_time", w2, False),
                                 ("newer_run_labelled_0500", w3, True), ("future_value_in_day2_rows", w4, True)]:
            rm, rl = mine_rank(wm), lab_rank(lab_frame(wm))
            rec["tests"][name] = {
                "rule_broken": broken,
                "mine_rank": None if not np.isfinite(rm) else round(float(rm), 4),
                "lab_rank": None if not np.isfinite(rl) else round(float(rl), 4),
                "mine_decision_changed": bool((rm > cut) != dec0), "lab_decision_changed": bool((rl > cut) != dec0)}
        # filter opened by 24 hours, with the fake run present
        dl_open = dl.copy(); dl_open[D1] = T + 24 * H
        rm = A.score(w2, deadline=dl_open)["rank"][D1]
        r_open, _ = L.weather_surprise(F, lab_frame(w2), shift_h=24)
        rec["tests"]["filter_opened_24h_newer_run"] = {
            "rule_broken": True, "mine_rank": None if not np.isfinite(rm) else round(float(rm), 4),
            "lab_rank": None if not np.isfinite(r_open[i]) else round(float(r_open[i]), 4),
            "mine_decision_changed": bool((rm > cut) != dec0), "lab_decision_changed": bool((r_open[i] > cut) != dec0)}
        inj.append(rec)
        print(d, {k: (v["mine_decision_changed"], v["lab_decision_changed"]) for k, v in rec["tests"].items()}, flush=True)
    res["injections"] = inj
    summ = {}
    for k in inj[0]["tests"]:
        summ[k] = {"rule_broken": inj[0]["tests"][k]["rule_broken"], "days": len(inj),
                   "mine_changed": sum(r["tests"][k]["mine_decision_changed"] for r in inj),
                   "lab_changed": sum(r["tests"][k]["lab_decision_changed"] for r in inj)}
    res["injection_summary"] = summ
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("injections", "dst_days", "by_local_hour", "runs_example")},
                     indent=1, default=str))
    print(json.dumps(res["by_local_hour"], indent=0)[:3000])
    print(json.dumps(res["dst_days"], indent=0))
    print(json.dumps(res["runs_example"], indent=0)[:4000])


if __name__ == "__main__":
    main()
