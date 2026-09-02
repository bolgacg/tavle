"""Six versions of one hour: the study behind docs/versions/.

Writes research/results/versions.json. Every number on the versions page
that is not live (the recent fortnight, the revision log) comes from here,
and the three tolerance lines in dbt/dbt_project.yml are rewritten by this
script so the test the pipeline runs and the rule the page quotes are the
same numbers.

Fit and check: the agreement rule between the real-time hour and the
settled hour is fitted on 2025 (the 99th percentile of the disagreement)
and checked on 2026, which was not looked at while choosing the rule. The
cost act reuses the imbalance economics of the wind study (research/
study2.py): a producer owning a fixed share of the zone's wind sells one
version and delivers the settled amount; the difference is settled at the
imbalance price. Single-pricing era only, to the end of hourly imbalance
settlement on 4 March 2025."""
import datetime as dt
import json
import pathlib
import re

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
OUT = ROOT / "research" / "results" / "versions.json"
PROJECT = ROOT / "dbt" / "dbt_project.yml"
VERSIONS = [("day-ahead", "v_day_ahead"), ("intraday", "v_intraday"), ("five-hour", "v_5h"),
            ("one-hour", "v_1h"), ("real-time", "v_realtime")]
FIT_YEAR, CHECK_YEAR = 2025, 2026
SINGLE = pd.Timestamp("2021-11-01"); IMB_END = pd.Timestamp("2025-03-05"); SHARE = 0.05
MIN_SETTLED = 100.0     # below this, percent disagreement is dominated by the denominator
PCTL = 99               # the rule: the 99th percentile of 2025 disagreement
MAX_BREACH_PCT = 5.0    # the alarm: five times the fitted breach rate

con = duckdb.connect(str(DB), read_only=True)
W = con.execute("""
    select area, hour_utc, v_day_ahead, v_intraday, v_5h, v_1h, v_realtime, realtime_readings, v_settled, versions_present
    from wind_versions order by 1, 2""").df()
W["year"] = W.hour_utc.dt.year
full = W[(W.versions_present == 6) & (W.realtime_readings == 12)].copy()


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


# ---------- act two: does it converge ----------
def convergence(df):
    rows = []
    for (area, year), g in df.groupby(["area", "year"]):
        big = g[g.v_settled >= MIN_SETTLED]
        for name, col in VERSIONS:
            d = g[col] - g.v_settled
            pct = (big[col] - big.v_settled).abs() / big.v_settled * 100
            rows.append({"area": area, "year": int(year), "version": name, "hours": int(len(g)),
                         "mae_mwh": float(d.abs().mean()), "median_abs_mwh": float(d.abs().median()),
                         "p90_abs_mwh": q(d.abs(), 90), "bias_mwh": float(d.mean()),
                         "within_5pct": float((pct <= 5).mean()) if len(big) else None,
                         "within_10pct": float((pct <= 10).mean()) if len(big) else None,
                         "median_abs_pct": float(pct.median()) if len(big) else None,
                         "mean_settled_mwh": float(g.v_settled.mean())})
    return rows


def by_wind_level(df):
    """The five disagreements by how windy the settled hour was: terciles within each zone, 2025 to 2026."""
    rows = []
    for area, g in df[df.year >= FIT_YEAR].groupby("area"):
        cuts = g.v_settled.quantile([1 / 3, 2 / 3]).values
        lab = np.where(g.v_settled <= cuts[0], "calm", np.where(g.v_settled <= cuts[1], "medium", "windy"))
        for level in ("calm", "medium", "windy"):
            h = g[lab == level]
            row = {"area": area, "level": level, "hours": int(len(h)), "settled_from": float(h.v_settled.min()),
                   "settled_to": float(h.v_settled.max()), "mean_settled_mwh": float(h.v_settled.mean())}
            for name, col in VERSIONS:
                row[name] = float((h[col] - h.v_settled).abs().median())
            rows.append(row)
    return rows


# ---------- act three: the rule ----------
def breach(g, tol_pct, tol_mwh):
    tol = np.maximum(tol_mwh, tol_pct / 100 * g.v_settled)
    return ((g.v_realtime - g.v_settled).abs() > tol)


def fit_rule(df):
    f = df[df.year == FIT_YEAR]
    big, small = f[f.v_settled >= MIN_SETTLED], f[f.v_settled < MIN_SETTLED]
    tol_pct = q((big.v_realtime - big.v_settled).abs() / big.v_settled * 100, PCTL)
    tol_mwh = q((small.v_realtime - small.v_settled).abs(), PCTL) if len(small) else 0.0
    tol_pct, tol_mwh = round(tol_pct, 1), round(tol_mwh, 1)
    out = {"fit_year": FIT_YEAR, "check_year": CHECK_YEAR, "percentile": PCTL, "min_settled_mwh": MIN_SETTLED,
           "tolerance_pct": tol_pct, "tolerance_mwh": tol_mwh, "max_breach_pct": MAX_BREACH_PCT,
           "fit_hours": int(len(f)), "by_year": [], "curve": [], "monthly": [], "baseline_pct": 5.0}
    for (area, year), g in df.groupby(["area", "year"]):
        b = breach(g, tol_pct, tol_mwh)
        b5 = breach(g, out["baseline_pct"], tol_mwh)   # the rule a desk writes down: five percent, same floor
        out["by_year"].append({"area": area, "year": int(year), "hours": int(len(g)), "breach_pct": float(100 * b.mean()),
                               "baseline_breach_pct": float(100 * b5.mean()), "period": "fit" if year == FIT_YEAR else ("check" if year == CHECK_YEAR else "earlier")})
    for area, g in df[df.year >= FIT_YEAR].groupby("area"):
        for t in range(1, 31):
            row = {"area": area, "tolerance_pct": t}
            for y in (FIT_YEAR, CHECK_YEAR):
                h = g[g.year == y]
                row[str(y)] = float(100 * breach(h, float(t), tol_mwh).mean()) if len(h) else None
            out["curve"].append(row)
    for (area, m), g in df[df.year >= FIT_YEAR].groupby(["area", df.hour_utc.dt.strftime("%Y-%m")]):
        b = breach(g, tol_pct, tol_mwh)
        out["monthly"].append({"area": area, "month": m, "hours": int(len(g)), "breach_pct": float(100 * b.mean()),
                               "bias_mwh": float((g.v_realtime - g.v_settled).mean()),
                               "bias_pct": float(100 * ((g.v_realtime - g.v_settled).sum() / g.v_settled.sum()))})
    return out


# ---------- act one: the preset hours ----------
def presets(df):
    def pack(row, why):
        return {"why": why, "area": row.area, "hour_utc": row.hour_utc.isoformat(), "values": {
            "day-ahead": float(row.v_day_ahead), "intraday": float(row.v_intraday), "five-hour": float(row.v_5h),
            "one-hour": float(row.v_1h), "real-time": float(row.v_realtime), "settled": float(row.v_settled)}}
    g = df[(df.year == CHECK_YEAR)]
    d1 = g[g.area == "DK1"]
    out = []
    r = d1.loc[d1.v_settled.idxmax()]; out.append(pack(r, f"the windiest settled hour of {CHECK_YEAR} in DK1"))
    span = d1[["v_day_ahead", "v_intraday", "v_5h", "v_1h", "v_realtime", "v_settled"]]
    rng = (span.max(axis=1) - span.min(axis=1))
    r = d1.loc[rng.idxmax()]; out.append(pack(r, f"the hour whose six versions were furthest apart in {CHECK_YEAR}, DK1"))
    rt = (d1.v_realtime - d1.v_settled).abs()
    r = d1.loc[rt.idxmax()]; out.append(pack(r, f"the hour where the real-time feed and the settlement disagreed most in {CHECK_YEAR}, DK1"))
    big = d1[d1.v_settled >= MIN_SETTLED]
    pct = ((big.v_realtime - big.v_settled).abs() / big.v_settled)
    med = pct.sort_values().index[len(pct) // 2]
    r = d1.loc[med]; out.append(pack(r, f"a typical hour: the median real-time disagreement of {CHECK_YEAR}, DK1"))
    d2 = g[g.area == "DK2"]
    r = d2.loc[d2.v_settled.idxmax()]; out.append(pack(r, f"the windiest settled hour of {CHECK_YEAR} in DK2"))
    return out


# ---------- act four: what acting on a version costs ----------
def cost(df):
    fv = con.execute("""
        select w.area, w.hour_utc, p.price_eur, i.imbalance_price_eur,
               w.v_day_ahead, w.v_intraday, w.v_5h, w.v_1h, w.v_realtime, w.v_settled
        from wind_versions w
        join power_hourly p using (area, hour_utc)
        join stg_imbalance i using (area, hour_utc)
        where w.versions_present = 6 and w.realtime_readings = 12 and i.imbalance_price_eur is not null
        order by 1, 2""").df()
    fv = fv[(fv.hour_utc >= SINGLE) & (fv.hour_utc < IMB_END)]
    gap = fv.imbalance_price_eur - fv.price_eur
    out = {"share": SHARE, "from": str(fv.hour_utc.min())[:10], "to": str(fv.hour_utc.max())[:10], "rows": [], "by_year": []}
    for area, g in fv.groupby("area"):
        gp = g.imbalance_price_eur - g.price_eur
        A = SHARE * g.v_settled
        meas = (-(SHARE * (g.v_settled - g.v_realtime)) * gp)      # the part that exists even with a perfect forecast of the feed
        for name, col in VERSIONS[:4]:
            N = SHARE * g[col]
            total = (-(A - N) * gp)
            fc = (-(SHARE * (g.v_realtime - g[col])) * gp)
            out["rows"].append({"area": area, "version": name, "hours": int(len(g)),
                                "eur_per_mwh": float(total.sum() / A.sum()), "total_keur": float(total.sum() / 1e3),
                                "forecast_part_eur_per_mwh": float(fc.sum() / A.sum()),
                                "measurement_part_eur_per_mwh": float(meas.sum() / A.sum()),
                                "imbalance_share": float((A - N).abs().sum() / A.sum()),
                                "measurement_share": float((SHARE * (g.v_settled - g.v_realtime)).abs().sum() / A.sum()),
                                "mwh_produced": float(A.sum())})
        for y, h in g.groupby(g.hour_utc.dt.year):
            gy = h.imbalance_price_eur - h.price_eur; Ay = SHARE * h.v_settled
            row = {"area": area, "year": int(y), "hours": int(len(h)),
                   "measurement_part": float((-(SHARE * (h.v_settled - h.v_realtime)) * gy).sum() / Ay.sum())}
            for name, col in VERSIONS[:4]:
                row[name] = float((-(Ay - SHARE * h[col]) * gy).sum() / Ay.sum())
            out["by_year"].append(row)
    return out


def spans(df):
    return {"first": str(df.hour_utc.min())[:10], "last": str(df.hour_utc.max())[:10], "hours_all_six": int(len(df)),
            "hours_any": int(len(W)), "share_all_six": float(len(df) / len(W)),
            "realtime_short_hours": int(((W.v_realtime.notna()) & (W.realtime_readings < 12)).sum()),
            "realtime_hours": int(W.v_realtime.notna().sum()),
            "by_versions_present": {int(k): int(v) for k, v in W.versions_present.value_counts().sort_index().items()}}


def write_vars(rule):
    s = PROJECT.read_text()
    for key, val in (("versions_tolerance_pct", rule["tolerance_pct"]), ("versions_tolerance_mwh", rule["tolerance_mwh"]),
                     ("versions_max_breach_pct", rule["max_breach_pct"])):
        s, n = re.subn(rf"^(\s*{key}:\s*)[-\d.]+", rf"\g<1>{val}", s, flags=re.M)
        assert n == 1, key
    PROJECT.write_text(s)


if __name__ == "__main__":
    res = {"computed_at": dt.date.today().isoformat(), "span": spans(full), "convergence": convergence(full),
           "by_wind_level": by_wind_level(full), "rule": fit_rule(full), "presets": presets(full), "cost": cost(full)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    write_vars(res["rule"])
    r = res["rule"]
    print(f"wrote {OUT}: {res['span']['hours_all_six']} hours with all six versions, {res['span']['first']} to {res['span']['last']}")
    print(f"rule: within {r['tolerance_pct']}% or {r['tolerance_mwh']} MWh; fit {FIT_YEAR}, check {CHECK_YEAR}")
    for row in r["by_year"]:
        if row["year"] >= 2024: print("  ", row)
