"""Inventory of raw files and parquet tables: COUNTS ONLY for every year (holdout rule).

    python inventory.py      # writes ~/nyiso-us/results/inventory.json and file_write_times.csv

The single exception to "counts only" is the congestion sign check, which reads prices and is
therefore restricted to delivery hours before HOLDOUT_START (2020-2023).
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

from common import (FIRST_DAY, HOLDOUT_START, LAST_DAY, PARQUET, RAW, RESULTS, GFS_RAW, TZ, ZONES,
                    decision_time, entries, expected_hours, local_midnight, month_keys)
from points import POINTS

SERIES = ["damlbmp_zone", "damlbmp_gen", "rtlbmp_zone", "rtlbmp_gen", "isolf", "outSched"]


def all_days():
    d = FIRST_DAY
    while d <= LAST_DAY:
        yield d
        d += dt.timedelta(days=1)


def raw_files():
    out, rows = {}, []
    for s in SERIES:
        zips = sum((RAW / f"{ym}{s}_csv.zip").exists() for ym in month_keys())
        have = {}
        for d, w, _, _ in entries(s):
            have[d] = w
            rows.append({"series": s, "file_date": d.isoformat(), "written_at_local": w.isoformat()})
        missing = [d.isoformat() for d in all_days() if d not in have]
        per_year = Counter(d.year for d in have)
        # Write-time checks against the 05:00 deadline (file named D is used for a bid made on D).
        late = Counter()
        for d, w in have.items():
            if s.startswith("damlbmp") and w > dt.datetime.combine(d - dt.timedelta(days=1), dt.time(11)):
                late[d.year] += 1                                   # after the 11:00 D-1 bound
            if s in ("isolf", "outSched") and w > dt.datetime.combine(d, dt.time(5)):
                late[d.year] += 1                                   # not public at 05:00 on D
            if s.startswith("rtlbmp") and w >= dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0)):
                late[d.year] += 1                                   # rewritten after its day (revision)
        clock = Counter(w.strftime("%H:%M") for w in have.values()).most_common(3)
        rel = Counter((w.date() - d).days for d, w in have.items())
        out[s] = {"monthly_zips": zips, "daily_files": len(have), "daily_files_per_year": dict(sorted(per_year.items())),
                  "missing_days": missing,
                  {"damlbmp_zone": "files_written_after_11h_on_D_minus_1",
                   "damlbmp_gen": "files_written_after_11h_on_D_minus_1",
                   "rtlbmp_zone": "files_rewritten_after_their_day",
                   "rtlbmp_gen": "files_rewritten_after_their_day",
                   "isolf": "files_named_D_not_public_by_05h_on_D",
                   "outSched": "files_named_D_not_public_by_05h_on_D"}[s]: dict(sorted(late.items())),
                  "most_common_write_clock_times": clock,
                  "write_day_minus_file_day_counts": {str(k): v for k, v in sorted(rel.items())}}
    pd.DataFrame(rows).to_csv(RESULTS / "file_write_times.csv", index=False)
    out["gfs_json"] = {"files": len(list(GFS_RAW.glob("*.json"))), "expected": len(POINTS) * 6}
    return out


def prices_zone_counts():
    pz = pd.read_parquet(PARQUET / "prices_zone.parquet",
                         columns=["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "rt_revised"])
    year = pz["delivery_hour"].dt.year
    rows = {str(y): int(n) for y, n in year.value_counts().sort_index().items()}
    missing = {}
    for y in range(FIRST_DAY.year, LAST_DAY.year + 1):
        exp = expected_hours(y)
        sub = pz[year == y]
        per = {}
        for z in ZONES:
            g = sub[sub["zone"] == z]
            per[z] = {"da": int(len(exp.difference(pd.DatetimeIndex(g.loc[g["da_lbmp"].notna(), "delivery_hour"])))),
                      "rt": int(len(exp.difference(pd.DatetimeIndex(g.loc[g["rt_lbmp"].notna(), "delivery_hour"]))))}
        missing[str(y)] = {"expected_hours": len(exp), "zones": per}
    jul = pz[(pz["delivery_hour"] >= pd.Timestamp("2026-07-01", tz=TZ)) &
             (pz["delivery_hour"] < pd.Timestamp("2026-08-01", tz=TZ))]
    exp = pd.date_range(pd.Timestamp("2026-07-01", tz=TZ), pd.Timestamp("2026-08-01", tz=TZ), freq="h", inclusive="left")
    have = pd.DatetimeIndex(jul.loc[jul["rt_lbmp"].notna() & (jul["zone"] == "N.Y.C."), "delivery_hour"])
    gap = exp.difference(have)
    gap_days = sorted({str(t.date()) for t in gap})
    revised = {str(y): int(n) for y, n in pz.loc[pz["rt_revised"], "delivery_hour"].dt.year.value_counts().sort_index().items()}
    return rows, missing, {"rt_zone_hours_missing_per_zone": len(gap), "days": gap_days}, revised


def prices_gen_counts():
    rows, pts = Counter(), defaultdict(lambda: {"da": set(), "rt": set(), "any": set(), "external": set()})
    jul_gen = 0
    for f in sorted((PARQUET / "prices_gen").glob("*.parquet")):
        t = pq.read_table(f, columns=["delivery_hour", "ptid", "point_type", "da_lbmp", "rt_lbmp"]).to_pandas()
        y = str(int(f.stem[:4]))
        rows[y] += len(t)
        pts[y]["da"] |= set(t.loc[t["da_lbmp"].notna(), "ptid"])
        pts[y]["rt"] |= set(t.loc[t["rt_lbmp"].notna(), "ptid"])
        pts[y]["any"] |= set(t["ptid"])
        pts[y]["external"] |= set(t.loc[t["point_type"] == "external", "ptid"])
        if f.stem == "202607":
            jul_gen = int(t["rt_lbmp"].notna().sum())
    first_seen = {}
    years = sorted(pts)
    for i, y in enumerate(years):
        prev = set().union(*[pts[p]["any"] for p in years[:i]]) if i else set()
        first_seen[y] = len(pts[y]["any"] - prev)
    gp = {y: {"with_da": len(v["da"]), "with_rt": len(v["rt"]), "any": len(v["any"]),
              "external_proxies": len(v["external"]), "new_this_year": first_seen[y]} for y, v in pts.items()}
    return dict(sorted(rows.items())), gp, jul_gen


def simple_counts(name, col, extra=None):
    df = pd.read_parquet(PARQUET / f"{name}.parquet")
    key = pd.to_datetime(df[col])
    yr = (key.dt.tz_convert(TZ).dt.year if hasattr(key.dtype, "tz") else key.dt.year)
    out = {"rows_per_year": {str(y): int(n) for y, n in yr.value_counts().sort_index().items()}}
    if extra:
        out.update(extra(df, yr))
    return out


def lf_extra(df, yr):
    files = df.groupby(yr)["issue_date"].nunique()
    leads = df["lead_days"].value_counts().sort_index()
    return {"files_per_year": {str(y): int(n) for y, n in files.items()},
            "rows_per_lead_day": {str(k): int(v) for k, v in leads.items()},
            "zones": sorted(df["zone"].unique().tolist())}


def og_extra(df, yr):
    return {"snapshots_per_year": {str(y): int(n) for y, n in df.groupby(yr)["snapshot_date"].nunique().items()},
            "distinct_equipment_per_year": {str(y): int(n) for y, n in df.groupby(yr)["equipment"].nunique().items()},
            "unparsed_out_or_in_times": int(df["sched_out"].isna().sum() + df["sched_in"].isna().sum())}


def wx_extra(df, yr):
    out = {}
    for p, g in df.groupby("point"):
        gy = yr[g.index]
        nn = g["temperature_2m_c"].notna()
        out[p] = {"zone": g["zone"].iloc[0],
                  "rows_per_year": {str(y): int(n) for y, n in gy.value_counts().sort_index().items()},
                  "null_values_per_year": {str(y): int(n) for y, n in (~nn).groupby(gy).sum().items()},
                  "first_non_null_target_hour": str(g.loc[nn, "target_hour"].min()) if nn.any() else None,
                  "grid_cell": [g["lat_grid"].iloc[0], g["lon_grid"].iloc[0]]}
    return {"points": out}


def congestion_sign_check_build_years():
    """Build years only. The energy component (reference-bus price) must be the same in every zone
    each hour. With energy = LBMP - losses + congestion_raw it is; with the naive sign it is not."""
    pz = pd.read_parquet(PARQUET / "prices_zone.parquet",
                         columns=["delivery_hour", "da_lbmp", "da_loss", "da_congestion_raw"])
    pz = pz[pz["delivery_hour"] < HOLDOUT_START]
    assert pz["delivery_hour"].max() < HOLDOUT_START
    fixed = pz["da_lbmp"] - pz["da_loss"] + pz["da_congestion_raw"]
    naive = pz["da_lbmp"] - pz["da_loss"] - pz["da_congestion_raw"]
    g = pz["delivery_hour"]
    spread_fixed = fixed.groupby(g).agg(lambda s: s.max() - s.min())
    spread_naive = naive.groupby(g).agg(lambda s: s.max() - s.min())
    out = {"years": "2020-2023 only", "hours": int(len(spread_fixed)),
           "note": "share of hours in which the energy component agrees across the 11 zones within a tolerance "
                   "(each published component is rounded to 0.01, so 0.02 is tight)"}
    for tol in (0.02, 0.05, 0.10):
        out[f"share_within_{tol:.2f}_raw_sign_reversed"] = round(float((spread_fixed <= tol).mean()), 4)
        out[f"share_within_{tol:.2f}_naive_sign"] = round(float((spread_naive <= tol).mean()), 4)
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    inv = {"generated": pd.Timestamp.now(tz=TZ).isoformat(timespec="seconds"),
           "rule": "counts only for every year; no price statistic on or after 2024-01-01",
           "raw_files": raw_files()}
    rows, missing, jul, revised = prices_zone_counts()
    gen_rows, gen_points, jul_gen = prices_gen_counts()
    jul["rtlbmp_gen_rows_in_july_2026"] = jul_gen
    inv["tables"] = {
        "prices_zone": {"rows_per_year": rows, "rt_rows_with_revised_publication_per_year": revised},
        "prices_gen": {"rows_per_year": gen_rows},
        "load_forecast": simple_counts("load_forecast", "issue_date", lf_extra),
        "outages": simple_counts("outages", "snapshot_date", og_extra),
        "weather_gfs": simple_counts("weather_gfs", "target_hour", wx_extra),
    }
    inv["missing_hours_per_zone_per_year"] = missing
    inv["generator_points_per_year"] = gen_points
    inv["july_2026_rtlbmp_gap"] = jul
    for f in sorted(PARQUET.glob("build_log_*.json")):
        inv.setdefault("build_logs", {})[f.stem] = json.loads(f.read_text())
    inv["congestion_sign_check"] = congestion_sign_check_build_years()
    (RESULTS / "inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    print("inventory written", RESULTS / "inventory.json")


if __name__ == "__main__":
    main()
