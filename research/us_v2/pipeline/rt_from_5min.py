"""Hourly real-time zone prices rebuilt from NYISO's own 5-minute real-time prices, for days the hourly archive lacks.
Written 10 Oct 2026 for the held-out run.

NYISO's monthly archive of hourly integrated real-time zone prices (rtlbmp_zone) for July 2026 holds only 12 daily
files; the other 19 days (1 to 18 and 20 July) are missing and the daily CSVs return 404. The 5-minute real-time
zone archive (realtime_zone) has every day. NYISO's hourly integrated price is the time-weighted average of the
5-minute prices inside the hour; this script rebuilds the missing days that way and checks the rule on every
official day first.

    python rt_from_5min.py check      list every held-out month (2024-01 to 2026-09) whose hourly real-time zip
                                      (zone or gen) lacks days; reads file names inside the zips only
    python rt_from_5min.py fetch      download the 5-minute monthly zip of each such month (about 1 MB each)
    python rt_from_5min.py selftest   rebuild every official day of those months and compare: every zone-hour
                                      present on both sides; LBMP, losses and congestion as unrounded averages
                                      within 0.005 USD/MWh of the official value, and within 0.01 once rounded
    python rt_from_5min.py rebuild    selftest, then write the missing days to raw/rt_rebuilt/<YYYYMM01>rtlbmp_zone_csv.zip
    python rt_from_5min.py all        check, fetch, rebuild

Rule (per zone, within one daily 5-minute file): rows sorted by time stamp; each row covers (previous stamp, its
stamp], the first row of a zone 5 minutes; the hour is (stamp - 1 s) floored to the hour; an interval is clipped
at the hour start; each component is weighted by seconds and rounded to 2 decimals. Every rebuilt hour must have
3,600 s of coverage. NYISO averages its unrounded 5-minute prices; the published ones carry 2 decimals, so an
average that lands exactly on a half cent can round to the other cent: the rebuilt price then differs by 0.01.

Output format: exactly NYISO's rtlbmp_zone daily CSV (quoted header and text, CRLF, "MM/DD/YYYY HH:MM" hour
beginning, 2 decimals), one entry per day, and each zip entry carries the write time of the 5-minute file it was
rebuilt from, so publication times follow the same rule as official rows (pipeline/build_tables.py pair()). The
table builders read these entries only for days the official zip lacks (research/us/pipeline/build_tables.py and
research/us_v2/pipeline/build_tables_v2.py read_price_month) and list them in build_log_prices.json.
Only the 5-minute ZONE file is published in this form; the gen files are complete for the whole window.
"""
from __future__ import annotations

import calendar
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path.home() / "nyiso-us" / "raw"
FIVE = RAW / "rt_5min"
OUT = RAW / "rt_rebuilt"
URL = "http://mis.nyiso.com/public/csv/realtime/{ym}realtime_zone_csv.zip"
WINDOW = ("202401", "202609")
HEADER = '"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)","Marginal Cost Congestion ($/MWHr)"'
COLS = ["ts", "name", "ptid", "lbmp", "loss", "cong"]
TOL = 0.005


def months():
    y, m = int(WINDOW[0][:4]), int(WINDOW[0][4:])
    while f"{y}{m:02d}" <= WINDOW[1]:
        yield f"{y}{m:02d}01"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def days_in(zp: Path) -> set[str]:
    return {i.filename[:8] for i in zipfile.ZipFile(zp).infolist()} if zp.exists() else set()


def check() -> dict:
    """{ym: [missing YYYYMMDD]} for the zone file; a gap in a gen file stops (no 5-minute gen source here)."""
    out = {}
    for ym in months():
        want = {f"{ym[:6]}{d:02d}" for d in range(1, calendar.monthrange(int(ym[:4]), int(ym[4:6]))[1] + 1)}
        for s in ("rtlbmp_zone", "rtlbmp_gen"):
            miss = sorted(want - days_in(RAW / f"{ym}{s}_csv.zip"))
            if miss and s == "rtlbmp_gen":
                raise SystemExit(f"{ym} rtlbmp_gen lacks {miss}: no rebuild for gen files")
            if miss:
                out[ym] = miss
    print(json.dumps({"months_with_missing_days": out}))
    return out


def fetch(gaps: dict):
    FIVE.mkdir(parents=True, exist_ok=True)
    for ym in gaps:
        p = FIVE / f"{ym}realtime_zone_csv.zip"
        if p.exists():
            continue
        with urllib.request.urlopen(URL.format(ym=ym), timeout=120) as r:
            p.write_bytes(r.read())
        print("fetched", p.name, p.stat().st_size, "bytes")


def read_entry(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> pd.DataFrame:
    df = pd.read_csv(z.open(info), header=0, names=COLS, usecols=range(6), dtype={"name": str, "ptid": "int64"})
    df["t"] = pd.to_datetime(df["ts"], format="%m/%d/%Y %H:%M:%S" if len(df["ts"].iloc[0]) > 16 else "%m/%d/%Y %H:%M")
    return df


def rebuild_day(five: pd.DataFrame, rounded: bool = True) -> pd.DataFrame:
    """Hourly rows (hour beginning, wall clock) from one day's 5-minute rows, in NYISO's row order."""
    order = list(dict.fromkeys(five["name"]))
    parts = []
    for name, g in five.groupby("name", sort=False):
        g = g.sort_values("t", kind="stable")
        t = g["t"].to_numpy()
        start = np.concatenate([[t[0] - np.timedelta64(5, "m")], t[:-1]])
        hour = (g["t"] - pd.Timedelta(seconds=1)).dt.floor("h").to_numpy()
        start = np.maximum(start, hour)
        w = (t - start) / np.timedelta64(1, "s")
        x = pd.DataFrame({"hour": hour, "w": w, "lbmp": g["lbmp"].to_numpy() * w, "loss": g["loss"].to_numpy() * w,
                          "cong": g["cong"].to_numpy() * w})
        s = x.groupby("hour").sum()
        if not np.allclose(s["w"], 3600.0):
            raise SystemExit(f"{name}: an hour without full 5-minute coverage: {s.index[~np.isclose(s['w'], 3600.0)][:3].tolist()}")
        h = pd.DataFrame({"t": s.index, "name": name, "ptid": int(g["ptid"].iloc[0])})
        for c in ("lbmp", "loss", "cong"):
            v = (s[c] / s["w"]).to_numpy()
            h[c] = np.round(v, 2) if rounded else v
        parts.append(h)
    out = pd.concat(parts, ignore_index=True)
    out["k"] = out["name"].map({n: i for i, n in enumerate(order)})
    out = out.sort_values(["t", "k"], kind="stable").drop(columns="k").reset_index(drop=True)
    if out["t"].dt.normalize().nunique() != 1 or len(out) != 24 * len(order):
        raise SystemExit(f"not one whole day of 24 hours x {len(order)} names (daylight-saving day?): {len(out)} rows")
    return out


def to_csv(h: pd.DataFrame) -> bytes:
    lines = [HEADER] + [f'"{t:%m/%d/%Y %H:%M}","{n}",{p},{a:.2f},{b:.2f},{c:.2f}'
                        for t, n, p, a, b, c in zip(h["t"], h["name"], h["ptid"], h["lbmp"], h["loss"], h["cong"])]
    return ("\r\n".join(lines) + "\r\n").encode().replace(b"-0.00", b"0.00")


def selftest(gaps: dict) -> dict:
    rep = {}
    for ym in gaps:
        oz = zipfile.ZipFile(RAW / f"{ym}rtlbmp_zone_csv.zip")
        fz = zipfile.ZipFile(FIVE / f"{ym}realtime_zone_csv.zip")
        five = {i.filename[:8]: i for i in fz.infolist()}
        n = worst = worst_r = equal = 0
        for info in oz.infolist():
            off = read_entry(oz, info)
            f = read_entry(fz, five[info.filename[:8]])
            for rounded in (False, True):
                reb = rebuild_day(f, rounded)
                j = off.merge(reb, on=["t", "name", "ptid"], how="outer", suffixes=("_o", "_r"), indicator=True)
                if not (j["_merge"] == "both").all():
                    raise SystemExit(f"{info.filename}: zone-hours differ between official and rebuilt")
                d = np.stack([(j[f"{c}_o"] - j[f"{c}_r"]).abs().to_numpy() for c in ("lbmp", "loss", "cong")], 1)
                if rounded:
                    worst_r = max(worst_r, float(d.max()))
                    equal += int((d.max(1) < 1e-9).sum())
                else:
                    worst = max(worst, float(d.max()))
            n += len(j)
        rep[ym] = {"official_days": len(oz.infolist()), "zone_hours_compared": n,
                   "worst_abs_diff_unrounded_average": round(worst, 6), "worst_abs_diff_after_rounding": round(worst_r, 6),
                   "zone_hours_equal_after_rounding": equal,
                   "pass": worst <= TOL + 1e-6 and worst_r <= 0.01 + 1e-6}
        if not rep[ym]["pass"]:
            raise SystemExit(f"selftest failed: {rep}")
    print(json.dumps({"selftest": rep}))
    return rep


def rebuild(gaps: dict):
    rep = selftest(gaps)
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    for ym, miss in gaps.items():
        fz = zipfile.ZipFile(FIVE / f"{ym}realtime_zone_csv.zip")
        five = {i.filename[:8]: i for i in fz.infolist()}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for d in miss:
                src = five[d]
                zi = zipfile.ZipInfo(f"{d}rtlbmp_zone.csv", date_time=src.date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, to_csv(rebuild_day(read_entry(fz, src))))
        (OUT / f"{ym}rtlbmp_zone_csv.zip").write_bytes(buf.getvalue())
        done[ym] = {"days": miss, "source": URL.format(ym=ym),
                    "source_entry_written": {d: "%04d-%02d-%02d %02d:%02d:%02d" % five[d].date_time for d in miss}}
    (OUT / "README.md").write_text(
        "# Hourly real-time zone prices rebuilt from NYISO's 5-minute prices\n\n"
        "NYISO's hourly integrated real-time archive (rtlbmp_zone) lacks these days; the daily files return 404. "
        "Each day here is the time-weighted average of NYISO's own 5-minute real-time zone prices (realtime_zone), "
        "in the rtlbmp_zone CSV format, written by research/us_v2/pipeline/rt_from_5min.py. Each zip entry keeps the "
        "write time of the 5-minute file it came from. The table builders use a day from here only when the official "
        "zip lacks it.\n\n```json\n" + json.dumps({"rebuilt": done, "selftest_on_official_days": rep,
                                                   "tolerance_usd_per_mwh": TOL}, indent=1) + "\n```\n")
    print(json.dumps({"rebuilt": {k: v["days"] for k, v in done.items()}}))


def rebuilt_entries(ym: str, series: str, have: set) -> list:
    """(zipfile, info) for rebuilt days of this month that the official zip lacks; used by the table builders."""
    p = OUT / f"{ym}{series}_csv.zip"
    if not p.exists():
        return []
    z = zipfile.ZipFile(p)
    return [(z, i) for i in sorted(z.infolist(), key=lambda i: i.filename) if i.filename[:8] not in have]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    g = check()
    if cmd in ("fetch", "all"):
        fetch(g)
    if cmd == "selftest":
        selftest(g)
    if cmd in ("rebuild", "all"):
        rebuild(g)
