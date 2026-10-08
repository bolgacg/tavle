"""GEFS reforecast v12 (AWS noaa-gefs-retrospective) 2 m temperature at the 12 study points.

Control member c00 only, 00 UTC run of every day 2010-01-01 to 2019-12-31, forecast leads 27 to 54 h
(3-hourly, 10 GRIB messages). Those leads bracket every local hour of D+1 for the run initialised at
00 UTC on D. Only the needed messages are fetched (HTTP range from the .idx file), about 7.5 MB a day,
because the full member file is 60 MB and gene is on a phone tether. Each day is decoded at once to the
12 nearest 0.25-degree grid cells and saved as one small JSON; the GRIB bytes are not kept.

    ~/nyiso-us/.venv_grib/bin/python -I fetch_reforecast.py [start_year end_year]   # skip-if-present
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "us" / "pipeline"))
try:
    from points import POINTS
except ImportError:                                   # gene layout: ~/nyiso-us/pipeline_v2 next to pipeline
    sys.path.insert(0, str(Path.home() / "nyiso-us" / "pipeline"))
    from points import POINTS

BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OUT = Path.home() / "nyiso-us" / "raw" / "gefs_reforecast"
LEADS = list(range(27, 55, 3))
MEMBER = "c00"


def get(url: str, rng: str | None = None, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:                        # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def ranges(idx: str, total_hint: int | None = None) -> dict[int, str]:
    lines = [l.split(":") for l in idx.strip().splitlines()]
    out = {}
    for i, l in enumerate(lines):
        lead = int(l[5].split()[0]) if "hour fcst" in l[5] else None
        if lead in LEADS:
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
            out[lead] = f"{l[1]}-{end}"
    return out


def nearest_index(lat: float, lon: float, ni: int, nj: int, lat0: float, lon0: float, d: float):
    j = int(round((lat0 - lat) / d))
    i = int(round(((lon % 360) - lon0) / d)) % ni
    return j * ni + i, lat0 - j * d, lon0 + i * d


def one_day(day: dt.date):
    f = OUT / str(day.year) / f"{day:%Y%m%d}.json"
    if f.exists():
        return "skip"
    init = f"{day:%Y%m%d}00"
    url = f"{BASE}/{day.year}/{init}/{MEMBER}/Days:1-10/tmp_2m_{init}_{MEMBER}.grib2"
    rg = ranges(get(url + ".idx").decode())
    if sorted(rg) != LEADS:
        return f"FAIL {day} leads {sorted(rg)}"
    rec = {"init_utc": f"{day}T00:00", "member": MEMBER, "source": url, "points": {}, "leads": {}}
    for lead in LEADS:
        h = eccodes.codes_new_from_message(get(url, rg[lead]))
        try:
            ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dd = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
            assert eccodes.codes_get(h, "forecastTime") == lead or eccodes.codes_get(h, "endStep") == lead
            assert eccodes.codes_get(h, "shortName") in ("2t", "t2m", "t")
            v = eccodes.codes_get_values(h)
            temps = {}
            for name, (_zone, lat, lon, _p) in POINTS.items():
                k, glat, glon = nearest_index(lat, lon, ni, nj, lat0, lon0, dd)
                temps[name] = round(float(v[k]) - 273.15, 2)
                rec["points"][name] = [glat, glon]
            rec["leads"][str(lead)] = temps
        finally:
            eccodes.codes_release(h)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(rec))
    tmp.rename(f)
    return "ok"


def main(y0: int, y1: int):
    days = [dt.date(y0, 1, 1) + dt.timedelta(n) for n in range((dt.date(y1 + 1, 1, 1) - dt.date(y0, 1, 1)).days)]
    t0, n = time.time(), 0
    with ThreadPoolExecutor(4) as ex:
        for d, r in zip(days, ex.map(lambda d: (lambda: _safe(d))(), days)):
            n += 1
            if r != "skip" and (r != "ok" or n % 50 == 0):
                print(time.strftime("%H:%M:%S"), d, r, f"{n}/{len(days)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", len(list(OUT.rglob("*.json"))), "days", flush=True)


def _safe(d):
    try:
        return one_day(d)
    except Exception as e:                            # noqa: BLE001
        return f"FAIL {d} {e!r}"


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2010, int(a[1]) if len(a) > 1 else 2019)
