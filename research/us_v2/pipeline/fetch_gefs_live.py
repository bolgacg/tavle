"""Archived live GEFS v12 (AWS noaa-gefs-pds) 2 m temperature at the 12 study points, idea V16.

Same model, member, run, leads, points and output format as fetch_reforecast.py: control member c00, 00 UTC
run of every day from 2020-09-23 (the first GEFS v12 cycle in the archive) to 2026-09-30, leads 27 to 54 h
(3-hourly, 10 GRIB messages, one file per lead: gefs.YYYYMMDD/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.fNNN).
Only the TMP:2 m message is fetched (HTTP range from each .idx), decoded at once to the 12 nearest
0.25-degree cells and saved as one small JSON in ~/nyiso-us/raw/gefs_live/YYYY/YYYYMMDD.json.

Version check (7 Oct 2026): atmos/pgrb2sp25 is GEFS v12 (0.25-degree grid, 1440 x 721, generating process
107, 30 perturbed members, as the reforecast's grid); the v11 files (pgrb2a, 1 degree, 20 members) stop
on 2020-09-23. Each message is asserted to be 2t, control (perturbationNumber 0), 0.25 degree, 30 members.

HOLDOUT: days on or after 2024-01-01 are downloaded and written for the held-out run, never printed; the log
shows status and counts only, for every year.

    nohup python3 fetch_gefs_live.py [YYYY-MM-DD YYYY-MM-DD] > ~/nyiso-us/logs/fetch_gefs_live.log 2>&1 &
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "us" / "pipeline"))
try:
    from points import POINTS
except ImportError:                                   # gene layout
    sys.path.insert(0, str(Path.home() / "nyiso-us" / "pipeline"))
    from points import POINTS

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
OUT = Path.home() / "nyiso-us" / "raw" / "gefs_live"
LEADS = list(range(27, 55, 3))
MEMBER = "c00"
FIRST, LAST = dt.date(2020, 9, 23), dt.date(2026, 9, 30)
THREADS = 16


def get(url: str, rng: str | None = None, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            if k == tries - 1:
                raise
        except Exception:                             # noqa: BLE001
            if k == tries - 1:
                raise
        time.sleep(5 * (k + 1))


def t2m_range(idx: str, lead: int) -> str:
    lines = [l.split(":") for l in idx.strip().splitlines()]
    hits = [i for i, l in enumerate(lines)
            if l[3] == "TMP" and l[4] == "2 m above ground" and l[5] == f"{lead} hour fcst"]
    if len(hits) != 1:
        raise ValueError(f"lead {lead}: {len(hits)} TMP:2 m messages in idx")
    i = hits[0]
    end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
    return f"{lines[i][1]}-{end}"


def nearest_index(lat: float, lon: float, ni: int, nj: int, lat0: float, lon0: float, d: float):
    j = int(round((lat0 - lat) / d))
    i = int(round(((lon % 360) - lon0) / d)) % ni
    return j * ni + i, lat0 - j * d, lon0 + i * d


def one_day(day: dt.date):
    f = OUT / str(day.year) / f"{day:%Y%m%d}.json"
    if f.exists():
        return "skip"
    run = f"{BASE}/gefs.{day:%Y%m%d}/00/atmos/pgrb2sp25"
    rec = {"init_utc": f"{day}T00:00", "run_time_utc": f"{day}T00:00:00Z", "cycle": "00",
           "model": "GEFS v12 (operational, archived)", "member": MEMBER,
           "source": f"{run}/gec00.t00z.pgrb2s.0p25.fNNN", "points": {}, "leads": {}}
    for lead in LEADS:
        url = f"{run}/gec00.t00z.pgrb2s.0p25.f{lead:03d}"
        try:
            rg = t2m_range(get(url + ".idx").decode(), lead)
        except FileNotFoundError:
            return f"MISSING {day} f{lead:03d}"
        h = eccodes.codes_new_from_message(get(url, rg))
        try:
            g = lambda k: eccodes.codes_get(h, k)    # noqa: E731
            ni, nj = g("Ni"), g("Nj")
            lat0, lon0, dd = (g("latitudeOfFirstGridPointInDegrees"), g("longitudeOfFirstGridPointInDegrees"),
                              g("iDirectionIncrementInDegrees"))
            assert g("shortName") == "2t" and g("perturbationNumber") == 0
            assert g("forecastTime") == lead and g("dataDate") == int(f"{day:%Y%m%d}") and g("dataTime") == 0
            assert (ni, nj, dd) == (1440, 721, 0.25) and g("numberOfForecastsInEnsemble") == 30
            assert g("generatingProcessIdentifier") == 107
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


def _safe(d):
    try:
        return one_day(d)
    except Exception as e:                            # noqa: BLE001  (message only, never a value)
        return f"FAIL {d} {type(e).__name__}: {str(e)[:160]}"


def main(a: dt.date, b: dt.date):
    days = [a + dt.timedelta(n) for n in range((b - a).days + 1)]
    t0, n = time.time(), 0
    with ThreadPoolExecutor(THREADS) as ex:
        for d, r in zip(days, ex.map(_safe, days)):
            n += 1
            if r != "skip" and (r != "ok" or n % 50 == 0):
                print(time.strftime("%H:%M:%S"), d, r, f"{n}/{len(days)} {time.time() - t0:.0f}s", flush=True)
    counts = {y.name: len(list(y.glob("*.json"))) for y in sorted(OUT.iterdir()) if y.is_dir()}
    print("DONE", sum(counts.values()), "days", json.dumps(counts), flush=True)


if __name__ == "__main__":
    x = sys.argv[1:]
    main(dt.date.fromisoformat(x[0]) if x else FIRST, dt.date.fromisoformat(x[1]) if len(x) > 1 else LAST)
