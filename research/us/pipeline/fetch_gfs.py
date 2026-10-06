"""Fetch archived GFS 2 m temperature forecasts (Open-Meteo previous-runs API) for one point per
NYISO load zone, one JSON per point, year and lead, into ~/nyiso-us/raw/gfs/:
    temperature_2m_previous_day2  (value predicted about 48 h before each hour)  <point>_<year>.json
    temperature_2m_previous_day3  (value predicted about 72 h before each hour)  <point>_<year>_day3.json
The three-day weather rule (OBJECTIVES addendum, 6 Oct evening) uses day2 for delivery hours up to
21:00 New York time and day3 for 22:00 and 23:00. Skips files already present and complete, so
re-running only fetches what is missing. 2020 is skipped: the archive returns nulls before 2021.
Stdlib only, so it runs before the venv exists.

    python3 fetch_gfs.py            # day2 and day3
    python3 fetch_gfs.py 3          # day3 only
"""
import json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import POINTS  # noqa: E402

OUT = Path.home() / "nyiso-us/raw/gfs"
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly={var}&start_date={s}&end_date={e}&models=gfs_seamless")
DAYS = [int(a) for a in sys.argv[1:]] or [2, 3]


def var(day: int) -> str:
    return f"temperature_2m_previous_day{day}"


def path(name: str, year: int, day: int) -> Path:
    return OUT / (f"{name}_{year}.json" if day == 2 else f"{name}_{year}_day{day}.json")


def complete(p: Path, day: int) -> bool:
    try:
        d = json.loads(p.read_text())
        return len(d["hourly"]["time"]) > 0 and var(day) in d["hourly"]
    except Exception:
        return False


fails = 0
for day in DAYS:
    for name, (zone, lat, lon, primary) in POINTS.items():
        for year in range(2021, 2027):
            s, e = f"{year}-01-01", ("2026-09-30" if year == 2026 else f"{year}-12-31")
            f = path(name, year, day)
            if f.exists() and complete(f, day):
                continue
            url = URL.format(lat=lat, lon=lon, var=var(day), s=s, e=e)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r:
                        body = r.read()
                    d = json.loads(body)
                    if "hourly" not in d or var(day) not in d["hourly"]:
                        raise ValueError(str(d)[:200])
                    d["_request"] = {"url": url, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    tmp = f.with_suffix(".part")
                    tmp.write_text(json.dumps(d))
                    tmp.rename(f)
                    print("ok", f.name, len(d["hourly"]["time"]), flush=True)
                    break
                except Exception as ex:
                    print("retry", f.name, attempt, repr(ex)[:200], flush=True)
                    time.sleep(20 * (attempt + 1))
            else:
                fails += 1
                print("FAIL", f.name, flush=True)
            time.sleep(3)  # be polite to the free API
print(f"GFS DONE days={DAYS} fails={fails} files={len(list(OUT.glob('*.json')))}", flush=True)
