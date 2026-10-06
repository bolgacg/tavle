"""Fetch archived GFS 2 m temperature forecasts (Open-Meteo previous-runs API, variable
temperature_2m_previous_day2 = the run issued about two days before each hour) for one point per
NYISO load zone, one JSON per point per year, into ~/nyiso-us/raw/gfs/. Skips files already
present and complete. 2020 is skipped: the archive returns nulls before 2021.
Stdlib only, so it runs before the venv exists."""
import json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import POINTS  # noqa: E402

OUT = Path.home() / "nyiso-us/raw/gfs"
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=temperature_2m_previous_day2&start_date={s}&end_date={e}&models=gfs_seamless")


def complete(p: Path) -> bool:
    try:
        d = json.loads(p.read_text())
        return len(d["hourly"]["time"]) > 0 and "temperature_2m_previous_day2" in d["hourly"]
    except Exception:
        return False


fails = 0
for name, (zone, lat, lon, primary) in POINTS.items():
    for year in range(2021, 2027):
        s, e = f"{year}-01-01", ("2026-09-30" if year == 2026 else f"{year}-12-31")
        f = OUT / f"{name}_{year}.json"
        if f.exists() and complete(f):
            continue
        url = URL.format(lat=lat, lon=lon, s=s, e=e)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    body = r.read()
                d = json.loads(body)
                if "hourly" not in d:
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
print(f"GFS DONE fails={fails} files={len(list(OUT.glob('*.json')))}", flush=True)
