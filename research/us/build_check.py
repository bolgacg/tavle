"""Day-one data check for the New York study, on the BUILD YEARS ONLY (2020 to 2023).

Reads NYISO monthly zone archives (day-ahead damlbmp, hourly integrated real-time rtlbmp) from
data/raw/us/, keeps the 11 load zones, pairs day-ahead with real-time hour by hour, and writes
research/us/results/build_check.json: per zone, the average gap (real-time minus day-ahead), the
share of hours real-time settled below day-ahead, and what "always virtual supply" earned per MWh
before fees, with and without the most extreme 1% of hours. 2024 onward is never read here.
"""
import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data/raw/us"
OUT = Path(__file__).resolve().parent / "results"
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
YEARS = range(2020, 2024)


def read_month(kind: str, ym: str) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(RAW / f"{ym}01{kind}_zone_csv.zip") as z:
        for name in sorted(z.namelist()):
            df = pd.read_csv(io.TextIOWrapper(z.open(name)))
            # Fall-back days repeat 01:00; row order is the only way to tell the two apart.
            df["dup"] = df.groupby(["Time Stamp", "Name"]).cumcount()
            frames.append(df[["Time Stamp", "Name", "dup", "LBMP ($/MWHr)"]])
    return pd.concat(frames)


rows, inventory = [], {"months": 0, "da_rows": 0, "rt_rows": 0, "paired_hours": 0}
for y in YEARS:
    for m in range(1, 13):
        ym = f"{y}{m:02d}"
        da, rt = read_month("damlbmp", ym), read_month("rtlbmp", ym)
        da, rt = da[da.Name.isin(ZONES)], rt[rt.Name.isin(ZONES)]
        pair = da.merge(rt, on=["Time Stamp", "Name", "dup"], suffixes=("_da", "_rt"))
        inventory["months"] += 1
        inventory["da_rows"] += len(da)
        inventory["rt_rows"] += len(rt)
        inventory["paired_hours"] += len(pair)
        rows.append(pair)

p = pd.concat(rows)
p["gap"] = p["LBMP ($/MWHr)_rt"] - p["LBMP ($/MWHr)_da"]
zones = []
for z, g in p.groupby("Name"):
    supply = -g["gap"].to_numpy()          # virtual supply earns day-ahead minus real-time
    cut = np.quantile(np.abs(g["gap"]), 0.99)
    calm = supply[np.abs(g["gap"]) <= cut]
    zones.append({
        "zone": z, "hours": int(len(g)),
        "mean_gap": round(float(g["gap"].mean()), 2),
        "share_rt_below_da": round(float((g["gap"] < 0).mean()), 3),
        "supply_per_mwh": round(float(supply.mean()), 2),
        "supply_per_mwh_without_top1pct": round(float(calm.mean()), 2),
        "worst_hour_for_supply": round(float(supply.min()), 2),
    })
OUT.mkdir(exist_ok=True)
json.dump({"years": "2020-2023 (build years only)", "inventory": inventory, "zones": zones},
          open(OUT / "build_check.json", "w"), indent=1)
print(json.dumps(inventory))
for z in zones: print(z)
