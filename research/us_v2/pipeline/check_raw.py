"""Format and naming checks on the 2010-2023 raw NYISO zips (no price statistic; names, PTIDs, column
headers, row counts, DST-day hour counts and zip-entry write times only).

    ../.venv/bin/python check_raw.py  ->  results/v2/raw_checks.json
"""
from __future__ import annotations

import collections
import datetime as dt
import io
import json
import zipfile

import pandas as pd

import common as C

SERIES = ["damlbmp_zone", "rtlbmp_zone", "damlbmp_gen", "rtlbmp_gen", "isolf", "outSched"]


def main():
    out = {"headers": collections.defaultdict(lambda: collections.Counter()),
           "zone_names": {}, "zone_ptids": {}, "gen_points": {}, "gen_renamed_ptids": {},
           "dst_rows_per_name": {}, "write_time_ranges": {}, "missing_months": [], "missing_days": {}}
    gen_names: dict[int, set] = collections.defaultdict(set)
    for y in range(C.FIRST_DAY.year, C.LAST_DAY.year + 1):
        names_y, ptid_y, gen_y = collections.Counter(), {}, set()
        wt = collections.defaultdict(list)
        days = collections.Counter()
        for m in range(1, 13):
            ym = f"{y}{m:02d}01"
            for s in SERIES:
                p = C.zip_path(ym, s)
                if not p.exists():
                    out["missing_months"].append(p.name)
                    continue
                z = zipfile.ZipFile(p)
                for info in z.infolist():
                    d = dt.datetime.strptime(info.filename[:8], "%Y%m%d").date()
                    days[s] += 1
                    w = dt.datetime(*info.date_time)
                    wt[s].append(((w - dt.datetime.combine(d, dt.time())).total_seconds() / 3600))
                    raw = z.open(info).read().decode("latin-1")
                    out["headers"][s][raw.splitlines()[0]] += 1
                    if s in ("damlbmp_zone", "damlbmp_gen") or (s == "isolf" and d.month in (3, 11)):
                        df = pd.read_csv(io.StringIO(raw), usecols=[0, 1, 2] if s != "isolf" else [0])
                        if s == "damlbmp_zone":
                            names_y.update(df.iloc[:, 1].unique())
                            for n, pt in zip(df.iloc[:, 1], df.iloc[:, 2]):
                                ptid_y.setdefault(n, set()).add(int(pt))
                        elif s == "damlbmp_gen":
                            for n, pt in set(zip(df.iloc[:, 1], df.iloc[:, 2])):
                                gen_y.add(int(pt))
                                gen_names[int(pt)].add(n)
                        # DST days: rows per name (expect 23 / 25)
                        if d.month in (3, 11) and d.weekday() == 6 and (
                                (d.month == 3 and 8 <= d.day <= 14) or (d.month == 11 and d.day <= 7)):
                            per = df.groupby(df.columns[1]).size() if s != "isolf" else None
                            key = f"{s} {d}"
                            if per is not None:
                                out["dst_rows_per_name"][key] = sorted(set(per.tolist()))
                            else:
                                ts = df.iloc[:, 0].str[:10]
                                out["dst_rows_per_name"][key] = int((ts == d.strftime("%m/%d/%Y")).sum())
        out["zone_names"][y] = sorted(names_y)
        out["zone_ptids"][y] = {k: sorted(v) for k, v in ptid_y.items()}
        out["gen_points"][y] = len(gen_y)
        out["missing_days"][y] = {s: (366 if y % 4 == 0 else 365) - days[s] for s in SERIES}
        out["write_time_ranges"][y] = {s: [round(min(v), 2), round(pd.Series(v).quantile(0.5), 2),
                                            round(pd.Series(v).quantile(0.99), 2), round(max(v), 2)]
                                       for s, v in wt.items()}
        print(y, out["gen_points"][y], out["missing_days"][y], flush=True)
    out["gen_renamed_ptids"] = {pt: sorted(n) for pt, n in gen_names.items() if len(n) > 1}
    out["headers"] = {s: dict(c) for s, c in out["headers"].items()}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "raw_checks.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
