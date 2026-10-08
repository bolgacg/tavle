"""v2 scorer: every positions file in results/v2/pos/ scored the same way, with the v1 lab's own functions
(research/us/side/lab.py: Frame money, evaluate, placebos, fragility, sizing view), over 2013..2023.

What changes from the lab, and only this: the scored years are 2013..2023 (11 years), costs per year come from
the extended fee table (rolling.costs, 2010 on), and the bar's year rule scales from "2 of 3" to "8 of 11"
(two thirds, rounded up). Everything else (average net >= 50,000 USD a year, whole-period Sharpe > 0.42,
positive total at the 0.50 stress cost, placebos, fragility flags, sizing view) is the lab's code unchanged.

    python score_v2.py [--draws 1000] [--quick]     -> results/v2/v2_results.json and v2_results.md

Columns: per-year net 2013..2023, total, total without the best 5 days, Sharpe, sizing view, PASS, FRAGILE,
placebos (one day late, permuted within year), and the tries count (every row x its settings grid, plus the
ideas tried in v1).

Run modes (rolling.MODE dryrun / heldout): the same scoring restricted to the test window (2023, or 2024-01-01 to
2026-09-30; 2026 is nine months), every metric, placebo and fragility check of the lab unchanged on that window, the
bar's year rule two thirds of the window's years; the sizing view is CARRIED from the build (each row's
scale_for_100k_drawdown in the build's v2_results.json, never refitted on the window) and reported as
sizing_view_carried. Writes v2_results_<mode>.{json,md}, v2_daily_<mode>.parquet (the window, the build file's format)
and v2_daily_all_<mode>.parquet (the build's daily table before the window, then the window).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import rolling as R

BUILD = R.MODE == "build"
FIRST = R.FIRST_SCORED if BUILD else R.TEST_WINDOW[0]
YEARS = list(range(FIRST.year, R.LAST_SCORED.year + 1))
BUILD_JSON = R.BUILD_RESULTS / "v2_results.json"
BUILD_DAILY = R.BUILD_RESULTS / "v2_daily.parquet"
BUILD_DAILY_INDEX = "__index_level_0__"              # the daily table's unnamed date index (naive local dates)


def import_lab():
    for p in (str(R.V1_SIDE), str(R.V1_PIPELINE)):
        if p not in sys.path:
            sys.path.insert(1, p)
    import lab
    sup, lod = R.costs()
    lab.YEARS = YEARS
    lab.COST = {y: sup[y] for y in YEARS}
    lab.LOADCOST = {y: lod[y] for y in YEARS}
    lab.BAR = {**lab.BAR, "positive_years": math.ceil(2 * len(YEARS) / 3)}
    return lab


class FrameV2:
    """lab.Frame for 2013..2023 (same fields); zone-hours without a settled gap are dropped (not tradable),
    and a missing whole day is logged instead of failing."""

    def __init__(self, lab, px: pd.DataFrame):
        h = px.copy()
        h["delivery_hour"] = h["delivery_hour"].dt.tz_convert(R.TZ)
        h = h.sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
        h["gap"] = h["rt_lbmp"] - h["da_lbmp"]
        h = h[h["gap"].notna()].reset_index(drop=True)
        h["ddate"] = h["delivery_hour"].dt.tz_localize(None).dt.normalize()
        h["hour"] = h["delivery_hour"].dt.hour
        occ = h.groupby(["ddate", "zone", "hour"]).cumcount()
        h["slot"] = np.where(occ > 0, 24, h["hour"])
        self.h = h
        self.days = pd.DatetimeIndex(sorted(h["ddate"].unique()))
        full = pd.date_range(self.days.min(), self.days.max(), freq="D")
        self.missing_days = [str(d.date()) for d in full.difference(self.days)]
        R.assert_pre2024(self.days, "frame days")
        self.ND = len(self.days)
        self.day_year = self.days.year.to_numpy()
        self.DI = self.days.get_indexer(h["ddate"])
        self.ZI = pd.Index(R.ZONES).get_indexer(h["zone"])
        self.SL = h["slot"].to_numpy()
        self.YEAR = h["ddate"].dt.year.to_numpy()
        self.GAP = h["gap"].to_numpy(float)
        self.SUPC = pd.Series(self.YEAR).map(lab.COST).to_numpy(float)
        self.LOADC = pd.Series(self.YEAR).map(lab.LOADCOST).to_numpy(float)
        self.NH = len(h)
        self.zone = h["zone"].to_numpy()

    day_pnl = None   # bound below from lab.Frame
    day_mwh = None
    rows = None
    to_cube = None


def bind_frame(lab):
    for k in ("day_pnl", "day_mwh", "rows", "to_cube"):
        setattr(FrameV2, k, getattr(lab.Frame, k))


def load_prices() -> pd.DataFrame:
    px = R.read_pre2024(R.PARQUET / "prices_zone.parquet", "delivery_hour",
                        ["delivery_hour", "zone", "da_lbmp", "rt_lbmp"], ds.field("zone").isin(R.ZONES))
    px["delivery_hour"] = px["delivery_hour"].dt.tz_convert(R.TZ)
    loc = px["delivery_hour"].dt.tz_localize(None)
    return px[(loc >= pd.Timestamp(FIRST)) & (loc < pd.Timestamp(R.LAST_SCORED) + pd.Timedelta(days=1))]


def positions_to_mw(F: FrameV2, pos: pd.DataFrame) -> np.ndarray:
    p = pos.copy()
    p["delivery_hour"] = pd.to_datetime(p["delivery_hour"]).dt.tz_convert(R.TZ)
    p = p.groupby(["delivery_hour", "zone"])["mw"].sum()
    mi = pd.MultiIndex.from_arrays([F.h["delivery_hour"], F.h["zone"]])
    return np.nan_to_num(p.reindex(mi).to_numpy(float))


def score_one(lab, F, name: str, mw: np.ndarray, meta: dict, RD, BT, placebo: bool = True) -> dict:
    cube = F.to_cube(mw)
    st = lab.Strat(name, meta.get("idea", "?"), meta.get("line", ""), {"mw": cube},
                   lambda F_, inp, p: np.nan_to_num(F_.rows(inp["mw"])), placebo=placebo and name != "V1_always_supply")
    res = lab.evaluate(F, mw)
    ch = {Y: (None, "per-quarter settings, see positions json") for Y in YEARS}
    total = res["three_year"]["net_usd"]
    plc = lab.placebos(F, st, st.inputs, total) if st.placebo else {"note": "not applicable"}
    frag = lab.fragility(F, st, st.inputs, ch, res, RD, BT) if RD is not None else {"FRAGILE": None,
                                                                                   "fragile_reasons": ["not run (--quick)"]}
    daily = res.pop("_daily")
    return {"name": name, "idea": meta.get("idea"), "line": meta.get("line"), "label": meta.get("label"),
            "comparison": bool(meta.get("comparison")),
            "settings_tried": int(meta.get("settings_tried", 1)),
            "years": {y: res["years"][str(y)]["net_usd"] for y in YEARS},
            "total": total, "total_without_best_5_days": res["three_year"]["net_without_best_5_days"],
            "sharpe": res["three_year"]["sharpe"], "max_drawdown_usd": res["three_year"]["max_drawdown_usd"],
            "stress_0.50_total": res["three_year"]["stress_0.50_usd"], "avg_net_per_year": res["avg_net_per_year"],
            "mwh_per_day": res["three_year"]["mwh_per_day"], "usd_per_mwh": res["three_year"]["usd_per_mwh"],
            "bar": res["bar"], "PASS": res["PASS"], "FRAGILE": frag.get("FRAGILE"),
            "fragile_reasons": frag.get("fragile_reasons"), "fragility": frag, "placebos": plc,
            "too_good": res["too_good"], "sizing_view": res["sizing_view"], "detail": res,
            "_daily": daily}


def tries(rows: list[dict], lab) -> dict:
    n = sum(r["settings_tried"] for r in rows if not r.get("comparison"))
    return {"v2_rows": len(rows), "v2_settings_tried": n,
            "v1_before": "v1 lab: ideas 1 to 14, the three basics, and their grids (research/us/side/lab.py TRIED_BEFORE)",
            "text": f"{len(rows)} v2 strategy rows, {n} settings tried in all on 2013..2023 rolling quarters, "
                    f"on top of everything v1 tried on 2021..2023."}


def _f(x):
    if x is None:
        return "n/a"
    return f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 10 else str(x)


def write_md(out: dict, path):
    span = f"{FIRST} to {R.LAST_SCORED}"
    title = ("# New York study v2: rolling-window results, 2013 to 2023 (build years only)" if BUILD else
             f"# New York study v2: {R.MODE} run, scored on {span}")
    read = ("Nothing on or after 2024-01-01 was read." if R.HOLDOUT <= dt.date(2024, 1, 1) else
            f"Data read up to {R.LAST_SCORED} (the held-out window).")
    L = [title, "",
         f"Written {out['written']}. Train on the previous 3 years, test on the next quarter, refit quarterly; "
         "settings chosen on the trailing 4 out-of-sample quarters. Net USD at full cost per year (1 MW per "
         f"position). {read}", "",
         f"Bar: average net >= 50,000 USD a year, positive in at least {out['bar_positive_years']} of {len(YEARS)} "
         f"years, Sharpe > 0.42 over {YEARS[0]} to {YEARS[-1]}, positive total at 0.50 USD/MWh stress cost.", ""]
    hdr = ["row"] + [str(y) for y in YEARS] + ["total", "w/o best 5 days", "Sharpe", "USD/MWh", "scale for 100k DD",
                                               "PASS", "FRAGILE", "placebo 1 day late", "placebo permuted mean", "tries"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in out["rows"]:
        p = r["placebos"]
        nm = r["name"] + (" (comparison)" if r.get("comparison") else "")
        L.append("| " + " | ".join([nm] + [_f(r["years"][y]) for y in YEARS] + [
            _f(r["total"]), _f(r["total_without_best_5_days"]), _f(r["sharpe"]), _f(r["usd_per_mwh"]),
            _f(r["sizing_view"]["scale_for_100k_drawdown"]), "yes" if r["PASS"] else "no",
            {True: "yes", False: "no"}.get(r["FRAGILE"], "n/a"), _f(p.get("shifted_one_day_total")),
            _f(p.get("shuffled_mean_total")), str(r["settings_tried"])]) + " |")
    labels = [r for r in out["rows"] if r.get("label")]
    if labels:
        L += ["", "Labels (addendum of 7 Oct):", ""] + [f"- {r['name']}: {r['label']}" for r in labels]
    L += ["", "Fragile reasons:", ""]
    for r in out["rows"]:
        if r["FRAGILE"]:
            L.append(f"- {r['name']}: {'; '.join(r['fragile_reasons'])}")
    L += ["", "Too-good flags (a year with Sharpe above 3, or one day above half the year's profit):", ""]
    for r in out["rows"]:
        for t in r["too_good"]:
            L.append(f"- {r['name']} {t['year']}: {t['why']}")
    if not BUILD:
        L += ["", "Sizing view carried from the build (scale = MW per position at which the build-year worst "
              "drawdown is 100,000 USD, capped at 5; not refitted here):", "",
              "| row | scale (build) | net at that scale | worst drawdown at that scale | return a year on 500,000 |",
              "|---|---|---|---|---|"]
        for r in out["rows"]:
            c = r.get("sizing_view_carried") or {}
            L.append(f"| {r['name']} | {_f(c.get('scale'))} | {_f(c.get('net_usd'))} | "
                     f"{_f(c.get('max_drawdown_usd'))} | {_f(c.get('return_on_500k_a_year_pct'))} |")
    L += ["", f"Tries: {out['tries']['text']}", "",
          "Sizing view (not part of the verdict): scale = MW per position at which the worst drawdown is 100,000 "
          "USD, capped at 5.", ""]
    if out.get("missing_days"):
        L.append(f"Days without any settled price, not scored: {len(out['missing_days'])}.")
    path.write_text("\n".join(L) + "\n")


def carried_sizing(lab, rows: list[dict], daily: pd.DataFrame) -> None:
    """Run modes: each row's build scale applied to the window's daily net (P&L, costs and drawdown scale
    linearly with the book)."""
    build = {r["name"]: r for r in json.loads(BUILD_JSON.read_text())["rows"]} if BUILD_JSON.exists() else {}
    for r in rows:
        b = build.get(r["name"])
        sc = ((b or {}).get("sizing_view") or {}).get("scale_for_100k_drawdown")
        if sc is None:
            r["sizing_view_carried"] = {"scale": None, "note": "row not in the build results"}
            continue
        d = sc * daily[r["name"]].to_numpy(float)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        r["sizing_view_carried"] = {
            "scale": sc, "source": str(BUILD_JSON), "net_usd": round(float(d.sum())),
            "max_drawdown_usd": round(float((cum - np.maximum.accumulate(cum)).min())),
            "return_on_500k_a_year_pct": round(100 * float(d.sum()) / lab.BANK * 365 / len(d), 1),
            "years": {y: round(float(d[daily.index.year == y].sum())) for y in YEARS},
            "days": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="no fragility (no random-days draws, no bootstrap)")
    ap.add_argument("--tag", default="" if BUILD else f"_{R.MODE}")
    a = ap.parse_args()
    lab = import_lab()
    bind_frame(lab)
    F = FrameV2(lab, load_prices())
    R.log(f"frame {F.NH:,} zone-hours, {F.ND} days, missing days {len(F.missing_days)}")
    RD = None if a.quick else lab.RandomDays(F, a.draws)
    BT = None if a.quick else lab.Boot(F.ND)
    rows = []
    for f in sorted(R.POS.glob("*.parquet")):
        meta_f = f.with_suffix(".json")
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        pos = R.read_pre2024(f, "delivery_hour")
        mw = positions_to_mw(F, pos)
        t = time.time()
        rows.append(score_one(lab, F, f.stem, mw, meta, RD, BT))
        R.log(f"scored {f.stem}: total {rows[-1]['total']:,}, PASS {rows[-1]['PASS']}, {time.time() - t:.0f} s")
    out = {"written": time.strftime("%Y-%m-%d %H:%M"), "years": YEARS, "costs": {"supply": lab.COST, "load": lab.LOADCOST},
           "bar_positive_years": lab.BAR["positive_years"], "missing_days": F.missing_days,
           "rows": rows, "tries": tries(rows, lab)}
    daily = pd.DataFrame({r["name"]: r.pop("_daily") for r in rows}, index=F.days)
    R.RESULTS.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(R.RESULTS / f"v2_daily{a.tag}.parquet")
    if not BUILD:
        carried_sizing(lab, rows, daily)
        out["window"] = [str(FIRST), str(R.LAST_SCORED)]
        out["build_results"] = str(BUILD_JSON)
        if BUILD_DAILY.exists():
            b = R.read_pre2024(BUILD_DAILY, BUILD_DAILY_INDEX)
            b = b.set_index(BUILD_DAILY_INDEX).rename_axis(None)
            b = b[b.index < pd.Timestamp(FIRST)]
            pd.concat([b, daily], axis=0).to_parquet(R.RESULTS / f"v2_daily_all{a.tag}.parquet")
    (R.RESULTS / f"v2_results{a.tag}.json").write_text(json.dumps(out, indent=1, default=lab._js))
    write_md(out, R.RESULTS / f"v2_results{a.tag}.md")
    R.log(f"-> {R.RESULTS / f'v2_results{a.tag}.md'}")


if __name__ == "__main__":
    main()
