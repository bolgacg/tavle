"""Chart data for the redesigned New York results page.

Reads (never writes) the scored results on gene and writes compact JSON files for the page's charts:

    equity_v1.json      daily cumulative net P&L (USD, 1 MW) and the bankroll path at the sizing view, every v1 row
    equity_v2.json      the same for every v2 row, 2013 onward, comparison rows flagged
    drawdown.json       underwater series (USD below the running peak, 1 MW), top rows, v1 and v2
    monthly.json        monthly net P&L per row (heatmap), every row, v1 and v2
    rolling_sharpe.json rolling 365-day Sharpe, top rows (v1 daily, v2 weekly)
    years.json          net P&L per calendar year, every row, checked against the results files
    placebo.json        real total against one day late, permuted and random days, every row (top rows flagged)
    seeds.json          the 20-seed spread per deep row (robustness checks)
    events.json         weather events and the dated addenda (date, label, commit) for the timeline

HOLDOUT. By default every daily table is filtered AT READ (pyarrow filter) to dates before 2024-01-01 and the
result is asserted; per-year fields from the summary files are dropped for 2024 onward. The held-out run has not
happened. When it has, rerun with --include-heldout (and point --v1-daily/--v2-daily etc. at the files that hold
it); the page data then runs to --end.

    gene:   cd ~/nyiso-us/chartdata && ../.venv/bin/python -I build_chartdata.py --out out
    laptop: python3 build_chartdata.py --only events --repo ~/projects/tavle --out ~/projects/tavle/docs/us/data

The data step needs gene's results tree (--root, default ~/nyiso-us). The events step needs the git repo for the
addenda commits (--repo); without it, the addenda already in --out/events.json are kept.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

BANK = 500_000
HOLDOUT_START = "2024-01-01"

# Flags from audits that make a row fragile for the selection rule (OBJECTIVES.md, 7 Oct: "including flags from
# audits"), and audit notes that do not change the flag but belong beside the row.
AUDIT_FRAGILE = {
    "idea10_weather_surprise": "fragile per idea 10 audit (side/idea10_audit.md)",
    "idea2_storm_flip": "fragile per storm audit (side/storm_audit.md)",
    "storm_day_flip [lead]": "fragile per storm audit (side/storm_audit.md)",
}
AUDIT_NOTES = {
    "V2_limit_gbm": "stage-1 audit: the profit is the gradient-boosting direction, the limit prices add nothing",
    "V3_tail_gbm": "stage-1 audit: the total reflects about 3 MW positions; per MWh it is the V2 direction bet",
    "V1_C_deep": "stage-1 audit: about 76 percent of the profit is a static pair spread",
    "V11_C_deep_ens4": "stage-1 audit: shares C's pair machinery, read as V1_C_deep (not separately audited)",
    "V10_C_deep_pretrain": "stage-1 audit: shares C's pair machinery, read as V1_C_deep (not separately audited)",
}
REFERENCE_ROWS = {"always_supply", "baseline_usual_side", "V1_always_supply", "V1_baseline"}

WEATHER_EVENTS = [
    {"id": "polar_vortex_2014", "start": "2014-01-06", "end": "2014-01-08",
     "label": "2014 polar vortex: Arctic air over New York, record winter demand and price spikes"},
    {"id": "cold_snap_2018", "start": "2017-12-27", "end": "2018-01-07",
     "label": "2018 cold snap: two weeks of extreme cold, the bomb cyclone of 4 January"},
    {"id": "elliott_2022", "start": "2022-12-23", "end": "2022-12-24",
     "label": "Winter Storm Elliott: flash freeze, real-time prices far above day-ahead"},
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ reading, with the holdout enforced at read
def read_daily(path: Path, end: str):
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).pandas_metadata or {}
    idx = (meta.get("index_columns") or [None])[0]
    if not isinstance(idx, str):
        raise SystemExit(f"{path}: no named date index")
    limit = pd.Timestamp(end).to_datetime64()
    tab = ds.dataset(str(path), format="parquet").to_table(filter=ds.field(idx) < limit)
    df = tab.to_pandas()
    if idx in df.columns:
        df = df.set_index(idx)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    assert df.index.max() < pd.Timestamp(end), f"HOLDOUT BREACH {path}: {df.index.max()}"
    return df.sort_index()


def years_ok(d: dict, end_year: int) -> dict:
    return {k: v for k, v in d.items() if str(k).isdigit() and int(k) < end_year}


# ------------------------------------------------------------------ row metadata
def pass_at_sizing_view(row: dict) -> bool:
    b, sv = row.get("bar") or {}, row.get("sizing_view") or {}
    ret = sv.get("return_on_500k_at_that_scale_pct")
    return bool(ret is not None and ret >= 10 and b.get("positive_years") and b.get("sharpe_3y")
                and b.get("stress_total"))


def row_meta(name: str, r: dict, version: str) -> dict:
    sv = r.get("sizing_view") or {}
    fragile = bool(r.get("FRAGILE"))
    audit_fragile = AUDIT_FRAGILE.get(name)
    psv = pass_at_sizing_view(r)
    candidate = psv and not fragile and not audit_fragile
    reasons = list(r.get("fragile_reasons") or (r.get("fragility") or {}).get("fragile_reasons") or [])
    if audit_fragile:
        reasons.append(audit_fragile)
    m = {"name": name, "idea": r.get("idea"), "line": r.get("line"), "source": r.get("source", "score_v2.py"),
         "pass_1mw": bool(r.get("PASS")), "pass_sizing_view": psv, "fragile": fragile or bool(audit_fragile),
         "fragile_reasons": reasons, "candidate_by_rule": candidate,
         "comparison": (not candidate) or bool(r.get("comparison")),
         "registered_comparison": bool(r.get("comparison")),
         "reference": name in REFERENCE_ROWS,
         "sizing_scale_mw": sv.get("scale_for_100k_drawdown"),
         "sizing_return_pct": sv.get("return_on_500k_at_that_scale_pct")}
    if name in AUDIT_NOTES:
        m["audit_note"] = AUDIT_NOTES[name]
    return m


def load_v1(a):
    res = json.loads(Path(a.v1_results).read_text())
    rows = res["strategies"]
    daily = read_daily(Path(a.v1_daily), a.end)
    return res, rows, daily


def load_v2(a, root: Path):
    rpath, dpath = a.v2_results, a.v2_daily
    if not rpath or not dpath:   # newest scored run: v2_daily{tag}.parquet with its v2_results{tag}.json
        cands = sorted((root / "results" / "v2").glob("v2_daily*.parquet"), key=lambda p: p.stat().st_mtime)
        cands = [p for p in cands if (p.parent / p.name.replace("v2_daily", "v2_results").replace(
            ".parquet", ".json")).exists()]
        if not cands:
            raise SystemExit("no scored v2 run found")
        dpath = cands[-1]
        rpath = dpath.parent / dpath.name.replace("v2_daily", "v2_results").replace(".parquet", ".json")
    res = json.loads(Path(rpath).read_text())
    rows = {r["name"]: r for r in res["rows"]}
    daily = read_daily(Path(dpath), a.end)
    return res, rows, daily, str(rpath), str(dpath)


# ------------------------------------------------------------------ series helpers
def ints(s):
    return [int(round(float(x))) for x in s]


def r2(s):
    return [None if (x is None or not math.isfinite(float(x))) else round(float(x), 2) for x in s]


def top_rows(metas: list[dict], extra: list[str]) -> list[str]:
    tops = [m["name"] for m in metas if m["candidate_by_rule"] or m["reference"]]
    return tops + [x for x in extra if x not in tops and x in {m["name"] for m in metas}]


def equity_file(version, daily, metas, start_year, src, holdout_note):
    cum = daily.cumsum()
    rows = []
    for m in metas:
        n = m["name"]
        scale = m["sizing_scale_mw"] or 0.0
        rows.append({**m, "cum_net_usd_1mw": ints(cum[n]),
                     "bankroll_usd_sizing_view": ints(BANK + scale * cum[n])})
    return {"meta": {"version": version, "written": time.strftime("%Y-%m-%d %H:%M %Z"),
                     "source": src, "unit": "USD, cumulative daily net after costs",
                     "resolution": "daily, every calendar day, no downsampling",
                     "first_day": str(daily.index[0].date()), "last_day": str(daily.index[-1].date()),
                     "days": len(daily), "scored_from_year": start_year,
                     "bankroll": f"{BANK:,} USD start; each row's daily net times its sizing-view scale "
                                 "(MW per position at which the build-year worst drawdown is 100,000 USD, "
                                 "capped at 5); costs scale linearly with size",
                     "flags": "candidate_by_rule = passes the bar at the sizing view and is not fragile "
                              "(lab flag or audit flag), computed here from the results file; comparison = "
                              "every other row (OBJECTIVES.md, 7 Oct selection rule) or a registered comparison "
                              "row; reference = always supply and the baseline; the official approved list is "
                              "published separately",
                     "holdout": holdout_note},
            "dates": [str(d.date()) for d in daily.index], "rows": rows}


def drawdown(daily, names):
    cum = daily[names].cumsum()
    under = cum - cum.cummax().clip(lower=0)
    out = {}
    for n in names:
        u = under[n]
        i = int(u.values.argmin())
        out[n] = {"underwater_usd_1mw": ints(u), "max_drawdown_usd": int(round(float(u.min()))),
                  "trough_day": str(u.index[i].date())}
    return out


def monthly(daily):
    m = daily.groupby(daily.index.to_period("M")).sum()
    return {"months": [str(p) for p in m.index], "rows": {n: ints(m[n]) for n in m.columns}}


def years(daily, rows_json, end_year, field):
    y = daily.groupby(daily.index.year).sum()
    out, checks = {}, {}
    for n in daily.columns:
        out[n] = ints(y[n])
        ref = years_ok(field(rows_json[n]), end_year) if n in rows_json else {}
        diffs = [abs(int(round(y.loc[int(k), n])) - int(v)) for k, v in ref.items() if int(k) in y.index]
        checks[n] = max(diffs) if diffs else None
    return {"years": [int(v) for v in y.index], "rows": out}, checks


def rolling_sharpe(daily, names, step):
    import numpy as np
    r = daily[names].rolling(365, min_periods=365)
    s = (r.mean() / r.std(ddof=1)) * np.sqrt(365)
    s = s.iloc[364:]
    if step > 1:
        s = s.iloc[::-1].iloc[::step].iloc[::-1]     # keep the last day, step back from it
    return {"dates": [str(d.date()) for d in s.index], "rows": {n: r2(s[n]) for n in names}}


def placebo_row(name, r, end_year, v1: bool):
    p = r.get("placebo") if v1 else r.get("placebos")
    p = p or {}
    rd = ((r.get("fragility") or {}).get("random_days")) or {}
    rd = rd if isinstance(rd, dict) else {}
    real_total = (r["three_year"]["net_usd"] if v1 else r["total"])
    real_years = ({k: v["net_usd"] for k, v in r["years"].items()} if v1 else r["years"])
    return {"real_total": real_total,
            "one_day_late_total": p.get("shifted_one_day_total"),
            "permuted_mean_total": p.get("shuffled_mean_total"),
            "permuted_p95_total": p.get("shuffled_p95_total"),
            "permuted_max_total": p.get("shuffled_max_total"),
            "permuted_share_at_or_above_real": p.get("shuffled_share_at_or_above_real"),
            "n_permutations": p.get("n_shuffles"),
            "random_days_median_total": rd.get("random_median_total"),
            "random_days_share_beating_rule": rd.get("share_random_beats_rule"),
            "random_days_draws": rd.get("draws"),
            "years": {k: {"real": real_years.get(k),
                          "one_day_late": (p.get("shifted_one_day_years") or {}).get(k),
                          "random_days_median": ((rd.get("years") or {}).get(k) or {}).get("random_median")}
                      for k in sorted(years_ok(real_years, end_year))},
            "note": p.get("note")}


def seeds_file(robust: dict, end_year):
    out = {}
    for name, v in (robust.get("seeds") or {}).items():
        def slim(x):
            return {"three_year_net": x.get("three_year_net"), "sharpe": x.get("sharpe"),
                    "max_drawdown": x.get("max_drawdown"), "years": years_ok(x.get("years") or {}, end_year),
                    "pass_1mw": x.get("PASS_1MW"), "pass_sizing_view": x.get("pass_at_sizing_view"),
                    "sizing_scale_mw": x.get("sizing_scale"), "sizing_return_pct": x.get("sizing_return_pct")}
        per = {k: slim(x) for k, x in sorted((v.get("per_seed") or {}).items())}
        nets = sorted(x["three_year_net"] for x in per.values())
        out[name] = {"kind": v.get("kind"), "registered_three_year_net": v.get("lab_three_year_net"),
                     "registered_seeds": "0 to 2 (mean of three)",
                     "per_seed": per, "mean_of_20": slim(v.get("mean_20") or {}),
                     "sets_of_three": {k: slim(x) for k, x in (v.get("triples") or {}).items()},
                     "spread": {"min": nets[0], "median": nets[len(nets) // 2] if len(nets) % 2 else
                                round((nets[len(nets) // 2 - 1] + nets[len(nets) // 2]) / 2),
                                "max": nets[-1], "n": len(nets),
                                "negative": sum(1 for x in nets if x < 0)} if nets else None}
    return out


# ------------------------------------------------------------------ events and addenda
def addenda_from_git(repo: Path, ref: str):
    files = ["research/us/OBJECTIVES.md", "research/us_v2/OBJECTIVES.md"]
    fmt = "%h|%ad|%s"
    outp = subprocess.run(["git", "-C", str(repo), "log", ref, "--reverse", f"--format={fmt}",
                           "--date=format:%Y-%m-%d %H:%M", "--"] + files,
                          capture_output=True, text=True, check=True).stdout
    seen, out = set(), []
    for line in outp.splitlines():
        h, when, subj = line.split("|", 2)
        if h in seen:
            continue
        seen.add(h)
        label = subj.split(": ", 1)[1] if subj.startswith(("us study", "us page")) and ": " in subj else subj
        out.append({"date": when[:10], "time_dk": when[11:], "commit": h, "label": label[0].upper() + label[1:],
                    "file": "v2" if "v2" in subj.split(":")[0] else "v1"})
    return out


def event_check(daily_sets):
    """For each weather event, the worst day of always supply inside its window, from the data on hand."""
    out = {}
    for ev in WEATHER_EVENTS:
        for label, daily, col in daily_sets:
            if daily is None or col not in daily.columns:
                continue
            w = daily.loc[ev["start"]:ev["end"], col]
            if len(w):
                out.setdefault(ev["id"], {})[label] = {"worst_day": str(w.idxmin().date()),
                                                      "worst_day_usd_1mw": int(round(float(w.min()))),
                                                      "window_net_usd_1mw": int(round(float(w.sum())))}
    return out


# ------------------------------------------------------------------ main
def write(out_dir: Path, name: str, obj) -> int:
    p = out_dir / name
    s = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    p.write_text(s)
    return len(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path.home() / "nyiso-us"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", choices=["all", "data", "events"], default="all")
    ap.add_argument("--repo", default=None, help="tavle git repo, for the addenda commits")
    ap.add_argument("--ref", default="origin/master", help="git ref holding the published record")
    ap.add_argument("--include-heldout", action="store_true",
                    help="read past 2024-01-01 up to --end; only once the single held-out run has happened")
    ap.add_argument("--end", default=None, help="exclusive end date with --include-heldout (default 2026-10-01)")
    ap.add_argument("--v1-results", default=None)
    ap.add_argument("--v1-daily", default=None)
    ap.add_argument("--v2-results", default=None)
    ap.add_argument("--v2-daily", default=None)
    ap.add_argument("--robust", default=None)
    ap.add_argument("--top-extra", default="", help="comma-separated row names added to the top rows")
    ap.add_argument("--max-bytes", type=int, default=1_500_000)
    a = ap.parse_args()

    if a.include_heldout:
        a.end = a.end or "2026-10-01"
        holdout_note = f"INCLUDES the held-out period: data read up to {a.end} (exclusive)"
    else:
        if a.end and a.end > HOLDOUT_START:
            raise SystemExit("--end past 2024-01-01 needs --include-heldout")
        a.end = a.end or HOLDOUT_START
        holdout_note = (f"build years only: every daily table filtered at read to dates before {a.end}, "
                        "asserted; nothing on or after 2024-01-01 was read")
    end_year = int(a.end[:4]) + (0 if a.end[5:] == "01-01" else 1)
    root = Path(a.root).expanduser()
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    a.v1_results = a.v1_results or str(root / "side" / "lab_results.json")
    a.v1_daily = a.v1_daily or str(root / "side" / "lab_daily.parquet")
    a.robust = a.robust or str(root / "results" / "robust" / "robustness_summary.json")
    extra = [x for x in a.top_extra.split(",") if x]
    sizes, report = {}, {}
    v1d = v2d = None

    if a.only in ("all", "data"):
        res1, rows1, v1d = load_v1(a)
        res2, rows2, v2d, r2path, d2path = load_v2(a, root)
        log(f"v1 {v1d.shape} {v1d.index[0].date()}..{v1d.index[-1].date()}; "
            f"v2 {v2d.shape} {v2d.index[0].date()}..{v2d.index[-1].date()} from {d2path}")
        metas1 = [row_meta(n, rows1[n], "v1") for n in v1d.columns]
        metas2 = [row_meta(n, rows2[n], "v2") for n in v2d.columns]
        pos_dir = root / "results" / "v2" / "pos"
        unscored = sorted(p.stem for p in pos_dir.glob("*.parquet") if p.stem not in v2d.columns)
        top1, top2 = top_rows(metas1, extra), top_rows(metas2, extra)

        e1 = equity_file("v1", v1d, metas1, 2021, f"{a.v1_daily} (daily), {a.v1_results} (flags, sizing)",
                         holdout_note)
        e2 = equity_file("v2", v2d, metas2, 2013, f"{d2path} (daily), {r2path} (flags, sizing)", holdout_note)
        e2["meta"]["positions_not_yet_scored"] = unscored
        e2["meta"]["note"] = ("v2 rolling windows: train on the previous 3 years, test the next quarter; "
                              "scored 2013 onward")
        sizes["equity_v1.json"] = write(out_dir, "equity_v1.json", e1)
        sizes["equity_v2.json"] = write(out_dir, "equity_v2.json", e2)

        common = {"written": time.strftime("%Y-%m-%d %H:%M %Z"), "holdout": holdout_note,
                  "top_rows_v1": top1, "top_rows_v2": top2,
                  "top_rows_rule": "rows that pass the bar at the sizing view and are not fragile, plus always "
                                   "supply and the baseline as references"}
        dd = {"meta": {**common, "unit": "USD below the running peak of cumulative net, 1 MW",
                       "resolution": "daily, no downsampling"},
              "v1": {"dates": e1["dates"], "rows": drawdown(v1d, top1)},
              "v2": {"dates": e2["dates"], "rows": drawdown(v2d, top2)}}
        sizes["drawdown.json"] = write(out_dir, "drawdown.json", dd)

        mo = {"meta": {**common, "unit": "USD net per calendar month, 1 MW", "rows": "every row"},
              "v1": monthly(v1d), "v2": monthly(v2d)}
        sizes["monthly.json"] = write(out_dir, "monthly.json", mo)

        rs = {"meta": {**common, "definition": "mean / std (ddof 1) of daily net over the trailing 365 days, "
                                               "times sqrt(365), as the lab's Sharpe; first value after a full "
                                               "365 days",
                       "resolution": "v1 daily; v2 weekly (every 7th day counted back from the last day)"},
              "v1": rolling_sharpe(v1d, top1, 1), "v2": rolling_sharpe(v2d, top2, 7)}
        sizes["rolling_sharpe.json"] = write(out_dir, "rolling_sharpe.json", rs)

        y1, c1 = years(v1d, rows1, end_year, lambda r: {k: v["net_usd"] for k, v in r["years"].items()})
        y2, c2 = years(v2d, rows2, end_year, lambda r: r["years"])
        yr = {"meta": {**common, "unit": "USD net per calendar year, 1 MW", "rows": "every row",
                       "check": "max absolute difference against the results files' per-year net, USD",
                       "check_v1": c1, "check_v2": c2}, "v1": y1, "v2": y2}
        sizes["years.json"] = write(out_dir, "years.json", yr)
        report["year_check_max_diff_usd"] = max([v for v in list(c1.values()) + list(c2.values()) if v is not None]
                                                or [None])

        pl = {"meta": {**common, "definitions": {
                  "one_day_late": "the same rule fed its inputs shifted one day late",
                  "permuted": "the rule's inputs permuted across days within each year, 20 permutations",
                  "random_days": "the same number of trading days drawn at random inside the same months, "
                                 "1000 draws (median total shown)"},
                  "source": "results files as scored; nothing recomputed here"},
              "v1": {n: {**placebo_row(n, rows1[n], end_year, True), "top": n in top1} for n in v1d.columns},
              "v2": {n: {**placebo_row(n, rows2[n], end_year, False), "top": n in top2} for n in v2d.columns}}
        sizes["placebo.json"] = write(out_dir, "placebo.json", pl)

        robust = json.loads(Path(a.robust).read_text())
        sd = {"meta": {"written": time.strftime("%Y-%m-%d %H:%M %Z"), "source": a.robust,
                       "robust_written": robust.get("written"), "holdout": holdout_note,
                       "what": "each deep row retrained with 20 random starts (seeds 0 to 19), every seed "
                               "scored on its own over 2021 to 2023, never selected; beside it the mean of all "
                               "20 and the six disjoint sets of three seeds (the registered ensemble size)",
                       "rows_covered": sorted((robust.get("seeds") or {}).keys())},
              "rows": seeds_file(robust, end_year)}
        sizes["seeds.json"] = write(out_dir, "seeds.json", sd)
        report.update({"v1_rows": len(metas1), "v2_rows": len(metas2), "top_v1": top1, "top_v2": top2,
                       "v2_unscored_positions": unscored})

    if a.only in ("all", "events"):
        ev_path = out_dir / "events.json"
        old = json.loads(ev_path.read_text()) if ev_path.exists() else {}
        repo = Path(a.repo).expanduser() if a.repo else None
        if repo and (repo / ".git").exists():
            addenda = addenda_from_git(repo, a.ref)
            how = f"git log {a.ref} of research/us*/OBJECTIVES.md in {repo}"
        else:
            addenda = old.get("addenda", [])
            how = (old.get("meta") or {}).get("addenda_source", "none: no --repo given and no earlier events.json")
        check = event_check([("v1_always_supply", v1d, "always_supply"), ("v2_always_supply", v2d,
                                                                         "V1_always_supply")])
        if not check:
            check = (old.get("meta") or {}).get("data_check", {})
        ev = {"meta": {"written": time.strftime("%Y-%m-%d %H:%M %Z"), "holdout": holdout_note,
                       "weather_dates": "event windows from public record (New York, Eastern time)",
                       "data_check": check,
                       "data_check_note": "worst day of always supply (1 MW) inside each window, from the daily "
                                          "tables; empty when the event is outside the scored years",
                       "addenda_source": how,
                       "addenda_note": "commit dates and times are Danish time; the commit time is the record"},
              "weather": WEATHER_EVENTS, "addenda": addenda}
        sizes["events.json"] = write(out_dir, "events.json", ev)

    big = {k: v for k, v in sizes.items() if v > a.max_bytes}
    for k, v in sizes.items():
        log(f"{k}: {v / 1000:.0f} kB")
    print(json.dumps({"sizes_bytes": sizes, **report}, default=str))
    if big:
        raise SystemExit(f"over {a.max_bytes} bytes: {big}")


if __name__ == "__main__":
    main()
