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
    heldout.json        (--only heldout) the held-out results the page leads with: daily net per row, the event split,
                        excess over the no-model rules with intervals, the registered verdicts, the zone-pair story,
                        the correlations and the two books, the S&P 500 and deposit baselines

HOLDOUT. By default every daily table is filtered AT READ (pyarrow filter) to dates before 2024-01-01 and the
result is asserted; per-year fields from the summary files are dropped for 2024 onward. The single held-out run
finished on 11 Oct 2026. --include-heldout reads up to --end; point --v1-daily/--v2-daily etc. at the run's files.

Fixed 11 Oct 2026 (page code, not frozen model code): the --include-heldout data step crashed with KeyError
'V15a_V4_limit', because the run's daily table keeps the three version 2 rows that read only the 2019 reforecast and
traded nothing, while its results file leaves them out. Daily columns without a results row are now reported as not
run instead of crashing, and V12_alloc, dropped from the held-out run on 9 Oct but run by mistake, is reported as not
run in held-out mode. The version 2 summary fields the page reads (total_1mw, total_wo_best5, sharpe_results,
years_positive) are now written by this script as well.

The held-out step (--only heldout) reads the run's outputs and the read-only audit of them from --heldout-dir:
    from_gene/lab_daily_heldout.parquet, from_gene/v2_daily_heldout.parquet   daily net per row, 1 MW, the window
    audit_scratch/analysis.json                     event split, excess with intervals, placebos, per row
    ho_v1/side/lab_results_heldout.json, ho_v2/results/v2/v2_results_heldout.json   the scorers' held-out results
    ho_reh/heldout_*.json                           the registered verdict run (ideas A to D)
    from_gene/b_books.csv, from_gene/b_corr.txt, from_gene/a_2025.txt   the books, correlations and the 2025 analysis
and from the repo: research/us*/HELDOUT-LIST.md (the build-year numbers each row carried into the run) and
research/us/chartdata/sources/ (FRED SP500 and Statistics Denmark DNRUIPI downloads for the baselines).

    laptop: python3 build_chartdata.py --only heldout --repo ~/projects/tavle --out ~/projects/tavle/docs/us/data

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
import re
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
# Dropped from the held-out run by the addendum of 9 Oct 2026 (it could not reproduce its own 2023 build); the run
# scored it anyway, so in held-out mode it is reported as not run.
NOT_RUN_HELDOUT = {"V12_alloc": "dropped on 9 Oct 2026 (6199e84): its build-year run could not be reproduced; "
                                "the held-out run scored it by mistake and it is excluded"}

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
    if version == "v2":     # the fields the page's version 2 ranking reads
        yrs = {k: v for k, v in (r.get("years") or {}).items()}
        m.update({"years_positive": sum(1 for v in yrs.values() if v > 0),
                  "years_traded": sum(1 for v in yrs.values() if v != 0),
                  "total_1mw": r.get("total"), "total_wo_best5": r.get("total_without_best_5_days"),
                  "sharpe_results": r.get("sharpe"), "pass_1mw_bar": bool(r.get("PASS"))})
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


# ------------------------------------------------------------------ the held-out step
HO_EVENTS = [
    {"id": "storm_2026", "start": "2026-01-24", "end": "2026-02-02",
     "label": "Winter storm, 24 January to 2 February 2026"},
    {"id": "heat_2026", "start": "2026-06-28", "end": "2026-07-05",
     "label": "Heat wave, 28 June to 5 July 2026"},
]
# The audit's verdict words, grouped into the families the page shows (heldout_results/AUDIT.md, section 6)
FAMILY_OF_VERDICT = {"event luck": "sell", "pair stance": "pairs", "mostly event luck": "mix", "comparison": "plain",
                     "broken": "not_run"}
PLAIN_ROWS = {"v1:always_supply": "always supply", "v1:baseline_usual_side": "habit rule",
              "v2:C_static_pairs": "zone pairs, no model"}
MIX_ROWS = {"v1:idea13_portfolio", "v1:idea14_learned_allocator_candidates"}


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def md_table_rows(path: Path):
    """Rows of every markdown table in a file, as lists of stripped cells (header and rule lines skipped)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def pre_values(repo: Path) -> dict:
    """Build-year numbers each row carried into the held-out run, from the two HELDOUT-LIST.md files."""
    num = lambda x: float(x.replace(",", "").replace("%", "").replace("−", "-"))
    out = {}
    for ver, f in (("v1", "research/us/HELDOUT-LIST.md"), ("v2", "research/us_v2/HELDOUT-LIST.md")):
        section = None
        for line in (repo / f).read_text().splitlines():
            if line.startswith("## "):
                section = "candidate" if line.startswith("## Candidates") else "comparison"
                continue
            if not line.startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c[0] in ("Strategy", "Row"):
                continue
            name = c[0].replace(" (registered comparison)", "").replace(" (second build)", " [lead]")
            if ver == "v1":    # Strategy | 2021 | 2022 | 2023 | net | Sharpe | MW | return | passes | fragile
                yrs = [num(x) for x in c[1:4]]
                out[f"v1:{name}"] = {"sharpe": num(c[5]), "scale_mw": num(c[6]), "ret_pct": num(c[7]),
                                     "years_pos": sum(1 for y in yrs if y > 0), "years_n": 3, "period": "2021 to 2023",
                                     "list": section}
            else:              # Row | net | avg | 2023 | positive years | Sharpe | MW | return | passes | fragile
                out[f"v2:{name}"] = {"sharpe": num(c[5]), "scale_mw": num(c[6]), "ret_pct": num(c[7]),
                                     "years_pos": int(num(c[4])), "years_n": 11, "period": "2013 to 2023",
                                     "list": section}
    return out


def parse_matrix_blocks(text: str, title: str) -> list[dict]:
    """Each whitespace table under a line equal to `title` in b_corr.txt, in order: {row: {col: value}}."""
    lines, out = text.splitlines(), []
    for i, l in enumerate(lines):
        if l.strip() != title:
            continue
        cols = lines[i + 1].split()
        m = {}
        for l2 in lines[i + 2:]:
            parts = l2.split()
            if len(parts) != len(cols) + 1:
                break
            m[parts[0]] = {c: float(v) for c, v in zip(cols, parts[1:])}
        out.append(m)
    return out


def parse_books(path: Path) -> list[dict]:
    import csv
    rows = []
    for r in csv.DictReader(open(path)):
        f = lambda k: float(r[k])
        rows.append({"book": "mix" if r["book"].startswith("mix") else "pairs", "limit_usd": int(f("limit")),
                     "size": "cautious" if r["size"].startswith("cautious") else "full",
                     "pairs_mw": f("pairs MW"), "wx_mw": f("V15c MW"),
                     "build": {"ret_pct": f("build ret/yr %"), "sharpe": f("build Sharpe"),
                               "worst_dd": int(f("build worst DD")), "worst_day": int(f("build worst day")),
                               "worst_day_date": r["build worst day date"]},
                     "ho": {"ret_pct": f("held-out ret/yr %"), "sharpe": f("held-out Sharpe"),
                            "worst_dd": int(f("held-out worst DD")), "worst_day": int(f("held-out worst day")),
                            "worst_day_date": r["held-out worst day date"]},
                     "ho_ex": {"ret_pct": f("held-out w/o events ret/yr %"), "sharpe": f("held-out w/o events Sharpe"),
                               "worst_dd": int(f("held-out w/o events worst DD"))},
                     "limit_held": r["held-out DD within limit"] == "True",
                     "years_ret_pct": {"2024": f("2024 ret %"), "2025": f("2025 ret %"), "2026": f("2026 ret %")}})
    return rows


def baselines_heldout(src: Path) -> dict:
    """S&P 500 (FRED SP500, price only) and Danish household deposits (DNRUIPI) over the held-out window."""
    import csv
    import statistics as st
    px = [(d, float(v)) for d, v in csv.reader(open(src / "sp500_fred.csv")) if d[:2] == "20" and v not in ("", ".")]
    start = [v for d, v in px if d < "2024-01-01"][-1]                 # close before the window
    win = [(d, v) for d, v in px if "2024-01-01" <= d <= "2026-09-30"]
    end_d, end = win[-1]
    years = (dt.date(2026, 9, 30) - dt.date(2023, 12, 31)).days / 365.25
    ann = (end / start) ** (1 / years) - 1
    rets = [b / a - 1 for (_, a), (_, b) in zip([("", start)] + win[:-1], win)]
    bill = {d: float(v) for d, v in csv.reader(open(src / "dtb3_fred.csv")) if d[:2] == "20" and v not in ("", ".")}
    last_bill, ex = None, []
    for (d, _), r in zip(win, rets):          # excess over the 3-month Treasury bill, the day's rate over 252 days
        last_bill = bill.get(d, last_bill)
        ex.append(r - last_bill / 100 / 252)
    sharpe = st.mean(ex) / st.stdev(ex) * math.sqrt(252)
    peak, mdd = start, 0.0
    for _, v in win:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    dep = {}
    for row in csv.reader(open(src / "deposit_dnruipi.csv", encoding="utf-8-sig"), delimiter=";"):
        if len(row) == 7 and row[5][:4].isdigit():
            dep[row[5]] = float(row[6].replace(",", "."))
    months = [m for m in sorted(dep) if "2024M01" <= m <= "2026M09"]
    dep_avg = sum(dep[m] for m in months) / len(months)
    return {"window": "2024-01-01 to 2026-09-30", "basis_usd": BANK,
            "sp500": {"source": "FRED series SP500 (S&P Dow Jones Indices), daily close, price only",
                      "url": "https://fred.stlouisfed.org/series/SP500", "file": "research/us/chartdata/sources/sp500_fred.csv",
                      "first_close": start, "last_close": end, "last_close_date": end_d,
                      "return_a_year_pct_price_only": round(100 * ann, 1),
                      "dividends_note": "dividends add about 1.3 percent a year (research/us/results/cost_capital_audit.md)",
                      "usd_a_year_on_basis": int(round(BANK * ann, -2)),
                      "sharpe_over_bills_price_only": round(sharpe, 2),
                      "sharpe_note": "daily price returns minus the 3-month Treasury bill rate (FRED DTB3, "
                                     "research/us/chartdata/sources/dtb3_fred.csv), times the square root of 252",
                      "worst_drawdown_pct": round(100 * mdd, 1)},
            "deposit": {"source": "Statistics Denmark table DNRUIPI: households' bank deposits, all currencies, all "
                                  "maturities, effective rate, monthly", "url": "https://www.statbank.dk/DNRUIPI",
                        "file": "research/us/chartdata/sources/deposit_dnruipi.csv",
                        "months": f"{months[0]} to {months[-1]}", "n_months": len(months),
                        "average_pct": round(dep_avg, 2), "usd_a_year_on_basis": int(round(BANK * dep_avg / 100, -2)),
                        "note": "a Danish krone rate applied to a dollar sum, as a stand-in for idle cash at a bank"}}


RUN_NAMES = {"v1_daily": "run/ho_v1/side/lab_daily_heldout.parquet",
             "v2_daily": "run/ho_v2/results/v2/v2_daily_heldout.parquet",
             "audit": "audit of the run, analysis.json", "v1_results": "run/ho_v1/side/lab_results_heldout.json",
             "v2_results": "run/ho_v2/results/v2/v2_results_heldout.json",
             "registered": "run/ho_reh/heldout_202401_202609_f0636cc3cc36_20261010T1138.json",
             "books": "analysis of 11 Oct 2026, b_books.csv", "corr": "analysis of 11 Oct 2026, b_corr.txt",
             "pairs2025": "analysis of 11 Oct 2026, a_2025.txt"}
PUBLISHED = {"audit": "docs/us/results/audit_analysis.json", "registered": "docs/us/results/registered_verdicts_heldout.json",
             "books": "docs/us/results/b_books.csv", "corr": "docs/us/results/b_corr.txt",
             "pairs2025": "docs/us/results/a_2025.txt", "v1_results": "docs/us/results/lab_results_heldout.md (table)",
             "v2_results": "docs/us/results/v2_results_heldout.md (table)"}


def build_heldout(a, repo: Path, out_dir: Path) -> dict:
    import pandas as pd
    hd = Path(a.heldout_dir).expanduser()
    f = {"v1_daily": hd / "from_gene/lab_daily_heldout.parquet", "v2_daily": hd / "from_gene/v2_daily_heldout.parquet",
         "audit": hd / "audit_scratch/analysis.json", "v1_results": hd / "ho_v1/side/lab_results_heldout.json",
         "v2_results": hd / "ho_v2/results/v2/v2_results_heldout.json",
         "registered": sorted((hd / "ho_reh").glob("heldout_*.json"))[-1],
         "books": hd / "from_gene/b_books.csv", "corr": hd / "from_gene/b_corr.txt", "pairs2025": hd / "from_gene/a_2025.txt"}
    aud = json.loads(f["audit"].read_text())
    r1 = json.loads(f["v1_results"].read_text())["strategies"]
    r2 = {r["name"]: r for r in json.loads(f["v2_results"].read_text())["rows"]}
    reh = json.loads(f["registered"].read_text())
    pre = pre_values(repo)

    # daily net per row, 1 MW, the window only
    d1 = pd.read_parquet(f["v1_daily"]); d2 = pd.read_parquet(f["v2_daily"])
    for d in (d1, d2):
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        assert str(d.index.min().date()) == "2024-01-01" and str(d.index.max().date()) == "2026-09-30", d.index
    assert (d1.index == d2.index).all()
    dates = [str(x.date()) for x in d1.index]

    # verdict words from the audit report's verdict table
    verdict = {}
    for c in md_table_rows(hd / "AUDIT.md"):
        if len(c) == 6 and ":" in c[0] and c[0] != "row":
            verdict[c[0]] = c[5]
    ev_mask = pd.Series(False, index=d1.index)
    for e in HO_EVENTS:
        ev_mask |= (d1.index >= e["start"]) & (d1.index <= e["end"])
    n_ev = int(ev_mask.sum())

    rows = []
    for key, x in aud.items():
        ver, name = key.split(":", 1)
        v = verdict.get(key, "")
        fam = "plain" if key in PLAIN_ROWS else "mix" if key in MIX_ROWS else \
            next((fam for w, fam in FAMILY_OF_VERDICT.items() if v.startswith(w)), None)
        assert fam, (key, v)
        daily = (d1 if ver == "v1" else d2)[name]
        assert abs(daily.sum() - x["total_1MW"]) < 1, (key, daily.sum(), x["total_1MW"])     # same numbers as the audit
        res = r1.get(name) if ver == "v1" else r2.get(name)
        ho_sharpe = (res["three_year"]["sharpe"] if ver == "v1" else res["sharpe"]) if res else x.get("sharpe")
        yrs = x["years"]
        passes = (x["ret_build_scale"] >= 10 and sum(1 for y in yrs.values() if y["net"] > 0) >= 2
                  and (x.get("sharpe") or 0) > 0.42 and (x.get("stress") or 0) > 0)
        row = {"key": key, "name": name, "version": ver, "family": fam, "verdict": v,
               "role": "not_run" if fam == "not_run" else "plain" if fam == "plain" else "candidate",
               "scale_mw": x["scale"], "pre": pre.get(key),
               "ho": {"net_1mw": x["total_1MW"], "storm": x["storm"], "heat": x["heat"],
                      "event_share_pct": x["event_share_pct"], "net_1mw_ex": x["total_ex_events"],
                      "ret_pct": x["ret_build_scale"], "ret_pct_ex": x["ret_build_scale_ex_events"],
                      "sharpe": ho_sharpe, "sharpe_ex": x.get("sharpe_ex_events"),
                      "years": {k: y["net"] for k, y in yrs.items()},
                      "years_ret_pct": {k: y["ret"] for k, y in yrs.items()},
                      "stress_050": x.get("stress"), "pass_sizing_view": bool(passes)},
               "excess_vs_supply": x.get("excess_vs_always_supply"),
               "excess_vs_static_pairs": x.get("excess_vs_C_static_pairs"),
               "excess_vs_habit": x.get("excess_vs_baseline_usual_side"),
               "hit_rate": {k: (x.get("hits") or {}).get(k) for k in ("event_days", "non_event", "build_2021_2023")},
               "one_day_late": (x.get("scorer_placebo") or {}).get("one_day_late")}
        rows.append(row)
    # The audit covers the candidates that cleared the bar on the window; the four that did not are computed here
    # from the same daily tables, the same way (return a year at build scale over 1,004 days, 986 without events).
    FAILED = {"v1:C_gbm": ("pairs", "negative at the 0.50 USD per MWh cost stress"),
              "v2:V14a_window_ens": ("pairs", "under 10 percent a year"),
              "v2:V14b_quantile_size": ("pairs", "under 10 percent a year"),
              "v2:V6_LI": ("sell", "under 10 percent a year")}
    listed = {k for k, v in pre.items() if v["list"] == "candidate"} - {"v2:V12_alloc"}
    assert listed - set(aud) == set(FAILED), sorted(listed - set(aud))
    ny = len(d1) / 365.25
    for key, (fam, why) in FAILED.items():
        ver, name = key.split(":", 1)
        daily = (d1 if ver == "v1" else d2)[name]
        res = r1[name] if ver == "v1" else r2[name]
        sc = pre[key]["scale_mw"]
        tot, ex = float(daily.sum()), float(daily[~ev_mask].sum())
        stress = (res.get("stress_0.50_total") if ver == "v2" else None)
        rows.append({"key": key, "name": name, "version": ver, "family": fam, "verdict": "fails the bar: " + why,
                     "role": "candidate", "scale_mw": sc, "pre": pre[key],
                     "ho": {"net_1mw": int(round(tot)), "net_1mw_ex": int(round(ex)),
                            "storm": int(round(float(daily[(d1.index >= "2026-01-24") & (d1.index <= "2026-02-02")].sum()))),
                            "heat": int(round(float(daily[(d1.index >= "2026-06-28") & (d1.index <= "2026-07-05")].sum()))),
                            "event_share_pct": round(100 * (tot - ex) / tot, 1) if tot else None,
                            "ret_pct": round(100 * tot * sc / BANK / ny, 1),
                            "ret_pct_ex": round(100 * ex * sc / BANK / ((len(d1) - n_ev) / 365.25), 1),
                            "sharpe": res["three_year"]["sharpe"] if ver == "v1" else res["sharpe"],
                            "years": {k: int(round(float(v))) for k, v in daily.groupby(daily.index.year).sum().items()},
                            "stress_050": stress, "pass_sizing_view": False},
                     "excess_vs_supply": None, "excess_vs_static_pairs": None, "excess_vs_habit": None,
                     "hit_rate": None, "one_day_late": None})
    cands = [r for r in rows if r["role"] == "candidate"]
    n_pass = sum(1 for r in cands if r["ho"]["pass_sizing_view"])
    assert (len(cands), n_pass) == (32, 28), (len(cands), n_pass)    # AUDIT.md: 28 of the 32 tested candidates

    # every scored row's held-out total at 1 MW, for the appendix tables
    totals = {"v1": {n: int(round(float(d1[n].sum()))) for n in d1.columns},
              "v2": {n: int(round(float(d2[n].sum()))) for n in d2.columns}}
    hero = sorted({r["name"] for r in rows if r["family"] in ("sell", "pairs", "plain")} - {"V3_tail_gbm"})
    daily = {n: ints((d1 if n in d1.columns and n not in d2.columns else d2)[n]) for n in hero}

    # correlations of daily net, 1 MW, three periods (b_corr.txt)
    txt = f["corr"].read_text()
    pear = parse_matrix_blocks(txt, "Pearson (pairwise complete)")
    assert len(pear) == 3
    corr = {"names": {"pairs": "zone pairs, no model", "V15c": "weather and border model (V15c)",
                      "V16": "weather model (V16)", "always_supply": "always supply", "idea13": "portfolio (idea 13)",
                      "V1_C_deep": "zone pairs, deep model (V1)"},
            "periods": {"build": "2013 to 2023 (version 1 rows 2021 to 2023)", "heldout": "held-out window",
                        "heldout_ex": "held-out window without the 18 event days"},
            "pearson": {"build": pear[0], "heldout": pear[1], "heldout_ex": pear[2]}}
    ysec = txt.split("yearly sums")[1].split("###")[0].splitlines()
    cols = ysec[1].split()
    yearly = {int(l.split()[0]): dict(zip(cols, [float(v) for v in l.split()[1:]])) for l in ysec[2:] if l.strip()}

    # the 2025 pair story (a_2025.txt)
    t25 = f["pairs2025"].read_text()
    spread = re.search(r"by year: (\[.*\])", t25).group(1)
    spread = [tuple(float(v) for v in m) for m in re.findall(r"\((\d+), (-?[\d.]+), (-?[\d.]+)\)", spread)]
    m25 = re.search(r"2025 CAPITL minus NYC.*?mean DA cong CAPITL ([\d.]+) NYC ([\d.]+); RT cong CAPITL ([\d.]+) NYC ([\d.]+)", t25)
    sp = t25.split("## C_static_pairs")[1]
    comp = re.search(r"\n2025\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", sp).groups()
    avg = re.search(r"2013-23 avg\s+(-?\d+\.?\d*)", sp).group(1)
    blk = re.search(r"\n16-21\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", sp).groups()
    mon = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"\n\s*(\d{1,2})\s+(-?\d+\.?\d*)\s+-?\d", sp.split("by month")[1].split("by hour")[0])}
    zone = {}
    for l in sp.split("by zone")[1].split("by month")[0].splitlines()[2:]:
        p = l.split()
        if len(p) == 10:
            zone[p[0]] = {"meanMW_2025": float(p[9]), "net_2025": float(p[7]), "meanMW_avg": float(p[3])}
    years_p = sorted(yearly)
    pairs = {"years": years_p, "net_1mw": [int(yearly[y]["pairs"]) for y in years_p],
             "nyc_minus_capitl_gap": [round(-c, 2) for y, c, _ in spread],
             "gap_years": [int(y) for y, _, _ in spread],
             "y2025": {"net": float(comp[0]), "energy": float(comp[1]), "loss": float(comp[2]),
                       "congestion": float(comp[3]), "fees": float(comp[4]), "avg_2013_2023_net": float(avg),
                       "da_cong_capitl": float(m25.group(1)), "da_cong_nyc": float(m25.group(2)),
                       "rt_cong_capitl": float(m25.group(3)), "rt_cong_nyc": float(m25.group(4)),
                       "hours_16_21": float(blk[0]), "hours_16_21_avg": float(blk[2]),
                       "jan_feb": mon[1] + mon[2], "july": mon[7], "zones": zone},
             "sharpe_pre": pre["v2:C_static_pairs"]["sharpe"],
             "sharpe_ho": r2["C_static_pairs"]["sharpe"]}
    assert all(v > 0 for v in pairs["net_1mw"]) and pairs["years"] == list(range(2013, 2027))

    ideas = {k: {"verdict": v["verdict"], "mean_daily_diff_vs_habit": round(v["mean_daily_diff"], 1),
                 "interval": [round(z, 1) for z in v["interval"]], "interval_level": round(v["interval_level"], 4),
                 "holm_rank": (v.get("holm") or {}).get("rank"),
                 "interval_95": [round(z, 1) for z in v["interval_95"]]} for k, v in reh["ideas"].items()}
    vsf = reh["value_of_the_spike_filter"]
    out = {"meta": {"written": time.strftime("%Y-%m-%d"),
                    "window": "2024-01-01 to 2026-09-30, 1,004 days, scored once from the code frozen at f0636cc",
                    "unit": "USD net after costs; 1 MW per position unless a field says build scale (the MW each row "
                            "carried from its build years, the size at which its build-year worst drawdown is 100,000)",
                    "event_days": n_ev,
                    "sources": {k: {"run_file": RUN_NAMES[k], "sha256_16": sha(v),
                                    "published_copy": PUBLISHED.get(k)} for k, v in f.items()},
                    "not_run": NOT_RUN_HELDOUT,
                    "hero_rows_note": "V3_tail_gbm is left out of the 1 MW chart: it averages about 2 MW a position"},
           "events": HO_EVENTS, "rows": rows, "totals_1mw": totals,
           "counts": {"tested_candidates": len(cands), "pass_sizing_view": n_pass,
                      "sell": sum(1 for r in cands if r["family"] == "sell"),
                      "pairs": sum(1 for r in cands if r["family"] == "pairs")},
           "registered": {"ideas": ideas, "spike_filter_minus_always_supply":
                          {"mean_daily": round(vsf["mean_daily_diff"], 2), "interval_95": [round(z, 2) for z in vsf["interval_95"]]}},
           "daily": {"dates": dates, "rows": daily},
           "books": parse_books(f["books"]), "corr": corr, "pairs": pairs,
           "baselines": baselines_heldout(repo / "research/us/chartdata/sources")}
    return {"heldout.json": write(out_dir, "heldout.json", out)}


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
    ap.add_argument("--only", choices=["all", "data", "events", "heldout"], default="all")
    ap.add_argument("--heldout-dir", default=str(Path.home() / "nyiso-us" / "heldout_results"),
                    help="the held-out run's outputs and their audit (see the docstring)")
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
    ap.add_argument("--max-bytes", type=int, default=3_000_000)
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

    if a.only == "heldout":
        sizes.update(build_heldout(a, Path(a.repo).expanduser(), out_dir))
        for k, v in sizes.items():
            log(f"{k}: {v / 1000:.0f} kB")
        return

    if a.only in ("all", "data"):
        res1, rows1, v1d = load_v1(a)
        res2, rows2, v2d, r2path, d2path = load_v2(a, root)
        log(f"v1 {v1d.shape} {v1d.index[0].date()}..{v1d.index[-1].date()}; "
            f"v2 {v2d.shape} {v2d.index[0].date()}..{v2d.index[-1].date()} from {d2path}")
        # A daily column with no results row (the three 2019-only weather rows, which traded nothing) is reported
        # as not run; in held-out mode so is a row the objectives dropped before the run.
        drop1 = {n: "no results row" for n in v1d.columns if n not in rows1}
        drop2 = {n: "no results row: it read only the 2019 reforecast and traded nothing" for n in v2d.columns
                 if n not in rows2}
        if a.include_heldout:
            drop2.update({n: why for n, why in NOT_RUN_HELDOUT.items() if n in v2d.columns})
        v1d, v2d = v1d.drop(columns=list(drop1)), v2d.drop(columns=list(drop2))
        report["not_run"] = {**drop1, **drop2}
        metas1 = [row_meta(n, rows1[n], "v1") for n in v1d.columns]
        metas2 = [row_meta(n, rows2[n], "v2") for n in v2d.columns]
        pos_dir = root / "results" / "v2" / "pos"
        unscored = sorted(p.stem for p in pos_dir.glob("*.parquet") if p.stem not in v2d.columns)
        top1, top2 = top_rows(metas1, extra), top_rows(metas2, extra)

        e1 = equity_file("v1", v1d, metas1, 2021, f"{a.v1_daily} (daily), {a.v1_results} (flags, sizing)",
                         holdout_note)
        e2 = equity_file("v2", v2d, metas2, 2013, f"{d2path} (daily), {r2path} (flags, sizing)", holdout_note)
        e2["meta"]["positions_not_yet_scored"] = unscored
        e2["meta"]["not_run"] = {k: v for k, v in report["not_run"].items() if k not in drop1}
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
