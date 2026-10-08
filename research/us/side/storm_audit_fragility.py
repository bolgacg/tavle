"""Audit part 2: independent re-implementation of the two storm strategies, selection and fragility checks.
Build years only (holdout enforced at load in storm_audit_lib). 1 MW per zone-hour, 11 zones, decision 05:00 on D.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from storm_audit_lib import (BID_DAYS, END, OUT, SCRIPT_CUTS, WIDE_CUTS, YEARS, assert_holdout, daily_book,
                             inputs, legs, load_lf, load_prices, load_wx, run_protocol, storm_score, strategy,
                             year_stats)

CLAIM = {"filter": {2021: 41620, 2022: 49655, 2023: 15866}, "flip": {2021: 78415, 2022: 290469, 2023: -48392},
         "supply": {2021: -5015, 2022: -198269, 2023: 68154}}
rng = np.random.default_rng(20261006)

px, lf, wx = load_prices(), load_lf(), load_wx()
assert_holdout((px, "delivery_hour"), (lf, "target_hour"), (wx, "target_hour"))
book = daily_book(px)
assert book.index.max() < pd.Timestamp("2024-01-01")
res = {"rows_without_rt_price_2020_2023": int(px["gap"].isna().sum()), "rows_2020_2023": len(px)}

inp_w = inputs(BID_DAYS, px, lf, wx, deadline="wallclock")
inp_s = inputs(BID_DAYS, px, lf, wx, deadline="script")
S_W, R_W = storm_score(inp_w)
S_S, _ = storm_score(inp_s)
yr = lambda s, y: s[s.index.year == y]


def nets(proto):
    return {y: proto[y]["net"] for y in YEARS}


# ---------------------------------------------------------------- 1. reproduction
res["always_supply"] = {y: year_stats(yr(book["sup"], y)) for y in YEARS}
res["repro_script_deadline"] = {k: run_protocol(book, S_S, k) for k in ("filter", "flip")}
res["audit_wallclock"] = {k: run_protocol(book, S_W, k) for k in ("filter", "flip")}
res["claim"] = CLAIM
res["variant_gfs_three_day_rule"] = {k: nets(run_protocol(book, storm_score(inputs(BID_DAYS, px, lf, wx, gfs="rule"))[0], k))
                                     for k in ("filter", "flip")}
res["variant_lf_newest_per_hour"] = {k: nets(run_protocol(book, storm_score(inputs(BID_DAYS, px, lf, wx, lfsel="per_hour"))[0], k))
                                     for k in ("filter", "flip")}

# ---------------------------------------------------------------- 2. every cut, every year (fixed cut, no selection)
grid = {}
for k in ("filter", "flip"):
    grid[k] = {}
    for y in [2020] + YEARS:
        grid[k][y] = {str(c): round(float(yr(strategy(book, S_W, c, k), y).sum())) for c in WIDE_CUTS}
        grid[k][y]["days_on_at_0.80"] = int((yr(S_W.reindex(book.index), y) > 0.8).sum())
res["cut_grid"] = grid
res["protocol_wide_grid"] = {k: {**nets(p := run_protocol(book, S_W, k, cuts=WIDE_CUTS)), "cuts": [p[y]["cut"] for y in YEARS]}
                             for k in ("filter", "flip")}

# ---------------------------------------------------------------- 3. components alone and removed; window
ALL = ("load", "cold", "heat", "rt")
comp = {}
for c in ALL:
    for label, cs in ((f"{c} alone", (c,)), (f"without {c}", tuple(x for x in ALL if x != c))):
        s, _ = storm_score(inp_w, comps=cs)
        comp[label] = {k: {**nets(p := run_protocol(book, s, k)), "sum": p["sum"], "cuts": [p[y]["cut"] for y in YEARS]}
                       for k in ("filter", "flip")}
for wdw in (180, 365):
    s, _ = storm_score(inp_w, window=wdw)
    comp[f"window {wdw}"] = {k: {**nets(p := run_protocol(book, s, k)), "sum": p["sum"], "cuts": [p[y]["cut"] for y in YEARS]}
                             for k in ("filter", "flip")}
res["components_and_window"] = comp

# ---------------------------------------------------------------- 4. costs
costs = {}
for label, c in (("base", None), ("0.50", 0.50), ("1.00", 1.00)):
    sup, _ = legs(book, c)
    costs[label] = {"supply": {y: round(float(yr(sup, y).sum())) for y in YEARS},
                    **{k: nets(run_protocol(book, S_W, k, cost=c)) for k in ("filter", "flip")}}
res["costs"] = costs

# ---------------------------------------------------------------- 5. what the storm days are
on_info = {}
for k in ("filter", "flip"):
    p = res["audit_wallclock"][k]
    on_info[k] = {}
    for y in YEARS:
        c = p[y]["cut"]
        s = yr(S_W.reindex(book.index), y)
        on = s > c
        sup, load = yr(book["sup"], y), yr(book["load"], y)
        by_month = on.groupby(on.index.month).mean().round(2)
        r = R_W.copy()
        r.index = r.index + pd.Timedelta(days=1)
        r = r.reindex(on.index)[on]
        on_info[k][y] = {"cut": c, "days_on": int(on.sum()),
                         "supply_pnl_on_days": round(float(sup[on].sum())),
                         "supply_pnl_off_days": round(float(sup[~on].sum())),
                         "load_pnl_on_days": round(float(load[on].sum())),
                         "share_on_by_month": {int(m): float(v) for m, v in by_month.items()},
                         "on_days_where_component_above_cut": {cc: int((r[cc] > c).sum()) for cc in ALL}}
res["storm_days"] = on_info

# ---------------------------------------------------------------- 6. placebos: same number of days, same months, random
def placebo(kind, y, draws=4000):
    c = res["audit_wallclock"][kind][y]["cut"]
    s = yr(S_W.reindex(book.index), y)
    on = (s > c).to_numpy()
    sup, load = yr(book["sup"], y).to_numpy(), yr(book["load"], y).to_numpy()
    alt = np.zeros_like(sup) if kind == "filter" else load
    delta = alt - sup                                  # what switching a day changes
    months = s.index.month.to_numpy()
    actual = float(sup.sum() + delta[on].sum())
    tot = np.full(draws, float(sup.sum()))
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        kOn = int(on[idx].sum())
        if kOn == 0:
            continue
        dm = delta[idx]
        pick = np.argsort(rng.random((draws, len(idx))), axis=1)[:, :kOn]
        tot += dm[pick].sum(1)
    return {"actual": round(actual), "placebo_median": round(float(np.median(tot))),
            "placebo_p05": round(float(np.quantile(tot, 0.05))), "placebo_p95": round(float(np.quantile(tot, 0.95))),
            "share_placebo_at_or_above_actual": round(float((tot >= actual).mean()), 4)}


res["placebo_month_matched"] = {k: {y: placebo(k, y) for y in YEARS} for k in ("filter", "flip")}

SEASON = {6, 7, 8, 12, 1, 2}
season_on = pd.Series(book.index.month.isin(SEASON), index=book.index).astype(float)  # 1 = 'storm'
res["season_rule_jun_aug_dec_feb"] = {k: {y: round(float(yr(strategy(book, season_on, 0.5, k), y).sum())) for y in YEARS}
                                      for k in ("filter", "flip")}

# ---------------------------------------------------------------- 7. stationary bootstrap over days (mean block 7)
def boot(daily: np.ndarray, draws=5000, block=7):
    n = len(daily)
    new = rng.random((draws, n)) < 1 / block
    start = rng.integers(0, n, size=(draws, n))
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
    out = daily[idx].sum(1)
    return [round(float(np.quantile(out, 0.025))), round(float(np.quantile(out, 0.975)))]


bt = {}
sup_all = book["sup"][book.index.year.isin(YEARS)]
for k in ("filter", "flip"):
    p = res["audit_wallclock"][k]
    d = pd.concat([yr(strategy(book, S_W, p[y]["cut"], k), y) for y in YEARS])
    bt[k] = {"net_2021_2023": round(float(d.sum())), "ci95": boot(d.to_numpy()),
             "minus_always_supply": round(float((d - sup_all).sum())), "minus_always_supply_ci95": boot((d - sup_all).to_numpy())}
    for y in YEARS:
        bt[k][f"ci95_{y}"] = boot(yr(d, y).to_numpy(), draws=2000)
res["bootstrap"] = bt

OUT.mkdir(exist_ok=True)
json.dump(res, open(OUT / "storm_audit_fragility.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
