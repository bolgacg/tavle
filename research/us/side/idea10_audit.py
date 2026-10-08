"""Idea 10 audit, parts 2 to 5: independent net, concentration, cuts, placebos, costs.

Build years only (holdout filter at load in idea10_audit_lib, asserted). Does not import lab.py.
Run on gene: cd ~/nyiso-us/side_audit && ../.venv/bin/python -I idea10_audit.py
Writes idea10_audit.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "side_audit"))
import idea10_audit_lib as A    # noqa: E402
import storm_audit_lib as S     # noqa: E402

OUT = HOME / "side_audit" / "idea10_audit.json"
ELLIOTT = pd.DatetimeIndex(["2022-12-23", "2022-12-24"])
CLAIM = {"2021": 3034, "2022": 311142, "2023": 37955}
SEED = 20261007


def r0(x):
    return None if x is None or not np.isfinite(x) else round(float(x))


def nets_of(d):
    return A.year_nets(d)


def main():
    res = {}
    w = A.load_wx()
    book = A.load_book()
    A.assert_naive_build(book.index, "book")
    sc = A.score(w)
    rank = sc["rank"]
    sup, load = A.legs(book)
    yrs = book.index.year
    in3 = yrs.isin(A.YEARS)

    # ================= 2. independent re-implementation
    ch, d, on = A.walk(book, rank)
    res["reimpl"] = {"cuts": {str(k): v for k, v in ch.items()}, "nets": nets_of(d), "claimed": CLAIM,
                     "sharpe_3y": A.sharpe(d), "load_days": {str(Y): int(on[yrs == Y].sum()) for Y in A.YEARS},
                     "always_supply": nets_of(sup), "gain_over_always_supply": nets_of(d - sup)}
    for c in ["sup", "load"]:
        pass
    # fee check (base costs used by the book)
    res["fees_per_mwh"] = {str(Y): {"supply": round(S.SUPPLY_FEE[Y], 4), "load": round(S.LOAD_FEE[Y], 4)} for Y in A.YEARS}

    # ================= 3. concentration
    conc = {}
    for Y in A.YEARS:
        dy = d[yrs == Y]
        top = dy.sort_values(ascending=False).head(10)
        conc[str(Y)] = {"net": r0(dy.sum()), "without_best_1": A.without_best(dy, 1), "without_best_3": A.without_best(dy, 3),
                        "without_best_10": A.without_best(dy, 10),
                        "top10": [{"date": str(t.date()), "side": "load" if on[t] else "supply", "usd": r0(v),
                                   "mean_gap": round(float(book.loc[t, "sum_gap"] / book.loc[t, "n"]), 1),
                                   "rank": None if not np.isfinite(rank[t]) else round(float(rank[t]), 3),
                                   "surprise_c": None if not np.isfinite(sc.loc[t, "surprise"]) else round(float(sc.loc[t, "surprise"]), 2)}
                                  for t, v in top.items()]}
        ld = on[yrs == Y]
        lday = ld[ld].index
        conc[str(Y)]["load_days"] = int(len(lday))
        conc[str(Y)]["load_legs_usd"] = r0(load[lday].sum())
        conc[str(Y)]["supply_on_those_days_usd"] = r0(sup[lday].sum())
        nE = lday.difference(ELLIOTT)
        conc[str(Y)]["load_legs_usd_without_elliott"] = r0(load[nE].sum())
        conc[str(Y)]["supply_on_those_days_without_elliott"] = r0(sup[nE].sum())
        conc[str(Y)]["load_days_by_month"] = {int(k): int(v) for k, v in pd.Series(1, index=lday).groupby(lday.month).sum().items()}
    res["concentration"] = conc
    dE = d.drop(ELLIOTT)
    res["without_elliott"] = {"nets": {str(Y): r0(dE[dE.index.year == Y].sum()) for Y in A.YEARS},
                              "sum": r0(dE[dE.index.year.isin(A.YEARS)].sum()), "sharpe_3y": A.sharpe(dE),
                              "elliott_days_usd": {str(t.date()): r0(d[t]) for t in ELLIOTT},
                              "elliott_days_supply_usd": {str(t.date()): r0(sup[t]) for t in ELLIOTT},
                              "elliott_rank": {str(t.date()): round(float(rank[t]), 4) for t in ELLIOTT},
                              "always_supply_without_elliott": r0(sup.drop(ELLIOTT)[in3[~book.index.isin(ELLIOTT)]].sum())}
    # December 2022 day by day
    dec = book.loc["2022-12-15":"2022-12-31"]
    res["dec2022"] = [{"date": str(t.date()), "rank": None if not np.isfinite(rank[t]) else round(float(rank[t]), 3),
                       "surprise_c": None if not np.isfinite(sc.loc[t, "surprise"]) else round(float(sc.loc[t, "surprise"]), 2),
                       "side": "load" if on[t] else "supply", "usd": r0(d[t]), "supply_usd": r0(sup[t]),
                       "mean_gap": round(float(book.loc[t, "sum_gap"] / book.loc[t, "n"]), 1)} for t in dec.index]

    # every cut fixed, per year
    grid = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.98]
    fixed = {}
    for c in grid:
        dd, oo = A.rule_daily(book, rank, {Y: c for Y in A.YEARS})
        fixed[str(c)] = {**{str(Y): r0(dd[yrs == Y].sum()) for Y in A.YEARS},
                         "2022_without_elliott": r0(dd[(yrs == 2022) & ~book.index.isin(ELLIOTT)].sum()),
                         "2020": r0(dd[yrs == 2020].sum()),
                         "load_days": {str(Y): int(oo[yrs == Y].sum()) for Y in A.YEARS}}
    fixed["always_supply"] = {str(Y): r0(sup[yrs == Y].sum()) for Y in A.YEARS}
    res["fixed_cuts"] = fixed
    # neighbouring cuts on the registered grid: one year at a time and all together
    g = list(A.CUTS)
    nb = {}
    for lab, step in [("higher", 1), ("lower", -1)]:
        sets = {Y: g[min(max(g.index(ch[Y]) + step, 0), len(g) - 1)] for Y in A.YEARS}
        dd, _ = A.rule_daily(book, rank, sets)
        nb[f"all_{lab}"] = {"cuts": {str(k): v for k, v in sets.items()}, "nets": nets_of(dd)}
        for Y in A.YEARS:
            s1 = dict(ch); s1[Y] = sets[Y]
            dd, _ = A.rule_daily(book, rank, s1)
            nb[f"{Y}_{lab}"] = {"cuts": {str(k): v for k, v in s1.items()}, "nets": nets_of(dd)}
    res["neighbours"] = nb
    # how close was each choice (prior-year net at each grid cut)
    res["choice_margins"] = {str(Y): {str(c): r0(A.rule_daily(book, rank, {Y - 1: c})[0][yrs == Y - 1].sum()) for c in A.CUTS}
                             for Y in [2022, 2023]}
    res["choice_2023_without_elliott_in_2022"] = A.choose_cut(book, rank, 2023, drop_dates=ELLIOTT)
    res["choice_without_31_dec"] = {str(Y): A.choose_cut(book, rank, Y, drop_dates=[pd.Timestamp(f"{Y - 1}-12-31")]) for Y in A.YEARS}
    ins = {Y: A.CUTS[int(np.argmax([A.rule_daily(book, rank, {Y: c})[0][yrs == Y].sum() for c in A.CUTS]))] for Y in A.YEARS}
    res["in_sample_cuts"] = {str(k): v for k, v in ins.items()}
    # wider grid in the walk-forward
    wide = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
    chw, dw, _ = A.walk(book, rank, cuts=wide)
    res["walk_wide_grid"] = {"cuts": {str(k): v for k, v in chw.items()}, "nets": nets_of(dw)}

    # hours cutoff (timing margin): only hours up to H, so the newest run used is older
    hc = {}
    for H in [21, 15, 9, 3]:
        mp = math.ceil(A.MIN_PAIRS * (H + 1) / 22)
        s2 = A.score(w, last_hour=H, min_pairs=mp)
        c2, d2, o2 = A.walk(book, s2["rank"])
        hc[str(H)] = {"min_rule_margin_h": 21 - H, "min_pairs": mp, "cuts": {str(k): v for k, v in c2.items()},
                      "nets": nets_of(d2), "without_elliott": r0(d2.drop(ELLIOTT)[d2.drop(ELLIOTT).index.year.isin(A.YEARS)].sum()),
                      "elliott_flags": {str(t.date()): bool(o2[t]) for t in ELLIOTT},
                      "corr_with_claimed_surprise": round(float(s2["surprise"].corr(sc["surprise"])), 3)}
    res["hours_cutoff"] = hc

    # does the surprise track the day's gap at all?
    gap = book["sum_gap"] / book["n"]
    cor = {}
    for Y in A.YEARS + ["all"]:
        m = in3 if Y == "all" else (yrs == Y)
        x = pd.DataFrame({"s": sc["surprise"], "g": gap})[m].dropna()
        xe = x.drop(ELLIOTT, errors="ignore")
        cor[str(Y)] = {"days": len(x), "spearman": round(float(x["s"].corr(x["g"], method="spearman")), 3),
                       "pearson": round(float(x["s"].corr(x["g"])), 3),
                       "pearson_without_elliott": round(float(xe["s"].corr(xe["g"])), 3)}
    res["surprise_vs_gap"] = cor
    # mean gap on load days and on supply days (USD/MWh), excluding Elliott
    gd = {}
    for Y in A.YEARS:
        m = (yrs == Y) & ~book.index.isin(ELLIOTT)
        gd[str(Y)] = {"load_days_mean_gap": round(float(gap[m & on.to_numpy()].mean()), 2) if (m & on.to_numpy()).any() else None,
                      "supply_days_mean_gap": round(float(gap[m & ~on.to_numpy()].mean()), 2),
                      "load_days_share_gap_above_load_fee": round(float((gap[m & on.to_numpy()] > S.LOAD_FEE[Y]).mean()), 3)
                      if (m & on.to_numpy()).any() else None}
    res["gap_by_side_without_elliott"] = gd

    # ================= 4. placebos
    pl = {}
    r = rank.to_numpy()
    for k in [1, 2, 3, 7, -1]:
        rs = np.full(len(r), np.nan)
        if k > 0:
            rs[k:] = r[:-k]
        else:
            rs[:k] = r[-k:]
        rs = pd.Series(rs, index=book.index)
        c_own, d_own, o_own = A.walk(book, rs)
        d_fix, o_fix = A.rule_daily(book, rs, ch)
        pl[f"shift_{k}"] = {"own_walk_cuts": {str(a): b for a, b in c_own.items()}, "own_walk": nets_of(d_own),
                            "real_cuts": nets_of(d_fix),
                            "real_cuts_without_elliott": r0(d_fix.drop(ELLIOTT)[d_fix.drop(ELLIOTT).index.year.isin(A.YEARS)].sum()),
                            "elliott_flags_real_cuts": {str(t.date()): bool(o_fix[t]) for t in ELLIOTT},
                            "elliott_usd_real_cuts": {str(t.date()): r0(d_fix[t]) for t in ELLIOTT},
                            "load_days_real_cuts": {str(Y): int(o_fix[yrs == Y].sum()) for Y in A.YEARS}}
    # same calendar day one year earlier
    prev = book.index - pd.DateOffset(years=1)
    ry = pd.Series(rank.reindex(prev).to_numpy(), index=book.index)
    c_own, d_own, o_own = A.walk(book, ry)
    d_fix, o_fix = A.rule_daily(book, ry, ch)
    pl["year_earlier"] = {"own_walk_cuts": {str(a): b for a, b in c_own.items()}, "own_walk": nets_of(d_own),
                          "real_cuts": nets_of(d_fix),
                          "real_cuts_without_elliott": r0(d_fix.drop(ELLIOTT)[d_fix.drop(ELLIOTT).index.year.isin(A.YEARS)].sum()),
                          "elliott_flags_real_cuts": {str(t.date()): bool(o_fix[t]) for t in ELLIOTT},
                          "load_days_real_cuts": {str(Y): int(o_fix[yrs == Y].sum()) for Y in A.YEARS},
                          "note": "2021 has no surprise a year earlier (archive starts 25 March 2021), so 2021 is always supply"}
    # same rule, real for 2022 and 2023 only (comparable span with the year-earlier placebo)
    pl["real_2022_2023"] = r0(d[yrs.isin([2022, 2023])].sum())
    # persistence
    s = sc["surprise"]
    fl = (rank > 0.9).astype(float).where(rank.notna())
    pers = {"surprise_autocorr": {str(k): round(float(s.autocorr(k)), 3) for k in [1, 2, 3, 7, 365]},
            "flag_above_0.90_autocorr": {str(k): round(float(fl.autocorr(k)), 3) for k in [1, 2, 3, 7]},
            "daily_gap_autocorr_2021_2023": {str(k): round(float(gap[in3].autocorr(k)), 3) for k in [1, 2, 3, 7]},
            "daily_gap_autocorr_without_elliott": {str(k): round(float(gap[in3].drop(ELLIOTT).autocorr(k)), 3) for k in [1, 2, 3]}}
    pl["persistence"] = pers
    # shuffles: within year and within month, fixed real cuts and own walk
    rng = np.random.default_rng(SEED)
    real3 = float(d[in3].sum())
    real_y = {Y: float(d[yrs == Y].sum()) for Y in A.YEARS}
    for scope in ["year", "month"]:
        key = yrs if scope == "year" else book.index.to_period("M")
        groups = [np.where(key == k)[0] for k in pd.unique(key)]
        tots, tots_y = [], {Y: [] for Y in A.YEARS}
        for _ in range(500):
            perm = np.arange(len(r))
            for gi in groups:
                perm[gi] = rng.permutation(gi)
            rp = pd.Series(r[perm], index=book.index)
            _, dp, _ = A.walk(book, rp)
            tots.append(float(dp[in3].sum()))
            for Y in A.YEARS:
                tots_y[Y].append(float(dp[yrs == Y].sum()))
        tots = np.array(tots)
        pl[f"shuffle_within_{scope}"] = {"draws": 500, "mean": r0(tots.mean()), "p95": r0(np.quantile(tots, 0.95)),
                                         "share_at_or_above_real": round(float((tots >= real3).mean()), 3),
                                         "years": {str(Y): {"mean": r0(np.mean(tots_y[Y])),
                                                            "share_at_or_above_real": round(float((np.array(tots_y[Y]) >= real_y[Y]).mean()), 3)}
                                                   for Y in A.YEARS}}
    res["placebos"] = pl

    # ================= 5. costs
    cs = {}
    for c in [None, 0.50, 1.00]:
        dd, _ = A.rule_daily(book, rank, ch, cost=c)
        chs, ds_, _ = A.walk(book, rank, choose_cost=c, eval_cost=c)
        su = A.legs(book, c)[0]
        cs["base" if c is None else str(c)] = {"cut_at_base": nets_of(dd), "cut_chosen_at_this_cost": nets_of(ds_),
                                              "cuts_at_this_cost": {str(a): b for a, b in chs.items()},
                                              "without_elliott": r0(dd.drop(ELLIOTT)[dd.drop(ELLIOTT).index.year.isin(A.YEARS)].sum()),
                                              "always_supply": nets_of(su)}
    # break-even uniform fee: gross / MWh
    gross = d + np.where(on, book["n"] * pd.Series(yrs, index=book.index).map(S.LOAD_FEE),
                         book["n"] * pd.Series(yrs, index=book.index).map(S.SUPPLY_FEE))
    be = {}
    for Y in A.YEARS + ["3y"]:
        m = in3 if Y == "3y" else (yrs == Y)
        be[str(Y)] = round(float(gross[m].sum() / book["n"][m].sum()), 3)
    mE = in3 & ~book.index.isin(ELLIOTT)
    be["3y_without_elliott"] = round(float(gross[mE].sum() / book["n"][mE].sum()), 3)
    cs["break_even_fee_usd_per_mwh"] = be
    cs["mwh_per_day"] = float(book["n"][in3].mean())
    res["costs"] = cs

    # ================= bootstrap (stationary, block 7) with and without Elliott
    def boot(x, draws=5000, block=7):
        x = x.to_numpy()
        n = len(x)
        rg = np.random.default_rng(SEED + 1)
        new = rg.random((draws, n)) < 1 / block
        start = rg.integers(0, n, size=(draws, n))
        idx = np.empty((draws, n), dtype=np.int64)
        idx[:, 0] = start[:, 0]
        for t in range(1, n):
            idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
        m = x[idx].mean(1) * 365
        return [r0(np.quantile(m, 0.025)), r0(np.quantile(m, 0.975))]

    res["bootstrap_ci95_usd_per_year"] = {"all": boot(d[in3]), "without_elliott": boot(dE[dE.index.year.isin(A.YEARS)]),
                                          "gain_over_supply": boot((d - sup)[in3]),
                                          "gain_over_supply_without_elliott": boot((d - sup).drop(ELLIOTT)[lambda z: z.index.year.isin(A.YEARS)])}
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
