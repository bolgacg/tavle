"""Positions for the v2 strategy rows, from the rolling predictions (rolling.py, gbm_rolling.py, deep_rolling.py).

Every setting used in a quarter is chosen on earlier out-of-sample quarters only (rolling.choose_setting: the
trailing 4 quarters, outcomes public by 05:00 on the quarter's first bid day; the 2012 warm-up quarters serve
2013). Rows (name: idea, rule):
  V1_baseline            side of the trailing 365-day zone-hour mean gap (v1 baseline)
  V1_always_supply       supply in every zone-hour
  V1_A_spike_gbm         supply unless P(gap >= S) > p*; (S, p*) from 3 x 2 (v1 grid), chosen per quarter
  V1_C_gbm, V1_C_deep    v1 pair rule: per pair (i, j) load i + supply j when pred_i - pred_j >= load + supply
                         cost, the reverse when <= -(load + supply cost); the 5 best pairs with positive net on the
                         trailing 4 quarters (all 55 scored), chosen per quarter
  V2_limit_gbm           price-sensitive bids: per zone-hour one bid with a limit price from the conditional
                         model's predicted gap on a grid of day-ahead levels (v2_limits); the bid counts only when
                         the realised day-ahead price clears at or past the limit (supply: DA >= limit; load: DA <= limit)
  V3_tail_gbm            side of the mean model (two-sided v1 rule), size = min(CAP, k * edge / tail), edge =
                         |mean| - cost of the side, tail = q90 (supply) or -q10 (load), floored at 1 USD/MWh;
                         k from (5, 10, 20, 40), chosen per quarter; CAP = 3 MW (declared)
  V10_C_deep_pretrain    the pair rule on the pretrained-then-fine-tuned deep model
  V11_C_gbm_ens4, V11_C_deep_ens4   the pair rule on the mean of the last 4 windows' models
Positions cover 2012 (warm-up, not scored; the allocator learns from them) to 2023.

    python strategies_v2.py [--only V1,V2,...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools

import numpy as np
import pandas as pd

import rolling as R

ZONES = R.ZONES
PAIRS = list(itertools.combinations(ZONES, 2))
MAX_PAIRS = 5
A_GRID = [(S, p) for S in (25, 50, 100) for p in (0.02, 0.05)]
V3_K = (5.0, 10.0, 20.0, 40.0)
V3_CAP = 3.0
TAIL_FLOOR = 1.0


# ============================================================================ helpers
def attach(preds: pd.DataFrame, panel: pd.DataFrame, cols=("gap", "y_da_lbmp", "gap_365d_h")) -> pd.DataFrame:
    keep = [c for c in cols if c in panel.columns]
    x = preds.merge(panel[["delivery_hour", "zone"] + keep], on=["delivery_hour", "zone"], how="left",
                    validate="one_to_one")
    R.assert_pre2024(x["delivery_hour"], "attach")
    return x


def per_quarter(rows: pd.DataFrame, settings, pos_fn, default=None):
    """Positions for every quarter with its setting chosen on earlier rows only."""
    mw = np.zeros(len(rows))
    ch = {}
    for q in R.quarters():
        sel = R.test_mask(rows, q)
        if not sel.any():
            continue
        hist = rows[pd.to_datetime(rows["delivery_date"]) < pd.Timestamp(q[0])]
        s, how = R.choose_setting(hist, settings, pos_fn, q[0], default)
        mw[sel] = pos_fn(rows[sel], s)
        ch[q[2]] = {"setting": s, "how": how}
    return mw, ch


def out(name, rows, mw, meta):
    d = rows[["delivery_hour", "zone"]].copy()
    d["mw"] = np.nan_to_num(mw)
    R.write_positions(name, d, meta)
    R.log(f"{name}: {np.abs(d['mw']).sum():,.0f} MWh, mean |mw| {np.abs(d['mw']).mean():.3f}")


# ============================================================================ simple rows
def basics(panel: pd.DataFrame):
    rows = panel[pd.to_datetime(panel["delivery_date"]).dt.year >= R.WARMUP_YEAR]
    s = rows["gap_365d_h"].to_numpy(float)
    mw = np.where(np.isnan(s), 0.0, np.sign(s))
    out("V1_baseline", rows, mw, {"idea": "V1", "line": "side of the trailing 365-day zone-hour mean gap",
                                   "settings_tried": 1})
    out("V1_always_supply", rows, -np.ones(len(rows)), {"idea": "V1", "line": "supply in every zone-hour",
                                                         "settings_tried": 1})


# ============================================================================ A spike
def a_pos(rows, s):
    S, pstar = s
    p = rows[f"p_s{S}"].to_numpy(float)
    return np.where(np.isnan(p) | (p > pstar), 0.0, -1.0)


def idea_a(g: pd.DataFrame):
    mw, ch = per_quarter(g, A_GRID, a_pos)
    out("V1_A_spike_gbm", g, mw, {"idea": "V1", "line": "supply unless P(spike) > p*", "settings_tried": len(A_GRID),
                                  "grid": A_GRID, "choices": ch})


# ============================================================================ C pairs
class Pairs:
    """Hourly pair legs and net per pair for one prediction column; rows indexed like the long frame."""

    def __init__(self, rows: pd.DataFrame, col: str):
        w = rows.pivot_table(index="delivery_hour", columns="zone", values=[col, "gap"], aggfunc="first")
        self.hours = w.index
        P = w[col].reindex(columns=ZONES).to_numpy(float)
        Gp = w["gap"].reindex(columns=ZONES).to_numpy(float)
        dd = self.hours.tz_convert(R.TZ).tz_localize(None).normalize()
        self.dd = pd.Series(dd)
        sup, lod = R.row_costs(dd.year)
        pc = sup + lod
        n = len(PAIRS)
        self.leg = np.zeros((len(self.hours), n))
        self.pnl = np.zeros((len(self.hours), n))
        for k, (i, j) in enumerate(PAIRS):
            a, b = ZONES.index(i), ZONES.index(j)
            d = P[:, a] - P[:, b]
            leg = np.where(d >= pc, 1.0, np.where(d <= -pc, -1.0, 0.0))
            ok = np.isfinite(d) & np.isfinite(Gp[:, a]) & np.isfinite(Gp[:, b])
            leg = np.where(ok, leg, 0.0)
            self.leg[:, k] = leg
            self.pnl[:, k] = leg * np.nan_to_num(Gp[:, a] - Gp[:, b]) - np.abs(leg) * pc

    def run(self, rows: pd.DataFrame):
        H = pd.Index(self.hours)
        mwz = np.zeros((len(H), len(ZONES)))
        ch = {}
        frame = pd.DataFrame({"delivery_date": self.dd})
        for q in R.quarters():
            qs = ((self.dd >= pd.Timestamp(q[0])) & (self.dd <= pd.Timestamp(q[1]))).to_numpy()
            if not qs.any():
                continue
            m = R.trailing_mask(frame, q[0])
            tot = self.pnl[m].sum(0)
            order = np.argsort(-tot, kind="stable")
            chosen = [k for k in order if tot[k] > 0][:MAX_PAIRS] if m.any() else []
            for k in chosen:
                a, b = ZONES.index(PAIRS[k][0]), ZONES.index(PAIRS[k][1])
                mwz[qs, a] += self.leg[qs, k]
                mwz[qs, b] -= self.leg[qs, k]
            ch[q[2]] = {"pairs": ["|".join(PAIRS[k]) for k in chosen],
                        "how": (f"best {MAX_PAIRS} positive of 55 on trailing {int(m.sum())} hours" if m.any()
                                else "no earlier out-of-sample quarter: no pairs")}
        long = pd.DataFrame(mwz, index=H, columns=ZONES).stack().rename("mw").reset_index()
        long.columns = ["delivery_hour", "zone", "mw"]
        return long, ch


def idea_c(name: str, g: pd.DataFrame, col: str, idea: str, line: str):
    p = Pairs(g, col)
    long, ch = p.run(g)
    R.write_positions(name, long, {"idea": idea, "line": line, "settings_tried": 1, "pair_choice": "5 of 55 per quarter",
                                   "choices": ch})
    R.log(f"{name}: {np.abs(long['mw']).sum():,.0f} MWh")


# ============================================================================ V2 limit prices
def v2_limits(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(side, limit, expected) per row from the grid only (05:00 information): supply limit = the lowest grid
    day-ahead level from which supply's predicted profit (-gap - supply cost) is positive at every higher level;
    load limit = the highest level up to which load's predicted profit (gap - load cost) is positive at every
    lower level. One bid per zone-hour: the side with the larger predicted profit at the median grid level
    (today's day-ahead price), supply on ties."""
    K = len([c for c in g.columns if c.startswith("cond_")])
    C = np.stack([g[f"cond_{k}"].to_numpy(float) for k in range(K)], 1)
    L = np.stack([g[f"dag_{k}"].to_numpy(float) for k in range(K)], 1)
    order = np.argsort(np.nan_to_num(L, nan=0.0), axis=1)          # levels ascending (negative DA flips order)
    C, L = np.take_along_axis(C, order, 1), np.take_along_axis(L, order, 1)
    sup, lod = R.row_costs(pd.to_datetime(g["delivery_date"]).dt.year)
    ps = -C - sup[:, None] > 0                                      # supply profitable at level k
    pl = C - lod[:, None] > 0
    s_ok = np.flip(np.cumprod(np.flip(ps, 1), 1), 1).astype(bool)  # profitable at k and every higher level
    l_ok = np.cumprod(pl, 1).astype(bool)                           # profitable at k and every lower level
    has_s, has_l = s_ok.any(1), l_ok.any(1)
    s_lim = np.where(has_s, L[np.arange(len(g)), np.argmax(s_ok, 1)], np.nan)
    l_lim = np.where(has_l, L[np.arange(len(g)), K - 1 - np.argmax(np.flip(l_ok, 1), 1)], np.nan)
    mid = K // 2
    es, el = -C[:, mid] - sup, C[:, mid] - lod
    side = np.where(has_s & (~has_l | (es >= el)), -1.0, np.where(has_l, 1.0, 0.0))
    valid = np.isfinite(L).all(1) & np.isfinite(C).all(1)
    side = np.where(valid, side, 0.0)
    limit = np.where(side < 0, s_lim, np.where(side > 0, l_lim, np.nan))
    return side, limit, np.where(side < 0, es, el)


def v2_pos(g: pd.DataFrame) -> np.ndarray:
    side, limit, _ = v2_limits(g)
    da = g["y_da_lbmp"].to_numpy(float)
    clears = np.where(side < 0, da >= limit, np.where(side > 0, da <= limit, False))
    return np.where(clears & np.isfinite(da), side, 0.0)


def idea_v2(g: pd.DataFrame):
    mw = v2_pos(g)
    out("V2_limit_gbm", g, mw, {"idea": "V2", "line": "limit-price bids from the gap predicted at 9 day-ahead levels; "
                                "counts only when day-ahead clears at or past the limit", "settings_tried": 1,
                                "levels": "da_d0_h x (0.5, 0.7, 0.85, 1, 1.15, 1.3, 1.6, 2, 3)"})


# ============================================================================ V3 tail-risk sizing
def v3_pos(rows: pd.DataFrame, k: float) -> np.ndarray:
    mu = rows["mean"].to_numpy(float)
    sup, lod = R.row_costs(pd.to_datetime(rows["delivery_date"]).dt.year)
    side = np.where(mu <= -sup, -1.0, np.where(mu >= lod, 1.0, 0.0))
    edge = np.where(side < 0, -mu - sup, np.where(side > 0, mu - lod, 0.0))
    tail = np.where(side < 0, rows["q90"].to_numpy(float), -rows["q10"].to_numpy(float))
    tail = np.maximum(np.nan_to_num(tail, nan=np.inf), TAIL_FLOOR)
    size = np.minimum(V3_CAP, k * edge / tail)
    return np.nan_to_num(side * size)


def idea_v3(g: pd.DataFrame):
    mw, ch = per_quarter(g, list(V3_K), v3_pos, default=10.0)
    out("V3_tail_gbm", g, mw, {"idea": "V3", "line": f"mean-model side, size min({V3_CAP}, k * edge / tail quantile)",
                               "settings_tried": len(V3_K), "grid": V3_K, "default": 10.0, "choices": ch})


# ============================================================================ registered comparison rows (addendum 7 Oct,
# after the stage-1 audit; logic of research/us_v2/audit/audit_stage1.py)
def static_pairs(panel: pd.DataFrame):
    """C_static_pairs, no model: per quarter, each pair's usual side per local hour = sign of its mean spread
    (gap_i - gap_j) over the 3-year window (delivery dates up to quarter start - 2 days); the pairs are the 5 with
    the best positive net of that static rule over the trailing 4 quarters (sides from the window ending a year
    earlier, as the audit's 'no model at all' row); 1 MW per leg, same costs."""
    rows = panel[pd.to_datetime(panel["delivery_date"]).dt.year >= R.WARMUP_YEAR - 2]
    w = rows.pivot_table(index="delivery_hour", columns="zone", values="gap", aggfunc="first").reindex(columns=ZONES)
    R.assert_pre2024(w.index, "static pairs")
    G = w.to_numpy(float)
    H = w.index
    loc = H.tz_convert(R.TZ)
    dd = pd.Series(loc.tz_localize(None).normalize())
    hod = loc.hour.to_numpy()
    sup, lod = R.row_costs(dd.dt.year)
    pc = sup + lod
    SP = np.stack([G[:, ZONES.index(i)] - G[:, ZONES.index(j)] for i, j in PAIRS], 1)
    legs = np.zeros_like(SP)
    ch = {}

    def side_hod(tr):
        mu = np.full((24, len(PAIRS)), np.nan)
        for h in range(24):
            sel = tr & (hod == h)
            if sel.any():
                mu[h] = np.nanmean(SP[sel], 0)
        return np.sign(np.nan_to_num(mu))

    for q in R.quarters():
        qs = ((dd >= pd.Timestamp(q[0])) & (dd <= pd.Timestamp(q[1]))).to_numpy()
        if not qs.any():
            continue
        lo, hi = R.window(q[0])
        tr = ((dd >= pd.Timestamp(lo)) & (dd <= pd.Timestamp(hi))).to_numpy()
        s_now = side_hod(tr)
        t4 = ((dd >= pd.Timestamp(q[0]) - pd.DateOffset(months=12)) & (dd <= pd.Timestamp(hi))).to_numpy()
        lo4, hi4 = R.window(q[0] - dt.timedelta(days=365))
        tr4 = ((dd >= pd.Timestamp(lo4)) & (dd <= pd.Timestamp(hi4))).to_numpy()
        if t4.any() and tr4.any():
            s4 = side_hod(tr4)[hod[t4]]
            net4 = np.nansum(s4 * SP[t4] - np.abs(s4) * pc[t4][:, None], 0)
            best = [k for k in np.argsort(-net4, kind="stable") if net4[k] > 0][:MAX_PAIRS]
            how = "best 5 positive of 55 by trailing-4Q net of the static rule"
        else:
            best, how = [], "no earlier window: no pairs"
        for k in best:
            legs[qs, k] = s_now[hod[qs], k]
        ch[q[2]] = {"pairs": ["|".join(PAIRS[k]) for k in best], "how": how}
    ok = np.isfinite(SP)
    legs = np.where(ok, legs, 0.0)
    Z = np.zeros((len(H), len(ZONES)))
    for k, (i, j) in enumerate(PAIRS):
        Z[:, ZONES.index(i)] += legs[:, k]
        Z[:, ZONES.index(j)] -= legs[:, k]
    long = pd.DataFrame(Z, index=H, columns=ZONES).stack().rename("mw").reset_index()
    long.columns = ["delivery_hour", "zone", "mw"]
    long = long[long["delivery_hour"] >= pd.Timestamp(dt.date(R.WARMUP_YEAR, 1, 1), tz=R.TZ)]
    R.write_positions("C_static_pairs", long, {"idea": "comparison", "comparison": True, "settings_tried": 1,
                      "line": "no model: 5 pairs by trailing static net, usual side per pair x local hour from the "
                              "3-year window, 1 MW", "choices": ch})
    R.log(f"C_static_pairs: {np.abs(long['mw']).sum():,.0f} MWh")


def comparisons(g: pd.DataFrame):
    side, _, _ = v2_limits(g)
    out("V2_sides_no_limit", g, side, {"idea": "comparison", "comparison": True, "settings_tried": 1,
                                       "line": "the gradient-boosting conditional model's side at every hour, 1 MW, "
                                               "no limit price"})
    mw, ch = per_quarter(g, list(V3_K), v3_pos, default=10.0)
    out("V3_tail_gbm_1MW", g, np.sign(mw), {"idea": "comparison", "comparison": True, "settings_tried": 0,
                                            "line": "V3's positions at 1 MW (same sides and hours, size removed)",
                                            "choices": ch})


def relabel():
    """Labels corrected by the addendum (the row names stay, so earlier files still match)."""
    import json
    for name, label in (("V2_limit_gbm", "V2 = gradient-boosting direction model (the limit prices add nothing; "
                                         "see V2_sides_no_limit)"),
                        ("V3_tail_gbm", "V3 tail-risk sizing: read per MWh and at 1 MW (V3_tail_gbm_1MW); its total "
                                        "reflects position size")):
        f = R.POS / f"{name}.json"
        if f.exists():
            m = json.loads(f.read_text())
            m["label"] = label
            f.write_text(json.dumps(m, indent=1, default=str))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    want = lambda k: only is None or k in only  # noqa: E731
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp", "gap_365d_h"])
    if a.only == "comparisons":
        static_pairs(panel)
        comparisons(attach(R.load_preds("gbm"), panel))
        relabel()
        return
    if want("V1"):
        basics(panel)
    if (R.PREDS / "gbm.parquet").exists():
        g = attach(R.load_preds("gbm"), panel)
        if want("V1"):
            idea_a(g)
            idea_c("V1_C_gbm", g, "mean", "V1", "pair rule on the gradient-boosting gap model")
        if want("V11"):
            idea_c("V11_C_gbm_ens4", g, "ens4", "V11", "pair rule on the mean of the last 4 windows' GBM models")
        if want("V2"):
            idea_v2(g)
        if want("V3"):
            idea_v3(g)
    else:
        R.log("gbm predictions missing: A, C GBM, V2, V3, V11 GBM not written")
    for f, col, name, idea, line in (
            ("deep_c", "pred", "V1_C_deep", "V1", "pair rule on the deep gap model"),
            ("deep_c", "ens4", "V11_C_deep_ens4", "V11", "pair rule on the mean of the last 4 windows' deep models"),
            ("deep_v10", "pred", "V10_C_deep_pretrain", "V10", "pair rule on the deep model pretrained from 2010, "
                                                               "fine-tuned on the window")):
        if want(idea) and (R.PREDS / f"{f}.parquet").exists():
            idea_c(name, attach(R.load_preds(f), panel), col, idea, line)
        elif want(idea):
            R.log(f"{f} predictions missing: {name} not written")


if __name__ == "__main__":
    main()
