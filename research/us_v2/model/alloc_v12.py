"""V12: learned allocator over the v2 survivors (the idea 14 lineage, v1 deep_alloc14.py reused unchanged).

Per delivery day a small deep gate (deep_alloc14.GateNet over the day-feature matrix, deep_policy encoders)
splits one unit of risk between the candidate rows; position per zone-hour = sum_c w_c(D) * scale_c * mw_c.
Candidates: every positions file in results/v2/pos/ except V12 itself (rows from 2012 on; 2012 = warm-up).
Survivor rule, per refit (declared before any v2 result): a candidate takes part only with at least 60 training
days of its own profits AND a positive net over the trailing 365 days (delivery dates up to 2 days before the
quarter). Each taking-part candidate is pre-scaled to a 100,000 USD worst drawdown on its training days (cap 5x),
as idea 14. Refits quarterly on the 3-year window (rolling.py). kappa (0.01, 0.1, 1): the one whose own walk-forward
weights earned most over the trailing 4 quarters; declared default 0.1 when there is none.

    python alloc_v12.py [--device cuda] [--wait-ideas-hours 48]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rolling as R
import deep_policy as DP      # v1
import deep_alloc14 as A      # v1

NAME = "V12_alloc"
TRAIN_START = pd.Timestamp(dt.date(R.WARMUP_YEAR, 1, 1))


def candidate_profits(days: pd.DatetimeIndex):
    """P [n_days, C] realised day profit of each candidate at its own size, full cost by side; NaN where the
    candidate has no rows that day. Also returns the hourly mw per candidate on the price frame."""
    import pyarrow.dataset as ds
    px = R.read_pre2024(R.PARQUET / "prices_zone.parquet", "delivery_hour",
                        ["delivery_hour", "zone", "da_lbmp", "rt_lbmp"], ds.field("zone").isin(R.ZONES))
    px["delivery_hour"] = px["delivery_hour"].dt.tz_convert(R.TZ)
    px["ddate"] = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    px = px[px["ddate"] >= TRAIN_START].reset_index(drop=True)
    px["gap"] = px["rt_lbmp"] - px["da_lbmp"]
    sup, lod = R.row_costs(px["ddate"].dt.year)
    def is_comparison(f):
        m = f.with_suffix(".json")
        return m.exists() and bool(json.loads(m.read_text()).get("comparison"))
    names = sorted(f.stem for f in R.POS.glob("*.parquet") if not f.stem.startswith("V12") and not is_comparison(f))
    P = np.full((len(days), len(names)), np.nan)
    MW = np.zeros((len(px), len(names)))
    mi = pd.MultiIndex.from_arrays([px["delivery_hour"], px["zone"]])
    di = days.get_indexer(px["ddate"])
    for j, n in enumerate(names):
        pos = R.read_pre2024(R.POS / f"{n}.parquet", "delivery_hour")
        pos["delivery_hour"] = pos["delivery_hour"].dt.tz_convert(R.TZ)
        s = pos.groupby(["delivery_hour", "zone"])["mw"].sum()
        have = s.reindex(mi).notna().to_numpy()
        mw = np.nan_to_num(s.reindex(mi).to_numpy(float))
        MW[:, j] = mw
        g = np.nan_to_num(px["gap"].to_numpy(float))
        pnl = mw * g - np.abs(mw) * np.where(mw < 0, sup, lod)
        ok = (di >= 0) & have
        tot = np.bincount(di[ok], weights=pnl[ok], minlength=len(days))
        cnt = np.bincount(di[ok], minlength=len(days))
        P[:, j] = np.where(cnt > 0, tot, np.nan)
    return names, P, px, MW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--wait-ideas-hours", type=float, default=48.0)
    ap.add_argument("--grace-hours", type=float, default=16.0)
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    # Wait for every idea row (v2_ideas.done, written after the weather idea V4); once the price-only rows are in
    # (v2_ideas_price.done), allow at most --grace-hours more for V4; hard cap --wait-ideas-hours from the start.
    ideas, price = R.RESULTS.parent / "v2_ideas.done", R.RESULTS.parent / "v2_ideas_price.done"
    t0, tp = time.time(), None
    while not ideas.exists() and time.time() - t0 < a.wait_ideas_hours * 3600:
        if price.exists() and tp is None:
            tp = time.time()
        if tp is not None and time.time() - tp > a.grace_hours * 3600:
            break
        time.sleep(120)
    R.log(f"ideas: all={ideas.exists()}, price={price.exists()} after {(time.time() - t0) / 3600:.1f} h")
    feats = DP.load_day_features(R.DAYFEATS)
    feats = feats[feats["delivery_date"] >= TRAIN_START].reset_index(drop=True)
    R.assert_pre2024(feats["delivery_date"], "v12 day features")
    days = pd.DatetimeIndex(feats["delivery_date"])
    names, P, px, MW = candidate_profits(days)
    C = len(names)
    A.CANDS, A.NC = tuple(names), C
    R.log(f"V12: {C} candidates {names}; features {feats.shape}")
    cfg = {"max_epochs": a.max_epochs} if a.max_epochs else None
    W = {k: np.zeros((len(days), C)) for k in A.KAPPAS}
    SC = np.zeros((len(days), C))
    recs = []
    dser = pd.Series(days)
    for q in R.quarters():
        if q[3]:
            continue
        lo, hi = R.window(q[0])
        te = ((dser >= pd.Timestamp(q[0])) & (dser <= pd.Timestamp(q[1]))).to_numpy()
        tr = ((dser >= max(pd.Timestamp(lo), TRAIN_START)) & (dser <= pd.Timestamp(hi))).to_numpy()
        trail = ((dser >= pd.Timestamp(q[0]) - pd.DateOffset(years=1)) & (dser <= pd.Timestamp(hi))).to_numpy()
        Ptr = P[tr]
        avail = (np.isfinite(Ptr).sum(0) >= A.MIN_HIST) & (np.nansum(P[trail], 0) > 0)
        scale = np.zeros(C)
        for j in np.flatnonzero(avail):
            scale[j] = A.dd_scale(Ptr[np.isfinite(Ptr[:, j]), j])
        rec = {"quarter": q[2], "taking_part": [n for n, x in zip(names, avail) if x],
               "scale": dict(zip(names, np.round(scale, 4).tolist()))}
        ts = time.time()
        if avail.any():
            ok = np.isfinite(Ptr[:, avail]).all(1)
            S = np.nan_to_num(Ptr[ok] * scale[None, :])
            trd = feats[tr][ok].reset_index(drop=True)
            m = A.Alloc14(A.KAPPAS, seeds, cfg, a.device).fit(trd, S, avail)
            w = m.predict(feats[te].reset_index(drop=True))
            for k in A.KAPPAS:
                W[k][te] = w[k]
            rec.update({k: v for k, v in m.info.items() if k != "best_epochs"})
        else:
            rec["note"] = "no survivor: sit out"
        SC[te] = scale[None, :]
        rec["seconds"] = round(time.time() - ts, 1)
        recs.append(rec)
        R.log(f"V12 {q[2]}: part {rec['taking_part']}, {rec['seconds']} s")
    # kappa per quarter on the trailing 4 quarters of its own walk-forward weights (training-objective net)
    Pn = np.nan_to_num(P)
    net = {k: (W[k] * SC * Pn).sum(1) for k in A.KAPPAS}
    frame = pd.DataFrame({"delivery_date": days})
    wsel = np.zeros((len(days), C))
    choice = {}
    for q in R.quarters():
        if q[3]:
            continue
        te = ((dser >= pd.Timestamp(q[0])) & (dser <= pd.Timestamp(q[1]))).to_numpy()
        m = R.trailing_mask(frame, q[0]) & (dser >= pd.Timestamp(R.FIRST_SCORED)).to_numpy()
        if m.any():
            k = max(A.KAPPAS, key=lambda kk: (net[kk][m].sum(), -A.KAPPAS.index(kk)))
            how = "best trailing net"
        else:
            k, how = A.KAPPAS[len(A.KAPPAS) // 2], "declared default"
        wsel[te] = W[k][te]
        choice[q[2]] = {"kappa": k, "how": how}
    # hourly positions
    di = days.get_indexer(px["ddate"])
    ok = di >= 0
    mw = np.zeros(len(px))
    mw[ok] = (wsel[di[ok]] * SC[di[ok]] * MW[ok]).sum(1)
    d = px[["delivery_hour", "zone"]].copy()
    d["mw"] = mw
    R.write_positions(NAME, d[d["delivery_hour"].dt.year >= R.FIRST_SCORED.year],
                      {"idea": "V12", "line": "learned allocator (idea 14 gate) over the v2 survivors",
                       "settings_tried": len(A.KAPPAS), "candidates": names, "kappa_choice": choice, "refits": recs})
    R.done("v2_v12")


if __name__ == "__main__":
    main()
