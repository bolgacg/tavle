"""The decision machine: one intraday position per hour, run end to end on Energinet's hourly
balancing data, with a registered rule, a learned gate given only the authority its measured
error earns, and guards that refuse.

Reads research/results/decision_hours.parquet (built by decision_data.py). Writes
research/results/decision.json for the page (tavle/decisionpage.py). Every number on the page
comes from this file or is recomputed in the browser from the per-hour arrays exported here,
by the same rules.

Periods: training = December 2019 to December 2023; the machine and the gate use its
single-pricing part, November 2021 onward; odd ISO weeks fit the gate, even ISO weeks measure
everything that is a choice (the age limit, which guards are on, the veto threshold);
separate = January 2024 to 4 March 2025, computed once with those choices frozen.
Registration and disclosure: research/PREREGISTRATION.md, section "The decision machine".
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN = ROOT / "research" / "results" / "decision_hours.parquet"
CAP = ROOT / "research" / "results" / "capacity_by_zone.json"
OUT = ROOT / "research" / "results" / "decision.json"

COST = 0.6            # EUR per unit hour, the registered v1 cost
HALF_VALUE = 0.5      # the feed may be late while the rule keeps at least this share of its fresh-feed value (even weeks)
EDGE_FLOOR = COST     # an optional guard: median |gap| of the last six settled hours must clear the cost
SINGLE = pd.Timestamp("2021-11-01")
SEP = pd.Timestamp("2024-01-01")
SEED = 0
FEATS = ["dir", "dir_age", "run2", "log_abs_gap2", "net_act2", "imb2", "rev5", "fc_share", "hod_sin", "hod_cos"]
QGRID = [round(q, 2) for q in np.arange(0.50, 0.8001, 0.05)]   # veto thresholds on P(rule direction is right)
SHARES = [round(x, 2) for x in np.arange(0, 1.0001, 0.1)]      # override shares, kept for the null table
REASON = {0: "traded", 1: "feed gap", 2: "no direction in the last settled hour", 3: "dual pricing regime", 4: "edge below cost",
          5: "feed outage", 6: "capacity step month", 7: "vetoed by the learned gate", 8: "feed too late"}
DEFAULT_GUARDS = {"feed": True, "direction": True, "late": True, "regime": True, "edge": False, "capstep": False}


# ---------------------------------------------------------------- learned gate (torch)
def fit_gate(Xf, yf, Xc, yc, hidden=16, seed=SEED, epochs=600, patience=50):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xf_t = torch.tensor(Xf, dtype=torch.float32, device=dev); yf_t = torch.tensor(yf, dtype=torch.float32, device=dev)
    Xc_t = torch.tensor(Xc, dtype=torch.float32, device=dev); yc_t = torch.tensor(yc, dtype=torch.float32, device=dev)
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    else:
        net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], 1))
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad, ep = 1e9, None, 0, 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = loss_fn(net(Xf_t).squeeze(1), yf_t); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            cl = float(loss_fn(net(Xc_t).squeeze(1), yc_t))
        if cl < best - 1e-5:
            best, bad = cl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(1)).cpu().numpy()
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    return predict, {"device": dev_name, "torch": torch.__version__, "epochs_run": ep + 1, "best_cal_logloss": round(best, 4),
                     "params": int(sum(p.numel() for p in net.parameters())), "hidden": hidden, "seed": seed}


# ---------------------------------------------------------------- helpers
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (p >= q[i]) & (p <= q[i + 1]) if i == bins - 1 else (p >= q[i]) & (p < q[i + 1])
        if m.sum():
            out.append({"decile": i + 1, "predicted": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3), "n": int(m.sum())})
    return out


def summarise(pnl, pos, gap, traded, years, reason, cf):
    t = traded
    nz = t & (gap != 0)
    pt = pnl[t]
    srt = np.sort(pt)[::-1]
    top = srt[: max(1, len(srt) // 10)].sum() if len(srt) else 0.0
    out = {"hours": int(len(t)), "traded": int(t.sum()), "refused": int((~t).sum()),
           "hit": round(float(np.mean(np.sign(gap[nz]) == pos[nz])), 3) if nz.sum() else None,
           "mean": round(float(pt.mean()), 2) if len(pt) else None,
           "se": round(float(pt.std(ddof=1) / np.sqrt(len(pt))), 2) if len(pt) > 1 else None,
           "median": round(float(np.median(pt)), 2) if len(pt) else None,
           "total": round(float(pt.sum()), 0) if len(pt) else 0.0,
           "best_decile_share": round(float(top / pt.sum()), 2) if len(pt) and pt.sum() != 0 else None,
           "by_reason": {str(k): int(v) for k, v in zip(*np.unique(reason[~t], return_counts=True))} if (~t).sum() else {},
           "refused_counterfactual_mean": round(float(np.nanmean(cf[~t])), 2) if (~t).sum() and np.isfinite(cf[~t]).any() else None,
           "by_year": []}
    for yv in np.unique(years):
        m = t & (years == yv)
        if m.sum():
            nzm = m & (gap != 0)
            out["by_year"].append({"year": int(yv), "traded": int(m.sum()), "mean": round(float(pnl[m].mean()), 2), "total": round(float(pnl[m].sum()), 0),
                                   "hit": round(float(np.mean(np.sign(gap[nzm]) == pos[nzm])), 3) if nzm.sum() else None})
    return out


def direction(gap, delay=0):
    """The sign of the latest settled hour known at the gate (t-2-delay), zero when that hour had no
    regulation, NaN when it is missing. Mirrors the browser's function."""
    n = len(gap)
    dir_ = np.full(n, np.nan); settled = np.zeros(n, dtype=bool)
    k = 2 + delay
    if k < n:
        src = gap[:n - k]
        settled[k:] = ~np.isnan(src)
        dir_[k:] = np.where(np.isnan(src), np.nan, np.sign(src))
    return dir_, settled


def simulate(z, veto_q=None, delay=0, outage_hours=(), guards=None, override=False, always=False):
    g = dict(DEFAULT_GUARDS); g.update(guards or {})
    gap = z["gap"]; n = len(gap)
    dir_, settled = direction(gap, delay)
    edge = np.roll(z["edge"], delay) if delay else z["edge"]
    p = z["p"]
    reason = np.zeros(n, dtype=int)
    if g["regime"]:
        reason[(reason == 0) & (~z["single"])] = 3
    if len(outage_hours):
        reason[(reason == 0) & np.isin(z["hod"], list(outage_hours))] = 5
    if g["feed"]:
        reason[(reason == 0) & (~settled)] = 1
    if g["late"] and delay > z["late_max"]:
        reason[(reason == 0)] = 8
    if g["direction"]:
        reason[(reason == 0) & (np.isnan(dir_) | (dir_ == 0))] = 2
    if g["edge"]:
        reason[(reason == 0) & edge] = 4
    if g["capstep"]:
        reason[(reason == 0) & z["capmonth"]] = 6
    rule = np.where(np.isnan(dir_) | (dir_ == 0), z["majority"], dir_)
    pv = np.nan_to_num(p, nan=0.5)
    prule = np.where(rule > 0, pv, 1 - pv)
    if veto_q is not None:
        reason[(reason == 0) & ~np.isnan(p) & (prule < veto_q)] = 7
    if always:
        reason[:] = 0
    pos = rule.copy()
    if override:
        pos = np.where(np.isnan(p), rule, np.where(pv >= 0.5, 1.0, -1.0))
    traded = reason == 0
    gap0 = np.nan_to_num(gap, nan=0.0)
    pnl = np.where(traded, pos * gap0 - COST, 0.0)
    cf = np.where(~traded & ~np.isnan(gap), rule * gap0 - COST, np.nan)
    return pos, pnl, traded, reason, cf, prule


def zone_run(df, area):
    df = df.sort_values("hour_utc").reset_index(drop=True)
    df["year"] = df.hour_utc.dt.year
    iso = df.hour_utc.dt.isocalendar()
    df["odd_week"] = (iso.week.astype(int) % 2 == 1)
    df["log_abs_gap2"] = np.log1p(df["abs_gap2"])
    df["hod"] = df.hour_utc.dt.hour
    feat_ok = (df["settled2"] & df["gap2"].notna() & df["wind_fc_5h_mwh"].notna() & df["wind_fc_da_mwh"].notna()
               & df["consumption_mwh"].notna() & df["dir"].notna())
    X = df[FEATS].copy()
    X["dir_age"] = X["dir_age"].fillna(12).clip(upper=12)
    X = X.fillna(0.0)
    single = df["single_pricing"].values
    train = (df["period"] == "training").values
    sep = (df["period"] == "separate").values
    yv = df["y"].values
    y01 = (yv > 0).astype(float)
    nz = yv != 0
    odd = df["odd_week"].values
    fit_m = single & train & odd & feat_ok.values & nz
    cal_m = single & train & (~odd) & feat_ok.values & nz
    sep_m = sep & feat_ok.values & nz
    mu = X[fit_m].mean(); sd = X[fit_m].std().replace(0, 1)
    Xs = ((X - mu) / sd).values.astype(np.float32)

    predict, info = fit_gate(Xs[fit_m], y01[fit_m], Xs[cal_m], y01[cal_m], hidden=16)
    predict_lin, _ = fit_gate(Xs[fit_m], y01[fit_m], Xs[cal_m], y01[cal_m], hidden=0)
    ok = feat_ok.values & single
    p = np.full(len(df), np.nan); p[ok] = predict(Xs[ok])
    p_lin = np.full(len(df), np.nan); p_lin[ok] = predict_lin(Xs[ok])

    maj_share = float(y01[fit_m].mean())
    majority_sign = 1.0 if maj_share >= 0.5 else -1.0
    dir_arr = df["dir"].values
    rule_hit_fit = float(np.mean(np.sign(yv[fit_m]) == dir_arr[fit_m]))
    p_rule = np.where(np.isnan(dir_arr), maj_share, np.where(dir_arr > 0, rule_hit_fit, 1 - rule_hit_fit))

    gap = np.round(df["gap"].values.astype(float), 2)  # the browser replays the same two-decimal values
    z = {"gap": gap, "p": p, "single": single, "hod": df["hod"].values, "year": df["year"].values,
         "edge": (df["med_abs_gap6"].fillna(0).values < EDGE_FLOOR), "majority": majority_sign}
    cap = json.load(open(CAP))["zones"].get(area, {})
    jumps = sorted({j["month"] for j in cap.get("jumps", [])})
    z["capmonth"] = df.hour_utc.dt.strftime("%Y-%m").isin(jumps).values
    gap0 = np.nan_to_num(gap, nan=0.0)

    # ---- measured on the even weeks of the single-pricing training years
    meas = single & train & (~odd)
    d0, s0 = direction(gap, 0)
    base = np.where(np.isnan(d0) | (d0 == 0), majority_sign, d0) * gap0 - COST
    okrow = s0 & ~np.isnan(gap) & (d0 != 0)
    # the lag table: the rule's worth on the even weeks when the freshest settled hour is d hours late
    lag_table = []
    for dl in range(0, 7):
        dd, ss = direction(gap, dl)
        m = meas & ss & ~np.isnan(gap) & (dd != 0) & ~np.isnan(dd)
        if m.sum():
            pnl_d = dd[m] * gap0[m] - COST
            nzm = gap[m] != 0
            lag_table.append({"delay": dl, "lag": 2 + dl, "hours": int(m.sum()), "mean": round(float(pnl_d.mean()), 2),
                              "hit": round(float(np.mean(np.sign(gap[m][nzm]) == dd[m][nzm])), 3) if nzm.sum() else None})
    fresh = lag_table[0]["mean"] if lag_table else 0.0
    late_max = max([l["delay"] for l in lag_table if l["mean"] >= HALF_VALUE * fresh] or [0])
    z["late_max"] = late_max
    # the zero-gap hour: what the rule earns when the latest settled hour had no direction and the last sign is used instead
    prev_sign = pd.Series(np.where(np.isnan(gap), np.nan, np.sign(gap))).replace(0, np.nan).ffill(limit=12).shift(2).values
    m = meas & s0 & (d0 == 0) & ~np.isnan(gap) & ~np.isnan(prev_sign)
    zero_gap_hours = {"hours": int(m.sum()), "mean_if_traded_on_last_sign": round(float((prev_sign[m] * gap0[m] - COST).mean()), 2) if m.sum() else None}
    # the edge guard, measured: what it would have refused and what those hours did
    m = meas & okrow & z["edge"]
    edge_measured = {"would_refuse": int(m.sum()), "counterfactual_mean": round(float(base[m].mean()), 2) if m.sum() else None}
    # the veto, measured: keep hours where the gate's probability of the rule's direction clears q
    pv = np.nan_to_num(p, nan=0.5)
    prule0 = np.where(d0 > 0, pv, 1 - pv)
    tm = meas & okrow & ~np.isnan(p)
    total_all = float(base[tm].sum())
    veto_curve = []
    for q in QGRID:
        keep = tm & (prule0 >= q); drop = tm & (prule0 < q)
        veto_curve.append({"q": q, "kept": int(keep.sum()), "removed": int(drop.sum()), "total_kept": round(float(base[keep].sum()), 0),
                           "mean_kept": round(float(base[keep].mean()), 2) if keep.sum() else None,
                           "removed_mean": round(float(base[drop].mean()), 2) if drop.sum() else None,
                           "removed_share": round(float(drop.sum() / tm.sum()), 3)})
    earned_q = 0.5
    for c in veto_curve:
        if c["total_kept"] >= total_all - 1e-9:
            earned_q = c["q"]
    earned_share = next(c["removed_share"] for c in veto_curve if c["q"] == earned_q)
    # twice the earned share: the smallest q on the grid whose removed share is at least double
    double_q = earned_q
    for c in veto_curve:
        if c["removed_share"] >= 2 * earned_share and earned_share > 0:
            double_q = c["q"]; break
    if earned_share == 0:
        double_q = next((c["q"] for c in veto_curve if c["removed_share"] > 0), earned_q)
    # the override, measured: does the gate's sign beat the rule's where it is confident? (the null)
    conf = np.abs(pv - 0.5)
    override_curve = []
    c_meas = conf[tm]
    for a in SHARES:
        if a == 0:
            continue
        tau = float(np.quantile(c_meas, 1 - a)) if a < 1 else 0.0
        m = tm & (conf >= tau)
        override_curve.append({"share": a, "n": int(m.sum()),
                               "hit_learned": round(float(np.mean(np.sign(pv[m] - 0.5) == np.sign(gap[m]))), 3) if m.sum() else None,
                               "hit_rule": round(float(np.mean(d0[m] == np.sign(gap[m]))), 3) if m.sum() else None,
                               "agree": round(float(np.mean(np.sign(pv[m] - 0.5) == d0[m])), 3) if m.sum() else None})

    card = {"framework": f"PyTorch {info['torch']}", "device": info["device"], "inputs": FEATS, "hidden": 16, "params": info["params"],
            "fit": {"hours": int(fit_m.sum()), "from": str(df.hour_utc[fit_m].min())[:10], "to": str(df.hour_utc[fit_m].max())[:10], "weeks": "odd ISO weeks"},
            "measured": {"hours": int(cal_m.sum()), "weeks": "even ISO weeks"},
            "separate": {"hours": int(sep_m.sum()), "from": str(df.hour_utc[sep_m].min())[:10], "to": str(df.hour_utc[sep_m].max())[:10]},
            "epochs_run": info["epochs_run"], "seed": info["seed"],
            "brier": {"measured": {"gate": round(brier(p[cal_m], y01[cal_m]), 4), "logistic": round(brier(p_lin[cal_m], y01[cal_m]), 4),
                                   "rule_as_probability": round(brier(p_rule[cal_m], y01[cal_m]), 4), "majority": round(brier(np.full(cal_m.sum(), maj_share), y01[cal_m]), 4)},
                      "separate": {"gate": round(brier(p[sep_m], y01[sep_m]), 4), "logistic": round(brier(p_lin[sep_m], y01[sep_m]), 4),
                                   "rule_as_probability": round(brier(p_rule[sep_m], y01[sep_m]), 4), "majority": round(brier(np.full(sep_m.sum(), maj_share), y01[sep_m]), 4)}},
            "reliability": {"measured": reliability(p[cal_m], y01[cal_m]), "separate": reliability(p[sep_m], y01[sep_m])},
            "majority_share": round(maj_share, 3), "rule_hit_fit": round(rule_hit_fit, 3),
            "override_curve": override_curve, "veto_curve": veto_curve, "earned_q": earned_q, "earned_removed_share": earned_share, "double_q": double_q}

    def arm(**kw):
        pos, pnl, traded, reason, cf, _ = simulate(z, **kw)
        res = {}
        for name, m in [("separate", sep), ("measured", meas), ("all", np.ones(len(df), bool))]:
            res[name] = summarise(pnl[m], pos[m], gap0[m], traded[m], z["year"][m], reason[m], cf[m])
        return res, pos, pnl, traded, reason, cf

    arms = {}
    arms["always_rule"], *_ = arm(always=True)
    arms["rule_guards"], pos_r, pnl_r, tr_r, rs_r, cf_r = arm()
    arms["veto_earned"], pos_v, pnl_v, tr_v, rs_v, cf_v = arm(veto_q=earned_q)
    arms["veto_double"], *_ = arm(veto_q=double_q)
    arms["learned_override"], *_ = arm(override=True)

    audit = []
    for code, label in REASON.items():
        if code == 0:
            continue
        m = sep & (rs_v == code)
        if m.sum():
            audit.append({"code": code, "guard": label, "refused": int(m.sum()),
                          "counterfactual_mean": round(float(np.nanmean(cf_v[m])), 2) if np.isfinite(cf_v[m]).any() else None})
    faults = {
        "feed_late_1h": arm(veto_q=earned_q, delay=1)[0]["separate"],
        "feed_late_3h": arm(veto_q=earned_q, delay=3)[0]["separate"],
        "feed_late_6h": arm(veto_q=earned_q, delay=6)[0]["separate"],
        "feed_late_6h_late_guard_off": arm(veto_q=earned_q, delay=6, guards={"late": False})[0]["separate"],
        "outage_00_05": arm(veto_q=earned_q, outage_hours=range(0, 6))[0]["separate"],
        "regime_guard_off": arm(veto_q=earned_q, guards={"regime": False})[0]["all"],
        "edge_guard_on": arm(veto_q=earned_q, guards={"edge": True})[0]["separate"],
        "capstep_guard_on": arm(veto_q=earned_q, guards={"capstep": True})[0]["separate"],
        "direction_guard_off": arm(veto_q=earned_q, guards={"direction": False})[0]["separate"],
    }
    idx = np.where(sep & tr_v)[0]
    worst = idx[np.argsort(pnl_v[idx])[:10]]
    worst_rows = [{"hour_utc": str(df.hour_utc[i])[:16], "gap": round(float(gap0[i]), 1), "position": int(pos_v[i]), "pnl": round(float(pnl_v[i]), 1),
                   "run": int(df["run2"][i]), "p_rule_right": round(float(np.where(pos_v[i] > 0, pv[i], 1 - pv[i])), 3) if not np.isnan(p[i]) else None} for i in worst]
    feed_gap_hours = [{"hour_utc": str(df.hour_utc[i])[:16], "gap": round(float(gap0[i]), 1)} for i in np.where(sep & (rs_v == 1) & ~np.isnan(gap))[0]]

    export = {
        "start": str(df.hour_utc.iloc[0])[:16], "hours": int(len(df)),
        "gap": [None if np.isnan(v) else round(float(v), 2) for v in gap],
        "p": [None if np.isnan(v) else round(float(v), 3) for v in p],
        "edge": [int(v) for v in z["edge"]], "capmonth": [int(v) for v in z["capmonth"]],
        "majority_sign": majority_sign, "late_max": late_max, "card": card, "lag_table": lag_table, "zero_gap_hours": zero_gap_hours, "edge_measured": edge_measured,
        "arms": arms, "guard_audit": audit, "faults": faults, "worst_hours": worst_rows, "feed_gap_hours": feed_gap_hours,
        "capacity_jumps": jumps, "sep_start_index": int(np.argmax(sep)), "single_start_index": int(np.argmax(single)),
    }
    return export


def main():
    df = pd.read_parquet(IN)
    out = {"cost": COST, "half_value": HALF_VALUE, "edge_floor": EDGE_FLOOR, "single_from": str(SINGLE.date()), "separate_from": str(SEP.date()),
           "qgrid": QGRID, "reasons": REASON, "default_guards": DEFAULT_GUARDS, "zones": {}}
    for area in ["DK1", "DK2"]:
        e = zone_run(df[df.area == area].copy(), area)
        out["zones"][area] = e
        a = e["arms"]
        print(f"{area}: late_max {e['late_max']} | earned q {e['card']['earned_q']} (removes {e['card']['earned_removed_share']}), double q {e['card']['double_q']} | separate: always {a['always_rule']['separate']['mean']} "
              f"rule {a['rule_guards']['separate']['mean']} veto {a['veto_earned']['separate']['mean']} double {a['veto_double']['separate']['mean']} override {a['learned_override']['separate']['mean']} "
              f"| traded {a['veto_earned']['separate']['traded']} refused {a['veto_earned']['separate']['refused']} {a['veto_earned']['separate']['by_reason']} | device {e['card']['device']}")
    OUT.write_text(json.dumps(out, default=float))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
