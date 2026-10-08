"""V14 positions -> results/v2/pos/V14*.parquet (+ .json), scored by score_v2.py like every other row.

All rows use the C pair rule (load i + supply j when the predicted spread clears both costs; 1 MW per leg at
most) with a fixed menu chosen per quarter on the trailing 4 out-of-sample quarters (v14_core.select_quarterly).
  V14a_window_ens     window-ensemble gap: menu gru_win (v14a_gru), gbm_win (v14a_gbm), their mean       (3)
  V14b_quantile_size  quantile heads size each pair leg: size = min(1, k * (|d| - cost) / width), d = spread of
                      the medians, width = mean of the two zones' q90 - q10 (floor 1 USD/MWh);
                      menu model {gru (v14b_gru), gbm (gbm q10/q50/q90), mean of both} x k {0.5, 1, 2}     (9)
  V14c_conformal_skip split conformal on the pair spread: half-width = the coverage quantile of |spread residual|
                      over the trailing 4 quarters' out-of-sample rows (public by 05:00 on the quarter's first
                      bid day), per pair; a leg is taken only when the interval [d - w, d + w] lies wholly past the
                      fee-adjusted break-even (+-both costs), else the pair-hour is skipped;
                      menu base {gbm mean, deep_c GRU} x coverage {0.5, 0.8}                              (4)
  V14d_seed_ens       mean of 10 GRU seeds (deep_c seeds 0..4 and v14d seeds 5..9) averaged with the gbm mean  (1)
  V14e_mlp_multi      Lago-style multi-output DNN: menu mlp_nowx, mlp_wx                                  (2)
  V14f_global_zone    global row model with zone embeddings: menu glob_mse, glob_dfl                       (2)

    python v14_positions.py [--only a,b,c,d,e,f] [--wait-hours 0]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import rolling as R
import v14_core as C

B_K = (0.5, 1.0, 2.0)
C_COV = (0.5, 0.8)
WIDTH_FLOOR = 1.0
NEED = {"a": ["v14a_gru", "v14a_gbm"], "b": ["v14b_gru", "gbm"], "c": ["gbm", "deep_c"],
        "d": ["v14d_gru", "deep_c", "gbm"], "e": ["v14e_mlp"], "f": ["v14f_mlp"]}
FLAGS = {"v14a_gru": "v14a_gru", "v14a_gbm": "v14a_gbm", "v14b_gru": "v14b_gru", "v14d_gru": "v14d_gru",
         "v14e_mlp": "v14e_mlp", "v14f_mlp": "v14f_mlp", "gbm": "v2_gbm", "deep_c": "v2_deep_c"}


def rows_for(names: list[str], panel: pd.DataFrame) -> pd.DataFrame:
    x = panel[["delivery_hour", "zone", "delivery_date", "gap"]]
    x = x[pd.to_datetime(x["delivery_date"]).dt.year >= R.WARMUP_YEAR]
    for n in names:
        p = R.load_preds(n)
        keep = [c for c in p.columns if c not in ("delivery_date", "quarter", "warmup", "ens_n", "fit_s",
                                                  "gbm_seconds") and not c.startswith(("dag_", "cond_"))]
        p = p[keep].rename(columns={c: f"{n}:{c}" for c in keep if c not in ("delivery_hour", "zone")})
        x = x.merge(p, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
    R.assert_pre2024(x["delivery_hour"], "v14 rows")
    return x


# ============================================================================ rules
def legs_quantile(W, lo, med, hi, k):
    d = C.spread(med)
    width = np.maximum((hi - lo)[:, C.PI] + (hi - lo)[:, C.PJ], 2 * WIDTH_FLOOR) / 2
    pc = W["pc"][:, None]
    edge = np.abs(d) - pc
    size = np.where(edge >= 0, np.minimum(1.0, k * edge / width), 0.0)
    return np.nan_to_num(np.sign(d) * size)


def legs_conformal(W, P, cov):
    """Per quarter: half-width per pair from the trailing out-of-sample residuals (earlier quarters only)."""
    d = C.spread(P)
    resid = np.abs(C.spread(W["gap"]) - d)
    pc = W["pc"][:, None]
    leg = np.zeros_like(d)
    dser = pd.Series(W["dd"])
    frame = pd.DataFrame({"delivery_date": W["dd"]})
    widths = {}
    for q in R.quarters():
        qs = ((dser >= pd.Timestamp(q[0])) & (dser <= pd.Timestamp(q[1]))).to_numpy()
        m = R.trailing_mask(frame, q[0])
        if not qs.any() or not m.any():
            continue
        w = np.nanquantile(resid[m], cov, axis=0)
        widths[q[2]] = w
        dq = d[qs]
        leg[qs] = np.where(dq - w >= pc[qs], 1.0, np.where(dq + w <= -pc[qs], -1.0, 0.0))
    return np.where(np.isfinite(d), leg, 0.0), widths


# ============================================================================ rows
def write(name, idea, line, W, settings, extra=None):
    long, ch = C.select_quarterly(W, settings)
    long = long[long["delivery_hour"] >= pd.Timestamp(f"{R.WARMUP_YEAR}-01-01", tz=R.TZ)]
    meta = {"idea": idea, "line": line, "settings_tried": len(settings), "menu": list(settings), "choices": ch,
            **(extra or {})}
    R.write_positions(name, long, meta)
    R.log(f"{name}: {np.abs(long['mw']).sum():,.0f} MWh over {len(settings)} settings")


def row_a(x):
    W = C.wide(x, ["v14a_gru:gru_win", "v14a_gbm:gbm_win"])
    g, b = W["v14a_gru:gru_win"], W["v14a_gbm:gbm_win"]
    write("V14a_window_ens", "V14a", "window ensemble (1, 2, 3, 5 years, expanding; recency half-life 12 months), "
          "pair rule", W, {"gru": C.leg_point(W, g), "gbm": C.leg_point(W, b), "both": C.leg_point(W, (g + b) / 2)})


def row_b(x):
    cols = ["v14b_gru:gq10", "v14b_gru:gq50", "v14b_gru:gq90", "gbm:q10", "gbm:q50", "gbm:q90"]
    W = C.wide(x, cols)
    m = {"gru": [W[c] for c in cols[:3]], "gbm": [W[c] for c in cols[3:]]}
    m["both"] = [(a + b) / 2 for a, b in zip(m["gru"], m["gbm"])]
    st = {f"{mod}_k{k:g}": legs_quantile(W, *m[mod], k) for mod in m for k in B_K}
    write("V14b_quantile_size", "V14b", "pair rule on the quantile medians, leg size min(1, k * edge / band width)",
          W, st)


def row_c(x):
    W = C.wide(x, ["gbm:mean", "deep_c:pred"])
    st, wd = {}, {}
    for base, col in (("gbm", "gbm:mean"), ("gru", "deep_c:pred")):
        for cov in C_COV:
            st[f"{base}_cov{cov:g}"], w = legs_conformal(W, W[col], cov)
            wd[f"{base}_cov{cov:g}"] = {k: float(np.nanmedian(v)) for k, v in w.items()}
    write("V14c_conformal_skip", "V14c", "pair rule, skipped when the conformal interval of the pair spread "
          "includes the fee-adjusted break-even", W, st, {"median_half_width_per_quarter": wd})


def row_d(x):
    W = C.wide(x, ["deep_c:pred", "v14d_gru:pred5", "gbm:mean"])
    gru10 = (W["deep_c:pred"] + W["v14d_gru:pred5"]) / 2
    write("V14d_seed_ens", "V14d", "pair rule on the mean of 10 GRU seeds and the gradient-boosting model", W,
          {"gru10_gbm": C.leg_point(W, (gru10 + W["gbm:mean"]) / 2)})


def row_mlp(x, src, name, idea, line):
    vs = [c for c in x.columns if c.startswith(f"{src}:")]
    W = C.wide(x, vs)
    write(name, idea, line, W, {v.split(":")[1]: C.leg_point(W, W[v]) for v in vs})


def ready(names) -> list[str]:
    return [n for n in names if not ((R.RESULTS / f"{FLAGS[n]}.done").exists() and (R.PREDS / f"{n}.parquet").exists())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b,c,d,e,f")
    ap.add_argument("--wait-hours", type=float, default=0.0)
    a = ap.parse_args()
    want = a.only.split(",")
    t0 = time.time()
    while True:
        missing = sorted({n for k in want for n in ready(NEED[k])})
        if not missing or time.time() - t0 > a.wait_hours * 3600:
            break
        time.sleep(300)
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap"])
    fns = {"a": row_a, "b": row_b, "c": row_c, "d": row_d,
           "e": lambda x: row_mlp(x, "v14e_mlp", "V14e_mlp_multi", "V14e", "pair rule on the multi-output DNN "
                                  "(day matrix -> 24 hours x 11 zones)"),
           "f": lambda x: row_mlp(x, "v14f_mlp", "V14f_global_zone", "V14f", "pair rule on one global model with "
                                  "learned zone and hour embeddings")}
    done_rows = []
    for k in want:
        miss = ready(NEED[k])
        if miss:
            R.log(f"V14{k} NOT WRITTEN: predictions missing {miss}")
            continue
        fns[k](rows_for(NEED[k], panel))
        done_rows.append(k)
    if len(done_rows) == len(want) == 6:
        R.done("v14_positions")
    else:
        raise SystemExit(f"V14 rows written: {done_rows} of {want}")


if __name__ == "__main__":
    main()
