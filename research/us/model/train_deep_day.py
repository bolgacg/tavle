"""Walk-forward runs of the day-level deep model (owner: deep agent). On gene from ~/nyiso-us/model:

    ../.venv/bin/python train_deep_day.py features   day-feature matrix, bid days 2019-12-31 .. 2023-12-30
                                                     -> results/day_features_2020_2023.parquet
    ../.venv/bin/python train_deep_day.py wf         monthly refits 2021-01 .. 2023-12, both tasks
                                                     -> results/deep_day_wf_2021_2023.parquet and .json

Walk-forward: a refit on the 1st of each month, trained on delivery dates from 2020-01-01 up to two days
before the month whose prices were all public by 05:00 on the month's first bid day (walkforward.py's
rule, applied to whole days); storm threshold L = the 10th percentile of the training days' always-supply
P&L (refit by refit); weather inputs are absent, and masked, before 2021-03-25. Configuration fixed in
deep_day.CONFIG, 5 seeds.

Output (long): per scored delivery date one row zone = 'ALL' (p_storm; pred_zone_supply_pnl = the sum of
the zone predictions) and one row per zone (p_storm repeated; that zone's predicted supply P&L), with
bid_date, refit_month and storm_L. The JSON holds per refit L, training days, epochs and seconds, and per
year the AUC of p_storm against "book P&L below that refit's L" beside the hand-made storm score's AUC on
the same days and target (side/storm_audit_lib.py on gene, ~/nyiso-us/side_audit).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import deep
import deep_day as DY
import lock
import panel as P
import walkforward as W
from common import HOME, RESULTS, ZONES, decision_time

FEATS = RESULTS / "day_features_2020_2023.parquet"
OUT = RESULTS / "deep_day_wf_2021_2023"
SPAN = (dt.date(2021, 1, 1), dt.date(2023, 12, 31))
BID_DAYS = [dt.date(2019, 12, 31) + dt.timedelta(days=i) for i in range((dt.date(2023, 12, 30) - dt.date(2019, 12, 31)).days + 1)]


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def features():
    t0 = time.time()
    df = DY.build_day_features(BID_DAYS, P.LockedStore(), log=log)
    lock.assert_build_only(df["delivery_date"])
    df.to_parquet(FEATS)
    nfeat = len([c for c in df.columns if "__" in c and not c.startswith("y__")])
    log(f"day features: {len(df)} days x {nfeat} features, {time.time() - t0:.0f} s -> {FEATS}")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATS)
    lock.assert_build_only(df["delivery_date"])
    return df


def hand_score() -> pd.Series | None:
    """The hand-made storm score per delivery date (side/storm_audit_lib.py), or None if absent."""
    lib_dir = HOME / "side_audit"
    if not (lib_dir / "storm_audit_lib.py").exists():
        return None
    sys.path.insert(0, str(lib_dir))
    import storm_audit_lib as L
    px, lf, wx = L.load_prices(), L.load_lf(), L.load_wx()
    s, _ = L.storm_score(L.inputs(L.BID_DAYS, px, lf, wx))
    return s


def walk(df: pd.DataFrame, span=SPAN, seeds=DY.SEEDS, device=None, config=None, log=log):
    """Monthly refits over span on the day-feature matrix: (long predictions, refit records)."""
    rows, recs = [], []
    for ms in W.month_starts(*span):
        me = min((pd.Timestamp(ms) + pd.offsets.MonthEnd(0)).date(), span[1])
        cut = pd.Timestamp(ms - dt.timedelta(days=W.TRAIN_GAP_DAYS))
        known = decision_time(ms - dt.timedelta(days=1))
        tr = df[(df["delivery_date"] <= cut) & (df["y__label_published_at"] <= known) & (df["y__n_rows"] > 0)]
        te = df[(df["delivery_date"] >= pd.Timestamp(ms)) & (df["delivery_date"] <= pd.Timestamp(me))]
        assert tr["delivery_date"].max() < te["delivery_date"].min()
        ts = time.time()
        sm = DY.DeepDayModel("storm", config, seeds=seeds, device=device)
        sm.fit(tr)
        ps = sm.predict(te)
        zm = DY.DeepDayModel("zone", config, seeds=seeds, device=device)
        zm.fit(tr)
        pz = zm.predict(te)
        sec = round(time.time() - ts, 1)
        for i in te.index:
            base = {"delivery_date": te.at[i, "delivery_date"], "bid_date": te.at[i, "bid_date"],
                    "p_storm": float(ps[i]), "refit_month": str(ms), "storm_L": sm.L}
            rows.append({**base, "zone": "ALL", "pred_zone_supply_pnl": float(pz.loc[i].sum())})
            rows += [{**base, "zone": z, "pred_zone_supply_pnl": float(pz.at[i, z])} for z in ZONES]
        recs.append({"month": str(ms), "train_days": sm.info["n_days"], "train_first": sm.info["train_first"],
                     "train_last": sm.info["train_last"], "L": sm.L, "storm_rate_train": sm.info["storm_rate_train"],
                     "storm_best_epochs": [s["best_epoch"] for s in sm.info["seeds"]],
                     "zone_best_epochs": [s["best_epoch"] for s in zm.info["seeds"]], "seconds": sec})
        log(f"refit {ms}: {sm.info['n_days']} days to {sm.info['train_last']}, L {sm.L:,.0f}, "
            f"epochs storm {recs[-1]['storm_best_epochs']} zone {recs[-1]['zone_best_epochs']}, {sec} s")
    out = pd.DataFrame(rows)
    lock.assert_build_only(out["delivery_date"])
    return out, recs


def evaluate(out: pd.DataFrame, df: pd.DataFrame, hand: pd.Series | None) -> dict:
    """Per year: AUC of p_storm (and of the hand-made score on the same days) for 'book P&L below the
    refit's L'; correlation and MAE of the zone predictions against realised zone supply P&L."""
    day = out[out["zone"] == "ALL"].merge(df[["delivery_date", "y__book_supply_pnl"]], on="delivery_date")
    day["storm"] = (day["y__book_supply_pnl"] < day["storm_L"]).astype(float)
    day["hand"] = hand.reindex(pd.DatetimeIndex(day["delivery_date"])).to_numpy() if hand is not None else np.nan
    real = df.set_index("delivery_date")[[f"y__zone_supply_pnl__{DY._slug(z)}" for z in ZONES]]
    real.columns = ZONES
    real = real.stack()
    zd = out[out["zone"] != "ALL"].copy()
    zd["real"] = real.reindex(pd.MultiIndex.from_arrays([zd["delivery_date"], zd["zone"]])).to_numpy()
    ev = {}
    for y, g in day.groupby(day["delivery_date"].dt.year):
        ok = np.isfinite(g["hand"].to_numpy())
        z = zd[(zd["delivery_date"].dt.year == y) & zd["real"].notna()]
        ev[int(y)] = {"days": int(len(g)), "storm_days": int(g["storm"].sum()),
                      "auc_p_storm": deep.auc(g["p_storm"].to_numpy(), g["storm"].to_numpy()),
                      "hand_days": int(ok.sum()),
                      "auc_hand_score_same_days": deep.auc(g["hand"].to_numpy()[ok], g["storm"].to_numpy()[ok]),
                      "auc_p_storm_same_days": deep.auc(g["p_storm"].to_numpy()[ok], g["storm"].to_numpy()[ok]),
                      "zone_pred_corr": float(np.corrcoef(z["pred_zone_supply_pnl"], z["real"])[0, 1]) if len(z) > 2 else None,
                      "zone_mae": float(np.mean(np.abs(z["pred_zone_supply_pnl"] - z["real"]))),
                      "zone_mae_zero": float(np.mean(np.abs(z["real"])))}
    return ev


def wf(args):
    t0 = time.time()
    df = load_features()
    out, recs = walk(df, SPAN, args.seeds, args.device)
    out.to_parquet(OUT.with_suffix(".parquet"))
    ev = evaluate(out, df, hand_score())
    summ = {"span": [str(SPAN[0]), str(SPAN[1])], "config": DY.CONFIG, "seeds": list(args.seeds), "refits": recs,
            "seconds_total": round(time.time() - t0, 1), "seconds_per_refit_mean": float(np.mean([r["seconds"] for r in recs])),
            "eval_by_year": ev, "hand_score_source": "side_audit/storm_audit_lib.storm_score(inputs(...)) defaults"}
    OUT.with_suffix(".json").write_text(json.dumps(summ, indent=1, default=str))
    log("eval " + json.dumps({y: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for y, e in ev.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "wf"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (gene is shared)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    a.seeds = tuple(int(x) for x in a.seeds.split(","))
    log(f"train_deep_day {a.mode}")
    features() if a.mode == "features" else wf(a)


if __name__ == "__main__":
    main()
