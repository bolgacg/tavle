"""The 2023 rehearsal, version 2 (CONTRACT.md "Rehearsal"; OBJECTIVES addendum of 6 Oct, late evening),
and the prepared held-out run. 2023 is development data.

    python rehearsal.py gbm        panel, tuning on 2020-2022, pairs for C, walk-forward 2023 with monthly
                                   refits; writes results/rehearsal_2023_v2.json, model/spike_config.json
                                   and model/frozen_choices.json. results/rehearsal_2023.json (v1) is
                                   never written again.
    python rehearsal.py deep       the deep model inside B and D (outage features recounted for v2), then
                                   rescore
    python rehearsal.py rescore    every score again from cached predictions (no refit), including the
                                   deep agent's spike predictions once delivered (spike_config.json says how)
    python rehearsal.py heldout    THE HELD-OUT RUN. Only through heldout.sh, on the orchestrator's
                                   instruction; the lock refuses it otherwise.

Version 2 changes idea A only: a LightGBM classifier gives P(gap >= S) per zone-hour, and A supplies in
every zone-hour except where that probability exceeds p_star. S (25, 50 or 100 USD/MWh) and p_star
(0.02 or 0.05) are chosen together on 2022 by A's mean daily net P&L, with the classifier trained on
delivery dates up to 30 Dec 2021: 6 configurations. B, C and D are unchanged; the base regression that C
uses is still selected with the v1 A rule, so C's predictions and pairs are the v1 ones. For the rehearsal,
D's 20 outage sites are counted on snapshots up to 30 Dec 2022.
Every loader goes through lock.py; while locked nothing on or after 2024-01-01 is read.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import lock

import numpy as np
import pandas as pd

import context as CX
import fees
import gbm
import panel as P
import score as SC
import strategies as S
import walkforward as W

RESULTS = lock.STUDY_DIR / "results"
OUT = RESULTS / "rehearsal_2023_v2.json"
CACHE = P.CACHE
SPIKE_CONFIG = lock.MODEL_DIR / "spike_config.json"
CHOICES = lock.MODEL_DIR / "frozen_choices.json"
FIRST_DAY = dt.date(2020, 1, 1)
BUILD = (FIRST_DAY, dt.date(2023, 12, 31))
TUNE_VALID = (dt.date(2022, 1, 1), dt.date(2022, 12, 31))
REH = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))
HOLDOUT = (dt.date(2024, 1, 1), dt.date(2026, 9, 30))
B_START = dt.date(2021, 3, 25)               # weather archive starts (OBJECTIVES addendum)
SITES_THROUGH: dt.date | None = dt.date(2022, 12, 30)   # rehearsal; None = frozen 2020-2023 list (held-out)
READ_END: dt.date | None = None              # None = the lock's default (2024-01-01)
WORKERS = 4
HOLDOUT_DAYS = (HOLDOUT[1] - HOLDOUT[0]).days + 1     # a calendar count, no data
# "A" here is the v1 rule: it still selects the base regression whose predictions idea C uses.
RULES = {"A": S.idea_A_v1, "B": S.idea_B, "D": S.idea_D, "A_gen": S.idea_A_v1}
SETS = {"base": ("A", None), "weather": ("B", B_START), "outages": ("D", None)}
SPIKE_SETS = {"spike": "base", "spike_gen": "gen"}     # classifier name -> feature set
IDEAS = ("A", "B", "C", "D")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _sites_tag() -> str:
    return f"sites{SITES_THROUGH:%Y%m%d}" if SITES_THROUGH else "sitesfrozen"


# ----------------------------------------------------------------- data

def _cached(name: str, build):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        log(f"cache {path}")
        return P.read_locked(path, "delivery_hour", READ_END or lock.read_end())
    t0 = time.time()
    df = build()
    P.save_panel(df, path)
    log(f"built {name}: {len(df)} rows in {time.time() - t0:.0f} s")
    return df


def load_all(store: P.LockedStore) -> tuple[pd.DataFrame, dict]:
    lock.assert_build_only(list(BUILD))
    tag = "" if BUILD == (dt.date(2020, 1, 1), dt.date(2023, 12, 31)) else f"_{BUILD[1]:%Y%m%d}"
    panel = _cached(f"panel_base{tag}", lambda: P.build_panel(*BUILD, store=store, workers=WORKERS))
    panel = panel.reset_index(drop=True)
    cols = {"base": list(P.BASE_FEATURES)}
    key = ["delivery_hour", "zone"]

    gen = _cached(f"feat_gen{tag}", lambda: P.build_gen(*BUILD, store=store, log=log))
    panel = panel.merge(gen, on=key, how="left", validate="one_to_one")
    cols["gen"] = cols["base"] + list(P.GEN_FEATURES)

    def ext(fn, **kw):
        def b():
            f = fn(panel, store, **kw).add_prefix("x_")
            f[key] = panel[key]
            return f
        return b
    for name, fn, kw, cname in (("weather", S.weather_features, {}, f"feat_weather{tag}"),
                                ("outages", S.outage_features, {"sites_through": SITES_THROUGH},
                                 f"feat_outages_{_sites_tag()}{tag}")):
        try:
            f = _cached(cname, ext(fn, **kw))
        except Exception as e:                      # the idea is reported as not run, the rest goes on
            log(f"!! {name} features failed: {type(e).__name__}: {e}")
            cols[name] = None
            continue
        new = [c for c in f.columns if c not in key]
        panel = panel.merge(f, on=key, how="left", validate="one_to_one")
        cols[name] = cols["base"] + new
    lock.assert_build_only(panel["delivery_hour"])
    return panel, cols


def rows(panel: pd.DataFrame, lo: dt.date, hi: dt.date) -> pd.DataFrame:
    return panel[(panel["delivery_date"] >= pd.Timestamp(lo)) & (panel["delivery_date"] <= pd.Timestamp(hi))]


# ----------------------------------------------------------------- tuning

def tune(panel: pd.DataFrame, feats: list[str], idea: str, train_start: dt.date | None):
    """A regression set: 8 configurations, chosen on 2022 by the idea's own rule against the baseline. The
    2022 rows are those public by 05:00 on the scored period's first bid day (walkforward.choice_rows)."""
    va = W.check_choice_rows(W.choice_rows(rows(panel, *TUNE_VALID), REH[0]), REH[0], f"tune {idea}")
    tr = panel[W.train_mask(panel, TUNE_VALID[0], train_start)]
    days = pd.DatetimeIndex(sorted(va["delivery_date"].unique()))
    base = S.baseline(va)
    recs, best, best_pred = [], None, None
    for cfg in gbm.GRID:
        t0 = time.time()
        m = gbm.GBMModel(feats, cfg)
        m.fit(tr)
        p = m.predict(va)
        led = RULES[idea](va, p)
        diff = SC.daily(led, days) - SC.daily(base, days)
        r = {"config": gbm.config_name(cfg), **cfg, "mean_daily_diff": float(diff.mean()),
             "idea_pnl_usd": float(led["pnl"].sum()), "mwh": float(led["mwh"].sum()),
             "rmse": float(np.sqrt(np.nanmean((p - va["gap"]) ** 2))),
             "corr": float(pd.Series(p).corr(va["gap"])), "seconds": round(time.time() - t0, 1)}
        recs.append(r)
        log(f"  tune {idea} {r['config']}: diff/day {r['mean_daily_diff']:.1f}, rmse {r['rmse']:.2f}, {r['seconds']} s")
        if best is None or r["mean_daily_diff"] > best[0]:
            best, best_pred = (r["mean_daily_diff"], cfg), p
    return {"train_rows": int(len(tr)), "train_first": str(tr["delivery_date"].min().date()),
            "train_last": str(tr["delivery_date"].max().date()), "valid": [str(d) for d in TUNE_VALID],
            "criterion": "mean daily P&L minus baseline on 2022, the idea's own rule",
            "chosen": gbm.config_name(best[1]), "chosen_config": best[1], "configs": recs}, best[1], best_pred, va


def choose_p_star(va: pd.DataFrame, p: pd.Series, grid=gbm.P_STAR_GRID) -> tuple[float, list[dict]]:
    """p_star from the grid with the highest mean daily net P&L of idea A on these (2022) rows."""
    days = pd.DatetimeIndex(sorted(va["delivery_date"].unique()))
    n = int(va["gap"].notna().sum())
    recs = []
    for ps in grid:
        led = S.idea_A(va, p, ps)
        recs.append({"p_star": ps, "mean_daily_net": float(SC.daily(led, days).mean()),
                     "mwh": float(led["mwh"].sum()), "sat_out_share": 1 - float(led["mwh"].sum()) / n})
    best = max(recs, key=lambda r: r["mean_daily_net"])
    return best["p_star"], recs


def tune_spike(panel: pd.DataFrame, feats: list[str], S_grid=gbm.S_GRID, p_grid=gbm.P_STAR_GRID):
    """Idea A: classifier per S trained to 30 Dec 2021, scored on 2022 for every p_star (len(S_grid) x
    len(p_grid) configurations); the best mean daily net P&L wins."""
    va = W.check_choice_rows(W.choice_rows(rows(panel, *TUNE_VALID), REH[0]), REH[0], "tune spike")
    tr = panel[W.train_mask(panel, TUNE_VALID[0])]
    recs, best = [], None
    for s_ in S_grid:
        t0 = time.time()
        m = gbm.SpikeClassifier(feats, s_)
        m.fit(tr)
        p = m.predict(va)
        spike = (va["gap"] >= s_).astype(float).where(va["gap"].notna())
        brier = float(np.nanmean((p - spike) ** 2))
        _, sub = choose_p_star(va, p, p_grid)
        for r in sub:
            r.update({"S": s_, "train_base_rate": m.base_rate, "valid_base_rate": float(spike.mean()),
                      "brier": brier, "seconds": round(time.time() - t0, 1)})
            recs.append(r)
            log(f"  tune A spike S={s_} p*={r['p_star']}: net/day {r['mean_daily_net']:.1f}, "
                f"sat out {r['sat_out_share']:.3f}, base rate {m.base_rate:.4f}")
            if best is None or r["mean_daily_net"] > best[0]:
                best = (r["mean_daily_net"], s_, r["p_star"], p)
    info = {"train_rows": int(len(tr)), "train_last": str(tr["delivery_date"].max().date()),
            "valid": [str(d) for d in TUNE_VALID], "configurations": len(recs),
            "criterion": "idea A mean daily net P&L on 2022", "chosen_S": best[1], "chosen_p_star": best[2],
            "fixed_lightgbm": {**gbm.SPIKE_FIXED, "rounds": gbm.N_ROUNDS, "learning_rate": gbm.FIXED["learning_rate"]},
            "configs": recs}
    return info, best[1], best[2], best[3]


# ----------------------------------------------------------------- money and scores

def ledgers(test: pd.DataFrame, preds: dict, pairs: list[str], fee=None, spike: dict | None = None,
            a_v1: bool = True) -> dict[str, pd.DataFrame]:
    """Hourly ledgers. `spike` = {"spike": {"p", "p_star"}, "spike_gen": {...}} gives the v2 A (and its
    generator ablation); without it A follows the v1 rule on preds["base"] when a_v1 is True."""
    L = {"baseline": S.baseline(test, fee), "always_supply": S.always_supply(test, fee)}
    if spike and "spike" in spike:
        L["A"] = S.idea_A(test, spike["spike"]["p"], spike["spike"]["p_star"], fee)
    elif a_v1 and "base" in preds:
        L["A"] = S.idea_A_v1(test, preds["base"], fee)
    if "base" in preds:
        L["C"] = S.idea_C(test, preds["base"], [tuple(p.split("|")) for p in pairs], fee)
    if "weather" in preds:
        L["B"] = S.idea_B(test, preds["weather"], fee)
    if "outages" in preds:
        L["D"] = S.idea_D(test, preds["outages"], fee)
    if spike and "spike_gen" in spike:
        L["A_gen"] = S.idea_A(test, spike["spike_gen"]["p"], spike["spike_gen"]["p_star"], fee)
    elif a_v1 and "gen" in preds:
        L["A_gen"] = S.idea_A_v1(test, preds["gen"], fee)
    return L


def score_all(test: pd.DataFrame, preds: dict, pairs: list[str], model: str, spike: dict | None = None,
              a_v1: bool = True) -> dict:
    settled = test.loc[test["gap"].notna(), "delivery_date"]
    days = pd.DatetimeIndex(sorted(settled.unique()))            # a day with no settled price is not scored
    idx = SC.stationary_indices(len(days))
    cut = float(np.quantile(test["gap"].abs().dropna(), 0.99))
    L = ledgers(test, preds, pairs, spike=spike, a_v1=a_v1)
    pred_of = {"B": "weather", "D": "outages"}
    res = {}
    for k in IDEAS:
        if k not in L:
            continue
        if k == "A" and spike and "spike" in spike:
            pr = spike["spike"]["p"]
            gp = (test["gap"] >= spike["spike"]["S"]).astype(float).where(test["gap"].notna())
        elif k == "A":
            pr, gp = preds.get("base"), test["gap"]
        elif k in pred_of:
            pr, gp = preds[pred_of[k]], test["gap"]
        else:
            pr, gp = None, None
        res[k] = SC.summarize(k, L[k], L["baseline"], days, idx, extreme_cut=cut, pred=pr, gap=gp)
    SC.apply_holm(res)
    for k in IDEAS:
        if k not in res:
            res[k] = {"verdict": "not run"}
    ctx = CX.block({k: v for k, v in L.items()}, days, idx)
    out = {"model": model, "scored_days": int(len(days)),
           "days_without_settled_prices": int(test["delivery_date"].nunique() - len(days)),
           "extreme_cut_abs_gap_usd": cut,
           "baseline": {**SC.money(L["baseline"]), "mean_daily": float(SC.daily(L["baseline"], days).mean())},
           "always_supply": {**SC.money(L["always_supply"]), "mean_daily": float(SC.daily(L["always_supply"], days).mean())},
           "ideas": res, "context": ctx}
    if spike and "spike" in spike and "A" in L:
        n = int(test["gap"].notna().sum())
        res["A"]["rule"] = {"S": spike["spike"]["S"], "p_star": spike["spike"]["p_star"],
                            "sat_out_share": 1 - float(L["A"]["mwh"].sum()) / n}
        out["value_of_the_spike_filter"] = {
            "what": "idea A minus always-supply (every zone-hour supply, same fees), daily net P&L",
            **SC.compare(L["A"], L["always_supply"], days, idx)}
    stress = {}
    for f in fees.STRESS:
        Ls = ledgers(test, preds, pairs, fee=f, spike=spike, a_v1=a_v1)
        stress[str(f)] = {k: SC.stress(k, Ls[k], Ls["baseline"], days, idx) for k in Ls if k != "baseline"}
    out["fee_stress"] = stress
    if "A_gen" in L and "A" in L:
        out["ablation_gen"] = {
            "what": "idea A with base + zone-plus-generator summary features",
            "vs_baseline": SC.summarize("A_gen", L["A_gen"], L["baseline"], days, idx, extreme_cut=cut),
            "vs_A_base": SC.deep_vs_gbm(L["A_gen"], L["A"], days, idx)}
        if spike and "spike_gen" in spike:
            out["ablation_gen"]["p_star"] = spike["spike_gen"]["p_star"]
    out["mde"] = {"power": SC.POWER, "holdout_days": HOLDOUT_DAYS, "note": "projection assumes this period's daily variance",
                  **{k: {"mde_usd_per_day": r.get("mde_80pct_power"), "se_usd_per_day": r.get("se_daily_diff"),
                         "projected_holdout_mde": (r["mde_80pct_power"] * np.sqrt(len(days) / HOLDOUT_DAYS)
                                                   if r.get("mde_80pct_power") is not None else None)}
                     for k, r in res.items()}}
    return out


# ----------------------------------------------------------------- the gbm stage

def _preds_path(model: str) -> Path:
    return CACHE / f"preds_{model}_{REH[0]:%Y}_v2.parquet"


def stage_gbm(store: P.LockedStore | None = None):
    t_start = time.time()
    store = store or P.LockedStore()
    log(f"lock: read end {store.end}; {lock.unlock_status()[1]}")
    panel, cols = load_all(store)
    log(f"panel {len(panel)} rows, {panel['delivery_date'].nunique()} days, "
        f"{panel['delivery_date'].min().date()}..{panel['delivery_date'].max().date()}")
    out = {"version": "v2", "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "status": "2023 is development data; build years only; no date on or after 2024-01-01 was read",
           "v1": "results/rehearsal_2023.json, unchanged", "lock": lock.unlock_status()[1],
           "fees": {"usd_per_mwh": fees.RATES, "source": fees.SOURCE},
           "outage_sites_counted_through": str(SITES_THROUGH),
           "panel": {"rows": int(len(panel)), "days": int(panel["delivery_date"].nunique()),
                     "first": str(panel["delivery_date"].min().date()), "last": str(panel["delivery_date"].max().date()),
                     "feature_sets": {k: (len(v) if v else None) for k, v in cols.items()},
                     "gap_missing_rows": int(panel["gap"].isna().sum())},
           "tuning": {}, "walk_forward": {}, "importance": {}}

    log("tuning the spike classifier for idea A")
    info, s_, pstar, _ = tune_spike(panel, cols["base"])
    out["tuning"]["spike"] = info
    spike_cfg = {"S_usd_per_mwh": s_, "spike": "gap = rt_lbmp - da_lbmp >= S",
                 "S_grid": list(gbm.S_GRID), "p_star_grid": list(gbm.P_STAR_GRID),
                 "chosen_on": "2022, idea A mean daily net P&L, gradient-boosting classifier trained on delivery "
                              "dates up to 2021-12-30 (rehearsal.py tune_spike)",
                 "gbm_p_star": pstar,
                 "deep_delivery": {
                     "predictions_2023": "results/deep_spike_2023.parquet: delivery_hour, zone, p_spike "
                                         "(walk-forward 2023, monthly refits, same S)",
                     "choice": "results/deep_spike_choice.json: {\"S\": S, \"p_star\": chosen from p_star_grid on "
                               "2022 by idea A mean daily net P&L}, or results/deep_spike_2022.parquet "
                               "(delivery_hour, zone, p_spike; trained to 2021-12-30) and rescore chooses it"}}
    SPIKE_CONFIG.write_text(json.dumps(spike_cfg, indent=1) + "\n")
    log(f"spike: S {s_}, p* {pstar}; wrote {SPIKE_CONFIG}")
    chosen = {}
    for fs, (idea, start) in SETS.items():
        if cols.get(fs) is None:
            out["tuning"][fs] = "not run: features missing"
            continue
        log(f"tuning {fs} (regression) with the {idea}{' v1' if idea == 'A' else ''} rule")
        info, cfg, vpred, va = tune(panel, cols[fs], idea, start)
        out["tuning"][fs] = info
        chosen[fs] = cfg
        if fs == "base":
            pairs, table = S.choose_pairs(va, vpred)
            out["pairs"] = {"chosen": ["|".join(p) for p in pairs], "chosen_on": "2022 predictions of the tuned "
                            "base regression trained to 30 Dec 2021", "table": table}
            log(f"pairs for C: {pairs}")

    pstar_gen = None
    if cols.get("gen"):
        info_g, _, pstar_gen, _ = tune_spike(panel, cols["gen"], S_grid=(s_,))
        out["tuning"]["spike_gen"] = info_g

    preds = {}
    for fs, cfg in chosen.items():
        _, start = SETS[fs]
        log(f"walk-forward {REH[0].year}: {fs} ({gbm.config_name(cfg)})")
        last = {}

        def make(fs=fs, cfg=cfg):
            last["m"] = gbm.GBMModel(cols[fs], cfg)
            return last["m"]
        preds[fs], out["walk_forward"][fs] = W.walk_forward(panel, make, *REH, train_start=start, log=log)
        out["importance"][fs] = dict(list(last["m"].importance().items())[:15])
    for name, fs in SPIKE_SETS.items():
        if not cols.get(fs) or (name == "spike_gen" and pstar_gen is None):
            continue
        log(f"walk-forward {REH[0].year}: {name} (classifier, S {s_})")
        last = {}

        def make(fs=fs):
            last["m"] = gbm.SpikeClassifier(cols[fs], s_)
            return last["m"]
        preds[name], out["walk_forward"][name] = W.walk_forward(panel, make, *REH, log=log)
        out["importance"][name] = dict(list(last["m"].importance().items())[:15])
    test = rows(panel, *REH)
    pd.DataFrame(preds).assign(delivery_hour=test["delivery_hour"], zone=test["zone"]).to_parquet(_preds_path("gbm"))

    spike = {"spike": {"p": preds["spike"], "p_star": pstar, "S": s_}}
    if "spike_gen" in preds:
        spike["spike_gen"] = {"p": preds["spike_gen"], "p_star": pstar_gen, "S": s_}
    out["spike_rule"] = {"S": s_, "p_star": pstar, "p_star_gen": pstar_gen}
    out.update(score_all(test, preds, out.get("pairs", {}).get("chosen", []), "gbm", spike=spike, a_v1=False))
    out["seconds"] = round(time.time() - t_start)
    write(out)
    write_choices(out)
    log(f"done in {out['seconds']} s")
    return out


def write_choices(out: dict) -> None:
    """Everything the held-out run needs that was chosen on build years, in one frozen file."""
    ch = {"source": f"results/{OUT.name}", "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
          "regression_configs": {fs: t["chosen_config"] for fs, t in out["tuning"].items()
                                 if isinstance(t, dict) and "chosen_config" in t},
          "train_start": {fs: (str(st) if st else None) for fs, (_, st) in SETS.items()},
          "spike": out["spike_rule"], "pairs": out.get("pairs", {}).get("chosen", []),
          "outage_sites": "features_outages.TOP_SITES (counted on 2020 to 2023 snapshots)"}
    CHOICES.write_text(json.dumps(ch, indent=1, default=str) + "\n")
    log(f"wrote {CHOICES}")


# ----------------------------------------------------------------- deep and rescore

def cached_deep_preds(test: pd.DataFrame, panel: pd.DataFrame | None = None) -> tuple[dict, dict, dict | None]:
    """Deep predictions on disk: regressions for B, C, D and, once the deep agent delivers them, spike
    probabilities for A (spike_config.json, deep_delivery)."""
    key = ["delivery_hour", "zone"]
    out, source = {}, {}
    files = {"base": [RESULTS / "deep_rehearsal_2023.parquet", CACHE / "preds_deep_base_2023.parquet"],
             "weather": [CACHE / "preds_deep_weather_2023.parquet"],
             "outages": [CACHE / f"preds_deep_outages_{_sites_tag()}_2023.parquet"]}

    def take(f, col):
        d = P.read_locked(f, "delivery_hour", READ_END or lock.read_end())
        m = test[key].merge(d[key + [col]], on=key, how="left", validate="one_to_one")
        return pd.Series(m[col].to_numpy(), index=test.index) if m[col].notna().all() else None

    for fs, cands in files.items():
        for f in cands:
            if f.exists() and (s := take(f, "pred_gap")) is not None:
                out[fs], source[fs] = s, f.name
                break
    spike = None
    f23 = next((f for f in (RESULTS / "deep_spike_2023.parquet", RESULTS / "deep_spike_rehearsal_2023.parquet")
                if f.exists()), None)
    if f23 is not None and SPIKE_CONFIG.exists():
        S_ = json.loads(SPIKE_CONFIG.read_text())["S_usd_per_mwh"]
        thr = pd.read_parquet(f23, columns=["spike_threshold"])["spike_threshold"] if "spike_threshold" in \
            pd.read_parquet(f23).columns else None
        if thr is not None and not (thr == S_).all():
            log(f"!! {f23.name} uses S {sorted(thr.unique())}, spike_config says {S_}: deep A left pending")
            return out, {**source, "spike": f"pending: deep S differs from {S_}"}, None
        p = take(f23, "p_spike")
        choice, f22 = RESULTS / "deep_spike_choice.json", RESULTS / "deep_spike_2022.parquet"
        pstar, how = None, None
        if choice.exists():
            c = json.loads(choice.read_text())
            if c.get("S") == S_:
                pstar, how = c["p_star"], choice.name
        elif f22.exists() and panel is not None:
            va = W.check_choice_rows(W.choice_rows(rows(panel, *TUNE_VALID), REH[0]), REH[0], "deep p*")
            p22 = take_rows(va, f22)
            if p22 is not None:
                pstar, _ = choose_p_star(va, p22)
                how = f"chosen here on {f22.name}"
        if p is not None and pstar is not None:
            spike = {"spike": {"p": p, "p_star": pstar, "S": S_}}
            source["spike"] = f"{f23.name}; p_star {pstar} ({how})"
    return out, source, spike


def take_rows(rws: pd.DataFrame, f: Path) -> pd.Series | None:
    key = ["delivery_hour", "zone"]
    d = P.read_locked(f, "delivery_hour", READ_END or lock.read_end())
    m = rws[key].merge(d[key + ["p_spike"]], on=key, how="left", validate="one_to_one")
    return pd.Series(m["p_spike"].to_numpy(), index=rws.index) if m["p_spike"].notna().all() else None


def stage_deep(config: str | None = None, store: P.LockedStore | None = None, sets=("outages",)):
    """Deep regressions inside B and D that are not cached yet (v2: D, with the recounted outage sites),
    then rescore. Base (C) comes from the deep agent's own rehearsal; A's deep spike model from its delivery."""
    import deep as DP
    store = store or P.LockedStore()
    panel, cols = load_all(store)
    test = rows(panel, *REH)
    key = ["delivery_hour", "zone"]
    tuned = RESULTS / "deep_tune_2022.json"
    if config is None and tuned.exists():
        config = json.loads(tuned.read_text()).get("selected")
    config = config or "c1"
    have, _, _ = cached_deep_preds(test)
    extra = {"weather": [c for c in (cols.get("weather") or []) if c.startswith("x_")],
             "outages": [c for c in (cols.get("outages") or []) if c.startswith("x_")]}
    need = [fs for fs in sets if fs not in have and extra.get(fs)]
    if need:
        DP.preload(REH[1], log=log)
    for fs in need:
        _, start = SETS[fs]
        log(f"deep walk-forward {REH[0].year}: {fs} ({len(extra[fs])} extra columns, config {config})")
        p, _ = W.walk_forward(panel, lambda ex=extra[fs]: DP.DeepModel(config, extra_cols=ex, log=log), *REH,
                              train_start=start, log=log)
        name = f"preds_deep_{fs}_{_sites_tag()}_2023.parquet" if fs == "outages" else f"preds_deep_{fs}_2023.parquet"
        pd.DataFrame({"pred_gap": p}).join(test[key]).to_parquet(CACHE / name)
    res = json.loads(OUT.read_text())
    res.setdefault("deep", {})["config"] = config
    write(res)
    stage_rescore(store)


def stage_rescore(store: P.LockedStore | None = None):
    """Every score again from cached predictions (no refit); keeps tuning and walk-forward records."""
    store = store or P.LockedStore()
    panel, _ = load_all(store)
    test = rows(panel, *REH)
    key = ["delivery_hour", "zone"]
    res = json.loads(OUT.read_text())
    pairs = res["pairs"]["chosen"]
    gp = P.read_locked(_preds_path("gbm"), "delivery_hour", READ_END or lock.read_end())
    gm = test[key].merge(gp, on=key, how="left", validate="one_to_one")
    g = {c: pd.Series(gm[c].to_numpy(), index=test.index) for c in gm.columns if c not in key}
    sr = res["spike_rule"]
    spike = {"spike": {"p": g.pop("spike"), "p_star": sr["p_star"], "S": sr["S"]}}
    if "spike_gen" in g:
        spike["spike_gen"] = {"p": g.pop("spike_gen"), "p_star": sr["p_star_gen"], "S": sr["S"]}
    res.update(SC.strip(score_all(test, g, pairs, "gbm", spike=spike, a_v1=False)))
    dpreds, source, dspike = cached_deep_preds(test, panel)
    if dpreds or dspike:
        days = pd.DatetimeIndex(sorted(test.loc[test["gap"].notna(), "delivery_date"].unique()))
        idx = SC.stationary_indices(len(days))
        Lg = ledgers(test, g, pairs, spike=spike, a_v1=False)
        Ld = ledgers(test, dpreds, pairs, spike=dspike, a_v1=False)
        d = res.setdefault("deep", {})
        d.update({"source": source,
                  "scores": SC.strip(score_all(test, dpreds, pairs, f"deep_{d.get('config', '')}", spike=dspike, a_v1=False)),
                  "deep_vs_gbm": {k: SC.deep_vs_gbm(Ld[k], Lg[k], days, idx) for k in IDEAS if k in Ld and k in Lg}})
        d["pending"] = [k for k in IDEAS if k not in d["deep_vs_gbm"]]
    res["rescored"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write(res)


# ----------------------------------------------------------------- the held-out run (prepared, not run)

def stage_heldout():
    """The single held-out run (OBJECTIVES item 4): January 2024 to September 2026, each month predicted
    from data up to two days earlier, with every choice read from frozen_choices.json. The lock refuses
    it unless research/us/FREEZE names a commit, this checkout's model and pipeline equal that commit,
    and US_HOLDOUT_RUN=1 (heldout.sh arranges all three on a clean clone)."""
    global BUILD, READ_END, SITES_THROUGH, OUT, REH, HOLDOUT, RESULTS
    import heldout_mode as HM
    if HM.mode() == "dryrun":                                     # the same path on 2023 (heldout_mode.py)
        HOLDOUT, RESULTS = HM.window(), HM.run_dir() / "results"
        if CACHE.resolve() == (Path.home() / "nyiso-us" / "cache").resolve():
            raise HM.ModeError("set US_CACHE_DIR: the dry run never writes the build cache")
        RESULTS.mkdir(parents=True, exist_ok=True)
    lock.assert_build_only([HOLDOUT[1]])                          # raises while locked
    ch = json.loads(CHOICES.read_text())
    commit = lock.freeze_hash() or "nofreeze"                     # only a synthetic test lacks FREEZE here
    BUILD, READ_END, SITES_THROUGH, REH = (FIRST_DAY, HOLDOUT[1]), HOLDOUT[1] + dt.timedelta(days=1), None, HOLDOUT
    OUT = RESULTS / f"heldout_{HOLDOUT[0]:%Y%m}_{HOLDOUT[1]:%Y%m}_{commit[:12]}_{dt.datetime.now():%Y%m%dT%H%M}.json"
    store = P.LockedStore(end=READ_END)
    panel, cols = load_all(store)
    test = rows(panel, *HOLDOUT)
    out = {"run": "held-out", "freeze_commit": commit, "lock": lock.unlock_status()[1], "choices": ch,
           "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "fees": {"usd_per_mwh": fees.RATES, "source": fees.SOURCE}, "walk_forward": {}}
    preds = {}
    for fs, cfg in ch["regression_configs"].items():
        if fs not in SETS or cols.get(fs) is None:
            continue
        st = ch["train_start"].get(fs)
        start = dt.date.fromisoformat(st) if st else None
        preds[fs], out["walk_forward"][fs] = W.walk_forward(
            panel, lambda fs=fs, cfg=cfg: gbm.GBMModel(cols[fs], cfg), *HOLDOUT, train_start=start, log=log)
    sp = ch["spike"]
    spike = {}
    for name, fs, ps in (("spike", "base", sp["p_star"]), ("spike_gen", "gen", sp.get("p_star_gen"))):
        if ps is None or not cols.get(fs):
            continue
        p, out["walk_forward"][name] = W.walk_forward(panel, lambda fs=fs: gbm.SpikeClassifier(cols[fs], sp["S"]),
                                                      *HOLDOUT, log=log)
        spike[name] = {"p": p, "p_star": ps, "S": sp["S"]}
    pd.DataFrame({**preds, **{k: v["p"] for k, v in spike.items()}}).assign(
        delivery_hour=test["delivery_hour"], zone=test["zone"]).to_parquet(OUT.with_suffix(".preds.parquet"))
    out.update(score_all(test, preds, ch["pairs"], "gbm", spike=spike, a_v1=False))
    write(out)
    log(f"held-out run written to {OUT}")


def write(out: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(SC.strip(out), indent=1, default=str))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "gbm"
    if stage == "gbm":
        stage_gbm()
    elif stage == "rescore":
        stage_rescore()
    elif stage == "deep":
        stage_deep(None, sets=tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("outages",))
    elif stage == "heldout":
        stage_heldout()
    else:
        raise SystemExit("usage: rehearsal.py gbm | deep [outages,weather] | rescore | heldout")
