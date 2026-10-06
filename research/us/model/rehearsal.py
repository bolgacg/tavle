"""The 2023 rehearsal (CONTRACT.md, "Rehearsal"). 2023 is development data.

    python rehearsal.py gbm      panel, tuning on 2020-2022, pairs for C, walk-forward 2023 with monthly
                                 refits for every idea, scores written to results/rehearsal_2023.json
    python rehearsal.py deep     the deep model inside every idea (needs the gbm stage's cache), adds the
                                 deep-against-gradient-boosting comparison to the same JSON

Tuning: each feature set's LightGBM configuration (gbm.GRID, 8) is fitted once on delivery dates up to
30 Dec 2021 (labels public by 05:00 on 31 Dec 2021) and scored on 2022 by its own idea's rule: the
chosen one has the highest mean daily P&L minus the baseline. C's pairs are chosen on the same 2022
predictions of the base model. Nothing from 2023 enters tuning or the pair choice.
Every loader goes through lock.py; nothing on or after 2024-01-01 is read.
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

import fees
import gbm
import panel as P
import score as SC
import strategies as S
import walkforward as W

RESULTS = lock.STUDY_DIR / "results"
OUT = RESULTS / "rehearsal_2023.json"
CACHE = P.CACHE
BUILD = (dt.date(2020, 1, 1), dt.date(2023, 12, 31))
TUNE_VALID = (dt.date(2022, 1, 1), dt.date(2022, 12, 31))
REH = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))
B_START = dt.date(2021, 3, 25)               # weather archive starts (OBJECTIVES addendum)
WORKERS = 4
HOLDOUT_DAYS = (dt.date(2026, 9, 30) - dt.date(2024, 1, 1)).days + 1   # a calendar count, no data
RULES = {"A": S.idea_A, "B": S.idea_B, "D": S.idea_D, "A_gen": S.idea_A}
SETS = {"base": ("A", None), "gen": ("A_gen", None), "weather": ("B", B_START), "outages": ("D", None)}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------- data

def _cached(name: str, build):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        log(f"cache {path}")
        return P.read_locked(path, "delivery_hour", lock.read_end())
    t0 = time.time()
    df = build()
    P.save_panel(df, path)
    log(f"built {name}: {len(df)} rows in {time.time() - t0:.0f} s")
    return df


def load_all(store: P.LockedStore) -> tuple[pd.DataFrame, dict]:
    lock.assert_build_only(list(BUILD))
    panel = _cached("panel_base", lambda: P.build_panel(*BUILD, store=store, workers=WORKERS))
    panel = panel.reset_index(drop=True)
    cols = {"base": list(P.BASE_FEATURES)}
    key = ["delivery_hour", "zone"]

    gen = _cached("feat_gen", lambda: P.build_gen(*BUILD, store=store, log=log))
    panel = panel.merge(gen, on=key, how="left", validate="one_to_one")
    cols["gen"] = cols["base"] + list(P.GEN_FEATURES)

    def ext(fn):
        def b():
            f = fn(panel, store)
            f = f.add_prefix("x_")
            f[key] = panel[key]
            return f
        return b
    for name, fn in (("weather", S.weather_features), ("outages", S.outage_features)):
        try:
            f = _cached(f"feat_{name}", ext(fn))
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
    va = rows(panel, *TUNE_VALID)
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


# ----------------------------------------------------------------- the gbm stage

def stage_gbm(store: P.LockedStore | None = None):
    t_start = time.time()
    store = store or P.LockedStore()
    log(f"lock: read end {store.end}; {lock.unlock_status()[1]}")
    panel, cols = load_all(store)
    log(f"panel {len(panel)} rows, {panel['delivery_date'].nunique()} days, "
        f"{panel['delivery_date'].min().date()}..{panel['delivery_date'].max().date()}")
    out = {"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "status": "2023 is development data; build years only; no date on or after 2024-01-01 was read",
           "lock": lock.unlock_status()[1], "fees": {"usd_per_mwh": fees.RATES, "source": fees.SOURCE},
           "panel": {"rows": int(len(panel)), "days": int(panel["delivery_date"].nunique()),
                     "first": str(panel["delivery_date"].min().date()), "last": str(panel["delivery_date"].max().date()),
                     "feature_sets": {k: (len(v) if v else None) for k, v in cols.items()},
                     "gap_missing_rows": int(panel["gap"].isna().sum())},
           "tuning": {}, "walk_forward": {}, "importance": {}}

    preds, chosen = {}, {}
    for fs, (idea, start) in SETS.items():
        if cols.get(fs) is None:
            out["tuning"][fs] = "not run: features missing"
            continue
        log(f"tuning {fs} for idea {idea}")
        info, cfg, vpred, va = tune(panel, cols[fs], idea, start)
        out["tuning"][fs] = info
        chosen[fs] = cfg
        if fs == "base":
            pairs, table = S.choose_pairs(va, vpred)
            out["pairs"] = {"chosen": ["|".join(p) for p in pairs], "chosen_on": "2022 predictions of the tuned "
                            "base model trained to 30 Dec 2021", "table": table}
            log(f"pairs for C: {pairs}")

    for fs, cfg in chosen.items():
        idea, start = SETS[fs]
        log(f"walk-forward 2023: {fs} ({gbm.config_name(cfg)})")
        last = {}

        def make(fs=fs, cfg=cfg):
            m = gbm.GBMModel(cols[fs], cfg)
            last["m"] = m
            return m
        p, fits = W.walk_forward(panel, make, *REH, train_start=start, log=log)
        preds[fs] = p
        out["walk_forward"][fs] = fits
        out["importance"][fs] = dict(list(last["m"].importance().items())[:15])
    pd.DataFrame({fs: p for fs, p in preds.items()}).assign(
        delivery_hour=rows(panel, *REH)["delivery_hour"], zone=rows(panel, *REH)["zone"]).to_parquet(
        CACHE / "preds_gbm_2023.parquet")

    test = rows(panel, *REH)
    out.update(score_all(test, preds, out.get("pairs", {}).get("chosen", []), "gbm"))
    out["seconds"] = round(time.time() - t_start)
    write(out)
    log(f"done in {out['seconds']} s")
    return out


def ledgers(test: pd.DataFrame, preds: dict, pairs: list[str], fee=None) -> dict[str, pd.DataFrame]:
    L = {"baseline": S.baseline(test, fee)}
    if "base" in preds:
        L["A"] = S.idea_A(test, preds["base"], fee)
        L["C"] = S.idea_C(test, preds["base"], [tuple(p.split("|")) for p in pairs], fee)
    if "weather" in preds:
        L["B"] = S.idea_B(test, preds["weather"], fee)
    if "outages" in preds:
        L["D"] = S.idea_D(test, preds["outages"], fee)
    if "gen" in preds:
        L["A_gen"] = S.idea_A(test, preds["gen"], fee)
    return L


def score_all(test: pd.DataFrame, preds: dict, pairs: list[str], model: str) -> dict:
    days = pd.DatetimeIndex(sorted(test["delivery_date"].unique()))
    idx = SC.stationary_indices(len(days))
    cut = float(np.quantile(test["gap"].abs().dropna(), 0.99))
    L = ledgers(test, preds, pairs)
    pred_of = {"A": "base", "C": "base", "B": "weather", "D": "outages", "A_gen": "gen"}
    res = {}
    for k in ("A", "B", "C", "D"):
        if k not in L:
            continue
        pr = preds[pred_of[k]] if k != "C" else None
        res[k] = SC.summarize(k, L[k], L["baseline"], days, idx, extreme_cut=cut, pred=pr,
                              gap=test["gap"] if pr is not None else None)
    SC.apply_holm(res)
    for k in ("B", "C", "D"):
        if k not in res:
            res[k] = {"verdict": "not run"}
    stress = {}
    for f in fees.STRESS:
        Ls = ledgers(test, preds, pairs, fee=f)
        stress[str(f)] = {k: SC.stress(k, Ls[k], Ls["baseline"], days, idx) for k in Ls if k != "baseline"}
    extra = {}
    if "A_gen" in L:
        extra["ablation_gen"] = {
            "what": "idea A rule with base + zone-plus-generator summary features",
            "vs_baseline": SC.summarize("A_gen", L["A_gen"], L["baseline"], days, idx, extreme_cut=cut,
                                        pred=preds["gen"], gap=test["gap"]),
            "vs_A_base": SC.deep_vs_gbm(L["A_gen"], L["A"], days, idx)}
    base_money = SC.money(L["baseline"])
    mdes = {k: {"mde_usd_per_day_2023": r.get("mde_80pct_power"),
                "se_usd_per_day_2023": r.get("se_daily_diff"),
                "projected_holdout_mde": (r["mde_80pct_power"] * np.sqrt(len(days) / HOLDOUT_DAYS)
                                          if r.get("mde_80pct_power") is not None else None)}
            for k, r in res.items()}
    return {"model": model, "scored_days": int(len(days)), "extreme_cut_abs_gap_usd": cut,
            "baseline": {**base_money, "mean_daily": float(SC.daily(L["baseline"], days).mean())},
            "ideas": res, "fee_stress": stress, "mde": {"power": SC.POWER, "holdout_days": HOLDOUT_DAYS,
                                                        "note": "projection assumes 2023's daily variance", **mdes},
            **extra}


# ----------------------------------------------------------------- the deep stage

def stage_deep(config: str | None = None, store: P.LockedStore | None = None, sets=("base", "weather", "outages")):
    """Deep model inside every idea. Base predictions (ideas A and C) are taken from the deep agent's
    own rehearsal (results/deep_rehearsal_2023.parquet, train_deep.py) when it exists; B and D need the
    weather or outage columns as extra inputs and are walked forward here with the same configuration."""
    import deep as DP
    store = store or P.LockedStore()
    panel, cols = load_all(store)
    test = rows(panel, *REH)
    key = ["delivery_hour", "zone"]
    tuned = RESULTS / "deep_tune_2022.json"
    if config is None and tuned.exists():
        config = json.loads(tuned.read_text()).get("selected")
    config = config or "c1"
    log(f"deep config {config}")
    dpreds, fits, source = {}, {}, {}
    dr = RESULTS / "deep_rehearsal_2023.parquet"
    if "base" in sets and dr.exists():
        d = P.read_locked(dr, "delivery_hour", lock.read_end())
        m = test[key].merge(d[key + ["pred_gap"]], on=key, how="left", validate="one_to_one")
        if m["pred_gap"].notna().all():
            dpreds["base"] = pd.Series(m["pred_gap"].to_numpy(), index=test.index)
            source["base"] = f"{dr.name} (train_deep.py rehearsal)"
            rj = RESULTS / "deep_rehearsal_2023.json"
            fits["base"] = json.loads(rj.read_text()) if rj.exists() else None
        else:
            log(f"!! {dr.name} lacks {int(m['pred_gap'].isna().sum())} test rows; refitting base here")
    extra = {"base": [], "weather": [c for c in (cols.get("weather") or []) if c.startswith("x_")],
             "outages": [c for c in (cols.get("outages") or []) if c.startswith("x_")]}
    need = [fs for fs in sets if fs not in dpreds and (fs == "base" or extra[fs])]
    if need:
        DP.preload(REH[1], log=log)
    for fs in need:
        idea, start = SETS[fs]
        log(f"deep walk-forward 2023: {fs} ({len(extra[fs])} extra columns)")
        p, f = W.walk_forward(panel, lambda ex=extra[fs]: DP.DeepModel(config, extra_cols=ex, log=log), *REH,
                              train_start=start, log=log)
        dpreds[fs], fits[fs], source[fs] = p, f, "rehearsal.py deep"
        pd.DataFrame({"pred_gap": p}).join(test[key]).to_parquet(CACHE / f"preds_deep_{fs}_2023.parquet")
    gp = P.read_locked(CACHE / "preds_gbm_2023.parquet", "delivery_hour", lock.read_end())
    gm = test[key].merge(gp, on=key, how="left", validate="one_to_one")
    g = {fs: pd.Series(gm[fs].to_numpy(), index=test.index) for fs in ("base", "weather", "outages") if fs in gm}
    res = json.loads(OUT.read_text())
    pairs = res["pairs"]["chosen"]
    deep_scores = score_all(test, dpreds, pairs, f"deep_{config}")
    days = pd.DatetimeIndex(sorted(test["delivery_date"].unique()))
    idx = SC.stationary_indices(len(days))
    Lg, Ld = ledgers(test, g, pairs), ledgers(test, dpreds, pairs)
    comp = {k: SC.deep_vs_gbm(Ld[k], Lg[k], days, idx) for k in ("A", "B", "C", "D") if k in Ld and k in Lg}
    res["deep"] = {"config": config, "source": source, "walk_forward": fits, "scores": SC.strip(deep_scores),
                   "deep_vs_gbm": comp,
                   "not_run": [k for k in ("A", "B", "C", "D") if k not in comp]}
    write(res)
    log("deep stage written")


def cached_deep_preds(test: pd.DataFrame) -> tuple[dict, dict]:
    """Deep predictions already on disk: the deep agent's base rehearsal and this file's B/D runs."""
    key = ["delivery_hour", "zone"]
    out, source = {}, {}
    files = {"base": [RESULTS / "deep_rehearsal_2023.parquet", CACHE / "preds_deep_base_2023.parquet"],
             "weather": [CACHE / "preds_deep_weather_2023.parquet"],
             "outages": [CACHE / "preds_deep_outages_2023.parquet"]}
    for fs, cands in files.items():
        for f in cands:
            if f.exists():
                d = P.read_locked(f, "delivery_hour", lock.read_end())
                m = test[key].merge(d[key + ["pred_gap"]], on=key, how="left", validate="one_to_one")
                if m["pred_gap"].notna().all():
                    out[fs], source[fs] = pd.Series(m["pred_gap"].to_numpy(), index=test.index), f.name
                    break
    return out, source


def stage_rescore(store: P.LockedStore | None = None):
    """Recompute every score from cached predictions (no refit); keeps tuning and walk-forward records."""
    store = store or P.LockedStore()
    panel, _ = load_all(store)
    test = rows(panel, *REH)
    key = ["delivery_hour", "zone"]
    res = json.loads(OUT.read_text())
    pairs = res["pairs"]["chosen"]
    gp = P.read_locked(CACHE / "preds_gbm_2023.parquet", "delivery_hour", lock.read_end())
    gm = test[key].merge(gp, on=key, how="left", validate="one_to_one")
    g = {fs: pd.Series(gm[fs].to_numpy(), index=test.index) for fs in SETS if fs in gm}
    res.update(SC.strip(score_all(test, g, pairs, "gbm")))
    if "deep" in res:
        dpreds, source = cached_deep_preds(test)
        days = pd.DatetimeIndex(sorted(test["delivery_date"].unique()))
        idx = SC.stationary_indices(len(days))
        Lg, Ld = ledgers(test, g, pairs), ledgers(test, dpreds, pairs)
        res["deep"]["source"] = source
        res["deep"]["scores"] = SC.strip(score_all(test, dpreds, pairs, f"deep_{res['deep']['config']}"))
        res["deep"]["deep_vs_gbm"] = {k: SC.deep_vs_gbm(Ld[k], Lg[k], days, idx)
                                      for k in ("A", "B", "C", "D") if k in Ld and k in Lg}
        res["deep"]["not_run"] = [k for k in ("A", "B", "C", "D") if k not in res["deep"]["deep_vs_gbm"]]
    res["rescored"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write(res)


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
        sets = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("base", "weather", "outages")
        stage_deep(None, sets=sets)
    else:
        raise SystemExit("usage: rehearsal.py gbm | deep [base,weather,outages] | rescore")
