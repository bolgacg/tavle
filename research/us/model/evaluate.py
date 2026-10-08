"""Every strategy walked forward over 2021 to 2023, for the list Bo approves before any freeze
(coordinator, 6 Oct 2026, night). Build years only: nothing on or after 2024-01-01 is read (lock.py).

    python evaluate.py compute   walk-forward predictions, cached per (model, configuration, year)
    python evaluate.py score     results/strategy_list_2021_2023.json and .md

Rules
  * Each month is predicted by a model refit on the 1st, trained on delivery dates up to two days before
    the month whose prices were public by 05:00 on its first bid day (walkforward.py), expanding from
    January 2020 (idea B from 25 March 2021, the start of the weather archive).
  * A choice is never made on the year it is scored. For year Y every configuration (for A the spike
    threshold S and cut p*; for C the five pairs) is fitted once on delivery dates up to two days before
    year Y-1 and scored on Y-1 by the idea's own mean daily net P&L, exactly as the rehearsal tuned for
    2023 (so 2023 reuses the v2 rehearsal's choices and predictions); only the chosen configuration is
    walked forward through Y. 2021 has no earlier year, so it uses defaults declared here before any
    result: regression gbm.GRID[0]; spike S = 50 USD/MWh, p* = 0.05. C's 2021 pairs are chosen on a
    2020 walk-forward (March to December) of the default regression. B's 2022 choice is fitted on its
    first five weeks of weather data and scored on May to December 2021. D's 20 outage sites for year Y
    are counted on snapshots up to 30 December of Y-1.
  * Deep strategies: the deep agent kept its configuration c2 (chosen on 2022) for every year, and the
    spike threshold S = 25 (chosen on 2022) for every year, so deep 2021 and 2022 numbers carry an
    in-sample configuration choice; the JSON says so per strategy.
  * Idea B is scored from 1 May 2021 (its first month with a month of weather-era training data).
  * Costs: full per-MWh costs of the cost audit (fees.py); 0.50 USD/MWh stress replaces them.
  * Deep strategies use the deep agent's predictions where they exist (2023 now); years without them
    are reported as pending and such a strategy is not assessed against the bar.
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
import rehearsal as R
import score as SC
import strategies as S
import walkforward as W

PERIODS = [(2021, dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
           (2022, dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
           (2023, dt.date(2023, 1, 1), dt.date(2023, 12, 31))]
PRE = (dt.date(2020, 3, 1), dt.date(2020, 12, 31))        # only for C's first pairs
B_FIRST = dt.date(2021, 5, 1)
B_TRAIN = dt.date(2021, 3, 25)
DEFAULT_REG = gbm.GRID[0]
DEFAULT_SPIKE = (50, 0.05)
STRESS = 0.50
BAR_SHARPE = 0.42
# Candidates added on the coordinator's request (6 Oct night); every choice on earlier years only.
STORM_GRID = (0.80, 0.90, 0.95, 0.98)       # storm day: score above this (coordinator's grid)
DEFAULT_STORM = 0.95                         # only if the year before has no scores
TAIL_S = 100                                 # extreme-tail spike flag: classifier target gap >= 100
ROLL_K = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)   # deep rolling cut: top k of the previous 30 days
DEFAULT_ROLL_K = 0.10
ROLL_DAYS = 30
MAX_ZONES = 6
RANK_DAYS, MIN_RANK_DAYS = 365, 60
OUT = R.RESULTS / "strategy_list_2021_2023"
KEY = ["delivery_hour", "zone"]
log = R.log


def wf_dir() -> Path:
    return R.CACHE / "wf"


def labels():
    return [p[0] for p in PERIODS]


def period(label, idea: str = "") -> tuple[dt.date, dt.date]:
    lab, lo, hi = next(p for p in PERIODS if p[0] == label)
    if idea == "B":
        lo = max(lo, B_FIRST)
    return lo, hi


def prev(label):
    i = labels().index(label)
    return labels()[i - 1] if i > 0 else None


# ----------------------------------------------------------------- data

def load(store=None):
    store = store or P.LockedStore()
    panel, cols = R.load_all(store)                  # base, gen, weather, outages counted through SITES_THROUGH
    outage_cols = {}
    for lab in labels():
        cut = dt.date(int(lab) - 1, 12, 30) if isinstance(lab, int) else period(lab)[0] - dt.timedelta(days=2)
        if R.SITES_THROUGH and cut == R.SITES_THROUGH and cols.get("outages"):
            outage_cols[lab] = cols["outages"]
            continue
        pre = f"o{cut:%Y%m%d}_"

        def build(cut=cut, pre=pre):
            f = S.outage_features(panel, store, sites_through=cut).add_prefix(pre)
            f[KEY] = panel[KEY]
            return f
        f = R._cached(f"feat_outages_sites{cut:%Y%m%d}_prefixed", build)
        new = [c for c in f.columns if c not in KEY]
        panel = panel.merge(f, on=KEY, how="left", validate="one_to_one")
        outage_cols[lab] = cols["base"] + new
    cols["outages_by_period"] = outage_cols
    lock.assert_build_only(panel["delivery_hour"])
    return panel, cols, store


# ----------------------------------------------------------------- cached walk-forward predictions

def _name(kind: str, key: str, lab) -> Path:
    return wf_dir() / f"{kind}_{key}_{lab}.parquet"


def wf(panel, kind: str, key: str, lab, lo: dt.date, hi: dt.date, make, train_start=None) -> pd.Series:
    f = _name(kind, key, lab)
    rws = R.rows(panel, lo, hi)
    if f.exists():
        d = P.read_locked(f, "delivery_hour", R.READ_END or lock.read_end())
        m = rws[KEY].merge(d, on=KEY, how="left", validate="one_to_one")
        return pd.Series(m["pred"].to_numpy(), index=rws.index)
    t0 = time.time()
    p, fits = W.walk_forward(panel, make, lo, hi, train_start=train_start, log=lambda *_: None)
    wf_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred": p}).join(rws[KEY]).to_parquet(f)
    log(f"  {kind} {key} {lab}: {len(fits)} refits, {time.time() - t0:.0f} s")
    return p.reindex(rws.index)


def reg_feats(cols, kind, lab):
    return cols["outages_by_period"][lab] if kind == "outages" else cols[kind]


V2_PRED_COL = {"base": "base", "weather": "weather", "outages": "outages", "spike": "spike"}


def _v2():
    """The v2 rehearsal (2023) made its choices by exactly this procedure; reuse them and its predictions."""
    if not R.OUT.exists() or not R._preds_path("gbm").exists():
        return None
    return json.loads(R.OUT.read_text())


def _v2_pred(panel, kind, lab):
    v2 = _v2()
    if v2 is None or str(lab) != str(R.REH[0].year):
        return None
    rws = R.rows(panel, *period(lab, "B" if kind == "weather" else ""))
    d = P.read_locked(R._preds_path("gbm"), "delivery_hour", R.READ_END or lock.read_end())
    m = rws[KEY].merge(d[KEY + [V2_PRED_COL[kind]]], on=KEY, how="left", validate="one_to_one")
    col = m[V2_PRED_COL[kind]]
    return pd.Series(col.to_numpy(), index=rws.index) if col.notna().all() else None


def reg_pred(panel, cols, kind, cfg, lab):
    """Walk-forward predictions of `cfg` for period `lab` (the v2 rehearsal's own when it chose the same)."""
    v2 = _v2()
    if v2 is not None and str(lab) == str(R.REH[0].year) and isinstance(v2["tuning"].get(kind), dict) \
            and v2["tuning"][kind].get("chosen_config") == cfg:
        p = _v2_pred(panel, kind, lab)
        if p is not None:
            return p
    lo, hi = period(lab, "B" if kind == "weather" else "")
    feats = reg_feats(cols, kind, lab)
    return wf(panel, kind, gbm.config_name(cfg), lab, lo, hi, lambda: gbm.GBMModel(feats, cfg),
              B_TRAIN if kind == "weather" else None)


def spike_pred(panel, cols, s_, lab):
    v2 = _v2()
    if v2 is not None and str(lab) == str(R.REH[0].year) and v2["spike_rule"]["S"] == s_:
        p = _v2_pred(panel, "spike", lab)
        if p is not None:
            return p
    lo, hi = period(lab)
    return wf(panel, "spike", f"S{s_}", lab, lo, hi, lambda: gbm.SpikeClassifier(cols["base"], s_))


# ----------------------------------------------------------------- choices made on the year before
# As in the rehearsal (rehearsal.tune, tune_spike): for period Y, one fit per configuration, trained on
# delivery dates up to two days before period Y-1 starts, scored on Y-1 by the idea's own mean daily net
# P&L. The first period uses the declared defaults. For 2023 this is exactly the v2 rehearsal's tuning,
# whose recorded choices are reused.

RULE = {"base": S.idea_A_v1, "weather": S.idea_B, "outages": S.idea_D}
MIN_TRAIN_DAYS = 28


def mean_net(led, rws) -> float:
    days = pd.DatetimeIndex(sorted(rws.loc[rws["gap"].notna(), "delivery_date"].unique()))
    return float(SC.daily(led, days).mean())


def _sel_fit(panel, kind, key, lab, make, train_start=None) -> tuple[pd.Series | None, pd.DataFrame]:
    """One fit trained up to two days before period prev(lab) starts, predicting prev(lab); cached."""
    p0 = prev(lab)
    start = period(lab)[0]
    va = W.check_choice_rows(W.choice_rows(R.rows(panel, *period(p0, "B" if kind == "weather" else "")), start),
                             start, f"select {kind} {key} for {lab}")
    f = _name(f"sel_{kind}", key, lab)
    if f.exists():
        d = P.read_locked(f, "delivery_hour", R.READ_END or lock.read_end())
        m = va[KEY].merge(d, on=KEY, how="left", validate="one_to_one")
        return pd.Series(m["pred"].to_numpy(), index=va.index), va
    tr = panel[W.train_mask(panel, va["delivery_date"].min().date(), train_start)]
    if tr["delivery_date"].nunique() < MIN_TRAIN_DAYS:
        return None, va
    m = make()
    m.fit(tr)
    p = m.predict(va)
    wf_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred": p}).join(va[KEY]).to_parquet(f)
    return p, va


def choose_reg(panel, cols, kind, lab) -> tuple[dict, dict]:
    p0 = prev(lab)
    if p0 is None:
        return DEFAULT_REG, {"how": "default (no earlier period)", "config": gbm.config_name(DEFAULT_REG)}
    v2 = _v2()
    if v2 is not None and str(lab) == str(R.REH[0].year) and isinstance(v2["tuning"].get(kind), dict):
        t = v2["tuning"][kind]
        return t["chosen_config"], {"how": f"v2 rehearsal tuning: best of {len(t['configs'])}, trained to "
                                           f"{t['train_last']}, scored on {p0}", "config": t["chosen"]}
    feats = reg_feats(cols, kind, lab)
    scores = {}
    for cfg in gbm.GRID:
        p, va = _sel_fit(panel, kind, gbm.config_name(cfg), lab, lambda cfg=cfg: gbm.GBMModel(feats, cfg),
                         B_TRAIN if kind == "weather" else None)
        if p is None:
            return DEFAULT_REG, {"how": f"default (under {MIN_TRAIN_DAYS} days of training data before {p0})",
                                 "config": gbm.config_name(DEFAULT_REG)}
        scores[gbm.config_name(cfg)] = mean_net(RULE[kind](va, p), va)
    best = max(gbm.GRID, key=lambda c: scores[gbm.config_name(c)])
    return best, {"how": f"best of {len(gbm.GRID)}: one fit trained to two days before {p0}, scored on {p0}",
                  "config": gbm.config_name(best), "scores_mean_daily_net": scores}


def choose_spike(panel, cols, lab) -> tuple[tuple, dict]:
    p0 = prev(lab)
    if p0 is None:
        return DEFAULT_SPIKE, {"how": "default (no earlier period)", "S": DEFAULT_SPIKE[0], "p_star": DEFAULT_SPIKE[1]}
    v2 = _v2()
    if v2 is not None and str(lab) == str(R.REH[0].year):
        sr = v2["spike_rule"]
        return (sr["S"], sr["p_star"]), {"how": f"v2 rehearsal tuning on {p0}", "S": sr["S"], "p_star": sr["p_star"]}
    scores = {}
    for s_ in gbm.S_GRID:
        p, va = _sel_fit(panel, "spike", f"S{s_}", lab, lambda s_=s_: gbm.SpikeClassifier(cols["base"], s_))
        for ps in gbm.P_STAR_GRID:
            scores[f"S{s_}_p{ps}"] = mean_net(S.idea_A(va, p, ps), va)
    k = max(scores, key=scores.get)
    s_, ps = int(k.split("_p")[0][1:]), float(k.split("_p")[1])
    return (s_, ps), {"how": f"best of {len(scores)}: one fit per S trained to two days before {p0}, scored on {p0}",
                      "S": s_, "p_star": ps, "scores_mean_daily_net": scores}


def choose_pairs(panel, cols, lab, base_cfg) -> tuple[list, dict]:
    p0 = prev(lab)
    v2 = _v2()
    if v2 is not None and str(lab) == str(R.REH[0].year):
        pairs = [tuple(x.split("|")) for x in v2["pairs"]["chosen"]]
        return pairs, {"pairs": v2["pairs"]["chosen"], "chosen_on": "v2 rehearsal: " + v2["pairs"]["chosen_on"]}
    if p0 is None:
        rws = W.check_choice_rows(W.choice_rows(R.rows(panel, *PRE), period(lab)[0]), period(lab)[0], f"pairs for {lab}")
        p = wf(panel, "base", gbm.config_name(DEFAULT_REG), "pre", *PRE, lambda: gbm.GBMModel(cols["base"], DEFAULT_REG))
        on = f"{PRE[0]}..{PRE[1]} walk-forward of the default regression"
    else:
        p, rws = _sel_fit(panel, "base", gbm.config_name(base_cfg), lab,
                          lambda: gbm.GBMModel(cols["base"], base_cfg))
        on = f"{p0} predictions of this period's chosen regression ({gbm.config_name(base_cfg)}), one fit trained before {p0}"
    pairs, table = S.choose_pairs(rws, p)
    return pairs, {"pairs": ["|".join(x) for x in pairs], "chosen_on": on, "top": table[:8]}


def compute(panel, cols):
    """Every prediction the choices and the scores need, cached; restartable."""
    for lab in labels():
        for kind in ("base", "weather", "outages"):
            if kind == "weather" and cols.get("weather") is None:
                continue
            cfg, info = choose_reg(panel, cols, kind, lab)
            reg_pred(panel, cols, kind, cfg, lab)
            log(f"  {lab} {kind}: {info['how']}; {info['config']}")
        (s_, ps), info = choose_spike(panel, cols, lab)
        spike_pred(panel, cols, s_, lab)
        log(f"  {lab} spike: {info['how']}; S {s_}, p* {ps}")
        ps_t, info = choose_tail(panel, cols, lab)
        tail_pred(panel, cols, lab)
        log(f"  {lab} tail spike S={TAIL_S}: {info['how']}; p* {ps_t}")
        base_cfg, _ = choose_reg(panel, cols, "base", lab)
        pairs, _ = choose_pairs(panel, cols, lab, base_cfg)
        log(f"period {lab}: predictions ready; pairs {pairs}")


# ----------------------------------------------------------------- the added candidates

def storm_score(panel) -> pd.Series:
    """Per delivery date D+1: the largest of four percentile ranks, each against the previous 365 days
    (strictly earlier, at least 60 values): the operator's NYISO peak load forecast for D+1, minus the
    coldest and the hottest GFS forecast temperature for D+1 over the zones (idea B's columns), and the
    mean gap of D's hours published by 05:00 (over zones). All are public at 05:00 on D. This matches the
    coordinator's side script (side/storm_value.py). It was designed AFTER seeing that a few storm days
    drive the losses: a forking-paths risk that only the held-out years can settle."""
    g = panel.groupby("delivery_date")
    day = pd.DataFrame({"heat": g["x_wx_d1_max_f"].max() if "x_wx_d1_max_f" in panel else np.nan,
                        "cold": -g["x_wx_d1_min_f"].min() if "x_wx_d1_min_f" in panel else np.nan,
                        "load": g["lf_nyiso_peak"].max(), "gap_d0": g["gap_d0_early"].mean()}).sort_index()
    ranks = pd.DataFrame(index=day.index)
    for c in day.columns:
        v = day[c].to_numpy(float)
        r = np.full(len(v), np.nan)
        for i in range(len(v)):
            prev_v = v[max(0, i - RANK_DAYS):i]
            prev_v = prev_v[~np.isnan(prev_v)]
            if len(prev_v) >= MIN_RANK_DAYS and not np.isnan(v[i]):
                r[i] = float((prev_v < v[i]).mean())
        ranks[c] = r
    return ranks.max(axis=1, skipna=True)


def storm_ledger(rws, score, q, fee=None):
    sc = rws["delivery_date"].map(score).to_numpy(float)
    pos = np.where(sc > q, S.NONE, S.SUPPLY)                 # a missing score: supply as usual
    return S.ledger(rws, pos, None, None, fee)


def storm_flip_ledger(rws, score, q, fee=None):
    """On storm days virtual LOAD in every zone-hour, otherwise supply."""
    sc = rws["delivery_date"].map(score).to_numpy(float)
    pos = np.where(sc > q, S.LOAD, S.SUPPLY)
    return S.ledger(rws, pos, None, None, fee)


def choose_storm(panel, lab, score, flip: bool = False) -> tuple[float, dict]:
    """The cut chosen on the 365 days before the period (for 2021: 2020), by the rule's own mean daily net."""
    lo = pd.Timestamp(period(lab)[0])
    rws = panel[(panel["delivery_date"] >= lo - pd.Timedelta(days=365)) & (panel["delivery_date"] < lo)]
    rws = W.check_choice_rows(W.choice_rows(rws, period(lab)[0]), period(lab)[0], f"storm cut for {lab}")
    if score.reindex(rws["delivery_date"].unique()).notna().sum() < 30:
        return DEFAULT_STORM, {"how": "default (no storm scores the year before)", "cut": DEFAULT_STORM}
    fn = storm_flip_ledger if flip else storm_ledger
    scores = {str(q): mean_net(fn(rws, score, q), rws) for q in STORM_GRID}
    best = max(STORM_GRID, key=lambda q: scores[str(q)])
    return best, {"how": f"best of {len(STORM_GRID)} on {rws['delivery_date'].min().date()}.."
                         f"{rws['delivery_date'].max().date()}", "cut": best, "scores_mean_daily_net": scores}


def rolling_rank(rws_all: pd.DataFrame, score: pd.Series, k: float | None = None):
    """Causal per-row figures against the pooled scores of the previous 30 delivery dates (strictly earlier):
    with k, True where the score is above their (1-k) quantile (no earlier dates: never); without k, the
    score's percentile rank among them (no earlier dates: 0, i.e. full size)."""
    sc = score.reindex(rws_all.index).to_numpy(float)
    pos_by = pd.Series(np.arange(len(sc))).groupby(rws_all["delivery_date"].to_numpy()).indices
    order = sorted(pos_by)
    by = {d: np.sort(sc[pos_by[d]][~np.isnan(sc[pos_by[d]])]) for d in order}
    out = np.zeros(len(sc), dtype=float if k is None else bool)
    for i, d in enumerate(order):
        prev_arr = [by[x] for x in order[max(0, i - ROLL_DAYS):i]]
        m = pos_by[d]
        if not prev_arr:
            continue
        pool = np.sort(np.concatenate(prev_arr))
        if len(pool) == 0:
            continue
        if k is None:
            out[m] = np.searchsorted(pool, sc[m], side="left") / len(pool)
        else:
            out[m] = sc[m] > np.quantile(pool, 1 - k)
    return pd.Series(out, index=rws_all.index)


def choose_zones(panel, lab) -> tuple[list, dict]:
    """Up to 6 zones whose supply profit on all earlier build years, without each zone's most extreme 1%
    of hours, was positive, ranked by the smallest worst-day loss."""
    lo = period(lab)[0]
    h = W.check_choice_rows(W.choice_rows(panel, lo), lo, f"zones for {lab}")
    rows_ = []
    for z, g in h.groupby("zone"):
        sup = -g["gap"].to_numpy(float) - fees.supply_fee_for(g["delivery_date"])
        cut = np.quantile(np.abs(g["gap"]), 0.99)
        calm = sup[np.abs(g["gap"].to_numpy(float)) <= cut]
        worst = float(pd.Series(sup, index=g.index).groupby(g["delivery_date"]).sum().min())
        rows_.append({"zone": z, "per_mwh_without_top1pct": float(calm.mean()), "worst_day_usd": worst})
    ok = sorted([r for r in rows_ if r["per_mwh_without_top1pct"] > 0], key=lambda r: -r["worst_day_usd"])
    chosen = [r["zone"] for r in ok[:MAX_ZONES]]
    return chosen, {"zones": chosen, "chosen_on": f"{h['delivery_date'].min().date()}..{h['delivery_date'].max().date()}",
                    "table": rows_}


def zone_ledger(rws, zones, fee=None):
    return S.ledger(rws, np.where(rws["zone"].isin(zones), S.SUPPLY, S.NONE), None, None, fee)


def tail_pred(panel, cols, lab):
    lo, hi = period(lab)
    return wf(panel, "spike", f"S{TAIL_S}", lab, lo, hi, lambda: gbm.SpikeClassifier(cols["base"], TAIL_S))


def choose_tail(panel, cols, lab) -> tuple[float, dict]:
    p0 = prev(lab)
    if p0 is None:
        return DEFAULT_SPIKE[1], {"how": "default (no earlier period)", "S": TAIL_S, "p_star": DEFAULT_SPIKE[1]}
    p, va = _sel_fit(panel, "spike", f"S{TAIL_S}", lab, lambda: gbm.SpikeClassifier(cols["base"], TAIL_S))
    scores = {str(ps): mean_net(S.idea_A(va, p, ps), va) for ps in gbm.P_STAR_GRID}
    best = max(gbm.P_STAR_GRID, key=lambda ps: scores[str(ps)])
    return best, {"how": f"best of {len(scores)} cuts, one fit trained before {p0}, scored on {p0}", "S": TAIL_S,
                  "p_star": best, "scores_mean_daily_net": scores}


def all_rows(panel):
    return R.rows(panel, PERIODS[0][1], PERIODS[-1][2])


def continuous(panel, parts: dict) -> pd.Series:
    """One score series over every period (each period's own predictions), for the 30-day rolling rules."""
    return pd.concat([v for v in parts.values() if v is not None]).reindex(all_rows(panel).index)


def choose_roll_k(panel, lab, dscore) -> tuple[float, dict]:
    p0 = prev(lab)
    if p0 is None:
        return DEFAULT_ROLL_K, {"how": "default (no earlier period)", "k": DEFAULT_ROLL_K}
    rws_all = all_rows(panel)
    va = W.check_choice_rows(W.choice_rows(R.rows(panel, *period(p0)), period(lab)[0]), period(lab)[0],
                             f"rolling k for {lab}")
    scores = {}
    for k in ROLL_K:
        flag = rolling_rank(rws_all, dscore, k).reindex(va.index)
        led = S.ledger(va, np.where(flag.to_numpy(bool), S.NONE, S.SUPPLY), None, None, None)
        scores[str(k)] = mean_net(led, va)
    best = max(ROLL_K, key=lambda k: scores[str(k)])
    return best, {"how": f"best of {len(ROLL_K)} on {p0}", "k": best, "scores_mean_daily_net": scores}


# ----------------------------------------------------------------- deep predictions (deep agent)

def deep_pred(panel, kind: str, lab) -> pd.Series | None:
    """results/deep_wf_<kind>_<period>.parquet (delivery_hour, zone, pred_gap or p_spike), or for 2023
    the files already delivered for the rehearsal."""
    rws = R.rows(panel, *period(lab, "B" if kind == "weather" else ""))
    col = "p_spike" if kind == "spike" else "pred_gap"
    cands = [R.RESULTS / f"deep_wf_{kind}_{lab}.parquet"]
    if lab == 2023:
        cands += {"base": [R.RESULTS / "deep_rehearsal_2023.parquet"],
                  "weather": [R.CACHE / "preds_deep_weather_2023.parquet"],
                  "outages": [R.CACHE / "preds_deep_outages_sites20221230_2023.parquet"],
                  "spike": [R.RESULTS / "deep_spike_2023.parquet", R.RESULTS / "deep_spike_rehearsal_2023.parquet"]}[kind]
    for f in cands:
        if f.exists():
            d = P.read_locked(f, "delivery_hour", R.READ_END or lock.read_end())
            if col not in d.columns:
                continue
            if kind == "spike" and "spike_threshold" in d.columns and R.SPIKE_CONFIG.exists():
                s_cfg = json.loads(R.SPIKE_CONFIG.read_text())["S_usd_per_mwh"]
                if not (d["spike_threshold"] == s_cfg).all():
                    log(f"!! {f.name}: spike threshold {sorted(d['spike_threshold'].unique())} is not S {s_cfg}; skipped")
                    continue
            m = rws[KEY].merge(d[KEY + [col]], on=KEY, how="left", validate="one_to_one")
            if m[col].notna().all():
                return pd.Series(m[col].to_numpy(), index=rws.index)
    return None


def deep_p_star(lab) -> float | None:
    if prev(lab) is None:
        return DEFAULT_SPIKE[1]                           # the first period: the declared default cut
    for f, k in ((R.RESULTS / "deep_wf_spike_choice.json", str(lab)), (R.RESULTS / "deep_spike_choice.json", None)):
        if f.exists():
            c = json.loads(f.read_text())
            v = c.get(k) if k else (c.get("p_star") if lab == 2023 else None)
            if v is not None:
                return float(v)
    return None


# ----------------------------------------------------------------- ledgers per strategy and period

DESCRIBE = {
    "always_supply": "Sell virtual supply in every zone and hour.",
    "baseline": "In each zone-hour, take the side that won on average over the past 365 days.",
    "A_hourly_mean": "Sell supply only in zone-hours whose past-365-day average gap paid more than the cost; no model.",
    "A_regression_v1": "Sell supply where a gradient-boosting forecast of the gap is below minus the cost (A as first registered).",
    "A_spike_gbm": "Sell supply everywhere except hours a gradient-boosting model flags as likely real-time spikes.",
    "B_gbm": "Trade either side where a forecast using weather forecasts expects the gap to beat the cost.",
    "C_gbm": "Buy one zone and sell another in five fixed zone pairs when the forecast spread beats both legs' costs.",
    "D_gbm": "Trade either side where a forecast using scheduled transmission outages expects the gap to beat the cost.",
    "A_regression_v1_deep": "As A regression, with the deep model's forecast.",
    "A_spike_deep": "As A spike flag, with the deep model's spike probability.",
    "B_deep": "As B, with the deep model.",
    "C_deep": "As C (same pairs), with the deep model's forecasts.",
    "D_deep": "As D, with the deep model.",
    "storm_day_filter": "Sell supply everywhere, but skip whole days whose forecast load, forecast cold or heat, or this morning's real-time gap is extreme for the past year.",
    "storm_day_flip": "As the storm-day filter, but on storm days buy virtual load everywhere instead of sitting out.",
    "A_spike_tail_S100_gbm": "As A spike flag, but the model looks only for extreme spikes (real time 100 USD/MWh or more above day ahead).",
    "risk_sized_supply_gbm": "Sell supply everywhere, sized down from 1 MW in proportion to how high the hour's spike score ranks in the last 30 days.",
    "zone_subset_supply": "Sell supply every hour, but only in up to six zones that earned without their worst hours and had the mildest worst day in earlier years.",
    "A_spike_deep_rolling_cut": "Sell supply everywhere except hours whose deep spike score is in the top k of the last 30 days (coordinator's rule).",
}


def ledgers_for(panel, cols, lab, fee=None, choices=None) -> tuple[dict, dict]:
    ch = choices if choices is not None else {}
    rws = R.rows(panel, *period(lab))
    rws_b = R.rows(panel, *period(lab, "B"))
    L = {"always_supply": S.always_supply(rws, fee), "baseline": S.baseline(rws, fee),
         "A_hourly_mean": S.idea_A_hourly_mean(rws, fee)}
    if "base" not in ch:
        ch["base"] = choose_reg(panel, cols, "base", lab)
        ch["weather"] = choose_reg(panel, cols, "weather", lab) if cols.get("weather") else (None, {"how": "not run"})
        ch["outages"] = choose_reg(panel, cols, "outages", lab)
        ch["spike"] = choose_spike(panel, cols, lab)
        ch["pairs"] = choose_pairs(panel, cols, lab, ch["base"][0])
    pairs = [tuple(x.split("|")) for x in ch["pairs"][1]["pairs"]]
    pb = reg_pred(panel, cols, "base", ch["base"][0], lab)
    L["A_regression_v1"] = S.idea_A_v1(rws, pb, fee)
    L["C_gbm"] = S.idea_C(rws, pb, pairs, fee)
    s_, ps = ch["spike"][0]
    L["A_spike_gbm"] = S.idea_A(rws, spike_pred(panel, cols, s_, lab), ps, fee)
    if ch["weather"][0] is not None:
        L["B_gbm"] = S.idea_B(rws_b, reg_pred(panel, cols, "weather", ch["weather"][0], lab), fee)
    L["D_gbm"] = S.idea_D(rws, reg_pred(panel, cols, "outages", ch["outages"][0], lab), fee)
    db = deep_pred(panel, "base", lab)
    if db is not None:
        L["A_regression_v1_deep"] = S.idea_A_v1(rws, db, fee)
        L["C_deep"] = S.idea_C(rws, db, pairs, fee)
    dw = deep_pred(panel, "weather", lab)
    if dw is not None:
        L["B_deep"] = S.idea_B(rws_b, dw, fee)
    do = deep_pred(panel, "outages", lab)
    if do is not None:
        L["D_deep"] = S.idea_D(rws, do, fee)
    dsp, dps = deep_pred(panel, "spike", lab), deep_p_star(lab)
    if dsp is not None and dps is not None:
        L["A_spike_deep"] = S.idea_A(rws, dsp, dps, fee)
    # the added candidates
    G = _shared(panel, cols)
    if "storm" not in ch:
        ch["storm"] = choose_storm(panel, lab, G["storm"])
        ch["storm_flip"] = choose_storm(panel, lab, G["storm"], flip=True)
        ch["tail"] = choose_tail(panel, cols, lab)
        ch["zones"] = choose_zones(panel, lab)
        ch["roll_k"] = choose_roll_k(panel, lab, G["deep_spike"]) if G["deep_spike"] is not None else (None, {"how": "deep spike scores missing"})
    L["storm_day_filter"] = storm_ledger(rws, G["storm"], ch["storm"][0], fee)
    L["storm_day_flip"] = storm_flip_ledger(rws, G["storm"], ch["storm_flip"][0], fee)
    L["A_spike_tail_S100_gbm"] = S.idea_A(rws, tail_pred(panel, cols, lab), ch["tail"][0], fee)
    mw = (1 - G["gbm_rank"].reindex(rws.index)).clip(lower=0).to_numpy(float)
    L["risk_sized_supply_gbm"] = S.ledger(rws, -mw, None, None, fee)
    L["zone_subset_supply"] = zone_ledger(rws, ch["zones"][0], fee)
    if G["deep_spike"] is not None and ch["roll_k"][0] is not None:
        flag = rolling_rank(all_rows(panel), G["deep_spike"], ch["roll_k"][0]).reindex(rws.index)
        L["A_spike_deep_rolling_cut"] = S.ledger(rws, np.where(flag.to_numpy(bool), S.NONE, S.SUPPLY), None, None, fee)
    return L, ch


_SHARED: dict = {}


def _shared(panel, cols) -> dict:
    """Series that span every period: the storm score, the GBM spike score's 30-day rank, deep spike scores."""
    if _SHARED.get("n") != len(panel):
        _SHARED.clear()
        _SHARED["n"] = len(panel)
        _SHARED["storm"] = storm_score(panel)
        gs = {lab: spike_pred(panel, cols, choose_spike(panel, cols, lab)[0][0], lab) for lab in labels()}
        _SHARED["gbm_rank"] = rolling_rank(all_rows(panel), continuous(panel, gs))
        ds = {lab: deep_pred(panel, "spike", lab) for lab in labels()}
        _SHARED["deep_spike"] = continuous(panel, ds) if all(v is not None for v in ds.values()) else None
    return _SHARED


# ----------------------------------------------------------------- metrics

def metrics(led: pd.DataFrame, days: pd.DatetimeIndex, led_stress: pd.DataFrame | None, idx=None) -> dict:
    d = SC.daily(led, days)
    n = len(days)
    net, mwh = float(led["pnl"].sum()), float(led["mwh"].sum())
    month = pd.Series(days.to_period("M").astype(str), index=days)
    mo = d.groupby(month).sum()
    annual = float(d.mean()) * CX.DAYS_PER_YEAR
    mdd = CX.max_drawdown(d)
    out = {"days": n, "first": str(days.min().date()), "last": str(days.max().date()),
           "net_usd": round(net, 2), "mwh": mwh, "profit_per_mwh": (net / mwh) if mwh else None,
           "sharpe_annualised": CX.sharpe_annualised(d), "max_drawdown_usd": round(mdd, 2),
           "return_on_bankroll_annualised": annual / CX.BANKROLL_USD,
           "return_over_max_drawdown": (annual / mdd) if mdd > 0 else None,
           "worst_day": {"date": str(d.idxmin().date()), "usd": round(float(d.min()), 2)},
           "months_positive": int((mo > 0).sum()), "months": int(len(mo)),
           "net_usd_at_0_50_stress": round(float(led_stress["pnl"].sum()), 2) if led_stress is not None else None,
           "net_usd_without_best_5_days": round(float(d.sum() - d.nlargest(5).sum()), 2),
           "best_5_days": {str(i.date()): round(float(v), 2) for i, v in d.nlargest(5).items()}}
    if idx is not None:
        out["mean_daily_net_interval_95"] = list(SC.interval(SC.boot_means(d.to_numpy(), idx), 0.95))
    return out


def score(panel, cols) -> dict:
    per, choices, all_led, all_stress = {}, {}, {}, {}
    for lab in labels():
        L, ch = ledgers_for(panel, cols, lab)
        Ls, _ = ledgers_for(panel, cols, lab, fee=STRESS, choices=ch)
        choices[str(lab)] = {k: v[1] for k, v in ch.items()}
        for k, led in L.items():
            all_led.setdefault(k, {})[lab] = led
            all_stress.setdefault(k, {})[lab] = Ls.get(k)
    rows_out = []
    for k in DESCRIBE:
        if k not in all_led:
            rows_out.append({"strategy": k, "what": DESCRIBE[k], "status": "pending: no predictions"})
            continue
        years = {}
        for lab, led in all_led[k].items():
            rws = R.rows(panel, *period(lab, "B" if k.startswith("B_") else ""))
            days = pd.DatetimeIndex(sorted(rws.loc[rws["gap"].notna(), "delivery_date"].unique()))
            years[str(lab)] = metrics(led, days, all_stress[k][lab])
        led = pd.concat(all_led[k].values(), ignore_index=True)
        lst = pd.concat([v for v in all_stress[k].values() if v is not None], ignore_index=True)
        lo = min(pd.Timestamp(y["first"]) for y in years.values())
        hi = max(pd.Timestamp(y["last"]) for y in years.values())
        rws = R.rows(panel, lo.date(), hi.date())
        days = pd.DatetimeIndex(sorted(rws.loc[rws["gap"].notna(), "delivery_date"].unique()))
        complete = len(years) == len(PERIODS)
        tot = metrics(led, days, lst, SC.stationary_indices(len(days)))
        pos_years = sum(1 for y in years.values() if y["net_usd"] > 0)
        bar = {"positive_total": tot["net_usd"] > 0, "positive_years_at_least_2_of_3": pos_years >= 2,
               "sharpe_above_0_42": (tot["sharpe_annualised"] or -1) > BAR_SHARPE,
               "positive_at_0_50_stress": (tot["net_usd_at_0_50_stress"] or -1) > 0}
        note = None
        if "deep" in k:
            note = ("deep configuration c2 and spike threshold S = 25 were chosen on 2022 and kept for every year, "
                    "so 2021 and 2022 are not out of sample in their configuration choice")
        rows_out.append({"strategy": k, "what": DESCRIBE[k], "status": "complete" if complete else
                         f"incomplete: only {', '.join(map(str, years))} available", "total": tot, "years": years,
                         "positive_years": pos_years, "bar": bar, "selection_note": note,
                         "passes_bar": bool(complete and all(bar.values()))})
    done = [r for r in rows_out if "total" in r]
    done.sort(key=lambda r: -r["total"]["net_usd"])
    for i, r in enumerate(done, 1):
        r["rank"] = i
    return {"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "build years only (2021 to 2023 scored, 2020 used for training); nothing on or after "
                      "2024-01-01 was read",
            "rules": __doc__.split("Rules", 1)[1].strip(), "bankroll_usd": CX.BANKROLL_USD,
            "bar": {"positive_total_net": True, "positive_in_at_least": "2 of 3 years", "sharpe_above": BAR_SHARPE,
                    "positive_at_cost_stress_usd_per_mwh": STRESS},
            "costs": {"load_usd_per_mwh": fees.RATES, "supply_usd_per_mwh": fees.SUPPLY_RATES},
            "choices": choices, "strategies": done + [r for r in rows_out if "total" not in r],
            "tries": {"strategies_in_this_list": len(rows_out),
                      "configurations_compared_per_choice": {"regression (A v1, B, C, D)": len(gbm.GRID),
                                                             "spike S x p*": len(gbm.S_GRID) * len(gbm.P_STAR_GRID),
                                                             "tail spike p*": len(gbm.P_STAR_GRID),
                                                             "storm cut (filter; flip)": len(STORM_GRID), "deep rolling k": len(ROLL_K),
                                                             "C pairs": 55, "zone subset": "rule, no grid",
                                                             "risk sizing": "rule, no grid"},
                      "earlier_versions": "idea A v1 (regression rule) is in the list; the v1 and v2 rehearsals "
                                          "scored 2023 only"}}


def _f(x, nd=0, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:+.1f}%"
    return f"{x:+,.{nd}f}" if nd else f"{x:+,.0f}"


def write_md(res: dict, path: Path, intro: str = "") -> None:
    yrs = [str(p[0]) for p in PERIODS]
    head = ("| Rank | Strategy | What it does | Total net, USD | " + " | ".join(yrs) +
            " | Per MWh | Sharpe | Max drawdown | Return on 500k a year | Worst day | Months positive"
            " | Total at 0.50 cost | Total without its 5 best days | Passes the bar |")
    ncol = head.count("|") - 1
    lines = [head, "|" + "---|" * ncol]
    for r in res["strategies"]:
        if "total" not in r:
            lines.append(f"| | {r['strategy']} | {r['what']} | {r['status']} |" + " |" * (ncol - 4))
            continue
        t, y = r["total"], r["years"]
        fails = {"positive_total": "total not positive", "positive_years_at_least_2_of_3": "under 2 positive years",
                 "sharpe_above_0_42": "Sharpe 0.42 or less", "positive_at_0_50_stress": "loses at 0.50"}
        why = [fails[k] for k, v in r["bar"].items() if not v]
        if r["status"] != "complete":
            why.append(r["status"])
        flag = "**yes**" if r["passes_bar"] else "no (" + "; ".join(why) + ")"
        name = r["strategy"] + (" *" if "deep" in r["strategy"] else "") + (" †" if r["strategy"].startswith("storm") else "")
        lines.append(f"| {r['rank']} | {name} | {r['what']} | {_f(t['net_usd'])} | "
                     + " | ".join(_f(y[k]['net_usd']) if k in y else "pending" for k in yrs)
                     + f" | {_f(t['profit_per_mwh'], 2)} | {_f(t['sharpe_annualised'], 2)} | {t['max_drawdown_usd']:,.0f}"
                     f" | {_f(t['return_on_bankroll_annualised'], pct=True)} | {t['worst_day']['usd']:,.0f}"
                     f" | {t['months_positive']} of {t['months']} | {_f(t['net_usd_at_0_50_stress'])}"
                     f" | {_f(t['net_usd_without_best_5_days'])} | {flag} |")
    tr = res["tries"]
    notes = ["", f"Strategies in this list: {tr['strategies_in_this_list']}. Configurations compared inside each "
             "choice: " + "; ".join(f"{k} {v}" for k, v in tr["configurations_compared_per_choice"].items()) + ".",
             "", "\\* Deep strategies kept configuration c2 and spike threshold S = 25, both chosen on 2022, in every "
             "year: their 2021 and 2022 figures are not out of sample in that choice.", "",
             "† The storm score was designed after seeing that a few storm days drive the losses: a forking-paths "
             "risk that only the held-out years can settle. Its cut is chosen on the year before, as for the others.", "",
             "Years are calendar years of delivery; B's 2021 covers 1 May to 31 December. Worst day is the lowest "
             "daily net P&L over 2021 to 2023. Return on 500k a year = mean daily net x 365 / 500,000.", ""]
    path.write_text(intro + "\n".join(lines) + "\n" + "\n".join(notes) + "\n")


def export_idea9(panel, cols) -> Path:
    """Idea 9 (extreme-tail spike flag, S = 100) for the strategy lab: the 2021 to 2023 walk-forward
    probabilities (monthly refits, as cached by compute) and, beside them, each year's cut p* chosen on
    the year before (2021: the declared default)."""
    parts, choice = [], {}
    for lab in labels():
        rws = R.rows(panel, *period(lab))
        p = tail_pred(panel, cols, lab)
        parts.append(pd.DataFrame({"delivery_hour": rws["delivery_hour"], "zone": rws["zone"], "p_spike": p,
                                   "refit_month": rws["delivery_date"].dt.to_period("M").dt.to_timestamp()}))
        ps, info = choose_tail(panel, cols, lab)
        choice[str(lab)] = {"p_star": ps, "how": info["how"], "scores_mean_daily_net": info.get("scores_mean_daily_net")}
    out = pd.concat(parts, ignore_index=True)
    lock.assert_build_only(out["delivery_hour"])
    f = R.RESULTS / "idea9_wf_2021_2023.parquet"
    out.to_parquet(f)
    f.with_suffix(".json").write_text(json.dumps({
        "idea": 9, "what": "extreme-tail spike flag: supply in every zone-hour except where P(gap >= 100) > p_star",
        "S": TAIL_S, "model": "gbm.SpikeClassifier, LightGBM binary, fixed settings (gbm.SPIKE_FIXED), monthly refits "
        "on delivery dates up to two days before each month, expanding from 2020-01-01", "p_star_grid": list(gbm.P_STAR_GRID),
        "p_star_by_year": choice, "rows": int(len(out))}, indent=1, default=str))
    log(f"wrote {f}")
    return f


def main(stage: str):
    panel, cols, _ = load()
    log(f"panel {len(panel)} rows {panel['delivery_date'].min().date()}..{panel['delivery_date'].max().date()}")
    if stage in ("compute", "all"):
        compute(panel, cols)
    if stage == "export9":
        export_idea9(panel, cols)
    if stage in ("score", "all"):
        res = score(panel, cols)
        OUT.with_suffix(".json").write_text(json.dumps(SC.strip(res), indent=1, default=str))
        write_md(res, OUT.parent / (OUT.name + "_table.md"))      # the hand-written .md embeds this table
        log(f"wrote {OUT.with_suffix('.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
