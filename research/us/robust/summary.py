"""Idea 13 (portfolio of the survivors), the scoring of the GPU runs, and results/robust/robustness_summary.md.
OBJECTIVES.md addendum with idea 13 and the overnight checks (commit b31b516); sizing decision in the addendum
before it (commit 7504c92).

    .venv/bin/python robust/summary.py        (on gene from ~/nyiso-us; `smoke` for synthetic data)

Run last in the CPU lane and again at the end of the GPU lane; each run writes whatever is complete and marks
the rest pending, so the later run leaves the complete summary. Writes only in results/robust/.

Idea 13, as registered: every strategy that passes Bo's bar at the sizing view and is not flagged fragile (flags
from the final side/lab_results.json; idea 10 and both storm-day flips count as fragile per their audits), each
scaled so its worst 2021 to 2023 drawdown equals the same risk budget (the sizing view's 100,000 USD, at most
5 MW, as side/lab.py computes it), summed with equal risk weights (1/N each), positions netted per zone-hour, and
scored by side/lab.py like every other row. The approved list does not exist yet, so every scored row is eligible.

Deep runs (robust/deep_runs.py): each seed's predictions, the mean of all 20, the registered seeds 0 to 2, and
six disjoint sets of three seeds are turned into positions by the registered rules (strategies.idea_C with each
year's registered pairs for C deep, strategies.idea_B for B deep) and scored by side/lab.py. No seed is selected.
Holdout: every loader is a locked one; nothing on or after 2024-01-01 is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEEP = {"C_deep": "base", "B_deep": "weather"}
WF_NAME = {"base": "base", "weather": "B"}
VARIANTS = ["dropout_0.2", "dropout_0.4", "weight_decay_1e-3", "weight_decay_1e-1"]
VARIANT_TEXT = {"dropout_0.2": "dropout 0.2 (c2: 0.3)", "dropout_0.4": "dropout 0.4 (c2: 0.3)",
                "weight_decay_1e-3": "weight decay 0.001 (c2: 0.01)", "weight_decay_1e-1": "weight decay 0.1 (c2: 0.01)"}
N_SEEDS = 20


# ------------------------------------------------------------------ deep predictions to positions
class DeepScorer:
    def __init__(self, F, rows, choices):
        import evaluate as E
        import rehearsal as R
        import strategies as S
        self.F, self.E, self.R, self.S, self.choices = F, E, R, S, choices
        self.rows = rows
        self.labels = E.labels()

    def score(self, strat: str, pred: pd.DataFrame, col: str) -> dict:
        """pred: delivery_hour, zone and a prediction column; returns rb.brief of the lab's evaluate."""
        E, R, S = self.E, self.R, self.S
        leds = []
        for lab in self.labels:
            rws = R.rows(self.rows, *E.period(lab, "B" if strat == "B_deep" else ""))
            m = rws[["delivery_hour", "zone"]].merge(pred[["delivery_hour", "zone", col]], on=["delivery_hour", "zone"],
                                                    how="left", validate="one_to_one")
            p = pd.Series(m[col].to_numpy(float), index=rws.index)
            if p.isna().any():
                raise ValueError(f"{strat} {lab}: {int(p.isna().sum())} rows without a prediction in {col}")
            if strat == "B_deep":
                leds.append(S.idea_B(rws, p))
            else:
                pairs = [tuple(x.split("|")) for x in self.choices[str(lab)]["pairs"]["pairs"]]
                leds.append(S.idea_C(rws, p, pairs))
        return rb.brief(rb.score(self.F, rb.ledger_to_mw(self.F, pd.concat(leds, ignore_index=True))))


def registered_deep(kind: str) -> pd.DataFrame:
    import lock
    import panel as P
    f = rb.RESULTS / f"deep_wf_2021_2023_{WF_NAME[kind]}.parquet"
    return P.read_locked(f, "delivery_hour", lock.read_end(), columns=["delivery_hour", "zone", "pred_gap"])


def read_pred(path: Path, cols=None) -> pd.DataFrame:
    import lock
    import panel as P
    return P.read_locked(path, "delivery_hour", lock.read_end(), columns=cols)


def seed_check(DS: DeepScorer, labres) -> dict:
    out = {}
    for strat, kind in DEEP.items():
        d = rb.OUT / "deep" / f"seeds_{kind}"
        if not (d / "done.json").exists():
            out[strat] = {"pending": f"waiting for {d / 'done.json'}"}
            continue
        per = read_pred(d / "per_seed.parquet")
        seeds = [c for c in per.columns if c.startswith("seed_")]
        assert len(seeds) == N_SEEDS, seeds
        per["mean_20"] = per[seeds].mean(axis=1)
        per["seeds_0_2"] = per[seeds[:3]].mean(axis=1)
        triples = [seeds[i:i + 3] for i in range(0, 18, 3)]
        for t in triples:
            per["triple_" + "_".join(s[-2:] for s in t)] = per[t].mean(axis=1)
        reg = registered_deep(kind)
        rows = {s: DS.score(strat, per, s) for s in seeds}
        mean20 = DS.score(strat, per, "mean_20")
        s02 = DS.score(strat, per, "seeds_0_2")
        trip = {c: DS.score(strat, per, c) for c in per.columns if c.startswith("triple_")}
        regs = DS.score(strat, reg, "pred_gap")
        m = per.merge(reg, on=["delivery_hour", "zone"], how="inner")
        net = np.array([r["three_year_net"] for r in rows.values()], float)
        sh = np.array([r["sharpe"] if r["sharpe"] is not None else np.nan for r in rows.values()], float)
        lab_net = ((labres or {}).get("strategies", {}).get(strat, {}).get("three_year", {}) or {}).get("net_usd")
        out[strat] = {
            "kind": kind, "per_seed": rows, "mean_20": mean20, "seeds_0_2_retrained": s02, "triples": trip,
            "registered_file": regs, "lab_three_year_net": lab_net,
            "registered_reproduces_lab": None if lab_net is None else bool(round(regs["three_year_net"]) == round(lab_net)),
            "spread": {"net_min": float(net.min()), "net_median": float(np.median(net)), "net_max": float(net.max()),
                       "sharpe_min": float(np.nanmin(sh)), "sharpe_median": float(np.nanmedian(sh)),
                       "sharpe_max": float(np.nanmax(sh)),
                       "pass_at_sizing_view": int(sum(r["pass_at_sizing_view"] for r in rows.values())),
                       "pass_1MW": int(sum(bool(r["PASS_1MW"]) for r in rows.values())),
                       "negative_total": int((net < 0).sum()), "n": len(rows),
                       "triples_net_min": min(r["three_year_net"] for r in trip.values()),
                       "triples_net_max": max(r["three_year_net"] for r in trip.values())},
            "retrained_0_2_vs_registered": {
                "rows": int(len(m)), "corr": float(np.corrcoef(m["seeds_0_2"], m["pred_gap"])[0, 1]),
                "mean_abs_diff_usd": float(np.mean(np.abs(m["seeds_0_2"] - m["pred_gap"])))},
            "run": rb.read_json(d / "done.json")}
        rb.log(f"seeds {strat}: net {net.min():,.0f} / {np.median(net):,.0f} / {net.max():,.0f}")
    return out


def deep_neighbours(DS: DeepScorer) -> dict:
    out = {}
    for strat, kind in DEEP.items():
        rows = {}
        for v in VARIANTS:
            d = rb.OUT / "deep" / f"nb_{kind}_{v}"
            if not (d / "done.json").exists():
                rows[v] = {"pending": f"waiting for {d / 'done.json'}"}
                continue
            p = read_pred(d / f"deep_wf_2021_2023_{WF_NAME[kind]}.parquet", ["delivery_hour", "zone", "pred_gap"])
            rows[v] = DS.score(strat, p, "pred_gap")
        reg = DS.score(strat, registered_deep(kind), "pred_gap")
        done = [r for r in rows.values() if "pending" not in r]
        out[strat] = {"registered": reg, "variants": rows,
                      "sign_kept_at_every_neighbour": (all(np.sign(r["three_year_net"]) == np.sign(reg["three_year_net"])
                                                           for r in done) if len(done) == len(VARIANTS) else None)}
    return out


# ------------------------------------------------------------------ idea 13
def survivors(labres: dict) -> tuple[list, dict]:
    keep, why_not = [], {}
    for name, r in labres["strategies"].items():
        if not rb.sizing_pass(r):
            continue
        reasons = []
        if r.get("FRAGILE"):
            reasons.append("lab: " + "; ".join(r.get("fragility", {}).get("fragile_reasons", [])))
        if name in rb.FRAGILE_BY_AUDIT:
            reasons.append("fragile per " + rb.FRAGILE_BY_AUDIT[name])
        if reasons:
            why_not[name] = reasons
        else:
            keep.append(name)
    return keep, why_not


def build_strats(F, px, lf, wx, need: set) -> dict:
    """The lab's own strategy objects for the names needed, built by side/lab.py's builders in main()'s order,
    a group only when a needed name is still missing."""
    LB = rb.lab()
    got = rb.lead_strats(F)
    groups = [
        lambda: LB.mk_basics(F, LB.baseline_cube(F, px)),
        lambda: LB.mk_storm(F, LB.storm_score(LB.storm_inputs(F, px, lf, wx))[0]),
        lambda: [LB.mk_weather(F, LB.weather_surprise(F, wx)[0])],
        lambda: ([] if LB.load_tail_cube(F)[0] is None else
                 [LB.mk_spike_rule("idea9_tail_spike_S100", "9", "", "p_tail", LB.load_tail_cube(F)[0])]),
        lambda: [LB.mk_hist(F)],
        lambda: LB.policy_strats(F, {}, {}),
        lambda: LB.deep_day_strats(F, px, LB.load_deep_spike_cube(F)[0], {}, {}),
    ]
    for g in groups:
        if need <= set(got):
            break
        got.update({s.name: s for s in g()})
    missing = need - set(got)
    if missing:
        raise KeyError(f"no builder found for {sorted(missing)}")
    return {k: got[k] for k in need}


def idea13(F, px, lf, wx, labres) -> dict:
    LB = rb.lab()
    keep, why_not = survivors(labres)
    out = {"survivors": keep, "excluded_fragile": why_not,
           "eligible": "every row of side/lab_results.json (the approved list does not exist yet)",
           "lab_results_written": labres["meta"]["written"]}
    if not keep:
        out["result"] = "not run: no strategy passes at the sizing view without a fragile flag"
        return out
    strats = build_strats(F, px, lf, wx, set(keep))
    comps, mw13 = {}, np.zeros(F.NH)
    for name in keep:
        mw = rb.mw_of(F, strats[name])
        res = rb.score(F, mw)
        mdd = -float(res["three_year"]["max_drawdown_usd"])
        scale = min(LB.SIZE_CAP, LB.SIZE_DD / mdd) if mdd > 0 else LB.SIZE_CAP
        w = scale / len(keep)
        lab_net = labres["strategies"][name]["three_year"]["net_usd"]
        comps[name] = {"scale_100k_drawdown": round(scale, 4), "weight_mw": round(w, 4), "max_drawdown_1MW": -mdd,
                       "three_year_net_1MW": res["three_year"]["net_usd"], "lab_three_year_net": lab_net,
                       "reproduces_lab": bool(round(res["three_year"]["net_usd"]) == round(lab_net))}
        mw13 += w * mw
    res13 = rb.score(F, mw13)
    big = float(np.abs(mw13).max())
    mdd13 = -float(res13["three_year"]["max_drawdown_usd"])
    k = min(LB.SIZE_DD / mdd13 if mdd13 > 0 else np.inf, LB.SIZE_CAP / big if big > 0 else np.inf)
    ret = 100 * k * res13["avg_net_per_year"] / LB.BANK
    b = res13["bar"]
    out.update({"components": comps, "as_built": rb.brief(res13), "largest_position_mw": round(big, 3),
                "sizing": {"multiplier_for_100k_drawdown_capped_at_5MW_per_zone_hour": round(float(k), 3),
                           "largest_position_mw_at_that_size": round(float(k * big), 3),
                           "return_on_500k_pct": round(float(ret), 1),
                           "pass_at_sizing_view": bool(b["positive_years"] and b["sharpe_3y"] and b["stress_total"]
                                                       and ret >= rb.SIZING_BAR_PCT)},
                "stress_and_seasons": rb.stress_and_seasons(F, mw13)})
    rb.log(f"idea 13: {len(keep)} components, three-year {res13['three_year']['net_usd']:,}")
    return out


# ------------------------------------------------------------------ the report
def f0(x):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:,.0f}"


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def yn(x):
    return "n/a" if x is None else ("yes" if x else "no")


def write_md(S: dict) -> str:
    L = []
    a = L.append
    a("# Robustness checks on the strongest rows, 2021 to 2023")
    a("")
    a(f"Written {S['written']} on gene by `robust/summary.py`. Registered before they ran (OBJECTIVES.md, addendum "
      "of 6 Oct, commit b31b516; sizing decision commit 7504c92). Build years only: every table and prediction file "
      "was read through the locked loaders, which stop at 1 January 2024. These checks cannot change a registered "
      "rule; they can only add a fragile flag. Lab table used: side/lab_results.json written "
      f"{S['lab_results_written']}; lead choices written {S['choices_written']}. Scoring is side/lab.py's for every "
      "number here: 1 MW per zone-hour, full costs, 0.50 stress, Bo's bar, and the sizing view (the scale whose "
      "three-year worst drawdown is 100,000 USD, at most 5 MW).")
    a("")
    pend = S["pending"]
    a("Status: " + ("complete." if not pend else "pending: " + "; ".join(pend) + "."))
    a("")
    # seeds
    a("## Seed spread: each deep model retrained with 20 random starts")
    a("")
    a("Same walk-forward as registered (monthly refits, configuration c2), seeds 0 to 19 in one process on the GPU. "
      "Each seed is scored on its own and never selected; beside it the mean of all 20, and the six disjoint sets of "
      "three seeds (the registered ensemble size).")
    a("")
    a("| Strategy | Registered (seeds 0 to 2), three-year net | 20 seeds: three-year net, min / median / max | "
      "Sharpe, min / median / max | Seeds passing at the sizing view | Seeds passing at 1 MW | Seeds with a negative "
      "total | Mean of 20 seeds: net, Sharpe, passes at sizing view | Six sets of three seeds: net, lowest to highest |")
    a("|---|---|---|---|---|---|---|---|---|")
    for strat, r in S["seeds"].items():
        if "pending" in r:
            a(f"| {strat} | pending | | | | | | | |")
            continue
        sp, m = r["spread"], r["mean_20"]
        a(f"| {strat} | {f0(r['registered_file']['three_year_net'])} | {f0(sp['net_min'])} / {f0(sp['net_median'])} / "
          f"{f0(sp['net_max'])} | {f2(sp['sharpe_min'])} / {f2(sp['sharpe_median'])} / {f2(sp['sharpe_max'])} | "
          f"{sp['pass_at_sizing_view']} of {sp['n']} | {sp['pass_1MW']} of {sp['n']} | {sp['negative_total']} of "
          f"{sp['n']} | {f0(m['three_year_net'])}, {f2(m['sharpe'])}, {yn(m['pass_at_sizing_view'])} | "
          f"{f0(sp['triples_net_min'])} to {f0(sp['triples_net_max'])} |")
    for strat, r in S["seeds"].items():
        if "pending" not in r:
            c = r["retrained_0_2_vs_registered"]
            a("")
            a(f"{strat}: seeds 0 to 2 retrained here against the registered predictions: correlation {c['corr']:.3f}, "
              f"mean absolute difference {c['mean_abs_diff_usd']:.2f} USD per MWh; scored, "
              f"{f0(r['seeds_0_2_retrained']['three_year_net'])} against {f0(r['registered_file']['three_year_net'])} "
              f"(GPU training is not bit-for-bit repeatable). The registered file scored here "
              f"{'reproduces' if r['registered_reproduces_lab'] else 'does not reproduce'} the lab's row "
              f"({f0(r['lab_three_year_net'])}).")
    a("")
    # neighbours
    a("## Neighbouring settings")
    a("")
    a("Gradient boosting: each year's chosen configuration moved one step on one of the grid's three settings, in "
      "every year at once (C keeps each year's registered pairs). Deep: configuration c2 with dropout or weight decay "
      "one step either side, three seeds as registered. Same walk-forward and scoring.")
    a("")
    a("| Strategy | Setting | Settings 2021 / 2022 / 2023 | Three-year net | Sharpe | Passes at 1 MW | Passes at the "
      "sizing view | Same sign as the chosen setting |")
    a("|---|---|---|---|---|---|---|---|")
    G = S["gbm_neighbours"]
    for strat in ("B_gbm", "C_gbm"):
        r = (G or {}).get(strat)
        if r is None:
            a(f"| {strat} | pending | | | | | | |")
            continue
        base = r["variants"]["chosen"]["three_year_net"]
        for v, x in r["variants"].items():
            same = "" if v == "chosen" else yn(np.sign(x["three_year_net"]) == np.sign(base))
            label = "chosen" + (" (reproduces the lab)" if r["reproduces_lab"] else
                                " (differs from the lab's " + f0(r["lab_three_year_net"]) + ")") if v == "chosen" else v
            a(f"| {strat} | {label} | {' / '.join(x['settings'].values())} | {f0(x['three_year_net'])} | "
              f"{f2(x['sharpe'])} | {yn(x['PASS_1MW'])} | {yn(x['pass_at_sizing_view'])} | {same} |")
    D = S["deep_neighbours"]
    for strat in ("B_deep", "C_deep"):
        r = (D or {}).get(strat)
        if r is None:
            a(f"| {strat} | pending | | | | | | |")
            continue
        reg = r["registered"]
        a(f"| {strat} | registered c2 (dropout 0.3, weight decay 0.01) | c2 / c2 / c2 | {f0(reg['three_year_net'])} | "
          f"{f2(reg['sharpe'])} | {yn(reg['PASS_1MW'])} | {yn(reg['pass_at_sizing_view'])} | |")
        for v, x in r["variants"].items():
            if "pending" in x:
                a(f"| {strat} | {VARIANT_TEXT[v]} | pending | | | | | |")
                continue
            a(f"| {strat} | {VARIANT_TEXT[v]} | same in every year | {f0(x['three_year_net'])} | {f2(x['sharpe'])} | "
              f"{yn(x['PASS_1MW'])} | {yn(x['pass_at_sizing_view'])} | "
              f"{yn(np.sign(x['three_year_net']) == np.sign(reg['three_year_net']))} |")
    a("")
    # stress and seasons
    a("## Costs at 1.00 USD per MWh, seasons and half-years")
    a("")
    a("Net USD at 1 MW per zone-hour (idea 13 at its built size), full costs except the second column. Seasons pool "
      "the three years; winter is December, January and February of 2021 to 2023 (January and February 2024 are held "
      "out).")
    a("")
    hy = [f"{y} {h}" for y in (2021, 2022, 2023) for h in ("H1", "H2")]
    a("| Strategy | Three-year net | At 1.00 USD per MWh | Winter | Spring | Summer | Autumn | " + " | ".join(hy) + " |")
    a("|---|" + "---|" * (6 + len(hy)))
    rows = dict((S["stress_seasons"] or {}).items())
    i13 = S["idea13"]
    if i13.get("stress_and_seasons"):
        rows["idea13_portfolio"] = {"three_year_net": i13["as_built"]["three_year_net"], **i13["stress_and_seasons"]}
    for strat in rb.TOP4 + ["idea13_portfolio"]:
        r = rows.get(strat)
        if r is None or "pending" in r:
            a(f"| {strat} | pending |" + " |" * (5 + len(hy)))
            continue
        sea = [f0(r["seasons"][k]["pooled"]["net_usd"]) for k in rb.SEASONS]
        hh = [f0(r["half_years"][k]["net_usd"]) for k in hy]
        a(f"| {strat} | {f0(r['three_year_net'])} | {f0(r['stress_1.00']['three_year']['net_usd'])} | "
          + " | ".join(sea) + " | " + " | ".join(hh) + " |")
    a("")
    # idea 13
    a("## Idea 13, portfolio of the survivors")
    a("")
    if "as_built" not in i13:
        a(i13.get("result", "pending"))
    else:
        comp = "; ".join(f"{k} {v['weight_mw']:.2f} MW" for k, v in i13["components"].items())
        ab, sz = i13["as_built"], i13["sizing"]
        a("| Components (MW per position) | Three-year net | Average a year | Sharpe | Max drawdown | Passes the bar as "
          "built | Multiplier for a 100,000 drawdown | Return on 500,000 a year at that size | Passes at the sizing "
          "view |")
        a("|---|---|---|---|---|---|---|---|---|")
        a(f"| {comp} | {f0(ab['three_year_net'])} | {f0(ab['avg_net_per_year'])} | {f2(ab['sharpe'])} | "
          f"{f0(ab['max_drawdown'])} | {yn(ab['PASS_1MW'])} | {sz['multiplier_for_100k_drawdown_capped_at_5MW_per_zone_hour']} | "
          f"{sz['return_on_500k_pct']}% | {yn(sz['pass_at_sizing_view'])} |")
        a("")
        a("Each component is sized so that its own 2021 to 2023 worst drawdown is 100,000 USD (at most 5 MW), then "
          f"weighted 1/{len(i13['components'])}; positions are netted per zone-hour before costs. Components "
          "reproduce the lab's rows: " + ", ".join(f"{k} {yn(v['reproduces_lab'])}" for k, v in i13["components"].items())
          + ".")
    if i13.get("excluded_fragile"):
        a("")
        a("Pass at the sizing view but left out as fragile: " + "; ".join(
            f"{k} ({', '.join(v)})" for k, v in i13["excluded_fragile"].items()) + ".")
    a("")
    a("## What the checks show for the fragile flag")
    a("")
    for strat in rb.TOP4:
        bits = []
        nb = (S["gbm_neighbours"] or {}).get(strat) or (S["deep_neighbours"] or {}).get(strat)
        if nb and nb.get("sign_kept_at_every_neighbour") is not None:
            bits.append("keeps its sign at every neighbouring setting" if nb["sign_kept_at_every_neighbour"]
                        else "changes sign at a neighbouring setting (the lab's rule for FRAGILE)")
        sd = S["seeds"].get(strat)
        if sd and "spread" in sd:
            bits.append(f"{sd['spread']['negative_total']} of 20 seeds lose money over three years, "
                        f"{sd['spread']['pass_at_sizing_view']} of 20 pass at the sizing view")
        st = (S["stress_seasons"] or {}).get(strat)
        if st and "stress_1.00" in st:
            v = st["stress_1.00"]["three_year"]["net_usd"]
            bits.append(f"{'stays positive' if v > 0 else 'turns negative'} at 1.00 USD per MWh ({f0(v)})")
        a(f"- {strat}: " + ("; ".join(bits) if bits else "pending") + ".")
    a("")
    text = "\n".join(L) + "\n"
    for bad in ("\u2014", "\u2013", "\u2192", "\u2190"):
        text = text.replace(bad, ",")
    return text


def main():
    t0 = time.time()
    labres = rb.lab_results()
    choices, when = rb.lead_choices()
    F, px, lf, wx = rb.frame()
    DS = DeepScorer(F, rb.label_rows(), choices)
    S = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "lab_results_written": labres["meta"]["written"],
         "choices_written": when}
    S["seeds"] = seed_check(DS, labres)
    S["deep_neighbours"] = deep_neighbours(DS)
    S["gbm_neighbours"] = (rb.read_json(rb.OUT / "gbm_neighbours.json") or {}).get("strategies")
    S["stress_seasons"] = (rb.read_json(rb.OUT / "stress_seasons.json") or {}).get("strategies")
    S["idea13"] = idea13(F, px, lf, wx, labres)
    pend = [f"seed check for {k}" for k, v in S["seeds"].items() if "pending" in v]
    pend += [f"deep neighbours for {k}" for k, v in S["deep_neighbours"].items()
             if any("pending" in x for x in v["variants"].values())]
    if S["gbm_neighbours"] is None:
        pend.append("gradient-boosting neighbours (robust/gbm_neighbours.py)")
    if S["stress_seasons"] is None:
        pend.append("stress and seasons for the top four (robust/stress_seasons.py)")
    S["pending"] = pend
    S["seconds"] = round(time.time() - t0)
    rb.write_json(rb.OUT / "robustness_summary.json", S)
    rb.write_text(rb.OUT / "robustness_summary.md", write_md(S))
    rb.log("wrote", rb.OUT / "robustness_summary.md", f"{S['seconds']} s; pending: {pend or 'none'}")


def smoke():
    """Synthetic frame and lab results: survivors, idea 13 and the report; the deep scorer on fake predictions."""
    import json
    import tempfile

    import lab as LB
    rng = np.random.default_rng(3)
    hours = pd.date_range("2021-01-01", "2024-01-01", freq="h", tz="America/New_York", inclusive="left")
    px = pd.DataFrame({"delivery_hour": np.repeat(hours, 11), "zone": np.tile(LB.ZONES, len(hours))})
    px["da_lbmp"] = 30.0
    px["rt_lbmp"] = 30.0 + rng.normal(-1.5, 6, len(px))
    F = LB.Frame(px)

    class St:
        def __init__(self, name, mw):
            self.name, self.inputs, self.mw = name, {}, mw
    fake = {"good_a": St("good_a", -np.ones(F.NH)), "good_b": St("good_b", np.where(F.ZI < 5, -1.0, 0.0))}
    global build_strats
    real_build, real_mw = build_strats, rb.mw_of
    build_strats = lambda F, px, lf, wx, need: {k: fake[k] for k in need}   # noqa: E731
    rb.mw_of = lambda F, st: st.mw
    strategies = {}
    for k, st in fake.items():
        strategies[k] = {**rb.score(F, st.mw), "FRAGILE": False}
        strategies[k].pop("_daily")
    strategies["fragile_c"] = {**strategies["good_a"], "FRAGILE": True,
                               "fragility": {"fragile_reasons": ["three-year total changes sign"]}}
    labres = json.loads(json.dumps({"meta": {"written": "synthetic"}, "strategies": strategies}, default=rb._js))
    for k in list(labres["strategies"]):
        r = labres["strategies"][k]
        r["bar"] = {"avg_net_per_year": True, "positive_years": True, "sharpe_3y": True, "stress_total": True}
        r["PASS"] = True
        r["sizing_view"]["return_on_500k_at_that_scale_pct"] = 50.0
    i13 = idea13(F, px, None, None, labres)
    assert i13["survivors"] == ["good_a", "good_b"] and "fragile_c" in i13["excluded_fragile"], i13
    rows = px[["delivery_hour", "zone"]].copy()
    rows["delivery_date"] = rows["delivery_hour"].dt.tz_localize(None).dt.normalize()
    rows["gap"] = px["rt_lbmp"] - px["da_lbmp"]
    pairs = {"pairs": {"pairs": ["CAPITL|NORTH", "LONGIL|WEST"]}}
    DS = DeepScorer(F, rows, {"2021": pairs, "2022": pairs, "2023": pairs})
    pred = rows[["delivery_hour", "zone"]].assign(p=rng.normal(-1, 3, len(rows)))
    c, b = DS.score("C_deep", pred, "p"), DS.score("B_deep", pred, "p")
    S = {"written": "synthetic", "lab_results_written": "synthetic", "choices_written": "synthetic",
         "seeds": {"C_deep": {"pending": "x"}, "B_deep": {"pending": "x"}}, "deep_neighbours": None,
         "gbm_neighbours": None, "stress_seasons": None, "idea13": i13, "pending": ["everything else"]}
    md = write_md(S)
    for bad in ("\u2014", "\u2013", "\u2192"):
        assert bad not in md
    with tempfile.TemporaryDirectory() as tmp:
        rb.write_text(Path(tmp) / "robustness_summary.md", md)
    build_strats, rb.mw_of = real_build, real_mw
    rb.log(f"smoke ok: idea 13 of 2 survivors, three-year {i13['as_built']['three_year_net']:,}; C deep {c['three_year_net']:,}, "
           f"B deep {b['three_year_net']:,}; report {len(md.splitlines())} lines")


if __name__ == "__main__":
    smoke() if len(sys.argv) > 1 and sys.argv[1] == "smoke" else main()
