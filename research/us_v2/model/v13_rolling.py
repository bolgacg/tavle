"""V13 runner: the all-inputs deep model on the v2 rolling quarters (OBJECTIVES.md, V13; rolling.py windows).

    python v13_rolling.py all   [--device cuda]   the whole V13 programme in priority order, one process, data loaded
                                                  once, every step restartable per quarter (rolling.run_rows parts):
        1. injected-lookahead self-test on the real tensors (a tampered cell must raise LookaheadError)
        2. full menu (gru, tcn, attn; every input) - needs results/v2_weather_data.done
        3. main rows -> results/v2/pos/V13_C_deep_all, V13_A_deep_all; results/v2/v2_v13.done
        4. no-weather menu (gru, tcn, attn without the weather group): runs first while the weather flag is missing
        5. ablations of the per-window chosen architecture: minus border, gen, load, outage (retrained), minus
           weather (the no-weather menu's run of the chosen architecture: the same model, no retrain)
        6. ablation scoring (score_v2's functions) -> results/v2/v13/v13_ablations.{json,md}; v2_v13_ablations.done
       A step is taken when it can run; while the weather flag is missing the no-weather parts run; the full model
       waits for the flag (at most --weather-wait-hours, then V13 full is reported NOT RUN).
    python v13_rolling.py smoke [--device cuda]   2 warm-up quarters, 2 seeds, 2 epochs per architecture (timing)
    python v13_rolling.py rows [--device cuda]    run modes (held-out, dry run): self-test, the full menu and the
                                                  main rows only (steps 1 to 3); the no-weather menu and the
                                                  ablations are diagnostics of the build years and are not rerun

Architecture choice (declared): for quarter q, the architecture whose OUT-OF-SAMPLE predictions earned the most
over the trailing 4 quarters (rolling.trailing_mask: outcomes public by 05:00 on q's first bid day) under the C
rule: the sum of the 5 best positive pair nets (strategies_v2.Pairs, full cost); ties and quarters without
earlier rows go to "gru" (first of the menu). The ablations retrain that chosen architecture.
Rows: V13_C_deep_all = the C pair rule on the chosen architecture's gap (settings tried 3: the menu);
V13_A_deep_all = the A rule (supply unless P(gap >= S) > p*) with (architecture, S, p*) chosen per quarter on
the trailing 4 quarters (settings tried 3 x 6). Ablation rows are diagnostics: they are written to
results/v2/v13/pos_ablations/ (never to results/v2/pos/, so neither the scorer's table nor V12 sees them).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import torch

import rolling as R
import strategies_v2 as SV
import v13_data as VD
import v13_model as VM
from timing import LookaheadError

V13DIR = R.RESULTS / "v13"
WEATHER_FLAG = R.RESULTS.parent / "v2_weather_data.done"
ABLATE = ("border", "gen", "load", "outage")              # retrained; weather comes from the no-weather menu
DEFAULT_ARCH = VM.ARCHS[0]


def name_of(arch: str, nowx: bool = False, drop: str | None = None) -> str:
    if drop:
        return f"v13_abl_minus_{drop}"
    return f"v13_{arch}" + ("_nowx" if nowx else "")


def finished(name: str) -> bool:
    parts = R.PREDS / f"{name}_parts"
    return (R.PREDS / f"{name}.parquet").exists() and all((parts / f"{q[2]}.parquet").exists() for q in R.quarters())


# ============================================================================ self-test on the real tensors
def lookahead_selftest(data: VD.Data, device, log) -> dict:
    """Make one not-yet-public cell look public (zone RT, zone DA of D+1, a generator point, the load forecast) and
    check that the batch guard raises; restore it. Then check that cells after 05:00 do not change the inputs."""
    A = VD.device_arrays(data, device)
    k = int(data.day_index(dt.date(2015, 7, 15))) + 1
    cols = torch.arange(min(50, len(data.ptids)), device=device)
    ref = VD.batch(data, [k], cols, device)
    t = int(data.dec_min[k])
    end = int(data.day_start[k])
    out = {}
    future_rt = end - 3                                    # an hour of D (bid day): RT not public at 05:00
    tests = {"zone_rt": ("Zp", (future_rt, 0, 1)), "zone_da": ("Zp", (end + 0 - 1, 0, 0)),
             "gen_rt": ("Gp", (future_rt, 0, 1)), "load_forecast": ("LFp", (k, 0, 0))}
    saved = {x: A[x].clone() for x in ("Zp", "Gp", "Hday", "LFp", "LFiss", "LFm")}
    for nm, (arr, ix) in tests.items():
        if nm == "zone_da":                       # every DA hour in the window is of D or earlier: label one D+1
            A["Hday"][end - 1] += 1
        if arr == "LFp":
            A["LFiss"][ix] = k + 5                 # a forecast vintage named after the bid day, marked public
            A["LFm"][ix] = True
        A[arr][ix] = t - 1
        try:
            VD.batch(data, [k], cols, device)
            out[nm] = "NOT CAUGHT"
        except LookaheadError as e:
            out[nm] = f"caught ({str(e)[:80]})"
        for x, v in saved.items():
            A[x].copy_(v)
    # values that are not public at 05:00 must not move any input
    zv_old = A["Zv"].clone()
    vis = (A["Zp"][end - 168:end] <= t)
    blk = A["Zv"][end - 168:end]
    blk[..., 3:][~vis[..., 1]] = 1e6
    blk[..., :3][~vis[..., 0]] = 1e6
    A["Zv"][end:end + 48] = 1e6                            # the delivery day itself and after
    new = VD.batch(data, [k], cols, device)
    same = all(torch.equal(ref[x], new[x]) for x in ("seq", "gx", "gm", "lf", "lfm", "lf_age", "dcal"))
    A["Zv"].copy_(zv_old)
    out["unpublished_values_change_nothing"] = bool(same)
    ok = all(v.startswith("caught") for kk, v in out.items() if kk != "unpublished_values_change_nothing") and same
    out["PASS"] = ok
    log(f"injected-lookahead self-test: {out}")
    if not ok:
        raise LookaheadError(f"self-test failed: {out}")
    return out


# ============================================================================ choice of architecture
class Chooser:
    """Per quarter: the architecture with the best trailing-4-quarter C-rule net (out-of-sample rows only)."""

    def __init__(self, panel: pd.DataFrame, nowx: bool = False):
        self.p = {}
        for a in VM.ARCHS:
            rows = SV.attach(R.load_preds(name_of(a, nowx)), panel, cols=("gap",))
            self.p[a] = SV.Pairs(rows, "pred")
        self.frame = pd.DataFrame({"delivery_date": self.p[VM.ARCHS[0]].dd})

    def score(self, a: str, q_start: dt.date) -> float | None:
        pr = self.p[a]
        m = R.trailing_mask(pd.DataFrame({"delivery_date": pr.dd}), q_start)
        if not m.any():
            return None
        tot = pr.pnl[m].sum(0)
        return float(np.sort(tot[tot > 0])[::-1][:SV.MAX_PAIRS].sum())

    def choose(self, q_start: dt.date) -> tuple[str, dict]:
        sc = {a: self.score(a, q_start) for a in VM.ARCHS}
        if all(v is None for v in sc.values()):
            return DEFAULT_ARCH, sc
        best = max(VM.ARCHS, key=lambda a: (sc[a] if sc[a] is not None else -np.inf, -VM.ARCHS.index(a)))
        return best, sc


def chosen_by_quarter(panel) -> dict:
    ch = Chooser(panel)
    return {q[2]: ch.choose(q[0]) for q in R.quarters()}


# ============================================================================ runs
def run_arch(data, blocks, panel, arch, drop, name, device, log, seeds, cfg=None, qs=None, arch_of=None):
    def fp(tr, te, q):
        a = arch_of[q[2]] if arch_of else arch
        m = VM.V13(data, blocks, a, drop=drop, seeds=seeds, device=device, cfg=cfg, log=log)
        m.fit(tr)
        out = m.predict(te)
        out["arch"] = a
        out["fit_s"] = m.info["fit_s"]
        out["best_epoch_mean"] = float(np.mean(m.info["best_epoch"]))
        del m
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
        return out
    t0 = time.time()
    res = R.run_rows(name, fp, panel, qs=qs, log=log)
    log(f"{name}: done in {(time.time() - t0) / 60:.1f} min")
    return res


# ============================================================================ positions
def a_pos_arch(rows, s):
    arch, S, p = s
    pr = rows[f"p_s{S}__{arch}"].to_numpy(float)
    return np.where(np.isnan(pr) | (pr > p), 0.0, -1.0)


def merged(panel, names: dict) -> pd.DataFrame:
    """One row per zone-hour with pred__<key>, p_s*__<key> for each preds file in names {key: file}."""
    base = None
    for key, f in names.items():
        p = R.load_preds(f)
        keep = ["delivery_hour", "zone", "delivery_date", "quarter", "warmup", "pred"] + [f"p_s{S}" for S in VM.SPIKES]
        p = p[keep].rename(columns={c: f"{c}__{key}" for c in ["pred"] + [f"p_s{S}" for S in VM.SPIKES]})
        base = p if base is None else base.merge(p.drop(columns=["delivery_date", "quarter", "warmup"]),
                                                 on=["delivery_hour", "zone"], how="inner", validate="one_to_one")
    return SV.attach(base, panel, cols=("gap",))


def write_main(panel, log):
    """V13_C_deep_all and V13_A_deep_all from the full menu."""
    choice = chosen_by_quarter(panel)
    g = merged(panel, {a: name_of(a) for a in VM.ARCHS})
    sel = np.full(len(g), np.nan)
    for qn, (a, _) in choice.items():
        m = (g["quarter"] == qn).to_numpy()
        sel[m] = g.loc[m, f"pred__{a}"].to_numpy()
    g["pred_sel"] = sel
    ch_json = {qn: {"arch": a, "trailing_c_net": sc} for qn, (a, sc) in choice.items()}
    SV.idea_c("V13_C_deep_all", g, "pred_sel", "V13",
              "pair rule on the all-inputs deep model; architecture (gru, tcn, attn) chosen per quarter on the "
              "trailing 4 out-of-sample quarters")
    meta_f = R.POS / "V13_C_deep_all.json"
    meta = json.loads(meta_f.read_text())
    meta.update({"settings_tried": len(VM.ARCHS), "arch_choice": ch_json})
    meta_f.write_text(json.dumps(meta, indent=1, default=str))
    grid = [(a, S, p) for a in VM.ARCHS for (S, p) in SV.A_GRID]
    mw, chA = SV.per_quarter(g, grid, a_pos_arch)
    SV.out("V13_A_deep_all", g, mw, {"idea": "V13", "line": "supply unless P(gap >= S) > p* from the all-inputs deep "
                                      "model; (architecture, S, p*) chosen per quarter", "settings_tried": len(grid),
                                      "grid": grid, "choices": chA})
    V13DIR.mkdir(parents=True, exist_ok=True)
    (V13DIR / "arch_choice.json").write_text(json.dumps(ch_json, indent=1, default=str))
    log(f"V13 main rows written; architecture choices: "
        f"{pd.Series([a for a, _ in choice.values()]).value_counts().to_dict()}")
    return choice


def write_ablations(panel, choice, log):
    """Ablation positions (C and A rules) into results/v2/v13/pos_ablations/, plus the full model's A rule on the
    C-chosen architecture (the like-for-like reference of the A ablations)."""
    out_dir = V13DIR / "pos_ablations"
    old = R.POS
    R.POS = out_dir
    try:
        g = merged(panel, {a: name_of(a) for a in VM.ARCHS} | {f"{a}_nowx": name_of(a, True) for a in VM.ARCHS})
        variants = {"full": None, "minus_weather": "nowx"}
        for d in ABLATE:
            p = R.load_preds(name_of("", drop=d))[["delivery_hour", "zone", "pred"] + [f"p_s{S}" for S in VM.SPIKES]]
            p = p.rename(columns={c: f"{c}__minus_{d}" for c in p.columns if c not in ("delivery_hour", "zone")})
            g = g.merge(p, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
            variants[f"minus_{d}"] = d
        for v, how in variants.items():
            cols = {}
            for c in ["pred"] + [f"p_s{S}" for S in VM.SPIKES]:
                x = np.full(len(g), np.nan)
                for qn, (a, _) in choice.items():
                    m = (g["quarter"] == qn).to_numpy()
                    src = f"{c}__{a}" if how is None else (f"{c}__{a}_nowx" if how == "nowx" else f"{c}__minus_{how}")
                    x[m] = g.loc[m, src].to_numpy()
                cols[c] = x
            h = g[["delivery_hour", "zone", "delivery_date", "gap"]].copy()
            for c, x in cols.items():
                h[f"{c}__{v}" if c != "pred" else "pred"] = x
            SV.idea_c(f"V13abl_C_{v}", h, "pred", "V13-ablation", f"C rule, chosen architecture, {v}")
            h2 = h.rename(columns={f"p_s{S}__{v}": f"p_s{S}" for S in VM.SPIKES})
            mw, chA = SV.per_quarter(h2, SV.A_GRID, SV.a_pos)
            SV.out(f"V13abl_A_{v}", h2, mw, {"idea": "V13-ablation", "line": f"A rule, C-chosen architecture, {v}",
                                              "settings_tried": len(SV.A_GRID), "choices": chA, "comparison": True})
    finally:
        R.POS = old
    log(f"ablation positions -> {out_dir}")


def score_ablations(log, draws=1000):
    import score_v2 as SC
    lab = SC.import_lab()
    SC.bind_frame(lab)
    F = SC.FrameV2(lab, SC.load_prices())
    RD, BT = lab.RandomDays(F, draws), lab.Boot(F.ND)
    rows = []
    src = V13DIR / "pos_ablations"
    for f in sorted(src.glob("*.parquet")):
        meta = json.loads(f.with_suffix(".json").read_text())
        mw = SC.positions_to_mw(F, R.read_pre2024(f, "delivery_hour"))
        r = SC.score_one(lab, F, f.stem, mw, meta, RD, BT)
        r.pop("_daily")
        rows.append(r)
        log(f"ablation {f.stem}: total {r['total']:,.0f}, Sharpe {r['sharpe']}")
    out = {"written": time.strftime("%Y-%m-%d %H:%M"), "note": "diagnostics, not strategies (OBJECTIVES V13)",
           "rows": rows}
    (V13DIR / "v13_ablations.json").write_text(json.dumps(out, indent=1, default=lab._js))
    years = SC.YEARS
    L = ["# V13 feature-group ablations (diagnostics, 2013 to 2023, net USD at full cost, 1 MW)", "",
         "Each row retrains the per-quarter chosen architecture with one input group removed (weather: the "
         "no-weather run of the same architecture). 'full' is the all-inputs model on the same architecture choice.",
         "", "| row | " + " | ".join(str(y) for y in years) + " | total | Sharpe | USD/MWh | PASS |",
         "|" + "---|" * (len(years) + 5)]
    for r in rows:
        L.append(f"| {r['name']} | " + " | ".join(SC._f(r["years"][y]) for y in years)
                 + f" | {SC._f(r['total'])} | {SC._f(r['sharpe'])} | {SC._f(r['usd_per_mwh'])} | "
                 f"{'yes' if r['PASS'] else 'no'} |")
    (V13DIR / "v13_ablations.md").write_text("\n".join(L) + "\n")
    log(f"-> {V13DIR / 'v13_ablations.md'}")


# ============================================================================ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "smoke", "selftest", "rows"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weather-wait-hours", type=float, default=30.0)
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    log = R.log
    V13DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap"])
    data = VD.load_data(log)
    log(f"v13 data {data.first_day}..{data.last_delivery}: {len(data.ptids)} generator points (borders excluded: "
        f"{data.external_ptids.tolist()}), grid {data.T} h; panel {panel.shape}; {time.time() - t0:.0f} s")
    st = lookahead_selftest(data, a.device, log)
    (V13DIR / "lookahead_selftest.json").write_text(json.dumps(st, indent=1))
    if a.mode == "selftest":
        return
    seeds = (0, 1, 2, 3, 4)
    if a.mode == "smoke":
        blocks = VD.DayBlocks(data, R.DAYFEATS, with_weather=WEATHER_FLAG.exists(), log=log)
        qs = R.quarters()[2:4]
        for arch in VM.ARCHS:
            t1 = time.time()
            run_arch(data, blocks, panel, arch, () if WEATHER_FLAG.exists() else ("weather",),
                     f"v13smoke_{arch}", a.device, log, seeds, cfg={"max_epochs": 2, "min_epochs": 1}, qs=qs)
            log(f"smoke {arch}: {time.time() - t1:.0f} s for 2 quarters x 2 epochs, 5 seeds")
        return

    if a.mode == "rows":
        if not WEATHER_FLAG.exists():
            raise SystemExit(f"V13 rows need {WEATHER_FLAG} (the build ran the full menu with weather)")
        blocks_full = VD.DayBlocks(data, R.DAYFEATS, with_weather=True, log=log)
        for arch in VM.ARCHS:
            run_arch(data, blocks_full, panel, arch, (), name_of(arch), a.device, log, seeds)
        write_main(panel, log)
        R.done("v2_v13")
        log(f"V13 rows complete in {(time.time() - t0) / 3600:.1f} h")
        return
    blocks_nowx = VD.DayBlocks(data, R.DAYFEATS, with_weather=False, log=log)
    blocks_full = None
    t_wait = time.time()
    choice = None
    while True:
        wx = WEATHER_FLAG.exists()
        if wx and blocks_full is None:
            blocks_full = VD.DayBlocks(data, R.DAYFEATS, with_weather=True, log=log)
        full_left = [x for x in VM.ARCHS if not finished(name_of(x))]
        nowx_left = [x for x in VM.ARCHS if not finished(name_of(x, True))]
        if wx and full_left:
            arch = full_left[0]
            run_arch(data, blocks_full, panel, arch, (), name_of(arch), a.device, log, seeds)
            continue
        if wx and not full_left and not (R.RESULTS / "v2_v13.done").exists():
            choice = write_main(panel, log)
            R.done("v2_v13")
            continue
        if nowx_left:
            arch = nowx_left[0]
            run_arch(data, blocks_nowx, panel, arch, ("weather",), name_of(arch, True), a.device, log, seeds)
            continue
        if wx and not full_left:
            choice = choice or chosen_by_quarter(panel)
            arch_of = {qn: x for qn, (x, _) in choice.items()}
            abl_left = [d for d in ABLATE if not finished(name_of("", drop=d))]
            if abl_left:
                d = abl_left[0]
                run_arch(data, blocks_full, panel, None, (d,), name_of("", drop=d), a.device, log, seeds,
                         arch_of=arch_of)
                continue
            write_ablations(panel, choice, log)
            score_ablations(log, a.draws)
            R.done("v2_v13_ablations")
            log(f"V13 complete in {(time.time() - t0) / 3600:.1f} h")
            return
        if time.time() - t_wait > a.weather_wait_hours * 3600:
            log(f"V13 full NOT RUN: no {WEATHER_FLAG.name} after {a.weather_wait_hours} h (no-weather menu is done)")
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
