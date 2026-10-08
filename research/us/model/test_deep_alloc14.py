"""Tests for deep_alloc14.py (idea 14). Synthetic tables from test_deep_policy.make_synth plus four synthetic
candidates; run on gene:  cd model && ../.venv/bin/python -m pytest -q -p no:cacheprovider test_deep_alloc14.py

  * candidate profits match a hand calculation; the drawdown scale is right by hand;
  * holdout: candidate rows on or after 2024-01-01 are filtered at load;
  * a candidate with fewer than 60 training days takes no part (weight 0, scale 0);
  * injected lookahead: corrupting every price not yet public at a refit (the gap days, the month, later) and later
    features moves neither the weights nor the prior-only scales; with the filter opened the same corruption does;
  * a large KL strength pins the weights to equal (the idea 13 fallback);
  * the yearly kappa is chosen on earlier data only;
  * a leaked feature (which candidate wins the target day) is exploited, so the machinery would show a leak if there were one.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import deep_alloc14 as A
import deep_policy as DP
from common import TZ, ZONES
from test_deep_policy import TINY, corrupt_prices, make_synth

torch = pytest.importorskip("torch")
MAY = dt.date(2021, 5, 1)
DOWN = set(DP.DOWNSTATE)
UP = set(DP.UPSTATE)


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


def make_cands(px, late_start="2021-03-15"):
    h = px[["delivery_hour", "zone"]].copy()
    z = h["zone"]
    parts = {"C_deep": -np.ones(len(h)), "B_gbm": np.where(z.isin(DOWN), 1.0, 0.0),
             "C_gbm": np.where(z.isin(UP), -1.0, 0.0), "B_deep": np.where(z.isin(DOWN), 1.0, -0.5)}
    out = []
    for c, mw in parts.items():
        g = h.assign(mw=mw, strategy=c)
        if c == "B_deep":
            g = g[g["delivery_hour"] >= pd.Timestamp(late_start, tz=TZ)]
        out.append(g)
    return pd.concat(out, ignore_index=True)


@pytest.fixture(scope="module")
def synth():
    px, feats, x = make_synth()
    return px, feats, make_cands(px)


def wf(feats, px, cand, kappas=(0.01,), cfg=None, ms=MAY, **kw):
    tg = DP.Targets(px)
    cp = A.CandProfits(px, cand, tg)
    me = (pd.Timestamp(ms) + pd.offsets.MonthEnd(0)).date()
    out, recs = A.walk_forward(feats, tg, cp, ms, me, kappas, (0,), {**TINY, **(cfg or {})}, "cpu",
                               log=lambda *_: None, **kw)
    return out.sort_values(["delivery_date", "kappa"]).reset_index(drop=True), recs, tg, cp


def W(out):
    return out[[f"w_{c}" for c in A.CANDS] + [f"scale_{c}" for c in A.CANDS]].to_numpy()


def test_candidate_profits_by_hand(synth):
    px, _, cand = synth
    tg = DP.Targets(px)
    cp = A.CandProfits(px, cand, tg)
    assert np.allclose(cp.P[:, 0], tg.Ps.sum(1))                              # always supply
    dn = [ZONES.index(z) for z in DP.DOWNSTATE]
    assert np.allclose(cp.P[:, 1], tg.Pl[:, dn].sum(1))
    late = tg.days < pd.Timestamp("2021-03-15")
    assert np.isnan(cp.P[late, 3]).all() and np.isfinite(cp.P[~late, 3]).all()


def test_dd_scale_by_hand():
    assert A.dd_scale(np.array([10.0, -30_000, 5_000, -30_000])) == pytest.approx(100_000 / 55_000)
    assert A.dd_scale(np.array([1.0, 2.0])) == A.SCALE_CAP
    assert A.dd_scale(np.array([-1_000.0])) == A.SCALE_CAP


def test_holdout_filtered_at_load(tmp_path, synth):
    px, _, cand = synth
    extra = cand.iloc[:3].copy()
    extra["delivery_hour"] = pd.Timestamp("2024-01-02 05:00", tz=TZ)
    p = tmp_path / "lead.parquet"
    pd.concat([cand.iloc[:100], extra]).to_parquet(p)
    got = A.load_candidates(p)
    assert len(got) == 100 and got["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    with pytest.raises(AssertionError):
        DP.assert_build(extra["delivery_hour"], "x")


def test_short_history_takes_no_part(synth):
    px, feats, cand = synth
    out, recs, _, _ = wf(feats, px, cand)
    assert recs[0]["taking_part"] == ["C_deep", "B_gbm", "C_gbm"]
    assert (out["w_B_deep"] == 0).all() and (out["scale_B_deep"] == 0).all()
    w = out[[f"w_{c}" for c in A.CANDS]].to_numpy()
    assert np.allclose(w.sum(1), 1.0)
    assert pd.Timestamp(recs[0]["train_last"]) <= pd.Timestamp(MAY) - pd.Timedelta(days=2)


def test_unpublished_prices_never_move_weights_or_scales(synth):
    px, feats, cand = synth
    base, _, _, _ = wf(feats, px, cand)
    ld = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    px_bad = corrupt_prices(px, (ld >= pd.Timestamp(MAY) - pd.Timedelta(days=1)).to_numpy())
    f_bad = feats.copy()
    num = [c for c in f_bad.columns if "__" in c]
    f_bad.loc[f_bad["delivery_date"] > pd.Timestamp("2021-05-31"), num] = 1e6
    got, _, _, _ = wf(f_bad, px_bad, cand)
    assert np.array_equal(W(got), W(base))
    opened, _, _, _ = wf(feats, px_bad, cand, gap_days=-31, known_shift=pd.Timedelta(days=40))
    ref, _, _, _ = wf(feats, px, cand, gap_days=-31, known_shift=pd.Timedelta(days=40))
    assert not np.allclose(W(opened), W(ref))


def test_large_kl_gives_equal_weights(synth):
    px, feats, cand = synth
    out, _, _, _ = wf(feats, px, cand, kappas=(1e4,))
    w = out[["w_C_deep", "w_B_gbm", "w_C_gbm"]].to_numpy()
    assert np.allclose(w, 1 / 3, atol=0.01)


def test_kappa_choice_uses_prior_year_only(synth):
    px, _, cand = synth
    tg = DP.Targets(px)
    cp = A.CandProfits(px, cand, tg)
    days = pd.date_range("2020-01-01", "2021-06-30")
    rng = np.random.default_rng(3)
    allw = pd.concat([pd.DataFrame({"delivery_date": days, "kappa": k,
                                    **{f"w_{c}": rng.dirichlet(np.ones(4), len(days))[:, j] for j, c in enumerate(A.CANDS)},
                                    **{f"scale_{c}": 1.0 for c in A.CANDS}}) for k in A.KAPPAS], ignore_index=True)
    ch = A.choose_kappas(allw, tg, cp, (2020, 2021))
    assert ch[2020]["how"].startswith("declared default") and ch[2020]["kappa"] == A.KAPPAS[1]
    ld = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    px_bad = corrupt_prices(px, (ld >= pd.Timestamp("2020-12-31")).to_numpy())
    tg2 = DP.Targets(px_bad)
    ch2 = A.choose_kappas(allw, tg2, A.CandProfits(px_bad, cand, tg2), (2021,))
    assert ch2[2021]["kappa"] == ch[2021]["kappa"]
    assert ch2[2021]["net_by_kappa"] == ch[2021]["net_by_kappa"]


def test_leaked_target_is_exploited():
    """Positive control: one input says which of two opposite candidates wins the target day, in data where that
    is mostly unpredictable; the gate must earn clearly more with it than without."""
    px, feats, _ = make_synth(seed=5, zd_sd=8.0)
    cand = make_cands(px, late_start="2020-01-01")
    tg = DP.Targets(px)
    leak = feats.copy()
    r = tg.index_of(leak["delivery_date"])
    cp0 = A.CandProfits(px, cand, tg)
    leak["out__n_active"] = np.sign(cp0.P[r, 1] - cp0.P[r, 0])      # which of two opposite candidates wins
    cfg = {"max_epochs": 40, "patience": 10, "lr": 3e-3, "dropout": 0.0, "weight_decay": 0.0}
    nets = {}
    for name, f in (("clean", feats), ("leak", leak)):
        out, _, tg_, cp = wf(f, px, cand, kappas=(0.001,), cfg=cfg)
        nets[name] = float(A.day_net(out, tg_, cp).sum())
    assert nets["leak"] > 1.5 * nets["clean"] + 10_000, nets
