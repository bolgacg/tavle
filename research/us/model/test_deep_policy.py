"""Tests for deep_policy.py (ideas 11 and 12; owner: policy agent). Synthetic day features and prices shaped like
the real tables; the *_real test needs gene's parquet and skips elsewhere.

    cd model && python -m pytest -q -p no:cacheprovider test_deep_policy.py

  * costs equal storm_value.py's tables and fees.py; expert profits match a hand calculation;
  * holdout: rows on or after 2024-01-01 are filtered at load, and anything that slips through raises;
  * weather is masked before 2021-03-25, and a refit whose window has no weather never reads it;
  * the objectives charge costs on |position| by side (12) and mix the experts' own profits (11);
  * K models trained side by side are the same as each trained alone;
  * injected lookahead: corrupting every price not yet public at a refit (the month itself, the two-day gap, a
    late-published day, later features) never moves a position; with the filter opened the same corruption does;
  * a leaked feature (the target day's own gap) is exploited, so the machinery would show a leak if there were one;
  * the yearly lambda and the pair are chosen on earlier data only.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import deep_policy as DP
import lock
from common import TZ, ZONES, decision_time

torch = pytest.importorskip("torch")

PX_STATS = ["da_mean", "da_max", "rt_mean", "rt_max", "gap_mean", "gap_max", "dacong_mean", "rtcong_mean"]
TINY = dict(h_enc=4, h_trunk=8, h_head=8, zone_emb=2, max_epochs=8, patience=3, val_min=20, val_max=40, batch=64,
            dropout=0.1, min_col_obs=10)
MAY = dt.date(2021, 5, 1)


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


def make_synth(start="2020-01-01", end="2021-06-30", seed=0, zd_sd=0.0):
    """Prices and day features. A day-level storm signal x (known by 05:00 on D, in the load-forecast and weather
    columns) drives real time above day ahead, mostly in N.Y.C. and LONGIL; calm days pay supply. zd_sd adds a
    zone-day shift no input can predict (the leak test)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    nd = len(days)
    x = rng.normal(0, 1, nd)
    hours = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1), freq="h",
                          inclusive="left")
    di = days.get_indexer(hours.tz_localize(None).normalize())
    parts = []
    for z in ZONES:
        da = 30 + rng.normal(0, 3, len(hours))
        w = 1.0 if z in DP.DOWNSTATE else 0.2
        gap = -1.0 + 8 * w * np.maximum(x[di] - 1.0, 0) + rng.normal(0, 4, len(hours))
        if zd_sd:
            gap = gap + rng.normal(0, zd_sd, nd)[di]
        parts.append(pd.DataFrame({"delivery_hour": hours, "zone": z, "da_lbmp": da, "rt_lbmp": da + gap,
                                   "published_at": hours + pd.Timedelta(minutes=75)}))
    px = pd.concat(parts, ignore_index=True)
    c = {}
    for p in ("P1", "P2"):
        for h in range(24):
            c[f"wx__{p}__h{h:02d}"] = 10 + 5 * x + rng.normal(0, 1, nd)
    for z in DP.LF_ZONES:
        for h in range(24):
            c[f"lf__{DP.slug(z)}__h{h:02d}"] = 1000 + 300 * x + rng.normal(0, 50, nd)
            c[f"lfwow__{DP.slug(z)}__h{h:02d}"] = rng.normal(0, 30, nd)
    for z in ZONES:
        for j in range(7):
            for s in PX_STATS:
                c[f"px__{DP.slug(z)}__d{j}__{s}"] = rng.normal(0, 10, nd)
        for h in range(6):
            for k in ("rt", "gap"):
                c[f"rtnow__{DP.slug(z)}__h{h:02d}__{k}"] = rng.normal(0, 10, nd)
    c.update({"out__n_active": rng.integers(0, 20, nd).astype(float), "cal__dow_sin": np.sin(days.dayofweek * 0.9),
              "cal__dow_cos": np.cos(days.dayofweek * 0.9), "rtnow__n_hours": np.full(nd, 5.0)})
    feats = pd.DataFrame({"bid_date": days - pd.Timedelta(days=1), "delivery_date": days, **c})
    tg = DP.Targets(px)
    feats["y__book_supply_pnl"] = tg.Ps.sum(1)                 # a label in the file: must never be an input
    return px, DP.mask_weather(feats), x


@pytest.fixture(scope="module")
def synth():
    return make_synth()


def wf(feats, px, kind, ms=MAY, cfg=None, lambdas=(0.01,), seeds=(0,), **kw):
    tg = DP.Targets(px)
    me = (pd.Timestamp(ms) + pd.offsets.MonthEnd(0)).date()
    out, recs = DP.walk_forward(feats, tg, kind, ms, me, lambdas, seeds, {**TINY, **(cfg or {})}, "cpu",
                                log=lambda *_: None, **kw)
    return out.sort_values(["delivery_date", "zone", "lam"]).reset_index(drop=True), recs


def corrupt_prices(px, mask, seed=1):
    rng = np.random.default_rng(seed)
    px = px.copy()
    px.loc[mask, "rt_lbmp"] = rng.uniform(-5000, 9000, int(mask.sum()))
    return px


# ----------------------------------------------------------------------------- costs and targets
def test_costs_equal_storm_value_and_fees():
    import fees
    for y in (2020, 2021, 2022, 2023):
        assert abs(DP.COST[y] - fees.SUPPLY_RATES[y]) < 1e-9
        assert abs(DP.LOADCOST[y] - fees.RATES[y]) < 1e-9


def test_expert_profits_by_hand(synth):
    px, feats, _ = synth
    tg = DP.Targets(px)
    d, z = pd.Timestamp("2021-03-14"), "N.Y.C."              # a spring-forward day: 23 hours
    h = px[(px["zone"] == z) & (px["delivery_hour"].dt.tz_localize(None).dt.normalize() == d)]
    assert len(h) == 23
    gap = (h["rt_lbmp"] - h["da_lbmp"]).to_numpy()
    r, zi = tg.index_of([d])[0], ZONES.index(z)
    assert np.isclose(tg.Ps[r, zi], (-gap - DP.COST[2021]).sum())
    assert np.isclose(tg.Pl[r, zi], (gap - DP.LOADCOST[2021]).sum())
    pair = ("WEST", "CENTRL")
    want = sum(tg.Pl[r, ZONES.index(q)] for q in DP.DOWNSTATE) + sum(tg.Ps[r, ZONES.index(q)] for q in pair)
    assert np.isclose(DP.pair_profit(tg, pair)[r], want)
    pos = np.array([[-0.5] * 5 + [0.7] * 6])
    net = tg.net(np.array([r]), pos)
    c = np.where(pos < 0, DP.COST[2021], DP.LOADCOST[2021])
    assert np.allclose(net, pos * tg.G[r] - np.abs(pos) * tg.N[r] * c)


def test_score_is_storm_value_score(synth):
    px, _, _ = synth
    h = DP.hourly_frame(px)
    g = h[h["year"] == 2021]
    s = DP.score(g, -np.ones(len(g)))
    tg = DP.Targets(px)
    r = np.flatnonzero(tg.year == 2021)
    assert s["net_usd"] == round(float(tg.Ps[r].sum()))


# ----------------------------------------------------------------------------- holdout
def test_holdout_filtered_at_load_and_asserted(tmp_path, synth):
    px, feats, _ = synth
    late = feats.tail(3).copy()
    late["delivery_date"] = pd.to_datetime(["2023-12-31", "2024-01-01", "2024-02-01"])
    f = pd.concat([feats, late], ignore_index=True)
    f.to_parquet(tmp_path / "f.parquet")
    got = DP.load_day_features(tmp_path / "f.parquet")
    assert got["delivery_date"].max() == pd.Timestamp("2023-12-31")
    p2 = px.head(5).copy()
    p2["delivery_hour"] = pd.date_range(pd.Timestamp("2023-12-31 22:00", tz=TZ), periods=5, freq="h")
    pd.concat([px, p2], ignore_index=True).to_parquet(tmp_path / "p.parquet")
    got = DP.load_prices(tmp_path / "p.parquet")
    assert got["delivery_hour"].max() < DP.END
    with pytest.raises((AssertionError, lock.HoldoutLocked)):
        DP.Targets(pd.concat([px, p2], ignore_index=True))
    with pytest.raises(lock.HoldoutLocked):
        DP.walk_forward(feats, DP.Targets(px), 12, dt.date(2023, 12, 1), dt.date(2024, 1, 31), (0.01,), (0,), TINY, "cpu")


# ----------------------------------------------------------------------------- weather mask
def test_weather_masked_before_start(synth):
    _, feats, _ = synth
    wx = [c for c in feats.columns if c.startswith("wx__")]
    early = feats["delivery_date"] < DP.WX_START
    assert feats.loc[early, wx].isna().all().all() and feats.loc[~early, wx].notna().all().all()
    prep = DP.Prep(feats[feats["delivery_date"] < pd.Timestamp("2021-03-01")], min_obs=10)
    a = prep.arrays(feats[feats["delivery_date"] >= pd.Timestamp("2021-04-01")])
    assert (a["wx_frac"] == 0).all() and (a["wx"] == 0).all()     # never seen in training: not read
    prep2 = DP.Prep(feats, min_obs=10)
    assert (prep2.arrays(feats[~early])["wx_frac"] == 1).all()
    assert not any(c.startswith("y__") for c in prep2.all())


# ----------------------------------------------------------------------------- objectives
def test_objective_idea12_costs_on_abs_position():
    a = torch.tensor([[[-1.0, 0.5, 0.0] + [0.0] * 8]])
    T = {"G": torch.tensor([[[-24.0, 10.0, 99.0] + [0.0] * 8]]), "N": torch.full((1, 1, 11), 24.0),
         "supc": torch.tensor([[0.1]]), "loadc": torch.tensor([[0.08]])}
    J = DP.day_objective(12, a, None, T, torch.tensor([0.0]), 0.0, 1.0)
    want = (-1 * -24 - 1 * 24 * 0.1) + (0.5 * 10 - 0.5 * 24 * 0.08)
    assert torch.isclose(J[0, 0], torch.tensor(want))
    J2 = DP.day_objective(12, -a, None, {**T, "G": -T["G"]}, torch.tensor([0.5]), 0.1, 10.0)
    p = (1 * -24 * -1 - 24 * 0.08 + (-0.5) * -10 - 0.5 * 24 * 0.1) / 10.0
    assert torch.isclose(J2[0, 0], torch.tensor(p - 0.5 * max(0, -p) ** 2 - 0.1 * 1.25))


def test_objective_idea11_mixed_profit():
    g = torch.zeros(1, 1, 11, 3)
    g[0, 0, 0] = torch.tensor([2.0, 0.0, -1.0])
    aux = torch.tensor([[0.3]])
    T = {"Ps": torch.full((1, 1, 11), 3.0), "Pl": torch.full((1, 1, 11), -5.0), "Pp": torch.tensor([[-40.0]])}
    J = DP.day_objective(11, g, aux, T, torch.tensor([0.0]), 0.0, 1.0)
    w = torch.softmax(g, -1)[0, 0]
    pi = torch.sigmoid(aux)[0, 0]
    want = (1 - pi) * (w[:, 0] * 3 + w[:, 2] * -5).sum() + pi * -40
    assert torch.isclose(J[0, 0], want)
    W = DP.mixed_weights(g, aux)
    assert torch.allclose(W.sum(-1), torch.ones(1, 1, 11))


def test_models_side_by_side_equal_models_alone(synth):
    px, feats, _ = synth
    tg = DP.Targets(px)
    tr = feats[(feats["delivery_date"] < pd.Timestamp("2021-04-01"))]
    cfg = {**TINY, "dropout": 0.0}
    m2 = DP.PolicyModel(12, (0.01, 0.1), (0, 1), cfg, "cpu")
    m2.fit(tr, tg)
    m1 = DP.PolicyModel(12, (0.1,), (1,), cfg, "cpu")
    m1.fit(tr, tg)
    k = int(np.flatnonzero((m2.lam_k == 0.1) & (m2.seed_k == 1))[0])
    assert m2.info["best_epochs"]["0.1"][1] == m1.info["best_epochs"]["0.1"][0]
    for (n, p2), (_, p1) in zip(m2.net.named_parameters(), m1.net.named_parameters()):
        assert torch.allclose(p2[k], p1[0], atol=1e-5), n


# ----------------------------------------------------------------------------- injected lookahead
@pytest.mark.parametrize("kind", [11, 12])
def test_unpublished_prices_never_move_positions(synth, kind):
    px, feats, _ = synth
    base, recs = wf(feats, px, kind)
    assert pd.Timestamp(recs[0]["train_last"]) <= pd.Timestamp(MAY) - pd.Timedelta(days=2)
    ld = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    late = (ld >= pd.Timestamp(MAY) - pd.Timedelta(days=1)).to_numpy()       # the gap days, the month, after
    px_bad = corrupt_prices(px, late)
    f_bad = feats.copy()
    after = f_bad["delivery_date"] > pd.Timestamp("2021-05-31")
    num = [c for c in f_bad.columns if "__" in c]
    f_bad.loc[after, num] = 1e6                                             # later days' features
    got, _ = wf(f_bad, px_bad, kind)
    assert np.array_equal(got["position"].to_numpy(), base["position"].to_numpy())
    opened, _ = wf(feats, px_bad, kind, gap_days=-31, known_shift=pd.Timedelta(days=40))
    ref, _ = wf(feats, px, kind, gap_days=-31, known_shift=pd.Timedelta(days=40))
    assert not np.allclose(opened["position"].to_numpy(), ref["position"].to_numpy())


def test_late_published_day_is_left_out(synth):
    px, feats, _ = synth
    day = pd.Timestamp("2021-04-20")
    ld = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    on = (ld == day).to_numpy()
    late = px.copy()
    late.loc[on, "published_at"] = decision_time(dt.date(2021, 4, 30)) + pd.Timedelta(hours=1)  # after 05:00 on 30 Apr
    a, ra = wf(feats, late, 12)
    b, rb = wf(feats, corrupt_prices(late, on), 12)
    assert ra[0]["n_days"] == rb[0]["n_days"]
    assert np.array_equal(a["position"].to_numpy(), b["position"].to_numpy())
    c, rc = wf(feats, corrupt_prices(px, on), 12)                          # control: public in time, so used
    d, rd = wf(feats, px, 12)
    assert rc[0]["n_days"] == rd[0]["n_days"] == ra[0]["n_days"] + 1
    assert not np.array_equal(c["position"].to_numpy(), d["position"].to_numpy())


def test_leaked_target_is_exploited():
    """Positive control: overwrite one input per zone with the target day's own gap (in data where that gap is mostly
    unpredictable); the policy must earn far more than without it."""
    px, feats, _ = make_synth(seed=5, zd_sd=8.0)
    tg = DP.Targets(px)
    leak = feats.copy()
    r = tg.index_of(leak["delivery_date"])
    for z in ZONES:
        leak[f"px__{DP.slug(z)}__d0__gap_mean"] = tg.G[r, ZONES.index(z)] / 24
    cfg = {"max_epochs": 40, "patience": 10, "lr": 3e-3, "dropout": 0.0}
    nets = {}
    for name, f in (("clean", feats), ("leak", leak)):
        out, _ = wf(f, px, 12, cfg=cfg)
        P = out.pivot(index="delivery_date", columns="zone", values="position").reindex(columns=ZONES)
        nets[name] = float(tg.net(tg.index_of(P.index), P.to_numpy()).sum())
    assert nets["leak"] > nets["clean"] + 10000, nets


# ----------------------------------------------------------------------------- choices
def test_lambda_choice_uses_prior_year_only(synth):
    px, _, _ = synth
    tg = DP.Targets(px)
    days = tg.days[(tg.days >= "2020-07-01")]
    rows = []
    for l, sgn in ((0.001, -1.0), (0.01, 1.0), (0.1, 0.0)):
        for z in ZONES:
            rows.append(pd.DataFrame({"delivery_date": days, "zone": z, "lam": l, "position": sgn}))
    allpos = pd.concat(rows, ignore_index=True)
    ch = DP.choose_lambdas(allpos, tg, (2021,))
    nets = {l: float(tg.net(tg.index_of(days[(days.year == 2020) & (days <= "2020-12-30")]),
                            np.full((len(days[(days.year == 2020) & (days <= "2020-12-30")]), 11), s)).sum())
            for l, s in ((0.001, -1.0), (0.01, 1.0), (0.1, 0.0))}
    assert ch[2021]["lam"] == max(nets, key=nets.get)
    bad = allpos.copy()
    m = (bad["delivery_date"] >= "2020-12-31").to_numpy()
    bad.loc[m, "position"] = np.random.default_rng(3).uniform(-1, 1, int(m.sum()))
    px2 = corrupt_prices(px, (px["delivery_hour"].dt.tz_localize(None) >= pd.Timestamp("2020-12-31")).to_numpy())
    ch2 = DP.choose_lambdas(bad, DP.Targets(px2), (2021,))
    assert ch2[2021]["lam"] == ch[2021]["lam"]
    assert DP.choose_lambdas(allpos[allpos["delivery_date"] >= "2021-01-01"], tg, (2021,))[2021]["how"].startswith("declared")


def test_pair_chosen_on_the_window_only(synth):
    px, feats, _ = synth
    tg = DP.Targets(px)
    rows = DP.train_rows(feats, tg, dt.date(2021, 1, 1))
    pair, info = DP.choose_pair(tg, rows)
    assert pair in DP.PAIRS and info["storm_days"] >= 1
    px2 = corrupt_prices(px, (px["delivery_hour"].dt.tz_localize(None) >= pd.Timestamp("2020-12-31")).to_numpy())
    tg2 = DP.Targets(px2)
    assert DP.choose_pair(tg2, DP.train_rows(feats, tg2, dt.date(2021, 1, 1)))[0] == pair


def test_output_schema(synth):
    px, feats, _ = synth
    out, recs = wf(feats, px, 11, lambdas=(0.01, 0.1))
    for c in ("delivery_date", "bid_date", "zone", "idea", "lam", "position", "refit_month", "w_supply", "w_out",
              "w_load", "w_pair", "pair_up1", "pair_up2"):
        assert c in out.columns
    assert (pd.to_datetime(out["refit_month"]) == out["delivery_date"].dt.to_period("M").dt.start_time).all()
    assert out["position"].between(-1, 1).all()
    assert np.allclose(out[["w_supply", "w_out", "w_load", "w_pair"]].sum(axis=1), 1.0)
    assert set(out["lam"]) == {0.01, 0.1} and len(out) == 2 * 31 * 11
    pv = DP.pair_vector((out["pair_up1"].iloc[0], out["pair_up2"].iloc[0]))
    zi = out["zone"].map(ZONES.index).to_numpy()
    assert np.allclose(out["position"], -out["w_supply"] + out["w_load"] + out["w_pair"] * pv[zi], atol=1e-6)


# ----------------------------------------------------------------------------- real data (gene only)
@pytest.mark.skipif(not DP.FEATS.exists(), reason="needs gene's day-feature matrix")
def test_real_unpublished_prices_never_move_positions():
    feats = DP.load_day_features()
    px = DP.load_prices()
    ms = dt.date(2022, 12, 1)
    feats = feats[feats["delivery_date"] <= pd.Timestamp("2022-12-31")]
    a, ra = wf(feats, px, 12, ms=ms, cfg={"max_epochs": 3, "patience": 2})
    ld = px["delivery_hour"].dt.tz_localize(None).dt.normalize()
    b, _ = wf(feats, corrupt_prices(px, (ld >= pd.Timestamp("2022-11-30")).to_numpy()), 12, ms=ms,
              cfg={"max_epochs": 3, "patience": 2})
    assert pd.Timestamp(ra[0]["train_last"]) <= pd.Timestamp("2022-11-29")
    assert np.array_equal(a["position"].to_numpy(), b["position"].to_numpy())
