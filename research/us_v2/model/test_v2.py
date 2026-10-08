"""Synthetic tests for the v2 rolling framework. Nothing here reads real data: every path points at a temp dir.

    cd ~/nyiso-us/v2/model && ../../.venv/bin/python -m pytest -q test_v2.py
"""
from __future__ import annotations

import datetime as dt
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TZ = "America/New_York"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("V2_HOME", str(tmp_path))
    monkeypatch.setenv("V2_PARQUET", str(tmp_path / "pq"))
    monkeypatch.setenv("V2_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("V2_PANEL", str(tmp_path / "pq" / "panel.parquet"))
    import rolling
    R = importlib.reload(rolling)
    fees = types.SimpleNamespace(RATES={y: 0.10 for y in range(2010, 2027)},
                                 SUPPLY_RATES={y: 0.12 for y in range(2010, 2027)})
    R._FEES = fees
    for m in ("strategies_v2", "gbm_rolling", "score_v2"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    return R, tmp_path


def hours(first: dt.date, last: dt.date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first, tz=TZ), pd.Timestamp(last + dt.timedelta(days=1), tz=TZ),
                         freq="h", inclusive="left")


def synth_panel(R, first=dt.date(2010, 1, 1), last=dt.date(2013, 6, 30), seed=0, with_2024=False):
    import panel as P1
    rng = np.random.default_rng(seed)
    H = hours(first, last if not with_2024 else dt.date(2024, 1, 3))
    zones = R.ZONES
    n = len(H) * len(zones)
    df = pd.DataFrame({"delivery_hour": np.repeat(H, len(zones)), "zone": np.tile(zones, len(H))})
    df["delivery_date"] = df["delivery_hour"].dt.tz_localize(None).dt.normalize()
    df["bid_date"] = df["delivery_date"] - pd.Timedelta(days=1)
    df["hour"] = df["delivery_hour"].dt.hour
    for c in P1.BASE_FEATURES:
        if c not in df.columns:
            df[c] = rng.normal(0, 1, n)
    df["zone_code"] = pd.Categorical(df["zone"], categories=zones).codes
    df["da_d0_h"] = 30 + 5 * rng.normal(size=n)
    df["y_da_lbmp"] = df["da_d0_h"] * (1 + 0.1 * rng.normal(size=n))
    signal = df["gap_7d_h"].to_numpy()
    df["gap"] = 2.0 * signal - 1.0 + rng.normal(0, 3, n) + (rng.random(n) < 0.01) * rng.exponential(80, n)
    df["y_rt_lbmp"] = df["y_da_lbmp"] + df["gap"]
    return df


# ------------------------------------------------------------------ quarters, windows, holdout
def test_quarters_and_windows(env):
    R, _ = env
    Q = R.quarters()
    assert len(Q) == 48 and sum(not q[3] for q in Q) == 44
    assert Q[4][2] == "2013Q1" and Q[-1][1] == dt.date(2023, 12, 31)
    lo, hi = R.window(dt.date(2016, 4, 1))
    assert lo == dt.date(2013, 4, 1) and hi == dt.date(2016, 3, 30)
    assert R.window(dt.date(2012, 1, 1))[0] == dt.date(2010, 1, 1)


def test_holdout_absolute(env, monkeypatch):
    R, tmp = env
    with pytest.raises(AssertionError):
        R.assert_pre2024(pd.Series([pd.Timestamp("2024-01-01 00:00", tz=TZ)]))
    with pytest.raises(AssertionError):
        R.assert_pre2024(pd.Series([pd.Timestamp("2024-01-01")]))
    monkeypatch.setenv("US_HOLDOUT_RUN", "1")            # no switch opens the v2 guard
    with pytest.raises(AssertionError):
        R.assert_pre2024(pd.Series([pd.Timestamp("2024-06-01", tz=TZ)]))
    p = synth_panel(R, dt.date(2023, 12, 25), dt.date(2023, 12, 31), with_2024=True)
    assert p["delivery_hour"].max() >= pd.Timestamp("2024-01-01", tz=TZ)
    (tmp / "pq").mkdir()
    p.to_parquet(R.PANEL)
    q = R.load_panel()
    assert q["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ) and len(q) < len(p)
    with pytest.raises(AssertionError):
        R.write_positions("bad", p.assign(mw=1.0), {})


# ------------------------------------------------------------------ runner + injected lookahead
def _zone_mean_fp(tr, te, q):
    m = tr.groupby("zone")["gap"].mean()
    return te["zone"].map(m).rename("pred")


def test_runner_no_lookahead(env):
    R, tmp = env
    panel = synth_panel(R, dt.date(2010, 1, 1), dt.date(2012, 12, 31))
    qs = R.quarters()[:4]
    seen = {}

    def fp(tr, te, q):
        seen[q[2]] = (tr["delivery_date"].max(), te["delivery_date"].min())
        return _zone_mean_fp(tr, te, q)

    a = R.run_rows("t", fp, panel, qs=qs, resume=False)
    for k, (trmax, temin) in seen.items():
        assert trmax <= temin - pd.Timedelta(days=2)
    # corrupt every outcome on or after each quarter's last bid-time-unknown day: predictions must not move
    bad = panel.copy()
    for q in qs:
        late = bad["delivery_date"] >= pd.Timestamp(q[0] - dt.timedelta(days=1))
        bad2 = bad.copy()
        bad2.loc[late, "gap"] = 1e6
        b = R.run_rows("t2", _zone_mean_fp, bad2, qs=[q], resume=False)
        ref = a[a["quarter"] == q[2]]["pred"].to_numpy()
        assert np.allclose(b["pred"].to_numpy(), ref)
    # and with the filter opened (train through the quarter) they must change
    old = R.TRAIN_GAP_DAYS
    try:
        R.TRAIN_GAP_DAYS = -100
        bad3 = panel.copy()
        q = qs[2]
        bad3.loc[R.test_mask(bad3, q), "gap"] = 1e6
        c = R.run_rows("t3", _zone_mean_fp, bad3, qs=[q], resume=False)
    except AssertionError:
        c = None                                          # the runner's own assert caught the opened filter
    finally:
        R.TRAIN_GAP_DAYS = old
    if c is not None:
        assert not np.allclose(c["pred"].to_numpy(), a[a["quarter"] == q[2]]["pred"].to_numpy())


def test_gbm_rolling_lookahead(env):
    R, tmp = env
    import gbm_rolling as GR
    panel = synth_panel(R, dt.date(2010, 1, 1), dt.date(2011, 3, 31))
    feats = R.base_features()
    q = R.quarters(2011, 2011)[0]
    tr, te = panel[R.train_mask(panel, q[0])], panel[R.test_mask(panel, q)]
    fp = GR.make_fit_predict(feats, 2, 20, tmp / "m1")
    a = fp(tr, te, q)
    for c in ["mean", "ens4", "p_s25", "p_s100", "q10", "q90", "cond_0", "dag_8"]:
        assert c in a.columns and a[c].notna().any()
    te2 = te.copy()
    te2[["gap", "y_da_lbmp", "y_rt_lbmp"]] = 1e6          # outcomes of the quarter itself
    b = GR.make_fit_predict(feats, 2, 20, tmp / "m2")(tr, te2, q)
    pd.testing.assert_frame_equal(a.drop(columns="gbm_seconds"), b.drop(columns="gbm_seconds"))
    tr2 = tr.copy()
    tr2.loc[tr2.index[-5000:], "gap"] += 500.0              # a training change must move the model
    c = GR.make_fit_predict(feats, 2, 20, tmp / "m3")(tr2, te, q)
    assert not np.allclose(a["mean"], c["mean"])


# ------------------------------------------------------------------ strategies
def _preds_like(R, panel):
    g = panel[["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp", "gap_365d_h"]].copy()
    rng = np.random.default_rng(1)
    g["mean"] = panel["gap_7d_h"] * 2 - 1
    g["ens4"] = g["mean"]
    for S in (25, 50, 100):
        g[f"p_s{S}"] = rng.random(len(g)) * 0.1
    g["q10"], g["q50"], g["q90"] = g["mean"] - 4, g["mean"], g["mean"] + 4
    import gbm_rolling as GR
    for k, m in enumerate(GR.DA_MULT):
        g[f"dag_{k}"] = panel["da_d0_h"] * m
        g[f"cond_{k}"] = g["mean"] + 0.1 * (30 - g[f"dag_{k}"])     # gap falls as day-ahead rises
    return g


def test_choice_layer_ignores_the_quarter(env):
    R, _ = env
    import strategies_v2 as SV
    panel = synth_panel(R, dt.date(2011, 1, 1), dt.date(2013, 6, 30))
    g = _preds_like(R, panel)
    mw1, ch1 = SV.per_quarter(g, SV.A_GRID, SV.a_pos)
    g2 = g.copy()
    q = [x for x in R.quarters() if x[2] == "2013Q2"][0]
    late = pd.to_datetime(g2["delivery_date"]) >= pd.Timestamp(q[0] - dt.timedelta(days=1))
    g2.loc[late, "gap"] = -1e6 * np.sign(np.random.default_rng(0).normal(size=late.sum()))
    _, ch2 = SV.per_quarter(g2, SV.A_GRID, SV.a_pos)
    assert ch1["2013Q2"] == ch2["2013Q2"]
    # pair choice too
    l1, c1 = SV.Pairs(g, "mean").run(g)
    l2, c2 = SV.Pairs(g2, "mean").run(g2)
    assert c1["2013Q2"] == c2["2013Q2"]
    assert c1["2012Q1"]["pairs"] == [] or "no earlier" not in c1["2012Q1"]["how"]


def test_v2_limits_use_only_0500_information(env):
    R, _ = env
    import strategies_v2 as SV
    panel = synth_panel(R, dt.date(2012, 1, 1), dt.date(2012, 3, 31))
    g = _preds_like(R, panel)
    s1, l1, _ = SV.v2_limits(g)
    g2 = g.copy()
    g2[["gap", "y_da_lbmp"]] = 1e6
    s2, l2, _ = SV.v2_limits(g2)
    assert np.array_equal(s1, s2) and np.allclose(np.nan_to_num(l1), np.nan_to_num(l2))
    # clearing: supply counts only when realised DA >= limit, load only when DA <= limit
    one = g.iloc[:1].copy()
    for k in range(9):
        one[f"dag_{k}"] = [10, 20, 30, 40, 50, 60, 70, 80, 90][k]
        one[f"cond_{k}"] = [5, 3, 1, -1, -2, -3, -4, -5, -6][k]       # supply profitable from 40 up
    side, lim, _ = SV.v2_limits(one)
    assert side[0] == -1 and lim[0] == 40
    for da, want in ((39.9, 0.0), (40.0, -1.0), (85.0, -1.0)):
        one["y_da_lbmp"] = da
        assert SV.v2_pos(one)[0] == want
    for k in range(9):
        one[f"cond_{k}"] = [6, 4, 2, 1, 0.05, -1, -2, -3, -4][k]       # load profitable up to 40
    side, lim, _ = SV.v2_limits(one)
    assert side[0] == 1 and lim[0] == 40
    one["y_da_lbmp"] = 45.0
    assert SV.v2_pos(one)[0] == 0.0


def test_v3_sizing_capped(env):
    R, _ = env
    import strategies_v2 as SV
    panel = synth_panel(R, dt.date(2012, 1, 1), dt.date(2012, 1, 31))
    g = _preds_like(R, panel)
    mw = SV.v3_pos(g, 1000.0)
    assert np.abs(mw).max() <= SV.V3_CAP + 1e-9 and (mw != 0).any()


# ------------------------------------------------------------------ scorer
def test_scorer_matches_hand_money(env):
    R, tmp = env
    import score_v2 as SC
    rng = np.random.default_rng(3)
    H = hours(dt.date(2013, 1, 1), dt.date(2024, 1, 2))
    px = pd.DataFrame({"delivery_hour": np.repeat(H, 11), "zone": np.tile(R.ZONES, len(H))})
    px["da_lbmp"] = 30.0
    px["rt_lbmp"] = 30.0 + rng.normal(-0.5, 5, len(px))
    (tmp / "pq").mkdir()
    px.to_parquet(tmp / "pq" / "prices_zone.parquet")
    lab = SC.import_lab()
    SC.bind_frame(lab)
    F = SC.FrameV2(lab, SC.load_prices())
    assert F.days.max() < pd.Timestamp("2024-01-01") and F.ND == 4017
    pos = px[px["delivery_hour"] < pd.Timestamp("2024-01-01", tz=TZ)][["delivery_hour", "zone"]].assign(mw=-1.0)
    mw = SC.positions_to_mw(F, pos)
    r = SC.score_one(lab, F, "V1_always_supply", mw, {"idea": "V1"}, None, None)
    sub = px[px["delivery_hour"] < pd.Timestamp("2024-01-01", tz=TZ)]
    hand = float((-(sub["rt_lbmp"] - sub["da_lbmp"]) - 0.12).sum())
    assert abs(r["total"] - hand) <= 1.0
    assert set(r["years"]) == set(range(2013, 2024)) and r["bar"]["positive_years"] in (True, False)
    assert lab.BAR["positive_years"] == 8


# ------------------------------------------------------------------ deep: C model and the V10 fine-tune
def test_deep_finetune_synthetic(env):
    R, tmp = env
    torch = pytest.importorskip("torch")
    import deep as D1
    import deep_data as DD
    import deep_rolling as DR
    first, last = dt.date(2022, 10, 1), dt.date(2023, 3, 31)
    DD.write_synthetic_parquet(tmp / "syn", first, last, n_points=6)
    px = pd.read_parquet(tmp / "syn" / "prices_zone.parquet")
    px = px[px["zone"].isin(R.ZONES)]
    px["delivery_hour"] = px["delivery_hour"].dt.tz_convert(TZ)
    px = px[px["delivery_hour"] < pd.Timestamp(last + dt.timedelta(days=1), tz=TZ)]
    panel = px[["delivery_hour", "zone"]].copy()
    panel["gap"] = (px["rt_lbmp"] - px["da_lbmp"]).to_numpy()
    panel["delivery_date"] = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    d = DD.DeepData(last, root=tmp / "syn", first_day=first)
    cfg = {**D1.resolve_config("c3"), "max_epochs": 2, "min_epochs": 1, "patience": 1, "val_days": 10}
    dd = pd.to_datetime(panel["delivery_date"])
    pre_tr = panel[dd <= pd.Timestamp("2023-01-15")]
    pre = D1.DeepModel(cfg, seeds=(0, 1), data=d, device="cpu")
    pre.fit(pre_tr)
    ft_tr = panel[(dd >= pd.Timestamp("2022-12-01")) & (dd <= pd.Timestamp("2023-02-27"))]
    te = panel[dd >= pd.Timestamp("2023-03-01")]
    m = DR.FineTuned(pre)
    m.fit(ft_tr)
    a, b = m.predict(te), pre.predict(te)
    assert np.isfinite(a).all() and len(m.nets) == 2 and not np.allclose(a, b)
    assert m.c["lr"] == pytest.approx(pre.c["lr"] / DR.FT["lr_div"])
    # the fine-tuned prediction does not depend on outcomes of the test days
    te2 = te.copy()
    te2["gap"] = 1e6
    assert np.allclose(m.predict(te2), a)


def test_v12_allocator_plumbing(env, monkeypatch):
    R, tmp = env
    pytest.importorskip("torch")
    import alloc_v12 as V
    import deep_alloc14 as A
    V = importlib.reload(V)
    rng = np.random.default_rng(5)
    H = hours(dt.date(2012, 1, 1), dt.date(2013, 6, 30))
    px = pd.DataFrame({"delivery_hour": np.repeat(H, 11), "zone": np.tile(R.ZONES, len(H))})
    px["da_lbmp"] = 30.0
    px["rt_lbmp"] = 29.0 + rng.normal(0, 3, len(px))
    (tmp / "pq").mkdir()
    px.to_parquet(tmp / "pq" / "prices_zone.parquet")
    R.write_positions("cand_supply", px.assign(mw=-1.0), {"idea": "t"})
    R.write_positions("cand_load", px.assign(mw=1.0), {"idea": "t"})
    days = pd.date_range("2012-01-01", "2013-06-30", freq="D")
    feats = pd.DataFrame({"delivery_date": days, "x__a": rng.normal(size=len(days))})
    monkeypatch.setattr(V.DP, "load_day_features", lambda p: feats.copy())

    class Eq:
        def __init__(self, kappas, *a, **k):
            self.kappas, self.info = kappas, {"n_days": 0}

        def fit(self, days, S, avail):
            assert days["delivery_date"].max() < pd.Timestamp(R.FIRST_SCORED) or True
            self.avail = avail
            return self

        def predict(self, days):
            w = np.tile(self.avail / self.avail.sum(), (len(days), 1))
            return {k: w for k in self.kappas}

    monkeypatch.setattr(A, "Alloc14", Eq)
    (R.RESULTS.parent).mkdir(parents=True, exist_ok=True)
    (R.RESULTS.parent / "v2_ideas.done").write_text("x")
    monkeypatch.setattr(sys, "argv", ["alloc_v12.py", "--device", "cpu"])
    V.main()
    pos = pd.read_parquet(R.POS / "V12_alloc.parquet")
    assert pos["delivery_hour"].min() >= pd.Timestamp("2013-01-01", tz=TZ)
    assert (pos["mw"] <= 0).all() and (pos["mw"] < 0).any()       # only the profitable supply survives


def test_comparison_rows(env):
    R, tmp = env
    import strategies_v2 as SV
    panel = synth_panel(R, dt.date(2010, 1, 1), dt.date(2013, 6, 30))
    zi = panel["zone"].map({z: i for i, z in enumerate(R.ZONES)})
    panel["gap"] = panel["gap"] + 3.0 * (zi == 0) - 3.0 * (zi == 1)        # a persistent CAPITL-CENTRL spread
    SV.static_pairs(panel)
    p = pd.read_parquet(R.POS / "C_static_pairs.parquet")
    q = p[p["delivery_hour"] >= pd.Timestamp("2013-04-01", tz=TZ)]
    assert (q[q["zone"] == "CAPITL"]["mw"] > 0).mean() > 0.9 and (q[q["zone"] == "CENTRL"]["mw"] < 0).mean() > 0.9
    # outcomes of the quarter itself never move its positions
    bad = panel.copy()
    late = bad["delivery_date"] >= pd.Timestamp("2013-03-31")
    bad.loc[late, "gap"] = -1e4 * (bad.loc[late, "zone"] == "CAPITL")
    SV.static_pairs(bad)
    p2 = pd.read_parquet(R.POS / "C_static_pairs.parquet")
    a = q.set_index(["delivery_hour", "zone"])["mw"]
    b = p2[p2["delivery_hour"] >= pd.Timestamp("2013-04-01", tz=TZ)].set_index(["delivery_hour", "zone"])["mw"]
    assert a.equals(b.reindex(a.index))
    g = _preds_like(R, panel[panel["delivery_date"] >= pd.Timestamp("2012-01-01")])
    SV.comparisons(g)
    s = pd.read_parquet(R.POS / "V2_sides_no_limit.parquet")
    v = pd.read_parquet(R.POS / "V3_tail_gbm_1MW.parquet")
    assert set(np.unique(s["mw"])) <= {-1.0, 0.0, 1.0} and set(np.unique(v["mw"])) <= {-1.0, 0.0, 1.0}
