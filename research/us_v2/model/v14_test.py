"""Unit tests for the V14 family on synthetic data (CPU, small).

    cd model && python -m pytest -q -p no:cacheprovider v14_test.py

  * label columns can never be inputs;
  * select_quarterly picks a setting and pairs from earlier quarters only (later gaps replaced: same choice);
  * the conformal half-widths and legs of a quarter ignore every gap after its cut (injected lookahead);
  * quantile sizing: never above 1 MW, smaller when the band is wider, zero below the break-even;
  * V14e DayModel, V14f GlobalModel and the V14a GBM window function: the prediction for a test day does not move
    when labels after the window cut and inputs of later bid days are replaced by noise; an input built from the
    label (positive control) does move it;
  * V14a and V14b GRUs (v1 synthetic tables): recency-weighted fit and non-crossing quantiles; tampered cell raises.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import rolling as R
import v14_core as C

Q = (dt.date(2016, 4, 1), dt.date(2016, 6, 30), "2016Q2", False)
FIRST, LAST = dt.date(2014, 1, 1), dt.date(2016, 6, 30)


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


def synth_panel(seed=0, first=FIRST, last=LAST):
    g = np.random.default_rng(seed)
    hours = pd.date_range(pd.Timestamp(first, tz=R.TZ), pd.Timestamp(last + dt.timedelta(days=1), tz=R.TZ),
                          freq="h", inclusive="left")
    p = pd.DataFrame({"delivery_hour": np.repeat(hours, C.NZ), "zone": np.tile(C.ZONES, len(hours))})
    p["delivery_date"] = p["delivery_hour"].dt.tz_localize(None).dt.normalize()
    zc = pd.Categorical(p["zone"], categories=C.ZONES).codes
    p["f1"] = g.normal(0, 1, len(p)).astype(np.float32)
    p["f2"] = (zc / 10.0).astype(np.float32)
    p["gap"] = 8 * p["f1"] + 5 * (zc - 5) + g.normal(0, 3, len(p))
    return p


# ============================================================================ guards
def test_labels_refused():
    for bad in (["gap"], ["y__book_supply_pnl"], ["y_da_lbmp"], ["label_published_at"]):
        with pytest.raises(C.LeakError):
            C.assert_inputs_public(["px__a"] + bad)
    C.assert_inputs_public(["px__a", "bord__x", "lf_h"])


# ============================================================================ pairs, selection, conformal, quantiles
def _rows(seed=0):
    p = synth_panel(seed, dt.date(2012, 1, 1), dt.date(2013, 12, 31))
    p["pred"] = p["gap"] + np.random.default_rng(1).normal(0, 4, len(p))
    return p


def test_select_quarterly_uses_earlier_quarters_only():
    p = _rows()
    W = C.wide(p, ["pred"])
    st = {"a": C.leg_point(W, W["pred"]), "b": -C.leg_point(W, W["pred"])}
    long0, ch0 = C.select_quarterly(W, st)
    assert ch0["2013Q1"]["setting"] == "a" and len(ch0["2013Q1"]["pairs"]) == 5
    assert ch0["2012Q1"]["setting"] is None
    cut = pd.Timestamp("2013-03-30")                       # 2013Q2 starts 1 Apr: public through 30 Mar
    W2 = dict(W)
    late = (W["dd"] > cut)
    W2["gap"] = W["gap"].copy()
    W2["gap"][late] = -W["gap"][late] * 50
    _, ch1 = C.select_quarterly(W2, st)
    assert ch1["2013Q2"] == ch0["2013Q2"]


def test_conformal_ignores_later_gaps():
    import v14_positions as VP
    p = _rows()
    W = C.wide(p, ["pred"])
    leg0, w0 = VP.legs_conformal(W, W["pred"], 0.8)
    W2 = dict(W)
    late = W["dd"] > pd.Timestamp("2013-03-30")
    W2["gap"] = W["gap"].copy()
    W2["gap"][late] = 1e4
    leg1, w1 = VP.legs_conformal(W2, W["pred"], 0.8)
    q2 = (W["dd"] >= pd.Timestamp("2013-04-01")) & (W["dd"] <= pd.Timestamp("2013-06-30"))
    np.testing.assert_array_equal(leg0[q2], leg1[q2])
    np.testing.assert_allclose(w0["2013Q2"], w1["2013Q2"])
    assert "2012Q1" not in w0                                   # no earlier quarter, no interval, no trade
    q1 = (W["dd"] < pd.Timestamp("2012-04-01"))
    assert not leg0[q1].any()
    # wider coverage skips more
    leg5, _ = VP.legs_conformal(W, W["pred"], 0.5)
    assert np.abs(leg5).sum() >= np.abs(leg0).sum()


def test_quantile_sizing():
    import v14_positions as VP
    H = 4
    W = {"pc": np.full(H, 0.2)}
    med = np.zeros((H, C.NZ))
    med[:, 0] = [0.1, 5, 5, 50]                                 # pair (0, 1): spread = med_0 - med_1
    lo, hi = med - 1, med + 1
    hi2 = med + 20
    s1 = VP.legs_quantile(W, lo, med, hi, 1.0)[:, 0]
    s2 = VP.legs_quantile(W, lo, med, hi2, 1.0)[:, 0]
    assert s1[0] == 0 and (np.abs(s1) <= 1).all() and s1[3] == 1
    assert (np.abs(s2) <= np.abs(s1) + 1e-12).all() and abs(s2[1]) < abs(s1[1])


# ============================================================================ V14e / V14f / GBM windows
def _day_matrix(p, seed=0, leak=False):
    days = pd.DatetimeIndex(sorted(p["delivery_date"].unique()))
    g = np.random.default_rng(seed)
    keys = pd.DataFrame({"bid_date": days - pd.Timedelta(days=1), "delivery_date": days})
    f = p.groupby("delivery_date")["f1"].mean().reindex(days).to_numpy(np.float32)
    X = np.c_[f, g.normal(0, 1, (len(days), 6)).astype(np.float32)]
    cols = ["px__f", "px__a", "px__b", "lf__c", "lf__d", "wxr__e", "cal__x"]
    if leak:                                                    # positive control: the day's own mean gap
        X[:, 1] = p.groupby("delivery_date")["gap"].mean().reindex(days).to_numpy(np.float32)
    return keys, X, cols


def _noisy_pair(p, keys, X, t):
    pn = C.noise_after(p, C.label_cut(Q), t - pd.Timedelta(days=1), ["f1", "f2"])
    Xn = X.copy()
    late = (keys["bid_date"] > t - pd.Timedelta(days=1)).to_numpy()
    Xn[late] = np.random.default_rng(9).normal(0, 500, Xn[late].shape)
    return pn, Xn


def test_day_model_injected_lookahead():
    import v14_mlp as VM
    p = synth_panel()
    keys, X, cols = _day_matrix(p)
    t = C.test_day(Q)
    te = p[p["delivery_date"] == t]
    a = VM.DayModel(keys, X, cols, p, "cpu").fit_predict(Q, te, "mlp_wx", fixed_epochs=2)
    pn, Xn = _noisy_pair(p, keys, X, t)
    b = VM.DayModel(keys, Xn, cols, pn, "cpu").fit_predict(Q, pn[pn["delivery_date"] == t], "mlp_wx", fixed_epochs=2)
    C.check_same(a, b, "day model")
    assert a.shape == (len(te),) and np.isfinite(a).all()
    nowx = VM.DayModel(keys, X, cols, p, "cpu")
    nowx.fit_predict(Q, te, "mlp_nowx", fixed_epochs=1)
    assert nowx.last_info["n_inputs"] == len(cols) - 1


def test_day_model_positive_control():
    """A label-derived input under a harmless name changes the prediction when the label is perturbed: the
    harness detects a leak when there is one."""
    import v14_mlp as VM
    p = synth_panel()
    t = C.test_day(Q)
    keys, X, cols = _day_matrix(p, leak=True)
    te = p[p["delivery_date"] == t]
    a = VM.DayModel(keys, X, cols, p, "cpu").fit_predict(Q, te, "mlp_wx", fixed_epochs=2)
    X2 = X.copy()
    X2[keys["delivery_date"] == t, 1] += 400.0                   # the test day's own gap moved
    b = VM.DayModel(keys, X2, cols, p, "cpu").fit_predict(Q, te, "mlp_wx", fixed_epochs=2)
    with pytest.raises(C.LeakError):
        C.check_same(a, b, "positive control")


def test_global_model_injected_lookahead():
    import v14_mlp as VM
    p = synth_panel()
    t = C.test_day(Q)
    te = p[p["delivery_date"] == t]
    out = {}
    for noisy in (False, True):
        pp = C.noise_after(p, C.label_cut(Q), t - pd.Timedelta(days=1), ["f1", "f2"]) if noisy else p
        m = VM.GlobalModel(pp, ["f1", "f2"], "cpu")
        cfg = {**VM.CFG_F, "batch": 4096}
        out[noisy] = {v: m.fit_predict(Q, pp[pp["delivery_date"] == t], v, cfg=cfg, fixed_epochs=2)
                      for v in ("glob_mse", "glob_dfl")}
    for v in out[False]:
        C.check_same(out[False][v], out[True][v], f"global {v}")
        assert out[False][v].shape == (len(te),)
    assert np.corrcoef(out[False]["glob_mse"], te["gap"])[0, 1] > 0.3     # it learns the synthetic signal


def test_gbm_windows_injected_lookahead(monkeypatch):
    import v14_gbm as VG
    p = synth_panel()
    t = C.test_day(Q)
    win = (("b1", 1), ("bexp", None))
    monkeypatch.setattr(R, "DATA_START", FIRST)
    a = VG.make_fp(p, ["f1", "f2"], 1, 20, win)(None, p[p["delivery_date"] == t], Q)
    pn = C.noise_after(p, C.label_cut(Q), t - pd.Timedelta(days=1), ["f1", "f2"])
    b = VG.make_fp(pn, ["f1", "f2"], 1, 20, win)(None, pn[pn["delivery_date"] == t], Q)
    for c in ("b1", "bexp", "gbm_win"):
        C.check_same(a[c].to_numpy(), b[c].to_numpy(), f"gbm {c}")
    with pytest.raises(C.LeakError):
        VG.make_fp(p, ["f1", "gap"], 1, 20, win)


def test_recency_weights():
    d = pd.Series(pd.date_range("2014-01-01", "2016-01-01", freq="D"))
    w = C.recency_weights(d, "2016-01-01")
    assert abs(w.mean() - 1) < 1e-5 and w[-1] > w[0]
    assert abs(w[-1] / w[-366] - 2.0) < 0.01


# ============================================================================ GRU variants (v1 synthetic tables)
@pytest.fixture(scope="module")
def gru_world(tmp_path_factory):
    import deep_data as DD
    first, last = dt.date(2023, 1, 1), dt.date(2023, 3, 31)
    root = tmp_path_factory.mktemp("v14_syn")
    frames = DD.write_synthetic_parquet(root, first, last, n_points=8, seed=3)
    dd = DD.DeepData(last, root=root, first_day=first)
    pz = frames["prices_zone"]
    panel = pd.DataFrame({"zone": pz["zone"].astype(str), "delivery_hour": pz["delivery_hour"],
                          "gap": pz["rt_lbmp"] - pz["da_lbmp"]})
    panel = panel[panel["zone"].isin(C.ZONES) & panel["gap"].notna()].reset_index(drop=True)
    panel["delivery_date"] = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    return dd, panel


TINY = dict(hidden=16, K=2, max_epochs=2, patience=1, min_epochs=1, val_days=10, batch=16, min_point_days=5)


def test_weighted_and_quantile_gru(gru_world):
    import deep as D1
    import v14_gru as VG
    dd, panel = gru_world
    tr = panel[panel["delivery_date"] <= pd.Timestamp("2023-03-10")]
    te = panel[panel["delivery_date"] >= pd.Timestamp("2023-03-20")]
    cfg = D1.resolve_config(TINY)
    w = VG.WeightedGRU(cfg, seeds=(0,), data=dd, device="cpu")
    w.fit(tr)
    pw = w.predict(te)
    assert np.isfinite(pw).all() and len(pw) == len(te)
    qm = VG.QuantileGRU(cfg, seeds=(0, 1), data=dd, device="cpu")
    qm.fit(tr)
    pq = qm.predict(te)
    assert list(pq.columns) == ["gq10", "gq25", "gq50", "gq75", "gq90"]
    assert (np.diff(pq.to_numpy(), axis=1) >= 0).all()
    # labels after the cut do not reach the fit
    tr2 = tr.copy()
    late = tr2["delivery_date"] > pd.Timestamp("2023-03-01")
    tr_cut = tr2[~late]
    a = VG.WeightedGRU(cfg, seeds=(0,), data=dd, device="cpu")
    a.fit(tr_cut)
    tr3 = pd.concat([tr_cut, panel[panel["delivery_date"] > pd.Timestamp("2023-03-25")].assign(gap=999.0)])
    b = VG.WeightedGRU(cfg, seeds=(0,), data=dd, device="cpu")
    b.fit(tr3[tr3["delivery_date"] <= pd.Timestamp("2023-03-01")])
    C.check_same(a.predict(te).to_numpy(), b.predict(te).to_numpy(), "weighted gru labels after cut")


def test_gru_guard_raises_on_tampered_cell(gru_world):
    import torch
    import deep_data as DD
    from timing import LookaheadError
    dd, _ = gru_world
    k = 40
    end = int(dd.day_start[k])
    old = dd.Zp[end - 3, 0, 1]
    dd.Zp[end - 3, 0, 1] = int(dd.dec_min[k]) - 1
    dd._dev = {}
    try:
        with pytest.raises(LookaheadError):
            DD.batch(dd, torch.as_tensor([k]), torch.arange(2), "cpu", check=True)
    finally:
        dd.Zp[end - 3, 0, 1] = old
        dd._dev = {}
