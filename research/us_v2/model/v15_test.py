"""Injected-lookahead tests for every new V15 feature path, on synthetic data (no real data, no holdout read).

  V15a  conditional model: the test quarter's realised day-ahead and gap overwritten, predictions identical;
        a learner that reads the realised day-ahead at bid time is caught; label columns refused as features.
  V15a/b limit bids: side and limit from the grid unchanged when realised day-ahead and gap are overwritten.
  V15b  composed grid: the deep level sits at today's day-ahead; built only from bid-time arrays.
  V15c  weather: day-matrix rows of other delivery dates overwritten (except D-7, the week-on-week lag), features
        identical; positive control moves; a join shifted one day later (tomorrow's forecast) is caught; a
        misaligned day matrix (delivery != bid + 1) is refused; reforecast first, GFS second, NaN in the gap.
  V15d  pair choice for a quarter unchanged when that quarter's and later gaps are overwritten.

    python -m pytest -q v15_test.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import v15_common as V
from v15_common import R, SV

BIG = 1e6


# ============================================================================ V15a conditional model
def frames(seed=0):
    rng = np.random.default_rng(seed)

    def mk(n, start):
        d = pd.date_range(start, periods=n, freq="h", tz=R.TZ)
        da0 = 40 + rng.normal(0, 10, n)
        da = da0 * rng.uniform(0.6, 1.6, n)
        x1 = rng.normal(0, 1, n)
        return pd.DataFrame({"delivery_hour": d, "delivery_date": d.tz_localize(None).normalize(), "x1": x1,
                             "x2": rng.normal(0, 1, n), "da_d0_h": da0, "y_da_lbmp": da,
                             "gap": 0.5 * (da0 - da) + 3 * x1 + rng.normal(0, 3, n)})
    return mk(6000, "2015-01-01"), mk(800, "2015-10-01")


def test_cond_model_ignores_test_labels():
    tr, te = frames()
    fp = V.make_cond_fp(["x1", "x2"], threads=1, n_rounds=40)
    a = fp(tr, te)
    te2 = te.copy()
    te2[["y_da_lbmp", "gap"]] = BIG
    pd.testing.assert_frame_equal(a, fp(tr, te2))
    # positive control: the bid-time inputs do move it
    te3 = te.copy()
    te3["da_d0_h"] = te3["da_d0_h"] * 2
    assert not a.equals(fp(tr, te3))


def test_cond_model_catches_injected_leak(monkeypatch):
    tr, te = frames()
    orig = V.cond_X

    def leaky(df, cols, da):
        Z = orig(df, cols, da)
        if "y_da_lbmp" in df:
            Z["cond_da"] = df["y_da_lbmp"].to_numpy(float)       # realised day-ahead at bid time
        return Z
    monkeypatch.setattr(V, "cond_X", leaky)
    fp = V.make_cond_fp(["x1", "x2"], threads=1, n_rounds=40)
    te2 = te.copy()
    te2["y_da_lbmp"] = BIG

    # the guarded fp hands only bid-time columns to cond_X, so even the leaky cond_X cannot see y_da_lbmp
    pd.testing.assert_frame_equal(fp(tr, te), fp(tr, te2))

    def unguarded(tr, te):                                   # the same model without the column guard
        import lightgbm as lgb
        m = lgb.train(dict(V.L.PARAMS, num_threads=1), lgb.Dataset(leaky(tr, ["x1", "x2"], tr["y_da_lbmp"]),
                                                                   tr["gap"]), 40)
        return m.predict(leaky(te, ["x1", "x2"], te["da_d0_h"]))
    assert not np.allclose(unguarded(tr, te), unguarded(tr, te2)), "injected leak not detected"


@pytest.mark.parametrize("bad", ["gap", "y_da_lbmp", "y_rt_lbmp", "cond_da"])
def test_labels_refused(bad):
    with pytest.raises((ValueError, AssertionError)):
        V.make_cond_fp(["x1", bad])


# ============================================================================ limit bids (V15a, V15b)
def grid_rows(seed=1, n=500):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2016-03-01", periods=n, freq="h", tz=R.TZ)
    g = pd.DataFrame({"delivery_hour": d, "delivery_date": d.tz_localize(None).normalize(),
                      "y_da_lbmp": rng.normal(40, 10, n), "gap": rng.normal(0, 10, n)})
    base = rng.normal(40, 8, n)
    for k, m in enumerate(V.DA_MULT):
        g[f"dag_{k}"] = base * m
        g[f"cond_{k}"] = rng.normal(0, 4, n) - 0.3 * (base * m - 40)
    return g


def test_limit_bids_ignore_realised():
    g = grid_rows()
    s, l = V.limit_bids(g)
    g2 = g.copy()
    g2[["y_da_lbmp", "gap"]] = BIG
    s2, l2 = V.limit_bids(g2)
    assert np.array_equal(s, s2) and np.array_equal(l, l2, equal_nan=True)
    assert (s != 0).any()
    g3 = g.copy()
    g3["cond_0"] = g3["cond_0"] + 50                     # positive control: the grid moves the bids
    s3, l3 = V.limit_bids(g3)
    assert not (np.array_equal(s, s3) and np.array_equal(l, l3, equal_nan=True))


def test_v15b_composed_grid():
    from v15_v13 import compose_limit_grid
    rng = np.random.default_rng(2)
    C = rng.normal(0, 5, (100, len(V.DA_MULT)))
    p = rng.normal(0, 5, 100)
    G = compose_limit_grid(p, C)
    assert np.allclose(G[:, V.I_D0], p)
    assert np.allclose(G - G[:, [0]], C - C[:, [0]])


# ============================================================================ V15c weather path
ZP = {"WEST": "buffalo", "N.Y.C.": "nyc"}


def day_matrix(n_days=40, start="2019-12-01", gfs_from=None, wxr_to=None, seed=3):
    rng = np.random.default_rng(seed)
    dd = pd.date_range(start, periods=n_days, freq="D")
    df = pd.DataFrame({"bid_date": dd - pd.Timedelta(days=1), "delivery_date": dd})
    for pre in ("wxr", "wx"):
        for pt in ZP.values():
            for h in range(24):
                v = rng.normal(5, 5, n_days)
                if pre == "wxr" and wxr_to is not None:
                    v = np.where(dd < pd.Timestamp(wxr_to), v, np.nan)
                if pre == "wx":
                    v = np.where(dd >= pd.Timestamp(gfs_from), v, np.nan) if gfs_from else np.full(n_days, np.nan)
                df[f"{pre}__{pt}__h{h:02d}"] = v
    return df


def keys_for(days):
    rows = [(pd.Timestamp(d), z, h) for d in days for z in ZP for h in range(24)]
    return pd.DataFrame(rows, columns=["delivery_date", "zone", "whour"])


def wcols(df):
    return [c for c in df.columns if c.startswith(("wx__", "wxr__"))]


def test_weather_ignores_other_days():
    day = day_matrix()
    D1 = pd.Timestamp("2019-12-20")
    k = keys_for([D1])
    a = V.weather_from_day(day, k, ZP)
    assert a["w15_temp_f"].notna().all() and a["w15_temp_f_wow"].notna().all()
    bad = day.copy()
    keep = day["delivery_date"].isin([D1, D1 - pd.Timedelta(days=7)])
    bad.loc[~keep, wcols(day)] = BIG
    pd.testing.assert_frame_equal(a, V.weather_from_day(bad, k, ZP))
    ctrl = day.copy()
    ctrl.loc[day["delivery_date"] == D1, wcols(day)] = BIG
    assert not a.equals(V.weather_from_day(ctrl, k, ZP))


def test_weather_catches_shifted_join():
    """A join that reads the next delivery day's forecast (one day later than public at 05:00) must be caught."""
    day = day_matrix()
    D1 = pd.Timestamp("2019-12-20")
    k = keys_for([D1])
    leak = day.copy()
    leak["delivery_date"] = leak["delivery_date"] - pd.Timedelta(days=1)      # row of D1+1 relabelled D1
    leak["bid_date"] = leak["bid_date"] - pd.Timedelta(days=1)
    bad = day.copy()
    bad.loc[day["delivery_date"] == D1 + pd.Timedelta(days=1), wcols(day)] = BIG
    bad_leak = bad.copy()
    bad_leak["delivery_date"] = leak["delivery_date"]
    bad_leak["bid_date"] = leak["bid_date"]
    # honest join: unchanged by the poisoned future row; shifted join: moved
    pd.testing.assert_frame_equal(V.weather_from_day(day, k, ZP), V.weather_from_day(bad, k, ZP))
    assert not V.weather_from_day(leak, k, ZP).equals(V.weather_from_day(bad_leak, k, ZP))


def test_weather_refuses_misaligned_matrix():
    day = day_matrix()
    day.loc[3, "bid_date"] = day.loc[3, "delivery_date"]
    with pytest.raises(V.LookaheadError):
        V.weather_from_day(day, keys_for([pd.Timestamp("2019-12-20")]), ZP)


def test_weather_source_order():
    day = day_matrix(n_days=40, start="2019-12-01", wxr_to="2019-12-15", gfs_from="2019-12-25")
    k = keys_for([pd.Timestamp("2019-12-10"), pd.Timestamp("2019-12-20"), pd.Timestamp("2019-12-30")])
    w = V.weather_from_day(day, k, ZP)
    src = w.groupby(k["delivery_date"])["w15_src"].first().to_list()
    assert src == [1.0, 0.0, 2.0]
    assert w.loc[k["delivery_date"] == pd.Timestamp("2019-12-20"), "w15_temp_f"].isna().all()
    r = k.index[(k["delivery_date"] == pd.Timestamp("2019-12-10")) & (k["zone"] == "WEST") & (k["whour"] == 7)][0]
    raw = day.loc[day["delivery_date"] == pd.Timestamp("2019-12-10"), "wxr__buffalo__h07"].iloc[0]
    assert np.isclose(w.loc[r, "w15_temp_f"], raw * 9 / 5 + 32)


def test_zone_points_cover_zones():
    zp = V.zone_points()
    assert len(zp) == 11 and zp["N.Y.C."] == "nyc" and zp["DUNWOD"] == "whiteplains"


# ============================================================================ V15d pair choice
def test_pair_choice_ignores_current_and_later_gaps():
    rng = np.random.default_rng(4)
    hours = pd.date_range("2012-01-01", "2013-12-31 23:00", freq="h", tz=R.TZ)
    rows = []
    for z in R.ZONES:
        n = len(hours)
        gap = rng.normal(0, 10, n) + (5 if z in ("N.Y.C.", "LONGIL") else 0)
        rows.append(pd.DataFrame({"delivery_hour": hours, "zone": z, "gap": gap,
                                  "pred": gap * 0.3 + rng.normal(0, 5, n)}))
    g = pd.concat(rows, ignore_index=True)
    g["delivery_date"] = g["delivery_hour"].dt.tz_localize(None).dt.normalize()
    q0 = dt.date(2013, 7, 1)
    _, ch = SV.Pairs(g, "pred").run(g)
    g2 = g.copy()
    late = g2["delivery_date"] >= pd.Timestamp(q0) - pd.Timedelta(days=1)
    g2.loc[late, "gap"] = rng.normal(0, 1000, int(late.sum()))
    _, ch2 = SV.Pairs(g2, "pred").run(g2)
    assert ch["2013Q3"]["pairs"] == ch2["2013Q3"]["pairs"]
    assert ch["2013Q4"]["pairs"] != ch2["2013Q4"]["pairs"], "positive control: trailing gaps must move the choice"


def test_combine_weather_prefers_reforecast():
    idx = pd.RangeIndex(3)
    refc = pd.DataFrame({c: [10.0, np.nan, np.nan] for c in V.V4_MAP}, index=idx)
    gfs = pd.DataFrame({c: [20.0, 20.0, np.nan] for c in V.W_COLS}, index=idx)
    gfs["w15_src"] = [2.0, 2.0, 0.0]
    w = V.combine_weather(refc, gfs)
    assert w["w15_temp_f"].tolist()[:2] == [10.0, 20.0] and np.isnan(w["w15_temp_f"].iloc[2])
    assert w["w15_src"].tolist() == [1.0, 2.0, 0.0]
