"""Injected-lookahead tests for V4 to V9, on synthetic data (no real data, no holdout read).

Method, per idea: compute the features of one bid day D, then overwrite every source value that was
NOT public at 05:00 on D (published_at later) with a huge number, recompute, and require identical
features. A positive control overwrites values that WERE public and requires a change, so each test
can fail. For the two-stage ideas (V8, V9) the test quarter's labels are overwritten and the
predictions must not move; a deliberately leaky learner must be caught. Run: pytest -q test_lookahead.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ideas_common as C  # noqa: E402
import v4_reforecast as V4  # noqa: E402
import v5_border as V5  # noqa: E402
import v6_specialists as V6  # noqa: E402
import v7_regimes as V7  # noqa: E402
import v8_error_mining as V8  # noqa: E402
import v9_decompose as V9  # noqa: E402
import v16_gefs_joined as V16  # noqa: E402

BIG = 1e6
START, DAYS = pd.Timestamp("2015-01-01"), 70
D = pd.Timestamp("2015-02-20")          # the bid day under test


def prices() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    hours = pd.date_range(START, periods=DAYS * 24, freq="h", tz=C.TZ)
    rows = []
    for z in C.ZONES + C.BORDERS:
        n = len(hours)
        da_e, rt_e = 30 + rng.normal(0, 5, n), 30 + rng.normal(0, 8, n)
        da_c, rt_c = rng.normal(0, 3, n), rng.normal(0, 5, n)
        da_l, rt_l = rng.normal(0, 1, n), rng.normal(0, 1, n)
        day = hours.tz_localize(None).normalize()
        rows.append(pd.DataFrame({
            "delivery_hour": hours, "zone": z,
            "da_energy": da_e, "rt_energy": rt_e, "da_congestion": da_c, "rt_congestion": rt_c,
            "da_loss": da_l, "rt_loss": rt_l,
            "da_lbmp": da_e + da_c + da_l, "rt_lbmp": rt_e + rt_c + rt_l,
            # day-ahead posted 11:00 the day before; real-time 30 minutes after the hour ends, except
            # odd days, whose file is rewritten at 06:00 the next day (after the 05:00 deadline), so
            # D-1 = 19 Feb is NOT public at 05:00 on D and a window that reads it is caught
            "da_published_at": (day - pd.Timedelta(hours=13)).tz_localize(C.TZ),
            "rt_published_at": np.where(day.day % 2 == 1,
                                        (day + pd.Timedelta(hours=30)).tz_localize(C.TZ),
                                        hours + pd.Timedelta(minutes=90))}))
    return pd.concat(rows, ignore_index=True)


def reforecast() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for d in pd.date_range(START, periods=DAYS, freq="D"):
        init = pd.Timestamp(d, tz="UTC")
        for z in C.ZONES:
            valid = [init + pd.Timedelta(hours=h) for h in range(27, 55, 3)]
            rows.append(pd.DataFrame({"init_utc": init, "target_hour": pd.DatetimeIndex(valid).tz_convert(C.TZ),
                                      "zone": z, "primary": True, "point": z,
                                      "temperature_2m_c": rng.normal(5, 5, len(valid)),
                                      "lead_hours": list(range(27, 55, 3)),
                                      "source": "noaa-gefs-retrospective GEFSv12 reforecast c00",
                                      "published_at": init + pd.Timedelta(hours=8)}))
    return pd.concat(rows, ignore_index=True)


def panel_index(bid_days) -> pd.MultiIndex:
    rows = []
    for b in bid_days:
        d1 = (b + pd.Timedelta(days=1)).tz_localize(C.TZ)
        hrs = pd.date_range(d1, (b + pd.Timedelta(days=2)).tz_localize(C.TZ), freq="h", inclusive="left")
        for z in C.ZONES:
            rows += [(b, z, h) for h in hrs]
    return pd.MultiIndex.from_tuples(rows, names=["bid_date", "zone", "delivery_hour"])


def poison(df: pd.DataFrame, pub_cols: dict, cutoff: pd.Timestamp, before: bool) -> pd.DataFrame:
    """Overwrite values whose publication is after (before=False) or at/before (True) the cutoff."""
    df = df.copy()
    for pub, vals in pub_cols.items():
        m = (df[pub] <= cutoff) if before else (df[pub] > cutoff)
        df.loc[m, vals] = BIG
    return df


def run(mod, store):
    for cache in (V5._GRID,):
        cache.clear()
    return mod.features(panel_index([D]), store=store)


PRICE_PUB = {"da_published_at": ["da_lbmp", "da_congestion", "da_energy", "da_loss"],
             "rt_published_at": ["rt_lbmp", "rt_congestion", "rt_energy", "rt_loss"]}


@pytest.mark.parametrize("mod", [V5, V6])
def test_price_features_ignore_unpublished(mod):
    cut = C.decision_time(D)
    pz = prices()
    base = run(mod, {"prices_zone": pz})
    assert base.notna().any().any()
    after = run(mod, {"prices_zone": poison(pz, PRICE_PUB, cut, before=False)})
    pd.testing.assert_frame_equal(base, after)
    # positive control: public values do move the features
    ctrl = run(mod, {"prices_zone": poison(pz, PRICE_PUB, cut, before=True)})
    assert not base.fillna(0).equals(ctrl.fillna(0))


def test_price_features_catch_injected_leak(monkeypatch):
    """A grid that treats every value as public at once must fail the same comparison."""
    cut = C.decision_time(D)
    pz = prices()
    orig = C.PriceGrid.__init__

    def leaky(self, *a, **k):
        orig(self, *a, **k)
        for s in self.usable:
            self.usable[s][:] = np.datetime64("1900-01-01", "ns")
    monkeypatch.setattr(C.PriceGrid, "__init__", leaky)
    base = run(V5, {"prices_zone": pz})
    after = run(V5, {"prices_zone": poison(pz, PRICE_PUB, cut, before=False)})
    assert not base.equals(after), "the injected leak was not detected"


def test_v4_ignores_unpublished_runs():
    cut = C.decision_time(D)
    wx = reforecast()
    base = run(V4, {"weather_reforecast": wx})
    assert base["wx_temp_f"].notna().all()
    after = run(V4, {"weather_reforecast": poison(wx, {"published_at": ["temperature_2m_c"]}, cut, False)})
    pd.testing.assert_frame_equal(base, after)
    ctrl = run(V4, {"weather_reforecast": poison(wx, {"published_at": ["temperature_2m_c"]}, cut, True)})
    assert not base.equals(ctrl)
    # injected leak: a run published at 05:01 on D must not be used
    late = wx.copy()
    m = late["init_utc"] == pd.Timestamp(D, tz="UTC")
    late.loc[m, "published_at"] = C.decision_time(D) + pd.Timedelta(minutes=1)
    late.loc[m, "temperature_2m_c"] = BIG
    assert not (run(V4, {"weather_reforecast": late})["wx_temp_f"] >= BIG / 2).any()   # older run or NaN


def test_v7_flag_not_on_before_effective_day():
    r = V7.regimes()
    eff = r["effective_date"].iloc[0]
    f_before = V7.features(panel_index([eff - pd.Timedelta(days=1)]))
    f_on = V7.features(panel_index([eff]))
    col = f"rg_{r['id'].iloc[0]}"
    assert (f_before[col] == 0).all() and (f_on[col] == 1).all()
    w = V7.sample_weight(pd.DataFrame({"bid_date": [D - pd.Timedelta(days=365), D]}), D, 365)
    assert np.allclose(w, [0.5, 1.0])
    with pytest.raises(ValueError):
        V7.sample_weight(pd.DataFrame({"bid_date": [D + pd.Timedelta(days=1)]}), D, 365)


def _frames():
    rng = np.random.default_rng(3)
    n_tr, n_te = 3 * 365 * 24, 24 * 30
    def mk(n, start):
        d = pd.date_range(start, periods=n, freq="h")
        return pd.DataFrame({"delivery_date": d.normalize(), "zone": rng.choice(C.ZONES, n),
                             "zone_code": rng.integers(0, 11, n), "hour": d.hour, "month": d.month,
                             "lf_h_rel": rng.normal(1, 0.1, n), "x": rng.normal(0, 1, n),
                             "gap": rng.normal(0, 10, n), "y_energy": rng.normal(0, 5, n),
                             "y_cong": rng.normal(0, 5, n), "y_loss": rng.normal(0, 1, n)})
    return mk(n_tr, "2012-01-01"), mk(n_te, "2015-01-05")


def lin_fp(train, test, target, cols, sample_weight=None):
    from learner import check_features
    check_features(cols)
    X = np.c_[np.ones(len(train)), train[cols].to_numpy(float)]
    b = np.linalg.lstsq(X, train[target].to_numpy(float), rcond=None)[0]
    return np.c_[np.ones(len(test)), test[cols].to_numpy(float)] @ b


def leaky_fp(train, test, target, cols, sample_weight=None):
    return lin_fp(train, test, target, cols) + 0.01 * test["gap"].to_numpy(float)


@pytest.mark.parametrize("fp_mod", [V8, V9])
def test_two_stage_ignores_test_labels(fp_mod):
    tr, te = _frames()
    base = fp_mod.fit_predict(tr, te, lin_fp, ["x"])
    te2 = te.copy()
    te2[["gap", "y_energy", "y_cong", "y_loss"]] = BIG
    pd.testing.assert_frame_equal(base, fp_mod.fit_predict(tr, te2, lin_fp, ["x"]))
    # injected leak is caught by the same comparison
    a, b = fp_mod.fit_predict(tr, te, leaky_fp, ["x"]), fp_mod.fit_predict(tr, te2, leaky_fp, ["x"])
    assert not a.equals(b)


def test_label_columns_refused_as_features():
    tr, te = _frames()
    with pytest.raises(ValueError):
        V9.fit_predict(tr, te, lin_fp, ["x", "y_cong"])


def test_v9_targets_add_up():
    pz = prices()
    idx = panel_index([D])
    p = C.panel_frame(idx)
    panel = pd.DataFrame({"zone": p["zone"].to_numpy(), "delivery_hour": p["delivery_hour"].to_numpy()})
    key = pz.set_index(["delivery_hour", "zone"])
    panel["gap"] = (key["rt_lbmp"] - key["da_lbmp"]).reindex(
        pd.MultiIndex.from_frame(panel[["delivery_hour", "zone"]])).to_numpy()
    t = V9.targets(panel, store={"prices_zone": pz})
    assert np.allclose(t.sum(axis=1), panel["gap"])


def test_holdout_refused():
    with pytest.raises(C.HoldoutError):
        C.panel_frame(panel_index([pd.Timestamp("2023-12-31")]))


@pytest.mark.parametrize("change", [dict(source="ERA5 reanalysis"), dict(source="ISD observed"), "drop_source",
                                    "drop_lead"])
def test_v4_refuses_non_forecast_weather(change):
    wx = reforecast()
    if change == "drop_source":
        wx = wx.drop(columns="source")
    elif change == "drop_lead":
        wx = wx.drop(columns="lead_hours")
    else:
        wx = wx.assign(**change)
    with pytest.raises(ValueError):
        run(V4, {"weather_reforecast": wx})


# ------------------------------------------------------------------------------ V16 joined GEFS v12
def joined(start, days, source) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for d in pd.date_range(start, periods=days, freq="D"):
        init = pd.Timestamp(d, tz="UTC")
        for z in C.ZONES:
            lead = list(range(27, 55, 3))
            rows.append(pd.DataFrame({"init_utc": init.tz_convert(C.TZ),
                                      "target_hour": (init + pd.to_timedelta(lead, unit="h")).tz_convert(C.TZ),
                                      "zone": z, "primary": True, "point": z, "lead_hours": lead,
                                      "temperature_2m_c": rng.normal(5, 5, len(lead)), "source": source,
                                      "published_at": (init + pd.Timedelta(hours=8)).tz_convert(C.TZ)}))
    return pd.concat(rows, ignore_index=True)


def joined_store():
    """Reforecast runs to 2019-12-31, nothing until 2020-09-22, live GEFS v12 from 2020-09-23."""
    a = joined(pd.Timestamp("2019-11-01"), 61, "GEFS reforecast v12 (AWS noaa-gefs-retrospective, c00)")
    b = joined(pd.Timestamp("2020-09-23"), 40, "GEFS v12 live forecast archive (AWS noaa-gefs-pds, c00)")
    return pd.concat([a, b], ignore_index=True)


D16 = [pd.Timestamp("2019-12-20"), pd.Timestamp("2020-10-15")]     # one bid day per source


def run16(store, days=D16):
    return V16.features(panel_index(days), store={V16.TABLE: store})


def test_v16_reads_the_joined_table_both_sources():
    wx = joined_store()
    f = run16(wx)
    assert f["wx_temp_f"].notna().all()
    assert not run16(wx.assign(temperature_2m_c=wx["temperature_2m_c"] + 1)).equals(f)


def test_v16_ignores_unpublished_runs():
    wx = joined_store()
    for D in D16:
        cut = C.decision_time(D)
        base = run16(wx, [D])
        after = run16(poison(wx, {"published_at": ["temperature_2m_c"]}, cut, False), [D])
        pd.testing.assert_frame_equal(base, after)
        ctrl = run16(poison(wx, {"published_at": ["temperature_2m_c"]}, cut, True), [D])
        assert not base.equals(ctrl)                       # positive control
        late = wx.copy()                                   # injected leak: run of D published 05:01 on D
        m = late["init_utc"] == pd.Timestamp(D, tz="UTC")
        assert m.any()
        late.loc[m, "published_at"] = C.decision_time(D) + pd.Timedelta(minutes=1)
        late.loc[m, "temperature_2m_c"] = BIG
        assert not (run16(late, [D])["wx_temp_f"] >= BIG / 2).any()


def test_v16_injected_lookahead_is_caught(monkeypatch):
    """A V4 that ignores publication times (every run public at once) uses the run of D+1 for bid day D;
    the poisoned comparison must then differ, so the test above can fail."""
    wx = joined_store()
    D = D16[1]
    cut = C.decision_time(D)
    orig = V4.per_bid_day

    def leaky(w, days):
        w = w.copy()
        w["avail"] = pd.to_datetime(w["avail"]) - pd.Timedelta(days=1)    # run of D+1 treated as public on D
        return orig(w, days)
    monkeypatch.setattr(V4, "per_bid_day", leaky)
    with pytest.raises(AssertionError):            # V4's own audit refuses, or the comparison differs
        base = run16(wx, [D])
        after = run16(poison(wx, {"published_at": ["temperature_2m_c"]}, cut, False), [D])
        pd.testing.assert_frame_equal(base, after)


def test_v16_gap_has_no_weather_and_is_not_traded():
    wx = joined_store()
    gap_day = pd.Timestamp("2020-05-10")
    f = run16(wx, [gap_day])
    assert f["wx_temp_f"].isna().all()                 # no older run carried into the gap
    dd = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-01", "2020-05-11", "2020-09-23", "2020-09-24"]))
    assert V16.traded(dd).tolist() == [True, False, False, False, True]


@pytest.mark.parametrize("change", [dict(source="ERA5 reanalysis"), "drop_source"])
def test_v16_refuses_non_forecast_weather(change):
    wx = joined_store()
    wx = wx.drop(columns="source") if change == "drop_source" else wx.assign(**change)
    with pytest.raises(ValueError):
        run16(wx)
