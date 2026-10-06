"""Lookahead test on the REAL tables (gene only; skipped where the parquet is absent).

For a handful of bid days in 2020-2023, every value published after 05:00 on D is replaced by noise,
in memory, in all sources (zone and generator prices, load forecasts, GFS weather, outage snapshots).
The base panel features, the generator summary, and the idea B and D features must not move.
Build years only: the LockedStore never holds a row for 2024 or later."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import panel as P
import strategies as S
from common import PARQUET, decision_time

pytestmark = pytest.mark.skipif(not (PARQUET / "prices_zone.parquet").exists(), reason="real tables absent")

FIXED_DAYS = [dt.date(2020, 3, 9), dt.date(2021, 11, 7), dt.date(2022, 7, 15), dt.date(2023, 3, 12),
              dt.date(2023, 8, 21)]


@pytest.fixture(scope="module")
def real():
    s = P.LockedStore()
    for n in ("prices_zone", "load_forecast", "outages", "weather_gfs"):
        s.table(n)
    return s


def _revised_day(real) -> dt.date:
    """A delivery day D+1 whose previous day's real-time file was rewritten after 05:00 on D."""
    pz = real.table("prices_zone")
    r = pz[pz["rt_revised"].astype(bool) & (pz["delivery_hour"].dt.year >= 2021)]
    for h, pub in zip(r["delivery_hour"], r["rt_published_at"]):
        d = h.tz_convert(P.TZ).date()
        if pub > decision_time(d + dt.timedelta(days=1)):
            return d + dt.timedelta(days=2)
    pytest.skip("no late-revised day found")


def _perturb_prices(df, t, rng):
    df = df.copy()
    fda = (df["da_published_at"] > t).to_numpy()
    frt = (df["rt_published_at"] > t).to_numpy()
    for c in ("da_lbmp", "da_loss", "da_congestion", "da_congestion_raw", "da_energy"):
        df.loc[fda, c] = rng.normal(0, 500, fda.sum())
    for c in ("rt_lbmp", "rt_loss", "rt_congestion", "rt_congestion_raw", "rt_energy"):
        df.loc[frt, c] = rng.normal(0, 500, frt.sum())
    return df


def _perturbed_store(real, t, d1):
    rng = np.random.default_rng(11)
    s = P.LockedStore(gen_cache_months=12)
    pz = _perturb_prices(real.table("prices_zone"), t, rng)
    lf = real.table("load_forecast").copy()
    lf["load_forecast_mw"] = lf["load_forecast_mw"].astype(float)          # stored as integers
    fut = (lf["published_at"] > t).to_numpy()
    lf.loc[fut, "load_forecast_mw"] = rng.normal(0, 1e4, fut.sum())
    wx = real.table("weather_gfs").copy()
    wx["temperature_2m_c"] = wx["temperature_2m_c"].astype(float)
    fut = (wx["published_at"] > t).to_numpy()
    wx.loc[fut, "temperature_2m_c"] = rng.normal(0, 30, fut.sum())
    og = real.table("outages").copy()
    fut = (og["published_at"] > t).to_numpy()
    shift = pd.to_timedelta(rng.integers(-5, 5, fut.sum()), unit="D")
    og.loc[fut, "sched_out"] = og.loc[fut, "sched_out"] + shift
    og.loc[fut, "sched_in"] = og.loc[fut, "sched_in"] - shift
    for name, df in (("prices_zone", pz), ("load_forecast", lf), ("weather_gfs", wx), ("outages", og)):
        s._t[name] = df
        s._key[name] = real._key[name]
    lo = pd.Timestamp(d1, tz=P.TZ) - pd.Timedelta(days=5)
    for p in pd.period_range(lo.date(), d1 + dt.timedelta(days=5), freq="M"):
        g = real._gen_month(p)
        if g is not None:
            s._gen[p] = _perturb_prices(g, t, rng)
    return s


def test_real_days_do_not_see_the_future(real):
    days = FIXED_DAYS + [_revised_day(real)]
    for d1 in days:
        t = decision_time(d1 - dt.timedelta(days=1))
        a = P.build_panel(d1, d1, store=real, include_gen=True)
        s = _perturbed_store(real, t, d1)
        b = P.build_panel(d1, d1, store=s, include_gen=True)
        feats = P.BASE_FEATURES + P.GEN_FEATURES
        pd.testing.assert_frame_equal(a[feats], b[feats], obj=f"features for {d1}")
        assert not np.allclose(a["gap"], b["gap"]), d1                       # the label did change
        wa, wb = S.weather_features(a, real), S.weather_features(b, s)
        pd.testing.assert_frame_equal(wa, wb, obj=f"weather features for {d1}", check_dtype=False)
        oa, ob = S.outage_features(a, real), S.outage_features(b, s)
        pd.testing.assert_frame_equal(oa, ob, obj=f"outage features for {d1}", check_dtype=False)


def test_real_store_holds_build_years_only(real):
    end = pd.Timestamp("2024-01-01", tz=P.TZ)
    assert real.table("prices_zone")["delivery_hour"].max() < end
    assert real.table("load_forecast")["target_hour"].max() < end
    assert real.table("weather_gfs")["target_hour"].max() < end
    assert pd.Timestamp(real.table("outages")["snapshot_date"].max()) <= pd.Timestamp("2023-12-30")
