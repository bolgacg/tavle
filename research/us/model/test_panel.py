"""Panel tests on synthetic tables (conftest.py): the lock holds, features match their definitions,
and no feature moves when every value published after 05:00 on D is replaced by noise."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import lock
import panel as P
import timing as T
from common import TZ, ZONES, decision_time

FEATS = P.BASE_FEATURES + P.GEN_FEATURES


def _one_day(root, d1: dt.date, include_gen=True):
    return P.build_panel(d1, d1, store=P.LockedStore(root), include_gen=include_gen)


def test_store_never_loads_holdout_rows(synth_root):
    s = P.LockedStore(synth_root)
    assert s.table("prices_zone")["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    assert s.table("load_forecast")["target_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    assert s.table("weather_gfs")["target_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    assert s.table("outages")["snapshot_date"].max() <= pd.Timestamp("2023-12-30")
    g = s.gen_window(pd.Timestamp("2023-12-30", tz=TZ), pd.Timestamp("2024-01-05", tz=TZ))
    assert len(g) and g["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    assert all(p.year < 2024 for p in s._gen)                         # no 2024 month file was opened
    # the raw file does hold 2024 rows: the filter, not the data, keeps them out
    raw = pd.read_parquet(synth_root / "prices_zone.parquet", columns=["delivery_hour"])
    assert raw["delivery_hour"].max() >= pd.Timestamp("2024-01-01", tz=TZ)


def test_loaders_refuse_holdout_requests(synth_root):
    with pytest.raises(lock.HoldoutLocked):
        P.LockedStore(synth_root, end=dt.date(2024, 2, 1), env={})
    with pytest.raises(lock.HoldoutLocked):
        P.build_panel(dt.date(2023, 12, 1), dt.date(2024, 1, 2), store=P.LockedStore(synth_root))
    with pytest.raises(lock.HoldoutLocked):
        P.bid_days(dt.date(2023, 12, 30), dt.date(2024, 1, 1))


def test_label_is_not_a_feature():
    assert not set(P.LABEL_COLS) & set(FEATS)
    assert not [c for c in FEATS if c.startswith("y_") or c == "gap"]
    assert len(set(FEATS)) == len(FEATS)


@pytest.mark.parametrize("d1", [dt.date(2023, 3, 11), dt.date(2023, 3, 12), dt.date(2023, 11, 5),
                                dt.date(2023, 6, 21), dt.date(2023, 12, 31)], ids=str)
def test_no_feature_moves_when_the_future_is_replaced(synth_root, tmp_path, d1):
    """Replace every value published after 05:00 on D (prices, load forecasts, generator prices) with
    noise; every feature for D must be identical. The label must change (the perturbation bites)."""
    t = decision_time(d1 - dt.timedelta(days=1))
    rng = np.random.default_rng(1)
    root = tmp_path / "perturbed"
    root.mkdir()
    for name in ("prices_zone", "load_forecast", "outages", "weather_gfs"):
        df = pd.read_parquet(synth_root / f"{name}.parquet")
        if name == "prices_zone":
            df = _perturb_prices(df, t, rng)
        elif name == "load_forecast":
            fut = df["published_at"] > t
            df.loc[fut, "load_forecast_mw"] = rng.normal(0, 1e4, fut.sum())
        df.to_parquet(root / f"{name}.parquet")
    (root / "prices_gen").mkdir()
    for f in (synth_root / "prices_gen").glob("*.parquet"):
        _perturb_prices(pd.read_parquet(f), t, rng).to_parquet(root / "prices_gen" / f.name)
    a = _one_day(synth_root, d1)
    b = _one_day(root, d1)
    pd.testing.assert_frame_equal(a[FEATS], b[FEATS])
    assert not np.allclose(a["gap"], b["gap"])


def _perturb_prices(df, t, rng):
    df = df.copy()
    fda = df["da_published_at"] > t
    frt = df["rt_published_at"] > t
    for c in ("da_lbmp", "da_loss", "da_congestion", "da_congestion_raw", "da_energy"):
        df.loc[fda, c] = rng.normal(0, 500, fda.sum())
    for c in ("rt_lbmp", "rt_loss", "rt_congestion", "rt_congestion_raw", "rt_energy"):
        df.loc[frt, c] = rng.normal(0, 500, frt.sum())
    return df


def _raw(root):
    pz = pd.read_parquet(root / "prices_zone.parquet")
    loc = pz["delivery_hour"].dt.tz_convert(TZ)
    pz["ldate"] = loc.dt.tz_localize(None).dt.normalize()
    pz["h"] = loc.dt.hour
    return pz


def test_features_match_definitions(synth_root):
    pz = _raw(synth_root)
    d1 = dt.date(2023, 8, 16)
    D = d1 - dt.timedelta(days=1)
    t = decision_time(D)
    p = _one_day(synth_root, d1, include_gen=False)
    assert len(p) == 24 * 11
    r = p[(p["zone"] == "WEST") & (p["hour"] == 14)].iloc[0]
    w = pz[pz["zone"] == "WEST"]
    at = lambda day, h: w[(w["ldate"] == pd.Timestamp(day)) & (w["h"] == h)].iloc[0]  # noqa: E731
    assert r["da_d0_h"] == pytest.approx(at(D, 14)["da_lbmp"])
    x = at(D - dt.timedelta(days=1), 14)
    assert r["gap_dm1_h"] == pytest.approx(x["rt_lbmp"] - x["da_lbmp"])
    win = w[(w["delivery_hour"] >= t - pd.Timedelta(days=365)) & (w["rt_published_at"] <= t) & (w["h"] == 14)]
    assert r["gap_365d_h"] == pytest.approx((win["rt_lbmp"] - win["da_lbmp"]).mean())
    assert r["n_365d_h"] == len(win)
    early = w[(w["ldate"] == pd.Timestamp(D)) & (w["h"] <= 3)]
    assert r["rt_d0_early"] == pytest.approx(early["rt_lbmp"].mean())
    assert r["hours_since_rt"] == pytest.approx(1.0)                   # last usable hour is 03:00-04:00
    assert r["gap"] == pytest.approx(at(d1, 14)["rt_lbmp"] - at(d1, 14)["da_lbmp"])
    assert r["n_rt_dm1"] == 24


def test_revised_rt_day_is_not_used_until_rewritten(synth_root):
    p = _one_day(synth_root, dt.date(2023, 3, 11), include_gen=False)   # D = 10 Mar, D-1 = 9 Mar revised
    assert p["gap_dm1_h"].isna().all() and (p["n_rt_dm1"] == 0).all()
    q = _one_day(synth_root, dt.date(2023, 3, 17), include_gen=False)   # 9 Mar rewritten 15 Mar 10:00
    pz = _raw(synth_root)
    t = decision_time(dt.date(2023, 3, 16))
    w = pz[(pz["zone"] == "NORTH") & (pz["h"] == 9) & (pz["ldate"] >= pd.Timestamp("2023-03-09"))
           & (pz["ldate"] <= pd.Timestamp("2023-03-15"))]
    assert (w["rt_published_at"] <= t).all()
    r = q[(q["zone"] == "NORTH") & (q["hour"] == 9)].iloc[0]
    assert r["gap_7d_h"] == pytest.approx((w["rt_lbmp"] - w["da_lbmp"]).mean())


def test_load_forecast_uses_the_newest_public_vintage(synth_root):
    s = P.LockedStore(synth_root)
    for D in (dt.date(2023, 5, 2), dt.date(2023, 6, 20), dt.date(2023, 11, 4)):
        t = decision_time(D)
        f = T.features_available_at(t, s, lookback_days=P.LOOKBACK_DAYS)
        mine = P.newest_vintage(f["load_forecast"], pd.Timestamp(D + dt.timedelta(days=1)))
        ref = T.bid_inputs(D, s, lookback_days=P.LOOKBACK_DAYS)["load_forecast_d1"]
        k = ["zone", "target_hour"]
        pd.testing.assert_frame_equal(mine.sort_values(k).reset_index(drop=True), ref.sort_values(k).reset_index(drop=True))
    late = _one_day(synth_root, dt.date(2023, 6, 21), include_gen=False)  # file named 20 Jun written 06:30
    assert (late["lf_lead"] == 2).all()
    normal = _one_day(synth_root, dt.date(2023, 6, 23), include_gen=False)
    assert (normal["lf_lead"] == 1).all() and normal["lf_h"].notna().all()


def test_dst_days(synth_root):
    assert len(_one_day(synth_root, dt.date(2023, 3, 12), include_gen=False)) == 23 * 11
    fall = _one_day(synth_root, dt.date(2023, 11, 5), include_gen=False)
    assert len(fall) == 25 * 11 and fall["lf_h"].notna().all()


def test_gen_features_present_and_parallel_build_matches(synth_root):
    p = P.build_panel(dt.date(2023, 9, 1), dt.date(2023, 9, 3), store=P.LockedStore(synth_root), include_gen=True)
    assert p[P.GEN_FEATURES].notna().mean().min() > 0.9
    a = P.build_panel(dt.date(2023, 9, 1), dt.date(2023, 9, 6), store=P.LockedStore(synth_root), workers=1)
    b = P.build_panel(dt.date(2023, 9, 1), dt.date(2023, 9, 6), store=P.LockedStore(synth_root), workers=2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["zone"]) == set(ZONES)
