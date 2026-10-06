"""Three-day weather rule on synthetic tables (runs anywhere, no parquet needed).

    python -m pytest -q -p no:cacheprovider test_gfs_rule.py

Day2 (run_lead_hours 48, published valid - 40 h) for hours up to 21:00 New York time, day3 (72,
published valid - 64 h) for 22:00 and 23:00, and day3 wherever day2 is missing or not yet public.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import timing as T
from common import TZ, decision_time, gfs_published_at
from points import POINTS


def empty_tables() -> dict:
    ts = pd.Series(pd.DatetimeIndex([], tz=TZ))
    pz = pd.DataFrame({c: pd.Series(dtype="float64") for c in T.DA_COLS + T.RT_COLS})
    for c in ("delivery_hour", "da_published_at", "rt_published_at"):
        pz[c] = ts
    lf = pd.DataFrame({"issue_date": pd.Series(dtype="datetime64[ns]"), "file_written_at": ts, "published_at": ts,
                       "target_hour": ts, "lead_days": pd.Series(dtype="int64"), "zone": pd.Series(dtype=str),
                       "load_forecast_mw": pd.Series(dtype="float64")})
    og = pd.DataFrame({"snapshot_date": pd.Series(dtype="datetime64[ns]"), "file_written_at": ts, "published_at": ts,
                       "ptid": pd.Series(dtype="Int64"), "equipment": pd.Series(dtype=str),
                       "sched_out": ts, "sched_in": ts})
    return {"prices_zone": pz, "load_forecast": lf, "outages": og}


def synthetic_weather(first: str, last: str, null_day2: list[pd.Timestamp] = ()) -> pd.DataFrame:
    """Every point, every hour in [first, last), both leads. Day2 temperature = 2.0, day3 = 3.0."""
    hours = pd.date_range(pd.Timestamp(first, tz=TZ), pd.Timestamp(last, tz=TZ), freq="h", inclusive="left")
    rows = []
    for name, (zone, lat, lon, primary) in POINTS.items():
        for lead, val in ((48, 2.0), (72, 3.0)):
            df = pd.DataFrame({"point": name, "zone": zone, "primary": primary, "lat_req": lat, "lon_req": lon,
                               "lat_grid": lat, "lon_grid": lon, "target_hour": hours,
                               "temperature_2m_c": val, "run_lead_hours": lead})
            if lead == 48 and len(null_day2):
                df.loc[df["target_hour"].isin(null_day2), "temperature_2m_c"] = np.nan
            rows.append(df)
    wx = pd.concat(rows, ignore_index=True)
    wx["published_at"] = gfs_published_at(wx["target_hour"], wx["run_lead_hours"])
    return wx


def store_with(wx: pd.DataFrame) -> T.Store:
    return T.Store(tables={**empty_tables(), "weather_gfs": wx})


def day3_hours(store: T.Store, d1: str, point: str = "albany") -> tuple[set, int]:
    D = (pd.Timestamp(d1) - pd.Timedelta(days=1)).date()
    x = T.bid_inputs(D, store)
    T.assert_no_lookahead(x, decision_time(D))
    w = x["weather_d1"]
    w = w[w["point"] == point]
    assert not w.duplicated("target_hour").any()
    h = w["target_hour"].dt.tz_convert(TZ).dt.hour
    return set(h[w["run_lead_hours"] == 72]), len(w)


@pytest.mark.parametrize("d1, expected, n_hours", [
    ("2023-07-11", {22, 23}, 24),            # ordinary summer day
    ("2023-01-17", {22, 23}, 24),            # ordinary winter day
    ("2023-03-12", {22, 23}, 23),            # spring forward: day2 for 22:00 EDT public at 05:00, rule says day3
    ("2023-11-05", {21, 22, 23}, 25),        # fall back: day2 for 21:00 EST published 06:00 EDT on D
])
def test_rule_by_hour(d1, expected, n_hours):
    s = store_with(synthetic_weather("2023-01-01", "2023-12-31"))
    got, n = day3_hours(s, d1)
    assert got == expected and n == n_hours


def test_published_offsets_are_40_and_64_hours():
    wx = synthetic_weather("2023-07-10", "2023-07-12")
    off = (wx["target_hour"] - wx["published_at"]).dt.total_seconds() / 3600
    assert set(off[wx["run_lead_hours"] == 48]) == {40.0} and set(off[wx["run_lead_hours"] == 72]) == {64.0}


def test_ordinary_day_boundary_is_exactly_the_deadline():
    """21:00 on D+1 minus 40 h is 05:00 on D: public at the deadline, not after it."""
    wx = synthetic_weather("2023-07-10", "2023-07-12")
    D = dt.date(2023, 7, 10)
    h21 = wx[(wx["run_lead_hours"] == 48) & (wx["target_hour"] == pd.Timestamp("2023-07-11 21:00", tz=TZ))]
    h22 = wx[(wx["run_lead_hours"] == 48) & (wx["target_hour"] == pd.Timestamp("2023-07-11 22:00", tz=TZ))]
    assert (h21["published_at"] == decision_time(D)).all() and (h22["published_at"] > decision_time(D)).all()


def test_null_day2_hours_fall_back_to_day3():
    gap = [pd.Timestamp("2023-07-11 08:00", tz=TZ), pd.Timestamp("2023-07-11 09:00", tz=TZ)]
    s = store_with(synthetic_weather("2023-07-01", "2023-07-20", null_day2=gap))
    got, n = day3_hours(s, "2023-07-11")
    assert got == {8, 9, 22, 23} and n == 24


def test_missing_day3_leaves_late_hours_empty():
    """With only day2 in the table (before the day3 fetch), hours 22 and 23 are absent, never day2."""
    wx = synthetic_weather("2023-07-01", "2023-07-20")
    s = store_with(wx[wx["run_lead_hours"] == 48])
    D = dt.date(2023, 7, 10)
    w = T.bid_inputs(D, s)["weather_d1"]
    assert w["target_hour"].dt.tz_convert(TZ).dt.hour.max() == 21


def test_injected_late_day2_is_caught():
    s = store_with(synthetic_weather("2023-07-01", "2023-07-20"))
    D = dt.date(2023, 7, 10)
    x = T.bid_inputs(D, s)
    wx = s.table("weather_gfs")
    late = wx[(wx["run_lead_hours"] == 48) & (wx["target_hour"] == pd.Timestamp("2023-07-11 23:00", tz=TZ))]
    bad = dict(x)
    bad["weather_d1"] = pd.concat([x["weather_d1"], late], ignore_index=True)
    with pytest.raises(T.LookaheadError):
        T.assert_no_lookahead(bad, decision_time(D))


def test_rule_violations_on_synthetic_table():
    wx = synthetic_weather("2023-07-10", "2023-07-12")
    assert T.rule_violations("weather_gfs", wx) == {}
    bad = wx.iloc[[3, 50]].copy()
    bad["published_at"] = bad["published_at"] - pd.Timedelta(hours=1)
    assert T.rule_violations("weather_gfs", bad)["published_not_target_minus_lead_plus_8h"] == 2
    odd = wx.iloc[[0]].copy()
    odd["run_lead_hours"] = 24
    assert T.rule_violations("weather_gfs", odd)["run_lead_not_48_or_72"] == 1
