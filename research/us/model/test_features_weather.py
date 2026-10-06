"""Tests for features_weather (idea B). Synthetic tests run anywhere; the last test needs the gene
parquet tables and is skipped without them. No price is read (holdout rule).

    cd research/us/model && python -m pytest -q -p no:cacheprovider test_features_weather.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_weather as W  # noqa: E402
import timing as T  # noqa: E402
from common import PARQUET, TZ, ZONES, decision_time, gfs_published_at  # noqa: E402
from points import POINTS  # noqa: E402

FIRST, LAST = "2023-06-20", "2023-11-20"


def day_base_c(ts: pd.Series) -> pd.Series:
    """A temperature that differs by day of year, so week-over-week changes are non-zero."""
    return 10 + (ts.dt.tz_convert(TZ).dt.dayofyear % 7).astype(float)


def synthetic_store(first=FIRST, last=LAST) -> T.Store:
    """Weather: day2 = base + hour/2 (C), day3 = day2 + 100 (a marker). Load forecast: one file per
    issue date I, written 07:05 on I-1, targets I..I+5, value 1000 + 100*lead_days + local hour."""
    hours = pd.date_range(pd.Timestamp(first, tz=TZ), pd.Timestamp(last, tz=TZ), freq="h", inclusive="left")
    th = pd.Series(hours)
    local_hour = th.dt.tz_convert(TZ).dt.hour.astype(float)
    rows = []
    for name, (zone, lat, lon, primary) in POINTS.items():
        for lead in (48, 72):
            temp = day_base_c(th) + local_hour / 2 + (100 if lead == 72 else 0)
            rows.append(pd.DataFrame({"point": name, "zone": zone, "primary": primary, "lat_req": lat,
                                      "lon_req": lon, "lat_grid": lat, "lon_grid": lon, "target_hour": th,
                                      "temperature_2m_c": temp, "run_lead_hours": lead}))
    wx = pd.concat(rows, ignore_index=True)
    wx["published_at"] = gfs_published_at(wx["target_hour"], wx["run_lead_hours"])

    lf_rows = []
    for issue in pd.date_range(first, last, freq="D")[:-6]:
        tgt = pd.Series(pd.date_range(pd.Timestamp(issue, tz=TZ), pd.Timestamp(issue + pd.Timedelta(days=6), tz=TZ),
                                      freq="h", inclusive="left"))
        lead = (tgt.dt.tz_localize(None).dt.normalize() - issue).dt.days
        written = pd.Timestamp(issue - pd.Timedelta(days=1) + pd.Timedelta(hours=7, minutes=5), tz=TZ)
        for z in ZONES + ["NYISO"]:
            lf_rows.append(pd.DataFrame({"issue_date": issue, "file_written_at": written, "published_at": written,
                                         "target_hour": tgt, "lead_days": lead, "zone": z,
                                         "load_forecast_mw": 1000 + 100 * lead + tgt.dt.tz_convert(TZ).dt.hour}))
    lf = pd.concat(lf_rows, ignore_index=True)
    import test_gfs_rule as G  # pipeline helper for the empty price and outage frames
    return T.Store(tables={**G.empty_tables(), "weather_gfs": wx, "load_forecast": lf})


def panel(bid_days, zones=ZONES) -> pd.MultiIndex:
    rows = []
    for D in bid_days:
        d1 = pd.Timestamp(D + dt.timedelta(days=1), tz=TZ)
        for h in pd.date_range(d1, pd.Timestamp(D + dt.timedelta(days=2), tz=TZ), freq="h", inclusive="left"):
            for z in zones:
                rows.append((pd.Timestamp(D), z, h))
    return pd.MultiIndex.from_tuples(rows, names=["bid_date", "zone", "delivery_hour"])


@pytest.fixture(scope="module")
def store():
    return synthetic_store()


@pytest.fixture(scope="module")
def july(store):
    idx = panel([dt.date(2023, 7, 10), dt.date(2023, 7, 11), dt.date(2023, 7, 17)])
    return idx, W.features(idx, store)


def at(idx, f, D, zone, hour):
    h = pd.Timestamp(D + dt.timedelta(days=1), tz=TZ) + pd.Timedelta(hours=hour)
    return f.loc[(pd.Timestamp(D), zone, h)]


def test_alignment_and_columns(july):
    idx, f = july
    assert f.index.equals(idx) and list(f.columns) == W.COLUMNS
    assert len(f) == 3 * 24 * 11


def test_three_day_rule_in_features(july):
    idx, f = july
    D = dt.date(2023, 7, 10)
    base = 10 + (pd.Timestamp("2023-07-11").dayofyear % 7)
    r21, r22 = at(idx, f, D, "N.Y.C.", 21), at(idx, f, D, "N.Y.C.", 22)
    assert r21["wx_run_lead_hours"] == 48 and r21["wx_temp_f"] == pytest.approx((base + 10.5) * 9 / 5 + 32)
    assert r22["wx_run_lead_hours"] == 72 and r22["wx_temp_f"] == pytest.approx((base + 11 + 100) * 9 / 5 + 32)


def test_operator_vintage_is_file_named_d(july):
    """File D (written 07:05 on D-1) gives D+1 at lead 1: 1100 + hour. File D+1 (written 07:05 on D,
    after the deadline) would give 1000 + hour and must not be used."""
    idx, f = july
    D = dt.date(2023, 7, 10)
    assert at(idx, f, D, "WEST", 15)["lf_mw"] == 1115
    assert at(idx, f, D, "WEST", 15)["lf_vintage_age_days"] == 0
    assert f["lf_d1_peak_mw"].eq(1123).all() and f["lf_d1_min_mw"].eq(1100).all()


def test_daily_weather_features(july):
    idx, f = july
    D = dt.date(2023, 7, 10)
    base = 10 + (pd.Timestamp("2023-07-11").dayofyear % 7)
    c = np.array([base + h / 2 + (100 if h >= 22 else 0) for h in range(24)])
    tf = c * 9 / 5 + 32
    r = at(idx, f, D, "CAPITL", 0)
    assert r["wx_d1_max_f"] == pytest.approx(tf.max()) and r["wx_d1_min_f"] == pytest.approx(tf.min())
    assert r["wx_d1_hdh65"] == pytest.approx(np.clip(65 - tf, 0, None).sum())
    assert r["wx_d1_cdh65"] == pytest.approx(np.clip(tf - 65, 0, None).sum())


def test_week_over_week(july):
    """D = 17 Jul against D-7 = 10 Jul: same rule, so the change is the day-base difference only."""
    idx, f = july
    d_now, d_wk = pd.Timestamp("2023-07-18"), pd.Timestamp("2023-07-11")
    diff_f = ((d_now.dayofyear % 7) - (d_wk.dayofyear % 7)) * 9 / 5
    r = at(idx, f, dt.date(2023, 7, 17), "LONGIL", 9)
    assert r["wx_temp_f_wow"] == pytest.approx(diff_f) and r["wx_d1_max_f_wow"] == pytest.approx(diff_f)
    assert r["lf_mw_wow"] == 0 and r["lf_d1_peak_wow_pct"] == 0


def test_dst_fall_back_day(store):
    """D+1 = 5 Nov 2023 has 25 hours; 21:00 EST uses day3 because its day2 value is published
    06:00 EDT on D, after the deadline."""
    D = dt.date(2023, 11, 4)
    idx = panel([D], zones=["CAPITL"])
    f = W.features(idx, store)
    assert len(f) == 25
    lead = f["wx_run_lead_hours"].to_numpy()
    hours = pd.Series(idx.get_level_values("delivery_hour")).dt.hour.to_numpy()
    assert set(hours[lead == 72]) == {21, 22, 23} and f["wx_temp_f"].notna().all()


def test_late_rows_in_store_are_not_used(store):
    """A day2 value for 22:00 and an operator file named D+1 both sit in the store; neither is used."""
    D = dt.date(2023, 7, 10)
    wx = store.table("weather_gfs")
    late = wx[(wx["run_lead_hours"] == 48) & (wx["target_hour"] == pd.Timestamp("2023-07-11 22:00", tz=TZ))]
    assert len(late) and (late["published_at"] > decision_time(D)).all()
    idx = panel([D], zones=["N.Y.C."])
    f = W.features(idx, store)
    assert f.iloc[22]["wx_run_lead_hours"] == 72 and f.iloc[15]["lf_mw"] == 1115


def test_injected_lookahead_raises(store, monkeypatch):
    """If bid_inputs ever handed over a row published after 05:00 on D, features() must refuse."""
    real = T.bid_inputs

    def leaky(bid_day, s=None, **kw):
        x = real(bid_day, s, **kw)
        lf = s.table("load_forecast")
        future = lf[lf["issue_date"] == pd.Timestamp(bid_day + dt.timedelta(days=1))].head(3)
        x["load_forecast_d1"] = pd.concat([x["load_forecast_d1"], future], ignore_index=True)
        return x

    monkeypatch.setattr(T, "bid_inputs", leaky)
    with pytest.raises(T.LookaheadError):
        W.features(panel([dt.date(2023, 7, 10)], zones=["WEST"]), store)


def test_injected_late_weather_raises(store, monkeypatch):
    real = T.bid_inputs

    def leaky(bid_day, s=None, **kw):
        x = real(bid_day, s, **kw)
        wx = s.table("weather_gfs")
        d1_23 = pd.Timestamp(bid_day + dt.timedelta(days=1), tz=TZ) + pd.Timedelta(hours=23)
        late = wx[(wx["run_lead_hours"] == 48) & (wx["target_hour"] == d1_23)]
        x["weather_d1"] = pd.concat([x["weather_d1"], late], ignore_index=True)
        return x

    monkeypatch.setattr(T, "bid_inputs", leaky)
    with pytest.raises(T.LookaheadError):
        W.features(panel([dt.date(2023, 7, 10)], zones=["WEST"]), store)


def test_panel_checks():
    D = pd.Timestamp("2023-07-10")
    wrong_day = pd.MultiIndex.from_tuples([(D, "WEST", pd.Timestamp("2023-07-10 05:00", tz=TZ))],
                                          names=["bid_date", "zone", "delivery_hour"])
    with pytest.raises(ValueError):
        W.panel_frame(wrong_day)
    naive = pd.MultiIndex.from_tuples([(D, "WEST", pd.Timestamp("2023-07-11 05:00"))],
                                      names=["bid_date", "zone", "delivery_hour"])
    with pytest.raises(TypeError):
        W.panel_frame(naive)


@pytest.mark.skipif(not (PARQUET / "weather_gfs.parquet").exists(), reason="gene tables not built")
def test_real_tables_build_years_coverage():
    """Counts only. 40 build-year bid days from 26 Mar 2021 (fixed seed), every zone and hour."""
    import random
    import json
    days = [dt.date(2021, 3, 26) + dt.timedelta(days=i) for i in range((dt.date(2023, 12, 30) - dt.date(2021, 3, 26)).days)]
    sample = sorted(random.Random(20261006).sample(days, 40))
    idx = panel(sample)
    f = W.features(idx, W.default_store())
    has_day3 = (W.default_store().table("weather_gfs")["run_lead_hours"] == 72).any()
    share = f.notna().mean().round(4).to_dict()
    late = pd.Series(idx.get_level_values("delivery_hour")).dt.hour.ge(22).to_numpy()
    out = {"bid_days": len(sample), "rows": len(f), "day3_in_table": bool(has_day3),
           "non_null_share": share,
           "rows_by_lead": f["wx_run_lead_hours"].value_counts(dropna=False).to_dict(),
           "late_hours_non_null_share": float(f.loc[late, "wx_temp_f"].notna().mean())}
    Path(PARQUET.parent / "results").mkdir(exist_ok=True)
    (PARQUET.parent / "results" / "features_weather_coverage.json").write_text(json.dumps(out, indent=1, default=str))
    assert share["lf_mw"] > 0.99 and share["lf_d1_peak_mw"] > 0.99
    assert f.loc[~late, "wx_temp_f"].notna().mean() > 0.97
    if has_day3:
        assert out["late_hours_non_null_share"] > 0.97 and share["wx_d1_cdh65"] > 0.97


def test_holdout_panel_is_refused(store, monkeypatch):
    """A panel row delivered on or after 2024-01-01 raises while the study is locked."""
    import lock
    monkeypatch.delenv(lock.ENV_FLAG, raising=False)
    with pytest.raises(lock.HoldoutLocked):
        W.features(panel([dt.date(2023, 12, 31)], zones=["WEST"]), store)
