"""v2 timing test (objective 2 over 2010-2023). Run on gene:
    cd ~/nyiso-us/pipeline_v2 && ../.venv/bin/python -m pytest -q -p no:cacheprovider test_timing_v2.py

Reuses the v1 tests unchanged (research/us/pipeline/test_timing.py) through the v2 `common` shim, so
SAMPLE = 500 bid days drawn with seed 20261006 from 1 Jan 2010 to 30 Dec 2023 and every table is read from
parquet_v2. Two v1 tests hard-code 2020-2026 dates and are replaced here; the reforecast table and the
built feature blocks get their own checks. Timestamps and counts only; no price statistic.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import timing as T  # noqa: E402
from test_timing import (SAMPLE, INJECT_DAYS, store, test_500_bid_days_no_lookahead,  # noqa: E402,F401
                         test_filter_drops_future_rows_placed_in_store, test_injected_lookahead_is_caught,
                         test_mislabelled_row_is_caught_by_rules, test_published_at_follows_rules,
                         test_published_at_rules_for_every_gen_month, test_three_day_rule_on_dst_days)

assert T.PARQUET == C.PARQUET, "timing did not pick up the v2 shim"

# v2 load_forecast rule: published_at = later(file_written_at, 06:00 on issue_date - 1) (build_tables_v2.
# lf_published_at). Patched into timing so the reused v1 rule test checks the v2 rule for that table.
_v1_rules = T.rule_violations


def _v2_rules(table, df):
    if table != "load_forecast":
        return _v1_rules(table, df)
    import build_tables_v2 as BV
    exp = BV.lf_published_at(df["issue_date"], df["file_written_at"])
    v = {"published_not_v2_rule": int((df["published_at"] != exp).sum()),
         "published_missing": int(df["published_at"].isna().sum())}
    return {k: n for k, n in v.items() if n}


T.rule_violations = _v2_rules


def test_no_vintage_named_after_bid_day_is_public_at_05(store):
    """Every load-forecast file named D+1 is published after 05:00 on D (all 5,113 files)."""
    lf = store.table("load_forecast")[["issue_date", "published_at"]].drop_duplicates("issue_date")
    dec = pd.Series([C.decision_time((d - pd.Timedelta(days=1)).date()) for d in pd.to_datetime(lf["issue_date"])],
                    index=lf.index)
    assert (lf["published_at"] > dec).all(), lf[lf["published_at"] <= dec]
RF = C.PARQUET / "weather_reforecast.parquet"
PARTS = C.PARQUET / "features" / "parts"


def _dst_days(years):
    spring, fall = [], []
    for y in years:
        mar = [dt.date(y, 3, d) for d in range(8, 15)]
        nov = [dt.date(y, 11, d) for d in range(1, 8)]
        spring.append(next(d for d in mar if d.weekday() == 6))
        fall.append(next(d for d in nov if d.weekday() == 6))
    return spring, fall


def test_sample_spans_every_year():
    assert {d.year for d in SAMPLE} == set(range(2010, 2024))
    assert max(SAMPLE) < dt.date(2023, 12, 31)


def test_no_row_on_or_after_holdout(store):
    end = pd.Timestamp("2024-01-01", tz=C.TZ)
    for name, col in (("prices_zone", "delivery_hour"), ("load_forecast", "target_hour"),
                      ("weather_gfs", "target_hour")):
        assert (store.table(name)[col] < end).all(), name


def test_dst_days_have_23_and_25_hours(store):
    pz = store.table("prices_zone")
    cap = pz[pz["zone"] == "CAPITL"]
    n = cap.groupby(C.local_midnight(cap["delivery_hour"])).size()
    spring, fall = _dst_days(range(2010, 2024))
    assert all(n[pd.Timestamp(d)] == 23 for d in spring), [(d, n[pd.Timestamp(d)]) for d in spring]
    assert all(n[pd.Timestamp(d)] == 25 for d in fall), [(d, n[pd.Timestamp(d)]) for d in fall]


# ------------------------------------------------------------------------------ reforecast
RFJ = C.PARQUET / "weather_gefs_joined.parquet"


@pytest.fixture(scope="module", params=["weather_reforecast", "weather_gefs_joined"])
def rf(request):
    """Every reforecast check runs on the reforecast table and on the V16 joined GEFS v12 table."""
    f = C.PARQUET / f"{request.param}.parquet"
    if not f.exists():
        pytest.skip(f"{request.param}.parquet not built")
    return pd.read_parquet(f)


def test_reforecast_published_at_rule(rf):
    assert (rf["published_at"] == rf["init_utc"] + C.REFORECAST_PUBLISH_LAG).all()
    assert (rf["init_utc"].dt.tz_convert("UTC").dt.hour == 0).all()
    assert (rf["target_hour"] == rf["init_utc"] + pd.to_timedelta(rf["lead_hours"], unit="h")).all()


def test_reforecast_run_public_before_its_decision(rf):
    """The 00 UTC run of D is public by 05:00 New York time on D, every day of the year (EST and EDT)."""
    D = rf["init_utc"].dt.tz_convert("UTC").dt.date
    dec = pd.Series([C.decision_time(d) for d in D.unique()], index=D.unique())
    assert (rf["published_at"].to_numpy() <= dec.reindex(D).to_numpy()).all()


def test_reforecast_leads_cover_every_hour_of_d1(rf):
    """Leads 27..54 bracket every local hour of D+1 for the run of D (23, 24 and 25-hour days)."""
    for init, g in list(rf.groupby("init_utc"))[::97]:
        D = init.tz_convert("UTC").date()
        d1 = pd.Timestamp(D + dt.timedelta(days=1), tz=C.TZ)
        hrs = pd.date_range(d1, pd.Timestamp(D + dt.timedelta(days=2), tz=C.TZ), freq="h", inclusive="left")
        assert g["target_hour"].min() <= hrs.min() and g["target_hour"].max() >= hrs.max(), D


def test_next_run_is_lookahead(rf):
    """Injected: the run of D+1 is published 03:00/04:00 on D+1, after the 05:00 deadline on D."""
    for D in INJECT_DAYS:
        nxt = rf[rf["init_utc"].dt.tz_convert("UTC").dt.date == D + dt.timedelta(days=1)]
        if len(nxt):
            assert (nxt["published_at"] > C.decision_time(D)).all()


def test_wxr_block_uses_only_public_runs():
    f = PARTS / "wxr_hourly.parquet"
    if not f.exists():
        pytest.skip("wxr block not built")
    w = pd.read_parquet(f, columns=["bid_date", "wxr_published_at", "delivery_hour"])
    dec = w["bid_date"].map(lambda d: C.decision_time(d.date()))
    assert (w["wxr_published_at"] <= dec).all()
    assert (C.local_midnight(w["delivery_hour"]) == w["bid_date"] + pd.Timedelta(days=1)).all()


def _wxr_fill(df, day_col, cols):
    """Share of non-missing wxr cells over delivery days 2010-2019 (the reforecast years)."""
    d = pd.to_datetime(df[day_col])
    d = d.dt.tz_convert(C.TZ) if d.dt.tz is not None else d
    m = (d.dt.year >= 2010) & (d.dt.year <= 2019)
    assert m.sum() > 0, f"no 2010-2019 delivery rows in {day_col}"
    return float(df.loc[m, cols].notna().to_numpy().mean()), int(m.sum())


def test_wxr_block_filled_2010_2019():
    """The reforecast weather block is not empty: at least 90 % filled for 2010-2019 delivery days in the
    hourly block, the panel's wxr_ columns and the day matrix's wxr__ block. Guards the 7 Oct 2026 bug where a
    seconds versus microseconds timestamp mismatch made every interpolated value NaN."""
    f = PARTS / "wxr_hourly.parquet"
    if not f.exists():
        pytest.skip("wxr block not built")
    w = pd.read_parquet(f, columns=["delivery_hour", "wxr_temp_c"])
    fill, n = _wxr_fill(w, "delivery_hour", ["wxr_temp_c"])
    assert fill >= 0.9, f"wxr_hourly {fill:.1%} filled over {n} rows"
    feat = C.PARQUET / "features"
    pcols = ["wxr_temp_f", "wxr_d1_max_f", "wxr_d1_min_f", "wxr_d1_mean_f", "wxr_d1_hdh65", "wxr_d1_cdh65"]
    p = pd.read_parquet(feat / "panel_2010_2023.parquet", columns=["delivery_hour"] + pcols)
    fill, n = _wxr_fill(p, "delivery_hour", pcols)
    assert fill >= 0.9, f"panel wxr_ columns {fill:.1%} filled over {n} rows"
    import pyarrow.parquet as pq
    dcols = [c for c in pq.read_schema(feat / "day_2010_2023.parquet").names if c.startswith("wxr__")]
    assert dcols, "day matrix has no wxr__ columns"
    d = pd.read_parquet(feat / "day_2010_2023.parquet", columns=["delivery_date"] + dcols)
    fill, n = _wxr_fill(d, "delivery_date", dcols)
    assert fill >= 0.9, f"day matrix wxr__ block {fill:.1%} filled over {n} days"


# ------------------------------------------------------------------------------ archived forecasts only
@pytest.mark.parametrize("name", ["weather_gfs", "weather_reforecast", "weather_gefs_joined"])
def test_weather_rows_are_archived_forecasts(name):
    """Bo's rule: weather inputs are archived forecasts only, never observations or reanalysis. Every row
    names its archive (source), run time (issue_time) and lead; published_at = run time + delay."""
    import build_tables_v2 as BV
    f = C.PARQUET / f"{name}.parquet"
    if not f.exists():
        pytest.skip(f"{name} not built")
    df = pd.read_parquet(f)
    assert BV.weather_violations(name, df) == {}
    assert set(df["source"].unique()) == BV.WEATHER_SOURCE[name]


def test_weather_used_for_a_bid_is_issued_before_05(store):
    """Every weather row a bid can see was issued and published before 05:00 New York time on D."""
    n = 0
    for D in SAMPLE[::5]:
        t = C.decision_time(D)
        f = T.features_available_at(t, store, lookback_days=2)
        w = f["weather_gfs"]
        if len(w):
            assert (w["published_at"] <= t).all()
            issue = w["target_hour"] - pd.to_timedelta(w["run_lead_hours"], unit="h") + C.GFS_NEAREST_RUN_SLACK
            assert (issue < t).all()
            n += len(w)
    assert n > 0


# ------------------------------------------------------------------------------ V16 joined GEFS v12
GAP0, LIVE0 = pd.Timestamp("2020-01-01"), pd.Timestamp(C.GEFS_LIVE_FIRST)


@pytest.fixture(scope="module")
def rfj():
    if not RFJ.exists():
        pytest.skip("weather_gefs_joined.parquet not built")
    return pd.read_parquet(RFJ)


def test_gefs_joined_sources_and_documented_gap(rfj):
    """Reforecast runs only to 2019-12-31, live GEFS v12 runs only from 2020-09-23, no run in between,
    every run 00 UTC, every valid time before 2024-01-01."""
    import build_tables_v2 as BV
    init = rfj["init_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    ref, live = rfj["source"] == BV.SOURCE_REFORECAST, rfj["source"] == BV.SOURCE_GEFS_LIVE
    assert (ref | live).all()
    assert (init[ref] < GAP0).all() and (init[live] >= LIVE0).all()
    assert not ((init >= GAP0) & (init < LIVE0)).any()
    assert (init.dt.hour == 0).all()
    assert (rfj["target_hour"] < pd.Timestamp("2024-01-01", tz=C.TZ)).all()
    yrs = init.dt.year.value_counts()
    assert set(range(2010, 2024)) <= set(yrs.index), sorted(yrs.index)


def test_gefs_live_files_stop_before_holdout():
    """The build chooses live files by name and never opens one dated on or after 2024-01-01."""
    import build_tables_v2 as BV
    files = BV.gefs_live_files()
    assert files and all(f.stem < "20240101" and f.stem >= "20200923" for f in files)


def test_wxr_joined_block_uses_only_public_runs():
    f = PARTS / "wxr_joined_hourly.parquet"
    if not f.exists():
        pytest.skip("wxr joined block not built")
    w = pd.read_parquet(f, columns=["bid_date", "wxr_published_at", "delivery_hour"])
    dec = w["bid_date"].map(lambda d: C.decision_time(d.date()))
    assert (w["wxr_published_at"] <= dec).all()
    assert (C.local_midnight(w["delivery_hour"]) == w["bid_date"] + pd.Timedelta(days=1)).all()


def test_wxr_joined_block_filled_except_documented_gap():
    """Fill test with the documented gap: at least 90 % of point-hours filled for bid days 2010-2019 and
    2020-09-23 onward; no row at all for bid days 2020-01-01 to 2020-09-22 (no run of either source)."""
    f = PARTS / "wxr_joined_hourly.parquet"
    if not f.exists():
        pytest.skip("wxr joined block not built")
    w = pd.read_parquet(f, columns=["bid_date", "wxr_temp_c", "source"])
    b = w["bid_date"]
    gap = (b >= GAP0) & (b < LIVE0)
    assert not gap.any(), int(gap.sum())
    for name, m in (("2010-2019", b < GAP0), ("2020-09-23..2023", b >= LIVE0)):
        assert m.sum() > 0, name
        fill = float(w.loc[m, "wxr_temp_c"].notna().mean())
        assert fill >= 0.9, f"{name}: {fill:.1%} filled"
    for y in range(2010, 2024):
        n = b.dt.year.eq(y).sum()
        assert n > 0, y
    assert (b.dt.year <= 2023).all()
