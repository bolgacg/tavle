"""Objective 2 timing test. Run on gene:  cd ~/nyiso-us/pipeline && ../.venv/bin/python -m pytest -q test_timing.py

For 500 random bid days D (Jan 2020 .. Sep 2026, fixed seed) every row used for delivery day D+1
must have published_at <= 05:00 New York time on D. Real lookahead rows (the isolf file named
D+1, the target day's own day-ahead prices, the outSched snapshot for D+1, real-time prices for
hours not yet over, GFS values not yet public) are injected and must be caught.
Only timestamps and row counts are compared; no price statistic is computed (holdout rule).
"""
from __future__ import annotations

import datetime as dt
import json
import random
from collections import Counter

import pandas as pd
import pytest

import timing as T
from common import FIRST_DAY, LAST_DAY, PARQUET, RESULTS, TZ, decision_time, local_midnight

pytestmark = pytest.mark.skipif(not (PARQUET / "prices_zone.parquet").exists(), reason="tables not built")

N_DAYS, SEED = 500, 20261006
ALL_BID_DAYS = [FIRST_DAY + dt.timedelta(days=i) for i in range((LAST_DAY - FIRST_DAY).days)]  # D+1 <= LAST_DAY
SAMPLE = sorted(random.Random(SEED).sample(ALL_BID_DAYS, N_DAYS))


@pytest.fixture(scope="module")
def store():
    s = T.Store()
    for name in T.TABLES:
        s.table(name)
    return s


def test_sample_spans_every_year():
    assert {d.year for d in SAMPLE} == set(range(2020, 2027))


def test_500_bid_days_no_lookahead(store):
    failures, cover = [], Counter()
    for D in SAMPLE:
        t = decision_time(D)
        x = T.bid_inputs(D, store, include_gen=True, gen_lookback_days=2)
        try:
            T.assert_no_lookahead(x, t)
        except T.LookaheadError as e:
            failures.append(f"{D}: {e}")
        d1 = pd.Timestamp(D + dt.timedelta(days=1), tz=TZ)
        # No price for the delivery day itself may be present.
        for k in ("da_prices_zone", "rt_prices_zone", "da_prices_gen", "rt_prices_gen"):
            if k in x and len(x[k]) and x[k]["delivery_hour"].max() >= d1:
                failures.append(f"{D}: {k} holds delivery-day rows")
        # The load-forecast vintage must be issued on or before D (file named D is written ~07:05 on D-1).
        lf = x["load_forecast_d1"]
        if len(lf) and pd.to_datetime(lf["issue_date"]).max() > pd.Timestamp(D):
            failures.append(f"{D}: load forecast from a file named after D")
        cover["days"] += 1
        cover["lf_d1_complete"] += int(len(lf) >= 12 * 23)
        cover["lf_from_file_D"] += int(len(lf) and (pd.to_datetime(lf["issue_date"]) == pd.Timestamp(D)).all())
        cover["outage_snapshot_D"] += int(len(x["outages_latest"]) and
                                          pd.Timestamp(x["outages_latest"]["snapshot_date"].iloc[0]) == pd.Timestamp(D))
        cover["gfs_d1_hours_ge_22"] += int(x["weather_d1"]["target_hour"].nunique() >= 22)
        cover["da_for_D_present"] += int(len(x["da_prices_zone"]) and
                                         local_midnight(x["da_prices_zone"]["delivery_hour"]).max() == pd.Timestamp(D))
        for k, n in x.excluded.items():
            cover[f"excluded_rows_{k}"] += n
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "timing_test.json").write_text(json.dumps(
        {"bid_days": N_DAYS, "seed": SEED, "failures": len(failures), "coverage_counts": dict(cover)}, indent=1))
    assert not failures, failures[:10]
    # The filter must actually have removed future rows from the candidate windows.
    for k in ("da_prices_zone", "rt_prices_zone", "load_forecast", "outages", "weather_gfs"):
        assert cover[f"excluded_rows_{k}"] > 0, k


def _inject(x: T.Frames, name: str, rows: pd.DataFrame) -> dict:
    bad = dict(x)
    bad[name] = pd.concat([x[name], rows], ignore_index=True)
    return bad


INJECT_DAYS = SAMPLE[::25]


@pytest.mark.parametrize("D", INJECT_DAYS, ids=str)
def test_injected_lookahead_is_caught(store, D):
    t = decision_time(D)
    x = T.bid_inputs(D, store)
    T.assert_no_lookahead(x, t)                                        # clean set passes
    d1 = D + dt.timedelta(days=1)
    caught = 0
    # 1. the isolf file named D+1 (written ~07:05 on D, after the deadline)
    lf = store.table("load_forecast")
    rows = lf[lf["issue_date"] == pd.Timestamp(d1)]
    # 2. the target day's own day-ahead prices
    pz = store.window("prices_zone", pd.Timestamp(d1, tz=TZ), pd.Timestamp(d1 + dt.timedelta(days=1), tz=TZ))
    da = pz[T.DA_COLS].rename(columns={"da_published_at": "published_at"})
    # 3. real-time price of the hour starting 05:00 on D (ends after the deadline)
    rt_d =store.window("prices_zone", pd.Timestamp(D, tz=TZ) + pd.Timedelta(hours=5),
                        pd.Timestamp(D, tz=TZ) + pd.Timedelta(hours=6))
    rt = rt_d[T.RT_COLS].rename(columns={"rt_published_at": "published_at"}).dropna(subset=["published_at"])
    # 4. the outSched snapshot for D+1 (written ~09:40 on D)
    og = store.table("outages")
    osd = og[og["snapshot_date"] == pd.Timestamp(d1)]
    # 5. a GFS value for the evening of D+1 issued too late (published_at after t)
    wx = store.window("weather_gfs", t, t + pd.Timedelta(days=3))
    wx = wx[wx["published_at"] > t].head(5)
    for name, inj in (("load_forecast_d1", rows), ("da_prices_zone", da), ("rt_prices_zone", rt),
                      ("outages_latest", osd), ("weather_d1", wx)):
        if len(inj) == 0:
            continue                                                   # source missing that day
        with pytest.raises(T.LookaheadError):
            T.assert_no_lookahead(_inject(x, name, inj), t)
        caught += 1
    assert caught >= 3


def test_filter_drops_future_rows_placed_in_store(store):
    """A store holding only future rows returns nothing for them."""
    D = SAMPLE[100]
    t = decision_time(D)
    lf = store.table("load_forecast")
    fut = lf[lf["published_at"] > t]
    fut = fut[fut["issue_date"] <= pd.Timestamp(D + dt.timedelta(days=2))].head(500)
    s2 = T.Store(tables={"prices_zone": store.table("prices_zone").head(0), "load_forecast": fut,
                         "outages": store.table("outages").head(0),
                         "weather_gfs": store.table("weather_gfs").head(0)})
    f = T.features_available_at(t, s2)
    assert len(fut) > 0 and len(f["load_forecast"]) == 0 and f.excluded["load_forecast"] == len(fut)


@pytest.mark.parametrize("table", T.TABLES)
def test_published_at_follows_rules(store, table):
    assert T.rule_violations(table, store.table(table)) == {}


def test_published_at_rules_for_every_gen_month():
    import pyarrow.parquet as pq
    bad = {}
    for f in sorted((PARQUET / "prices_gen").glob("*.parquet")):
        v = T.rule_violations("prices_gen", pq.read_table(f, columns=[
            "delivery_hour", "da_published_at", "da_file_written_at", "rt_published_at",
            "rt_file_written_at", "rt_revised", "published_at"]).to_pandas())
        if v:
            bad[f.stem] = v
    assert bad == {}


def test_mislabelled_row_is_caught_by_rules(store):
    """A target-day DA row whose published_at was wrongly set to D-1 would pass the filter, so
    the rule check must catch it."""
    pz = store.table("prices_zone")
    row = pz[pz["da_published_at"].notna()].iloc[[5000]].copy()
    row["da_published_at"] = row["da_published_at"] - pd.Timedelta(days=1)
    row["published_at"] = row["rt_published_at"]
    assert T.rule_violations("prices_zone", row).get("da_before_11h_bound") == 1
    g = store.table("weather_gfs").iloc[[100]].copy()
    g["published_at"] = g["published_at"] - pd.Timedelta(hours=1)
    assert T.rule_violations("weather_gfs", g)


def test_dst_days_have_23_and_25_hours(store):
    """Counts only: spring-forward days have 23 delivery hours per zone, fall-back days 25."""
    pz = store.table("prices_zone")
    n = pz[pz["zone"] == "CAPITL"].groupby(local_midnight(pz.loc[pz["zone"] == "CAPITL", "delivery_hour"])).size()
    spring = [pd.Timestamp(d) for d in ("2020-03-08", "2021-03-14", "2022-03-13", "2023-03-12",
                                        "2024-03-10", "2025-03-09", "2026-03-08")]
    fall = [pd.Timestamp(d) for d in ("2020-11-01", "2021-11-07", "2022-11-06", "2023-11-05",
                                      "2024-11-03", "2025-11-02")]
    assert all(n[d] == 23 for d in spring) and all(n[d] == 25 for d in fall)
