"""Every-day timing test for the run window (dryrun: delivery 2023-01-01..2023-12-31; heldout:
2024-01-01..2026-09-30). Skipped in build mode. Run by run_heldout_data.sh after test_timing_v2.py:

    US_RUN_MODE=dryrun US_RUN_DIR=... ../.venv/bin/python -m pytest -q -p no:cacheprovider test_window_v2.py

test_timing_v2 samples 500 bid days over the whole range; here EVERY bid day whose delivery day lies in
the window is checked: no input row published after 05:00 New York time on D, no delivery-day price, no
load-forecast file named after D, and every weather row an archived forecast issued and published before
05:00 on D. The built weather tables and blocks are checked row by row over the window, and the panel
and day matrix must hold one bid day before each delivery day. Timestamps and counts only.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pytest  # noqa: E402

import timing as T  # noqa: E402
from test_timing import store  # noqa: E402,F401

pytestmark = pytest.mark.skipif(C.RUN_MODE == "build", reason="window test runs in dryrun / heldout mode only")

if C.RUN_MODE != "build":
    W0, W1 = C._HM.window()
    assert W1 == C.LAST_DAY, (W1, C.LAST_DAY)
else:
    W0 = W1 = C.LAST_DAY
BID_DAYS = [W0 - dt.timedelta(days=1) + dt.timedelta(days=i) for i in range((W1 - W0).days + 1)]
LO = pd.Timestamp(W0, tz=C.TZ)
SPAN = f"{C.FIRST_DAY.year}_{C.LAST_DAY.year}"
PARTS = C.PARQUET / "features" / "parts"


def _window_rows(path: Path, col: str, lo, columns=None) -> pd.DataFrame:
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(col).type
    return d.to_table(filter=ds.field(col) >= ds.scalar(lo).cast(typ), columns=columns).to_pandas()


def test_every_bid_day_in_window(store):
    failures, cover = [], {"bid_days": 0, "lf_d1_complete": 0, "gfs_d1_rows": 0, "outage_snapshot": 0,
                           "da_for_D_present": 0}
    for D in BID_DAYS:
        t = C.decision_time(D)
        x = T.bid_inputs(D, store, include_gen=True, gen_lookback_days=2)
        try:
            T.assert_no_lookahead(x, t)
        except T.LookaheadError as e:
            failures.append(f"{D}: {e}")
        d1 = pd.Timestamp(D + dt.timedelta(days=1), tz=C.TZ)
        for k in ("da_prices_zone", "rt_prices_zone", "da_prices_gen", "rt_prices_gen"):
            if k in x and len(x[k]) and x[k]["delivery_hour"].max() >= d1:
                failures.append(f"{D}: {k} holds delivery-day rows")
        lf = x["load_forecast_d1"]
        if len(lf) and pd.to_datetime(lf["issue_date"]).max() > pd.Timestamp(D):
            failures.append(f"{D}: load forecast from a file named after D")
        wx = x["weather_d1"]
        if len(wx):
            issue = wx["target_hour"] - pd.to_timedelta(wx["run_lead_hours"], unit="h") + C.GFS_NEAREST_RUN_SLACK
            if not ((issue < t) & (wx["published_at"] <= t)).all():
                failures.append(f"{D}: GFS row issued or published after 05:00")
        cover["bid_days"] += 1
        cover["lf_d1_complete"] += int(len(lf) >= 12 * 23)
        cover["gfs_d1_rows"] += len(wx)
        cover["outage_snapshot"] += int(len(x["outages_latest"]) > 0)
        cover["da_for_D_present"] += int(len(x["da_prices_zone"]) > 0 and C.local_midnight(
            x["da_prices_zone"]["delivery_hour"]).max() == pd.Timestamp(D))
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "timing_window.json").write_text(json.dumps(
        {"mode": C.RUN_MODE, "window": [str(W0), str(W1)], "failures": len(failures), "coverage_counts": cover},
        indent=1))
    assert not failures, failures[:10]
    assert cover["bid_days"] == len(BID_DAYS)


@pytest.mark.parametrize("name", ["weather_gfs", "weather_gefs_joined"])
def test_weather_rows_in_window(name):
    """Archived forecasts only, and every run that serves a window bid day is public by 05:00 on its day."""
    import build_tables_v2 as BV
    w = _window_rows(C.PARQUET / f"{name}.parquet", "target_hour", LO - pd.Timedelta(days=3))
    assert len(w), name
    assert BV.weather_violations(name, w) == {}
    if name == "weather_gefs_joined":
        D = w["init_utc"].dt.tz_convert("UTC").dt.date
        dec = pd.Series({d: C.decision_time(d) for d in D.unique()})
        assert (w["published_at"].to_numpy() <= dec.reindex(D).to_numpy()).all()
        assert (w["issue_time"].to_numpy() < dec.reindex(D).to_numpy()).all()
        assert (w["target_hour"] < pd.Timestamp(C.LAST_DAY + dt.timedelta(days=1), tz=C.TZ)).all()


@pytest.mark.parametrize("part", ["wxr_hourly.parquet", "wxr_joined_hourly.parquet"])
def test_wxr_blocks_in_window(part):
    w = _window_rows(PARTS / part, "delivery_hour", LO, ["bid_date", "delivery_hour", "wxr_published_at"])
    if part == "wxr_hourly.parquet":
        assert len(w) == 0                          # reforecast block ends with 2019 runs
        return
    assert len(w), part
    dec = w["bid_date"].map(lambda d: C.decision_time(d.date()))
    assert (w["wxr_published_at"] <= dec).all()
    assert (C.local_midnight(w["delivery_hour"]) == w["bid_date"] + pd.Timedelta(days=1)).all()


def test_panel_and_day_cover_window():
    p = _window_rows(C.PARQUET / "features" / f"panel_{SPAN}.parquet", "delivery_hour", LO,
                     ["bid_date", "delivery_date", "delivery_hour", "zone", "label_published_at"])
    assert len(p)
    assert (p["delivery_date"] == p["bid_date"] + pd.Timedelta(days=1)).all()
    assert (C.local_midnight(p["delivery_hour"]) == p["delivery_date"]).all()
    dec = p["bid_date"].map(lambda d: C.decision_time(d.date()))
    assert (p["label_published_at"] > dec).all()        # the label is never public at the decision
    assert set(p["zone"]) == set(C.ZONES)
    d = _window_rows(C.PARQUET / "features" / f"day_{SPAN}.parquet", "delivery_date", W0,
                     ["bid_date", "delivery_date"])
    assert (pd.to_datetime(d["delivery_date"]) == pd.to_datetime(d["bid_date"]) + pd.Timedelta(days=1)).all()
    days = {pd.Timestamp(D + dt.timedelta(days=1)) for D in BID_DAYS}
    cover = {"panel_delivery_days": int(p["delivery_date"].nunique()),
             "day_matrix_delivery_days": int(pd.to_datetime(d["delivery_date"]).nunique()),
             "window_days": len(days),
             "panel_days_missing": sorted(str(x.date()) for x in days - set(p["delivery_date"]))[:50]}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "window_cover.json").write_text(json.dumps(cover, indent=1))
    assert set(p["delivery_date"]) <= days
