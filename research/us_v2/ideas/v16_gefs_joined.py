"""V16: idea V4 (weather forecasts against the operator's load forecast) on one consistent forecast source,
GEFS version 12 throughout (OBJECTIVES.md, V16 addendum, 7 Oct 2026):
    runs 2010-01-01 .. 2019-12-31   GEFS v12 reforecast (noaa-gefs-retrospective)
    runs 2020-09-23 .. 2023-12-31   archived live GEFS v12 (noaa-gefs-pds, atmos/pgrb2sp25)
control member c00, 00 UTC run of D, leads 27-54 h, the same 12 points, published_at = init + 8 h for both.
Table: parquet_v2/weather_gefs_joined.parquet (build_tables_v2.py gefs_joined), with a `source` column.

Model, features, rule and rolling windows are V4's, unchanged: this module only points V4 at the joined
table. January to September 2020 has no forecast and is not traded: delivery days 2020-01-01 to 2020-09-23
(bid days 2019-12-31 to 2020-09-22) are dropped from the positions (NOT_TRADED below).
"""
from __future__ import annotations

import pandas as pd

import v4_reforecast as V4

NAME = "V16"
TABLE = "weather_gefs_joined"
COLUMNS = V4.COLUMNS
ROWS = {"V16_B_gefs_joined": dict(features="base+V4", rule="two_sided",
                                  note="V4 on GEFS v12 reforecast to 2019 + live GEFS v12 from 2020-09-23")}
# delivery days with no run of either source behind them (bid days 2019-12-31 .. 2020-09-22 would use the
# runs of 2020-01-01 .. 2020-09-22, which do not exist; 1 Jan 2020 is dropped too: January 2020 not traded)
NOT_TRADED = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-23"))      # delivery date, inclusive both


def features(panel_index, panel=None, store=None) -> pd.DataFrame:
    return V4.features(panel_index, panel=panel, store=store, table=TABLE)


def traded(delivery_date) -> pd.Series:
    d = pd.to_datetime(pd.Series(delivery_date))
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    d = d.dt.normalize()
    return ~((d >= NOT_TRADED[0]) & (d <= NOT_TRADED[1]))


rule = V4.rule
