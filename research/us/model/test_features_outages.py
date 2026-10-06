"""Tests for features_outages (idea D). Synthetic tests run anywhere; the last test needs the gene
parquet tables and is skipped without them. No price is read (holdout rule).

    cd research/us/model && python -m pytest -q -p no:cacheprovider test_features_outages.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_outages as O  # noqa: E402
import timing as T  # noqa: E402
from common import PARQUET, TZ, ZONES, decision_time  # noqa: E402

D = dt.date(2023, 7, 10)            # bid day; delivery day D+1 = 11 Jul 2023


def ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)                                   # an explicit offset settles fall-back 01:xx
    return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)


# Outages as scheduled; each snapshot S lists the ones active on day S (as NYISO's files do).
OUTAGES = [
    ("FARRAGUT_345KV_7W",            "2023-07-01 08:00", "2023-07-20 16:00"),   # 345, all of D+1
    ("SWEDEN__-MORTIMER_115_111",    "2023-07-08 06:00", "2023-07-11 12:00"),   # other, ends 12:00 on D+1
    ("E13THSTA_345_138_BK 15",       "2023-07-10 09:00", "2023-07-13 00:00"),   # 345, started on D
    ("ASTORIAE-ASTORIAG_138_34124M", "2023-07-09 07:00", "2023-07-10 18:00"),   # 138, ends on D: not on D+1
    ("STLAWRNC-MOSES____230_L34P",   "2023-07-05 00:00", "2023-07-11 06:30"),   # 230, ends 06:30 on D+1
    ("SCH-HQ-NYISO_LIMIT_1200",      "2023-07-10 00:00", "2023-07-12 00:00"),   # interface limit, other
    ("LOOKAHEA_345KV_1",             "2023-07-11 00:00", "2023-07-11 23:00"),   # starts D+1: only in snapshot D+1
]


def synthetic_outages(first="2023-06-25", last="2023-11-10", extra=()) -> pd.DataFrame:
    rows = []
    for S in pd.date_range(first, last, freq="D"):
        written = ts(f"{(S - pd.Timedelta(days=1)).date()} 09:40")
        for eq, a, b in list(OUTAGES) + list(extra):
            out, back = ts(a), ts(b)
            if out < ts(str(S.date())) + pd.Timedelta(days=1) and back > ts(str(S.date())):
                rows.append({"snapshot_date": S, "file_written_at": written, "published_at": written, "ptid": 1,
                             "equipment": eq, "sched_out": out, "sched_in": back})
    return pd.DataFrame(rows)


def store_with(og: pd.DataFrame) -> T.Store:
    import test_gfs_rule as G
    e = G.empty_tables()
    wx = pd.DataFrame({"point": pd.Series(dtype=str), "zone": pd.Series(dtype=str),
                       "target_hour": pd.Series(pd.DatetimeIndex([], tz=TZ)),
                       "temperature_2m_c": pd.Series(dtype=float), "run_lead_hours": pd.Series(dtype=int),
                       "published_at": pd.Series(pd.DatetimeIndex([], tz=TZ))})
    return T.Store(tables={"prices_zone": e["prices_zone"], "load_forecast": e["load_forecast"],
                           "outages": og, "weather_gfs": wx})


def panel(bid_days, zones=ZONES) -> pd.MultiIndex:
    rows = []
    for B in bid_days:
        d1 = pd.Timestamp(B + dt.timedelta(days=1), tz=TZ)
        for h in pd.date_range(d1, pd.Timestamp(B + dt.timedelta(days=2), tz=TZ), freq="h", inclusive="left"):
            for z in zones:
                rows.append((pd.Timestamp(B), z, h))
    return pd.MultiIndex.from_tuples(rows, names=["bid_date", "zone", "delivery_hour"])


@pytest.fixture(scope="module")
def store():
    return store_with(synthetic_outages())


@pytest.fixture(scope="module")
def feats(store):
    idx = panel([D], zones=["N.Y.C.", "WEST"])
    return idx, O.features(idx, store, top=O.top_sites(store.table("outages")))


def row(idx, f, hour, zone="N.Y.C."):
    return f.loc[(pd.Timestamp(D), zone, ts("2023-07-11") + pd.Timedelta(hours=hour))]


def test_name_parsing():
    assert O.voltage_kv("FARRAGUT_345KV_7W") == 345 and O.voltage_kv("HUDSONP_-FARRAGUT_345_B3402") == 345
    assert O.voltage_kv("E13THSTA_345_138_BK 15") == 345 and O.voltage_kv("STLAWRNC_230B_230D_PS 34") == 230
    assert O.voltage_kv("KERHONKS_69_KV_MK-234-FK") == 69 and O.voltage_kv("SCH-HQ-NYISO_LIMIT_1200") is None
    assert O.voltage_class("RAMAPO___500KV_1") == "345kv_up" and O.voltage_class("RAINEY_138A_138B_PAR_5") == "138kv"
    assert O.voltage_class("SWEDEN__-MORTIMER_115_111") == "other_kv" and O.voltage_class("CHAT_DC_GC2") == "other_kv"
    assert O.sites("HUDSONP_-FARRAGUT_345_B3402") == ["HUDSONP", "FARRAGUT"]
    assert O.sites("LIGHTHSE-MALLORY__115_7") == ["LIGHTHSE", "MALLORY"]
    assert O.sites("BUCHAN_S_345KV_3") == ["BUCHAN_S"] and O.sites("RAINEY_138A_138B_PAR_5") == ["RAINEY"]
    assert O.sites("SCH-HQ_CEDARS-NYISO_LIMIT_200") == ["SCH-HQ_CEDARS-NYISO"]


def test_alignment_and_same_across_zones(feats):
    idx, f = feats
    assert f.index.equals(idx) and len(f) == 48
    nyc = f.xs("N.Y.C.", level="zone").to_numpy()
    west = f.xs("WEST", level="zone").to_numpy()
    assert (nyc == west).all()                                   # system-wide features


def test_hourly_counts_by_voltage(feats):
    idx, f = feats
    h0, h7, h13 = row(idx, f, 0), row(idx, f, 7), row(idx, f, 13)
    # 00:00: FARRAGUT, E13THSTA (345), SWEDEN (other), STLAWRNC (230), SCH (other); LOOKAHEA unseen
    assert (h0["out_n_active"], h0["out_n_345kv_up"], h0["out_n_230kv"], h0["out_n_138kv"], h0["out_n_other_kv"]) \
        == (5, 2, 1, 0, 2)
    assert (h7["out_n_active"], h7["out_n_230kv"]) == (4, 0)                 # STLAWRNC back at 06:30
    assert (h13["out_n_active"], h13["out_n_other_kv"]) == (3, 1)           # SWEDEN back at 12:00
    assert h0["out_n_138kv"] == 0                                           # ASTORIA ended on D


def test_daily_counts(feats):
    idx, f = feats
    r = row(idx, f, 0)
    # ending on D+1: SWEDEN (12:00), STLAWRNC (06:30); started on D: E13THSTA, SCH
    assert r["out_n_ending_d1"] == 2 and r["out_n_started_d"] == 2
    assert r["out_n_in_snapshot"] == 6 and r["out_snapshot_age_days"] == 0


def test_snapshot_d_plus_1_is_not_used(feats, store):
    """Snapshot D+1 (written 09:40 on D) holds LOOKAHEA, starting on D+1. It must not count."""
    og = store.table("outages")
    assert (og[og["snapshot_date"] == pd.Timestamp("2023-07-11")]["equipment"] == "LOOKAHEA_345KV_1").any()
    idx, f = feats
    assert "out_sub_lookahea" not in f.columns or (f["out_sub_lookahea"] == 0).all()
    assert row(idx, f, 1)["out_n_345kv_up"] == 2


def test_site_indicators(feats):
    idx, f = feats
    assert row(idx, f, 0)["out_sub_mortimer"] == 1 and row(idx, f, 13)["out_sub_mortimer"] == 0
    assert row(idx, f, 0)["out_sub_farragut"] == 1 and row(idx, f, 23)["out_sub_farragut"] == 1


def test_missing_snapshot_uses_previous(store):
    og = store.table("outages")
    s2 = store_with(og[og["snapshot_date"] != pd.Timestamp(D)])
    f = O.features(panel([D], zones=["WEST"]), s2, top=[])
    assert (f["out_snapshot_age_days"] == 1).all()


def test_top_sites_ignore_holdout_years():
    """A site with many outage starts in 2024 must not enter the build-year list."""
    extra = [(f"ZZHOLD__345KV_{i}", f"2024-0{1 + i % 6}-1{i % 9} 08:00", f"2024-0{1 + i % 6}-1{i % 9} 20:00")
             for i in range(40)]
    og = synthetic_outages("2023-06-25", "2024-07-10", extra=extra)
    top = O.top_sites(og)
    assert "ZZHOLD" not in top and "FARRAGUT" in top


def test_dst_fall_back_has_25_hours():
    """D+1 = 5 Nov 2023 (25 hours): an outage ending 01:30 EST is active in both 01:00 hours."""
    extra = [("CLAY_____345KV_R440", "2023-11-01 08:00", "2023-11-05 01:30-05:00")]
    s = store_with(synthetic_outages("2023-10-25", "2023-11-08", extra=extra))
    idx = panel([dt.date(2023, 11, 4)], zones=["CAPITL"])
    f = O.features(idx, s, top=[])
    hours = pd.Series(idx.get_level_values("delivery_hour"))
    assert len(f) == 25 and f["out_n_active"].notna().all()
    # hours 0, 1 EDT, 1 EST are active (the EST 01:00 hour starts before 01:30 EST), 2:00 on is not
    assert f["out_n_active"].tolist()[:4] == [1, 1, 1, 0] and hours.iloc[2].utcoffset() == pd.Timedelta(hours=-5)


def test_injected_lookahead_raises(store, monkeypatch):
    """If bid_inputs ever handed over the D+1 snapshot, features() must refuse."""
    real = T.bid_inputs

    def leaky(bid_day, s=None, **kw):
        x = real(bid_day, s, **kw)
        og = s.table("outages")
        x["outages_latest"] = pd.concat([x["outages_latest"],
                                         og[og["snapshot_date"] == pd.Timestamp(bid_day + dt.timedelta(days=1))]],
                                        ignore_index=True)
        return x

    monkeypatch.setattr(T, "bid_inputs", leaky)
    with pytest.raises(T.LookaheadError):
        O.features(panel([D], zones=["WEST"]), store, top=[])


def test_published_at_filter_is_what_excludes(store):
    """Mark snapshot D as written after 05:00 on D: the filter must fall back to snapshot D-1."""
    og = store.table("outages").copy()
    late = og["snapshot_date"] == pd.Timestamp(D)
    og.loc[late, "published_at"] = decision_time(D) + pd.Timedelta(minutes=1)
    f = O.features(panel([D], zones=["WEST"]), store_with(og), top=[])
    assert (f["out_snapshot_age_days"] == 1).all()


@pytest.mark.skipif(not (PARQUET / "outages.parquet").exists(), reason="gene tables not built")
def test_real_tables_build_years():
    """Counts only. The top-20 list from 2020 to 2023 and coverage on 60 build-year bid days."""
    import json
    import random
    s = O.default_store()
    top = O.top_sites(s.table("outages"))
    if O.TOP_SITES is not None:
        assert top == O.TOP_SITES
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range((dt.date(2023, 12, 30) - dt.date(2020, 1, 1)).days)]
    sample = sorted(random.Random(20261006).sample(days, 60))
    idx = panel(sample, zones=["CAPITL"])
    f = O.features(idx, s)
    out = {"top_sites": top, "bid_days": len(sample), "rows": len(f),
           "non_null_share": f.notna().mean().round(4).to_dict(),
           "snapshot_age_days": f["out_snapshot_age_days"].value_counts().to_dict(),
           "mean_active_by_class": {c: round(float(f[f"out_n_{c}"].mean()), 2) for c in O.CLASSES},
           "site_indicator_share_on": {c: round(float(f[c].mean()), 3) for c in f.columns if c.startswith("out_sub_")}}
    (PARQUET.parent / "results").mkdir(exist_ok=True)
    (PARQUET.parent / "results" / "features_outages_coverage.json").write_text(json.dumps(out, indent=1, default=str))
    assert len(top) == O.N_TOP and f["out_n_active"].notna().mean() > 0.99


def test_holdout_panel_is_refused(store, monkeypatch):
    """A panel row delivered on or after 2024-01-01 raises while the study is locked."""
    import lock
    monkeypatch.delenv(lock.ENV_FLAG, raising=False)
    with pytest.raises(lock.HoldoutLocked):
        O.features(panel([dt.date(2023, 12, 31)], zones=["WEST"]), store, top=[])


def test_frozen_site_list_gives_distinct_columns():
    assert O.TOP_SITES is not None and len(O.TOP_SITES) == O.N_TOP
    assert len(set(O.columns(O.TOP_SITES))) == 5 + O.N_TOP + 4
