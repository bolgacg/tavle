"""Idea D features: scheduled transmission outages (NYISO outSched snapshots).

    features(panel_index) -> DataFrame aligned to the panel index

Panel row = (bid_date D, zone, delivery_hour h of D+1), decided at 05:00 New York time on D.
The snapshot used is timing.bid_inputs(D)["outages_latest"]: the newest snapshot public at 05:00
on D, normally the file named D (written about 09:40 on D-1). It passes timing.assert_no_lookahead
before use. The features are system-wide (the same for every zone in an hour); the model pairs
them with the zone.

What a snapshot holds: snapshot D lists the scheduled outages ACTIVE ON DAY D (each row starts on
or before D and ends on or after D; checked on the January 2020, January 2023 and August 2026
files). An outage that starts on D+1 is therefore not visible at 05:00 on D, so "outages starting
on D+1" cannot be counted; the closest public signal is out_n_started_d (new on D).

    out_n_active                 outages active during the hour [h, h+1)
    out_n_345kv_up / _230kv / _138kv / _other_kv   the same by voltage class parsed from the
                                 equipment name (345 kV and above: 345, 500, 765; other: 115, 69,
                                 interface schedule limits "SCH-...", unparsed)
    out_sub_<site>               1 if any active outage in the hour touches that substation (either
                                 end of a line) or interface; the 20 sites with the most outage
                                 starts in the snapshots serving 2020 to 2023, see top_sites, TOP_SITES
    out_n_ending_d1              outages whose scheduled end falls on D+1
    out_n_started_d              outages whose scheduled start falls on D (new in this snapshot)
    out_n_in_snapshot            rows in the snapshot (after exact duplicates are dropped)
    out_snapshot_age_days        D minus the snapshot date (0 normally; 1 when snapshot D is late
                                 or missing, e.g. 25 Sep 2026)
No price is read here. The default store reads through the holdout lock (lock.py), and features()
refuses panel rows delivered on or after 2024-01-01 while the study is locked.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock  # noqa: E402  (holdout lock: no row for a delivery date on or after 2024-01-01 while locked)
import timing as T  # noqa: E402
from common import TZ, decision_time  # noqa: E402
from features_weather import locked_store, panel_frame  # noqa: E402  (shared panel and store helpers)

# Snapshot S serves delivery day S+1, so build-year delivery days 2020-01-02..2023-12-31 use snapshots
# up to 2023-12-30: the same window whether or not the holdout lock is in place.
BUILD_FIRST, BUILD_LAST_SNAPSHOT = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-30")
N_TOP = 20
KNOWN_KV = {765, 500, 345, 230, 220, 138, 115, 69, 46, 34}
CLASSES = ["345kv_up", "230kv", "138kv", "other_kv"]
_KV_TOKEN = re.compile(r"^(\d{2,3})(KV|[A-Z])?$")
_KV_ANY = re.compile(r"(\d{2,3})\s*KV")

# The N_TOP sites, computed by top_sites on gene from the real outages table on 6 Oct 2026 (snapshots
# 2020-01-01 to 2023-12-30, 184,164 rows, 868 distinct sites) and frozen here, so the held-out run
# uses exactly this list. test_real_tables_build_years recomputes it and must match.
TOP_SITES: list[str] | None = [
    "ASTORIAE", "LEEDS", "N.SCTLND", "FRASER", "GILBOA", "OAKDALE", "ROTTRDAM", "EDIC", "ASTORIAG", "MARCY",
    "ASTORIAW", "ASTORIA3", "ASTORIA4", "ASTORIA5", "NIAGARA", "FARRAGUT", "HANCOCK", "HAZEL",
    "SCH-NE-NYISO", "SCH-NYISO-NE"]


# ------------------------------------------------------------------------------- name parsing
def voltage_kv(name: str) -> int | None:
    """First token that is a known transmission voltage, e.g. FARRAGUT_345KV_7W -> 345,
    HUDSONP_-FARRAGUT_345_B3402 -> 345, E13THSTA_345_138_BK 15 -> 345 (high side first),
    STLAWRNC_230B_230D_PS 34 -> 230, KERHONKS_69_KV_MK-234-FK -> 69."""
    up = str(name).upper()
    for tok in re.split(r"[_\s]+", up):
        m = _KV_TOKEN.match(tok)
        if m and int(m.group(1)) in KNOWN_KV:
            return int(m.group(1))
    m = _KV_ANY.search(up)
    return int(m.group(1)) if m else None


def voltage_class(name: str) -> str:
    kv = voltage_kv(name)
    if kv is not None and kv >= 345:
        return "345kv_up"
    return {230: "230kv", 138: "138kv"}.get(kv, "other_kv")


def sites(name: str) -> list[str]:
    """Substations an equipment name touches. Names use 8-character padded station fields:
    'BUCHAN_S_345KV_3' -> [BUCHAN_S]; lines 'HUDSONP_-FARRAGUT_345_B3402' -> [HUDSONP, FARRAGUT];
    interface limits 'SCH-HQ-NYISO_LIMIT_1200' -> [SCH-HQ-NYISO]; otherwise the first '_' token."""
    up = str(name).upper().strip()
    if up.startswith("SCH-"):
        return [up.split("_LIMIT")[0] if "_LIMIT" in up else up.split("_")[0]]
    if len(up) > 8 and up[8] == "-":
        return [up[:8].strip("_"), up[9:17].strip("_")]
    if len(up) > 8 and up[8] == "_":
        return [up[:8].strip("_")]
    return [up.split("_")[0]]


def slug(site: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", site.lower()).strip("_")


# ------------------------------------------------------------------------------------ inputs
def default_store(end: dt.date | None = None) -> T.Store:
    """The outages table through the holdout lock (snapshots dated before read_end - 1 day); other
    sources empty."""
    return locked_store({"outages": "snapshot_date"}, end)


def snapshot_for(bid_day: dt.date, store: T.Store) -> pd.DataFrame:
    """The snapshot used for bid day D, checked against the 05:00 deadline (raises LookaheadError)."""
    og = T.bid_inputs(bid_day, store)["outages_latest"]
    T.assert_no_lookahead({"outages_latest": og}, decision_time(bid_day))
    return og.drop_duplicates(["snapshot_date", "equipment", "sched_out", "sched_in"])


def top_sites(outages: pd.DataFrame, n: int = N_TOP) -> list[str]:
    """The n sites with the most distinct outage starts (site, local start date), counted over the
    snapshots that serve build-year delivery days (dated 2020-01-01 to 2023-12-30) and starts on or
    after 2020-01-01. Counting starts, not days active, keeps multi-year outages (one start, always
    on) from filling the list with constant indicators. Ties are broken alphabetically. Rows from
    later snapshots are never read."""
    snap = pd.to_datetime(outages["snapshot_date"])
    og = outages[(snap >= BUILD_FIRST) & (snap <= BUILD_LAST_SNAPSHOT)]
    start = og["sched_out"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    og = og[start >= BUILD_FIRST]
    ev = pd.DataFrame({"equipment": og["equipment"].to_numpy(),
                       "start": og["sched_out"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize().to_numpy()})
    ev = ev.drop_duplicates()
    ev["site"] = ev["equipment"].map(sites)
    ev = ev.explode("site")[["site", "start"]].drop_duplicates()
    counts = ev.groupby("site").size().reset_index(name="n")
    counts = counts.sort_values(["n", "site"], ascending=[False, True])
    return counts["site"].head(n).tolist()


# ---------------------------------------------------------------------------------- features
def columns(top: list[str]) -> list[str]:
    slugs = [slug(s) for s in top]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"two sites share a column name: {slugs}")
    return (["out_n_active"] + [f"out_n_{c}" for c in CLASSES] + [f"out_sub_{slug(s)}" for s in top]
            + ["out_n_ending_d1", "out_n_started_d", "out_n_in_snapshot", "out_snapshot_age_days"])


def day_features(bid_day: dt.date, snap: pd.DataFrame, top: list[str]) -> pd.DataFrame:
    """One row per local hour of D+1 (23, 24 or 25), keyed by hour_utc (naive UTC)."""
    d1 = pd.Timestamp(bid_day + dt.timedelta(days=1), tz=TZ)
    hours = pd.date_range(d1, pd.Timestamp(bid_day + dt.timedelta(days=2), tz=TZ), freq="h", inclusive="left")
    start = hours.tz_convert("UTC").tz_localize(None).to_numpy()
    end = start + np.timedelta64(1, "h")
    n = len(snap)
    so = snap["sched_out"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    si = snap["sched_in"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    # active[i, j]: outage j overlaps hour i; an unparsed end point counts as open-ended.
    active = ((np.isnat(so)[None, :] | (so[None, :] < end[:, None]))
              & (np.isnat(si)[None, :] | (si[None, :] > start[:, None]))) if n else np.zeros((len(hours), 0), bool)
    eq = snap["equipment"].astype(str).tolist()
    cls = np.array([voltage_class(e) for e in eq])
    site_sets = [set(sites(e)) for e in eq]
    out = {"hour_utc": start.astype("datetime64[ns]"), "out_n_active": active.sum(axis=1)}
    for c in CLASSES:
        out[f"out_n_{c}"] = active[:, cls == c].sum(axis=1) if n else np.zeros(len(hours), int)
    for s in top:
        touch = np.array([s in ss for ss in site_sets], dtype=bool)
        out[f"out_sub_{slug(s)}"] = active[:, touch].any(axis=1).astype(int) if n else np.zeros(len(hours), int)
    local_in = snap["sched_in"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    local_out = snap["sched_out"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    out["out_n_ending_d1"] = int((local_in == pd.Timestamp(bid_day + dt.timedelta(days=1))).sum())
    out["out_n_started_d"] = int((local_out == pd.Timestamp(bid_day)).sum())
    out["out_n_in_snapshot"] = n
    out["out_snapshot_age_days"] = ((pd.Timestamp(bid_day) - pd.Timestamp(snap["snapshot_date"].iloc[0])).days
                                    if n else np.nan)
    f = pd.DataFrame(out)
    f["bid_date"] = pd.Timestamp(bid_day)
    return f


def features(panel_index: pd.MultiIndex, store: T.Store | None = None,
             top: list[str] | None = None) -> pd.DataFrame:
    """Idea D feature columns for every panel row, index = panel_index, same order. A bid day with
    no public snapshot in the 30-day lookback gets NaN for every column. `top` overrides the frozen
    TOP_SITES (tests on synthetic tables)."""
    p = panel_frame(panel_index)
    lock.assert_build_only(p["delivery_hour"])
    store = store or default_store()
    if top is None:
        top = TOP_SITES if TOP_SITES is not None else top_sites(store.table("outages"))
    cols = columns(top)
    parts = []
    for D in sorted(set(p["bid_date"].dt.date)):
        snap = snapshot_for(D, store)
        if len(snap):
            parts.append(day_features(D, snap, top))
    daily = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["bid_date", "hour_utc"] + cols))
    daily["bid_date"] = daily["bid_date"].astype("datetime64[ns]")
    daily["hour_utc"] = daily["hour_utc"].astype("datetime64[ns]")
    out = p.merge(daily, on=["bid_date", "hour_utc"], how="left", validate="many_to_one")
    res = out[cols].astype(float)
    res.index = panel_index
    return res
