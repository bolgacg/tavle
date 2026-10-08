"""v2 feature matrices, built once so models never rebuild features (delivery 2010-01-01 to 2023-12-31).

    cd ~/nyiso-us/pipeline_v2 && ../.venv/bin/python features_v2.py all
        steps: panel (v1 panel per year) | border | wxr | day (v1 deep_day matrix per year) | assemble
        wxrj (idea V16, separate): parts/wxr_joined_hourly.parquet from weather_gefs_joined

Outputs in ~/nyiso-us/parquet_v2/features/:
  panel_2010_2023.parquet   v1 schema (panel.KEY_COLS + LABEL_COLS + BASE_FEATURES), plus
                            bord__* border-proxy features and wxr_* reforecast weather (zone's primary point)
  day_2010_2023.parquet     v1 deep_day schema (bid_date, delivery_date, feature_columns, y__ labels), plus
                            bord__<n>__dJ__<stat> and wxr__<point>__hHH
  parts/                    per-year pieces and the border / weather blocks (assemble joins them)

Every input is filtered on its own published_at <= 05:00 on bid day D (v1 timing.features_available_at
for the v1 blocks; the same filter, asserted, for the border and reforecast blocks). The panel and day
matrix reuse v1 model code unchanged (panel.build_panel, deep_day.build_day_features) on a
panel.LockedStore rooted at parquet_v2; the lock refuses any delivery date on or after 2024-01-01.

Run mode (common.py, US_RUN_MODE): unset = the build above. dryrun / heldout build the same files under
$US_RUN_DIR/parquet_v2/features, named panel_2010_<last year> / day_2010_<last year> (dryrun 2023,
heldout 2026, delivery to 2026-09-30); the store ends at LAST_DAY + 1 day (2024-01-01 in build mode).
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402  (v2 shim first)

MODEL = next(p for p in (HERE.parent.parent / "us" / "model", Path.home() / "nyiso-us" / "model") if (p / "panel.py").exists())
if str(MODEL) not in sys.path:
    sys.path.append(str(MODEL))                     # after v2: `import fees` resolves to the v2 fees shim

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

import fees  # noqa: E402
import lock  # noqa: E402
import panel as P  # noqa: E402
import deep_day as DY  # noqa: E402
from points import POINTS  # noqa: E402

assert P.PARQUET == C.PARQUET and Path(fees.__file__).parent == HERE, "v2 shims not in effect"
FEAT = C.PARQUET / "features"
PARTS = FEAT / "parts"
YEARS = range(C.FIRST_DAY.year, C.LAST_DAY.year + 1)
END = pd.Timestamp(C.LAST_DAY + dt.timedelta(days=1), tz=C.TZ)
STORE_END = C.LAST_DAY + dt.timedelta(days=1)       # 2024-01-01 in build mode, the LockedStore default
SPAN = f"{C.FIRST_DAY.year}_{C.LAST_DAY.year}"       # 2010_2023 in build mode
PANEL_FILE = FEAT / f"panel_{SPAN}.parquet"
DAY_FILE = FEAT / f"day_{SPAN}.parquet"
BORDERS = {"H Q": "HQ", "NPX": "NPX", "O H": "OH", "PJM": "PJM"}
BORDER_DAY_STATS = DY.PX_STATS
WORKERS = 4


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def bid_day_range(y: int) -> tuple[dt.date, dt.date]:
    return dt.date(y, 1, 1), min(dt.date(y, 12, 31), C.LAST_DAY)


# ------------------------------------------------------------------------------------ panel (v1 code)
def panel_year(y: int):
    out = PARTS / f"panel_base_{y}.parquet"
    if out.exists():
        return
    t0 = time.time()
    s, e = bid_day_range(y)
    p = P.build_panel(s, e, store=P.LockedStore(end=STORE_END), workers=WORKERS)
    P.save_panel(p, out)
    log(f"panel {y}: {len(p)} rows {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------ border block
def border_rows() -> pd.DataFrame:
    files = sorted((C.PARQUET / "prices_gen").glob("*.parquet"))
    d = ds.dataset([str(f) for f in files], format="parquet")
    cols = ["delivery_hour", "name", "da_lbmp", "da_congestion", "da_published_at", "rt_lbmp", "rt_congestion",
            "rt_published_at"]
    t = d.to_table(filter=(ds.field("point_type") == "external"), columns=cols).to_pandas()
    t = t[t["delivery_hour"] < END]
    lock.assert_build_only(t["delivery_hour"])
    t["ld"], t["h"] = P._local(t["delivery_hour"])
    return t.sort_values("delivery_hour").reset_index(drop=True)


def _border_one(args):
    D, b = args
    t = C.decision_time(D)
    d0 = pd.Timestamp(D)
    da = b[(b["ld"] >= d0 - pd.Timedelta(days=7)) & (b["ld"] <= d0) & (b["da_published_at"] <= t)]
    rt = b[(b["ld"] >= d0 - pd.Timedelta(days=7)) & (b["ld"] <= d0) & (b["rt_published_at"] <= t)]
    assert (da["da_published_at"] <= t).all() and (rt["rt_published_at"] <= t).all()
    g = da[["delivery_hour", "name", "ld", "h", "da_lbmp"]].merge(
        rt[["delivery_hour", "name", "rt_lbmp"]], on=["delivery_hour", "name"], how="inner")
    g["gap"] = g["rt_lbmp"] - g["da_lbmp"]
    hourly = pd.DataFrame(index=pd.Index(range(24), name="hour"))
    day = {"bid_date": d0}
    for n, k in BORDERS.items():
        a0 = da[(da["name"] == n) & (da["ld"] == d0)].groupby("h")
        hourly[f"bord__da_d0_h_{k}"] = a0["da_lbmp"].mean()
        hourly[f"bord__dacong_d0_h_{k}"] = a0["da_congestion"].mean()
        gn = g[g["name"] == n]
        hourly[f"bord__gap_dm1_h_{k}"] = gn[gn["ld"] == d0 - pd.Timedelta(days=1)].groupby("h")["gap"].mean()
        hourly[f"bord__gap_7d_h_{k}"] = gn[gn["ld"] < d0].groupby("h")["gap"].mean()
        dan, rtn = da[da["name"] == n], rt[rt["name"] == n]
        for j in range(DY.N_DAYS_HIST):
            ld = d0 - pd.Timedelta(days=j)
            a, r, gg = dan[dan["ld"] == ld], rtn[rtn["ld"] == ld], gn[gn["ld"] == ld]
            vals = {"da_mean": a["da_lbmp"].mean(), "da_max": a["da_lbmp"].max(), "rt_mean": r["rt_lbmp"].mean(),
                    "rt_max": r["rt_lbmp"].max(), "gap_mean": gg["gap"].mean(), "gap_max": gg["gap"].max(),
                    "dacong_mean": a["da_congestion"].mean(), "rtcong_mean": r["rt_congestion"].mean()}
            for st in BORDER_DAY_STATS:
                day[f"bord__{k}__d{j}__{st}"] = float(vals[st]) if pd.notna(vals[st]) else np.nan
    hourly = hourly.reset_index()
    hourly.insert(0, "bid_date", d0)
    return hourly, day


def build_border():
    b = border_rows()
    days = [C.FIRST_DAY - dt.timedelta(days=1) + dt.timedelta(days=i)
            for i in range((C.LAST_DAY - C.FIRST_DAY).days + 1)]          # bid days for delivery 2010-01-01..
    hs, ds_ = [], []
    # chunks per worker, each with only the rows it needs
    import multiprocessing as mp
    chunks = [days[i::WORKERS] for i in range(WORKERS)]
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as ex:
        for res in ex.map(_border_chunk, [(c, b) for c in chunks]):
            for h, d in res:
                hs.append(h)
                ds_.append(d)
    H = pd.concat(hs, ignore_index=True).sort_values(["bid_date", "hour"])
    Dd = pd.DataFrame(ds_).sort_values("bid_date")
    H.to_parquet(PARTS / "border_hourly.parquet", index=False)
    Dd.to_parquet(PARTS / "border_day.parquet", index=False)
    log(f"border: hourly {len(H)} rows, day {len(Dd)} rows, {len(b)} border-hours read")


def _border_chunk(args):
    days, b = args
    out = []
    for D in days:
        lo = pd.Timestamp(D) - pd.Timedelta(days=8)
        bb = b[(b["ld"] >= lo) & (b["ld"] <= pd.Timestamp(D))]
        out.append(_border_one((D, bb)))
    return out


# ------------------------------------------------------------------------------------ reforecast block
def build_wxr(table: str = "weather_reforecast", out_name: str = "wxr_hourly.parquet"):
    """Hourly 2 m temperature for every local hour of D+1 per point, from the newest reforecast run public by
    05:00 on D (the 00 UTC run of D), linear in time between its 3-hourly leads. No other run is used.
    table="weather_gefs_joined" (step wxrj, idea V16) builds the same block from the joined GEFS v12 source
    into parts/wxr_joined_hourly.parquet with a `source` column; the main matrices are not changed."""
    f = C.PARQUET / f"{table}.parquet"
    if not f.exists():
        log(f"wxr: {table}.parquet missing, skipped")
        return
    w = pd.read_parquet(f)
    w = w[w["target_hour"] < END]
    lock.assert_build_only(w["target_hour"])
    rows = []
    for init, g in w.groupby("init_utc"):
        D = init.tz_convert("UTC").date()                         # 00 UTC on D
        t = C.decision_time(D)
        if g["published_at"].max() > t:
            raise AssertionError(f"reforecast run {init} published after 05:00 on {D}")
        d1 = pd.Timestamp(D + dt.timedelta(days=1), tz=C.TZ)
        d2 = pd.Timestamp(D + dt.timedelta(days=2), tz=C.TZ)      # local midnight, not d1 + 24 h (DST: 23/25 h)
        # Both interpolation axes in int64 nanoseconds. pandas 3 infers the unit (date_range from a date gives
        # seconds, the parquet column is microseconds), and mixed units made every lead fall outside the
        # bracket, so the whole block was NaN until 7 Oct 2026. Normalise both sides, never trust the default.
        hours = pd.date_range(d1, d2, freq="h", inclusive="left").as_unit("ns")
        hx = hours.tz_convert("UTC").asi8
        for p, gp in g.groupby("point"):
            gp = gp.sort_values("target_hour")
            x = gp["target_hour"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
            v = gp["temperature_2m_c"].to_numpy(float)
            ok = (hx >= x.min()) & (hx <= x.max())
            vals = np.where(ok, np.interp(hx, x, v), np.nan)
            rows.append(pd.DataFrame({"bid_date": pd.Timestamp(D), "point": p, "delivery_hour": hours,
                                      "wxr_temp_c": vals, "wxr_published_at": gp["published_at"].max(),
                                      "source": gp["source"].iloc[0]}))
    W = pd.concat(rows, ignore_index=True)
    W["bid_date"] = W["bid_date"].dt.as_unit("ns")
    W["delivery_hour"] = W["delivery_hour"].dt.as_unit("ns")
    if table != "weather_gefs_joined":
        W = W.drop(columns="source")                               # main block keeps its registered schema
    filled = W.loc[W["bid_date"].dt.year <= 2019, "wxr_temp_c"].notna().mean()
    if not filled >= 0.9:
        raise AssertionError(f"wxr: only {filled:.1%} of 2010-2019 point-hours filled (unit mismatch?)")
    if table == "weather_gefs_joined":
        live = W["bid_date"] >= pd.Timestamp(C.GEFS_LIVE_FIRST)
        gap = (W["bid_date"] >= pd.Timestamp("2020-01-01")) & ~live
        if gap.any():
            raise AssertionError(f"wxr joined: {int(gap.sum())} point-hours in the 2020-01-01..09-22 gap")
        fl = W.loc[live, "wxr_temp_c"].notna().mean() if live.any() else 0.0
        if not fl >= 0.9:
            raise AssertionError(f"wxr joined: only {fl:.1%} of 2020-09-23..{C.LAST_DAY} point-hours filled")
    tmp = PARTS / (out_name + ".tmp")
    W.to_parquet(tmp, index=False)
    os.replace(tmp, PARTS / out_name)
    log(f"wxr ({table}): {len(W)} point-hours, {W['bid_date'].nunique()} bid days")


# ------------------------------------------------------------------------------------ day matrix (v1 code)
def _day_year(y: int):
    out = PARTS / f"day_base_{y}.parquet"
    if out.exists():
        return y, 0
    t0 = time.time()
    s, e = bid_day_range(y)
    first = max(s, C.FIRST_DAY) - dt.timedelta(days=1) if y == C.FIRST_DAY.year else s - dt.timedelta(days=1)
    days = [first + dt.timedelta(days=i) for i in range((e - first).days)]   # delivery days s..e
    df = DY.build_day_features(days, P.LockedStore(end=STORE_END))
    df.to_parquet(out, index=False)
    return y, time.time() - t0


def build_day():
    import multiprocessing as mp
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as ex:
        for y, s in ex.map(_day_year, YEARS):
            log(f"day {y}: {s:.0f}s")


# ------------------------------------------------------------------------------------ assemble
def assemble():
    base = pd.concat([pd.read_parquet(PARTS / f"panel_base_{y}.parquet") for y in YEARS], ignore_index=True)
    bh = pd.read_parquet(PARTS / "border_hourly.parquet")
    base = base.merge(bh, on=["bid_date", "hour"], how="left", validate="many_to_one")
    for k in BORDERS.values():
        base[f"bord__spread_da_d0_h_{k}"] = base["da_d0_h"] - base[f"bord__da_d0_h_{k}"]
        base[f"bord__spread_gap_dm1_h_{k}"] = base["gap_dm1_h"] - base[f"bord__gap_dm1_h_{k}"]
    wxr_cols = []
    if (PARTS / "wxr_hourly.parquet").exists():
        W = pd.read_parquet(PARTS / "wxr_hourly.parquet")
        prim = {z: p for p, (z, _la, _lo, pr) in POINTS.items() if pr}
        W = W[W["point"].isin(prim.values())].assign(zone=lambda x: x["point"].map({v: k for k, v in prim.items()}))
        for k in ("bid_date", "delivery_hour"):                      # join keys in the panel's own unit
            W[k] = W[k].dt.as_unit(base[k].dt.unit)
        W["wxr_temp_f"] = W["wxr_temp_c"] * 9 / 5 + 32
        dly = W.groupby(["bid_date", "zone"])["wxr_temp_f"].agg(
            wxr_d1_max_f="max", wxr_d1_min_f="min", wxr_d1_mean_f="mean",
            wxr_d1_hdh65=lambda s: 24 * np.clip(65 - s, 0, None).mean(),
            wxr_d1_cdh65=lambda s: 24 * np.clip(s - 65, 0, None).mean()).reset_index()
        n = W.groupby(["bid_date", "zone"])["wxr_temp_f"].count().reset_index(name="_n")
        dly = dly.merge(n, on=["bid_date", "zone"])
        dly.loc[dly["_n"] < 20, [c for c in dly.columns if c.startswith("wxr_")]] = np.nan
        base = base.merge(W[["bid_date", "zone", "delivery_hour", "wxr_temp_f"]], on=["bid_date", "zone", "delivery_hour"],
                          how="left", validate="one_to_one")
        base = base.merge(dly.drop(columns="_n"), on=["bid_date", "zone"], how="left", validate="many_to_one")
        wxr_cols = ["wxr_temp_f", "wxr_d1_max_f", "wxr_d1_min_f", "wxr_d1_mean_f", "wxr_d1_hdh65", "wxr_d1_cdh65"]
    lock.assert_build_only(base["delivery_hour"])
    base = base.sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
    _atomic(base, PANEL_FILE)
    log(f"{PANEL_FILE.stem}: {len(base)} rows, {base.shape[1]} cols ({len(wxr_cols)} wxr)")

    day = pd.concat([pd.read_parquet(PARTS / f"day_base_{y}.parquet") for y in YEARS], ignore_index=True)
    bd = pd.read_parquet(PARTS / "border_day.parquet")
    day = day.merge(bd, on="bid_date", how="left", validate="one_to_one")
    if (PARTS / "wxr_hourly.parquet").exists():
        W = pd.read_parquet(PARTS / "wxr_hourly.parquet")
        W["bid_date"] = W["bid_date"].dt.as_unit(day["bid_date"].dt.unit)
        W["h"] = W["delivery_hour"].dt.tz_convert(C.TZ).dt.hour
        piv = W.groupby(["bid_date", "point", "h"])["wxr_temp_c"].mean().unstack(["point", "h"])
        piv.columns = [f"wxr__{DY._slug(p)}__h{h:02d}" for p, h in piv.columns]
        cols = [f"wxr__{DY._slug(p)}__h{h:02d}" for p in sorted(POINTS) for h in range(24)]
        day = day.merge(piv.reindex(columns=cols).reset_index(), on="bid_date", how="left", validate="one_to_one")
    lock.assert_build_only(day["delivery_date"])
    day = day.sort_values("bid_date").reset_index(drop=True)
    _atomic(day, DAY_FILE)
    log(f"{DAY_FILE.stem}: {len(day)} rows, {day.shape[1]} cols")


def _atomic(df: pd.DataFrame, path: Path):
    """Write to a temp file in the same folder, then os.replace, so a reader never sees a half-written file."""
    lock.assert_build_only(df["delivery_hour"] if "delivery_hour" in df else df["delivery_date"])
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main(what: str):
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if what in ("panel", "all"):
        for y in YEARS:
            panel_year(y)
    if what in ("border", "all"):
        build_border()
    if what in ("wxr", "all"):
        build_wxr()
    if what == "wxrj":                              # V16 joined GEFS v12 block (not part of "all")
        build_wxr("weather_gefs_joined", "wxr_joined_hourly.parquet")
    if what in ("day", "all"):
        build_day()
    if what in ("assemble", "all"):
        assemble()
    log(f"done {what} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
