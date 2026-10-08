"""V15 shared pieces (OBJECTIVES.md, V15 addendum of 7 Oct, binding): the conditional gap model behind the V2 limit
rule, the two-sided rule, and the weather block of V15c built from the pipeline's day matrix.

Reused code, unchanged: rolling.py (windows, costs, holdout, run_rows, write_positions), strategies_v2.py (v2_limits,
v2_pos, Pairs, out), gbm_rolling.DA_MULT (the 9 day-ahead levels), ideas/learner.py (the fixed LightGBM of V4 to
V9), ideas/run_ideas.py (the V4, V5, V8 feature sets).

Conditional model (V15a): the V2 construction on another feature set. In training the realised day-ahead price of
the delivery hour is an input (cond_da, cond_da_ratio = DA / da_d0_h); at bid time the model is evaluated on the 9
grid levels dag_k = da_d0_h x DA_MULT[k] only (da_d0_h = the same hour's day-ahead on D, public at 05:00 on D), and
strategies_v2.v2_limits picks side and limit from that grid. The test quarter's realised day-ahead and gap are never
read by the model (v15_test.py poisons them).

Weather (V15c): GEFS reforecast to 2019 through V4's path (ideas/v4_reforecast.py, see the note above V4_MAP: the
day matrix's own reforecast block is empty), GFS archive from 25 March 2021 from the day matrix
(parquet_v2/features/day_2010_2023.parquet, wx__<point>__hHH in degrees C for the 24 wall hours of D+1, filtered at
build time on published_at <= 05:00 on D by v1 timing.gfs_rule; pipeline_v2 README, test_timing_v2.py). Both at the
zone's primary point (v1 points.py); the reforecast where the row has it, else GFS, else NaN (1 January 2020 to
24 March 2021). Never observed weather.

HOLDOUT RULE (absolute): nothing on or after 2024-01-01 is read (rolling.read_pre2024, assert_pre2024).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDEAS = Path(os.environ.get("V2_IDEAS_DIR", str(HERE.parent / "ideas")))
for _p in (str(IDEAS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("US_LGBM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rolling as R  # noqa: E402
import strategies_v2 as SV  # noqa: E402
import learner as L  # noqa: E402

DA_MULT = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)      # gbm_rolling.DA_MULT (asserted below on gene)
I_D0 = DA_MULT.index(1.0)                                       # the grid level equal to today's day-ahead
V15DIR = R.RESULTS / "v15"
THREADS = int(os.environ.get("US_LGBM_THREADS", "2"))
REFORECAST_END = pd.Timestamp("2020-01-01")                     # V4 rows end here (run_ideas.py)


class LookaheadError(AssertionError):
    pass


def check_da_mult():
    try:
        import gbm_rolling
        assert tuple(gbm_rolling.DA_MULT) == DA_MULT, "DA_MULT differs from gbm_rolling"
    except ImportError:
        pass


# ============================================================================ conditional model (V2 rule)
def cond_X(df: pd.DataFrame, cols: list[str], da) -> pd.DataFrame:
    Z = df[cols].astype(float).copy()
    Z["cond_da"] = np.asarray(da, float)
    d0 = df["da_d0_h"].astype(float)
    Z["cond_da_ratio"] = np.asarray(da, float) / d0.where(d0.abs() > 1, np.nan).to_numpy()
    return Z


def make_cond_fp(cols: list[str], threads: int = THREADS, n_rounds: int = L.N_ROUNDS, params=None):
    """fit_predict for rolling.run_rows: the conditional regression of the gap on cols + (cond_da, cond_da_ratio),
    the ideas learner's fixed settings; returns dag_k, cond_k for the 9 levels."""
    cols = list(cols)
    L.check_features(cols)
    for bad in ("gap", "y_da_lbmp", "y_rt_lbmp", "cond_da", "cond_da_ratio"):
        if bad in cols:
            raise LookaheadError(f"{bad} may not be a bid-time feature")
    prm = dict(L.PARAMS, num_threads=threads, **(params or {}))

    def fp(tr: pd.DataFrame, te: pd.DataFrame, q=None) -> pd.DataFrame:
        import lightgbm as lgb
        if "delivery_hour" in tr:
            R.assert_pre2024(tr["delivery_hour"], "v15 cond train")
            R.assert_pre2024(te["delivery_hour"], "v15 cond test")
        y = tr["gap"].to_numpy(float)
        ok = np.isfinite(y)
        t = tr[ok]
        m = lgb.train(prm, lgb.Dataset(cond_X(t, cols, t["y_da_lbmp"].to_numpy(float)),
                                       np.clip(y[ok], -L.CLIP, L.CLIP)), n_rounds)
        out = pd.DataFrame(index=te.index)
        tx = te[list(dict.fromkeys(cols + ["da_d0_h"]))]                     # bid-time columns only from here on
        base = tx["da_d0_h"].to_numpy(float)
        for k, mlt in enumerate(DA_MULT):
            lvl = base * mlt
            out[f"dag_{k}"] = lvl
            out[f"cond_{k}"] = m.predict(cond_X(tx, cols, lvl))
        return out

    return fp


def limit_bids(g: pd.DataFrame):
    """(side, limit) of the V2 rule from the grid only (strategies_v2.v2_limits)."""
    side, limit, _ = SV.v2_limits(g)
    return side, limit


# ============================================================================ two-sided rule (v1 B/D, ideas_common)
def two_sided(rows: pd.DataFrame, pred) -> np.ndarray:
    sup, lod = R.row_costs(pd.to_datetime(rows["delivery_date"]).dt.year)
    p = np.asarray(pred, float)
    pos = np.where(p <= -sup, -1.0, np.where(p >= lod, 1.0, 0.0))
    return np.where(np.isfinite(p), pos, 0.0)


# ============================================================================ weather block (V15c)
W_COLS = ["w15_temp_f", "w15_d1_max_f", "w15_d1_min_f", "w15_d1_mean_f", "w15_d1_hdh65", "w15_d1_cdh65", "w15_src",
          "w15_temp_f_wow", "w15_d1_max_f_wow", "w15_d1_min_f_wow", "w15_d1_mean_f_wow"]
MIN_HOURS = 20
BASE_F = 65.0


def zone_points() -> dict:
    """zone -> slug of its primary point (v1 points.py, deep_day._slug)."""
    import re
    from points import POINTS
    slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "", s)   # noqa: E731  (deep_day._slug)
    zp = {v[0]: slug(p) for p, v in POINTS.items() if v[3]}
    assert sorted(zp) == sorted(R.ZONES), f"primary points do not cover the zones: {sorted(zp)}"
    return zp


def day_weather_columns(zp: dict) -> list[str]:
    pts = sorted(set(zp.values()))
    return [f"{pre}__{pt}__h{h:02d}" for pre in ("wxr", "wx") for pt in pts for h in range(24)]


def read_day_weather(path: Path = R.DAYFEATS) -> pd.DataFrame:
    import pyarrow.dataset as ds
    zp = zone_points()
    cols = day_weather_columns(zp)
    sch = set(ds.dataset(str(path), format="parquet").schema.names)
    miss = [c for c in cols if c not in sch]
    if miss:
        raise RuntimeError(f"day matrix lacks {len(miss)} weather columns, e.g. {miss[:3]}")
    return R.read_pre2024(Path(path), "delivery_date", ["bid_date", "delivery_date"] + cols)


def weather_from_day(day: pd.DataFrame, keys: pd.DataFrame, zp: dict | None = None,
                     use_reforecast: bool = True) -> pd.DataFrame:
    """keys: delivery_date (naive day of delivery), zone, whour (local wall hour) per panel row. Every value comes
    from the day-matrix row of the SAME delivery date (built from forecasts public by 05:00 on bid day = delivery
    date - 1), or from delivery date - 7 for the week-on-week columns. Returns W_COLS aligned to keys.index."""
    zp = zp or zone_points()
    dd = pd.to_datetime(day["delivery_date"]).dt.normalize()
    bd = pd.to_datetime(day["bid_date"]).dt.normalize()
    if not (dd == bd + pd.Timedelta(days=1)).all():
        raise LookaheadError("day matrix: delivery_date != bid_date + 1")
    if dd.duplicated().any():
        raise ValueError("day matrix: duplicate delivery dates")
    R.assert_pre2024(dd, "v15 weather day matrix")
    hourly, daily = [], []
    for z, pt in zp.items():
        Rm = day[[f"wxr__{pt}__h{h:02d}" for h in range(24)]].to_numpy(float)
        if not use_reforecast:
            Rm = np.full_like(Rm, np.nan)
        Gm = day[[f"wx__{pt}__h{h:02d}" for h in range(24)]].to_numpy(float)
        use_r = np.isfinite(Rm).any(1)
        use_g = ~use_r & np.isfinite(Gm).any(1)
        T = np.where(use_r[:, None], Rm, np.where(use_g[:, None], Gm, np.nan)) * 9 / 5 + 32
        n = np.isfinite(T).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                st = {"w15_d1_max_f": np.nanmax(T, 1), "w15_d1_min_f": np.nanmin(T, 1),
                      "w15_d1_mean_f": np.nanmean(T, 1),
                      "w15_d1_hdh65": 24 * np.nanmean(np.clip(BASE_F - T, 0, None), 1),
                      "w15_d1_cdh65": 24 * np.nanmean(np.clip(T - BASE_F, 0, None), 1)}
        st = {k: np.where(n >= MIN_HOURS, v, np.nan) for k, v in st.items()}
        daily.append(pd.DataFrame({"delivery_date": dd.to_numpy(), "zone": z,
                                   "w15_src": np.where(use_r, 1.0, np.where(use_g, 2.0, 0.0)), **st}))
        hourly.append(pd.DataFrame({"delivery_date": np.repeat(dd.to_numpy(), 24), "zone": z,
                                    "whour": np.tile(np.arange(24), len(dd)), "w15_temp_f": T.ravel()}))
    H = pd.concat(hourly, ignore_index=True)
    D = pd.concat(daily, ignore_index=True)
    k = keys[["delivery_date", "zone", "whour"]].copy()
    k["delivery_date"] = pd.to_datetime(k["delivery_date"]).dt.normalize().astype("datetime64[ns]")
    for f in (H, D):
        f["delivery_date"] = pd.to_datetime(f["delivery_date"]).astype("datetime64[ns]")
    k["whour"] = k["whour"].astype(int)
    out = k.merge(H, on=["delivery_date", "zone", "whour"], how="left", validate="many_to_one")
    out = out.merge(D, on=["delivery_date", "zone"], how="left", validate="many_to_one")
    lagH = H.assign(delivery_date=H["delivery_date"] + pd.Timedelta(days=7)).rename(columns={"w15_temp_f": "_t7"})
    lagD = D.assign(delivery_date=D["delivery_date"] + pd.Timedelta(days=7))[
        ["delivery_date", "zone", "w15_d1_max_f", "w15_d1_min_f", "w15_d1_mean_f"]].rename(
        columns=lambda c: c + "_7" if c.startswith("w15_") else c)
    out = out.merge(lagH, on=["delivery_date", "zone", "whour"], how="left", validate="many_to_one")
    out = out.merge(lagD, on=["delivery_date", "zone"], how="left", validate="many_to_one")
    out["w15_temp_f_wow"] = out["w15_temp_f"] - out["_t7"]
    for c in ("w15_d1_max_f", "w15_d1_min_f", "w15_d1_mean_f"):
        out[c + "_wow"] = out[c] - out[c + "_7"]
    res = out[W_COLS].astype(float)
    res.index = keys.index
    return res


def panel_keys(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"delivery_date": pd.to_datetime(panel["delivery_date"]).dt.normalize(),
                         "zone": panel["zone"].astype(str).to_numpy(),
                         "whour": panel["delivery_hour"].dt.tz_convert(R.TZ).dt.hour.to_numpy()}, index=panel.index)


# ============================================================================ reforecast through V4's path
# The day matrix's wxr__ block (and the panel's wxr_ columns) are empty on gene (7 Oct 18:30): pipeline_v2
# features_v2.build_wxr compares target_hour as int64 microseconds (pandas 3 reads parquet as datetime64[us]) with
# an hour grid in nanoseconds, so every interpolated value is NaN. V15 therefore takes the reforecast through
# V4's own tested path (ideas/v4_reforecast.py: newest 00 UTC run public by 05:00 on D, zone's primary point,
# hourly interpolation within the run; ideas/test_lookahead.py) and the GFS archive from the day matrix (v1 rule).
V4_MAP = {"wx_temp_f": "w15_temp_f", "wx_d1_max_f": "w15_d1_max_f", "wx_d1_min_f": "w15_d1_min_f",
          "wx_d1_mean_f": "w15_d1_mean_f", "wx_d1_hdh65": "w15_d1_hdh65", "wx_d1_cdh65": "w15_d1_cdh65",
          "wx_temp_f_wow": "w15_temp_f_wow", "wx_d1_max_f_wow": "w15_d1_max_f_wow",
          "wx_d1_min_f_wow": "w15_d1_min_f_wow", "wx_d1_mean_f_wow": "w15_d1_mean_f_wow"}


def v4_cache_path() -> Path:
    return V15DIR / "feat_V4_wx.parquet"


def save_cache(path: Path, panel: pd.DataFrame, extra: pd.DataFrame):
    V15DIR.mkdir(parents=True, exist_ok=True)
    f = pd.concat([panel[["delivery_hour", "zone"]], extra], axis=1)
    R.assert_pre2024(f["delivery_hour"], f"v15 cache {path.name}")
    f.to_parquet(path)


def load_cache(path: Path, panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    c = R.read_pre2024(path, "delivery_hour")
    c["delivery_hour"] = c["delivery_hour"].dt.tz_convert(R.TZ)
    x = panel[["delivery_hour", "zone"]].merge(c, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
    x.index = panel.index
    return x[cols]


def v4_weather(panel: pd.DataFrame) -> pd.DataFrame:
    import v4_reforecast as V4
    cols = list(V4_MAP)
    x = load_cache(v4_cache_path(), panel, cols)
    if x is None:
        idx = pd.MultiIndex.from_frame(panel[["bid_date", "zone", "delivery_hour"]])
        x = V4.features(idx, panel=panel)
        x.index = panel.index
        save_cache(v4_cache_path(), panel, x[[c for c in V4.COLUMNS]])
        x = x[cols]
    return x


def combine_weather(refc: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    """Reforecast (V4 columns, renamed) where the row has it, else the GFS columns; w15_src 1, 2 or 0."""
    r = refc.rename(columns=V4_MAP)
    cols = [c for c in W_COLS if c != "w15_src"]
    has_r = (r["w15_temp_f"].notna() | r["w15_d1_mean_f"].notna()).to_numpy()
    out = gfs[W_COLS].copy()
    out.loc[has_r, cols] = r.loc[has_r, cols].to_numpy()
    out.loc[has_r, "w15_src"] = 1.0
    return out


# ============================================================================ V5 border features, cached
def v5_cache_path() -> Path:
    return V15DIR / "feat_V5_bx.parquet"


def save_v5_cache(panel: pd.DataFrame, extra: pd.DataFrame):
    save_cache(v5_cache_path(), panel, extra)


def v5_features(panel: pd.DataFrame) -> pd.DataFrame:
    """V5's bx_ columns (ideas/v5_border.features), from the V15a cache when present, else built."""
    import v5_border as V5
    x = load_cache(v5_cache_path(), panel, V5.COLUMNS)
    if x is not None:
        return x
    idx = pd.MultiIndex.from_frame(panel[["bid_date", "zone", "delivery_hour"]])
    e = V5.features(idx, panel=panel)
    e.index = panel.index
    save_v5_cache(panel, e)
    return e
