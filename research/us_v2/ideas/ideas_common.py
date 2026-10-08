"""Shared engine for the v2 ideas V4 to V9 (us_v2/ideas/). OBJECTIVES.md is binding.

INTERFACE (every ideas/vN_*.py module exposes)
    NAME, ROWS                       idea id and the strategy rows it registers (dict name -> spec)
    features(panel_index, panel=None, store=None) -> DataFrame
        extra feature columns only, index = panel_index, same order. panel_index levels are found by
        name (bid_date, zone, delivery_hour) else by position 0, 1, 2; delivery_hour is tz-aware
        New York, hour beginning, on bid_date + 1. Inputs are only rows public at 05:00 New York on D.
    rule(panel, pred, costs) -> positions array in {-1, 0, +1} per panel row
        (-1 virtual supply, +1 virtual load). costs = (load_cost, supply_cost) arrays per row.
    optional hooks the framework runner calls when present:
        targets(panel) -> dict of extra label columns (never features)            V9
        sample_weight(train_panel, train_end) -> array                             V7
        train_mask(panel) -> bool array                                             V6
        fit_predict(train, test, base_fit_predict, feature_cols) -> array           V8, V9
            base_fit_predict(train, test, target, feature_cols, sample_weight=None) -> array
            is the framework's own model call, so two-stage ideas reuse the same learner.

HOLDOUT RULE (absolute): nothing on or after 2024-01-01 is read. read_table() filters at the parquet
level and assert_pre_holdout() refuses any panel row delivered on or after that date. Only the held-out run
mode (research/us/model/heldout_mode.py, US_RUN_MODE=heldout, open only through the v1 lock) moves the bound
to 2026-10-01; the dry-run mode keeps 2024-01-01.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"


def _read_end() -> dt.date:
    """Exclusive read bound of the run mode (heldout_mode next to the v1 lock); a build copy without it: 2024-01-01."""
    import sys
    here = Path(__file__).resolve().parent
    for p in (os.environ.get("V1_MODEL_DIR"), here.parents[1] / "us" / "model"):
        if p and (Path(p) / "heldout_mode.py").exists() and str(p) not in sys.path:
            sys.path.insert(1, str(p))
    try:
        import heldout_mode as HM
    except ImportError:
        if os.environ.get("US_RUN_MODE"):
            raise
        return dt.date(2024, 1, 1)
    return HM.read_end()


HOLDOUT = _read_end()
DECISION_HOUR = 5
PARQUET_V2 = Path(os.environ.get("NYISO_V2_PARQUET", str(Path.home() / "nyiso-us" / "parquet_v2")))
if HOLDOUT > dt.date(2024, 1, 1) and PARQUET_V2.resolve() == (Path.home() / "nyiso-us" / "parquet_v2").resolve():
    raise RuntimeError("held-out mode must not read the build tables (set NYISO_V2_PARQUET to the rebuilt tables)")
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.",
         "NORTH", "WEST"]
BORDERS = ["PJM", "NPX", "O H", "H Q"]          # NYISO proxy buses: PJM, New England, Ontario, Quebec
SUPPLY, NONE, LOAD = -1, 0, 1


class HoldoutError(RuntimeError):
    pass


def assert_pre_holdout(delivery) -> None:
    d = pd.to_datetime(pd.Series(np.asarray(delivery)))
    if d.dt.tz is not None:
        d = d.dt.tz_convert(TZ).dt.tz_localize(None)
    if len(d) and (d.dt.normalize() >= pd.Timestamp(HOLDOUT)).any():
        raise HoldoutError(f"a row is delivered on or after {HOLDOUT}: the holdout is closed to v2 design")


def decision_time(bid_day) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(pd.Timestamp(bid_day).date(), dt.time(DECISION_HOUR)), tz=TZ)


def read_table(name: str, time_col: str, columns=None, root: Path | None = None,
               filters=None) -> pd.DataFrame:
    """A parquet_v2 table with rows strictly before 2024-01-01 (New York; HOLDOUT of the run mode) on `time_col`."""
    import pyarrow.parquet as pq
    end = pd.Timestamp(HOLDOUT, tz=TZ)
    path = (root or PARQUET_V2) / f"{name}.parquet"
    schema = pq.read_schema(path)
    t = schema.field(time_col).type
    import pyarrow as pa
    if pa.types.is_timestamp(t):
        bound = end.tz_convert(t.tz).to_pydatetime() if t.tz else end.tz_localize(None).to_pydatetime()
    else:
        bound = HOLDOUT
    f = [(time_col, "<", bound)] + list(filters or [])
    df = pq.read_table(path, columns=columns, filters=f).to_pandas()
    return df


def panel_frame(panel_index: pd.MultiIndex) -> pd.DataFrame:
    """bid_date (naive midnight), zone, delivery_hour (tz New York), wall (naive local), whour (0..23)."""
    names = list(panel_index.names)

    def level(cands, pos):
        for c in cands:
            if c in names:
                return pd.Series(panel_index.get_level_values(names.index(c)))
        return pd.Series(panel_index.get_level_values(pos))

    hour = level(("delivery_hour",), 2)
    if not isinstance(hour.dtype, pd.DatetimeTZDtype):
        raise TypeError("delivery_hour must be tz-aware (New York, hour beginning)")
    p = pd.DataFrame({"bid_date": pd.to_datetime(level(("bid_date",), 0)).dt.tz_localize(None).dt.normalize()
                      if getattr(pd.to_datetime(level(("bid_date",), 0)).dt, "tz", None) is not None
                      else pd.to_datetime(level(("bid_date",), 0)).dt.normalize(),
                      "zone": level(("zone",), 1).astype(str).to_numpy(),
                      "delivery_hour": hour.dt.tz_convert(TZ).to_numpy()})
    p["delivery_hour"] = pd.to_datetime(p["delivery_hour"]).dt.tz_convert(TZ) \
        if isinstance(p["delivery_hour"].dtype, pd.DatetimeTZDtype) else hour.dt.tz_convert(TZ).to_numpy()
    p["bid_date"] = p["bid_date"].astype("datetime64[ns]")
    p["wall"] = p["delivery_hour"].dt.tz_localize(None).astype("datetime64[ns]")
    if not (p["wall"].dt.normalize() == p["bid_date"] + pd.Timedelta(days=1)).all():
        raise ValueError("every delivery_hour must fall on bid_date + 1 (New York time)")
    p["whour"] = p["wall"].dt.hour.astype(int)
    assert_pre_holdout(p["delivery_hour"])
    return p


# ------------------------------------------------------------------ hour grid of public prices
class PriceGrid:
    """Day x wall-hour x node arrays of a price table, each value with the first bid day on whose
    05:00 deadline it was public. The 25-hour fall-back day is averaged per wall hour; the missing
    spring hour stays NaN.

    prices: long frame with delivery_hour (tz), zone (node name), the value columns, and publication
    columns da_published_at and rt_published_at (tz-aware). Values of a node-hour count for bid day D
    only if their publication time <= 05:00 on D."""

    def __init__(self, prices: pd.DataFrame, nodes: list[str], cols: list[str]):
        p = prices[prices["zone"].isin(nodes)].copy()
        wall = p["delivery_hour"].dt.tz_convert(TZ).dt.tz_localize(None)
        p["day"] = wall.dt.normalize().astype("datetime64[ns]")
        p["whour"] = wall.dt.hour
        assert_pre_holdout(p["day"])
        self.days = pd.DatetimeIndex(sorted(p["day"].unique()))
        self.nodes = list(nodes)
        self.cols = list(cols)
        di = self.days.get_indexer(p["day"])
        ni = pd.Index(self.nodes).get_indexer(p["zone"])
        hi = p["whour"].to_numpy()
        shape = (len(self.days), 24, len(self.nodes))
        self.v: dict[str, np.ndarray] = {}
        for c in cols:
            s = np.zeros(shape)
            n = np.zeros(shape)
            x = p[c].to_numpy(float)
            ok = ~np.isnan(x)
            np.add.at(s, (di[ok], hi[ok], ni[ok]), x[ok])
            np.add.at(n, (di[ok], hi[ok], ni[ok]), 1)
            with np.errstate(invalid="ignore"):
                self.v[c] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        # first bid day each value is public: the smallest D with decision(D) >= published_at
        self.usable: dict[str, np.ndarray] = {}
        for side in ("da", "rt"):
            pub = p[f"{side}_published_at"].dt.tz_convert(TZ)
            first = (pub - pd.Timedelta(hours=DECISION_HOUR)).dt.tz_localize(None)
            first = first.dt.ceil("D").astype("datetime64[ns]")   # 05:00 exactly counts on that day
            arr = np.full(shape, np.datetime64("2262-01-01", "ns"))
            # latest publication wins when two rows share a cell (fall-back hour)
            order = np.argsort(first.to_numpy())
            arr[di[order], hi[order], ni[order]] = first.to_numpy()[order]
            self.usable[side] = arr

    def masked(self, col: str, side: str, bid_day: pd.Timestamp) -> np.ndarray:
        return np.where(self.usable[side] <= np.datetime64(bid_day, "ns"), self.v[col], np.nan)

    def day_slice(self, lo: pd.Timestamp, hi: pd.Timestamp) -> slice:
        return slice(self.days.searchsorted(lo), self.days.searchsorted(hi))


def trailing(grid: PriceGrid, col: str, side: str, bid_days, window: int, op=np.nanmean,
             transform=None) -> np.ndarray:
    """For each bid day D: op over delivery days D-window .. D-1 of a public value, per (hour, node).
    Returns array (len(bid_days), 24, nodes). `transform(values_dict_of_masked_arrays)` may combine
    columns (e.g. rt - da) before op; then `col` is ignored."""
    out = np.full((len(bid_days), 24, len(grid.nodes)), np.nan)
    for k, D in enumerate(pd.DatetimeIndex(bid_days)):
        sl = grid.day_slice(D - pd.Timedelta(days=window), D)
        if sl.stop <= sl.start:
            continue
        if transform is None:
            x = np.where(grid.usable[side][sl] <= np.datetime64(D, "ns"), grid.v[col][sl], np.nan)
        else:
            x = transform(grid, sl, np.datetime64(D, "ns"))
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[k] = op(x, axis=0)
    return out


def gap_transform(grid: PriceGrid, sl: slice, D) -> np.ndarray:
    """rt_lbmp - da_lbmp, counted only where both are public at 05:00 on D."""
    ok = (grid.usable["rt"][sl] <= D) & (grid.usable["da"][sl] <= D)
    return np.where(ok, grid.v["rt_lbmp"][sl] - grid.v["da_lbmp"][sl], np.nan)


def same_hour_on_D(grid: PriceGrid, col: str, side: str, bid_days) -> np.ndarray:
    """The value for delivery day D itself (day-ahead of D is posted on D-1), same wall hour."""
    out = np.full((len(bid_days), 24, len(grid.nodes)), np.nan)
    idx = grid.days.get_indexer(pd.DatetimeIndex(bid_days))
    for k, (D, i) in enumerate(zip(pd.DatetimeIndex(bid_days), idx)):
        if i >= 0:
            out[k] = np.where(grid.usable[side][i] <= np.datetime64(D, "ns"), grid.v[col][i], np.nan)
    return out


def to_rows(arr: np.ndarray, bid_days, nodes, p: pd.DataFrame, node_col: str | None = "zone",
            fixed_node: str | None = None) -> np.ndarray:
    """Pick arr[bid_day, whour, node] for each panel row (node = the row's zone, or `fixed_node`)."""
    bi = pd.DatetimeIndex(bid_days).get_indexer(p["bid_date"])
    if fixed_node is not None:
        ni = np.full(len(p), list(nodes).index(fixed_node))
    else:
        ni = pd.Index(list(nodes)).get_indexer(p[node_col])
    ok = (bi >= 0) & (ni >= 0)
    out = np.full(len(p), np.nan)
    out[ok] = arr[bi[ok], p["whour"].to_numpy()[ok], ni[ok]]
    return out


# ------------------------------------------------------------------------------ rules
def two_sided(panel: pd.DataFrame, pred, costs) -> np.ndarray:
    """Supply when pred <= -supply cost, load when pred >= load cost, else none (v1 B/D rule)."""
    load_c, supply_c = (np.asarray(c, float) for c in costs)
    p = np.asarray(pd.Series(pred).reindex(panel.index) if isinstance(pred, pd.Series) else pred, float)
    pos = np.where(p <= -supply_c, SUPPLY, np.where(p >= load_c, LOAD, NONE))
    return np.where(np.isnan(p), NONE, pos).astype(int)
