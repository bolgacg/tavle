"""Rolling-window runner for the New York study, version 2 (us_v2/OBJECTIVES.md, binding).

Design: train on the previous 3 years, test on the next quarter, refit every quarter, scored on the 44
quarters 2013Q1 to 2023Q4. The four quarters of 2012 are WARM-UP quarters (expanding window from
2010-01-01): their out-of-sample predictions exist only so that settings used in 2013 can be chosen on
earlier out-of-sample results; they are never scored.

    Q = quarters()                          [(start, end, label, warmup)] 2012Q1 .. 2023Q4
    train_mask(panel, q_start)              delivery dates in [q_start - 3 years, q_start - 2 days]
    run_rows(name, fit_predict, ...)        one prediction row set over every quarter -> results/v2/preds/<name>.parquet
    choose_setting(rows, settings, pos_fn, q_start)   setting with the best net over the trailing 4 quarters
    write_positions(name, df, meta)         -> results/v2/pos/<name>.parquet + .json, scored by score_v2.py

HOLDOUT (absolute): nothing on or after 2024-01-01 is read, computed or written. Every loader filters at
read time and asserts it (assert_pre2024 here, independent of any unlock switch, plus the v1 lock).
Paths come from the environment so the data agent's file names are a one-line change:
    V2_HOME (~/nyiso-us), V2_PARQUET ($V2_HOME/parquet_v2), V2_PANEL ($V2_PARQUET/features/panel_2010_2023.parquet),
    V2_DAY ($V2_PARQUET/features/day_2010_2023.parquet), V2_RESULTS ($V2_HOME/results/v2), V1_MODEL_DIR.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent


def _v1_model_dir() -> Path:
    for p in (os.environ.get("V1_MODEL_DIR"), HERE.parents[1] / "us" / "model", Path.home() / "nyiso-us" / "model"):
        if p and (Path(p) / "lock.py").exists():
            return Path(p)
    raise FileNotFoundError("v1 model directory not found (set V1_MODEL_DIR)")


V1_MODEL = _v1_model_dir()
V1_PIPELINE = V1_MODEL.parent / "pipeline"
V1_SIDE = V1_MODEL.parent / "side"
for _p in (HERE, V1_MODEL, V1_PIPELINE):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))

import lock  # noqa: E402  (v1 holdout lock)

TZ = "America/New_York"
HOLDOUT = dt.date(2024, 1, 1)
END_TS = pd.Timestamp(HOLDOUT, tz=TZ)
DATA_START = dt.date(2010, 1, 1)
FIRST_SCORED, LAST_SCORED = dt.date(2013, 1, 1), dt.date(2023, 12, 31)
WARMUP_YEAR = 2012
TRAIN_YEARS = 3
TRAIN_GAP_DAYS = 2                  # a refit before quarter start S learns from delivery dates <= S - 2 days
TRAIL_QUARTERS = 4                  # settings are chosen on the previous 4 out-of-sample quarters
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]

V2_HOME = Path(os.environ.get("V2_HOME", str(Path.home() / "nyiso-us")))
PARQUET = Path(os.environ.get("V2_PARQUET", str(V2_HOME / "parquet_v2")))
PANEL = Path(os.environ.get("V2_PANEL", str(PARQUET / "features" / "panel_2010_2023.parquet")))
DAYFEATS = Path(os.environ.get("V2_DAY", str(PARQUET / "features" / "day_2010_2023.parquet")))
RESULTS = Path(os.environ.get("V2_RESULTS", str(V2_HOME / "results" / "v2")))
PREDS, POS = RESULTS / "preds", RESULTS / "pos"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


# ============================================================================ holdout
class HoldoutBreach(AssertionError):
    pass


def assert_pre2024(values, what: str = "") -> None:
    """Raise if any date-like value is on or after 2024-01-01 New York time. No switch turns this off."""
    s = pd.Series(values).dropna() if not isinstance(values, pd.Series) else values.dropna()
    if not len(s):
        return
    mx = pd.Timestamp(s.max())
    late = (mx.tz_convert(TZ) >= END_TS) if mx.tzinfo is not None else (mx >= pd.Timestamp(HOLDOUT))
    if late:
        raise HoldoutBreach(f"HOLDOUT BREACH {what}: {mx}")
    lock.assert_build_only(s)


def read_pre2024(path: Path, timecol: str, columns=None, extra=None) -> pd.DataFrame:
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(timecol).type
    if pa.types.is_timestamp(typ):
        bound = pa.scalar(END_TS if typ.tz else pd.Timestamp(HOLDOUT), type=typ)
    elif pa.types.is_date(typ):
        bound = pa.scalar(HOLDOUT, type=typ)
    else:
        raise TypeError(f"{path}:{timecol} is not a time column")
    flt = ds.field(timecol) < bound
    if extra is not None:
        flt = flt & extra
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    assert_pre2024(df[timecol], f"{path}:{timecol}")
    return df


# ============================================================================ fees (per calendar year, 2010 on)
def fees_module():
    """The extended fee table: the data agent's shim (~/nyiso-us/pipeline_v2/fees.py, laptop us_v2/pipeline/fees.py),
    else fees_v2.py next to this file or in V2_PARQUET, else the v1 fees.py; it must
    cover every year 2010..2023 (raises otherwise, so no year is silently costed at zero)."""
    for p in (V2_HOME / "pipeline_v2", HERE.parent / "pipeline", HERE, PARQUET, V1_MODEL):
        for name in ("fees_v2", "fees"):
            f = Path(p) / f"{name}.py"
            if f.exists():
                spec = importlib.util.spec_from_file_location(f"v2fees_{name}_{abs(hash(str(f)))}", f)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                if all(y in m.RATES and y in m.SUPPLY_RATES for y in range(DATA_START.year, LAST_SCORED.year + 1)):
                    return m
    raise RuntimeError("no fee table covers 2010..2023 (data agent: extend fees.py or write fees_v2.py)")


_FEES = None


def costs() -> tuple[dict, dict]:
    """(supply cost per year, load cost per year), USD per cleared MWh."""
    global _FEES
    if _FEES is None:
        _FEES = fees_module()
    return dict(_FEES.SUPPLY_RATES), dict(_FEES.RATES)


def row_costs(years) -> tuple[np.ndarray, np.ndarray]:
    sup, lod = costs()
    y = pd.Series(np.asarray(years))
    return y.map(sup).to_numpy(float), y.map(lod).to_numpy(float)


# ============================================================================ quarters and windows
def quarters(first_year: int = WARMUP_YEAR, last_year: int = LAST_SCORED.year) -> list[tuple]:
    out = []
    for y in range(first_year, last_year + 1):
        for i, m in enumerate((1, 4, 7, 10)):
            s = dt.date(y, m, 1)
            e = (dt.date(y + (m == 10), (m + 3 - 1) % 12 + 1, 1) - dt.timedelta(days=1))
            out.append((s, e, f"{y}Q{i + 1}", y == WARMUP_YEAR))
    assert out[-1][1] <= LAST_SCORED
    return out


def window(q_start: dt.date, years: int = TRAIN_YEARS) -> tuple[dt.date, dt.date]:
    lo = max(DATA_START, dt.date(q_start.year - years, q_start.month, q_start.day))
    return lo, q_start - dt.timedelta(days=TRAIN_GAP_DAYS)


def _dd(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["delivery_date"]).dt.normalize()


def train_mask(panel: pd.DataFrame, q_start: dt.date, years: int = TRAIN_YEARS, start: dt.date | None = None) -> np.ndarray:
    lo, hi = window(q_start, years)
    if start is not None:
        lo = start
    d = _dd(panel)
    m = (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    if "gap" in panel.columns:
        m &= panel["gap"].notna()
    return m.to_numpy()


def test_mask(panel: pd.DataFrame, q) -> np.ndarray:
    d = _dd(panel)
    return ((d >= pd.Timestamp(q[0])) & (d <= pd.Timestamp(q[1]))).to_numpy()


# ============================================================================ panel
def load_panel(path: Path = PANEL, columns: list[str] | None = None) -> pd.DataFrame:
    p = read_pre2024(Path(path), "delivery_hour", columns)
    p["delivery_hour"] = p["delivery_hour"].dt.tz_convert(TZ)
    if "delivery_date" not in p.columns:
        p["delivery_date"] = p["delivery_hour"].dt.tz_localize(None).dt.normalize()
    p["delivery_date"] = pd.to_datetime(p["delivery_date"]).dt.normalize()
    assert_pre2024(p["delivery_date"], "panel delivery_date")
    p = p.sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
    return p


def base_features() -> list[str]:
    import panel as P1                                   # v1 feature names (imports v1 pipeline common)
    return list(P1.BASE_FEATURES)


# ============================================================================ the runner
def run_rows(name: str, fit_predict, panel: pd.DataFrame | None = None, extra: pd.DataFrame | None = None,
             qs: list | None = None, log=log, resume: bool = True) -> pd.DataFrame:
    """For every quarter: fit_predict(train_df, test_df, q) -> Series or DataFrame aligned to test_df.index.
    Per-quarter parts go to results/v2/preds/<name>_parts/ (a restart skips finished quarters);
    the merged file is results/v2/preds/<name>.parquet with delivery_hour, zone, delivery_date, quarter, warmup."""
    panel = load_panel() if panel is None else panel
    if extra is not None:
        panel = panel.join(extra)
    assert_pre2024(panel["delivery_hour"], f"{name} panel")
    parts = PREDS / f"{name}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    out = []
    for q in (qs or quarters()):
        f = parts / f"{q[2]}.parquet"
        if resume and f.exists():
            out.append(pd.read_parquet(f))
            continue
        t0 = time.time()
        tr = panel[train_mask(panel, q[0])]
        te = panel[test_mask(panel, q)]
        assert tr["delivery_date"].max() <= pd.Timestamp(q[0] - dt.timedelta(days=TRAIN_GAP_DAYS))
        assert_pre2024(te["delivery_hour"], f"{name} {q[2]} test")
        if not len(te) or not len(tr):
            log(f"{name} {q[2]}: no rows (train {len(tr)}, test {len(te)}), skipped")
            continue
        r = fit_predict(tr, te, q)
        r = r.to_frame("pred") if isinstance(r, pd.Series) else r
        r = r.reindex(te.index)
        keys = te[["delivery_hour", "zone", "delivery_date"]].copy()
        part = pd.concat([keys, r], axis=1)
        part["quarter"], part["warmup"] = q[2], q[3]
        part.to_parquet(f)
        out.append(part)
        log(f"{name} {q[2]}: train {len(tr):,} rows {tr['delivery_date'].min().date()}..{tr['delivery_date'].max().date()}, "
            f"test {len(te):,}, {time.time() - t0:.0f} s")
    res = pd.concat(out, ignore_index=True)
    assert_pre2024(res["delivery_hour"], f"{name} preds")
    res.to_parquet(PREDS / f"{name}.parquet")
    return res


def load_preds(name: str) -> pd.DataFrame:
    p = read_pre2024(PREDS / f"{name}.parquet", "delivery_hour")
    p["delivery_hour"] = p["delivery_hour"].dt.tz_convert(TZ)
    p["delivery_date"] = pd.to_datetime(p["delivery_date"]).dt.normalize()
    return p


# ============================================================================ settings: trailing out-of-sample choice
def trailing_mask(rows: pd.DataFrame, q_start: dt.date, n_quarters: int = TRAIL_QUARTERS) -> np.ndarray:
    """Rows of the n quarters before q_start whose outcomes were public by 05:00 on the bid day for q_start:
    delivery dates up to q_start - 2 days (the last full day settled before 05:00 on q_start - 1)."""
    lo = pd.Timestamp(q_start) - pd.DateOffset(months=3 * n_quarters)
    hi = pd.Timestamp(q_start - dt.timedelta(days=TRAIN_GAP_DAYS))
    d = _dd(rows)
    return ((d >= lo) & (d <= hi)).to_numpy()


def net_pnl(rows: pd.DataFrame, mw: np.ndarray) -> float:
    gap = rows["gap"].to_numpy(float)
    mw = np.where(np.isfinite(gap), np.nan_to_num(np.asarray(mw, float)), 0.0)
    sup, lod = row_costs(_dd(rows).dt.year)
    c = np.where(mw < 0, sup, lod)
    return float(np.sum(mw * np.nan_to_num(gap) - np.abs(mw) * c))


def choose_setting(rows: pd.DataFrame, settings: list, pos_fn, q_start: dt.date, default=None) -> tuple:
    """rows: earlier out-of-sample rows (delivery_date, gap and what pos_fn needs). pos_fn(rows, s) -> mw.
    Best net over the trailing 4 quarters at full cost; ties go to the first grid value; no rows = default."""
    m = trailing_mask(rows, q_start)
    if not m.any():
        return (settings[0] if default is None else default), "declared default (no earlier out-of-sample quarter)"
    sub = rows[m]
    vals = [net_pnl(sub, pos_fn(sub, s)) for s in settings]
    i = int(np.argmax(vals))
    return settings[i], f"best of {len(settings)} on {int(m.sum()):,} trailing rows"


# ============================================================================ positions
def write_positions(name: str, df: pd.DataFrame, meta: dict) -> Path:
    """df: delivery_hour (tz NY), zone, mw (signed; several rows per zone-hour are summed). Rows of the 2012
    warm-up may be included (the allocator learns from them); the scorer scores 2013..2023 only.
    meta: idea, line, settings_tried (int), choices (per quarter), notes."""
    d = df[["delivery_hour", "zone", "mw"]].copy()
    d["delivery_hour"] = pd.to_datetime(d["delivery_hour"]).dt.tz_convert(TZ)
    assert_pre2024(d["delivery_hour"], f"positions {name}")
    d = d.groupby(["delivery_hour", "zone"], as_index=False)["mw"].sum()
    d = d[d["delivery_hour"] >= pd.Timestamp(dt.date(WARMUP_YEAR, 1, 1), tz=TZ)]
    POS.mkdir(parents=True, exist_ok=True)
    f = POS / f"{name}.parquet"
    d.to_parquet(f)
    meta = {"name": name, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": int(len(d)), **meta}
    meta.setdefault("settings_tried", 1)
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return f


def done(marker: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{marker}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
