"""Tests for deep_day.py (owner: deep agent). Synthetic tables from conftest.write_synthetic (all five
sources, including January 2024 rows that the lock must keep out).

    cd model && python -m pytest -q -p no:cacheprovider test_deep_day.py

  * injected lookahead: every value published after 05:00 on D (prices by component, load forecasts,
    weather, outage snapshots) replaced with garbage changes no feature of D and no prediction for D+1;
    the control (values published before 05:00 replaced) does change them;
  * the hand-checked features match the sources (load forecast vintage, published real-time hours);
  * fit / predict for both tasks; the storm threshold L is the training 10th percentile;
  * the lock refuses delivery 2024-01-01.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import deep_day as DY
import lock
import panel as P
from common import TZ, decision_time

DAYS = [dt.date(2022, 11, 1) + dt.timedelta(days=i) for i in range(150)]
TINY = dict(h_enc=8, h_trunk=16, max_epochs=6, patience=3, val_min=10, batch=32)


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


@pytest.fixture(scope="module")
def feats(synth_root):
    return DY.build_day_features(DAYS, P.LockedStore(root=synth_root))


def _fcols(df):
    return [c for c in df.columns if "__" in c and not c.startswith("y__")]


def _perturb(root: Path, out: Path, t: pd.Timestamp, after: bool, seed=0) -> Path:
    """Copy the tables, replacing every value published after t (after=True) or at/before t."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    pick = (lambda pub: ~(pub <= t)) if after else (lambda pub: pub <= t)
    pz = pd.read_parquet(root / "prices_zone.parquet")
    for side in ("da", "rt"):
        m = pick(pz[f"{side}_published_at"]).to_numpy()
        for c in (f"{side}_lbmp", f"{side}_congestion", f"{side}_loss"):
            pz.loc[m, c] = rng.uniform(-5000, 9000, int(m.sum()))
    pz.to_parquet(out / "prices_zone.parquet")
    lf = pd.read_parquet(root / "load_forecast.parquet")
    m = pick(lf["published_at"]).to_numpy()
    lf.loc[m, "load_forecast_mw"] = rng.uniform(1e5, 1e6, int(m.sum()))
    lf.to_parquet(out / "load_forecast.parquet")
    wx = pd.read_parquet(root / "weather_gfs.parquet")
    m = pick(wx["published_at"]).to_numpy()
    wx.loc[m, "temperature_2m_c"] = rng.uniform(-60, 60, int(m.sum()))
    wx.to_parquet(out / "weather_gfs.parquet")
    og = pd.read_parquet(root / "outages.parquet")
    m = pick(og["published_at"]).to_numpy()
    og.loc[m, "equipment"] = "FAKE_345KV_LINE"
    og.loc[m, "sched_out"] = pd.Timestamp("2022-01-01", tz=TZ)
    og.loc[m, "sched_in"] = pd.Timestamp("2025-01-01", tz=TZ)
    og.to_parquet(out / "outages.parquet")
    return out


@pytest.mark.parametrize("D", [dt.date(2023, 1, 9), dt.date(2023, 3, 11), dt.date(2023, 3, 12)], ids=str)
def test_post_deadline_values_change_nothing(synth_root, feats, tmp_path, D):
    t = decision_time(D)
    clean = DY.build_day_features([D], P.LockedStore(root=synth_root), with_labels=False)
    pert = DY.build_day_features([D], P.LockedStore(root=_perturb(synth_root, tmp_path / "after", t, True)),
                                 with_labels=False)
    cols = _fcols(clean)
    a, b = clean[cols].to_numpy(float), pert[cols].to_numpy(float)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=0)
    # and no prediction moves: a model fitted on the clean days predicts D+1 identically
    fit_days = feats[feats["delivery_date"] < pd.Timestamp(D)]
    for task in ("storm", "zone"):
        m = DY.DeepDayModel(task, TINY, seeds=(0,), device="cpu")
        m.fit(fit_days)
        pa, pb = m.predict(clean), m.predict(pert)
        assert np.array_equal(np.asarray(pa), np.asarray(pb))
    # control: replacing what WAS public at 05:00 does change the features
    ctrl = DY.build_day_features([D], P.LockedStore(root=_perturb(synth_root, tmp_path / "before", t, False)),
                                 with_labels=False)
    c = ctrl[cols].to_numpy(float)
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(c))


def test_features_match_sources(synth_root, feats):
    D = dt.date(2023, 2, 1)
    row = feats[feats["bid_date"] == pd.Timestamp(D)].iloc[0]
    lf = pd.read_parquet(synth_root / "load_forecast.parquet")
    t = decision_time(D)
    d1 = lf[(lf["target_hour"] == pd.Timestamp("2023-02-02 18:00", tz=TZ)) & (lf["zone"] == "N.Y.C.")]
    d1 = d1[d1["published_at"] <= t].sort_values("published_at").iloc[-1]
    assert row["lf__NYC__h18"] == pytest.approx(d1["load_forecast_mw"])
    assert pd.Timestamp(d1["issue_date"]) == pd.Timestamp(D)           # the file named D, written 07:05 on D-1
    # real-time hours of D: 00..03 are public at 05:00 (hour end + 15 min), 04 and 05 are not
    assert np.isfinite(row["rtnow__NYC__h03__rt"]) and np.isnan(row["rtnow__NYC__h04__rt"])
    assert row["rtnow__n_hours"] == 4
    # weather for D+1 exists (synthetic two-day values up to 21:00), every point
    assert np.isfinite(row[[c for c in feats.columns if c.startswith("wx__") and c.endswith("__h12")]].to_numpy(float)).all()
    # labels: book = sum of zones
    z = row[[c for c in feats.columns if c.startswith("y__zone_supply_pnl__")]].to_numpy(float)
    assert row["y__book_supply_pnl"] == pytest.approx(z.sum())


def test_fit_predict(feats):
    tr, te = feats.iloc[:120], feats.iloc[120:]
    s = DY.DeepDayModel("storm", TINY, seeds=(0, 1), device="cpu")
    s.fit(tr)
    p = s.predict(te)
    assert p.index.equals(te.index) and ((p > 0) & (p < 1)).all()
    assert s.L == pytest.approx(np.quantile(tr["y__book_supply_pnl"], 0.10))
    z = DY.DeepDayModel("zone", TINY, seeds=(0, 1), device="cpu")
    z.fit(tr)
    q = z.predict(te)
    assert q.shape == (len(te), 11) and np.isfinite(q.to_numpy()).all()


def test_lock_refuses_holdout(synth_root):
    with pytest.raises(lock.HoldoutLocked):
        DY.build_day_features([dt.date(2023, 12, 31)], P.LockedStore(root=synth_root))


def test_walk_and_evaluate(feats):
    import train_deep_day as TDD
    out, recs = TDD.walk(feats, (dt.date(2023, 2, 1), dt.date(2023, 3, 31)), seeds=(0,), device="cpu", config=TINY,
                         log=lambda *_: None)
    assert [r["month"] for r in recs] == ["2023-02-01", "2023-03-01"]
    days = out["delivery_date"].nunique()
    assert days == 28 + 31 - 1 or days == 28 + 31            # the last bid day of the fixture is 30 March
    assert set(out["zone"]) == {"ALL", *__import__("common").ZONES}
    assert out["p_storm"].between(0, 1).all() and out["pred_zone_supply_pnl"].notna().all()
    a = out[out["zone"] == "ALL"].set_index("delivery_date")["pred_zone_supply_pnl"]
    b = out[out["zone"] != "ALL"].groupby("delivery_date")["pred_zone_supply_pnl"].sum()
    assert np.allclose(a.sort_index(), b.sort_index())
    hand = pd.Series(np.random.default_rng(0).random(400), index=pd.date_range("2022-11-01", periods=400))
    ev = TDD.evaluate(out, feats, hand)
    assert set(ev) == {2023} and 0 <= ev[2023]["auc_p_storm"] <= 1 and ev[2023]["hand_days"] > 50
