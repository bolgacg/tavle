"""Tests for deep.py (owner: deep agent). CPU, synthetic tables, tiny network.

    cd model && python -m pytest -q -p no:cacheprovider test_deep.py

  * fit/predict follow the Model interface: a Series aligned to the panel index (shuffled, with a
    non-default index), finite, the same for every row of a zone-hour;
  * the same seeds give the same predictions (CPU), and a saved model reloads to the same predictions;
  * end to end, the prediction for delivery day D+1 does not change when every value published after
    05:00 on D is replaced (data reloaded from perturbed tables);
  * extra panel columns (ideas B and D) are accepted;
  * the holdout lock refuses a 2024 delivery date in fit and in predict.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import deep
import deep_data as DD
import lock
from test_deep_data import _perturb_after, _rewrite

FIRST, LAST = dt.date(2023, 1, 1), dt.date(2023, 4, 30)
TINY = dict(hidden=16, K=2, max_epochs=3, patience=2, min_epochs=1, val_days=10, batch=16, min_point_days=5)


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("deep_syn")
    frames = DD.write_synthetic_parquet(root, FIRST, LAST, n_points=12, seed=11)
    dd = DD.DeepData(LAST, root=root, first_day=FIRST)
    pz = frames["prices_zone"]
    panel = pd.DataFrame({"zone": pz["zone"].astype(str), "delivery_hour": pz["delivery_hour"],
                          "gap": pz["rt_lbmp"] - pz["da_lbmp"]})
    panel = panel[panel["gap"].notna()].reset_index(drop=True)
    dday = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    panel["bid_date"] = (dday - pd.Timedelta(days=1)).dt.date
    train = panel[dday <= pd.Timestamp("2023-03-30")]
    test = panel[(dday >= pd.Timestamp("2023-04-01"))].sample(frac=1.0, random_state=0)
    test.index = [f"r{i}" for i in range(len(test))]
    return {"root": root, "frames": frames, "dd": dd, "train": train, "test": test}


def _fit(world, seeds=(0, 1), **kw):
    m = deep.DeepModel({**TINY, **kw}, seeds=seeds, data=world["dd"], device="cpu")
    m.fit(world["train"])
    return m


def test_fit_predict_interface(world):
    m = _fit(world)
    p = m.predict(world["test"])
    assert isinstance(p, pd.Series) and p.index.equals(world["test"].index)
    assert np.isfinite(p).all() and p.std() > 0
    # one prediction per (delivery date, local hour, zone): duplicates of the fall-back hour share it
    k = deep.panel_keys(world["test"])
    g = pd.DataFrame({"p": p.to_numpy(), "d": k["delivery_date"].to_numpy(), "s": k["slot"].to_numpy(),
                      "z": k["zone"].to_numpy()}).groupby(["d", "s", "z"])["p"].nunique()
    assert (g == 1).all()
    assert m.info["n_days"] > 50 and len(m.info["seeds"]) == 2


def test_same_seed_same_predictions_and_save_load(world, tmp_path):
    a = _fit(world)
    b = _fit(world)
    pa, pb = a.predict(world["test"]), b.predict(world["test"])
    assert np.allclose(pa, pb, atol=1e-5)
    a.save(tmp_path / "m.pt")
    c = deep.DeepModel.load(tmp_path / "m.pt", data=world["dd"], device="cpu")
    assert np.allclose(c.predict(world["test"]), pa, atol=1e-6)


def test_prediction_ignores_values_published_after_deadline(world, tmp_path):
    m = _fit(world, seeds=(0,))
    m.save(tmp_path / "m.pt")
    D = dt.date(2023, 4, 12)                                    # bid day; delivery 13 April
    rows = world["test"][deep.panel_keys(world["test"])["bid_date"] == pd.Timestamp(D)]
    base = m.predict(rows)
    _rewrite(tmp_path / "pert", _perturb_after(world["frames"], DD.decision_ts(D), np.random.default_rng(5)))
    dd2 = DD.DeepData(LAST, root=tmp_path / "pert", first_day=FIRST)
    m2 = deep.DeepModel.load(tmp_path / "m.pt", data=dd2, device="cpu")
    assert np.array_equal(m2.predict(rows).to_numpy(), base.to_numpy())
    # control: perturbing what WAS public at 05:00 on D does change the prediction
    _rewrite(tmp_path / "pert2", _perturb_after(world["frames"], DD.decision_ts(D) - pd.Timedelta(days=2),
                                                 np.random.default_rng(5)))
    m3 = deep.DeepModel.load(tmp_path / "m.pt", data=DD.DeepData(LAST, root=tmp_path / "pert2", first_day=FIRST),
                             device="cpu")
    assert not np.allclose(m3.predict(rows).to_numpy(), base.to_numpy())


def test_extra_columns(world):
    tr, te = world["train"].copy(), world["test"].copy()
    rng = np.random.default_rng(0)
    tr["x1"], te["x1"] = rng.normal(size=len(tr)), rng.normal(size=len(te))
    te.loc[te.index[:10], "x1"] = np.nan
    m = deep.DeepModel({**TINY}, seeds=(0,), data=world["dd"], device="cpu", extra_cols=["x1"])
    m.fit(tr)
    assert np.isfinite(m.predict(te)).all()


def test_no_points_config(world):
    m = _fit(world, seeds=(0,), K=0)
    assert len(m.cols) == 0 and np.isfinite(m.predict(world["test"])).all()


def test_tcn_config(world):
    m = _fit(world, seeds=(0,), arch="tcn")
    assert np.isfinite(m.predict(world["test"])).all()


def test_holdout_dates_refused(world):
    m = deep.DeepModel(TINY, seeds=(0,), data=world["dd"], device="cpu")
    bad = world["train"].head(5).copy()
    bad["delivery_hour"] = pd.Timestamp("2024-01-01 05:00", tz=DD.TZ)
    bad["bid_date"] = dt.date(2023, 12, 31)
    with pytest.raises(lock.HoldoutLocked):
        m.fit(pd.concat([world["train"], bad]))
    m = _fit(world, seeds=(0,))
    with pytest.raises(lock.HoldoutLocked):
        m.predict(bad)


@pytest.mark.parametrize("visible", [True, False], ids=["known_at_05h", "published_after_05h"])
def test_planted_signal(tmp_path, visible):
    """End-to-end alignment and leakage check. The gap of hour h of D+1 is planted as
      known_at_05h:        0.5 x (day-ahead price of the same hour of D, public at 11:00 on D-1), or
      published_after_05h: 0.5 x (day-ahead price of hour h of D+1 minus that of D), which needs D+1's
                           day-ahead prices (public 11:00 on D, after the decision),
    plus noise. The first must be learned (targets, windows and output slots line up); the second must
    not be (no path from data published after 05:00 on D into the prediction)."""
    first, last = dt.date(2023, 4, 1), dt.date(2023, 7, 31)
    frames = DD.write_synthetic_parquet(tmp_path / "raw", first, last, n_points=6, seed=21)
    pz = frames["prices_zone"].copy()
    rng = np.random.default_rng(0)
    da = pz.set_index(["zone", "delivery_hour"])["da_lbmp"]
    prev = da.reindex(pd.MultiIndex.from_arrays([pz["zone"], pz["delivery_hour"] - pd.Timedelta(hours=24)])).to_numpy()
    signal = 0.5 * (prev - np.nanmean(prev)) if visible else 0.5 * (pz["da_lbmp"].to_numpy() - prev)
    pz["rt_lbmp"] = np.where(pz["rt_lbmp"].isna(), np.nan, pz["da_lbmp"] + np.nan_to_num(signal) + rng.normal(0, 2, len(pz)))
    frames["prices_zone"] = pz
    _rewrite(tmp_path / "planted", frames)
    dd = DD.DeepData(last, root=tmp_path / "planted", first_day=first)
    panel = pd.DataFrame({"zone": pz["zone"].astype(str), "delivery_hour": pz["delivery_hour"],
                          "gap": pz["rt_lbmp"] - pz["da_lbmp"], "signal": signal})
    panel = panel[panel["gap"].notna() & np.isfinite(panel["signal"])].reset_index(drop=True)
    dday = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    train, test = panel[dday <= pd.Timestamp("2023-06-29")], panel[dday >= pd.Timestamp("2023-07-01")]
    m = deep.DeepModel({**TINY, "hidden": 32, "max_epochs": 40, "patience": 8, "min_epochs": 5, "val_days": 14,
                        "lr": 3e-3, "dropout": 0.1, "weight_decay": 1e-4}, seeds=(0,), data=dd, device="cpu")
    m.fit(train)
    p = m.predict(test)
    r_gap = np.corrcoef(p, test["gap"])[0, 1]
    print(f"planted visible={visible}: corr(pred, gap) = {r_gap:.3f}")
    if visible:
        assert r_gap > 0.6, r_gap
    else:
        assert abs(r_gap) < 0.15, r_gap


# ----------------------------------------------------------------------------- spike head
def test_spike_threshold_file(tmp_path):
    assert deep.spike_threshold(tmp_path / "absent.json") == (50.0, "default 50 (absent.json absent)")
    (tmp_path / "s.json").write_text('{"S": 75, "note": "chosen on 2020-2022"}')
    assert deep.spike_threshold(tmp_path / "s.json")[0] == 75.0
    (tmp_path / "lead.json").write_text('{"S_usd_per_mwh": 25, "S_grid": [25, 50, 100]}')
    assert deep.spike_threshold(tmp_path / "lead.json") == (25.0, "lead.json: S_usd_per_mwh=25")
    (tmp_path / "bad.json").write_text('{"cut": 0.3}')
    with pytest.raises(KeyError):
        deep.spike_threshold(tmp_path / "bad.json")


def test_spike_interface_and_save_load(world, tmp_path):
    m = deep.DeepSpikeModel({**TINY}, threshold=20.0, seeds=(0, 1), data=world["dd"], device="cpu")
    m.fit(world["train"])
    p = m.predict(world["test"])
    assert p.index.equals(world["test"].index) and ((p > 0) & (p < 1)).all() and p.std() > 0
    both = m.predict_both(world["test"])
    assert list(both.columns) == ["p_spike", "p_spike_weighted", "pred_gap"] and np.allclose(both["p_spike"], p)
    w = m.info["pos_weight"]
    assert m.info["threshold"] == 20.0 and w > 1
    assert (both["p_spike"] <= both["p_spike_weighted"] + 1e-12).all()
    # calibrated: the average probability is near the training spike rate, the weighted one is not
    rate = m.info["train_spike_rate"]
    assert 0.4 * rate < p.mean() < 2.5 * rate, (p.mean(), rate)
    assert both["p_spike_weighted"].mean() > 2 * rate
    one = deep.DeepSpikeModel({**TINY}, threshold=20.0, seeds=(0,), data=world["dd"], device="cpu")
    one.fit(world["train"])
    b1 = one.predict_both(world["test"])
    q, w1 = b1["p_spike_weighted"].to_numpy(), one.info["pos_weight"]
    assert np.allclose(b1["p_spike"], q / (q + w1 * (1 - q)))
    m.save(tmp_path / "s.pt")
    m2 = deep.DeepSpikeModel.load(tmp_path / "s.pt", data=world["dd"], device="cpu")
    assert m2.threshold == 20.0 and np.allclose(m2.predict(world["test"]), p, atol=1e-6)
    bad = world["train"].head(3).copy()
    bad["delivery_hour"] = pd.Timestamp("2024-01-02 05:00", tz=DD.TZ)
    bad["bid_date"] = dt.date(2024, 1, 1)
    with pytest.raises(lock.HoldoutLocked):
        m.predict(bad)


@pytest.mark.parametrize("visible", [True, False], ids=["known_at_05h", "published_after_05h"])
def test_spike_planted(tmp_path, visible):
    """Spikes (gap +80) planted in the hours where the day-ahead price of the same hour of D was in its
    top 15% (known at 05:00 on D): the spike output must find them. Planted instead where D+1's own
    day-ahead price rose most against D (published after 05:00 on D): it must not."""
    first, last = dt.date(2023, 4, 1), dt.date(2023, 7, 31)
    frames = DD.write_synthetic_parquet(tmp_path / "raw", first, last, n_points=6, seed=21)
    pz = frames["prices_zone"].copy()
    rng = np.random.default_rng(1)
    da = pz.set_index(["zone", "delivery_hour"])["da_lbmp"]
    prev = da.reindex(pd.MultiIndex.from_arrays([pz["zone"], pz["delivery_hour"] - pd.Timedelta(hours=24)])).to_numpy()
    sig = prev if visible else pz["da_lbmp"].to_numpy() - prev
    hot = sig > np.nanquantile(sig, 0.85)
    pz["rt_lbmp"] = np.where(pz["rt_lbmp"].isna(), np.nan, pz["da_lbmp"] + rng.normal(0, 3, len(pz)) + 80 * hot)
    frames["prices_zone"] = pz
    _rewrite(tmp_path / "planted", frames)
    dd = DD.DeepData(last, root=tmp_path / "planted", first_day=first)
    panel = pd.DataFrame({"zone": pz["zone"].astype(str), "delivery_hour": pz["delivery_hour"],
                          "gap": pz["rt_lbmp"] - pz["da_lbmp"], "ok": np.isfinite(sig)})
    panel = panel[panel["gap"].notna() & panel["ok"]].reset_index(drop=True)
    dday = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    train, test = panel[dday <= pd.Timestamp("2023-06-29")], panel[dday >= pd.Timestamp("2023-07-01")]
    m = deep.DeepSpikeModel({**TINY, "hidden": 32, "max_epochs": 40, "patience": 8, "min_epochs": 5, "val_days": 14,
                             "lr": 3e-3, "dropout": 0.1, "weight_decay": 1e-4}, threshold=50.0, seeds=(0,),
                            data=dd, device="cpu")
    m.fit(train)
    a = deep.auc(m.predict(test).to_numpy(), (test["gap"] >= 50).to_numpy(float))
    print(f"spike planted visible={visible}: AUC = {a:.3f}")
    if visible:
        assert a > 0.85, a
    else:
        assert abs(a - 0.5) < 0.1, a
