"""V13 tests on synthetic tables (nothing real is read): the injected-lookahead tests and a CPU fit of every
architecture. The real-data self-test runs at the start of every v13_rolling.py job (lookahead_selftest).

    cd ~/nyiso-us/v2/model && ../../.venv/bin/python -m pytest -q -p no:cacheprovider v13_test.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rolling as R  # noqa: E402
import deep_data as DD  # noqa: E402
import v13_data as VD  # noqa: E402
import v13_model as VM  # noqa: E402
import v13_rolling as VR  # noqa: E402
from timing import LookaheadError  # noqa: E402

FIRST, LAST = dt.date(2015, 5, 1), dt.date(2015, 8, 31)
TINY = dict(hidden=8, K=2, G=4, max_epochs=2, min_epochs=1, batch=8, val_days=10, heads=2, layers=1)


def day_matrix(first, last, seed=0, with_y=True):
    rng = np.random.default_rng(seed)
    dd = pd.date_range(first, last, freq="D")
    df = pd.DataFrame({"delivery_date": dd, "bid_date": dd - pd.Timedelta(days=1)})
    for j in range(6):
        df[f"bord__HQ__d{j}__da_mean"] = rng.normal(30, 5, len(dd))
    for h in range(24):
        df[f"lfwow__CAPITL__h{h:02d}"] = rng.normal(0, 50, len(dd))
        df[f"wxr__albany__h{h:02d}"] = np.where(dd < pd.Timestamp("2015-07-01"), rng.normal(20, 5, len(dd)), np.nan)
        df[f"wx__albany__h{h:02d}"] = np.where(dd >= pd.Timestamp("2015-08-01"), rng.normal(20, 5, len(dd)), np.nan)
    df["out__n_active"] = rng.integers(0, 20, len(dd)).astype(float)
    if with_y:
        df["y__book_supply_pnl"] = rng.normal(0, 1, len(dd))
    return df


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("pq")
    tabs = DD.write_synthetic_parquet(root, FIRST, LAST, n_points=24, seed=3)
    data = VD.Data(LAST, root=root, first_day=FIRST)
    dm = day_matrix(FIRST, LAST)
    f = root / "day.parquet"
    dm.to_parquet(f, index=False)
    pz = tabs["prices_zone"]
    panel = pz[["delivery_hour", "zone"]].copy()
    panel["gap"] = (pz["rt_lbmp"] - pz["da_lbmp"]).to_numpy()
    panel["delivery_hour"] = panel["delivery_hour"].dt.tz_convert(R.TZ)
    panel["delivery_date"] = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    return data, f, panel.reset_index(drop=True), root


def test_borders_excluded_and_masks(synth):
    data, f, panel, root = synth
    assert len(data.external_ptids) == 2 and not set(data.external_ptids) & set(data.ptids)
    # points that appear late or retire are masked outside their life
    alive = data.Gp[..., 0] < DD.NEVER
    assert (~alive).any() and alive.any()


def test_selftest_real_path(synth):
    data, f, panel, root = synth
    out = VR.lookahead_selftest(data, "cpu", print)
    assert out["PASS"]


def test_injected_rt_cell_is_caught(synth):
    data, f, panel, root = synth
    A = VD.device_arrays(data, "cpu")
    k = data.day_index(dt.date(2015, 6, 10)) + 1
    end = int(data.day_start[k])
    cols = torch.arange(len(data.ptids))
    VD.batch(data, [k], cols, "cpu")                        # clean
    old = A["Gp"][end - 2, 3, 1].clone()
    A["Gp"][end - 2, 3, 1] = int(data.dec_min[k]) - 10      # an RT point value of 22:00 on D marked public at 04:50
    with pytest.raises(LookaheadError):
        VD.batch(data, [k], cols, "cpu")
    A["Gp"][end - 2, 3, 1] = old


def test_day_matrix_guards(synth, tmp_path):
    data, f, panel, root = synth
    b = VD.DayBlocks(data, f, with_weather=True, log=print)
    assert not any(c.startswith("y__") for cs in b.cols.values() for c in cs)
    assert b.wxsrc[:, 0].sum() > 0 and b.wxsrc[:, 1].sum() > 0
    dm = day_matrix(FIRST, LAST)
    dm["delivery_date"] = dm["delivery_date"] - pd.Timedelta(days=1)    # D+1's row labelled as D's
    g = tmp_path / "bad.parquet"
    dm.to_parquet(g, index=False)
    with pytest.raises(LookaheadError):
        VD.DayBlocks(data, g, with_weather=True, log=print)
    b.delivery[5] += 1
    with pytest.raises(LookaheadError):
        b.assert_aligned(np.array([5]))
    b.delivery[5] -= 1
    with pytest.raises(RuntimeError):                      # weather asked, no wxr__ columns
        dm2 = day_matrix(FIRST, LAST).drop(columns=[c for c in dm.columns if c.startswith("wxr__")])
        h = tmp_path / "nowxr.parquet"
        dm2.to_parquet(h, index=False)
        VD.DayBlocks(data, h, with_weather=True, log=print)


@pytest.mark.parametrize("arch", VM.ARCHS)
def test_fit_predict_and_future_values_change_nothing(synth, arch):
    data, f, panel, root = synth
    blocks = VD.DayBlocks(data, f, with_weather=True, log=print)
    q0 = dt.date(2015, 8, 1)
    d = pd.to_datetime(panel["delivery_date"])
    tr = panel[(d <= pd.Timestamp(q0 - dt.timedelta(days=2))) & panel["gap"].notna()]
    te = panel[d == pd.Timestamp(q0)]
    m = VM.V13(data, blocks, arch, seeds=(0, 1), device="cpu", cfg=TINY, log=print)
    m.fit(tr)
    p1 = m.predict(te)
    assert np.isfinite(p1.to_numpy()).all() and (p1[[f"p_s{S}" for S in VM.SPIKES]].to_numpy() <= 1).all()
    # corrupt everything not public at 05:00 on the bid day of q0: RT of hours from 04:00 on, every value of q0
    # and later, day-level rows after q0, labels after the cut
    k0 = data.day_index(q0)
    t0 = int(data.dec_min[k0])
    A = VD.device_arrays(data, "cpu")
    saved = {x: A[x].clone() for x in ("Zv", "Gv")}
    late_rt = torch.as_tensor(data.Hmin + DD.RT_RULE_MIN > t0)
    late_day = torch.as_tensor(data.Hday >= k0)
    A["Zv"][late_rt, :, 3:] = 1e6
    A["Zv"][late_day] = 1e6
    A["Gv"][late_rt, :, 2:] = 1000
    A["Gv"][late_day] = 1000
    Dsave = {x: v.clone() for x, v in m.D.items()}
    for x in m.D:
        m.D[x][k0 + 1:] = 7.0
    p2 = m.predict(te)
    for x in ("Zv", "Gv"):
        A[x].copy_(saved[x])
    m.D = Dsave
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    # sanity: corrupting a public input DOES move the prediction
    A["Zv"][int(data.day_start[k0]) - 30, :, :3] += 500.0
    p3 = m.predict(te)
    A["Zv"].copy_(saved["Zv"])
    assert not np.allclose(p1["pred"].to_numpy(), p3["pred"].to_numpy())


def test_ablation_removes_group(synth):
    data, f, panel, root = synth
    blocks = VD.DayBlocks(data, f, with_weather=True, log=print)
    d = pd.to_datetime(panel["delivery_date"])
    tr = panel[(d <= pd.Timestamp("2015-07-20")) & panel["gap"].notna()]
    te = panel[d == pd.Timestamp("2015-07-25")]
    m = VM.V13(data, blocks, "tcn", drop=("weather", "outage"), seeds=(0,), device="cpu", cfg=TINY, log=print)
    m.fit(tr)
    p1 = m.predict(te)
    m.D["wx"] += 5.0
    m.D["out"] += 5.0
    p2 = m.predict(te)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    mg = VM.V13(data, blocks, "gru", drop=("gen",), seeds=(0,), device="cpu", cfg=TINY, log=print)
    mg.fit(tr)
    assert len(mg.cols) == 0
