"""Tests for deep_data.py (owner: deep agent).

    cd model && python -m pytest -q -p no:cacheprovider test_deep_data.py

On synthetic tables with the real schema and publication rules (DST days included):
  1. the cells deep_data marks visible for bid day D are exactly the rows timing.features_available_at
     and timing.bid_inputs return for 05:00 on D, with the same values;
  2. the GPU/torch batch path marks the same cells as the numpy reference;
  3. replacing every value published after 05:00 on D leaves the tensors for D bit-identical;
  4. injected lookahead is caught: a mislabelled (too early) published_at on a real-time hour, a
     generator hour or the isolf file named D+1, and a tampered mask;
  5. the holdout lock refuses 2024 before anything is read, and reads stop at the boundary.
On gene, test_real_tables_match_timing repeats test 1 on real build-year days (2023 only).
"""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import deep_data as DD
import lock
import timing as T
from common import PARQUET, TZ, ZONES

PERIODS = {"fall": (dt.date(2023, 10, 20), dt.date(2023, 11, 20)),
           "spring": (dt.date(2023, 2, 28), dt.date(2023, 3, 25))}


@pytest.fixture(autouse=True)
def _locked(monkeypatch):
    monkeypatch.delenv("US_HOLDOUT_RUN", raising=False)


@pytest.fixture(scope="module", params=list(PERIODS))
def synth(request, tmp_path_factory):
    first, last = PERIODS[request.param]
    root = tmp_path_factory.mktemp(f"syn_{request.param}")
    frames = DD.write_synthetic_parquet(root, first, last, seed=7)
    dd = DD.DeepData(last, root=root, first_day=first)
    days = [first + dt.timedelta(days=i) for i in range(8, (last - first).days)]
    return {"root": root, "frames": frames, "dd": dd, "days": days, "first": first, "last": last}


def _store(root: Path) -> T.Store:
    tz = f"datetime64[ns, {TZ}]"
    return T.Store(root=root, tables={
        "prices_zone": pd.read_parquet(root / "prices_zone.parquet"),
        "load_forecast": pd.read_parquet(root / "load_forecast.parquet"),
        "outages": pd.DataFrame({"snapshot_date": pd.to_datetime(pd.Series([], dtype="datetime64[ns]")),
                                 "published_at": pd.Series([], dtype=tz)}),
        "weather_gfs": pd.DataFrame({"target_hour": pd.Series([], dtype=tz), "temperature_2m_c": pd.Series([], dtype=float),
                                     "published_at": pd.Series([], dtype=tz)})})


def _mine(dd: DD.DeepData, D: dt.date) -> dict:
    """Visible cells from the numpy reference: {(source, hour_ns, key): value}."""
    s = dd.sample(D)
    dd.assert_sample_no_lookahead(D, s)
    hns = dd.origin_ns + (s["start"] + np.arange(DD.WINDOW_H, dtype=np.int64)) * DD.HOUR_NS
    out = {}
    for src, m, v in (("zda", s["m_da"], s["zv"][..., 0]), ("zrt", s["m_rt"], s["zv"][..., 3])):
        for i, z in zip(*np.nonzero(m)):
            out[(src, int(hns[i]), ZONES[z])] = float(v[i, z])
    for src, m, v in (("gda", s["m_gda"], s["gv"][..., 0]), ("grt", s["m_grt"], s["gv"][..., 1])):
        for i, p in zip(*np.nonzero(m)):
            out[(src, int(hns[i]), int(dd.ptids[p]))] = float(v[i, p])
    return out, (int(hns[0]), int(hns[-1]) + DD.HOUR_NS)


def _theirs(store: T.Store, D: dt.date, lo: int, hi: int) -> dict:
    f = T.features_available_at(DD.decision_ts(D), store, lookback_days=9, include_gen=True, gen_lookback_days=9)
    out = {}
    for src, name, col, key in (("zda", "da_prices_zone", "da_lbmp", "zone"), ("zrt", "rt_prices_zone", "rt_lbmp", "zone"),
                                ("gda", "da_prices_gen", "da_congestion", "ptid"),
                                ("grt", "rt_prices_gen", "rt_congestion", "ptid")):
        df = f.get(name, pd.DataFrame())
        if not len(df):
            continue
        h = DD._ns(df["delivery_hour"])
        v = df[col].to_numpy(float)
        keep = (h >= lo) & (h < hi) & np.isfinite(v)
        ks = df[key].to_numpy()[keep]
        for hh, kk, vv in zip(h[keep], ks, v[keep]):
            out[(src, int(hh), kk if key == "zone" else int(kk))] = float(vv)
    return out


def _compare(a: dict, b: dict):
    assert set(a) == set(b), (sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5])
    assert all(abs(a[k] - b[k]) < 1e-3 for k in a)


def test_visible_cells_equal_timing(synth):
    dd, store = synth["dd"], _store(synth["root"])
    for D in synth["days"][::3]:
        mine, (lo, hi) = _mine(dd, D)
        _compare(mine, _theirs(store, D, lo, hi))
        # rows the deadline removes exist in the window (the filter is doing work)
        s = dd.sample(D)
        assert (~s["m_rt"]).sum() > 0 and s["m_da"].sum() > 0


def test_load_forecast_equals_bid_inputs(synth):
    dd, store = synth["dd"], _store(synth["root"])
    rows = dd.lf_rows
    for D in synth["days"][::2]:
        lf = T.bid_inputs(D, store)["load_forecast_d1"]
        theirs = {(int(h), z): (float(v), str(i)) for h, z, v, i in
                  zip(DD._ns(lf["target_hour"]), lf["zone"], lf["load_forecast_mw"], lf["issue_date"].astype(str))}
        r = rows[DD.local_dates(rows["target_hour"]) == np.datetime64(D + dt.timedelta(days=1), "D")]
        mine = {(int(h), z): (float(v), str(i)) for h, z, v, i in
                zip(DD._ns(r["target_hour"]), r["zone"], r["load_forecast_mw"], pd.to_datetime(r["issue_date"]).astype(str))}
        assert set(mine) == set(theirs) and len(mine) > 0
        assert all(abs(mine[k][0] - theirs[k][0]) < 1e-6 and mine[k][1][:10] == theirs[k][1][:10] for k in mine)
        # the dense array holds the same values (two 01:00 hours of a fall-back day are averaged)
        k = dd.day_index(D) + 1
        for (h, z), (v, _) in mine.items():
            slot = pd.Timestamp(h, tz="UTC").tz_convert(TZ).hour
            if dd.LFm[k].sum() == 24 * DD.N_LF:
                assert dd.LF[k, slot, DD.LF_ZONES.index(z)] == pytest.approx(v, rel=1e-5) or slot == 1


def test_late_isolf_file_falls_back_to_older_vintage(synth):
    """When the file named D was written after 05:00 on D, the D+1 hours come from file D-1."""
    lf, dd = synth["frames"]["load_forecast"], synth["dd"]
    pub = lf.drop_duplicates("issue_date").set_index("issue_date")["published_at"]
    late = [d for d in synth["days"] if d in pub.index and pub[d] > DD.decision_ts(d)]
    for D in late:
        k = dd.day_index(D) + 1
        assert (dd.LFiss[k][dd.LFm[k]] == dd.day_index(D) - 1).all()


def _all_cols(dd):
    return torch.arange(len(dd.ptids), dtype=torch.long)


def test_torch_batch_masks_equal_numpy(synth):
    dd = synth["dd"]
    days = synth["days"][::4]
    k = torch.as_tensor([dd.day_index(D) + 1 for D in days])
    b = DD.batch(dd, k, _all_cols(dd), "cpu")
    nz = DD.N_Z
    for j, D in enumerate(days):
        s = dd.sample(D)
        assert np.array_equal(b["seq"][j, :, 7 * nz:8 * nz].numpy() > 0, s["m_da"])
        assert np.array_equal(b["seq"][j, :, 8 * nz:9 * nz].numpy() > 0, s["m_rt"])
        assert np.array_equal(b["gm"][j, ..., 0].numpy() > 0, s["m_gda"])
        assert np.array_equal(b["gm"][j, ..., 1].numpy() > 0, s["m_grt"])
        assert np.array_equal(b["lfm"][j].numpy() > 0, s["m_lf"])


def _rewrite(root: Path, frames: dict):
    (root / "prices_gen").mkdir(parents=True, exist_ok=True)
    frames["prices_zone"].to_parquet(root / "prices_zone.parquet", index=False)
    frames["load_forecast"].to_parquet(root / "load_forecast.parquet", index=False)
    pg = frames["prices_gen"]
    for m, part in pg.groupby(pg["delivery_hour"].dt.strftime("%Y%m")):
        part.to_parquet(root / "prices_gen" / f"{m}.parquet", index=False)


def _perturb_after(frames: dict, t: pd.Timestamp, rng) -> dict:
    """Replace every value published after t with garbage (huge numbers or NaN)."""
    out = {k: v.copy() for k, v in frames.items()}
    for name, cols in (("prices_zone", (["da_lbmp", "da_loss", "da_congestion"], ["rt_lbmp", "rt_loss", "rt_congestion"])),
                       ("prices_gen", (["da_congestion"], ["rt_congestion"]))):
        df = out[name]
        for side, cc in zip(("da", "rt"), cols):
            fut = ~(df[f"{side}_published_at"] <= t)
            for c in cc:
                g = rng.choice([1e4, -9e3, np.nan], size=int(fut.sum()))
                df.loc[fut, c] = g
    lf = out["load_forecast"]
    fut = ~(lf["published_at"] <= t)
    lf.loc[fut, "load_forecast_mw"] = rng.uniform(1e5, 1e6, int(fut.sum()))
    return out


def test_values_published_after_deadline_never_change_tensors(synth, tmp_path):
    rng = np.random.default_rng(1)
    dd = synth["dd"]
    for D in random.Random(3).sample(synth["days"], 3):
        root2 = tmp_path / f"p{D}"
        _rewrite(root2, _perturb_after(synth["frames"], DD.decision_ts(D), rng))
        dd2 = DD.DeepData(synth["last"], root=root2, first_day=synth["first"])
        assert np.array_equal(dd.ptids, dd2.ptids)
        k = torch.as_tensor([dd.day_index(D) + 1])
        a, b = DD.batch(dd, k, _all_cols(dd), "cpu"), DD.batch(dd2, k, _all_cols(dd2), "cpu")
        for name in a:
            assert torch.equal(a[name], b[name]), (D, name)
        # and the perturbation did reach data inside the window (the test is not vacuous)
        s1, s2 = dd.sample(D), dd2.sample(D)
        assert not np.array_equal(np.nan_to_num(s1["zv"]), np.nan_to_num(s2["zv"]))


def _mislabel(frames: dict, D: dt.date) -> dict:
    out = {k: v.copy() for k, v in frames.items()}
    t = DD.decision_ts(D)
    pz = out["prices_zone"]
    h5 = pd.Timestamp(dt.datetime.combine(D, dt.time(5)), tz=TZ)
    pz.loc[pz["delivery_hour"] == h5, "rt_published_at"] = t - pd.Timedelta(hours=1)
    pg = out["prices_gen"]
    h6 = h5 + pd.Timedelta(hours=1)
    pg.loc[(pg["delivery_hour"] == h6) & (pg["ptid"] == pg["ptid"].min()), "rt_published_at"] = t - pd.Timedelta(minutes=30)
    lf = out["load_forecast"]
    lf.loc[pd.to_datetime(lf["issue_date"]) == pd.Timestamp(D + dt.timedelta(days=1)), "published_at"] = \
        t - pd.Timedelta(minutes=30)
    return out


def test_injected_lookahead_is_caught(synth, tmp_path):
    dd = synth["dd"]
    D = synth["days"][len(synth["days"]) // 2]
    # clean day passes, in numpy and in the torch batch
    dd.assert_sample_no_lookahead(D, dd.sample(D))
    DD.batch(dd, torch.as_tensor([dd.day_index(D) + 1]), _all_cols(dd), "cpu", check=True)
    # mislabelled published_at in the tables: the cell passes the published_at filter, the rules catch it
    root3 = tmp_path / "mislabelled"
    _rewrite(root3, _mislabel(synth["frames"], D))
    dd3 = DD.DeepData(synth["last"], root=root3, first_day=synth["first"])
    s = dd3.sample(D)
    with pytest.raises(T.LookaheadError) as e:
        dd3.assert_sample_no_lookahead(D, s)
    for part in ("zone_rt", "gen_rt", "load_forecast"):
        assert part in str(e.value)
    with pytest.raises(T.LookaheadError):
        DD.batch(dd3, torch.as_tensor([dd3.day_index(D) + 1]), _all_cols(dd3), "cpu", check=True)
    # a tampered mask: the last hour of D (real-time not yet public) marked visible
    s = dd.sample(D)
    s["m_rt"] = s["m_rt"].copy()
    s["m_rt"][-1, 0] = True
    with pytest.raises(T.LookaheadError):
        dd.assert_sample_no_lookahead(D, s)
    s = dd.sample(D)
    s["m_gda"] = s["m_gda"].copy()
    s["m_gda"][:, :] = True                       # includes hours of points that never published
    with pytest.raises(T.LookaheadError):
        dd.assert_sample_no_lookahead(D, s)


class _Exploding(dict):
    def __getitem__(self, k):
        raise AssertionError("read before the lock check")

    def __contains__(self, k):
        raise AssertionError("read before the lock check")

    def __bool__(self):
        raise AssertionError("read before the lock check")


def test_lock_refuses_holdout_before_reading():
    with pytest.raises(lock.HoldoutLocked):
        DD.DeepData(dt.date(2024, 1, 1), tables=_Exploding(), first_day=dt.date(2023, 12, 1))


def test_reads_stop_at_the_boundary(tmp_path):
    """Tables that (synthetically) run into 2024: a 2023 load never holds a 2024 hour."""
    frames = DD.write_synthetic_parquet(tmp_path, dt.date(2023, 12, 20), dt.date(2024, 1, 3), n_points=4, seed=2)
    dd = DD.DeepData(dt.date(2023, 12, 31), root=tmp_path, first_day=dt.date(2023, 12, 20))
    last_hour = pd.Timestamp(dd.origin_ns + (dd.T - 1) * DD.HOUR_NS, tz="UTC").tz_convert(TZ)
    assert last_hour == pd.Timestamp("2023-12-31 23:00", tz=TZ)
    assert DD.local_dates(dd.lf_rows["target_hour"]).max() <= np.datetime64("2023-12-31")
    assert len(frames["prices_zone"]) > 0


@pytest.mark.skipif(not (PARQUET / "prices_zone.parquet").exists(), reason="real tables not on this machine")
def test_real_tables_match_timing():
    """Real build-year days (2023 only; nothing at or after 2024-01-01 is read)."""
    first, last = dt.date(2023, 2, 1), dt.date(2023, 4, 30)
    dd = DD.DeepData(last, first_day=first)

    class Clipped(T.Store):
        def gen_window(self, lo, hi):
            hi = min(hi, DD.local_midnight(last + dt.timedelta(days=1)))
            return super().gen_window(lo, hi)
    lo, hi = DD.local_midnight(first - dt.timedelta(days=12)), DD.local_midnight(last + dt.timedelta(days=1))
    pz = DD._read(PARQUET / "prices_zone.parquet", None, "delivery_hour", lo, hi)
    lf = DD._read(PARQUET / "load_forecast.parquet", None, "target_hour", lo, hi)
    lock.assert_build_only(pz["delivery_hour"])
    lock.assert_build_only(lf["target_hour"])
    tz = f"datetime64[ns, {TZ}]"
    store = Clipped(tables={"prices_zone": pz, "load_forecast": lf,
                            "outages": pd.DataFrame({"snapshot_date": pd.to_datetime(pd.Series([], dtype="datetime64[ns]")),
                                                     "published_at": pd.Series([], dtype=tz)}),
                            "weather_gfs": pd.DataFrame({"target_hour": pd.Series([], dtype=tz),
                                                         "temperature_2m_c": pd.Series([], dtype=float),
                                                         "published_at": pd.Series([], dtype=tz)})})
    days = sorted(random.Random(20261006).sample([first + dt.timedelta(days=i) for i in range(8, (last - first).days)], 8))
    days.append(dt.date(2023, 3, 11))                       # bid day before the spring-forward day
    for D in days:
        mine, (a, b) = _mine(dd, D)
        _compare(mine, _theirs(store, D, a, b))
