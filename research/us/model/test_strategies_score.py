"""Money, rules, scoring, walk-forward and the LightGBM wrapper, on small hand-made or synthetic data."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import fees
import gbm
import lock
import score as SC
import strategies as S
import walkforward as W
from common import TZ, decision_time


def tiny(gaps, zone="WEST", day="2023-05-02", s365=None, preds=None):
    n = len(gaps)
    dh = pd.date_range(pd.Timestamp(day, tz=TZ), periods=n, freq="h")
    p = pd.DataFrame({"delivery_date": pd.Timestamp(day), "delivery_hour": dh, "zone": zone,
                      "gap": np.asarray(gaps, float),
                      "gap_365d_h": np.asarray(s365 if s365 is not None else [0.0] * n, float)})
    pr = pd.Series(np.asarray(preds if preds is not None else [0.0] * n, float), index=p.index)
    return p, pr


def test_fees():
    assert fees.fee(2023) == 0.1066 and fees.fee(2024) == 0.1333 and fees.fee(2026) == 0.1919
    assert set(fees.RATES) == set(range(2020, 2027))
    f = fees.fee_for(pd.Series(pd.to_datetime(["2020-03-01", "2023-12-31"])))
    assert list(f) == [0.0862, 0.1066]
    assert list(fees.fee_for(pd.Series(pd.to_datetime(["2020-03-01"])), 0.5)) == [0.5]
    # 23:00 on 31 Dec 2022 local is 2023 in UTC: the local year counts
    assert fees.fee_for(pd.Series([pd.Timestamp("2022-12-31 23:00", tz=TZ)]))[0] == fees.fee(2022)


def test_money_signs_and_thresholds():
    fee = fees.fee(2023)
    p, pr = tiny([-10, 5, 3, -2, np.nan], s365=[-1, 2, np.nan, 0, -1],
                 preds=[-fee, -fee + 1e-9, 0.5, fee, -9])
    b = S.baseline(p)
    assert list(b["pos"]) == [-1, 1, 0, 0, 0]                          # NaN gap: not traded
    assert b["pnl"].iloc[0] == pytest.approx(10 - fee)                  # supply earns -gap - fee
    assert b["pnl"].iloc[1] == pytest.approx(5 - fee)                   # load earns gap - fee
    a = S.idea_A(p, pr)
    assert list(a["pos"]) == [-1, 0, 0, 0, 0]                           # supply at pred <= -fee only
    t = S.two_sided(p, pr)
    assert list(t["pos"]) == [-1, 0, 1, 1, 0]                          # 0.5 >= fee: load
    assert t["pnl"].iloc[3] == pytest.approx(-2 - fee)
    m = SC.money(a)
    assert m["break_even_fee"] == pytest.approx(10.0) and m["profit_per_mwh"] == pytest.approx(10 - fee)


def test_pairs():
    fee = fees.fee(2023)
    p1, _ = tiny([4, -3, 1], zone="N.Y.C.")
    p2, _ = tiny([1, 2, 1], zone="WEST")
    p = pd.concat([p1, p2], ignore_index=True)
    pred = pd.Series([1.0, -1.0, 0.1, 0.0, 0.0, 0.0], index=p.index)
    c = S.idea_C(p, pred, [("N.Y.C.", "WEST")])
    assert list(c["pos"]) == [1, -1, 0]                                 # 1 >= 2 fee; -1 <= -2 fee; 0.1 < 2 fee
    assert c["pnl"].iloc[0] == pytest.approx((4 - 1) - 2 * fee)
    assert c["pnl"].iloc[1] == pytest.approx(-(-3 - 2) - 2 * fee)
    assert list(c["mwh"]) == [2, 2, 0]
    chosen, table = S.choose_pairs(p, pred)
    assert chosen == [("N.Y.C.", "WEST")] and len(table) == 55


def test_ledger_refuses_holdout_rows():
    p, pr = tiny([1.0, 2.0], day="2024-01-01")
    with pytest.raises(lock.HoldoutLocked):
        S.idea_A(p, pr)


def test_stationary_bootstrap():
    idx = SC.stationary_indices(365, n_boot=2000, seed=1)
    assert idx.shape == (2000, 365) and idx.min() >= 0 and idx.max() < 365
    breaks = (np.diff(idx, axis=1) != 1) & ~((idx[:, :-1] == 364) & (idx[:, 1:] == 0))
    assert 5.5 < 1 / breaks.mean() < 8.5                                # mean block about 7 days
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 365)
    bm = SC.boot_means(x, idx)
    lo, hi = SC.interval(bm, 0.95)
    assert lo < x.mean() < hi and lo > 0.7 and hi < 1.3
    assert SC.p_two_sided(bm) < 0.01
    assert np.array_equal(SC.stationary_indices(50, 10), SC.stationary_indices(50, 10))   # fixed seed


def test_holm_and_words():
    h = SC.holm({"B": 0.001, "C": 0.04, "D": 0.02})
    assert h["B"]["rank"] == 1 and h["B"]["alpha"] == pytest.approx(0.05 / 3) and h["B"]["rejected"]
    assert h["D"]["alpha"] == pytest.approx(0.025) and h["D"]["rejected"]
    assert h["C"]["alpha"] == pytest.approx(0.05) and h["C"]["rejected"]
    h = SC.holm({"B": 0.03, "C": 0.001, "D": 0.04})
    assert h["C"]["rejected"] and not h["B"]["rejected"] and not h["D"]["reached"]
    assert SC.verdict(0.1, 2) == "pays" and SC.verdict(-2, -0.1) == "doesn't pay"
    assert SC.verdict(-1, 1) == "inconclusive" and SC.verdict(0.1, 2, reached=False) == "inconclusive"
    assert SC.mde(1.0, 0.05) == pytest.approx(1.96 + 0.8416, abs=1e-3)


def test_deep_vs_gbm_words():
    days = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=200))
    idx = SC.stationary_indices(len(days), 2000)
    def led(vals, mwh):
        return pd.DataFrame({"delivery_date": days, "pnl": vals, "mwh": mwh, "gross": vals, "zone": "W",
                             "absgap": 0.0})
    rng = np.random.default_rng(3)
    g = led(rng.normal(0, 1, 200), 100.0)
    same = led(g["pnl"] + rng.normal(0, 0.01, 200), 100.0)
    assert SC.deep_vs_gbm(same, g, days, idx)["verdict"] == "equivalent"
    better = led(g["pnl"] + 50 + rng.normal(0, 1, 200), 100.0)
    assert SC.deep_vs_gbm(better, g, days, idx)["verdict"] == "better"


def _synth_panel(n_days=200, start="2022-06-01"):
    rng = np.random.default_rng(5)
    d = pd.date_range(start, periods=n_days)
    rows = []
    for z in ("WEST", "N.Y.C."):
        dh = pd.date_range(pd.Timestamp(start, tz=TZ), periods=n_days * 24, freq="h")
        x = rng.normal(0, 1, len(dh))
        rows.append(pd.DataFrame({"delivery_hour": dh, "zone": z, "x": x, "zone_code": int(z == "WEST"),
                                  "gap": 3 * x + rng.normal(0, 1, len(dh))}))
    p = pd.concat(rows, ignore_index=True)
    p["delivery_date"] = p["delivery_hour"].dt.tz_localize(None).dt.normalize()
    p["label_published_at"] = p["delivery_hour"] + pd.Timedelta(hours=1, minutes=15)
    return p


def test_train_mask_two_day_gap_and_publication():
    p = _synth_panel()
    ms = dt.date(2022, 9, 1)
    m = W.train_mask(p, ms)
    assert p.loc[m, "delivery_date"].max() == pd.Timestamp("2022-08-30")
    late = p["delivery_date"] == pd.Timestamp("2022-08-20")
    p.loc[late, "label_published_at"] = decision_time(dt.date(2022, 8, 31)) + pd.Timedelta(minutes=1)
    m2 = W.train_mask(p, ms)
    assert not m2[late].any() and m2.sum() == m.sum() - late.sum()


def test_walk_forward_and_gbm():
    p = _synth_panel()
    seen = []

    class Spy(gbm.GBMModel):
        def fit(self, tr):
            seen.append((tr["delivery_date"].max(), tr["delivery_date"].min()))
            super().fit(tr)
    pred, fits = W.walk_forward(p, lambda: Spy(["x", "zone_code"], gbm.GRID[0], n_rounds=50),
                                dt.date(2022, 10, 1), dt.date(2022, 11, 30), log=lambda *_: None)
    assert [s[0] for s in seen] == [pd.Timestamp("2022-09-29"), pd.Timestamp("2022-10-30")]
    test = p[(p["delivery_date"] >= "2022-10-01") & (p["delivery_date"] <= "2022-11-30")]
    assert pred.index.equals(test.index.sort_values())
    assert np.corrcoef(pred, test.loc[pred.index, "gap"])[0, 1] > 0.8     # it learns the signal
    a = gbm.GBMModel(["x", "zone_code"], gbm.GRID[1], n_rounds=50)
    b = gbm.GBMModel(["x", "zone_code"], gbm.GRID[1], n_rounds=50)
    tr = p[p["delivery_date"] < "2022-10-01"]
    a.fit(tr), b.fit(tr)
    assert np.array_equal(a.predict(test), b.predict(test))                # deterministic
    with pytest.raises(lock.HoldoutLocked):
        a.predict(_synth_panel(10, "2023-12-28"))
    assert len(gbm.GRID) == 8
