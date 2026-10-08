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
    # cost audit (commit c3adae3): RS1 + FERC on every cleared MWh, + the uplift bound on supply legs
    assert fees.fee(2023) == pytest.approx(0.1066 + 0.017) and fees.supply_fee(2023) == pytest.approx(0.1236 + 0.026)
    assert fees.RS1[2021] == 0.0757 and fees.RS1[2025] == 0.1666 and fees.RS1[2026] == 0.1919
    assert set(fees.RATES) == set(fees.SUPPLY_RATES) == set(range(2020, 2027))
    f = fees.fee_for(pd.Series(pd.to_datetime(["2020-03-01", "2023-12-31"])))
    assert f.tolist() == pytest.approx([0.0962, 0.1236])
    c = fees.cost_for(pd.Series(pd.to_datetime(["2023-05-01"] * 3)), [-1, 1, 0])
    assert c.tolist() == pytest.approx([0.1496, 0.1236, 0.1236])
    assert list(fees.fee_for(pd.Series(pd.to_datetime(["2020-03-01"])), 0.5)) == [0.5]
    assert list(fees.cost_for(pd.Series(pd.to_datetime(["2020-03-01"])), [-1], 1.0)) == [1.0]   # stress replaces all
    # 23:00 on 31 Dec 2022 local is 2023 in UTC: the local year counts
    assert fees.fee_for(pd.Series([pd.Timestamp("2022-12-31 23:00", tz=TZ)]))[0] == fees.fee(2022)


def test_money_signs_and_thresholds():
    fs, fl = fees.supply_fee(2023), fees.fee(2023)
    p, pr = tiny([-10, 5, 3, -2, np.nan], s365=[-1, 2, np.nan, 0, -1],
                 preds=[-fs, -fs + 1e-9, 0.5, fl, -9])
    b = S.baseline(p)
    assert list(b["pos"]) == [-1, 1, 0, 0, 0]                          # NaN gap: not traded
    assert b["pnl"].iloc[0] == pytest.approx(10 - fs)                   # supply earns -gap - supply cost
    assert b["pnl"].iloc[1] == pytest.approx(5 - fl)                    # load earns gap - load cost
    a = S.idea_A_v1(p, pr)
    assert list(a["pos"]) == [-1, 0, 0, 0, 0]                           # v1: supply at pred <= -supply cost only
    t = S.two_sided(p, pr)
    assert list(t["pos"]) == [-1, 0, 1, 1, 0]                          # 0.5 >= fee: load
    assert t["pnl"].iloc[3] == pytest.approx(-2 - fl)                   # load at pred >= load cost
    m = SC.money(a)
    assert m["break_even_fee"] == pytest.approx(10.0) and m["profit_per_mwh"] == pytest.approx(10 - fs)


def test_pairs():
    pair = fees.fee(2023) + fees.supply_fee(2023)                       # one load leg + one supply leg
    p1, _ = tiny([4, -3, 1], zone="N.Y.C.")
    p2, _ = tiny([1, 2, 1], zone="WEST")
    p = pd.concat([p1, p2], ignore_index=True)
    pred = pd.Series([1.0, -1.0, 0.1, 0.0, 0.0, 0.0], index=p.index)
    c = S.idea_C(p, pred, [("N.Y.C.", "WEST")])
    assert list(c["pos"]) == [1, -1, 0]                                 # 1 >= pair cost; -1 <= -pair cost; 0.1 below
    assert c["pnl"].iloc[0] == pytest.approx((4 - 1) - pair)
    assert c["pnl"].iloc[1] == pytest.approx(-(-3 - 2) - pair)
    assert list(c["mwh"]) == [2, 2, 0]
    chosen, table = S.choose_pairs(p, pred)
    assert chosen == [("N.Y.C.", "WEST")] and len(table) == 55


def test_ledger_refuses_holdout_rows():
    p, pr = tiny([1.0, 2.0], day="2024-01-01")
    with pytest.raises(lock.HoldoutLocked):
        S.idea_A(p, pr, 0.05)
    with pytest.raises(lock.HoldoutLocked):
        S.always_supply(p)


def test_idea_A_supplies_except_spike_risk_and_always_supply():
    fee = fees.supply_fee(2023)
    p, _ = tiny([-10, 80, 3, -2, np.nan])
    ps = pd.Series([0.01, 0.30, 0.05, np.nan, 0.0], index=p.index)
    a = S.idea_A(p, ps, 0.05)
    assert list(a["pos"]) == [-1, 0, -1, 0, 0]          # 0.30 > p*: sit out; 0.05 is not above p*; NaN p or gap: none
    assert a["pnl"].iloc[2] == pytest.approx(-3 - fee)
    al = S.always_supply(p)
    assert list(al["pos"]) == [-1, -1, -1, -1, 0] and al["pnl"].iloc[1] == pytest.approx(-80 - fee)
    assert list(S.idea_A(p, ps, 0.05, fee_override=1.0)["pos"]) == list(a["pos"])   # the fee moves money, not the rule


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
    # a word other than inconclusive needs the Holm rejection (audit patch at apply_holm)
    rng = np.random.default_rng(0)
    res = {k: {"boot_means": rng.normal(m, 1, 10_000)} for k, m in (("B", 5.0), ("C", 2.2), ("D", 0.1))}
    for r in res.values():
        r["p_two_sided"], r["se_daily_diff"] = SC.p_two_sided(r["boot_means"]), 1.0
    SC.apply_holm(res)
    for r in res.values():
        assert (r["verdict"] in ("pays", "doesn't pay")) == r["holm"]["rejected"]
    assert res["B"]["verdict"] == "pays" and res["D"]["verdict"] == "inconclusive"
    assert SC.verdict(0.1, 2) == "pays" and SC.verdict(-2, -0.1) == "doesn't pay"
    assert SC.verdict(-1, 1) == "inconclusive" and SC.verdict(0.1, 2, reached=False) == "inconclusive"
    assert SC.mde(1.0, 0.05) == pytest.approx(1.96 + 0.8416, abs=1e-3)


def test_deep_vs_gbm_words():
    days = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=200))
    idx = SC.stationary_indices(len(days), 2000)
    def led(vals, mwh):
        return pd.DataFrame({"delivery_date": days, "pnl": vals, "mwh": mwh, "gross": vals + mwh * 0.1236,
                             "zone": "W", "absgap": 0.0})              # costs paid: 12.36 USD a day = the margin
    rng = np.random.default_rng(3)
    g = led(rng.normal(0, 1, 200), 100.0)
    same = led(g["pnl"] + rng.normal(0, 0.01, 200), 100.0)
    assert SC.deep_vs_gbm(same, g, days, idx)["verdict"] == "equivalent"
    better = led(g["pnl"] + 50 + rng.normal(0, 1, 200), 100.0)
    assert SC.deep_vs_gbm(better, g, days, idx)["verdict"] == "better"
    tiny_better = led(g["pnl"] + 0.5 + rng.normal(0, 0.01, 200), 100.0)   # inside the margin but above zero
    assert SC.deep_vs_gbm(tiny_better, g, days, idx)["verdict"] == "better"     # better/worse first
    wide = led(g["pnl"] + rng.normal(0, 400, 200), 100.0)                 # interval much wider than +-12.4
    assert SC.deep_vs_gbm(wide, g, days, idx)["verdict"] == "inconclusive"


def test_context_figures():
    import context as CX
    days = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=100))
    idx = SC.stationary_indices(len(days), 2000)
    v = np.r_[np.full(50, 2.0), np.full(50, 0.0)]
    led = pd.DataFrame({"delivery_date": days[v > 0], "pnl": v[v > 0], "mwh": 1.0, "gross": v[v > 0], "zone": "W",
                        "absgap": 0.0})                             # 50 trading days, 50 days without a trade
    c = CX.own(led, days, idx)
    d = pd.Series(v)
    assert c["mean_daily_net_usd"] == pytest.approx(1.0) and c["days_with_trades"] == 50
    assert c["sharpe_annualised"] == pytest.approx(d.mean() / d.std(ddof=1) * np.sqrt(365))
    assert c["annual_net_usd"] == pytest.approx(365.0) and c["return_on_bankroll"] == pytest.approx(365 / 500_000)
    assert c["max_drawdown_usd"] == 0 and c["return_over_max_drawdown"] is None   # never fell
    assert CX.max_drawdown(pd.Series([5.0, -3, -4, 6, -10, 2])) == pytest.approx(11.0)   # peak 5 to trough -6
    assert CX.max_drawdown(pd.Series([-2.0, 1])) == pytest.approx(2.0)                   # the peak starts at zero
    assert CX.fill_bankroll({"A": c}, 1000.0)["A"]["return_on_bankroll"] == pytest.approx(0.365)


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


def test_choice_guard_refuses_outcomes_not_public_at_the_first_bid():
    p = _synth_panel()                                     # delivery days from 2022-06-01
    start = dt.date(2022, 9, 1)
    ok = W.choice_rows(p, start)
    assert ok["delivery_date"].max() == pd.Timestamp("2022-08-30")
    W.check_choice_rows(ok, start, "fine")
    with pytest.raises(AssertionError, match="not public"):
        W.check_choice_rows(p[p["delivery_date"] <= "2022-08-31"], start, "day before the period")
    late = ok.copy()
    late.loc[late.index[-1], "label_published_at"] = decision_time(dt.date(2022, 8, 31)) + pd.Timedelta(hours=1)
    with pytest.raises(AssertionError, match="not public"):
        W.check_choice_rows(late, start, "revised after the deadline")


def test_rehearsal_tuning_ignores_the_last_day_before_the_scored_period(monkeypatch):
    import rehearsal as R
    p = _synth_panel(n_days=200, start="2022-06-01")
    p["gap_365d_h"] = -0.5
    p["bid_date"] = p["delivery_date"] - pd.Timedelta(days=1)
    monkeypatch.setattr(R, "TUNE_VALID", (dt.date(2022, 10, 1), dt.date(2022, 11, 30)))
    monkeypatch.setattr(R, "REH", (dt.date(2022, 12, 1), dt.date(2022, 12, 17)))
    monkeypatch.setattr(gbm, "GRID", gbm.GRID[:2])
    monkeypatch.setattr(gbm, "N_ROUNDS", 20)
    a = R.tune(p, ["x", "zone_code"], "A", None)[0]
    q = p.copy()
    last = q["delivery_date"] == pd.Timestamp("2022-11-30")          # not public at 05:00 on 30 Nov
    q.loc[last, "gap"] = -1e4                                          # a huge outcome that would sway the choice
    b = R.tune(q, ["x", "zone_code"], "A", None)[0]
    assert [c["mean_daily_diff"] for c in a["configs"]] == [c["mean_daily_diff"] for c in b["configs"]]
    assert W.CHOICE_LOG[-1]["last_delivery_day"] == "2022-11-29"
