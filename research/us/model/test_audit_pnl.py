"""P&L audit tests (6 Oct 2026). One concrete test per checklist item (the item number leads each test
name), run on hand-made rows and on conftest.py's synthetic NYISO tables, plus the second
implementation (audit_pnl.py) agreeing with the official code on identical synthetic inputs.
Nothing here reads real data. The one xfail documents a bug in score.apply_holm (see its reason)."""
from __future__ import annotations

import datetime as dt
import re
import shutil

import numpy as np
import pandas as pd
import pytest

import audit_pnl as AU
import fees
import gbm
import lock
import panel as P
import rehearsal as R
import score as SC
import strategies as S
import walkforward as W
from common import TZ, ZONES, decision_time

F22, F23 = fees.fee(2022), fees.fee(2023)


def rows(spec, preds=None):
    """spec: list of (zone, local hour start 'YYYY-MM-DD HH:MM', gap, trailing signal)."""
    dh = pd.DatetimeIndex([pd.Timestamp(h, tz=TZ) for _, h, _, _ in spec])
    p = pd.DataFrame({"zone": [z for z, *_ in spec], "delivery_hour": dh,
                      "delivery_date": dh.tz_localize(None).normalize(),
                      "gap": [g for *_, g, _ in spec], "gap_365d_h": [s for *_, s in spec]})
    p["gap"], p["gap_365d_h"] = p["gap"].astype(float), p["gap_365d_h"].astype(float)
    pr = pd.Series(np.asarray(preds if preds is not None else np.zeros(len(p)), float), index=p.index)
    return p, pr


# ================================================================ 1. sign conventions

def test_1_supply_earns_minus_gap_load_earns_gap_none_zero():
    p, pr = rows([("WEST", "2023-05-02 10:00", -10, -1), ("WEST", "2023-05-02 11:00", 7, 1),
                  ("WEST", "2023-05-02 12:00", 3, 0)], preds=[-5, 5, 0])
    for led in (S.baseline(p), S.two_sided(p, pr)):
        assert list(led["pos"]) == [-1, 1, 0]
        assert led["pnl"].tolist() == pytest.approx([10 - F23, 7 - F23, 0.0])   # rt - da = gap
        assert led["mwh"].tolist() == [1, 1, 0]
    a = S.idea_A(p, pr)                                                          # A never buys load
    assert list(a["pos"]) == [-1, 0, 0] and a["pnl"].tolist() == pytest.approx([10 - F23, 0, 0])


def test_1_pairs_are_two_legs_load_in_i_supply_in_j():
    p, _ = rows([("N.Y.C.", "2023-05-02 10:00", 4, 0), ("WEST", "2023-05-02 10:00", 1, 0),
                 ("N.Y.C.", "2023-05-02 11:00", -3, 0), ("WEST", "2023-05-02 11:00", 2, 0)])
    pred = pd.Series([1.0, 0.0, -1.0, 0.0], index=p.index)        # +1 then -1: load NYC, then supply NYC
    c = S.idea_C(p, pred, [("N.Y.C.", "WEST")])
    leg = lambda pos, g: pos * g - F23                              # one leg, one fee
    assert c["pnl"].tolist() == pytest.approx([leg(1, 4) + leg(-1, 1), leg(-1, -3) + leg(1, 2)])
    assert c["mwh"].tolist() == [2, 2]
    au = AU.pair_ledger(p.assign(ldate=p["delivery_date"]), pred.to_numpy(), [("N.Y.C.", "WEST")])
    assert au["pnl"].tolist() == pytest.approx(c["pnl"].tolist())


# ================================================================ 2. fees

def test_2_fee_is_the_delivery_years_rate_per_cleared_mwh():
    p, pr = rows([("WEST", "2022-12-31 23:00", -4, -1), ("WEST", "2023-01-01 00:00", -4, -1)], preds=[-5, -5])
    a = S.idea_A(p, pr)                                    # the second row is bid on 31 Dec 2022: delivery year counts
    assert a["pnl"].tolist() == pytest.approx([4 - F22, 4 - F23])


def test_2_stress_cost_replaces_the_fee_and_is_charged_per_leg():
    p, _ = rows([("N.Y.C.", "2023-05-02 10:00", 4, 1), ("WEST", "2023-05-02 10:00", 1, -1)])
    pred = pd.Series([3.0, 0.0], index=p.index)
    for f in fees.STRESS:
        L = R.ledgers(p, {"base": pred}, ["N.Y.C.|WEST"], fee=f)
        assert L["baseline"]["pnl"].tolist() == pytest.approx([4 - f, -1 - f])         # not 4 - 0.1066 - f
        assert L["C"]["pnl"].iloc[0] == pytest.approx((4 - 1) - 2 * f)
    assert fees.STRESS == (0.50, 1.00)


def test_2_fee_only_on_traded_hours():
    p, pr = rows([("WEST", "2023-05-02 10:00", -4, 0), ("WEST", "2023-05-02 11:00", 9, 0),
                  ("WEST", "2023-05-02 12:00", np.nan, 0)], preds=[-1, 0, -1])
    a = S.idea_A(p, pr)
    charged = (a["gross"] - a["pnl"]).to_numpy()
    assert charged.tolist() == pytest.approx([F23, 0, 0])          # untraded and unsettled hours pay nothing


# ================================================================ 3. days in New York time

def test_3_dst_days_have_23_and_25_hours_and_both_0100_hours_count(synth_root):
    st = P.LockedStore(synth_root)
    for day, n in ((dt.date(2023, 3, 12), 23), (dt.date(2023, 11, 5), 25), (dt.date(2023, 6, 1), 24)):
        lab = P._labels(st, day, day)
        assert (lab.groupby("zone").size() == n).all()
    lab = P._labels(st, dt.date(2023, 11, 5), dt.date(2023, 11, 5))
    one = lab[(lab["zone"] == "WEST") & (lab["delivery_hour"].dt.hour == 1)]
    assert sorted(str(t.utcoffset()) for t in one["delivery_hour"]) == ["-1 day, 19:00:00", "-1 day, 20:00:00"]
    led = S.ledger(lab, -np.ones(len(lab)), np.full(len(lab), F23))
    days = pd.DatetimeIndex([pd.Timestamp("2023-11-05")])
    assert SC.daily(led, days).iloc[0] == pytest.approx(float((-lab["gap"] - F23).sum()))   # 275 zone-hours


def test_3_days_without_trades_count_as_zero():
    p, pr = rows([("WEST", "2023-05-01 10:00", -10, 0), ("WEST", "2023-05-02 10:00", 5, 0),
                  ("WEST", "2023-05-03 10:00", 5, 0)], preds=[-1, 0, 0])
    a, base = S.idea_A(p, pr), S.baseline(p)                      # baseline signal 0: never trades
    days = pd.DatetimeIndex(sorted(p["delivery_date"].unique()))
    r = SC.summarize("A", a, base, days, SC.stationary_indices(3, 200))
    assert r["n_days"] == 3 and r["mean_daily_idea"] == pytest.approx((10 - F23) / 3)
    traded_only = a[a["mwh"] > 0].groupby("delivery_date")["pnl"].sum().mean()
    assert traded_only == pytest.approx(10 - F23)                 # what dropping the empty days would report
    empty = S.idea_C(p, pr, [])                                   # no pairs: still one zero per day
    assert SC.daily(empty, days).tolist() == [0.0, 0.0, 0.0]


# ================================================================ 4. information by 05:00 on D

def _store_with(synth_root, tmp_path, changes):
    """A copy of the synthetic store with WEST prices changed: changes = [(local hour, column, add)]."""
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    for n in ("load_forecast", "outages", "weather_gfs"):
        shutil.copy(synth_root / f"{n}.parquet", root / f"{n}.parquet")
    pz = pd.read_parquet(synth_root / "prices_zone.parquet")
    for hour, col, add in changes:
        m = (pz["zone"] == "WEST") & (pz["delivery_hour"] == pd.Timestamp(hour, tz=TZ))
        assert m.sum() == 1
        pz.loc[m, col] += add
    pz.to_parquet(root / "prices_zone.parquet")
    return P.LockedStore(root)


def test_4_no_feature_moves_when_data_published_after_0500_changes(synth_root, tmp_path):
    d1 = dt.date(2023, 12, 15)                  # bid day 14 Dec, decision 05:00 EST; 365 days back = 14 Dec 2022 05:00
    ref = P.build_panel(d1, d1, store=P.LockedStore(synth_root))
    late = [(f"2023-12-15 {h:02d}:00", c, add) for h in range(24)
            for c, add in (("rt_lbmp", 1000.0), ("da_lbmp", 400.0))]                     # the delivery day itself
    late += [("2023-12-14 04:00", "rt_lbmp", 1000.0),             # ends 05:00, published 05:15
             ("2022-12-14 04:00", "rt_lbmp", 1000.0)]             # one hour before the 365-day window opens
    got = P.build_panel(d1, d1, store=_store_with(synth_root, tmp_path, late))
    pd.testing.assert_frame_equal(ref[P.BASE_FEATURES], got[P.BASE_FEATURES])
    assert not np.allclose(ref["gap"], got["gap"])                 # the label did move: the perturbation was real


def test_4_trailing_window_ends_at_the_decision_and_starts_365_days_before(synth_root, tmp_path):
    d1 = dt.date(2023, 12, 15)
    ref = P.build_panel(d1, d1, store=P.LockedStore(synth_root))
    inside = [("2023-12-14 03:00", "rt_lbmp", 1000.0),             # ends 04:00, published 04:15: used
              ("2022-12-14 05:00", "rt_lbmp", 1000.0)]             # exactly 05:00 on D minus 365 days: used
    got = P.build_panel(d1, d1, store=_store_with(synth_root, tmp_path, inside))
    w = lambda x, h: x[(x["zone"] == "WEST") & (x["hour"] == h)].iloc[0]
    for h in (3, 5):
        r, g = w(ref, h), w(got, h)
        assert g["gap_365d_h"] - r["gap_365d_h"] == pytest.approx(1000.0 / r["n_365d_h"])
    assert w(got, 4)["gap_365d_h"] == w(ref, 4)["gap_365d_h"]


def test_4_audit_signal_equals_panel_signal_on_dst_and_revised_days(synth_root):
    st = P.LockedStore(synth_root)
    H = AU.hours_frame(AU.read_pre2024(synth_root / "prices_zone.parquet"))
    for lo, hi in ((dt.date(2023, 3, 10), dt.date(2023, 3, 13)), (dt.date(2023, 11, 3), dt.date(2023, 11, 6))):
        pan = P.build_panel(lo, hi, store=st)
        days = pd.date_range(lo, hi)
        sig = AU.trailing_signal(H, days)
        m = H.assign(sig=sig)[["delivery_hour", "zone", "sig"]].merge(pan, on=["delivery_hour", "zone"],
                                                                      validate="one_to_one")
        assert len(m) == len(pan)
        np.testing.assert_allclose(m["sig"], m["gap_365d_h"], rtol=0, atol=1e-9)


def _wf_panel(start="2022-10-01", n_days=160):
    rng = np.random.default_rng(5)
    dh = pd.date_range(pd.Timestamp(start, tz=TZ), periods=n_days * 24, freq="h")
    out = []
    for z in ("WEST", "N.Y.C."):
        x = rng.normal(0, 1, len(dh))
        out.append(pd.DataFrame({"delivery_hour": dh, "zone": z, "x": x, "zone_code": int(z == "WEST"),
                                 "gap": 3 * x + rng.normal(0, 1, len(dh)), "gap_365d_h": rng.normal(0, 1, len(dh))}))
    p = pd.concat(out, ignore_index=True)
    p["delivery_date"] = p["delivery_hour"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    p["label_published_at"] = p["delivery_hour"] + pd.Timedelta(hours=1, minutes=15)
    return p


def test_4_and_8_refits_learn_only_what_was_public_two_days_before_the_month():
    p = _wf_panel()
    late = (p["delivery_date"] == pd.Timestamp("2023-01-20")) & (p["zone"] == "WEST")
    p.loc[late, "label_published_at"] = decision_time(dt.date(2023, 1, 31)) + pd.Timedelta(minutes=1)
    seen = []

    class Spy(gbm.GBMModel):
        def fit(self, tr):
            seen.append(("fit", tr["delivery_date"].max(), int((tr.index.isin(p.index[late])).sum())))
            super().fit(tr)

        def predict(self, x):
            seen.append(("predict", x["delivery_date"].min(), x["delivery_date"].max()))
            return super().predict(x)
    W.walk_forward(p, lambda: Spy(["x", "zone_code"], gbm.GRID[0], n_rounds=20),
                   dt.date(2023, 1, 1), dt.date(2023, 3, 10), log=lambda *_: None)
    fits = [s for s in seen if s[0] == "fit"]
    assert [f[1] for f in fits] == [pd.Timestamp("2022-12-30"), pd.Timestamp("2023-01-30"), pd.Timestamp("2023-02-27")]
    assert fits[1][2] == 0 and fits[2][2] == late.sum()          # a label published after 05:00 waits for the next refit
    preds = [s for s in seen if s[0] == "predict"]
    assert [(a.month, b.month) for _, a, b in preds] == [(1, 1), (2, 2), (3, 3)]
    assert W.TRAIN_GAP_DAYS == 2


# ================================================================ 5. bootstrap, Holm, words

def test_5_stationary_bootstrap_over_days_mean_block_7_fixed_seed():
    assert (SC.N_BOOT, SC.MEAN_BLOCK) == (10_000, 7) and isinstance(SC.SEED, int)
    idx = SC.stationary_indices(365)
    assert idx.shape == (10_000, 365) and np.array_equal(idx, SC.stationary_indices(365))
    step = (idx[:, 1:] - idx[:, :-1]) % 365
    assert 6.8 < 1 / np.mean(step != 1) < 7.2                     # breaks (incl. a restart at the next day) 1 in 7
    # the series resampled is the daily one: hours are never resampled on their own
    p, pr = rows([("WEST", f"2023-05-0{d} {h:02d}:00", -1.0 * d, 0) for d in (1, 2) for h in range(24)],
                 preds=[-1] * 48)
    days = pd.DatetimeIndex(sorted(p["delivery_date"].unique()))
    small = SC.stationary_indices(2, 500)
    r = SC.summarize("A", S.idea_A(p, pr), S.baseline(p), days, small)
    daily = np.array([24 * (1 - F23), 24 * (2 - F23)])
    np.testing.assert_allclose(r["boot_means"], daily[small].mean(axis=1))


def test_5_holm_is_over_B_C_D_only_and_A_stays_at_95():
    rng = np.random.default_rng(1)
    base = {k: rng.normal(m, 1, 10_000) for k, m in (("A", 0.0), ("B", 3.0), ("C", 2.4), ("D", 0.1))}

    def run(a_shift):
        res = {k: {"boot_means": v + (a_shift if k == "A" else 0), "se_daily_diff": 1.0} for k, v in base.items()}
        for r in res.values():
            r["p_two_sided"] = SC.p_two_sided(r["boot_means"])
        SC.apply_holm(res)
        return res
    r1, r2 = run(0.0), run(50.0)                                  # A's p from about 1 to about 0
    assert r1["A"]["interval_level"] == r2["A"]["interval_level"] == 0.95 and "holm" not in r1["A"]
    for k in ("B", "C", "D"):
        assert r1[k]["holm"] == r2[k]["holm"] and r1[k]["verdict"] == r2[k]["verdict"]
    assert sorted(r1[k]["holm"]["alpha"] for k in "BCD") == pytest.approx([0.05 / 3, 0.05 / 2, 0.05])


def test_5_verdict_words_are_objective_6_byte_for_byte():
    text = (lock.STUDY_DIR / "OBJECTIVES.md").read_text()
    m = re.search(r"one word: (.+?)\. Pays", text)
    words = [w.strip() for w in re.split(r",\s*(?:or\s+)?", m.group(1))]
    assert words == ["pays", "doesn't pay", "inconclusive", "not run"] == list(AU.WORDS)
    produced = {SC.verdict(1, 2), SC.verdict(-2, -1), SC.verdict(-1, 1), SC.verdict(1, 2, reached=False)}
    assert produced | {"not run"} == set(words)


@pytest.mark.xfail(strict=True, reason="BUG score.py:192 - a secondary idea can be 'pays' while Holm did not "
                   "reject it (p uses +1/(n+1), the interval does not); the later ideas are then forced to "
                   "inconclusive. Patch: verdict only when info['rejected'].")
def test_5_holm_rejection_and_pays_agree_at_the_boundary():
    rng = np.random.default_rng(0)
    k = 83                                                         # bootstrap means <= 0 out of 10,000
    res = {"B": {"boot_means": np.concatenate([-rng.random(k) - 0.01, rng.random(10_000 - k) + 0.01])},
           "C": {"boot_means": rng.normal(0, 1, 10_000)}, "D": {"boot_means": rng.normal(0, 1, 10_000)}}
    for r in res.values():
        r["p_two_sided"], r["se_daily_diff"] = SC.p_two_sided(r["boot_means"]), 1.0
    SC.apply_holm(res)
    assert res["B"]["holm"]["rank"] == 1
    assert (res["B"]["verdict"] == "pays") == res["B"]["holm"]["rejected"]


# ================================================================ 6. ranking-table numbers

def test_6_break_even_fee_zeroes_the_profit():
    rng = np.random.default_rng(2)
    spec = [("WEST", f"2023-05-02 {h:02d}:00", g, s) for h, g, s in zip(range(24), rng.normal(-1, 5, 24), rng.normal(0, 1, 24))]
    p, _ = rows(spec)
    be = SC.money(S.baseline(p))["break_even_fee"]
    assert S.baseline(p, fee_override=be)["pnl"].sum() == pytest.approx(0.0, abs=1e-9)


def _score_panel(n_days=40, start="2023-05-01", seed=4):
    rng = np.random.default_rng(seed)
    dh = pd.date_range(pd.Timestamp(start, tz=TZ), periods=24 * n_days, freq="h")
    out = []
    for z in ZONES:
        g = rng.normal(-1, 6, len(dh)) + (rng.random(len(dh)) < 0.01) * rng.exponential(200, len(dh))
        out.append(pd.DataFrame({"delivery_hour": dh, "zone": z, "gap": g, "gap_365d_h": rng.normal(-0.5, 1, len(dh))}))
    p = pd.concat(out, ignore_index=True)
    p["delivery_date"] = p["delivery_hour"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    preds = {fs: pd.Series(rng.normal(-0.3, 1, len(p)), index=p.index) for fs in ("base", "weather", "outages")}
    return p, preds


def test_6_extreme_hours_cut_once_over_all_zone_hours_and_applied_to_both():
    p, preds = _score_panel()
    res = R.score_all(p, preds, ["N.Y.C.|WEST"], "gbm")
    cut = np.quantile(p["gap"].abs(), 0.99)
    assert res["extreme_cut_abs_gap_usd"] == pytest.approx(cut)
    a, b = S.idea_A(p, preds["base"]), S.baseline(p)
    w = res["ideas"]["A"]["without_extreme_1pct"]
    assert w["abs_gap_cut_usd"] == pytest.approx(cut)
    assert w["idea_money"]["mwh"] == a.loc[p["gap"].abs() <= cut, "mwh"].sum()
    assert w["baseline_money"]["mwh"] == b.loc[p["gap"].abs() <= cut, "mwh"].sum()


def test_6_worst_month_and_months_positive():
    vals = {"2023-01-15": 5.0, "2023-02-15": -7.0, "2023-03-15": 2.0}
    p, pr = rows([("WEST", f"{d} 10:00", -v, 0) for d, v in vals.items()], preds=[-1, -1, -1])
    days = pd.DatetimeIndex(sorted(p["delivery_date"].unique()))
    r = SC.summarize("A", S.idea_A(p, pr), S.baseline(p), days, SC.stationary_indices(3, 100))
    assert r["worst_month_idea"]["month"] == "2023-02"
    assert r["worst_month_idea"]["pnl_usd"] == pytest.approx(-7 - F23)
    assert r["months_positive_idea"] == 2 and r["months"] == 3


# ================================================================ 7. no annualisation

def test_7_nothing_is_annualised_or_compounded():
    pat = re.compile(r"sharpe|annuali|\*\s*365\b|\*\s*252\b|sqrt\(\s*365|cumprod|compound", re.I)
    for f in ("score.py", "rehearsal.py", "strategies.py", "fees.py"):
        hits = [ln for ln in (lock.MODEL_DIR / f).read_text().splitlines() if pat.search(ln)]
        assert hits == [], (f, hits)
    p, preds = _score_panel(n_days=20)
    res = R.score_all(p, preds, [], "gbm")
    days = pd.DatetimeIndex(sorted(p["delivery_date"].unique()))
    assert res["ideas"]["A"]["mean_daily_idea"] == pytest.approx(S.idea_A(p, preds["base"])["pnl"].sum() / len(days))


# ================================================================ 8. silent inflation

def test_8_a_duplicated_panel_row_stops_the_run(monkeypatch):
    p, _ = _score_panel(n_days=3)
    dup = pd.concat([p, p.iloc[[5]]], ignore_index=True)
    gen = p[["delivery_hour", "zone"]].assign(**{c: 0.0 for c in P.GEN_FEATURES})
    monkeypatch.setattr(R, "_cached", lambda name, build: dup if name == "panel_base" else gen)
    with pytest.raises(pd.errors.MergeError):
        R.load_all(store=None)


def test_8_nan_predictions_never_trade():
    p, _ = rows([("N.Y.C.", "2023-05-02 10:00", 4, 0), ("WEST", "2023-05-02 10:00", 1, 0)])
    nan = pd.Series([np.nan, np.nan], index=p.index)
    assert S.idea_A(p, nan)["mwh"].sum() == 0 and S.two_sided(p, nan)["mwh"].sum() == 0
    assert S.idea_C(p, nan, [("N.Y.C.", "WEST")])["mwh"].sum() == 0
    assert AU.pos_two_sided(np.array([np.nan]), np.array([F23])).tolist() == [0.0]


def test_8_tuning_and_pair_choice_never_see_2023(monkeypatch):
    p = _wf_panel(start="2021-11-01", n_days=480)                 # through Feb 2023
    seen = []

    class Spy(gbm.GBMModel):
        def fit(self, tr):
            seen.append(("fit", tr["delivery_date"].min(), tr["delivery_date"].max()))
            super().fit(tr)

        def predict(self, x):
            seen.append(("predict", x["delivery_date"].min(), x["delivery_date"].max()))
            return super().predict(x)
    monkeypatch.setattr(gbm, "GBMModel", lambda feats, cfg: Spy(feats, cfg, n_rounds=10))
    monkeypatch.setattr(gbm, "GRID", gbm.GRID[:1])
    info, cfg, vpred, va = R.tune(p, ["x", "zone_code"], "A", None)
    assert seen[0][2] == pd.Timestamp("2021-12-30")
    assert seen[1][1:] == (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
    assert va["delivery_date"].max() == pd.Timestamp("2022-12-31") and vpred.index.equals(va.index)


# ================================================================ the second implementation

def test_audit_reads_nothing_from_2024(synth_root):
    df = AU.read_pre2024(synth_root / "prices_zone.parquet")        # the file holds January 2024 rows
    assert df["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz=TZ)
    with pytest.raises(AU.HoldoutError):
        AU.guard(pd.Series([pd.Timestamp("2024-01-01 00:00", tz=TZ)]))


def test_audit_bootstrap_is_stationary_with_mean_block_7():
    idx = AU.boot_index(365, 4000, seed=3)
    assert idx.shape == (4000, 365) and idx.min() >= 0 and idx.max() < 365
    step = (idx[:, 1:] - idx[:, :-1]) % 365
    assert 6.6 < 1 / np.mean(step != 1) < 7.4


def test_audit_matches_official_on_synthetic_panel(synth_root):
    lo, hi = dt.date(2023, 10, 28), dt.date(2023, 11, 12)          # fall-back day and a revised RT day inside
    pan = P.build_panel(lo, hi, store=P.LockedStore(synth_root))
    rng = np.random.default_rng(9)
    preds = {fs: pd.Series(rng.normal(-0.2, 1.5, len(pan)), index=pan.index) for fs in ("base", "weather", "outages")}
    pairs = ["N.Y.C.|WEST", "CAPITL|LONGIL"]
    off = SC.strip(R.score_all(pan, preds, pairs, "gbm"))

    H = AU.hours_frame(AU.read_pre2024(synth_root / "prices_zone.parquet"))
    days = pd.date_range(lo, hi)
    sig = AU.trailing_signal(H, days)
    keep = H["ldate"].isin(days).to_numpy()
    T, sig = H[keep].reset_index(drop=True), sig[keep]
    m = T[["delivery_hour", "zone"]].merge(pan[["delivery_hour", "zone"]].reset_index(), on=["delivery_hour", "zone"],
                                           validate="one_to_one")
    arr = {fs: s.reindex(m["index"]).to_numpy() for fs, s in preds.items()}
    mine = AU.score(T, sig, arr, [tuple(x.split("|")) for x in pairs], days, SC.stationary_indices(len(days)))
    diffs = AU.compare({k: off[k] for k in ("scored_days", "extreme_cut_abs_gap_usd", "baseline", "ideas", "fee_stress")},
                       mine)
    assert diffs == []
    assert len(AU.flat(off["ideas"])) > 100                         # the comparison really covered the fields
