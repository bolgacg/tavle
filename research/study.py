"""The wind-forecast study, as pre-registered in PREREGISTRATION.md.

Run on the training period first (default). The holdout is read once,
with --holdout, after the code is frozen. Results are written to
research/results/*.json and never typed into the page by hand."""
import argparse
import json
import pathlib

import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
OUT = ROOT / "research" / "results"
TRAIN = ("2019-11-01", "2024-01-01")
HOLDOUT = ("2024-01-01", "2099-01-01")


def frame(con, lo, hi):
    return con.execute(f"""
        select area, hour_utc, price_eur, wind_fc_da_mwh, wind_fc_1h_mwh, wind_actual_mwh,
               consumption_mwh, wind_fc_share, wind_err_da_mwh, wind_err_1h_mwh,
               hour((hour_utc at time zone 'UTC') at time zone 'Europe/Copenhagen') as hour_dk,
               month(hour_utc) as month, year(hour_utc) as year
        from forecast_hourly
        where hour_utc >= timestamp '{lo}' and hour_utc < timestamp '{hi}'
          and wind_fc_da_mwh is not null and consumption_mwh > 0
        order by area, hour_utc""").df()


def design(df, with_forecast):
    """Linear model with hour-of-day and month fixed effects, optionally the
    forecast share. Plain least squares; nothing fancier is needed to answer
    'how much does the forecast add'."""
    cols = [np.ones(len(df))]
    for h in range(1, 24):
        cols.append((df.hour_dk == h).astype(float).values)
    for m in range(2, 13):
        cols.append((df.month == m).astype(float).values)
    if with_forecast:
        cols.append(df.wind_fc_share.values)
        cols.append(df.wind_fc_share.values ** 2)
    return np.column_stack(cols)


def fit_predict(train, test, with_forecast):
    Xtr, Xte = design(train, with_forecast), design(test, with_forecast)
    beta, *_ = np.linalg.lstsq(Xtr, train.price_eur.values, rcond=None)
    return Xte @ beta


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def q1(df):
    """Explanatory R2 per zone and year. The pre-registration said hour and
    month fixed effects, out of sample, reported per year. Fitting on other
    years and predicting this one fails for a reason that is itself a
    finding: the price LEVEL is set by gas and by the year (2020 was cheap,
    2022 was the crisis), so a model must be allowed to know the year. The
    split is therefore within each zone-year, alternating weeks: odd weeks
    fit, even weeks scored. The forecast's contribution is the difference
    between the two R2 values."""
    out = {}
    for area in ("DK1", "DK2"):
        d = df[df.area == area].copy()
        d["week"] = d.hour_utc.dt.isocalendar().week.astype(int).values
        for y in sorted(d.year.unique()):
            dy = d[d.year == y]
            tr, te = dy[dy.week % 2 == 1], dy[dy.week % 2 == 0]
            if len(te) < 1000 or len(tr) < 1000:
                continue
            base = r2(te.price_eur.values, fit_predict(tr, te, False))
            full = r2(te.price_eur.values, fit_predict(tr, te, True))
            out[f"{area}|{y}"] = {"r2_without": round(float(base), 3), "r2_with": round(float(full), 3),
                                  "forecast_adds": round(float(full - base), 3), "hours": int(len(te))}
    p = df.pivot_table(index="hour_utc", columns="area", values=["price_eur", "wind_fc_share"]).dropna()
    spread = (p["price_eur"]["DK1"] - p["price_eur"]["DK2"]).values
    dshare = (p["wind_fc_share"]["DK1"] - p["wind_fc_share"]["DK2"]).values
    ok = np.isfinite(spread) & np.isfinite(dshare)
    c = float(np.corrcoef(spread[ok], dshare[ok])[0, 1]) if ok.sum() > 100 else None
    # the same, but only in hours where the two zones actually differ (spread beyond 2 EUR)
    big = ok & (np.abs(spread) > 2)
    c2 = float(np.corrcoef(spread[big], dshare[big])[0, 1]) if big.sum() > 100 else None
    out["spread"] = {"corr_all_hours": round(c, 3) if c is not None else None, "hours": int(ok.sum()),
                     "corr_when_spread_exceeds_2eur": round(c2, 3) if c2 is not None else None,
                     "hours_when_spread_exceeds_2eur": int(big.sum())}
    return out


def q2(df):
    out = {}
    for area in ("DK1", "DK2"):
        d = df[df.area == area].copy()
        e = d.wind_err_da_mwh.values
        e1 = d.wind_err_1h_mwh.dropna().values
        out[area] = {
            "mae_day_ahead_mwh": round(float(np.mean(np.abs(e))), 1),
            "bias_day_ahead_mwh": round(float(np.mean(e)), 1),
            "mae_one_hour_mwh": round(float(np.mean(np.abs(e1))), 1) if len(e1) else None,
            "hours": int(len(d)),
        }
        d["decile"] = np.minimum(9, (d.wind_fc_da_mwh.rank(pct=True) * 10).astype(int))
        out[area]["mae_by_forecast_decile"] = [round(float(v), 1) for v in d.groupby("decile").wind_err_da_mwh.apply(lambda s: np.mean(np.abs(s)))]
        out[area]["bias_by_hour"] = [round(float(v), 1) for v in d.groupby("hour_dk").wind_err_da_mwh.mean()]
    return out


def q3(con, lo, hi):
    d = con.execute(f"""
        select area, wind_err_da_mwh, imbalance_minus_spot_eur, wind_fc_da_mwh
        from forecast_vs_imbalance
        where hour_utc >= timestamp '{lo}' and hour_utc < timestamp '{hi}'
          and wind_err_da_mwh is not null and imbalance_minus_spot_eur is not null""").df()
    out = {}
    for area in ("DK1", "DK2"):
        x = d[d.area == area].copy()
        if len(x) < 1000:
            continue
        agree = float(np.mean(np.sign(x.wind_err_da_mwh) == -np.sign(x.imbalance_minus_spot_eur)))
        x["decile"] = np.minimum(9, (x.wind_err_da_mwh.rank(pct=True) * 10).astype(int))
        by_dec = x.groupby("decile").imbalance_minus_spot_eur.mean()
        out[area] = {"sign_agreement": round(agree, 3), "hours": int(len(x)),
                     "mean_gap_by_error_decile_eur": [round(float(v), 2) for v in by_dec],
                     "mean_gap_more_wind_than_forecast": round(float(x[x.wind_err_da_mwh > 0].imbalance_minus_spot_eur.mean()), 2),
                     "mean_gap_less_wind_than_forecast": round(float(x[x.wind_err_da_mwh < 0].imbalance_minus_spot_eur.mean()), 2)}
    return out


def q4_untestable():
    return {"verdict": "cannot be tested as pre-registered",
            "reason": "the day-ahead forecast is published at 18:00 the day before delivery, about six hours "
                      "after the auction that sets the price the rule would trade; using it is look-ahead"}


def q4_prime(con, lo, hi, threshold=None):
    """Added after the documentation finding. The 06:00 revision of the wind
    forecast as a signal for the sign of the imbalance gap in delivery hours.
    A proxy for an intraday position; intraday prices are not public here."""
    d = con.execute(f"""
        select area, cast(hour_utc as date) as day, hour_utc,
               wind_fc_id_mwh - wind_fc_da_mwh as revision_mwh,
               imbalance_price_eur - price_eur as gap_eur
        from forecast_hourly h join stg_imbalance using (area, hour_utc)
        where hour_utc >= timestamp '{lo}' and hour_utc < timestamp '{hi}'
          and wind_fc_id_mwh is not null and wind_fc_da_mwh is not null and imbalance_price_eur is not null
        order by area, hour_utc""").df()
    out = {}
    for area in ("DK1", "DK2"):
        x = d[d.area == area].copy()
        if len(x) < 1000:
            continue
        thr = threshold[area] if threshold else float(np.percentile(np.abs(x.revision_mwh), 75))
        take = x[np.abs(x.revision_mwh) > thr]
        # more wind than the evening forecast implied means a lower imbalance price: short the gap
        pnl = -np.sign(take.revision_mwh) * take.gap_eur - 0.6
        n = int(len(take)); mean = float(pnl.mean()) if n else 0.0
        se = float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        agree = float(np.mean(np.sign(take.revision_mwh) == -np.sign(take.gap_eur))) if n else None
        # where the money is: a mean can be carried by a few hours, and the page must say so
        srt = np.sort(pnl.values)[::-1] if n else np.array([])
        top_share = float(srt[: max(1, n // 10)].sum() / srt.sum()) if n and srt.sum() > 0 else None
        out[area] = {"threshold_revision_mwh": round(thr, 1), "hours_traded": n, "hours_total": int(len(x)),
                     "sign_agreement": round(agree, 3) if agree is not None else None,
                     "median_pnl_eur": round(float(pnl.median()), 3) if n else None,
                     "share_of_total_from_best_decile": round(top_share, 3) if top_share is not None else None,
                     "mean_pnl_eur_per_mwh_hour": round(mean, 3), "standard_error": round(se, 3),
                     "verdict": ("null" if abs(mean) <= se else ("positive" if mean > 0 else "negative")) if n else "no trades"}
    return out


def q4(df, threshold=None):
    """The pre-registered spread rule. Kept for the record; not run, see q4_untestable."""
    p = df.pivot_table(index="hour_utc", columns="area", values=["price_eur", "wind_fc_share"]).dropna()
    daily = p.groupby(p.index.date).mean()
    dshare = daily["wind_fc_share"]["DK1"] - daily["wind_fc_share"]["DK2"]
    spread = daily["price_eur"]["DK1"] - daily["price_eur"]["DK2"]
    if threshold is None:
        threshold = float(np.percentile(np.abs(dshare), 75))
    take = np.abs(dshare) > threshold
    # more forecast wind in DK1 than DK2 means DK1 cheaper: short the spread
    direction = -np.sign(dshare[take])
    pnl = direction * spread[take] - 0.6  # pre-registered cost: 0.5 round trip + 0.1 bid-ask proxy
    n = int(take.sum())
    mean = float(pnl.mean()) if n else 0.0
    se = float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"threshold_share_diff": round(threshold, 4), "days_traded": n, "days_total": int(len(daily)),
            "mean_pnl_eur_per_mwh_day": round(mean, 3), "standard_error": round(se, 3),
            "verdict": ("null" if abs(mean) <= se else ("positive" if mean > 0 else "negative")) if n else "no trades"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="read the holdout once; code must be frozen first")
    a = ap.parse_args()
    lo, hi = HOLDOUT if a.holdout else TRAIN
    con = duckdb.connect(str(DB), read_only=True)
    df = frame(con, lo, hi)
    res = {"period": [lo, hi], "hours": int(len(df)), "q1": q1(df), "q2": q2(df), "q3": q3(con, lo, hi)}
    res["q4"] = q4_untestable()
    if a.holdout:
        trq = q4_prime(con, *TRAIN)
        thr = {k: v["threshold_revision_mwh"] for k, v in trq.items()}
        res["q4_prime"] = q4_prime(con, lo, hi, threshold=thr)
    else:
        res["q4_prime"] = q4_prime(con, lo, hi)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "holdout" if a.holdout else "training"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
