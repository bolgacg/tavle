"""A model card for the wind study's price model, and the anatomy of the forecast drift.

What is trained: the pre-registered explanatory model of the day-ahead price: hour-of-day and
month effects, with and without Energinet's expected wind as a share of consumption (linear and
squared), fitted by least squares on the odd ISO weeks of a zone-year and scored on the even
weeks. Here it is scored the way a reader would ask: error in euros per megawatt-hour, against
two things anyone could do without a model (yesterday's price for the same hour; the mean of the
training weeks). Also: the monthly bias of the day-ahead wind forecast split into its parts, so
the drift chart can say what moves it.
"""
import json
import sys
import numpy as np
import duckdb

sys.path.insert(0, "research")
from study import design, fit_predict, r2  # noqa: E402

con = duckdb.connect("data/tavle.duckdb", read_only=True)
df = con.execute("""
    select f.area, f.hour_utc, p.price_eur, f.wind_fc_da_mwh, f.wind_actual_mwh, f.wind_err_da_mwh, f.wind_fc_share
    from forecast_hourly f join power_hourly p on p.area = f.area and p.hour_utc = f.hour_utc
    where p.price_eur is not null and f.wind_fc_share is not null
    order by 1, 2""").df()
local = df.hour_utc.dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
df["hour_dk"] = local.dt.hour.values; df["month"] = local.dt.month.values
df["year"] = df.hour_utc.dt.year
df["week"] = df.hour_utc.dt.isocalendar().week.astype(int).values
out = {"model": {"target": "day-ahead price, EUR/MWh, per hour and zone",
                 "features": "23 hour-of-day dummies, 11 month dummies, expected wind share of consumption and its square",
                 "fit": "ordinary least squares on odd ISO weeks of each zone-year, scored on even weeks (alternating-week holdout within the year)",
                 "why_within_year": "price levels shifted by an order of magnitude between years (2021 to 2023), so a model fitted on one year and scored on another is dominated by the level; alternating weeks hold the level fixed and ask only whether wind explains the hour-to-hour movement"},
       "rows": [], "drift": {}}
for area in ("DK1", "DK2"):
    da = df[df.area == area]
    for y in sorted(da.year.unique()):
        dy = da[da.year == y].copy()
        tr, te = dy[dy.week % 2 == 1], dy[dy.week % 2 == 0]
        if len(tr) < 500 or len(te) < 500:
            continue
        yhat_with = fit_predict(tr, te, True); yhat_without = fit_predict(tr, te, False)
        y_te = te.price_eur.values
        # naive: same hour the previous day (needs the previous day's price in the frame)
        prev = dy.set_index("hour_utc").price_eur.shift(24, freq="h")
        naive = prev.reindex(te.hour_utc).values
        ok = ~np.isnan(naive)
        mae = lambda a, b: float(np.mean(np.abs(a - b)))
        out["rows"].append({"area": area, "year": int(y), "period": "holdout" if y >= 2024 else "training", "hours": int(len(te)),
                            "price_mean": float(y_te.mean()), "price_sd": float(y_te.std()),
                            "mae_with_wind": mae(y_te, yhat_with), "mae_without_wind": mae(y_te, yhat_without),
                            "mae_naive_yesterday": mae(y_te[ok], naive[ok]), "mae_train_mean": mae(y_te, np.full_like(y_te, tr.price_eur.mean())),
                            "r2_with": float(r2(y_te, yhat_with)), "r2_without": float(r2(y_te, yhat_without))})
    # drift anatomy: monthly bias in MWh, as a share of actual wind, and the actual wind level
    m = da.copy(); m["month_key"] = m.hour_utc.dt.strftime("%Y-%m")
    g = m.groupby("month_key").agg(bias=("wind_err_da_mwh", "mean"), actual=("wind_actual_mwh", "mean"), forecast=("wind_fc_da_mwh", "mean"), n=("wind_err_da_mwh", "size")).reset_index()
    g["bias_pct_of_actual"] = 100 * g.bias / g.actual
    g["year"] = g.month_key.str[:4].astype(int)
    by_year = g.groupby("year").agg(bias=("bias", "mean"), pct=("bias_pct_of_actual", "mean"), actual=("actual", "mean")).reset_index()
    # within-year seasonality: correlation of monthly bias with monthly actual wind
    corr = float(np.corrcoef(g.bias, g.actual)[0, 1])
    out["drift"][area] = {"monthly": [{"month": r.month_key, "bias": round(float(r.bias)), "actual": round(float(r.actual)), "forecast": round(float(r.forecast)), "pct": round(float(r.bias_pct_of_actual), 1)} for r in g.itertuples()],
                          "by_year": [{"year": int(r.year), "bias": round(float(r.bias)), "pct": round(float(r.pct), 1), "actual": round(float(r.actual))} for r in by_year.itertuples()],
                          "corr_bias_vs_actual_wind": round(corr, 2)}
json.dump(out, open("research/results/model_card.json", "w"), indent=1)
for r in out["rows"]:
    print(f"{r['area']} {r['year']} {r['period']:8s} price {r['price_mean']:6.1f}±{r['price_sd']:5.1f}  MAE with {r['mae_with_wind']:5.1f} without {r['mae_without_wind']:5.1f} yesterday {r['mae_naive_yesterday']:5.1f} mean {r['mae_train_mean']:5.1f}  R2 {r['r2_with']:.2f}/{r['r2_without']:.2f}")
for a, v in out["drift"].items():
    print(a, "bias by year", [(x["year"], x["bias"], f"{x['pct']}%") for x in v["by_year"]], "corr(bias, actual wind) =", v["corr_bias_vs_actual_wind"])
