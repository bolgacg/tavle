"""Extra scrutiny of idea B with gradient boosting (the lead's top row), build years only.

1. Inputs: B's hourly weather and load-forecast inputs (cache/feat_weather.parquet, the matrix B trained and
   predicted on) are rebuilt from the raw tables for sample bid days, once with the 05:00 filter and once with
   the filter opened to 24:00 on D. The cached values must equal the 05:00 version and not the opened one.
2. Where the money comes from: by side, month and zone; Winter Storm Elliott; the 2023 loss.
Positions are the lead's (side/lead_positions_2021_2023.parquet). Writes ~/nyiso-us/side/b_audit.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lab as L  # noqa: E402

px, lf_nyiso, wx = L.load_tables()
F = L.Frame(px)
lf = L.read_build(L.PQ / "load_forecast.parquet", "target_hour", None)
lf = lf[lf["zone"].isin(L.ZONES)]
fw = L.read_build(Path.home() / "nyiso-us" / "cache" / "feat_weather.parquet", "delivery_hour")
out = {}

# ---------------------------------------------------------------- 1. input timing
rng = np.random.default_rng(L.SEED)
cand = F.days[(F.days >= "2021-05-01") & (F.days <= "2023-12-31")]
sample = sorted(set(pd.Timestamp(x) for x in rng.choice(cand, 40, replace=False)) | {pd.Timestamp("2022-12-23"), pd.Timestamp("2022-12-24"),
                                                            pd.Timestamp("2023-02-04"), pd.Timestamp("2022-06-13")})


def rebuild(d1, shift_h):
    D = d1 - pd.Timedelta(days=1)
    t = L.common.decision_time(D.date()) + pd.Timedelta(hours=shift_h)
    w = wx[(wx["target_hour"].dt.tz_localize(None).dt.normalize() == d1) & (wx["published_at"] <= t)]
    hr = w["target_hour"].dt.hour
    if shift_h == 0:     # the three-day rule (timing.gfs_rule): day2 to 21:00, day3 for 22, 23 and gaps
        w = w[((w["run_lead_hours"] == 48) & (hr <= 21)) | (w["run_lead_hours"] == 72)]
        w = w.sort_values(["point", "target_hour", "run_lead_hours"], kind="stable")
        w = w.drop_duplicates(["point", "target_hour"], keep="first")
    else:                # opened: the newest value public by then
        w = w.sort_values(["point", "target_hour", "published_at"], kind="stable")
        w = w.drop_duplicates(["point", "target_hour"], keep="last")
    w = w.assign(f=w["temperature_2m_c"] * 9 / 5 + 32)[["zone", "target_hour", "f"]].rename(columns={"target_hour": "delivery_hour"})
    l = lf[(lf["target_hour"].dt.tz_localize(None).dt.normalize() == d1) & (lf["published_at"] <= t)]
    l = l.sort_values("published_at", kind="stable").drop_duplicates(["zone", "target_hour"], keep="last")
    l = l[["zone", "target_hour", "load_forecast_mw"]].rename(columns={"target_hour": "delivery_hour"})
    return w.merge(l, on=["zone", "delivery_hour"], how="outer")


recs = []
for d1 in sample:
    c = fw[fw["delivery_hour"].dt.tz_localize(None).dt.normalize() == d1][["zone", "delivery_hour", "x_wx_temp_f", "x_lf_mw"]]
    a = c.merge(rebuild(d1, 0), on=["zone", "delivery_hour"], how="left")
    b = c.merge(rebuild(d1, 19), on=["zone", "delivery_hour"], how="left")
    recs.append({"delivery_day": str(d1.date()), "rows": len(c),
                 "wx_equal_0500": int(np.isclose(a["x_wx_temp_f"], a["f"], atol=1e-6, equal_nan=True).sum()),
                 "wx_equal_opened": int(np.isclose(b["x_wx_temp_f"], b["f"], atol=1e-6, equal_nan=True).sum()),
                 "lf_equal_0500": int(np.isclose(a["x_lf_mw"], a["load_forecast_mw"], atol=1e-6, equal_nan=True).sum()),
                 "lf_equal_opened": int(np.isclose(b["x_lf_mw"], b["load_forecast_mw"], atol=1e-6, equal_nan=True).sum())})
r = pd.DataFrame(recs)
out["inputs"] = {"days": len(r), "rows": int(r["rows"].sum()),
                 "weather_equal_to_0500_rebuild": int(r["wx_equal_0500"].sum()),
                 "weather_equal_to_opened_rebuild": int(r["wx_equal_opened"].sum()),
                 "load_forecast_equal_to_0500_rebuild": int(r["lf_equal_0500"].sum()),
                 "load_forecast_equal_to_opened_rebuild": int(r["lf_equal_opened"].sum()),
                 "days_detail": recs}

# ---------------------------------------------------------------- 2. money
lp = L.read_build(L.SIDE / "lead_positions_2021_2023.parquet", "delivery_hour")
b = lp[lp["strategy"] == "B_gbm"]
m = F.h[["delivery_hour", "zone"]].merge(b[["delivery_hour", "zone", "mw", "pred"]], on=["delivery_hour", "zone"], how="left")
mw = np.nan_to_num(m["mw"].to_numpy(float))
pred = m["pred"].to_numpy(float)
c = np.where(mw < 0, F.SUPC, np.where(mw > 0, F.LOADC, 0.0))
pnl = mw * F.GAP - np.abs(mw) * c
d = pd.DataFrame({"day": F.h["ddate"], "year": F.YEAR, "zone": F.zone, "mw": mw, "pnl": pnl, "gap": F.GAP, "pred": pred})
d = d[d["year"] >= 2021]
d["side"] = np.where(d["mw"] > 0, "load", np.where(d["mw"] < 0, "supply", "none"))
daily = d.groupby("day")["pnl"].sum()
out["by_year_side"] = {f"{y} {s}": {"usd": round(float(g["pnl"].sum())), "mwh": int(len(g))}
                       for (y, s), g in d[d["side"] != "none"].groupby(["year", "side"])}
out["by_month"] = {str(k): round(float(v)) for k, v in d.groupby(d["day"].dt.to_period("M"))["pnl"].sum().items()}
out["corr_pred_gap"] = {str(y): round(float(np.corrcoef(g["pred"], g["gap"])[0, 1]), 3)
                        for y, g in d[d["pred"].notna()].groupby("year")}
cut = d[d["pred"].notna()].copy()
cut["q"] = cut.groupby("year")["gap"].transform(lambda x: x.abs() <= x.abs().quantile(0.99))
out["corr_pred_gap_without_top1pct"] = {str(y): round(float(np.corrcoef(g["pred"], g["gap"])[0, 1]), 3)
                                        for y, g in cut[cut["q"]].groupby("year")}
ell = ["2022-12-23", "2022-12-24"]
e = d[d["day"].isin(pd.to_datetime(ell))]
out["elliott"] = {"pnl_by_day": {str(k.date()): round(float(v)) for k, v in e.groupby("day")["pnl"].sum().items()},
                  "position_mix": {str(k.date()): g["side"].value_counts().to_dict() for k, g in e.groupby("day")},
                  "mean_pred_by_day": {str(k.date()): round(float(g["pred"].mean()), 1) for k, g in e.groupby("day")},
                  "mean_gap_by_day": {str(k.date()): round(float(g["gap"].mean()), 1) for k, g in e.groupby("day")},
                  "three_year_total": round(float(daily.sum())),
                  "three_year_without_elliott": round(float(daily.drop(pd.to_datetime(ell)).sum())),
                  "2022_without_elliott": round(float(daily[daily.index.year == 2022].drop(pd.to_datetime(ell)).sum()))}
dec = d[(d["day"] >= "2022-12-01") & (d["day"] <= "2022-12-31")]
out["december_2022_daily_mix"] = {str(k.date()): {"load": int((g["side"] == "load").sum()), "supply": int((g["side"] == "supply").sum()),
                                                   "usd": round(float(g["pnl"].sum()))} for k, g in dec.groupby("day")}
srt = daily.sort_values(ascending=False)
out["best_days"] = {str(k.date()): round(float(v)) for k, v in srt.head(15).items()}
out["without_best_k"] = {str(k): round(float(srt.iloc[k:].sum())) for k in (0, 2, 5, 10, 20, 50)}
for y in (2021, 2022, 2023):
    dy = daily[daily.index.year == y].sort_values(ascending=False)
    out[f"{y}_without_best_k"] = {str(k): round(float(dy.iloc[k:].sum())) for k in (0, 5, 10, 20)}
z = d[d["year"] == 2023].groupby(["zone", "side"])["pnl"].sum().unstack(fill_value=0).round()
out["2023_by_zone_side"] = z.to_dict(orient="index")
out["2022_by_zone_side"] = d[d["year"] == 2022].groupby(["zone", "side"])["pnl"].sum().unstack(fill_value=0).round().to_dict(orient="index")
(L.SIDE / "b_audit.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ("december_2022_daily_mix",)}, default=str)[:6000])
