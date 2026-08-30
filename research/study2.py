"""Pre-registration v2: who pays when the wind forecast is wrong. Writes research/results/study2.json
with every quantity registered in PREREGISTRATION.md (v2), for training and holdout separately."""
import json
import numpy as np
import pandas as pd
import duckdb

TRAIN = ("2019-12-01", "2024-01-01"); HOLD = ("2024-01-01", "2025-03-05")
SINGLE = pd.Timestamp("2021-11-01"); SHARE = 0.05; COST = 0.6
con = duckdb.connect("data/tavle.duckdb", read_only=True)
fv = con.execute("""select v.area, v.hour_utc, v.price_eur, v.wind_actual_mwh, v.imbalance_price_eur, v.imbalance_minus_spot_eur as gap,
                           f.wind_fc_da_mwh, f.wind_fc_id_mwh, f.wind_fc_5h_mwh, f.wind_fc_1h_mwh, f.consumption_mwh
                    from forecast_vs_imbalance v join forecast_hourly f using (area, hour_utc)
                    where v.imbalance_minus_spot_eur is not null and f.wind_fc_da_mwh is not null order by 1, 2""").df()
fv["err"] = fv.wind_actual_mwh - fv.wind_fc_da_mwh
HORIZONS = [("evening before", "wind_fc_da_mwh"), ("same morning", "wind_fc_id_mwh"), ("five hours ahead", "wind_fc_5h_mwh"), ("one hour ahead", "wind_fc_1h_mwh")]


def period(df, lo, hi):
    return df[(df.hour_utc >= lo) & (df.hour_utc < hi)]


def h1(d):
    d = d[d.hour_utc >= SINGLE].dropna(subset=[c for _, c in HORIZONS])
    out = {}
    for area in ("DK1", "DK2"):
        x = d[d.area == area]; A = SHARE * x.wind_actual_mwh; rows = []
        for name, col in HORIZONS:
            N = SHARE * x[col]; cost = -(A - N) * (x.imbalance_price_eur - x.price_eur)
            rows.append({"horizon": name, "eur_per_mwh": float(cost.sum() / A.sum()), "total_keur": float(cost.sum() / 1e3),
                         "imbalance_share": float((A - N).abs().sum() / A.sum()), "hours": int(len(x))})
        by_year = []
        for y, g in x.groupby(x.hour_utc.dt.year):
            Ay = SHARE * g.wind_actual_mwh
            by_year.append({"year": int(y), **{name: float((-(Ay - SHARE * g[col]) * (g.imbalance_price_eur - g.price_eur)).sum() / Ay.sum()) for name, col in HORIZONS}})
        out[area] = {"rows": rows, "by_year": by_year, "mwh_produced": float(A.sum()), "share": SHARE}
    return out


def h2(d):
    out = {}
    for area in ("DK1", "DK2"):
        x = d[d.area == area]; out[area] = {}
        for regime, sel in (("dual", x.hour_utc < SINGLE), ("single", x.hour_utc >= SINGLE)):
            g = x[sel]
            if len(g) < 500:
                continue
            out[area][regime] = {"hours": int(len(g)), "gap_more_wind": float(g.gap[g.err > 0].mean()), "gap_less_wind": float(g.gap[g.err < 0].mean()),
                                 "zero_share": float((g.gap == 0).mean()), "mean_abs_gap": float(g.gap.abs().mean()),
                                 "from": str(g.hour_utc.min())[:10], "to": str(g.hour_utc.max())[:10]}
    return out


def h3(d):
    d = d[d.hour_utc >= SINGLE]
    out = {}
    for area in ("DK1", "DK2"):
        x = d[d.area == area].set_index("hour_utc").sort_index()
        g = np.sign(x.gap); e = np.sign(x.err); nz = g != 0
        maj = float(max((g[nz] > 0).mean(), (g[nz] < 0).mean()))
        lags = []
        for k in (1, 2, 3, 6):
            gl = g.shift(k, freq="h").reindex(g.index); el = e.shift(k, freq="h").reindex(g.index)
            m = nz & (gl != 0) & gl.notna(); m2 = nz & (el != 0) & el.notna()
            lags.append({"lag": k, "persistence": float((g[m] == gl[m]).mean()), "wind_error_rule": float((g[m2] == -el[m2]).mean()), "n": int(m.sum())})
        rules = {}
        for k in (1, 2):  # k = 1 as registered; k = 2 added after registration because the intraday gate closes an hour before delivery
            gl = g.shift(k, freq="h").reindex(g.index); m = (gl != 0) & gl.notna()
            pnl = (gl[m] * x.gap[m] - COST)
            srt = np.sort(pnl.values)[::-1]; top = srt[: max(1, len(srt) // 10)].sum()
            rules[str(k)] = {"hours": int(len(pnl)), "hit_rate": float((np.sign(x.gap[m]) == gl[m]).mean()), "mean": float(pnl.mean()),
                             "se": float(pnl.std(ddof=1) / np.sqrt(len(pnl))), "median": float(pnl.median()),
                             "best_decile_share": float(top / pnl.sum()) if pnl.sum() != 0 else None, "cost": COST,
                             "by_year": [{"year": int(y), "mean": float(v.mean()), "hit": float((np.sign(x.gap[m][v.index]) == gl[m][v.index]).mean())} for y, v in pnl.groupby(pnl.index.year)]}
        out[area] = {"majority_baseline": maj, "lags": lags, "rule": rules["1"], "rule_lag2": rules["2"]}
    return out


def h4(lo, hi):
    fh = con.execute(f"""select area, hour_utc, price_eur, wind_fc_da_mwh, wind_actual_mwh, consumption_mwh from forecast_hourly
                         where price_eur is not null and consumption_mwh > 0 and wind_fc_da_mwh is not null and wind_actual_mwh is not null
                           and hour_utc >= timestamp '{lo}' and hour_utc < timestamp '{hi}'""").df()
    local = fh.hour_utc.dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen"); fh["h"] = local.dt.hour.values; fh["m"] = local.dt.month.values; fh["year"] = local.dt.year.values
    out = {"market": {}, "bias": {}}
    for area in ("DK1", "DK2"):
        rows = []
        for y, d in fh[fh.area == area].groupby("year"):
            if len(d) < 3000:
                continue
            X = [np.ones(len(d))] + [(d.h == h).astype(float).values for h in range(1, 24)] + [(d.m == mm).astype(float).values for mm in range(2, 13)]
            fs = (d.wind_fc_da_mwh / d.consumption_mwh).values; ds = (d.wind_actual_mwh / d.consumption_mwh).values - fs
            Xf = np.column_stack(X + [fs, fs ** 2]); Xb = np.column_stack(X + [fs, fs ** 2, ds]); y_ = d.price_eur.values
            bf = np.linalg.lstsq(Xf, y_, rcond=None)[0]; bb = np.linalg.lstsq(Xb, y_, rcond=None)[0]
            r2 = lambda yh: float(1 - ((y_ - yh) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum())
            rows.append({"year": int(y), "coef_forecast": float(bf[-2]), "coef_added": float(bb[-1]), "r2_forecast": r2(Xf @ bf), "r2_added": r2(Xb @ bb), "hours": int(len(d))})
        out["market"][area] = rows
        x = fh[fh.area == area]
        out["bias"][area] = [{"year": int(y), "bias_mwh": float((g.wind_actual_mwh - g.wind_fc_da_mwh).mean()), "pct": float(100 * (g.wind_actual_mwh - g.wind_fc_da_mwh).mean() / g.wind_actual_mwh.mean()), "hours": int(len(g))} for y, g in x.groupby("year")]
    pc = con.execute(f"select area, hour_utc, price_eur, wind_mwh, solar_mwh from power_context where price_eur is not null and hour_utc >= timestamp '{lo}' and hour_utc < timestamp '{hi}'").df()
    local = pc.hour_utc.dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen"); pc["h"] = local.dt.hour.values; pc["year"] = local.dt.year.values
    out["negative"] = {}
    for area in ("DK1", "DK2"):
        neg = pc[(pc.area == area) & (pc.price_eur <= 0)]
        out["negative"][area] = [{"year": int(y), "hours": int(len(g)), "midday_share": float(((g.h >= 10) & (g.h <= 16)).mean()), "night_share": float(((g.h >= 22) | (g.h <= 5)).mean()),
                                  "solar_mwh": float(g.solar_mwh.mean()), "wind_mwh": float(g.wind_mwh.mean()), "by_hour": [int((g.h == h).sum()) for h in range(24)]} for y, g in neg.groupby("year")]
    return out


res = {}
for name, (lo, hi) in (("training", TRAIN), ("holdout", HOLD)):
    d = period(fv, pd.Timestamp(lo), pd.Timestamp(hi))
    res[name] = {"period": [lo, hi], "hours": int(len(d)), "h1": h1(d), "h2": h2(d), "h3": h3(d), "h4": h4(lo, hi)}
res["all"] = {"h4": h4("2019-12-01", "2099-01-01")}
res["definitions"] = {"share": SHARE, "cost": COST, "single_pricing_from": str(SINGLE.date()), "horizons": [h for h, _ in HORIZONS],
                      "disclosure": "an exploratory pass over all data was run on 30 August 2026 before v2 was registered; the holdout is a separate period, not an unread one"}
json.dump(res, open("research/results/study2.json", "w"), indent=1)
for name in ("training", "holdout"):
    r = res[name]; print(f"== {name} {r['period']} hours {r['hours']}")
    for a in ("DK1", "DK2"):
        print("  H1", a, [(x["horizon"][:8], round(x["eur_per_mwh"], 2)) for x in r["h1"][a]["rows"]])
        print("  H2", a, {k: (round(v["gap_more_wind"], 2), round(v["gap_less_wind"], 2), round(v["zero_share"], 2)) for k, v in r["h2"][a].items()})
        print("  H3", a, "maj", round(r["h3"][a]["majority_baseline"], 3), [(l["lag"], round(l["persistence"], 3), round(l["wind_error_rule"], 3)) for l in r["h3"][a]["lags"]], "rule", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["h3"][a]["rule"].items()})
        print("  H4", a, [(x["year"], round(x["r2_added"] - x["r2_forecast"], 4)) for x in r["h4"]["market"][a]])
print("negative DK1:", [(x["year"], x["hours"], round(x["midday_share"], 2)) for x in res["all"]["h4"]["negative"]["DK1"]])
