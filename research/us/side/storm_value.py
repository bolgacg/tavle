"""Side analysis 2 (Bo, 6 Oct): the four adjusted strategies plus a storm flip, build years only (2020 to 2023).

Decision at 05:00 New York time on D for delivery day D+1. Every input is filtered on published_at <= D 05:00.
Choices (cuts, zone lists) are made on the year BEFORE the scored year and applied unchanged.
"""
import json, numpy as np, pandas as pd
P = '/home/bo/nyiso-us'; TZ = 'America/New_York'
END = pd.Timestamp('2024-01-01', tz=TZ)
COST = {2020: 0.0862+0.010+0.007, 2021: 0.0757+0.015+0.004, 2022: 0.0853+0.016+0.003, 2023: 0.1066+0.017+0.026}
LOADCOST = {y: c - u for (y, c), u in zip(COST.items(), [0.007, 0.004, 0.003, 0.026])}   # load legs pay no uplift
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
BANK = 500_000

px = pd.read_parquet(f'{P}/parquet/prices_zone.parquet', columns=['delivery_hour', 'zone', 'da_lbmp', 'rt_lbmp', 'rt_published_at'])
px = px[(px.delivery_hour < END) & px.zone.isin(ZONES)].copy()
assert px.delivery_hour.max() < END
px['gap'] = px.rt_lbmp - px.da_lbmp
px['ddate'] = px.delivery_hour.dt.tz_localize(None).dt.normalize()          # delivery date (local)
px['year'] = px.ddate.dt.year
bid_dates = pd.date_range('2020-01-02', '2023-12-30', freq='D')            # D; delivery = D+1

# --- day-level storm inputs, each known by 05:00 on D ---
lf = pd.read_parquet(f'{P}/parquet/load_forecast.parquet')
lf = lf[lf.zone == 'NYISO']
w = pd.read_parquet(f'{P}/parquet/weather_gfs.parquet')
w = w[w.primary & w.temperature_2m_c.notna()]
rows = []
for D in bid_dates:
    t05 = pd.Timestamp(D, tz=TZ) + pd.Timedelta(hours=5)
    d1s, d1e = pd.Timestamp(D + pd.Timedelta(days=1), tz=TZ), pd.Timestamp(D + pd.Timedelta(days=2), tz=TZ)
    f = lf[(lf.published_at <= t05) & (lf.target_hour >= d1s) & (lf.target_hour < d1e)]
    f = f[f.issue_date == f.issue_date.max()] if len(f) else f
    ww = w[(w.published_at <= t05) & (w.target_hour >= d1s) & (w.target_hour < d1e)]
    r = px[(px.ddate == pd.Timestamp(D)) & (px.rt_published_at <= t05)]
    rows.append({'D': D, 'peak_load_fc': f.load_forecast_mw.max() if len(f) else np.nan,
                 'tmin': ww.temperature_2m_c.min() if len(ww) else np.nan, 'tmax': ww.temperature_2m_c.max() if len(ww) else np.nan,
                 'rt_stress_now': r.gap.mean() if len(r) else np.nan})
day = pd.DataFrame(rows).set_index('D')
def trailing_pct(s, window=365):
    """Percentile rank of today's value among the previous `window` days' values (strictly earlier)."""
    out = pd.Series(np.nan, index=s.index)
    vals = s.to_numpy()
    for i in range(len(s)):
        prev = vals[max(0, i - window):i]; prev = prev[~np.isnan(prev)]
        if len(prev) >= 60 and not np.isnan(vals[i]): out.iloc[i] = (prev < vals[i]).mean()
    return out
day['p_load'] = trailing_pct(day.peak_load_fc)
day['p_cold'] = trailing_pct(-day.tmin)
day['p_heat'] = trailing_pct(day.tmax)
day['p_rt'] = trailing_pct(day.rt_stress_now)
day['storm'] = day[['p_load', 'p_cold', 'p_heat', 'p_rt']].max(axis=1)      # most extreme of the four, NaN-safe
day['ddate'] = day.index + pd.Timedelta(days=1)
px = px.merge(day[['ddate', 'storm']], on='ddate', how='left')

# deep spike scores (S=25) for 2022 and 2023, out of sample
pred = pd.concat([pd.read_parquet(f'{P}/results/deep_spike_2022.parquet'), pd.read_parquet(f'{P}/results/deep_spike_rehearsal_2023.parquet')])
assert set(pred.spike_threshold.unique()) == {25.0}
px = px.merge(pred[['delivery_hour', 'zone', 'p_spike']], on=['delivery_hour', 'zone'], how='left')

def score(g, mw):
    """mw: signed MW per row (-x = supply x MW, +x = load x MW)."""
    cost = np.where(mw < 0, g.year.map(COST), g.year.map(LOADCOST))
    pnl = mw * g.gap - np.abs(mw) * cost
    dd = pd.Series(pnl, index=g.index).groupby(g.ddate).sum()
    cum = dd.cumsum(); mdd = float((cum - cum.cummax()).min()); mwh = float(np.abs(mw).sum())
    return {'net_usd': round(float(dd.sum())), 'usd_per_mwh': round(float(dd.sum()) / max(mwh, 1), 2),
            'sharpe': round(float(dd.mean() / dd.std() * np.sqrt(365)), 2) if dd.std() > 0 else None,
            'max_drawdown_usd': round(mdd), 'worst_day_usd': round(float(dd.min())),
            'return_on_500k_pct': round(100 * float(dd.sum()) / BANK * 365 / dd.size, 1), 'mwh_per_day': round(mwh / dd.size),
            'net_without_best_3_days': round(float(dd.sort_values().iloc[:-3].sum())), 'best_day_usd': round(float(dd.max()))}

def stress(g, mw):
    pnl = mw * g.gap - np.abs(mw) * 0.50
    return round(float(pd.Series(pnl, index=g.index).groupby(g.ddate).sum().sum()))

CUTS = [0.80, 0.90, 0.95, 0.98]
res = {}
for Y in [2021, 2022, 2023]:
    g = px[px.year == Y]; prior = px[px.year == Y - 1]
    sup = -np.ones(len(g)); out = {'always_supply': score(g, sup)}
    # 1 storm-day filter: cut chosen on the prior year
    def filt(h, c): return np.where(h.storm > c, 0.0, -1.0)
    def flip(h, c): return np.where(h.storm > c, 1.0, -1.0)
    cf = max(CUTS, key=lambda c: score(prior, filt(prior, c))['net_usd'])
    cl = max(CUTS, key=lambda c: score(prior, flip(prior, c))['net_usd'])
    out['storm_day_filter'] = {**score(g, filt(g, cf)), 'cut': cf, 'stress_0.50': stress(g, filt(g, cf)), 'days_sat_out': int((g.groupby('ddate').storm.first() > cf).sum())}
    out['storm_day_flip_to_load'] = {**score(g, flip(g, cl)), 'cut': cl, 'stress_0.50': stress(g, flip(g, cl))}
    # 3 risk-sized by the deep score: MW = 1 - percentile rank among the previous 30 days of scores
    ranks = []
    dates = sorted(g.ddate.unique()) if g.p_spike.notna().all() else []
    for i, dt in enumerate(dates):
        cur = g[g.ddate == dt]; prev = g[(g.ddate < dt) & (g.ddate >= dt - pd.Timedelta(days=30))].p_spike.to_numpy()
        r = np.searchsorted(np.sort(prev), cur.p_spike.to_numpy()) / len(prev) if len(prev) else np.zeros(len(cur))
        ranks.append(pd.Series(r, index=cur.index))
    rank = pd.concat(ranks).reindex(g.index) if ranks else None
    if g.p_spike.notna().all():
        mw = -(1 - rank.to_numpy())
        out['risk_sized_deep'] = {**score(g, mw), 'stress_0.50': stress(g, mw)}
    # 4 zone subset chosen on all prior years: positive supply profit without the extreme 1% and smallest worst day
    hist = px[px.year < Y]
    cut99 = np.quantile(np.abs(hist.gap), 0.99)
    calm = hist[np.abs(hist.gap) <= cut99].groupby('zone').gap.mean().mul(-1)
    worst = hist.assign(s=-hist.gap).groupby(['zone', 'ddate']).s.sum().groupby('zone').min()
    keep = [z for z in worst.sort_values(ascending=False).index if calm[z] > 0][:6]
    mwz = np.where(g.zone.isin(keep), -1.0, 0.0)
    out['zone_subset'] = {**score(g, mwz), 'zones': keep, 'stress_0.50': stress(g, mwz)}
    res[str(Y)] = out
json.dump(res, open(f'{P}/results/side_storm_value.json', 'w'), indent=1, default=str)
for y, o in res.items():
    for k, v in o.items(): print(y, k, {x: v[x] for x in v if x in ('net_usd', 'sharpe', 'max_drawdown_usd', 'worst_day_usd', 'best_day_usd', 'net_without_best_3_days', 'return_on_500k_pct', 'stress_0.50', 'cut', 'days_sat_out')})
