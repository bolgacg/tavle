"""Side analysis (Bo, 6 Oct): what the deep spike score (AUC 0.78 in 2023) could be worth, build years only.

Reads prices for 2020 to 2023 ONLY (hard filter below) and the deep spike predictions for 2022 (one model trained to
2021-12-30) and 2023 (monthly refits). Strategies, all supply-only, 1 MW per zone-hour, full corrected costs:
  always supply; deep filter (sit out when p_spike > cut); history filter (sit out when the zone-hour's past spike
  frequency > cut); oracle (sit out the hours that actually spiked: the ceiling, not a strategy).
The cut for each filter is chosen on 2022 and applied unchanged to 2023, so 2023 is out of sample for model and cut.
"""
import json, numpy as np, pandas as pd
P = '/home/bo/nyiso-us'
S = 25.0   # must match spike_threshold in the prediction files (checked below)
COST = {2020: 0.0862+0.010+0.007, 2021: 0.0757+0.015+0.004, 2022: 0.0853+0.016+0.003, 2023: 0.1066+0.017+0.026}
BANKROLL = 500_000

px = pd.read_parquet(f'{P}/parquet/prices_zone.parquet', columns=['delivery_hour', 'zone', 'da_lbmp', 'rt_lbmp'])
px = px[px.delivery_hour < pd.Timestamp('2024-01-01', tz='America/New_York')]          # holdout guard
assert px.delivery_hour.max() < pd.Timestamp('2024-01-01', tz='America/New_York')
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
px = px[px.zone.isin(ZONES)].copy()
px['gap'] = px.rt_lbmp - px.da_lbmp
px['year'] = px.delivery_hour.dt.year
px['hod'] = px.delivery_hour.dt.hour
px['date'] = px.delivery_hour.dt.date

pred = pd.concat([pd.read_parquet(f'{P}/results/deep_spike_2022.parquet'),
                  pd.read_parquet(f'{P}/results/deep_spike_rehearsal_2023.parquet')])
d = px[px.year.isin([2022, 2023])].merge(pred[['delivery_hour', 'zone', 'p_spike']], on=['delivery_hour', 'zone'], how='left')
assert d.p_spike.notna().all(), 'missing predictions'
assert set(pred.spike_threshold.unique()) == {S}, pred.spike_threshold.unique()

# history filter: the zone-hour's spike frequency in all years before the scored year
def hist_rate(y):
    h = px[px.year < y]
    return (h.gap >= S).groupby([h.zone, h.hod]).mean().rename('p_hist')
d = pd.concat([g.join(hist_rate(y), on=['zone', 'hod']) for y, g in d.groupby('year')])

def run(g, sit_out, cost=None):
    t = ~sit_out
    c = g.year.map(COST) if cost is None else cost
    pnl = np.where(t, -g.gap - c, 0.0)
    day = pd.Series(pnl, index=g.index).groupby(g.date).sum()
    cum = day.cumsum(); dd = float((cum - cum.cummax()).min())
    sharpe = float(day.mean() / day.std() * np.sqrt(365)) if day.std() > 0 else float('nan')
    mwh = int(t.sum())
    return {'net_usd': round(float(day.sum())), 'usd_per_day': round(float(day.mean()), 1), 'mwh_per_day': round(mwh / len(day)),
            'usd_per_mwh': round(float(day.sum()) / max(mwh, 1), 2), 'sharpe': round(sharpe, 2), 'max_drawdown_usd': round(dd),
            'worst_day_usd': round(float(day.min())), 'return_on_500k_pct': round(100 * float(day.sum()) / BANKROLL * 365 / len(day), 1),
            'share_hours_sat_out': round(float(sit_out.mean()), 3)}

def rolling_cut(g, col, k):
    """Per delivery date: the (1-k) quantile of col over the previous 30 delivery dates (strictly earlier)."""
    daily = g.groupby('date')[col].apply(lambda x: x.to_numpy())
    dates = list(daily.index); thr = {}
    for i, dt in enumerate(dates):
        prev = daily.iloc[max(0, i - 30):i]
        thr[dt] = np.quantile(np.concatenate(prev.to_list()), 1 - k) if len(prev) else np.inf
    return g.date.map(thr)

out = {'S': S, 'costs': COST, 'rule': 'sit out a zone-hour when its score is in the top k of the previous 30 days of scores',
       'note': 'prices 2020-2023 only; k chosen on 2022 (in sample for 2022); 2023 out of sample for model and k'}
KS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
col = {'deep': 'p_spike', 'hist': 'p_hist'}
for y, g in d.groupby('year'):
    out[str(y)] = {'always_supply': run(g, pd.Series(False, index=g.index)),
                   'oracle_ceiling': run(g, g.gap >= S),
                   'curves': {m: {str(k): run(g, g[col[m]] > rolling_cut(g, col[m], k)) for k in KS} for m in col}}
chosen = {m: max(KS, key=lambda k: out['2022']['curves'][m][str(k)]['net_usd']) for m in col}
out['chosen_on_2022'] = chosen
for m, k in chosen.items():
    g = d[d.year == 2023]; cut = rolling_cut(g, col[m], k)
    out['2023'][f'{m}_filter_out_of_sample'] = run(g, g[col[m]] > cut)
    out['2023'][f'{m}_filter_stress_0.50'] = run(g, g[col[m]] > cut, cost=0.50)
    g2 = d[d.year == 2022]
    out['2022'][f'{m}_filter_in_sample_k'] = run(g2, g2[col[m]] > rolling_cut(g2, col[m], k))
# AUC check
from sklearn.metrics import roc_auc_score
for y, g in d.groupby('year'):
    out[str(y)]['auc_deep'] = round(float(roc_auc_score(g.gap >= S, g.p_spike)), 3)
    out[str(y)]['auc_hist'] = round(float(roc_auc_score(g.gap >= S, g.p_hist)), 3)
    out[str(y)]['spike_rate'] = round(float((g.gap >= S).mean()), 4)
    # where do the spike losses sit: share of always-supply losses in spike hours
    out[str(y)]['spike_hours_cost_usd'] = round(float((-g.gap[g.gap >= S]).sum()))
    top = g.p_spike >= g.p_spike.quantile(0.9)
    out[str(y)]['share_of_spike_hours_in_top10pct_deep'] = round(float(((g.gap >= S) & top).sum() / max((g.gap >= S).sum(), 1)), 3)
json.dump(out, open(f'{P}/results/side_spike_value.json', 'w'), indent=1)
print(json.dumps({y: {k: v for k, v in out[y].items() if k != 'curves'} for y in ['2022', '2023']}, indent=1))
print('chosen', chosen)
