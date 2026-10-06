# P&L audit of the 2023 rehearsal (6 Oct 2026, 18:35)

The profit code passes every check except one boundary case in the Holm verdict. The real-data comparison is still pending, because gene has been offline since about 17:55. Tests: `model/test_audit_pnl.py`, 24 pass and 1 xfail that documents the bug.

## Checks

1. **Signs: pass.** strategies.py:54 books pnl = pos x gap - MWh x fee, with supply at -1. Idea C books two legs at 2 MWh (strategies.py:99-106).
2. **Fees: pass.** The rate is the delivery year's (fees.py:50). Stress replaces the fee (fees.py:45-46). Only traded, settled hours are charged (strategies.py:48).
3. **Days: pass.** The local date comes from the tz-aware hour (panel.py:311), and empty days count as zero (score.py:98). Real 2023: 365 days and 96,360 baseline MWh, which is every zone-hour including the 23- and 25-hour days.
4. **Information by 05:00: pass.** The trailing mean (panel.py:201) ends at 05:00 on D and starts 365 days earlier. Changing the delivery day, D 04:00, or the hour before the window opens moves no feature. Refits stop at the month start minus 2 days (walkforward.py:36-38).
5. **Bootstrap: pass, one bug.** The settings and the Holm set match the contract. Bug at score.py:192: an idea can get "pays" without being Holm-rejected, when exactly 83 of 10,000 means are at or below zero.
6. **Ranking table: pass** (score.py:106 and 152-154; rehearsal.py:205 cuts 1.00% of baseline hours).
7. **Annualisation:** none exists.
8. **Inflation: pass**, covering duplicates (rehearsal.py:73 and 91 raise), NaN predictions, boundaries, and 2022-only tuning and pairs. Exception: TOP_SITES (features_outages.py:58-60) was counted through 2023, which makes it in-sample for D.

## Second implementation

`model/audit_pnl.py` shares no profit code with the official files. On synthetic inputs it reproduces 411 of the 422 official fields exactly when given the same resamples, and it flags a wrong fee in 228 fields. Its own bootstrap matches the official one in distribution, so the remaining differences are Monte Carlo noise. On the real data, the official JSON is consistent to the cent, but it predates the 17:37 code edits.

## Patches, largest effect first

1. features_outages.py:190: count TOP_SITES only through 2022-12-30 for the rehearsal (affects D only).
2. Rerun `rehearsal.py rescore`, so the JSON comes from the code being frozen.
3. score.py:192: `r["verdict"] = verdict(*r["interval"]) if info["rejected"] else "inconclusive"`.

Before the freeze, commit or move `audit_pnl.py` and `test_audit_pnl.py`, because lock.py refuses untracked .py files.

## To finish the real-data comparison (writes nothing on gene)

```
cd research/us/model; S=/tmp/audit; mkdir -p $S; X="nyiso-us/.venv/bin/python - extract"
ssh gene "$X nyiso-us/parquet/prices_zone.parquet delivery_hour 2021-12-25 delivery_hour,zone,da_lbmp,rt_lbmp,da_published_at,rt_published_at,rt_revised" < audit_pnl.py > $S/prices.parquet
ssh gene "$X nyiso-us/cache/preds_gbm_2023.parquet delivery_hour 2023-01-01" < audit_pnl.py > $S/preds.parquet
ssh gene "$X nyiso-us/cache/panel_base.parquet delivery_hour 2023-01-01 delivery_hour,zone,delivery_date,bid_date,gap,gap_365d_h" < audit_pnl.py > $S/panel.parquet
python3 audit_pnl.py --prices $S/prices.parquet --preds $S/preds.parquet --panel $S/panel.parquet \
  --official ../results/rehearsal_2023.json --out $S/audit_2023.json
```
The last line prints the fields that differ by more than 0.5% beyond Monte Carlo noise. Details are in `diffs_gbm`, and `diffs_gbm_with_panel_inputs` separates data-layer causes from logic causes.
