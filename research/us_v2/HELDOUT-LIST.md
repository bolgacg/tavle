# New York study, version 2: the held-out list

Written 8 Oct 2026, before any held-out data is read. Applies the selection rule I fixed on 7 Oct (commit 99234c2, version 1 objectives; it covers version 1 and version 2) mechanically and with no judgement: a row is a candidate when it passes the bar at the sizing view and is not flagged fragile, flags from audits included. Every other row runs beside the candidates as a comparison. Rows I registered as comparison rows (the addendum after the stage-1 audit, commit 10c01b4) stay comparisons whatever they score.

Build-year numbers: rolling windows, train on the previous three years, test on the next quarter, 44 quarters from 2013 to 2023, full costs, net USD at 1 MW per position (results/v2/v2_results.md on the training machine, final scoring of 8 Oct after the V15a reruns). The bar at the sizing view: return of at least 10 percent a year on the 500,000 USD bankroll at the size whose worst drawdown is 100,000 USD (at most 5 MW), positive in at least 8 of the 11 years, Sharpe above 0.42, positive at 0.50 USD per MWh stress. Too-good flags (a year with Sharpe above 3, or one day above half of a year's profit) are reasons for an audit, not exclusions.

## Candidates (26)

| Row | Net 2013 to 2023 | Average a year | 2023 | Positive years of 11 | Sharpe | MW for a 100,000 drawdown | Return a year at that size | Passes at the sizing view | Fragile |
|---|---|---|---|---|---|---|---|---|---|
| V8_error_mining | 1,451,096 | 131,918 | 21,662 | 11 | 2.08 | 2.26 | 59.6% | yes | no |
| V15c_wx_border_gbm | 1,465,173 | 133,198 | 2,921 | 11 | 2.18 | 2.01 | 53.6% | yes | no |
| V15a_V5_limit | 1,125,624 | 102,329 | 27,734 | 11 | 2.29 | 2.51 | 51.4% | yes | no |
| V15a_V8_limit | 1,212,347 | 110,213 | 13,146 | 11 | 2.57 | 2.17 | 47.8% | yes | no |
| V16_B_gefs_joined | 1,690,124 | 153,648 | 27,364 | 11 | 2.46 | 1.54 | 47.4% | yes | no |
| V12_alloc | 5,677,372 | 516,125 | -36,267 | 10 | 2.4 | 0.41 | 42.5% | yes | no |
| V9_decompose | 1,208,319 | 109,847 | -4,605 | 10 | 1.77 | 1.82 | 39.9% | yes | no |
| V2_limit_gbm | 1,178,549 | 107,141 | 2,064 | 11 | 2.41 | 1.77 | 38.0% | yes | no |
| V3_tail_gbm | 3,907,473 | 355,225 | 70,495 | 11 | 1.9 | 0.53 | 37.4% | yes | no |
| V5_border_inputs | 1,260,003 | 114,546 | 45,823 | 11 | 1.96 | 1.52 | 34.8% | yes | no |
| V14f_global_zone | 576,202 | 52,382 | 1,159 | 11 | 2.28 | 3.19 | 33.5% | yes | no |
| V1_C_deep | 668,609 | 60,783 | 33,677 | 10 | 2.25 | 2.6 | 31.7% | yes | no |
| V14e_mlp_multi | 510,722 | 46,429 | 9,703 | 11 | 2.14 | 3.33 | 30.9% | yes | no |
| V7_recency | 1,314,936 | 119,540 | -5,930 | 10 | 1.88 | 1.22 | 29.2% | yes | no |
| V7_flags | 1,307,159 | 118,833 | -34,780 | 10 | 1.88 | 1.22 | 29.0% | yes | no |
| V11_C_deep_ens4 | 642,305 | 58,391 | 32,855 | 11 | 2.1 | 2.39 | 28.0% | yes | no |
| V15b_V13_both | 834,728 | 75,884 | -34,397 | 10 | 1.65 | 1.52 | 23.0% | yes | no |
| V14d_seed_ens | 583,112 | 53,010 | 20,259 | 11 | 1.93 | 2.08 | 22.0% | yes | no |
| V14b_quantile_size | 203,417 | 18,492 | 3,301 | 11 | 1.89 | 5.0 | 18.5% | yes | no |
| V14a_window_ens | 639,744 | 58,159 | 13,544 | 11 | 1.91 | 1.58 | 18.4% | yes | no |
| V13_C_deep_all | 451,367 | 41,033 | 17,001 | 11 | 1.73 | 1.84 | 15.1% | yes | no |
| V15b_V13_limit | 652,873 | 59,352 | -44,685 | 9 | 1.46 | 1.25 | 14.8% | yes | no |
| V14c_conformal_skip | 208,183 | 18,926 | -8,334 | 10 | 1.19 | 3.59 | 13.6% | yes | no |
| V11_C_gbm_ens4 | 422,684 | 38,426 | -8,435 | 10 | 1.45 | 1.75 | 13.4% | yes | no |
| V1_C_gbm | 368,164 | 33,469 | -3,750 | 10 | 1.29 | 1.81 | 12.1% | yes | no |
| V6_LI | 128,768 | 11,706 | 11,654 | 10 | 1.26 | 5.0 | 11.7% | yes | no |

Notes that change no selection:
- V12_alloc, the learned allocator over the version 2 survivors, ran before the V15a reruns and before V16 finished, so it allocates over the survivors as they stood then. It is carried into the held-out run as it ran and is not rerun.
- V3_tail_gbm is, by the stage-1 audit, the same direction bet as V2 at about 3 MW (audit/v2_stage1_audit.md); read it per MWh and beside V3_tail_gbm_1MW. V2_limit_gbm is the gradient-boosting direction model; its limit prices add nothing.
- V1_C_deep, V10 and V11's C rows are mostly a persistent pair spread (same audit); C_static_pairs is their model-free benchmark.

## Comparisons (12 rows, plus market baselines and version 1)

| Row | Net 2013 to 2023 | Average a year | 2023 | Positive years of 11 | Sharpe | MW for a 100,000 drawdown | Return a year at that size | Passes at the sizing view | Fragile |
|---|---|---|---|---|---|---|---|---|---|
| V15a_V4_limit | 1,110,359 | 100,942 | 0 | 7 | 2.65 | 2.41 | 48.6% | no | no |
| V3_tail_gbm_1MW (registered comparison) | 1,356,902 | 123,355 | 26,326 | 11 | 1.94 | 1.55 | 38.3% | yes | no |
| V2_sides_no_limit (registered comparison) | 1,193,308 | 108,482 | -19,496 | 10 | 2.15 | 1.62 | 35.2% | yes | no |
| V4_B_reforecast | 1,115,684 | 101,426 | 0 | 7 | 2.83 | 1.54 | 31.3% | no | no |
| V10_C_deep_pretrain | 539,422 | 49,038 | 28,733 | 11 | 1.78 | 2.02 | 19.8% | yes | yes, random days in the same months beat the rule more than 20 percent of the time |
| C_static_pairs (registered comparison) | 507,922 | 46,175 | 24,153 | 11 | 1.9 | 1.78 | 16.4% | yes | yes, random days in the same months beat the rule more than 20 percent of the time |
| V6_NYC | 102,025 | 9,275 | -263 | 10 | 1.59 | 5.0 | 9.3% | no | no |
| V15d_V4_pairs | 212,044 | 19,277 | 0 | 6 | 0.83 | 1.54 | 5.9% | no | no |
| V13_A_deep_all | 307,062 | 27,915 | 11,063 | 8 | 0.66 | 0.59 | 3.3% | no | no |
| V1_A_spike_gbm | 261,243 | 23,749 | 25,390 | 7 | 0.74 | 0.68 | 3.2% | no | yes, random days in the same months beat the rule more than 20 percent of the time |
| V1_baseline | 204,294 | 18,572 | -72,836 | 6 | 0.58 | 0.8 | 3.0% | no | yes, random days in the same months beat the rule more than 20 percent of the time |
| V1_always_supply | 247,363 | 22,488 | 68,154 | 7 | 0.27 | 0.27 | 1.2% | no | yes, three-year total turns negative without one year's best 10 days |

V4_B_reforecast, V15a_V4_limit and V15d_V4_pairs read the GEFS reforecast, which ends in 2019; they traded nothing from 2020 to 2023 and have no forecast source in the held-out years, so they will be reported as not run unless their own code finds forecasts. V16 is the same idea with a consistent forecast source.

Market baselines over the same held-out months: the S&P 500 total return and its Sharpe ratio, and the Danish bank deposit rate with the Nationalbank current-account rate. Version 1's candidates and comparisons (research/us/HELDOUT-LIST.md) run in the same held-out run.

Count of tries: 38 version 2 rows and 87 settings on the 2013 to 2023 rolling quarters, on top of everything version 1 tried on 2021 to 2023.

The held-out run: January 2024 to September 2026, once, the rolling windows continuing quarter by quarter exactly as in the build years (train on the previous three years up to two days before each quarter, settings chosen on the trailing four out-of-sample quarters), sizes carried over unchanged, weather only from archived forecasts.
