# New York study, version 1: the held-out list

Written 8 Oct 2026, before any held-out data is read. Applies the selection rule I fixed on 7 Oct (commit 99234c2) with the sizing view of 6 Oct (commit 7504c92), mechanically and with no judgement: a strategy is a candidate when it passes the bar at the sizing view and is not flagged fragile, flags from the audits included. Every other strategy runs beside the candidates as a comparison, on the same unseen months, together with version 2's rows.

Build-year numbers: walk-forward over 2021 to 2023, each month predicted only from earlier data, full costs, net USD at 1 MW per zone-hour (side/lab_results.md, last written 7 Oct with idea 14; idea 13 from results/robustness_summary.md). The bar at the sizing view: return of at least 10 percent a year on the 500,000 USD bankroll at the size whose worst 2021 to 2023 drawdown is 100,000 USD (at most 5 MW), positive in at least two of the three years, Sharpe above 0.42, positive at 0.50 USD per MWh stress. A year with Sharpe above 3, or one day above half of a year's profit, is a reason for an audit, not an exclusion.

## Candidates (7)

| Strategy | 2021 | 2022 | 2023 | Three-year net | Sharpe | MW for a 100,000 drawdown | Return a year at that size | Passes at the sizing view | Fragile |
|---|---|---|---|---|---|---|---|---|---|
| B_gbm | 51,187 | 555,120 | -22,361 | 583,946 | 1.96 | 2.3 | 89.5% | yes | no |
| B_deep | 35,464 | 310,719 | -35,731 | 310,453 | 1.18 | 1.45 | 29.9% | yes | no |
| C_gbm | 41,457 | 124,029 | 4,403 | 169,889 | 1.67 | 3.41 | 38.6% | yes | no |
| C_deep | 57,838 | 73,206 | 34,363 | 165,407 | 2.15 | 3.92 | 43.2% | yes | no |
| A_regression_v1 | 25,769 | 32,400 | 39,058 | 97,227 | 0.72 | 1.71 | 11.1% | yes | no |
| idea13_portfolio | 123,226 | 507,435 | 33,485 | 664,146 | 3.12 | 0.594 | 26.3% | yes | no |
| idea14_learned_allocator_candidates | 136,420 | 1,489,127 | -104,490 | 1,521,057 | 1.68 | 0.55 | 55.7% | yes | no |

Idea 14 was registered after the full table was seen; a pass there weighs less than a pass on the earlier list (addendum of 7 Oct). Idea 13 is the equal-risk portfolio of the five survivors.

## Comparisons (27 strategy rows, plus two market baselines)

| Strategy | 2021 | 2022 | 2023 | Three-year net | Sharpe | MW for a 100,000 drawdown | Return a year at that size | Passes at the sizing view | Fragile |
|---|---|---|---|---|---|---|---|---|---|
| idea10_weather_surprise | 3,034 | 311,142 | 37,955 | 352,130 | 1.03 | 1.21 | 28.5% | yes | yes, fragile by its audit (side/idea10_audit.md): at the next cut the three-year total falls to 23,925 |
| idea2_storm_flip | 78,415 | 290,469 | -48,392 | 320,492 | 0.94 | 0.74 | 15.9% | yes | yes, fragile by the storm audit (side/storm_audit.md): an artefact of Winter Storm Elliott; three-year total turns negative without one year's best 10 days |
| storm_day_flip (second build) | 74,421 | 282,065 | -49,753 | 306,733 | 0.9 | 0.74 | 15.2% | yes | yes, fragile by the storm audit (side/storm_audit.md): an artefact of Winter Storm Elliott; three-year total turns negative without one year's best 10 days |
| idea5_deep_zone_day | 20,081 | 149,388 | 46,173 | 215,642 | 0.66 | 0.59 | 8.5% | no | yes, three-year total turns negative without one year's best 10 days |
| idea11_learned_allocator | -28,374 | 209,765 | -11,074 | 170,316 | 0.7 | 1.79 | 20.3% | no | yes, three-year total turns negative without one year's best 10 days |
| idea12_policy_network | -11,441 | 206,885 | -29,216 | 166,228 | 0.67 | 1.76 | 19.5% | no | yes, three-year total turns negative without one year's best 10 days |
| D_gbm | 40,450 | 145,900 | -39,033 | 147,317 | 0.84 | 2.08 | 20.4% | yes | yes, three-year total turns negative without one year's best 10 days |
| idea7_gbm_storm_filter | 7,643 | 70,444 | 30,689 | 108,777 | 0.8 | 1.11 | 8.0% | no | no |
| idea1_storm_filter | 41,620 | 49,655 | 15,866 | 107,141 | 0.91 | 1.07 | 7.6% | no | yes, fragile by the storm audit (side/storm_audit.md); three-year total changes sign at a neighbouring setting |
| storm_day_filter (second build) | 39,476 | 45,535 | 15,185 | 100,196 | 0.85 | 1.0 | 6.7% | no | yes, fragile by the storm audit (side/storm_audit.md); three-year total changes sign at a neighbouring setting |
| risk_sized_supply_gbm | -8,110 | 45,128 | 21,425 | 58,443 | 0.89 | 2.34 | 9.1% | no | no |
| A_spike_gbm | -16,822 | 46,476 | 20,542 | 50,195 | 0.73 | 1.66 | 5.6% | no | no |
| idea9_tail_spike_S100 | -2,378 | 377 | 49,792 | 47,791 | 0.34 | 0.7 | 2.2% | no | yes, three-year total turns negative without one year's best 10 days |
| A_spike_tail_S100_gbm (second build) | -2,378 | -28,960 | 49,792 | 18,454 | 0.1 | 0.64 | 0.8% | no | yes, three-year total turns negative without one year's best 10 days |
| A_spike_deep_rolling_cut | 12,369 | -59,902 | 22,117 | -25,416 | -0.14 | 0.59 | -1.0% | no | yes, three-year total turns negative without one year's best 10 days |
| idea8_storm_zone_pairs | 51,935 | -116,133 | 36,071 | -28,128 | -0.09 | 0.47 | -0.9% | no | yes, three-year total changes sign at a neighbouring setting; three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| A_spike_deep | -9,241 | -31,639 | -1,063 | -41,943 | -0.62 | 1.13 | -3.2% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| D_deep | 11,714 | -19,132 | -40,505 | -47,922 | -0.3 | 0.97 | -3.1% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| zone_subset_supply | 9,665 | -89,969 | 27,324 | -52,979 | -0.3 | 0.61 | -2.2% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| idea6_deep_two_gates | -22,823 | -31,639 | -2,425 | -56,888 | -0.81 | 0.98 | -3.7% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| baseline_usual_side | 12,655 | -8,643 | -72,836 | -68,824 | -0.81 | 0.8 | -3.7% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| A_regression_v1_deep | 9,335 | -91,649 | 12,745 | -69,568 | -0.33 | 0.65 | -3.0% | no | yes, three-year total turns negative without one year's best 10 days |
| A_hourly_mean | 7,484 | -87,770 | 2,638 | -77,648 | -0.52 | 0.72 | -3.7% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| side_past_spike_rate_filter | 7,360 | -136,202 | 37,604 | -91,237 | -0.33 | 0.39 | -2.4% | no | yes, three-year total turns negative without one year's best 10 days |
| always_supply | -5,015 | -198,269 | 68,154 | -135,130 | -0.4 | 0.29 | -2.6% | no | yes, three-year total turns negative without one year's best 10 days |
| idea4_deep_storm_three_way | -12,724 | -198,269 | 10,172 | -200,822 | -0.6 | 0.28 | -3.8% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| idea3_deep_storm_filter | -12,724 | -198,269 | 9,337 | -201,657 | -0.6 | 0.28 | -3.8% | no | yes, three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |

Market baselines over the same held-out months, as fixed in the cost and capital addendum: the S&P 500 total return (and its Sharpe ratio) and the Danish bank deposit rate with the Nationalbank current-account rate. Always supply, the usual-side baseline and the hourly-mean version of idea A are in the table above.

The held-out run: January 2024 to September 2026, once, every model refitted month by month on data up to two days before each month exactly as in the build years, sizes carried over unchanged from 2021 to 2023.
