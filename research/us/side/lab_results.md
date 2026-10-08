# Strategy lab: every strategy scored the same way on 2021 to 2023

Written 2026-10-07 09:54:46 CEST on gene by `side/lab.py`. Build years only: every table and prediction file was filtered at load to delivery times before 1 January 2024 New York time, and every loader asserts it. Nothing was committed, pushed or sent.

9 of 33 scored strategies meet Bo's bar; 3 of those 9 are also flagged FRAGILE. Nothing is pending.

**Bottom line.** Seven rows meet Bo's bar at 1 MW: B with gradient boosting (583,946 over three years), idea 10 weather surprise (352,130), the storm-day flip in both builds (320,492 here, 306,733 in the lead's), B deep (310,453), C gradient boosting (169,889) and C deep (165,407). The two storm flips are also FRAGILE. Every passing row except C makes most of its money in 2022: Winter Storm Elliott's two days give B 180,520, idea 10 204,836 and the flip 204,836, and in 2023 B, the flip and B deep lose money while idea 10 earns less than always supply. B's inputs pass an independent rebuild at 05:00 and its date placebos earn far less than it does, so its 2022 is read from public data, but it is one year. Among the passing rows, only C is positive in all three years; its shuffled-date placebos earn most of what it earns (C deep: 142,531 of 165,407), so C's profit is a steady pair stance, not day-by-day timing.

## Ranking by three-year net (USD, full costs, 1 MW per zone-hour)

The bar (registered, unchanged): average at least 50,000 a year, positive in at least 2 of 3 years, three-year Sharpe above 0.42, positive three-year total at a cost of 0.50 per MWh. FRAGILE (coordinator's flag, shown beside PASS): the total changes sign at a neighbouring setting, or turns negative without one year's best 10 days, or random days in the same months beat the rule more than 20 percent of the time.

| # | Strategy | From | 2021 | 2022 | 2023 | Three-year | Without best 5 days | Average a year | Sharpe, 3 years | Total at 0.50 stress | Placebo: score one day late | Placebo: dates shuffled (mean of 20; share at or above real) | Random days beat rule | PASS | FRAGILE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea14_learned_allocator_candidates | model/deep_alloc14.py | 136,420 | 1,489,127 | -104,490 | 1,521,057 | 749,048 | 507,019 | 1.68 | 1,265,855 | 1,442,852 | 1,410,132 (0.20) | 0% | PASS | no |
| 2 | B_gbm | lead | 51,187 | 555,120 | -22,361 | 583,946 | 329,676 | 194,649 | 1.96 | 487,581 | 92,197 | -16,950 (0.00) | 0% | PASS | no |
| 3 | idea10_weather_surprise | lab | 3,034 | 311,142 | 37,955 | 352,130 | 88,091 | 117,377 | 1.03 | 240,551 | 120,955 | -154,089 (0.00) | 0% | PASS | no |
| 4 | idea2_storm_flip | lab | 78,415 | 290,469 | -48,392 | 320,492 | 38,019 | 106,831 | 0.94 | 208,088 | -104,338 | -63,194 (0.05) | 6% | PASS | FRAGILE |
| 5 | B_deep | lead | 35,464 | 310,719 | -35,731 | 310,453 | 70,950 | 103,484 | 1.18 | 215,302 | 149,007 | 123,723 (0.00) | 0% | PASS | no |
| 6 | storm_day_flip [lead] | lead | 74,421 | 282,065 | -49,753 | 306,733 | 24,260 | 102,244 | 0.90 | 194,333 | 76,347 | -51,414 (0.05) | 7% | PASS | FRAGILE |
| 7 | idea5_deep_zone_day | deep agent | 20,081 | 149,388 | 46,173 | 215,642 | -3,489 | 71,881 | 0.66 | 110,737 | -24,708 | -163,819 (0.00) | 0% | PASS | FRAGILE |
| 8 | idea11_learned_allocator | policy agent | -28,374 | 209,765 | -11,074 | 170,316 | -12,425 | 56,772 | 0.70 | 116,108 | -52,415 | -12,930 (0.00) | 3% | no | FRAGILE |
| 9 | C_gbm | lead | 41,457 | 124,029 | 4,403 | 169,889 | 107,596 | 56,630 | 1.67 | 81,574 | 132,664 | 20,889 (0.00) | 3% | PASS | no |
| 10 | idea12_policy_network | policy agent | -11,441 | 206,885 | -29,216 | 166,228 | -19,658 | 55,409 | 0.67 | 126,143 | -94,333 | -28,666 (0.00) | 2% | no | FRAGILE |
| 11 | C_deep | lead | 57,838 | 73,206 | 34,363 | 165,407 | 122,663 | 55,136 | 2.15 | 74,670 | 173,231 | 142,531 (0.30) | 2% | PASS | no |
| 12 | D_gbm | lead | 40,450 | 145,900 | -39,033 | 147,317 | 36,893 | 49,106 | 0.84 | 39,622 | -204,299 | -33,258 (0.10) | 0% | no | FRAGILE |
| 13 | idea7_gbm_storm_filter | lab | 7,643 | 70,444 | 30,689 | 108,777 | 48,148 | 36,259 | 0.80 | 20,075 | -15,042 | -103,164 (0.05) | 8% | no | no |
| 14 | idea1_storm_filter | lab | 41,620 | 49,655 | 15,866 | 107,141 | 44,273 | 35,714 | 0.91 | 46,740 | -110,963 | -88,870 (0.05) | 6% | no | FRAGILE |
| 15 | storm_day_filter [lead] | lead | 39,476 | 45,535 | 15,185 | 100,196 | 37,328 | 33,399 | 0.85 | 39,467 | -15,021 | -78,877 (0.05) | 7% | no | FRAGILE |
| 16 | A_regression_v1 | lead | 25,769 | 32,400 | 39,058 | 97,227 | 41,713 | 32,409 | 0.72 | 41,546 | -55,691 | -95,941 (0.00) | 0% | no | no |
| 17 | risk_sized_supply_gbm | lead | -8,110 | 45,128 | 21,425 | 58,443 | 32,314 | 19,481 | 0.89 | 3,685 | -41,339 | -51,954 (0.00) | 0% | no | no |
| 18 | A_spike_gbm | lead | -16,822 | 46,476 | 20,542 | 50,195 | 32,268 | 16,732 | 0.73 | -27,212 | -14,935 | -23,979 (0.05) | 6% | no | no |
| 19 | idea9_tail_spike_S100 | lead | -2,378 | 377 | 49,792 | 47,791 | 4,862 | 15,930 | 0.34 | -55,416 | -39,993 | -114,135 (0.00) | 1% | no | FRAGILE |
| 20 | A_spike_tail_S100_gbm [lead] | lead | -2,378 | -28,960 | 49,792 | 18,454 | -44,893 | 6,151 | 0.10 | -88,208 | -113,574 | -130,668 (0.00) | 0% | no | FRAGILE |
| 21 | A_spike_deep_rolling_cut | lead | 12,369 | -59,902 | 22,117 | -25,416 | -88,924 | -8,472 | -0.14 | -117,796 | -116,556 | -102,940 (0.05) | 14% | no | FRAGILE |
| 22 | idea8_storm_zone_pairs | lab | 51,935 | -116,133 | 36,071 | -28,128 | -93,352 | -9,376 | -0.09 | -111,088 | -67,312 | -77,447 (0.25) | 25% | no | FRAGILE |
| 23 | A_spike_deep | lead | -9,241 | -31,639 | -1,063 | -41,943 | -63,275 | -13,981 | -0.62 | -103,322 | -61,666 | 587 (0.85) | 84% | no | FRAGILE |
| 24 | D_deep | lead | 11,714 | -19,132 | -40,505 | -47,922 | -98,397 | -15,974 | -0.30 | -153,431 | -18,426 | 108,972 (0.95) | 24% | no | FRAGILE |
| 25 | zone_subset_supply | lead | 9,665 | -89,969 | 27,324 | -52,979 | -106,127 | -17,660 | -0.30 | -113,497 | -51,405 | -53,071 (0.20) | 100% | no | FRAGILE |
| 26 | idea6_deep_two_gates | deep agent | -22,823 | -31,639 | -2,425 | -56,888 | -79,002 | -18,962 | -0.81 | -118,889 | -35,972 | 18,301 (0.95) | 84% | no | FRAGILE |
| 27 | baseline_usual_side | lab | 12,655 | -8,643 | -72,836 | -68,824 | -126,628 | -22,941 | -0.81 | -181,321 | -73,235 | 221,654 (1.00) | 100% | no | FRAGILE |
| 28 | A_regression_v1_deep | lead | 9,335 | -91,649 | 12,745 | -69,568 | -121,975 | -23,190 | -0.33 | -123,835 | -37,350 | -11,881 (0.95) | 16% | no | FRAGILE |
| 29 | A_hourly_mean | lab | 7,484 | -87,770 | 2,638 | -77,648 | -107,266 | -25,883 | -0.52 | -126,549 | -83,839 | 61,731 (1.00) | 100% | no | FRAGILE |
| 30 | side_past_spike_rate_filter | lab | 7,360 | -136,202 | 37,604 | -91,237 | -163,955 | -30,413 | -0.33 | -184,346 | -91,171 | -91,233 (0.90) | 20% | no | FRAGILE |
| 31 | always_supply | lab | -5,015 | -198,269 | 68,154 | -135,130 | -216,721 | -45,043 | -0.40 | -246,079 | n/a | n/a | n/a | no | FRAGILE |
| 32 | idea4_deep_storm_three_way | deep agent | -12,724 | -198,269 | 10,172 | -200,822 | -287,211 | -66,940 | -0.60 | -302,045 | 9,237 | -52,673 (1.00) | 61% | no | FRAGILE |
| 33 | idea3_deep_storm_filter | deep agent | -12,724 | -198,269 | 9,337 | -201,657 | -280,817 | -67,219 | -0.60 | -301,687 | -87,420 | -70,687 (1.00) | 74% | no | FRAGILE |

## What each strategy does

- **idea14_learned_allocator_candidates** (idea 14; from model/deep_alloc14.py (idea14_wf_2021_2023)). A deep gate splits one unit of risk per day between C deep, B gbm, C gbm and B deep, each pre-scaled to a 100,000 USD drawdown with scales from data before each refit; trained on earlier days' realised profits, pulled toward equal weights (registered after the table was seen).
- **B_gbm** (idea B (gbm); from lead modeller (model/evaluate.py positions)). Trade either side where a forecast using weather forecasts expects the gap to beat the cost.
- **idea10_weather_surprise** (idea 10; from lab). Virtual load all day when the latest GFS run, against the run before it, moves D+1's temperatures further from 18.3 C than on most days of the trailing year (rank above a cut); otherwise supply.
- **idea2_storm_flip** (idea 2; from lab). Same storm score; virtual load in every zone-hour on flagged days, supply on the others.
- **B_deep** (idea B (deep); from lead modeller (model/evaluate.py positions)). As B, with the deep model.
- **storm_day_flip [lead]** (idea 2 (lead's build); from lead modeller (model/evaluate.py positions)). As the storm-day filter, but on storm days buy virtual load everywhere instead of sitting out.
- **idea5_deep_zone_day** (idea 5; from deep agent (deep_day_wf_2021_2023)). The deep model predicts each zone's next-day supply profit after cost; supply that zone all day when the prediction is above zero, virtual load when it is below minus (the load break-even plus a margin chosen on the year before), otherwise nothing.
- **idea11_learned_allocator** (idea 11; from policy agent (deep_policy_wf_2021_2023)). A deep gate network weights four simple experts per zone and day (always supply, sit out, virtual load, the storm-day zone pair), trained on their realised profit in earlier days; the position is the weighted mix, held all day.
- **C_gbm** (idea C (gbm); from lead modeller (model/evaluate.py positions)). Buy one zone and sell another in five fixed zone pairs when the forecast spread beats both legs' costs.
- **idea12_policy_network** (idea 12; from policy agent (deep_policy_wf_2021_2023)). A deep network outputs each zone's position for the next day directly, from minus 1 (supply) to plus 1 (load), trained on next-day profit after costs minus a penalty on losses; held all day.
- **C_deep** (idea C (deep); from lead modeller (model/evaluate.py positions)). As C (same pairs), with the deep model's forecasts.
- **D_gbm** (idea D (gbm); from lead modeller (model/evaluate.py positions)). Trade either side where a forecast using scheduled transmission outages expects the gap to beat the cost.
- **idea7_gbm_storm_filter** (idea 7; from lab (LightGBM, monthly refits from April 2020)). LightGBM reads the deep agent's day features and gives the probability that the next day's always-supply book loses more than the worst-tenth day of its training window; sit out the whole day when it is above a cut chosen on the year before, otherwise supply.
- **idea1_storm_filter** (idea 1; from lab). Sit out the whole day when the hand-made storm score (highest of four trailing-year percentile ranks: peak load forecast, coldest and hottest forecast temperature, the gap already published for D) is above a cut; otherwise supply everywhere.
- **storm_day_filter [lead]** (idea 1 (lead's build); from lead modeller (model/evaluate.py positions)). Sell supply everywhere, but skip whole days whose forecast load, forecast cold or heat, or this morning's real-time gap is extreme for the past year.
- **A_regression_v1** (idea A v1 (superseded); from lead modeller (model/evaluate.py positions)). Sell supply where a gradient-boosting forecast of the gap is below minus the cost (A as first registered).
- **risk_sized_supply_gbm** (idea side (risk sizing); from lead modeller (model/evaluate.py positions)). Sell supply everywhere, sized down from 1 MW in proportion to how high the hour's spike score ranks in the last 30 days.
- **A_spike_gbm** (idea A (main, gbm); from lead modeller (model/evaluate.py positions)). Sell supply everywhere except hours a gradient-boosting model flags as likely real-time spikes.
- **idea9_tail_spike_S100** (idea 9; from lead modeller (cache/wf/spike_S100_*.parquet)). Idea A with spikes defined as real time at least 100 USD above day ahead: supply in every zone-hour except where the gradient-boosting probability of such a spike exceeds p* (0.02 or 0.05, chosen on the year before).
- **A_spike_tail_S100_gbm [lead]** (idea 9 (lead's choice of p*); from lead modeller (model/evaluate.py positions)). As A spike flag, but the model looks only for extreme spikes (real time 100 USD/MWh or more above day ahead).
- **A_spike_deep_rolling_cut** (idea side (deep rolling cut); from lead modeller (model/evaluate.py positions)). Sell supply everywhere except hours whose deep spike score is in the top k of the last 30 days (coordinator's rule).
- **idea8_storm_zone_pairs** (idea 8; from lab). On flagged storm days, load in N.Y.C. and LONGIL against supply in two upstate zones (cut and the two zones chosen on the year before); supply everywhere on the other days.
- **A_spike_deep** (idea A (main, deep); from lead modeller (model/evaluate.py positions)). As A spike flag, with the deep model's spike probability.
- **D_deep** (idea D (deep); from lead modeller (model/evaluate.py positions)). As D, with the deep model.
- **zone_subset_supply** (idea side (zone subset); from lead modeller (model/evaluate.py positions)). Sell supply every hour, but only in up to six zones that earned without their worst hours and had the mildest worst day in earlier years.
- **idea6_deep_two_gates** (idea 6; from deep agent (deep_day_wf and deep_wf_spike)). Idea 3's day gate (sit out days the deep day model flags), then inside the days that pass, the deep hourly spike model: supply every zone-hour except where its spike probability is above p*.
- **baseline_usual_side** (idea basic; from lab). Each zone-hour takes the side of its average gap over the trailing 365 days (supply if real time usually settles below day ahead, load if above).
- **A_regression_v1_deep** (idea A v1 (deep); from lead modeller (model/evaluate.py positions)). As A regression, with the deep model's forecast.
- **A_hourly_mean** (idea basic (idea A, simple); from lab). Supply only in the zone-hours whose trailing 365-day average gap is at or below minus the supply cost; no model.
- **side_past_spike_rate_filter** (idea side (spike_value.py history filter); from lab). Supply everywhere except zone-hours whose spike frequency (real time 25 USD or more above day ahead) in all earlier years is in the top k of the previous 30 days' scores; k chosen on the year before.
- **always_supply** (idea basic; from lab). Virtual supply in every zone-hour, every day.
- **idea4_deep_storm_three_way** (idea 4; from deep agent (deep_day_wf_2021_2023)). Same deep probability with two cuts: supply below the first, sit out between, virtual load in every zone-hour at or above the second.
- **idea3_deep_storm_filter** (idea 3; from deep agent (deep_day_wf_2021_2023)). The deep day model's probability that the next day's always-supply book has a worst-tenth loss; sit out the whole day when it is above a cut, otherwise supply.

## Each year

Return is net over the 500,000 USD bankroll. Sharpe is daily, times the square root of 365, with zero days included. Drawdown runs from the start of the year.

| Strategy | Year | Net | Return | Sharpe | Max drawdown | Worst day | Best day | Without best 3 days | Without best 5 days | Without best 10 days | MWh a day | At 0.50 stress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea14_learned_allocator_candidates | 2021 | 136,420 | 27.3% | 1.52 | -63,561 | -44,201 (2021-08-27) | 46,729 (2021-08-22) | 55,654 | 26,661 | -26,402 | 641.9 | 40,905 |
| idea14_learned_allocator_candidates | 2022 | 1,489,127 | 297.8% | 2.99 | -58,897 | -16,412 (2022-08-06) | 459,341 (2022-12-24) | 833,205 | 748,507 | 600,090 | 613.7 | 1,400,145 |
| idea14_learned_allocator_candidates | 2023 | -104,490 | -20.9% | -0.87 | -182,187 | -55,956 (2023-02-04) | 69,358 (2023-01-11) | -227,320 | -244,610 | -274,723 | 531.8 | -175,195 |
| idea14_learned_allocator_candidates | 2021 to 2023 | 1,521,057 | 101.4% a year | 1.68 | -182,187 | -55,956 (2023-02-04) | 459,341 (2022-12-24) | 865,135 | 749,048 | | 595.8 | 1,265,855 |
| B_gbm | 2021 | 51,187 | 10.2% | 2.43 | -14,644 | -3,159 (2021-11-15) | 7,256 (2021-06-28) | 34,042 | 25,318 | 8,467 | 174.8 | 25,138 |
| B_gbm | 2022 | 555,120 | 111.0% | 3.35 | -22,350 | -12,800 (2022-06-13) | 141,753 (2022-12-24) | 338,604 | 300,850 | 216,206 | 257.0 | 517,859 |
| B_gbm | 2023 | -22,361 | -4.5% | -0.62 | -43,515 | -12,506 (2023-02-04) | 15,546 (2023-01-11) | -57,293 | -65,610 | -80,683 | 248.5 | -55,416 |
| B_gbm | 2021 to 2023 | 583,946 | 38.9% a year | 1.96 | -43,515 | -12,800 (2022-06-13) | 141,753 (2022-12-24) | 367,430 | 329,676 | | 226.8 | 487,581 |
| idea10_weather_surprise | 2021 | 3,034 | 0.6% | 0.08 | -63,111 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -24,128 | -33,678 | -51,279 | 264.0 | -36,034 |
| idea10_weather_surprise | 2022 | 311,142 | 62.2% | 1.67 | -58,128 | -35,598 (2022-01-16) | 148,035 (2022-12-24) | 86,263 | 47,102 | -17,265 | 264.0 | 272,985 |
| idea10_weather_surprise | 2023 | 37,955 | 7.6% | 0.77 | -47,560 | -20,681 (2023-09-05) | 16,074 (2023-01-11) | 574 | -12,032 | -34,593 | 264.0 | 3,600 |
| idea10_weather_surprise | 2021 to 2023 | 352,130 | 23.5% a year | 1.03 | -82,452 | -35,598 (2022-01-16) | 148,035 (2022-12-24) | 127,251 | 88,091 | | 264.0 | 240,551 |
| idea2_storm_flip | 2021 | 78,415 | 15.7% | 2.07 | -22,425 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | 50,887 | 40,238 | 16,457 | 264.0 | 39,148 |
| idea2_storm_flip | 2022 | 290,469 | 58.1% | 1.55 | -134,609 | -26,790 (2022-03-29) | 148,035 (2022-12-24) | 50,089 | 8,573 | -70,276 | 264.0 | 252,236 |
| idea2_storm_flip | 2023 | -48,392 | -9.7% | -0.98 | -104,544 | -16,146 (2023-01-11) | 20,609 (2023-09-05) | -93,733 | -104,120 | -125,999 | 264.0 | -83,296 |
| idea2_storm_flip | 2021 to 2023 | 320,492 | 21.4% a year | 0.94 | -134,609 | -26,790 (2022-03-29) | 148,035 (2022-12-24) | 80,112 | 38,019 | | 264.0 | 208,088 |
| B_deep | 2021 | 35,464 | 7.1% | 1.98 | -11,512 | -3,125 (2021-08-22) | 5,257 (2021-07-07) | 22,427 | 15,711 | 1,912 | 172.1 | 9,847 |
| B_deep | 2022 | 310,719 | 62.1% | 2.12 | -23,080 | -11,083 (2022-12-26) | 127,580 (2022-12-24) | 107,290 | 74,048 | 27,126 | 256.0 | 273,591 |
| B_deep | 2023 | -35,731 | -7.1% | -1.07 | -68,205 | -12,472 (2023-02-04) | 16,018 (2023-01-11) | -59,690 | -66,363 | -78,486 | 245.7 | -68,136 |
| B_deep | 2021 to 2023 | 310,453 | 20.7% a year | 1.18 | -69,152 | -12,472 (2023-02-04) | 127,580 (2022-12-24) | 107,024 | 70,950 | | 224.6 | 215,302 |
| storm_day_flip [lead] | 2021 | 74,421 | 14.9% | 1.96 | -22,425 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | 46,892 | 36,243 | 12,463 | 264.0 | 35,160 |
| storm_day_flip [lead] | 2022 | 282,065 | 56.4% | 1.51 | -134,282 | -26,790 (2022-03-29) | 148,035 (2022-12-24) | 41,685 | 169 | -78,680 | 264.0 | 243,829 |
| storm_day_flip [lead] | 2023 | -49,753 | -10.0% | -1.01 | -102,967 | -16,146 (2023-01-11) | 20,609 (2023-09-05) | -95,094 | -105,480 | -127,360 | 264.0 | -84,656 |
| storm_day_flip [lead] | 2021 to 2023 | 306,733 | 20.4% a year | 0.90 | -134,282 | -26,790 (2022-03-29) | 148,035 (2022-12-24) | 66,353 | 24,260 | | 264.0 | 194,333 |
| idea5_deep_zone_day | 2021 | 20,081 | 4.0% | 0.55 | -42,311 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -7,448 | -16,506 | -33,988 | 245.7 | -16,312 |
| idea5_deep_zone_day | 2022 | 149,388 | 29.9% | 0.83 | -134,636 | -35,598 (2022-01-16) | 148,035 (2022-12-24) | -39,259 | -65,828 | -119,606 | 252.7 | 112,878 |
| idea5_deep_zone_day | 2023 | 46,173 | 9.2% | 1.06 | -33,919 | -14,606 (2023-09-06) | 16,074 (2023-01-11) | 8,862 | -3,927 | -27,872 | 244.7 | 14,171 |
| idea5_deep_zone_day | 2021 to 2023 | 215,642 | 14.4% a year | 0.66 | -169,814 | -35,598 (2022-01-16) | 148,035 (2022-12-24) | 26,995 | -3,489 | | 247.7 | 110,737 |
| idea11_learned_allocator | 2021 | -28,374 | -5.7% | -1.09 | -38,433 | -7,400 (2021-01-29) | 5,988 (2021-06-28) | -43,410 | -50,653 | -64,836 | 170.5 | -53,659 |
| idea11_learned_allocator | 2022 | 209,765 | 42.0% | 1.55 | -27,776 | -26,420 (2022-12-23) | 128,078 (2022-12-24) | 54,027 | 32,121 | -353 | 74.7 | 198,934 |
| idea11_learned_allocator | 2023 | -11,074 | -2.2% | -0.42 | -39,991 | -10,726 (2023-02-03) | 15,973 (2023-01-11) | -34,092 | -38,707 | -48,693 | 136.2 | -29,167 |
| idea11_learned_allocator | 2021 to 2023 | 170,316 | 11.4% a year | 0.70 | -55,944 | -26,420 (2022-12-23) | 128,078 (2022-12-24) | 12,033 | -12,425 | | 127.1 | 116,108 |
| C_gbm | 2021 | 41,457 | 8.3% | 1.67 | -21,011 | -6,429 (2021-08-08) | 8,819 (2021-08-12) | 21,869 | 13,703 | -3,934 | 221.0 | 8,599 |
| C_gbm | 2022 | 124,029 | 24.8% | 2.48 | -29,328 | -12,568 (2022-05-10) | 23,803 (2022-12-25) | 77,563 | 63,655 | 37,937 | 222.3 | 91,799 |
| C_gbm | 2023 | 4,403 | 0.9% | 0.25 | -15,347 | -5,056 (2023-02-05) | 6,347 (2023-02-28) | -10,627 | -16,000 | -25,117 | 175.1 | -18,824 |
| C_gbm | 2021 to 2023 | 169,889 | 11.3% a year | 1.67 | -29,328 | -12,568 (2022-05-10) | 23,803 (2022-12-25) | 123,423 | 107,596 | | 206.1 | 81,574 |
| idea12_policy_network | 2021 | -11,441 | -2.3% | -0.48 | -32,870 | -7,314 (2021-01-29) | 7,246 (2021-06-28) | -28,419 | -36,295 | -50,212 | 146.2 | -33,141 |
| idea12_policy_network | 2022 | 206,885 | 41.4% | 1.49 | -32,512 | -32,512 (2022-12-23) | 130,325 (2022-12-24) | 48,397 | 25,835 | -6,912 | 53.4 | 199,152 |
| idea12_policy_network | 2023 | -29,216 | -5.8% | -1.13 | -42,542 | -12,469 (2023-02-04) | 15,960 (2023-01-11) | -49,688 | -53,670 | -61,688 | 79.2 | -39,869 |
| idea12_policy_network | 2021 to 2023 | 166,228 | 11.1% a year | 0.67 | -56,728 | -32,512 (2022-12-23) | 130,325 (2022-12-24) | 4,836 | -19,658 | | 92.9 | 126,143 |
| C_deep | 2021 | 57,838 | 11.6% | 2.30 | -15,516 | -10,990 (2021-08-27) | 10,946 (2021-08-22) | 36,402 | 28,370 | 11,902 | 221.4 | 24,919 |
| C_deep | 2022 | 73,206 | 14.6% | 2.52 | -25,500 | -4,204 (2022-12-29) | 7,413 (2022-06-08) | 51,846 | 40,525 | 17,468 | 214.2 | 42,158 |
| C_deep | 2023 | 34,363 | 6.9% | 1.55 | -12,245 | -11,384 (2023-02-04) | 8,940 (2023-02-27) | 11,170 | 5,717 | -3,048 | 201.8 | 7,593 |
| C_deep | 2021 to 2023 | 165,407 | 11.0% a year | 2.15 | -25,500 | -11,384 (2023-02-04) | 10,946 (2021-08-22) | 137,162 | 122,663 | | 212.5 | 74,670 |
| D_gbm | 2021 | 40,450 | 8.1% | 1.41 | -33,991 | -10,862 (2021-02-19) | 7,160 (2021-06-28) | 21,168 | 12,884 | -5,291 | 256.5 | 2,323 |
| D_gbm | 2022 | 145,900 | 29.2% | 1.61 | -48,109 | -30,520 (2022-01-16) | 58,259 (2022-12-24) | 62,325 | 39,203 | -7,947 | 260.7 | 108,118 |
| D_gbm | 2023 | -39,033 | -7.8% | -1.21 | -47,689 | -12,689 (2023-09-06) | 15,218 (2023-01-11) | -63,662 | -71,195 | -84,708 | 241.0 | -70,818 |
| D_gbm | 2021 to 2023 | 147,317 | 9.8% a year | 0.84 | -48,109 | -30,520 (2022-01-16) | 58,259 (2022-12-24) | 60,677 | 36,893 | | 252.7 | 39,622 |
| idea7_gbm_storm_filter | 2021 | 7,643 | 1.5% | 0.23 | -53,675 | -7,305 (2021-06-28) | 10,813 (2021-02-19) | -19,518 | -27,495 | -43,893 | 222.0 | -25,205 |
| idea7_gbm_storm_filter | 2022 | 70,444 | 14.1% | 1.22 | -61,450 | -20,097 (2022-06-13) | 13,966 (2022-12-28) | 36,183 | 21,977 | -9,779 | 166.4 | 46,413 |
| idea7_gbm_storm_filter | 2023 | 30,689 | 6.1% | 0.74 | -67,173 | -20,681 (2023-09-05) | 14,410 (2023-07-28) | 4,434 | -4,255 | -22,876 | 248.8 | -1,133 |
| idea7_gbm_storm_filter | 2021 to 2023 | 108,777 | 7.3% a year | 0.80 | -90,322 | -20,681 (2023-09-05) | 14,410 (2023-07-28) | 68,421 | 48,148 | | 212.4 | 20,075 |
| idea1_storm_filter | 2021 | 41,620 | 8.3% | 1.65 | -18,263 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | 17,617 | 11,197 | -2,869 | 118.6 | 24,077 |
| idea1_storm_filter | 2022 | 49,655 | 9.9% | 0.85 | -87,791 | -26,790 (2022-03-29) | 21,484 (2022-12-27) | 7,060 | -5,992 | -31,445 | 169.2 | 25,210 |
| idea1_storm_filter | 2023 | 15,866 | 3.2% | 0.66 | -21,674 | -16,146 (2023-01-11) | 4,391 (2023-08-19) | 3,980 | -1,039 | -11,901 | 144.0 | -2,547 |
| idea1_storm_filter | 2021 to 2023 | 107,141 | 7.1% a year | 0.91 | -93,495 | -26,790 (2022-03-29) | 21,484 (2022-12-27) | 60,878 | 44,273 | | 143.9 | 46,740 |
| storm_day_filter [lead] | 2021 | 39,476 | 7.9% | 1.56 | -19,784 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | 15,473 | 9,053 | -5,013 | 122.9 | 21,291 |
| storm_day_filter [lead] | 2022 | 45,535 | 9.1% | 0.78 | -88,689 | -26,790 (2022-03-29) | 21,484 (2022-12-27) | 2,939 | -10,113 | -35,248 | 167.1 | 21,403 |
| storm_day_filter [lead] | 2023 | 15,185 | 3.0% | 0.63 | -23,179 | -16,146 (2023-01-11) | 4,391 (2023-08-19) | 3,300 | -1,720 | -12,581 | 144.0 | -3,227 |
| storm_day_filter [lead] | 2021 to 2023 | 100,196 | 6.7% a year | 0.85 | -99,762 | -26,790 (2022-03-29) | 21,484 (2022-12-27) | 53,933 | 37,328 | | 144.7 | 39,467 |
| A_regression_v1 | 2021 | 25,769 | 5.2% | 1.35 | -17,060 | -5,263 (2021-11-05) | 2,788 (2021-11-24) | 17,486 | 12,211 | 1,035 | 135.0 | 5,799 |
| A_regression_v1 | 2022 | 32,400 | 6.5% | 0.44 | -54,319 | -31,715 (2022-01-16) | 12,153 (2022-12-26) | -3,074 | -23,114 | -58,198 | 153.0 | 10,308 |
| A_regression_v1 | 2023 | 39,058 | 7.8% | 2.33 | -19,095 | -5,774 (2023-09-06) | 5,564 (2023-07-28) | 24,837 | 19,014 | 9,889 | 106.5 | 25,439 |
| A_regression_v1 | 2021 to 2023 | 97,227 | 6.5% a year | 0.72 | -58,514 | -31,715 (2022-01-16) | 12,153 (2022-12-26) | 61,753 | 41,713 | | 131.5 | 41,546 |
| risk_sized_supply_gbm | 2021 | -8,110 | -1.6% | -0.62 | -21,899 | -3,555 (2021-01-01) | 2,100 (2021-02-24) | -14,159 | -17,365 | -23,624 | 127.1 | -26,918 |
| risk_sized_supply_gbm | 2022 | 45,128 | 9.0% | 1.35 | -28,074 | -12,821 (2022-12-24) | 7,916 (2022-12-25) | 27,352 | 18,999 | 3,134 | 127.3 | 26,743 |
| risk_sized_supply_gbm | 2023 | 21,425 | 4.3% | 1.73 | -16,805 | -3,936 (2023-01-11) | 2,755 (2023-02-01) | 14,553 | 10,884 | 3,307 | 137.3 | 3,860 |
| risk_sized_supply_gbm | 2021 to 2023 | 58,443 | 3.9% a year | 0.89 | -42,710 | -12,821 (2022-12-24) | 7,916 (2022-12-25) | 40,667 | 32,314 | | 130.6 | 3,685 |
| A_spike_gbm | 2021 | -16,822 | -3.4% | -0.60 | -44,997 | -7,708 (2021-01-29) | 3,728 (2021-01-31) | -27,132 | -32,918 | -46,090 | 239.0 | -52,180 |
| A_spike_gbm | 2022 | 46,476 | 9.3% | 2.15 | -22,302 | -9,081 (2022-01-04) | 3,846 (2022-06-19) | 36,532 | 31,113 | 19,949 | 115.8 | 29,753 |
| A_spike_gbm | 2023 | 20,542 | 4.1% | 1.15 | -27,565 | -3,652 (2023-11-10) | 3,691 (2023-08-19) | 11,010 | 5,973 | -3,107 | 198.0 | -4,784 |
| A_spike_gbm | 2021 to 2023 | 50,195 | 3.3% a year | 0.73 | -60,197 | -9,081 (2022-01-04) | 3,846 (2022-06-19) | 38,930 | 32,268 | | 184.3 | -27,212 |
| idea9_tail_spike_S100 | 2021 | -2,378 | -0.5% | -0.06 | -66,623 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -27,811 | -35,587 | -52,376 | 262.0 | -41,135 |
| idea9_tail_spike_S100 | 2022 | 377 | 0.1% | 0.01 | -83,235 | -35,824 (2022-01-16) | 8,194 (2022-02-28) | -22,900 | -36,739 | -66,226 | 227.6 | -32,494 |
| idea9_tail_spike_S100 | 2023 | 49,792 | 10.0% | 1.68 | -39,422 | -8,421 (2023-08-11) | 6,253 (2023-09-08) | 32,270 | 23,041 | 5,667 | 246.9 | 18,214 |
| idea9_tail_spike_S100 | 2021 to 2023 | 47,791 | 3.2% a year | 0.34 | -142,728 | -35,824 (2022-01-16) | 10,813 (2021-02-19) | 19,945 | 4,862 | | 245.5 | -55,416 |
| A_spike_tail_S100_gbm [lead] | 2021 | -2,378 | -0.5% | -0.06 | -66,623 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -27,811 | -35,587 | -52,376 | 262.0 | -41,135 |
| A_spike_tail_S100_gbm [lead] | 2022 | -28,960 | -5.8% | -0.30 | -96,235 | -38,708 (2022-12-24) | 16,032 (2022-12-27) | -69,970 | -92,307 | -133,937 | 251.5 | -65,287 |
| A_spike_tail_S100_gbm [lead] | 2023 | 49,792 | 10.0% | 1.68 | -39,422 | -8,421 (2023-08-11) | 6,253 (2023-09-08) | 32,270 | 23,041 | 5,667 | 246.9 | 18,214 |
| A_spike_tail_S100_gbm [lead] | 2021 to 2023 | 18,454 | 1.2% a year | 0.10 | -155,726 | -38,708 (2022-12-24) | 16,032 (2022-12-27) | -22,556 | -44,893 | | 253.5 | -88,208 |
| A_spike_deep_rolling_cut | 2021 | 12,369 | 2.5% | 0.40 | -48,231 | -5,761 (2021-06-28) | 10,634 (2021-02-19) | -13,823 | -20,073 | -33,846 | 237.3 | -22,732 |
| A_spike_deep_rolling_cut | 2022 | -59,902 | -12.0% | -0.61 | -125,741 | -43,259 (2022-12-24) | 18,458 (2022-12-27) | -103,002 | -121,883 | -161,322 | 227.7 | -92,792 |
| A_spike_deep_rolling_cut | 2023 | 22,117 | 4.4% | 1.18 | -25,952 | -5,551 (2023-01-11) | 4,991 (2023-02-01) | 9,056 | 3,035 | -6,328 | 190.7 | -2,273 |
| A_spike_deep_rolling_cut | 2021 to 2023 | -25,416 | -1.7% a year | -0.14 | -169,905 | -43,259 (2022-12-24) | 18,458 (2022-12-27) | -68,516 | -88,924 | | 218.6 | -117,796 |
| idea8_storm_zone_pairs | 2021 | 51,935 | 10.4% | 1.90 | -19,475 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | 27,773 | 20,793 | 4,957 | 171.5 | 26,531 |
| idea8_storm_zone_pairs | 2022 | -116,133 | -23.2% | -0.67 | -213,497 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -161,084 | -177,990 | -212,715 | 230.9 | -149,487 |
| idea8_storm_zone_pairs | 2023 | 36,071 | 7.2% | 1.41 | -18,905 | -16,146 (2023-01-11) | 4,391 (2023-08-19) | 24,058 | 17,705 | 5,799 | 187.6 | 11,868 |
| idea8_storm_zone_pairs | 2021 to 2023 | -28,128 | -1.9% a year | -0.09 | -213,497 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -74,391 | -93,352 | | 196.6 | -111,088 |
| A_spike_deep | 2021 | -9,241 | -1.8% | -0.41 | -39,057 | -5,480 (2021-06-28) | 5,466 (2021-02-10) | -23,516 | -28,315 | -38,423 | 203.6 | -39,354 |
| A_spike_deep | 2022 | -31,639 | -6.3% | -1.10 | -53,860 | -21,131 (2022-01-16) | 3,226 (2022-01-14) | -40,640 | -45,289 | -55,170 | 84.4 | -43,832 |
| A_spike_deep | 2023 | -1,063 | -0.2% | -0.08 | -24,432 | -4,559 (2023-10-10) | 3,831 (2023-08-19) | -9,053 | -12,112 | -18,448 | 149.1 | -20,136 |
| A_spike_deep | 2021 to 2023 | -41,943 | -2.8% a year | -0.62 | -88,248 | -21,131 (2022-01-16) | 5,466 (2021-02-10) | -56,670 | -63,275 | | 145.7 | -103,322 |
| D_deep | 2021 | 11,714 | 2.3% | 0.59 | -23,121 | -4,202 (2021-06-28) | 5,503 (2021-01-29) | -684 | -7,380 | -20,534 | 247.7 | -25,106 |
| D_deep | 2022 | -19,132 | -3.8% | -0.23 | -88,389 | -55,065 (2022-12-24) | 9,065 (2022-01-19) | -45,399 | -61,746 | -96,924 | 251.6 | -55,605 |
| D_deep | 2023 | -40,505 | -8.1% | -1.09 | -59,871 | -15,397 (2023-09-06) | 15,291 (2023-01-11) | -71,610 | -78,572 | -91,186 | 244.6 | -72,720 |
| D_deep | 2021 to 2023 | -47,922 | -3.2% a year | -0.30 | -102,860 | -55,065 (2022-12-24) | 15,291 (2023-01-11) | -81,195 | -98,397 | | 248.0 | -153,431 |
| zone_subset_supply | 2021 | 9,665 | 1.9% | 0.48 | -30,617 | -3,931 (2021-06-28) | 6,618 (2021-02-10) | -6,491 | -10,824 | -20,571 | 144.0 | -11,638 |
| zone_subset_supply | 2022 | -89,969 | -18.0% | -0.93 | -141,620 | -75,290 (2022-12-24) | 12,522 (2022-12-27) | -126,764 | -142,674 | -170,354 | 144.0 | -110,767 |
| zone_subset_supply | 2023 | 27,324 | 5.5% | 1.04 | -37,845 | -11,337 (2023-09-05) | 7,661 (2023-07-28) | 7,565 | -305 | -12,477 | 144.0 | 8,907 |
| zone_subset_supply | 2021 to 2023 | -52,979 | -3.5% a year | -0.30 | -164,178 | -75,290 (2022-12-24) | 12,522 (2022-12-27) | -89,774 | -106,127 | | 144.0 | -113,497 |
| idea6_deep_two_gates | 2021 | -22,823 | -4.6% | -1.03 | -50,248 | -5,480 (2021-06-28) | 5,466 (2021-02-10) | -37,098 | -41,897 | -51,647 | 187.7 | -50,588 |
| idea6_deep_two_gates | 2022 | -31,639 | -6.3% | -1.10 | -53,860 | -21,131 (2022-01-16) | 3,226 (2022-01-14) | -40,640 | -45,289 | -55,170 | 84.4 | -43,832 |
| idea6_deep_two_gates | 2023 | -2,425 | -0.5% | -0.14 | -34,741 | -5,235 (2023-10-10) | 4,545 (2023-08-19) | -13,124 | -17,821 | -26,769 | 172.4 | -24,469 |
| idea6_deep_two_gates | 2021 to 2023 | -56,888 | -3.8% a year | -0.81 | -101,830 | -21,131 (2022-01-16) | 5,466 (2021-02-10) | -72,329 | -79,002 | | 148.2 | -118,889 |
| baseline_usual_side | 2021 | 12,655 | 2.5% | 0.80 | -16,181 | -3,284 (2021-03-16) | 5,322 (2021-01-29) | -1,207 | -7,111 | -17,526 | 264.0 | -26,579 |
| baseline_usual_side | 2022 | -8,643 | -1.7% | -0.22 | -51,164 | -12,922 (2022-12-27) | 13,872 (2022-12-23) | -44,949 | -61,553 | -85,140 | 264.0 | -46,960 |
| baseline_usual_side | 2023 | -72,836 | -14.6% | -3.05 | -79,740 | -11,412 (2023-09-05) | 12,490 (2023-01-11) | -91,239 | -96,157 | -105,751 | 264.0 | -107,782 |
| baseline_usual_side | 2021 to 2023 | -68,824 | -4.6% a year | -0.81 | -124,750 | -12,922 (2022-12-27) | 13,872 (2022-12-23) | -107,119 | -126,628 | | 264.0 | -181,321 |
| A_regression_v1_deep | 2021 | 9,335 | 1.9% | 0.41 | -32,295 | -5,059 (2021-07-07) | 5,995 (2021-02-10) | -4,738 | -10,381 | -21,933 | 141.2 | -11,556 |
| A_regression_v1_deep | 2022 | -91,649 | -18.3% | -0.78 | -128,937 | -93,251 (2022-12-24) | 11,592 (2022-01-19) | -125,144 | -144,055 | -176,733 | 127.2 | -110,027 |
| A_regression_v1_deep | 2023 | 12,745 | 2.5% | 0.49 | -43,513 | -14,091 (2023-09-05) | 5,966 (2023-09-08) | -503 | -6,924 | -17,750 | 117.3 | -2,252 |
| A_regression_v1_deep | 2021 to 2023 | -69,568 | -4.6% a year | -0.33 | -154,103 | -93,251 (2022-12-24) | 11,592 (2022-01-19) | -103,063 | -121,975 | | 128.6 | -123,835 |
| A_hourly_mean | 2021 | 7,484 | 1.5% | 0.38 | -33,973 | -3,777 (2021-03-16) | 5,940 (2021-02-19) | -5,455 | -9,827 | -18,454 | 129.5 | -11,670 |
| A_hourly_mean | 2022 | -87,770 | -17.6% | -1.09 | -108,282 | -67,629 (2022-12-24) | 6,190 (2022-01-14) | -104,977 | -113,854 | -132,390 | 88.0 | -100,475 |
| A_hourly_mean | 2023 | 2,638 | 0.5% | 0.10 | -45,300 | -15,063 (2023-09-05) | 6,472 (2023-07-28) | -11,647 | -17,857 | -27,626 | 133.2 | -14,404 |
| A_hourly_mean | 2021 to 2023 | -77,648 | -5.2% a year | -0.52 | -138,071 | -67,629 (2022-12-24) | 6,472 (2023-07-28) | -96,250 | -107,266 | | 116.9 | -126,549 |
| side_past_spike_rate_filter | 2021 | 7,360 | 1.5% | 0.24 | -53,570 | -5,962 (2021-06-28) | 9,318 (2021-02-19) | -16,200 | -22,639 | -36,306 | 240.1 | -28,154 |
| side_past_spike_rate_filter | 2022 | -136,202 | -27.2% | -0.87 | -208,905 | -123,372 (2022-12-24) | 19,023 (2022-12-27) | -186,080 | -208,920 | -253,227 | 237.2 | -170,466 |
| side_past_spike_rate_filter | 2023 | 37,604 | 7.5% | 1.65 | -31,092 | -5,715 (2023-09-05) | 7,360 (2023-02-04) | 16,567 | 9,030 | -5,304 | 182.4 | 14,273 |
| side_past_spike_rate_filter | 2021 to 2023 | -91,237 | -6.1% a year | -0.33 | -257,699 | -123,372 (2022-12-24) | 19,023 (2022-12-27) | -141,115 | -163,955 | | 219.9 | -184,346 |
| always_supply | 2021 | -5,015 | -1.0% | -0.13 | -73,583 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -32,176 | -40,153 | -57,055 | 264.0 | -44,070 |
| always_supply | 2022 | -198,269 | -39.7% | -1.06 | -277,995 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -252,848 | -277,429 | -327,819 | 264.0 | -236,399 |
| always_supply | 2023 | 68,154 | 13.6% | 1.38 | -67,173 | -20,681 (2023-09-05) | 14,410 (2023-07-28) | 30,681 | 17,254 | -6,691 | 264.0 | 34,389 |
| always_supply | 2021 to 2023 | -135,130 | -9.0% a year | -0.40 | -344,445 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -190,152 | -216,721 | | 264.0 | -246,079 |
| idea4_deep_storm_three_way | 2021 | -12,724 | -2.5% | -0.34 | -81,292 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -39,886 | -47,863 | -64,765 | 245.2 | -48,997 |
| idea4_deep_storm_three_way | 2022 | -198,269 | -39.7% | -1.06 | -277,995 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -252,848 | -277,429 | -327,819 | 264.0 | -236,399 |
| idea4_deep_storm_three_way | 2023 | 10,172 | 2.0% | 0.26 | -40,023 | -12,472 (2023-02-04) | 16,074 (2023-01-11) | -28,167 | -38,202 | -54,251 | 209.1 | -16,649 |
| idea4_deep_storm_three_way | 2021 to 2023 | -200,822 | -13.4% a year | -0.60 | -352,154 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -257,508 | -287,211 | | 239.4 | -302,045 |
| idea3_deep_storm_filter | 2021 | -12,724 | -2.5% | -0.34 | -81,292 | -9,351 (2021-01-29) | 10,813 (2021-02-19) | -39,886 | -47,863 | -64,765 | 245.2 | -48,997 |
| idea3_deep_storm_filter | 2022 | -198,269 | -39.7% | -1.06 | -277,995 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -252,848 | -277,429 | -327,819 | 264.0 | -236,399 |
| idea3_deep_storm_filter | 2023 | 9,337 | 1.9% | 0.38 | -42,880 | -9,067 (2023-08-11) | 6,529 (2023-02-01) | -5,694 | -12,237 | -24,518 | 200.4 | -16,291 |
| idea3_deep_storm_filter | 2021 to 2023 | -201,657 | -13.4% a year | -0.60 | -352,154 | -148,090 (2022-12-24) | 21,484 (2022-12-27) | -256,235 | -280,817 | | 236.5 | -301,687 |

## Fragility

Neighbours: each year's chosen setting moved one step on its grid, in every year at once. The bootstrap is stationary over days (mean block 7, 5,000 draws) on the three-year daily net, shown per year. Random days: the rule's own daily books dealt to random days of the same month, 1,000 draws; percentile of the real rule.

| Strategy | Settings chosen 2021 / 2022 / 2023 | Three-year total at neighbouring settings | 95% interval, USD a year | Random days: percentile of rule | Without each year's best 10 days, three-year total | FRAGILE because |
|---|---|---|---|---|---|---|
| idea14_learned_allocator_candidates | none / none / none | no setting | 149,650 to 1,059,266 | 100.0 | 1,358,236 / 632,020 / 1,350,824 | no |
| B_gbm | none / none / none | no setting | 68,182 to 389,784 | 100.0 | 541,227 / 245,032 / 525,624 | no |
| idea10_weather_surprise | 0.95 / 0.9 / 0.8 | higher 23,925; lower 427,931 | -17,812 to 316,272 | 99.9 | 297,817 / 23,723 / 279,582 | no |
| idea2_storm_flip | 0.8 / 0.9 / 0.8 | higher 11,107; lower 391,019 | -23,250 to 283,970 | 93.6 | 258,534 / -40,253 / 242,885 | three-year total turns negative without one year's best 10 days |
| B_deep | none / none / none | no setting | 16,279 to 229,476 | 99.9 | 276,901 / 26,860 / 267,698 | no |
| storm_day_flip [lead] | 0.8 / 0.9 / 0.8 | higher 37,834; lower 390,472 | -27,630 to 282,182 | 92.9 | 244,775 / -54,012 / 229,126 | three-year total turns negative without one year's best 10 days |
| idea5_deep_zone_day | 0.0 / 25.0 / 0.0 | higher 187,156; lower 220,521 | -52,852 to 243,017 | 99.7 | 161,573 / -53,353 / 141,597 | three-year total turns negative without one year's best 10 days |
| idea11_learned_allocator | none / none / none | no setting | -22,484 to 178,923 | 97.4 | 133,855 / -39,801 / 132,697 | three-year total turns negative without one year's best 10 days |
| C_gbm | none / none / none | no setting | 16,316 to 97,674 | 96.9 | 124,498 / 83,796 / 140,370 | no |
| idea12_policy_network | none / none / none | no setting | -24,930 to 176,988 | 97.8 | 127,457 / -47,569 / 133,756 | three-year total turns negative without one year's best 10 days |
| C_deep | none / none / none | no setting | 28,178 to 83,110 | 98.1 | 119,472 / 109,669 / 127,996 | no |
| D_gbm | none / none / none | no setting | -10,184 to 116,690 | 99.8 | 101,576 / -6,530 / 101,642 | three-year total turns negative without one year's best 10 days |
| idea7_gbm_storm_filter | 0.1 / 0.1 / 0.2 | higher 154,050; lower 122,959 | -25,951 to 98,732 | 92.3 | 57,240 / 28,554 / 55,212 | no |
| idea1_storm_filter | 0.8 / 0.9 / 0.8 | higher -54,388; lower 144,330 | -20,513 to 94,900 | 93.6 | 62,652 / 26,041 / 79,375 | three-year total changes sign at a neighbouring setting |
| storm_day_filter [lead] | 0.8 / 0.9 / 0.8 | higher -40,923; lower 144,018 | -23,250 to 92,966 | 92.9 | 55,707 / 19,414 / 72,430 | three-year total changes sign at a neighbouring setting |
| A_regression_v1 | none / none / none | no setting | -15,366 to 78,438 | 100.0 | 72,494 / 6,628 / 68,058 | no |
| risk_sized_supply_gbm | none / none / none | no setting | -8,942 to 49,129 | 99.7 | 42,928 / 16,449 / 40,324 | no |
| A_spike_gbm | 0.05 / 0.05 / 0.05 | lower 4,096 | -16,352 to 50,188 | 93.7 | 20,928 / 23,668 / 26,547 | no |
| idea9_tail_spike_S100 | 0.05 / 0.02 / 0.02 | higher 26,323; lower 55,289 | -46,756 to 79,898 | 99.4 | -2,207 / -18,812 / 3,666 | three-year total turns negative without one year's best 10 days |
| A_spike_tail_S100_gbm [lead] | 0.05 / 0.05 / 0.02 | higher 26,323; lower 55,289 | -66,430 to 80,336 | 100.0 | -31,544 / -86,523 / -25,671 | three-year total turns negative without one year's best 10 days |
| A_spike_deep_rolling_cut | none / none / none | no setting | -77,052 to 58,327 | 86.5 | -71,631 / -126,836 / -53,860 | three-year total turns negative without one year's best 10 days |
| idea8_storm_zone_pairs | (0.8, (WEST, CENTRL)) / (0.95, (GENESE, CENTRL)) / (0.8, (WEST, NORTH)) | higher cut -18,853; lower cut 160,358 | -146,219 to 90,520 | 74.9 | -75,106 / -124,710 / -58,400 | three-year total changes sign at a neighbouring setting; three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| A_spike_deep | 0.05 / 0.05 / 0.02 | higher -23,280; lower -25,352 | -43,581 to 15,074 | 16.4 | -71,125 / -65,474 / -59,328 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| D_deep | none / none / none | no setting | -78,183 to 37,303 | 75.9 | -80,171 / -125,715 / -98,603 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| zone_subset_supply | none / none / none | no setting | -84,242 to 41,902 | 0.0 | -83,215 / -133,364 / -92,780 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| idea6_deep_two_gates | (0.2, 0.05) / (0.5, 0.05) / (0.1, 0.05) | higher day cut -29,495; lower day cut -55,178; lower p* -41,055 | -49,676 to 11,498 | 15.6 | -85,712 / -80,419 / -81,232 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| baseline_usual_side | none / none / none | no setting | -60,152 to 13,213 | 0.0 | -99,005 / -145,322 / -101,739 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| A_regression_v1_deep | none / none / none | no setting | -107,858 to 43,180 | 83.6 | -100,836 / -154,652 / -100,064 | three-year total turns negative without one year's best 10 days |
| A_hourly_mean | none / none / none | no setting | -90,666 to 26,754 | 0.0 | -103,586 / -122,268 / -107,912 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| side_past_spike_rate_filter | 0.1 / 0.1 / 0.3 | higher -77,362; lower -119,140 | -137,350 to 59,385 | 80.2 | -134,904 / -208,263 / -134,146 | three-year total turns negative without one year's best 10 days |
| always_supply | none / none / none | no setting | -180,784 to 70,482 | not applicable: the same book every day | -187,171 / -264,680 / -209,975 | three-year total turns negative without one year's best 10 days |
| idea4_deep_storm_three_way | (0.2, 0.5) / (0.4, 0.5) / (0.1, 0.2) | higher load cut -230,249; higher sit-out cut -135,384; lower load cut -204,420; lower sit-out cut -101,789 | -202,758 to 40,953 | 38.8 | -252,862 / -330,371 / -265,244 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |
| idea3_deep_storm_filter | 0.2 / 0.5 / 0.1 | higher -136,219; lower -199,065 | -201,736 to 41,610 | 26.5 | -253,697 / -331,206 / -235,511 | three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time |

## Sizing view (not part of the registered verdict)

From the 2021 to 2023 walk-forward only. Sortino uses the downside deviation of daily net (zero days included). Return over max drawdown = average net a year over the three-year max drawdown. Scale = the MW per position that would make the three-year max drawdown 100,000 USD (20 percent of the bankroll), capped at 5; profit, costs and drawdown all scale linearly with it. This is a sizing view only; PASS above is computed at 1 MW as registered.

| Strategy | Sharpe | Sortino | Return over max drawdown | Max drawdown at 1 MW | Scale for a 100,000 drawdown | Return on 500,000 a year at that scale |
|---|---|---|---|---|---|---|
| idea14_learned_allocator_candidates | 1.68 | 7.23 | 2.78 | -182,187 | 0.55 | 55.7% |
| B_gbm | 1.96 | 8.25 | 4.47 | -43,515 | 2.3 | 89.5% |
| idea10_weather_surprise | 1.03 | 2.31 | 1.42 | -82,452 | 1.21 | 28.5% |
| idea2_storm_flip | 0.94 | 2.37 | 0.79 | -134,609 | 0.74 | 15.9% |
| B_deep | 1.18 | 4.50 | 1.50 | -69,152 | 1.45 | 29.9% |
| storm_day_flip [lead] | 0.90 | 2.27 | 0.76 | -134,282 | 0.74 | 15.2% |
| idea5_deep_zone_day | 0.66 | 1.38 | 0.42 | -169,814 | 0.59 | 8.5% |
| idea11_learned_allocator | 0.70 | 2.44 | 1.01 | -55,944 | 1.79 | 20.3% |
| C_gbm | 1.67 | 2.73 | 1.93 | -29,328 | 3.41 | 38.6% |
| idea12_policy_network | 0.67 | 2.21 | 0.98 | -56,728 | 1.76 | 19.5% |
| C_deep | 2.15 | 3.48 | 2.16 | -25,500 | 3.92 | 43.2% |
| D_gbm | 0.84 | 1.37 | 1.02 | -48,109 | 2.08 | 20.4% |
| idea7_gbm_storm_filter | 0.80 | 1.05 | 0.40 | -90,322 | 1.11 | 8.0% |
| idea1_storm_filter | 0.91 | 1.20 | 0.38 | -93,495 | 1.07 | 7.6% |
| storm_day_filter [lead] | 0.85 | 1.12 | 0.33 | -99,762 | 1.0 | 6.7% |
| A_regression_v1 | 0.72 | 0.88 | 0.55 | -58,514 | 1.71 | 11.1% |
| risk_sized_supply_gbm | 0.89 | 1.16 | 0.46 | -42,710 | 2.34 | 9.1% |
| A_spike_gbm | 0.73 | 0.95 | 0.28 | -60,197 | 1.66 | 5.6% |
| idea9_tail_spike_S100 | 0.34 | 0.42 | 0.11 | -142,728 | 0.7 | 2.2% |
| A_spike_tail_S100_gbm [lead] | 0.10 | 0.12 | 0.04 | -155,726 | 0.64 | 0.8% |
| A_spike_deep_rolling_cut | -0.14 | -0.16 | -0.05 | -169,905 | 0.59 | -1.0% |
| idea8_storm_zone_pairs | -0.09 | -0.10 | -0.04 | -213,497 | 0.47 | -0.9% |
| A_spike_deep | -0.62 | -0.73 | -0.16 | -88,248 | 1.13 | -3.2% |
| D_deep | -0.30 | -0.34 | -0.16 | -102,860 | 0.97 | -3.1% |
| zone_subset_supply | -0.30 | -0.33 | -0.11 | -164,178 | 0.61 | -2.2% |
| idea6_deep_two_gates | -0.81 | -0.96 | -0.19 | -101,830 | 0.98 | -3.7% |
| baseline_usual_side | -0.81 | -1.19 | -0.18 | -124,750 | 0.8 | -3.7% |
| A_regression_v1_deep | -0.33 | -0.35 | -0.15 | -154,103 | 0.65 | -3.0% |
| A_hourly_mean | -0.52 | -0.55 | -0.19 | -138,071 | 0.72 | -3.7% |
| side_past_spike_rate_filter | -0.33 | -0.35 | -0.12 | -257,699 | 0.39 | -2.4% |
| always_supply | -0.40 | -0.43 | -0.13 | -344,445 | 0.29 | -2.6% |
| idea4_deep_storm_three_way | -0.60 | -0.64 | -0.19 | -352,154 | 0.28 | -3.8% |
| idea3_deep_storm_filter | -0.60 | -0.64 | -0.19 | -352,154 | 0.28 | -3.8% |

## Against the lead modeller's list

The lead's strategies are scored here from their own hourly positions (model/evaluate.py ledgers, exported read-only by side/lead_export.py), so every row has the same columns. Every year that differs from the lead's own figure by more than 1 percent is listed with the reason.

| Strategy | Year | Lab | Lead | Difference | Why |
|---|---|---|---|---|---|
| C_gbm | 2023 | 4,403 | 3,911 | +12.6% | Same gross profit. Where two pairs put opposite legs in the same zone-hour, the lab nets them into one position before charging costs, so it pays fewer fees than the lead's leg-by-leg ledger. |

**Storm-day filter: lab 107,141 against the lead's 144,018.** Two things differ, separated in side/recon.py (recon.json). First, the score's inputs: the lab reads the GFS forecast as storm_value.py did (every value of either run that is public at 05:00); the lead reads one value per point and hour by the pipeline's three-day rule and takes the load forecast hour by hour. Both are legal at 05:00 (storm audit, section 1). The two scores are equal on 861 of 1,400 days, and the flag at 0.80 differs on 16 days. That accounts for 2021 (41,620 against 39,476). Second, the 2022 cut. On the lead's own score, 0.80 wins only because the choice counts 31 December 2021: 0.80 beat 0.90 on 2021 by 0.08 USD a day, about 28 USD over the year. The 2022 setting is first used at 05:00 on 31 December 2021, before that day's prices are public, so that day has to be left out. Without it the cut is 0.90 and 2022 earns 45,535 instead of 89,357. The right figures are therefore 100,196 on the lead's score (39,476, 45,535 and 15,185) and 107,141 on storm_value's inputs. The lead's 144,018 rests on one day of lookahead in the cut choice and should read 100,196.

**Storm-day flip: lab 320,492 against the lead's 306,733.** Here only the score's inputs differ: both choose 0.80, 0.90 and 0.80, with or without 31 December. Built with the three-day GFS rule, the lab's score matches the lead in 2022 and 2023 to the dollar (282,065 and -49,753); 2021 still differs (76,325 against 74,421) through the remaining input details. Neither has lookahead. The lead's version follows the pipeline's weather rule, so 306,733 is the figure for the list.

**Idea 9, 2022: lab 377 against the lead's -28,960.** The predictions are the same file; p* is not. The lab picks p* on the 2021 walk-forward predictions it actually traded (0.02 earned 14.59 USD a day, 0.05 lost 5.99); the lead picks it on a separate model trained once before 2021 and scored on 2021 (0.05 lost 12.11 a day, 0.02 lost 13.62). At p* = 0.05 the lab reproduces -28,960 exactly. Neither uses 2022 data. The lab's choice is closer to what would have been traded, but the main fact is that a near coin-flip between two cuts moves 2022 by 29,337, and idea 9 fails the bar in both versions (negative at the 0.50 stress).

**Choice window.** Every lab choice for year Y now uses the year before only up to 30 December, because the setting is first used at 05:00 on 31 December. No lab choice changed because of it.

**B starts on 1 May 2021** (its weather archive starts on 25 March). The lab counts 1 January to 30 April 2021 as days without positions, so B's net is the lead's, while its 2021 and three-year Sharpe and return per year are lower than the lead's (which count only the days B could trade).

## Safeguards

**(a) Injected lookahead.** Inputs: values published after 05:00 on D are corrupted (load forecast times 3, temperatures plus 40 C, real-time prices plus 1,000, a fake newer GFS run at 60 C, labels after the training cut scrambled). With the 05:00 filter the decision for D+1 must not change; with the filter opened to midnight it must. Rule: corrupting every later day's score never changes today's positions. Choice: scrambling the outcomes of the scored year and after never changes the setting chosen for it; choosing on the scored year itself does change settings (shown as a count of years).

| Strategy | Input test | Decisions changed, filter at 05:00 | Decisions changed, filter opened | Rule test | Choice test | Years whose setting moves if chosen in-sample | Pass |
|---|---|---|---|---|---|---|---|
| idea14_learned_allocator_candidates | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | no setting | pass |
| B_gbm | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea10_weather_surprise | raw tables, 20 days | 0 | 15 | 0 changed | 0 changed | 3 | pass |
| idea2_storm_flip | raw tables, 20 days | 0 | 7 | 0 changed | 0 changed | 3 | pass |
| B_deep | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| storm_day_flip [lead] | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea5_deep_zone_day | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | 3 | pass |
| idea11_learned_allocator | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | no setting | pass |
| C_gbm | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea12_policy_network | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | no setting | pass |
| C_deep | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| D_gbm | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea7_gbm_storm_filter | label window, 3 months | max change 0.000 | max change 0.884 | 0 changed | 0 changed | 2 | pass |
| idea1_storm_filter | raw tables, 20 days | 0 | 7 | 0 changed | 0 changed | 3 | pass |
| storm_day_filter [lead] | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| A_regression_v1 | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| risk_sized_supply_gbm | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| A_spike_gbm | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea9_tail_spike_S100 | inputs not tested here: the lead modeller's walk-forward (model/walkforward.py, model/evaluate.py tail_pred) owns the feature and label timing; its prediction file carries no refit dates to check | n/a | n/a | 0 changed | 0 changed | 2 | pass (rule, choice); inputs not tested here |
| A_spike_tail_S100_gbm [lead] | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| A_spike_deep_rolling_cut | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea8_storm_zone_pairs | raw tables, 20 days | 0 | 10 | 0 changed | 0 changed | 3 | pass |
| A_spike_deep | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| D_deep | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| zone_subset_supply | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| idea6_deep_two_gates | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | 3 | pass |
| baseline_usual_side | raw tables, 8 days (zone-hours) | 0 | 827 | 0 changed | 0 changed | no setting | pass |
| A_regression_v1_deep | inputs and choices are the lead's (timing test and the real-data noise test in model/); the lab tests only the rule layer | n/a | n/a | 0 changed | chosen by the lead (not re-run here) changed | chosen by the lead | pass (rule, choice); inputs not tested here |
| A_hourly_mean | raw tables, 8 days (zone-hours) | 0 | 771 | 0 changed | 0 changed | no setting | pass |
| side_past_spike_rate_filter | n/a | n/a | n/a | 0 changed | 0 changed | 2 | pass |
| always_supply | no inputs: the position never changes | n/a | n/a | n/a changed | n/a changed | n/a | pass |
| idea4_deep_storm_three_way | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | 3 | pass |
| idea3_deep_storm_filter | refit dates checked: yes | n/a | n/a | 0 changed | 0 changed | 3 | pass |

**(b) Placebos** are in the ranking table: the same rule and walk-forward fed its score one day late, and fed its score with dates permuted within each year (20 permutations; the share of them at or above the real rule is in brackets). A rule whose placebos earn about what it earns is not reading the day.

**(c) Too good.** A year is flagged when its Sharpe is above 3 or one day earns more than half its profit.

- B_gbm, 2022: Sharpe 3.35 above 3. Year without that day: 413,367. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. Written check (side/b_audit.py, b_audit.json). Inputs: all 11,605 sampled hourly weather and load-forecast values in B's training matrix (44 delivery days, Elliott included) equal a rebuild from the raw tables at 05:00; with the filter opened to midnight only 398 of the load-forecast values would still match. Placebos: one day late 92,197, dates shuffled -16,950 on average (none of 20 at or above the real rule), random days in the same months never beat it. So the 2022 result is read from data public at 05:00. It is still one exceptional year: December 2022 earned 300,208 and Elliott's two days 180,520 (on 24 December B held load in 261 of 264 zone-hours with a predicted gap of +34 USD; the realised gap was +561). Without those two days 2022 earns 374,601. B's forecast correlates 0.12 with the hourly gap in 2022, 0.04 in 2021 and -0.03 in 2023, when its load legs lost 51,023 in 10 of 11 zones.
- idea10_weather_surprise, 2021: best day 2021-02-19 earned 10,813 of the year's 3,034 (356 percent). Year without that day: -7,779. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- idea2_storm_flip, 2022: best day 2022-12-24 earned 148,035 of the year's 290,469 (51 percent). Year without that day: 142,434. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. Written check: Winter Storm Elliott. On 24 December 2022 real time settled on average 561 USD/MWh above day ahead, and the rule held virtual load in every zone-hour because the storm score was just above the 0.90 cut (0.9014 in the lab's build, through the cold component alone). The storm audit (side/storm_audit.md) reproduced this to the dollar and found no lookahead; outside Elliott's two days the flip's load legs earned 8,515 USD over three years. The flag stands: the 2022 profit is one storm.
- storm_day_flip [lead], 2022: best day 2022-12-24 earned 148,035 of the year's 282,065 (52 percent). Year without that day: 134,030. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. Written check: Winter Storm Elliott. On 24 December 2022 real time settled on average 561 USD/MWh above day ahead, and the rule held virtual load in every zone-hour because the storm score was just above the 0.90 cut (0.9014 in the lab's build, through the cold component alone). The storm audit (side/storm_audit.md) reproduced this to the dollar and found no lookahead; outside Elliott's two days the flip's load legs earned 8,515 USD over three years. The flag stands: the 2022 profit is one storm.
- idea5_deep_zone_day, 2021: best day 2021-02-19 earned 10,813 of the year's 20,081 (54 percent). Year without that day: 9,268. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- idea5_deep_zone_day, 2022: best day 2022-12-24 earned 148,035 of the year's 149,388 (99 percent). Year without that day: 1,353. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. 
- idea11_learned_allocator, 2022: best day 2022-12-24 earned 128,078 of the year's 209,765 (61 percent). Year without that day: 81,687. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. 
- C_gbm, 2023: best day 2023-02-28 earned 6,347 of the year's 4,403 (144 percent). Year without that day: -1,944. That day the mean real-time minus day-ahead gap was 7.2 USD/MWh and always supply made -1,939. Written check: the year earned only 4,403 USD, so an ordinary good pair day exceeds half of it. The flag says 2023 was thin, not that the day is suspect.
- idea12_policy_network, 2022: best day 2022-12-24 earned 130,325 of the year's 206,885 (63 percent). Year without that day: 76,560. That day the mean real-time minus day-ahead gap was 560.8 USD/MWh and always supply made -148,090. 
- idea7_gbm_storm_filter, 2021: best day 2021-02-19 earned 10,813 of the year's 7,643 (141 percent). Year without that day: -3,170. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- idea9_tail_spike_S100, 2022: best day 2022-02-28 earned 8,194 of the year's 377 (2173 percent). Year without that day: -7,817. That day the mean real-time minus day-ahead gap was -39.6 USD/MWh and always supply made 10,418. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- A_spike_deep_rolling_cut, 2021: best day 2021-02-19 earned 10,634 of the year's 12,369 (86 percent). Year without that day: 1,735. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- zone_subset_supply, 2021: best day 2021-02-10 earned 6,618 of the year's 9,665 (68 percent). Year without that day: 3,047. That day the mean real-time minus day-ahead gap was -35.9 USD/MWh and always supply made 9,460. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- A_regression_v1_deep, 2021: best day 2021-02-10 earned 5,995 of the year's 9,335 (64 percent). Year without that day: 3,340. That day the mean real-time minus day-ahead gap was -35.9 USD/MWh and always supply made 9,460. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- A_hourly_mean, 2021: best day 2021-02-19 earned 5,940 of the year's 7,484 (79 percent). Year without that day: 1,544. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- A_hourly_mean, 2023: best day 2023-07-28 earned 6,472 of the year's 2,638 (245 percent). Year without that day: -3,834. That day the mean real-time minus day-ahead gap was -54.7 USD/MWh and always supply made 14,410. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- side_past_spike_rate_filter, 2021: best day 2021-02-19 earned 9,318 of the year's 7,360 (127 percent). Year without that day: -1,958. That day the mean real-time minus day-ahead gap was -41.1 USD/MWh and always supply made 10,813. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.
- idea4_deep_storm_three_way, 2023: best day 2023-01-11 earned 16,074 of the year's 10,172 (158 percent). Year without that day: -5,902. That day the mean real-time minus day-ahead gap was 61.0 USD/MWh and always supply made -16,146. 
- idea3_deep_storm_filter, 2023: best day 2023-02-01 earned 6,529 of the year's 9,337 (70 percent). Year without that day: 2,808. That day the mean real-time minus day-ahead gap was -24.9 USD/MWh and always supply made 6,529. Written check: a small year, not an outsized day. On that day the strategy earned no more than always supply did, from ordinary supply positions; nothing about the day suggests lookahead.

## How many strategies were tried

Registered so far: ideas 1 to 12, the three basic strategies and ideas A to D, 19 strategies. 19 of them are scored in this table. The table also has 14 further rows: the deep-model versions of A to D (the registered secondary comparison), A's first rule (A v1) in both models, the lead's builds of ideas 1, 2 and 9 beside the lab's, and four side rules (risk-sized supply, the zone subset, the deep rolling cut, the past spike rate filter). Choices inside the rows searched 156 grid settings in total, counting each grid value of each row once. Before this lab, on the same build years: storm_value.py tried 4 rules beside always supply (storm filter, storm flip, deep risk sizing, a six-zone subset), spike_value.py 2 (deep and history spike filters, 6 cuts each), the storm audit 22 diagnostic variants of ideas 1 and 2, and model/evaluate.py the registered ideas A to D with both models. Ideas 1 and 2 were found in that exploration (disclosed in OBJECTIVES.md).

## Notes

- baseline_crosscheck_vs_panel_base: `{"rows_compared_2021_2023": 289080, "max_abs_diff": 4.884981308350689e-14, "sign_agreement": 0.9999930814999308}`
- storm_deadline: `{"rule": "05:00 wall clock New York on D", "days_where_storm_value_deadline_differs": 8, "examples": ["2020-03-08", "2020-11-01", "2021-03-14", "2021-11-07", "2022-03-13", "2022-11-06", "2023-03-12", "2023-11-05"]}`
- weather_surprise: `{"first_day_with_rank": "2021-05-25", "days_with_rank_2021_2023": 949, "runs_used": "newest and previous public run per point-hour; in the archive the 48-hour and 72-hour leads"}`
- idea9_source: `"cache/wf/spike_S100_2021..2023.parquet, coverage {'2021': 1.0, '2022': 1.0, '2023': 1.0}"`
- deep_spike_file: `{"source": "results/deep_wf_2021_2023_spike.parquet", "timing_metadata": {"rows_in_own_month_refit": true, "train_last_at_least_2_days_before_month": true, "refits": 36, "config": "c2", "threshold": 25.0}}`
- idea7_gbm: `{"features": 1626, "params": {"objective": "binary", "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 20, "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 20261006, "n_jobs": 1, "deterministic": true, "force_row_wise": true}, "first_refit": "2020-04-01"}`
- deep_day_file: `{"rows": 13140, "first": "2021-01-01", "last": "2023-12-31", "rows_in_own_month_refit": true, "train_last_at_least_2_days_before_month": true, "refits": 36, "eval_by_year": {"2021": {"days": 365, "storm_days": 67, "auc_p_storm": 0.5128217970549935, "hand_days": 365, "auc_hand_score_same_days": 0.613593108284083, "auc_p_storm_same_days": 0.5128217970549935, "zone_pred_corr": 0.15147896257883717, "zone_mae": 163.8785509224542, "zone_mae_zero": 165.75918839352428}, "2022": {"days": 365, "storm_days": 74, "auc_p_storm": 0.5953840438376521, "hand_days": 365, "auc_hand_score_same_days": 0.6903269248`
- policy_file: `{"rows": 24090, "columns": ["delivery_date", "bid_date", "zone", "idea", "lam", "position", "refit_month", "w_supply", "w_out", "w_load", "w_pair", "pair_up1", "pair_up2"], "position_column": "position", "rows_in_own_month_refit": true, "train_last_at_least_2_days_before_month": true}`
- idea14_file: `{"rows_in_own_month_refit": true, "train_last_at_least_2_days_before_month": true, "kappa_choice": {"2021": 0.1, "2022": 1.0, "2023": 0.01}}`
- lead_export: `{"strategies": 19, "rows": 5183902, "same_strategies_scored_once": {"always_supply": "always_supply", "baseline": "baseline_usual_side", "A_hourly_mean": "A_hourly_mean"}}`
- reproduction_vs_storm_value_json: `{"always_supply": {"2021": {"lab": -5015, "storm_value": -5015, "lab_cut": null, "storm_value_cut": null}, "2022": {"lab": -198269, "storm_value": -198269, "lab_cut": null, "storm_value_cut": null}, "2023": {"lab": 68154, "storm_value": 68154, "lab_cut": null, "storm_value_cut": null}}, "idea1_storm_filter": {"2021": {"lab": 41620, "storm_value": 41620, "lab_cut": 0.8, "storm_value_cut": 0.8}, "2022": {"lab": 49655, "storm_value": 49655, "lab_cut": 0.9, "storm_value_cut": 0.9}, "2023": {"lab": 15866, "storm_value": 15866, "lab_cut": 0.8, "storm_value_cut": 0.8}}, "idea2_storm_flip": {"2021": {`
