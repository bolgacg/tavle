# Robustness checks on the strongest rows, 2021 to 2023

Written 2026-10-07 05:33:44 CEST on gene by `robust/summary.py`. Registered before they ran (OBJECTIVES.md, addendum of 6 Oct, commit b31b516; sizing decision commit 7504c92). Build years only: every table and prediction file was read through the locked loaders, which stop at 1 January 2024. These checks cannot change a registered rule; they can only add a fragile flag. Lab table used: side/lab_results.json written 2026-10-06 23:32:43 CEST; lead choices written 2026-10-06 23:21:35. Scoring is side/lab.py's for every number here: 1 MW per zone-hour, full costs, 0.50 stress, Bo's bar, and the sizing view (the scale whose three-year worst drawdown is 100,000 USD, at most 5 MW).

Status: complete.

## Seed spread: each deep model retrained with 20 random starts

Same walk-forward as registered (monthly refits, configuration c2), seeds 0 to 19 in one process on the GPU. Each seed is scored on its own and never selected; beside it the mean of all 20, and the six disjoint sets of three seeds (the registered ensemble size).

| Strategy | Registered (seeds 0 to 2), three-year net | 20 seeds: three-year net, min / median / max | Sharpe, min / median / max | Seeds passing at the sizing view | Seeds passing at 1 MW | Seeds with a negative total | Mean of 20 seeds: net, Sharpe, passes at sizing view | Six sets of three seeds: net, lowest to highest |
|---|---|---|---|---|---|---|---|---|
| C_deep | 165,407 | 103,607 / 149,213 / 220,562 | 1.22 / 1.79 / 2.47 | 20 of 20 | 10 of 20 | 0 of 20 | 169,263, 2.10, yes | 148,898 to 188,420 |
| B_deep | 310,453 | 121,971 / 248,632 / 312,241 | 0.92 / 1.26 / 1.55 | 19 of 20 | 17 of 20 | 0 of 20 | 283,064, 1.24, yes | 209,407 to 310,453 |

C_deep: seeds 0 to 2 retrained here against the registered predictions: correlation 1.000, mean absolute difference 0.00 USD per MWh; scored, 165,407 against 165,407 (GPU training is not bit-for-bit repeatable). The registered file scored here reproduces the lab's row (165,407).

B_deep: seeds 0 to 2 retrained here against the registered predictions: correlation 1.000, mean absolute difference 0.00 USD per MWh; scored, 310,453 against 310,453 (GPU training is not bit-for-bit repeatable). The registered file scored here reproduces the lab's row (310,453).

## Neighbouring settings

Gradient boosting: each year's chosen configuration moved one step on one of the grid's three settings, in every year at once (C keeps each year's registered pairs). Deep: configuration c2 with dropout or weight decay one step either side, three seeds as registered. Same walk-forward and scoring.

| Strategy | Setting | Settings 2021 / 2022 / 2023 | Three-year net | Sharpe | Passes at 1 MW | Passes at the sizing view | Same sign as the chosen setting |
|---|---|---|---|---|---|---|---|
| B_gbm | chosen (reproduces the lab) | leaves15_minleaf200_clipnone / leaves15_minleaf200_clipnone / leaves15_minleaf2000_clip250.0 | 583,946 | 1.96 | yes | yes |  |
| B_gbm | num_leaves one step | leaves63_minleaf200_clipnone / leaves63_minleaf200_clipnone / leaves63_minleaf2000_clip250.0 | 571,384 | 2.14 | yes | yes | yes |
| B_gbm | min_data_in_leaf one step | leaves15_minleaf2000_clipnone / leaves15_minleaf2000_clipnone / leaves15_minleaf200_clip250.0 | 502,575 | 1.78 | yes | yes | yes |
| B_gbm | target_clip one step | leaves15_minleaf200_clip250.0 / leaves15_minleaf200_clip250.0 / leaves15_minleaf2000_clipnone | 563,025 | 1.86 | yes | yes | yes |
| C_gbm | chosen (reproduces the lab) | leaves15_minleaf200_clipnone / leaves63_minleaf2000_clipnone / leaves15_minleaf200_clip250.0 | 169,889 | 1.67 | yes | yes |  |
| C_gbm | num_leaves one step | leaves63_minleaf200_clipnone / leaves15_minleaf2000_clipnone / leaves63_minleaf200_clip250.0 | 139,430 | 1.47 | no | yes | yes |
| C_gbm | min_data_in_leaf one step | leaves15_minleaf2000_clipnone / leaves63_minleaf200_clipnone / leaves15_minleaf2000_clip250.0 | 61,703 | 0.62 | no | no | yes |
| C_gbm | target_clip one step | leaves15_minleaf200_clip250.0 / leaves63_minleaf2000_clip250.0 / leaves15_minleaf200_clipnone | 187,341 | 1.64 | yes | yes | yes |
| B_deep | registered c2 (dropout 0.3, weight decay 0.01) | c2 / c2 / c2 | 310,453 | 1.18 | yes | yes | |
| B_deep | dropout 0.2 (c2: 0.3) | same in every year | 280,289 | 1.33 | yes | yes | yes |
| B_deep | dropout 0.4 (c2: 0.3) | same in every year | 302,123 | 1.09 | yes | yes | yes |
| B_deep | weight decay 0.001 (c2: 0.01) | same in every year | 311,302 | 1.18 | yes | yes | yes |
| B_deep | weight decay 0.1 (c2: 0.01) | same in every year | 299,575 | 1.14 | yes | yes | yes |
| C_deep | registered c2 (dropout 0.3, weight decay 0.01) | c2 / c2 / c2 | 165,407 | 2.15 | yes | yes | |
| C_deep | dropout 0.2 (c2: 0.3) | same in every year | 172,707 | 2.22 | yes | yes | yes |
| C_deep | dropout 0.4 (c2: 0.3) | same in every year | 161,135 | 2.31 | yes | yes | yes |
| C_deep | weight decay 0.001 (c2: 0.01) | same in every year | 164,693 | 2.16 | yes | yes | yes |
| C_deep | weight decay 0.1 (c2: 0.01) | same in every year | 159,434 | 2.13 | yes | yes | yes |

## Costs at 1.00 USD per MWh, seasons and half-years

Net USD at 1 MW per zone-hour (idea 13 at its built size), full costs except the second column. Seasons pool the three years; winter is December, January and February of 2021 to 2023 (January and February 2024 are held out).

| Strategy | Three-year net | At 1.00 USD per MWh | Winter | Spring | Summer | Autumn | 2021 H1 | 2021 H2 | 2022 H1 | 2022 H2 | 2023 H1 | 2023 H2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_deep | 165,407 | -41,658 | 82,416 | 58,035 | 3,277 | 21,679 | 38,365 | 19,473 | 60,236 | 12,970 | 26,722 | 7,641 |
| B_gbm | 583,946 | 363,414 | 363,998 | 75,704 | 91,673 | 52,571 | 24,508 | 26,678 | 152,270 | 402,851 | -17,938 | -4,423 |
| C_gbm | 169,889 | -31,291 | 84,184 | 45,003 | 5,192 | 35,510 | 14,332 | 27,125 | 48,839 | 75,190 | 10,551 | -6,149 |
| B_deep | 310,453 | 92,335 | 222,670 | 37,812 | 41,705 | 8,266 | 22,303 | 13,161 | 111,821 | 198,898 | -6,017 | -29,714 |
| idea13_portfolio | 664,146 | 374,884 | 377,421 | 125,249 | 75,937 | 85,539 | 64,396 | 58,830 | 177,198 | 330,237 | 32,932 | 553 |

## Idea 13, portfolio of the survivors

| Components (MW per position) | Three-year net | Average a year | Sharpe | Max drawdown | Passes the bar as built | Multiplier for a 100,000 drawdown | Return on 500,000 a year at that size | Passes at the sizing view |
|---|---|---|---|---|---|---|---|---|
| A_regression_v1 0.34 MW; B_deep 0.29 MW; B_gbm 0.46 MW; C_deep 0.78 MW; C_gbm 0.68 MW | 664,146 | 221,382 | 3.12 | -31,564 | yes | 0.594 | 26.3% | yes |

Each component is sized so that its own 2021 to 2023 worst drawdown is 100,000 USD (at most 5 MW), then weighted 1/5; positions are netted per zone-hour before costs. Components reproduce the lab's rows: A_regression_v1 yes, B_deep yes, B_gbm yes, C_deep yes, C_gbm yes.

Pass at the sizing view but left out as fragile: idea2_storm_flip (lab: three-year total turns negative without one year's best 10 days, fragile per storm audit (side/storm_audit.md)); idea10_weather_surprise (fragile per idea 10 audit (side/idea10_audit.md)); D_gbm (lab: three-year total turns negative without one year's best 10 days); storm_day_flip [lead] (lab: three-year total turns negative without one year's best 10 days, fragile per storm audit (side/storm_audit.md)).

## What the checks show for the fragile flag

- C_deep: keeps its sign at every neighbouring setting; 0 of 20 seeds lose money over three years, 20 of 20 pass at the sizing view; turns negative at 1.00 USD per MWh (-41,658).
- B_gbm: keeps its sign at every neighbouring setting; stays positive at 1.00 USD per MWh (363,414).
- C_gbm: keeps its sign at every neighbouring setting; turns negative at 1.00 USD per MWh (-31,291).
- B_deep: keeps its sign at every neighbouring setting; 0 of 20 seeds lose money over three years, 19 of 20 pass at the sizing view; stays positive at 1.00 USD per MWh (92,335).

