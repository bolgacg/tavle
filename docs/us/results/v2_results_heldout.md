# New York study v2: heldout run, scored on 2024-01-01 to 2026-09-30

Written 2026-10-11 01:56. Train on the previous 3 years, test on the next quarter, refit quarterly; settings chosen on the trailing 4 out-of-sample quarters. Net USD at full cost per year (1 MW per position). Data read up to 2026-09-30 (the held-out window).

Bar: average net >= 50,000 USD a year, positive in at least 2 of 3 years, Sharpe > 0.42 over 2024 to 2026, positive total at 0.50 USD/MWh stress cost.

| row | 2024 | 2025 | 2026 | total | w/o best 5 days | Sharpe | USD/MWh | scale for 100k DD | PASS | FRAGILE | placebo 1 day late | placebo permuted mean | tries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_static_pairs (comparison) | 17,294 | 102,043 | 27,938 | 147,275 | 80,172 | 2.29 | 0.715 | 5.0 | no | yes | 147,264 | 174,749 | 1 |
| V10_C_deep_pretrain | 11,743 | 72,670 | 28,311 | 112,724 | 59,571 | 1.97 | 0.806 | 5.0 | no | no | 82,748 | 90,850 | 1 |
| V11_C_deep_ens4 | 10,175 | 74,580 | 23,958 | 108,714 | 70,699 | 2.26 | 0.668 | 5.0 | no | no | 105,824 | 125,372 | 1 |
| V11_C_gbm_ens4 | 17,599 | 43,396 | 54,163 | 115,158 | 62,035 | 1.89 | 0.776 | 5.0 | no | no | 38,772 | -40,166 | 1 |
| V12_alloc | 59,070 | -158,157 | 318,530 | 219,443 | -78,755 | 0.35 | 0.822 | 0.22 | no | yes | 199,692 | 136,679 | 3 |
| V13_A_deep_all | 52,738 | 28,587 | 254,005 | 335,331 | 22,716 | 1.05 | 1.305 | 1.27 | yes | yes | 279,099 | 353,586 | 18 |
| V13_C_deep_all | 13,124 | 91,541 | 12,440 | 117,104 | 76,040 | 2.48 | 0.705 | 5.0 | no | no | 114,690 | 85,177 | 3 |
| V14a_window_ens | 35,242 | 12,279 | 30,754 | 78,275 | 33,025 | 1.48 | 0.558 | 5.0 | no | no | 4,650 | 25,654 | 3 |
| V14b_quantile_size | 3,327 | 4,329 | 2,451 | 10,107 | 5,298 | 1.34 | 0.629 | 5.0 | no | no | 4,435 | 719 | 9 |
| V14c_conformal_skip | 17,417 | 35,468 | 29,524 | 82,409 | 26,903 | 1.59 | 1.905 | 5.0 | no | no | 43,400 | 42,063 | 4 |
| V14d_seed_ens | 32,327 | 34,008 | 37,479 | 103,814 | 60,303 | 2.25 | 0.728 | 5.0 | no | no | 31,971 | 21,521 | 1 |
| V14e_mlp_multi | -11,367 | 58,711 | 7,230 | 54,574 | 17,347 | 1.28 | 0.314 | 5.0 | no | yes | 48,526 | 113,885 | 2 |
| V14f_global_zone | 17,138 | 39,419 | 32,969 | 89,526 | 52,759 | 2.1 | 0.529 | 5.0 | no | no | 24,447 | -27,711 | 2 |
| V15a_V5_limit | 41,672 | 61,707 | 311,797 | 415,175 | 99,124 | 1.23 | 2.021 | 0.73 | yes | no | 187,117 | -39,743 | 1 |
| V15a_V8_limit | 16,782 | 23,076 | 294,595 | 334,453 | 6,690 | 0.97 | 1.587 | 0.69 | yes | yes | 164,253 | 7,569 | 2 |
| V15b_V13_both | 54,277 | 107,405 | 228,533 | 390,214 | 129,183 | 1.49 | 1.611 | 1.72 | yes | no | 274,402 | 108,183 | 3 |
| V15b_V13_limit | 18,186 | 48,126 | 238,640 | 304,952 | 9,780 | 1.06 | 1.578 | 1.48 | yes | yes | 198,296 | 84,714 | 3 |
| V15c_wx_border_gbm | 85,116 | 30,145 | 318,450 | 433,710 | 101,483 | 1.24 | 1.781 | 0.91 | yes | no | 209,784 | -50,984 | 1 |
| V16_B_gefs_joined | 36,530 | 28,070 | 346,350 | 410,950 | 77,193 | 1.17 | 1.674 | 0.84 | yes | no | 129,846 | -15,337 | 1 |
| V1_A_spike_gbm | 18,104 | 27,400 | -1,113 | 44,391 | -15,157 | 0.34 | 0.196 | 1.72 | no | yes | -32,861 | 269,654 | 6 |
| V1_C_deep | 16,151 | 90,959 | 22,513 | 129,623 | 79,044 | 2.39 | 0.793 | 5.0 | no | no | 107,914 | 117,880 | 1 |
| V1_C_gbm | 20,543 | 28,829 | 40,485 | 89,857 | 51,488 | 1.83 | 0.621 | 5.0 | no | no | -24,050 | -46,616 | 1 |
| V1_always_supply | 58,640 | -14,709 | 303,266 | 347,197 | -22,053 | 0.89 | 1.31 | 0.63 | yes | yes | n/a | n/a | 1 |
| V1_baseline | 39,434 | 22,797 | 41,118 | 103,349 | -8,640 | 0.71 | 0.39 | 1.92 | no | yes | 31,264 | 478,932 | 1 |
| V2_limit_gbm | 26,963 | 19,491 | 344,177 | 390,630 | 28,835 | 1.09 | 1.964 | 0.66 | yes | yes | 238,129 | -4,541 | 1 |
| V2_sides_no_limit (comparison) | 45,179 | 49,190 | 367,351 | 461,720 | 94,633 | 1.26 | 1.823 | 0.77 | yes | no | 320,542 | 20,722 | 1 |
| V3_tail_gbm | 120,061 | 186,854 | 951,812 | 1,258,726 | 267,545 | 1.21 | 2.315 | 0.32 | yes | no | 623,751 | -16,159 | 4 |
| V3_tail_gbm_1MW (comparison) | 35,982 | 72,097 | 323,511 | 431,591 | 94,572 | 1.22 | 1.802 | 0.92 | yes | no | 195,576 | -17,264 | 0 |
| V5_border_inputs | 61,389 | 96,820 | 134,878 | 293,087 | 63,836 | 1.05 | 1.197 | 1.02 | yes | no | 56,955 | -46,610 | 1 |
| V6_LI | 5,479 | -13,304 | 19,564 | 11,739 | -6,977 | 0.38 | 0.507 | 5.0 | no | yes | 27,901 | -10,721 | 1 |
| V6_NYC | 7,448 | 7,516 | 30,240 | 45,204 | 10,764 | 1.24 | 1.967 | 5.0 | no | no | 7,909 | -11,725 | 1 |
| V7_flags | 64,037 | -10,674 | 346,599 | 399,962 | 78,444 | 1.15 | 1.556 | 0.83 | yes | no | 312,409 | -161,062 | 1 |
| V7_recency | 43,657 | -45,020 | 315,817 | 314,453 | -17,503 | 0.91 | 1.221 | 0.7 | yes | yes | 363,046 | -149,851 | 3 |
| V8_error_mining | 35,123 | 14,892 | 292,062 | 342,077 | 29,400 | 1.0 | 1.371 | 0.68 | yes | yes | 128,278 | 26,969 | 2 |
| V9_decompose | 17,076 | 71,301 | 203,628 | 292,005 | 24,027 | 0.98 | 1.184 | 1.18 | yes | yes | 49,557 | -38,615 | 1 |

Not run:

- V4_B_reforecast: not run: forecast source ends in 2019 (superseded by V16)
- V15a_V4_limit: not run: forecast source ends in 2019 (superseded by V16)
- V15d_V4_pairs: not run: forecast source ends in 2019 (superseded by V16)

Labels (addendum of 7 Oct):

- V2_limit_gbm: V2 = gradient-boosting direction model (the limit prices add nothing; see V2_sides_no_limit)
- V3_tail_gbm: V3 tail-risk sizing: read per MWh and at 1 MW (V3_tail_gbm_1MW); its total reflects position size

Fragile reasons:

- C_static_pairs: random days in the same months beat the rule more than 20 percent of the time
- V12_alloc: three-year total turns negative without one year's best 10 days
- V13_A_deep_all: three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time
- V14e_mlp_multi: random days in the same months beat the rule more than 20 percent of the time
- V15a_V8_limit: three-year total turns negative without one year's best 10 days
- V15b_V13_limit: three-year total turns negative without one year's best 10 days
- V1_A_spike_gbm: three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time
- V1_always_supply: three-year total turns negative without one year's best 10 days
- V1_baseline: three-year total turns negative without one year's best 10 days; random days in the same months beat the rule more than 20 percent of the time
- V2_limit_gbm: three-year total turns negative without one year's best 10 days
- V6_LI: three-year total turns negative without one year's best 10 days
- V7_recency: three-year total turns negative without one year's best 10 days
- V8_error_mining: three-year total turns negative without one year's best 10 days
- V9_decompose: three-year total turns negative without one year's best 10 days

Too-good flags (a year with Sharpe above 3, or one day above half the year's profit):

- C_static_pairs 2025: Sharpe 3.74 above 3
- C_static_pairs 2026: best day 2026-07-01 earned 16,300 of the year's 27,938 (58 percent)
- V10_C_deep_pretrain 2025: Sharpe 3.02 above 3
- V10_C_deep_pretrain 2026: best day 2026-07-01 earned 19,134 of the year's 28,311 (68 percent)
- V11_C_deep_ens4 2025: Sharpe 3.41 above 3
- V13_A_deep_all 2025: best day 2025-01-20 earned 18,331 of the year's 28,587 (64 percent)
- V13_C_deep_all 2025: Sharpe 3.75 above 3
- V14c_conformal_skip 2026: best day 2026-07-01 earned 17,253 of the year's 29,524 (58 percent)
- V14e_mlp_multi 2025: Sharpe 3.07 above 3
- V14e_mlp_multi 2026: best day 2026-07-01 earned 7,468 of the year's 7,230 (103 percent)
- V15a_V8_limit 2025: best day 2025-07-01 earned 21,877 of the year's 23,076 (95 percent)
- V15c_wx_border_gbm 2025: best day 2025-07-01 earned 19,558 of the year's 30,145 (65 percent)
- V16_B_gefs_joined 2025: best day 2025-01-20 earned 16,295 of the year's 28,070 (58 percent)
- V1_A_spike_gbm 2024: best day 2024-01-14 earned 10,347 of the year's 18,104 (57 percent)
- V1_A_spike_gbm 2025: best day 2025-01-20 earned 16,128 of the year's 27,400 (59 percent)
- V1_C_deep 2025: Sharpe 3.3 above 3
- V1_baseline 2025: best day 2025-07-01 earned 21,349 of the year's 22,797 (94 percent)
- V1_baseline 2026: best day 2026-01-29 earned 31,950 of the year's 41,118 (78 percent)
- V2_limit_gbm 2025: best day 2025-01-20 earned 19,163 of the year's 19,491 (98 percent)
- V5_border_inputs 2026: best day 2026-01-28 earned 97,577 of the year's 134,878 (72 percent)
- V6_NYC 2025: best day 2025-07-01 earned 5,611 of the year's 7,516 (75 percent)
- V8_error_mining 2025: best day 2025-01-20 earned 18,673 of the year's 14,892 (125 percent)
- V9_decompose 2024: best day 2024-01-17 earned 9,142 of the year's 17,076 (54 percent)
- V9_decompose 2026: best day 2026-01-28 earned 115,411 of the year's 203,628 (57 percent)

Sizing view carried from the build (scale = MW per position at which the build-year worst drawdown is 100,000 USD, capped at 5; not refitted here):

| row | scale (build) | net at that scale | worst drawdown at that scale | return a year on 500,000 |
|---|---|---|---|---|
| C_static_pairs | 1.78 | 262,150 | -30,095 | 19 |
| V10_C_deep_pretrain | 2.02 | 227,702 | -26,520 | 17 |
| V11_C_deep_ens4 | 2.39 | 259,825 | -25,557 | 19 |
| V11_C_gbm_ens4 | 1.75 | 201,527 | -28,053 | 15 |
| V12_alloc | 0.41 | 89,972 | -182,513 | 6.5 |
| V13_A_deep_all | 0.59 | 197,845 | -46,514 | 14 |
| V13_C_deep_all | 1.84 | 215,472 | -15,567 | 16 |
| V14a_window_ens | 1.58 | 123,674 | -24,207 | 9.0 |
| V14b_quantile_size | 5.0 | 50,535 | -13,875 | 3.7 |
| V14c_conformal_skip | 3.59 | 295,847 | -49,885 | 22 |
| V14d_seed_ens | 2.08 | 215,933 | -26,118 | 16 |
| V14e_mlp_multi | 3.33 | 181,731 | -43,378 | 13 |
| V14f_global_zone | 3.19 | 285,588 | -30,657 | 21 |
| V15a_V5_limit | 2.51 | 1,042,089 | -343,765 | 76 |
| V15a_V8_limit | 2.17 | 725,762 | -312,782 | 53 |
| V15b_V13_both | 1.52 | 593,126 | -88,141 | 43 |
| V15b_V13_limit | 1.25 | 381,190 | -84,580 | 28 |
| V15c_wx_border_gbm | 2.01 | 871,757 | -222,092 | 63 |
| V16_B_gefs_joined | 1.54 | 632,863 | -182,786 | 46 |
| V1_A_spike_gbm | 0.68 | 30,186 | -39,454 | 2.2 |
| V1_C_deep | 2.6 | 337,019 | -29,555 | 24 |
| V1_C_gbm | 1.81 | 162,642 | -25,585 | 12 |
| V1_always_supply | 0.27 | 93,743 | -43,010 | 6.8 |
| V1_baseline | 0.8 | 82,679 | -41,767 | 6.0 |
| V2_limit_gbm | 1.77 | 691,416 | -270,020 | 50 |
| V2_sides_no_limit | 1.62 | 747,986 | -211,301 | 54 |
| V3_tail_gbm | 0.53 | 667,125 | -167,171 | 48 |
| V3_tail_gbm_1MW | 1.55 | 668,966 | -167,733 | 49 |
| V5_border_inputs | 1.52 | 445,493 | -149,600 | 32 |
| V6_LI | 5.0 | 58,695 | -99,730 | 4.3 |
| V6_NYC | 5.0 | 226,022 | -45,671 | 16 |
| V7_flags | 1.22 | 487,953 | -147,583 | 36 |
| V7_recency | 1.22 | 383,633 | -174,217 | 28 |
| V8_error_mining | 2.26 | 773,094 | -330,093 | 56 |
| V9_decompose | 1.82 | 531,449 | -154,737 | 39 |

Tries: 35 v2 strategy rows, 84 settings tried in all on 2013..2023 rolling quarters, on top of everything v1 tried on 2021..2023.

Sizing view (not part of the verdict): scale = MW per position at which the worst drawdown is 100,000 USD, capped at 5.

