# New York study v2: rolling-window results, 2013 to 2023 (build years only)

Written 2026-10-08 04:56. Train on the previous 3 years, test on the next quarter, refit quarterly; settings chosen on the trailing 4 out-of-sample quarters. Net USD at full cost per year (1 MW per position). Nothing on or after 2024-01-01 was read.

Bar: average net >= 50,000 USD a year, positive in at least 8 of 11 years, Sharpe > 0.42 over 2013 to 2023, positive total at 0.50 USD/MWh stress cost.

| row | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | total | w/o best 5 days | Sharpe | USD/MWh | scale for 100k DD | PASS | FRAGILE | placebo 1 day late | placebo permuted mean | tries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_static_pairs (comparison) | 112,377 | 60,192 | 78,914 | 21,658 | 9,767 | 16,695 | 20,374 | 14,051 | 119,452 | 30,287 | 24,153 | 507,922 | 433,746 | 1.9 | 0.565 | 1.78 | no | yes | 510,613 | 661,544 | 1 |
| V10_C_deep_pretrain | 130,206 | 41,136 | 88,953 | 60,540 | 56,600 | 30,567 | 19,464 | 36,194 | 2,489 | 44,541 | 28,733 | 539,422 | 468,898 | 1.78 | 0.662 | 2.02 | no | yes | 588,706 | 417,384 | 1 |
| V11_C_deep_ens4 | 150,389 | 75,421 | 93,618 | 70,131 | 67,221 | 46,793 | 18,803 | 28,905 | 18,919 | 39,249 | 32,855 | 642,305 | 569,130 | 2.1 | 0.778 | 2.39 | yes | no | 654,510 | 498,247 | 1 |
| V11_C_gbm_ens4 | 9,329 | 80,595 | 40,702 | 15,514 | 36,765 | 38,242 | 2,475 | 20,184 | 50,281 | 137,032 | -8,435 | 422,684 | 361,009 | 1.45 | 0.534 | 1.75 | no | no | 265,154 | -11,946 | 1 |
| V12_alloc | 1,416,566 | 1,076,470 | 188,438 | 420,219 | 332,039 | 379,369 | 243,893 | 155,003 | 410,793 | 1,090,850 | -36,267 | 5,677,372 | 4,906,962 | 2.4 | 1.689 | 0.41 | yes | no | 3,056,242 | 971,084 | 3 |
| V13_A_deep_all | 74,512 | 208,196 | 21,008 | 48,966 | 36,733 | 9,991 | -9,847 | 2,691 | -23,038 | -73,212 | 11,063 | 307,062 | 198,859 | 0.66 | 0.379 | 0.59 | no | no | 292,185 | 441,736 | 18 |
| V13_C_deep_all | 127,898 | 40,585 | 63,259 | 25,204 | 56,697 | 34,020 | 9,722 | 1,439 | 38,196 | 37,345 | 17,001 | 451,367 | 381,510 | 1.73 | 0.545 | 1.84 | no | no | 457,258 | 388,903 | 3 |
| V14a_window_ens | 66,516 | 74,200 | 81,426 | 39,901 | 79,178 | 53,762 | 28,276 | 46,409 | 40,981 | 115,551 | 13,544 | 639,744 | 561,207 | 1.91 | 0.788 | 1.58 | yes | no | 566,291 | 354,274 | 3 |
| V14b_quantile_size | 33,258 | 50,647 | 12,823 | 15,719 | 15,370 | 17,515 | 1,047 | 7,927 | 18,333 | 27,477 | 3,301 | 203,417 | 181,022 | 1.89 | 0.955 | 5.0 | no | no | 165,912 | 66,914 | 9 |
| V14c_conformal_skip | 9,424 | 52,522 | 23,553 | 9,170 | 18,247 | 15,185 | 936 | 9,605 | 25,320 | 52,556 | -8,334 | 208,183 | 154,295 | 1.19 | 1.398 | 3.59 | no | no | 208,565 | 50,458 | 4 |
| V14d_seed_ens | 50,362 | 129,696 | 98,457 | 56,416 | 44,023 | 42,484 | 5,647 | 19,927 | 19,803 | 96,038 | 20,259 | 583,112 | 518,422 | 1.93 | 0.718 | 2.08 | yes | no | 477,679 | 265,253 | 1 |
| V14e_mlp_multi | 135,595 | 58,601 | 50,032 | 44,566 | 43,802 | 42,608 | 8,664 | 9,927 | 48,608 | 58,617 | 9,703 | 510,722 | 452,191 | 2.14 | 0.618 | 3.33 | no | no | 533,320 | 499,539 | 2 |
| V14f_global_zone | 64,756 | 69,843 | 72,485 | 60,860 | 54,877 | 67,681 | 15,492 | 11,628 | 49,303 | 108,118 | 1,159 | 576,202 | 525,256 | 2.28 | 0.705 | 3.19 | yes | no | 334,606 | 59,848 | 2 |
| V15a_V4_limit | 247,208 | 300,410 | 105,323 | 146,664 | 118,697 | 152,430 | 39,627 | 0 | 0 | 0 | 0 | 1,110,359 | 974,187 | 2.65 | 1.983 | 2.41 | no | no | 654,590 | 196,756 | 1 |
| V15a_V5_limit | 194,863 | 182,351 | 49,231 | 83,432 | 94,657 | 101,979 | 52,071 | 46,050 | 86,882 | 206,374 | 27,734 | 1,125,624 | 970,485 | 2.29 | 1.365 | 2.51 | yes | no | 664,143 | 118,974 | 1 |
| V15a_V8_limit | 180,407 | 259,639 | 73,748 | 60,383 | 99,365 | 110,098 | 67,591 | 43,214 | 112,629 | 192,127 | 13,146 | 1,212,347 | 1,077,924 | 2.57 | 1.452 | 2.17 | yes | no | 695,129 | 194,897 | 2 |
| V15b_V13_both | 137,664 | 257,755 | 108,712 | 100,565 | 81,813 | 43,162 | 19,482 | 52,698 | 19,924 | 47,351 | -34,397 | 834,728 | 679,281 | 1.65 | 0.824 | 1.52 | yes | no | 716,671 | 365,907 | 3 |
| V15b_V13_limit | 99,304 | 207,797 | 110,951 | 89,496 | 85,741 | 39,879 | 19,936 | 48,270 | 26,507 | -30,324 | -44,685 | 652,873 | 529,172 | 1.46 | 0.828 | 1.25 | yes | no | 568,111 | 327,070 | 3 |
| V15c_wx_border_gbm | 241,030 | 288,087 | 66,677 | 128,010 | 105,245 | 136,876 | 30,807 | 31,246 | 48,405 | 385,870 | 2,921 | 1,465,173 | 1,207,174 | 2.18 | 1.435 | 2.01 | yes | no | 813,666 | 136,813 | 1 |
| V15d_V4_pairs | -362 | 41,027 | 49,492 | 26,126 | 56,741 | 34,456 | 4,564 | 0 | 0 | 0 | 0 | 212,044 | 159,969 | 0.83 | 0.41 | 1.54 | no | no | 140,348 | 33,857 | 1 |
| V16_B_gefs_joined | 264,606 | 299,137 | 60,268 | 155,285 | 121,438 | 162,361 | 52,588 | 31,175 | 74,820 | 441,081 | 27,364 | 1,690,124 | 1,416,110 | 2.46 | 1.769 | 1.54 | yes | no | 762,331 | 190,605 | 1 |
| V1_A_spike_gbm | 92,255 | 121,429 | 6,714 | 44,474 | 51,909 | 26,403 | -8,431 | -2,508 | -31,116 | -65,276 | 25,390 | 261,243 | 225,188 | 0.74 | 0.317 | 0.68 | no | yes | 135,672 | 402,714 | 6 |
| V1_C_deep | 169,476 | 70,388 | 89,641 | 69,078 | 68,286 | 45,258 | 29,478 | 25,019 | -3,299 | 71,607 | 33,677 | 668,609 | 596,499 | 2.25 | 0.8 | 2.6 | yes | no | 698,712 | 541,511 | 1 |
| V1_C_gbm | 6,157 | 72,714 | 60,174 | 12,134 | 32,930 | 28,623 | 848 | 17,146 | 42,985 | 98,202 | -3,750 | 368,164 | 313,749 | 1.29 | 0.469 | 1.81 | no | no | 238,933 | 12,255 | 1 |
| V1_always_supply | 46,127 | 255,468 | 37,082 | 27,942 | 42,984 | -35,266 | 32,766 | -24,609 | -5,015 | -198,269 | 68,154 | 247,363 | 65,940 | 0.27 | 0.233 | 0.27 | no | yes | n/a | n/a | 1 |
| V1_baseline | 62,472 | 128,148 | 53,046 | -3,181 | 30,463 | -19,552 | -4,594 | 26,308 | 12,665 | -8,643 | -72,836 | 204,294 | 119,196 | 0.58 | 0.193 | 0.8 | no | yes | 198,134 | 1,015,067 | 1 |
| V2_limit_gbm | 128,839 | 272,485 | 63,681 | 102,728 | 86,119 | 104,341 | 77,815 | 35,088 | 106,860 | 198,527 | 2,064 | 1,178,549 | 1,032,758 | 2.41 | 1.447 | 1.77 | yes | no | 564,630 | 167,763 | 1 |
| V2_sides_no_limit (comparison) | 137,602 | 243,839 | 43,830 | 102,606 | 68,346 | 88,641 | 89,174 | 35,983 | 120,567 | 282,214 | -19,496 | 1,193,308 | 1,019,236 | 2.15 | 1.183 | 1.62 | yes | no | 575,082 | 119,350 | 1 |
| V3_tail_gbm | 529,051 | 721,292 | 78,731 | 216,365 | 242,303 | 387,080 | 222,663 | 125,541 | 305,761 | 1,008,190 | 70,495 | 3,907,473 | 3,166,317 | 1.9 | 1.57 | 0.53 | yes | no | 1,432,875 | 501,730 | 4 |
| V3_tail_gbm_1MW (comparison) | 174,934 | 256,374 | 26,910 | 98,331 | 78,240 | 129,288 | 79,997 | 43,236 | 107,028 | 336,238 | 26,326 | 1,356,902 | 1,107,875 | 1.94 | 1.34 | 1.55 | yes | no | 490,599 | 179,046 | 0 |
| V4_B_reforecast | 264,606 | 299,137 | 60,268 | 155,285 | 121,438 | 162,361 | 52,588 | 0 | 0 | 0 | 0 | 1,115,684 | 997,769 | 2.83 | 1.706 | 1.54 | no | no | 655,024 | 212,920 | 1 |
| V5_border_inputs | 208,853 | 259,860 | 7,532 | 78,580 | 82,967 | 106,407 | 50,360 | 39,941 | 50,959 | 328,721 | 45,823 | 1,260,003 | 1,023,111 | 1.96 | 1.238 | 1.52 | yes | no | 748,645 | 128,110 | 1 |
| V6_LI | -9,133 | 15,766 | 3,073 | 14,926 | 9,131 | 14,444 | 11,374 | 11,900 | 3,823 | 41,812 | 11,654 | 128,768 | 97,050 | 1.26 | 1.357 | 5.0 | no | no | 85,471 | 7,381 | 1 |
| V6_NYC | 16,736 | 36,144 | 12,703 | 7,075 | 2,071 | 6,968 | 7,232 | 4,268 | 8,099 | 991 | -263 | 102,025 | 87,483 | 1.59 | 1.084 | 5.0 | no | no | 65,776 | 4,441 | 1 |
| V7_flags | 186,401 | 277,110 | 16,658 | 65,021 | 72,211 | 137,659 | 71,885 | 51,465 | 48,753 | 414,777 | -34,780 | 1,307,159 | 1,060,221 | 1.88 | 1.28 | 1.22 | yes | no | 556,633 | 153,701 | 1 |
| V7_recency | 132,355 | 277,110 | 26,904 | 92,228 | 79,166 | 113,650 | 71,885 | 57,290 | 38,845 | 431,433 | -5,930 | 1,314,936 | 1,067,657 | 1.88 | 1.286 | 1.22 | yes | no | 533,189 | 147,709 | 3 |
| V8_error_mining | 217,651 | 269,221 | 57,006 | 48,046 | 99,636 | 170,181 | 64,453 | 33,804 | 83,377 | 386,059 | 21,662 | 1,451,096 | 1,197,233 | 2.08 | 1.417 | 2.26 | yes | no | 719,792 | 221,931 | 2 |
| V9_decompose | 206,879 | 255,222 | 42,483 | 102,004 | 82,935 | 116,592 | 38,819 | 42,820 | 101,031 | 224,138 | -4,605 | 1,208,319 | 966,230 | 1.77 | 1.178 | 1.82 | yes | no | 474,654 | 127,522 | 1 |

Labels (addendum of 7 Oct):

- V2_limit_gbm: V2 = gradient-boosting direction model (the limit prices add nothing; see V2_sides_no_limit)
- V3_tail_gbm: V3 tail-risk sizing: read per MWh and at 1 MW (V3_tail_gbm_1MW); its total reflects position size

Fragile reasons:

- C_static_pairs: random days in the same months beat the rule more than 20 percent of the time
- V10_C_deep_pretrain: random days in the same months beat the rule more than 20 percent of the time
- V1_A_spike_gbm: random days in the same months beat the rule more than 20 percent of the time
- V1_always_supply: three-year total turns negative without one year's best 10 days
- V1_baseline: random days in the same months beat the rule more than 20 percent of the time

Too-good flags (a year with Sharpe above 3, or one day above half the year's profit):

- C_static_pairs 2013: Sharpe 3.64 above 3
- C_static_pairs 2016: best day 2016-03-28 earned 18,257 of the year's 21,658 (84 percent)
- C_static_pairs 2021: Sharpe 3.59 above 3
- V10_C_deep_pretrain 2013: Sharpe 3.49 above 3
- V10_C_deep_pretrain 2015: Sharpe 3.05 above 3
- V10_C_deep_pretrain 2017: Sharpe 3.97 above 3
- V10_C_deep_pretrain 2020: Sharpe 3.01 above 3
- V10_C_deep_pretrain 2021: best day 2021-08-20 earned 6,167 of the year's 2,489 (248 percent)
- V11_C_deep_ens4 2013: Sharpe 4.15 above 3
- V11_C_deep_ens4 2015: Sharpe 3.1 above 3
- V11_C_deep_ens4 2017: Sharpe 4.32 above 3
- V11_C_deep_ens4 2021: best day 2021-08-22 earned 14,955 of the year's 18,919 (79 percent)
- V11_C_gbm_ens4 2013: best day 2013-03-28 earned 9,372 of the year's 9,329 (100 percent)
- V11_C_gbm_ens4 2019: best day 2019-03-06 earned 2,532 of the year's 2,475 (102 percent)
- V11_C_gbm_ens4 2022: Sharpe 3.14 above 3
- V12_alloc 2013: Sharpe 4.76 above 3
- V12_alloc 2014: Sharpe 4.23 above 3
- V12_alloc 2016: Sharpe 3.17 above 3
- V12_alloc 2017: Sharpe 4.14 above 3
- V12_alloc 2019: Sharpe 3.03 above 3
- V12_alloc 2021: Sharpe 3.15 above 3
- V13_A_deep_all 2015: best day 2015-02-23 earned 10,546 of the year's 21,008 (50 percent)
- V13_A_deep_all 2020: best day 2020-06-24 earned 1,858 of the year's 2,691 (69 percent)
- V13_A_deep_all 2023: best day 2023-07-28 earned 6,128 of the year's 11,063 (55 percent)
- V13_C_deep_all 2013: Sharpe 3.59 above 3
- V13_C_deep_all 2016: best day 2016-03-28 earned 17,558 of the year's 25,204 (70 percent)
- V13_C_deep_all 2017: Sharpe 3.85 above 3
- V13_C_deep_all 2020: best day 2020-08-29 earned 2,028 of the year's 1,439 (141 percent)
- V14a_window_ens 2015: Sharpe 3.17 above 3
- V14a_window_ens 2017: Sharpe 4.42 above 3
- V14a_window_ens 2020: Sharpe 3.47 above 3
- V14b_quantile_size 2013: Sharpe 3.47 above 3
- V14b_quantile_size 2014: Sharpe 3.38 above 3
- V14b_quantile_size 2018: Sharpe 3.2 above 3
- V14b_quantile_size 2019: best day 2019-09-17 earned 895 of the year's 1,047 (85 percent)
- V14c_conformal_skip 2016: best day 2016-07-12 earned 4,856 of the year's 9,170 (53 percent)
- V14c_conformal_skip 2019: best day 2019-07-22 earned 1,839 of the year's 936 (196 percent)
- V14d_seed_ens 2014: Sharpe 4.37 above 3
- V14d_seed_ens 2015: Sharpe 3.57 above 3
- V14d_seed_ens 2021: best day 2021-08-22 earned 13,898 of the year's 19,803 (70 percent)
- V14e_mlp_multi 2013: Sharpe 4.44 above 3
- V14e_mlp_multi 2014: Sharpe 3.64 above 3
- V14e_mlp_multi 2017: Sharpe 4.56 above 3
- V14e_mlp_multi 2023: best day 2023-02-05 earned 8,010 of the year's 9,703 (83 percent)
- V14f_global_zone 2015: Sharpe 3.49 above 3
- V14f_global_zone 2017: Sharpe 3.39 above 3
- V14f_global_zone 2018: Sharpe 3.74 above 3
- V14f_global_zone 2023: best day 2023-02-27 earned 6,846 of the year's 1,159 (591 percent)
- V15a_V4_limit 2013: Sharpe 4.64 above 3
- V15a_V4_limit 2014: Sharpe 3.33 above 3
- V15a_V4_limit 2016: Sharpe 4.55 above 3
- V15a_V4_limit 2017: Sharpe 4.56 above 3
- V15a_V4_limit 2018: Sharpe 4.16 above 3
- V15a_V5_limit 2013: Sharpe 3.95 above 3
- V15a_V5_limit 2014: Sharpe 3.08 above 3
- V15a_V5_limit 2017: Sharpe 3.76 above 3
- V15a_V5_limit 2018: Sharpe 3.27 above 3
- V15a_V5_limit 2021: Sharpe 3.34 above 3
- V15a_V5_limit 2023: best day 2023-09-05 earned 18,650 of the year's 27,734 (67 percent)
- V15a_V8_limit 2013: Sharpe 3.74 above 3
- V15a_V8_limit 2014: Sharpe 3.19 above 3
- V15a_V8_limit 2017: Sharpe 3.75 above 3
- V15a_V8_limit 2018: Sharpe 3.1 above 3
- V15a_V8_limit 2019: Sharpe 3.37 above 3
- V15a_V8_limit 2021: Sharpe 4.3 above 3
- V15a_V8_limit 2023: best day 2023-09-05 earned 18,496 of the year's 13,146 (141 percent)
- V15b_V13_both 2016: Sharpe 3.31 above 3
- V15b_V13_both 2017: Sharpe 3.45 above 3
- V15b_V13_both 2020: Sharpe 4.19 above 3
- V15b_V13_both 2022: best day 2022-12-24 earned 45,181 of the year's 47,351 (95 percent)
- V15b_V13_limit 2015: Sharpe 3.14 above 3
- V15b_V13_limit 2017: Sharpe 3.86 above 3
- V15b_V13_limit 2020: Sharpe 4.33 above 3
- V15c_wx_border_gbm 2013: Sharpe 4.57 above 3
- V15c_wx_border_gbm 2014: Sharpe 3.93 above 3
- V15c_wx_border_gbm 2016: Sharpe 4.23 above 3
- V15c_wx_border_gbm 2017: Sharpe 3.83 above 3
- V15c_wx_border_gbm 2018: Sharpe 3.16 above 3
- V15c_wx_border_gbm 2023: best day 2023-09-05 earned 18,778 of the year's 2,921 (643 percent)
- V15d_V4_pairs 2019: best day 2019-01-12 earned 3,508 of the year's 4,564 (77 percent)
- V16_B_gefs_joined 2013: Sharpe 4.59 above 3
- V16_B_gefs_joined 2014: Sharpe 4.32 above 3
- V16_B_gefs_joined 2016: Sharpe 4.62 above 3
- V16_B_gefs_joined 2017: Sharpe 4.52 above 3
- V16_B_gefs_joined 2018: Sharpe 3.72 above 3
- V16_B_gefs_joined 2023: best day 2023-09-05 earned 19,784 of the year's 27,364 (72 percent)
- V1_A_spike_gbm 2015: best day 2015-06-02 earned 4,909 of the year's 6,714 (73 percent)
- V1_C_deep 2013: Sharpe 4.19 above 3
- V1_C_deep 2015: Sharpe 3.18 above 3
- V1_C_deep 2017: Sharpe 4.62 above 3
- V1_C_gbm 2013: best day 2013-01-20 earned 9,035 of the year's 6,157 (147 percent)
- V1_C_gbm 2016: best day 2016-03-28 earned 10,118 of the year's 12,134 (83 percent)
- V1_C_gbm 2019: best day 2019-08-11 earned 3,586 of the year's 848 (423 percent)
- V1_baseline 2020: Sharpe 3.38 above 3
- V2_limit_gbm 2014: Sharpe 3.25 above 3
- V2_limit_gbm 2016: Sharpe 3.31 above 3
- V2_limit_gbm 2017: Sharpe 3.12 above 3
- V2_limit_gbm 2019: Sharpe 3.58 above 3
- V2_limit_gbm 2021: Sharpe 3.94 above 3
- V2_limit_gbm 2023: best day 2023-09-05 earned 19,499 of the year's 2,064 (945 percent)
- V2_sides_no_limit 2013: Sharpe 3.01 above 3
- V2_sides_no_limit 2019: Sharpe 3.78 above 3
- V2_sides_no_limit 2021: Sharpe 4.08 above 3
- V3_tail_gbm 2013: Sharpe 3.69 above 3
- V3_tail_gbm 2018: Sharpe 3.08 above 3
- V3_tail_gbm 2019: Sharpe 3.35 above 3
- V3_tail_gbm 2020: Sharpe 3.14 above 3
- V3_tail_gbm 2021: Sharpe 3.75 above 3
- V3_tail_gbm 2023: best day 2023-09-05 earned 59,060 of the year's 70,495 (84 percent)
- V3_tail_gbm_1MW 2013: Sharpe 3.51 above 3
- V3_tail_gbm_1MW 2019: Sharpe 3.4 above 3
- V3_tail_gbm_1MW 2021: Sharpe 3.8 above 3
- V3_tail_gbm_1MW 2023: best day 2023-09-05 earned 20,427 of the year's 26,326 (78 percent)
- V4_B_reforecast 2013: Sharpe 4.59 above 3
- V4_B_reforecast 2014: Sharpe 4.32 above 3
- V4_B_reforecast 2016: Sharpe 4.62 above 3
- V4_B_reforecast 2017: Sharpe 4.52 above 3
- V4_B_reforecast 2018: Sharpe 3.72 above 3
- V5_border_inputs 2013: Sharpe 3.93 above 3
- V5_border_inputs 2014: Sharpe 3.84 above 3
- V5_border_inputs 2015: best day 2015-09-29 earned 10,636 of the year's 7,532 (141 percent)
- V5_border_inputs 2017: Sharpe 3.23 above 3
- V6_LI 2015: best day 2015-06-19 earned 1,618 of the year's 3,073 (53 percent)
- V6_LI 2020: Sharpe 3.26 above 3
- V6_LI 2021: best day 2021-08-22 earned 3,246 of the year's 3,823 (85 percent)
- V6_NYC 2014: Sharpe 3.79 above 3
- V6_NYC 2022: best day 2022-06-13 earned 1,900 of the year's 991 (192 percent)
- V7_flags 2013: Sharpe 3.29 above 3
- V7_flags 2014: Sharpe 3.74 above 3
- V7_flags 2015: best day 2015-09-29 earned 10,001 of the year's 16,658 (60 percent)
- V7_flags 2018: Sharpe 3.13 above 3
- V7_flags 2019: Sharpe 3.37 above 3
- V7_flags 2020: Sharpe 3.08 above 3
- V7_recency 2014: Sharpe 3.74 above 3
- V7_recency 2019: Sharpe 3.37 above 3
- V7_recency 2020: Sharpe 3.64 above 3
- V8_error_mining 2013: Sharpe 3.82 above 3
- V8_error_mining 2014: Sharpe 3.7 above 3
- V8_error_mining 2017: Sharpe 3.58 above 3
- V8_error_mining 2018: Sharpe 3.8 above 3
- V8_error_mining 2019: Sharpe 3.17 above 3
- V8_error_mining 2021: Sharpe 3.18 above 3
- V8_error_mining 2023: best day 2023-09-05 earned 20,335 of the year's 21,662 (94 percent)
- V9_decompose 2013: Sharpe 4.49 above 3
- V9_decompose 2014: Sharpe 3.52 above 3
- V9_decompose 2021: Sharpe 3.93 above 3
- V9_decompose 2022: best day 2022-12-24 earned 148,035 of the year's 224,138 (66 percent)

Tries: 38 v2 strategy rows, 87 settings tried in all on 2013..2023 rolling quarters, on top of everything v1 tried on 2021..2023.

Sizing view (not part of the verdict): scale = MW per position at which the worst drawdown is 100,000 USD, capped at 5.

