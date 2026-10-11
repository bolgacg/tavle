# Version 1, heldout run: 2024-01-01 to 2026-09-30

Written 2026-10-10 22:55:04 CEST by `model/heldout_v1.py`. Every row walked forward as in the build years, scored by side/lab.py's scoring (full costs, 1 MW per zone-hour). 'three_year' in each row is the whole window (the lab's field name); 'years' are calendar years of the window; 2026 runs to 30 September; sizing_view is recomputed on the window, sizing_view_carried uses the build's scale.

| # | Strategy | 2024 | 2025 | 2026 | Window total | Sharpe | Max drawdown | At 0.50 stress | One day late | Shuffled (share at or above) | Random days beat it | Build scale (MW) | Window net at that scale | Return a year at that scale | FRAGILE on the window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea13_portfolio | 61,354 | 146,822 | 436,632 | 644,808 | 1.70 | -67,895 | 560,245 | n/a | n/a | 0% | 0.594 | 383,016 | 27.8% | no |
| 2 | idea11_learned_allocator | 69,959 | 42,360 | 415,808 | 528,126 | 1.54 | -53,490 | 467,768 | 397,517 | 252,773 (0.05) | 1% | 1.79 | 945,346 | 68.7% | no |
| 3 | idea12_policy_network | 94,466 | 39,271 | 378,017 | 511,754 | 1.66 | -50,534 | 466,485 | 363,769 | 84,223 (0.00) | 0% | 1.76 | 900,687 | 65.5% | no |
| 4 | D_gbm | 55,266 | 92,928 | 360,103 | 508,296 | 1.41 | -117,676 | 434,624 | 361,279 | 46,970 (0.00) | 0% | 2.08 | 1,057,256 | 76.9% | no |
| 5 | A_spike_deep_rolling_cut | 69,389 | 84,604 | 353,970 | 507,963 | 1.45 | -56,836 | 431,695 | 439,048 | 351,188 (0.00) | 0% | 0.59 | 299,698 | 21.8% | no |
| 6 | idea14_learned_allocator_candidates | 15,555 | 136,334 | 345,393 | 497,282 | 1.68 | -50,599 | 411,401 | 473,239 | 522,975 (0.80) | 0% | 0.55 | 273,505 | 19.9% | no |
| 7 | B_gbm | 14,771 | 24,143 | 427,638 | 466,552 | 1.29 | -77,292 | 391,830 | 251,325 | 76,298 (0.00) | 0% | 2.3 | 1,073,069 | 78.0% | no |
| 8 | idea5_deep_zone_day | 78,701 | -5,547 | 367,704 | 440,858 | 1.16 | -133,329 | 373,860 | 461,840 | 284,702 (0.10) | 6% | 0.59 | 260,106 | 18.9% | no |
| 9 | B_deep | 44,528 | 39,707 | 348,359 | 432,594 | 1.30 | -42,508 | 361,180 | 462,644 | 86,063 (0.00) | 3% | 1.45 | 627,262 | 45.6% | no |
| 10 | A_regression_v1 | 60,522 | 33,440 | 273,391 | 367,353 | 1.18 | -104,714 | 325,265 | 202,610 | 192,034 (0.05) | 1% | 1.71 | 628,174 | 45.7% | no |
| 11 | A_regression_v1_deep | 57,917 | 4,244 | 304,505 | 366,667 | 1.06 | -101,329 | 324,210 | 417,787 | 218,386 (0.00) | 4% | 0.65 | 238,334 | 17.3% | FRAGILE |
| 12 | side_past_spike_rate_filter | 58,791 | 38,670 | 261,300 | 358,761 | 1.18 | -100,734 | 287,151 | 358,694 | 358,764 (1.00) | 1% | 0.39 | 139,917 | 10.2% | no |
| 13 | always_supply | 58,640 | -14,709 | 303,266 | 347,197 | 0.89 | -159,297 | 268,950 | n/a | n/a | n/a | 0.29 | 100,687 | 7.3% | FRAGILE |
| 14 | idea10_weather_surprise | 57,430 | -69,039 | 343,606 | 331,997 | 0.85 | -158,206 | 253,340 | 394,344 | 299,873 (0.45) | 58% | 1.21 | 401,716 | 29.2% | FRAGILE |
| 15 | D_deep | 70,308 | -12,171 | 273,406 | 331,543 | 0.98 | -111,129 | 260,167 | 379,505 | 84,351 (0.00) | 51% | 0.97 | 321,597 | 23.4% | FRAGILE |
| 16 | idea7_gbm_storm_filter | 32,999 | -46,910 | 328,804 | 314,892 | 0.83 | -164,954 | 241,720 | 220,842 | 314,947 (0.70) | 24% | 1.11 | 349,530 | 25.4% | FRAGILE |
| 17 | idea3_deep_storm_filter | 48,382 | 86,637 | 155,699 | 290,718 | 0.88 | -70,657 | 217,560 | 34,288 | 288,046 (0.60) | 55% | 0.28 | 81,401 | 5.9% | FRAGILE |
| 18 | A_hourly_mean | 49,740 | 10,002 | 157,568 | 217,310 | 1.00 | -68,879 | 159,731 | 178,674 | 421,825 (1.00) | 100% | 0.72 | 156,463 | 11.4% | FRAGILE |
| 19 | idea8_storm_zone_pairs | 56,325 | 107,190 | 45,616 | 209,131 | 1.26 | -55,170 | 135,124 | 91,118 | 250,310 (0.70) | 74% | 0.47 | 98,291 | 7.1% | FRAGILE |
| 20 | zone_subset_supply | 39,103 | -2,706 | 168,568 | 204,965 | 1.02 | -83,537 | 162,285 | 204,811 | 204,843 (0.10) | 0% | 0.61 | 125,029 | 9.1% | FRAGILE |
| 21 | idea9_tail_spike_S100 | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | -63,784 | 100,721 | 131,595 | 318,398 (1.00) | 80% | 0.7 | 121,653 | 8.8% | FRAGILE |
| 22 | A_spike_gbm | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | -63,784 | 100,721 | 131,610 | 312,222 (1.00) | 80% | 1.66 | 288,491 | 21.0% | FRAGILE |
| 23 | A_spike_tail_S100_gbm [second build] | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | -63,784 | 100,721 | 131,595 | 312,222 (1.00) | 80% | 0.64 | 111,225 | 8.1% | FRAGILE |
| 24 | storm_day_filter [second build] | 55,504 | 83,939 | 30,777 | 170,220 | 1.05 | -70,900 | 99,060 | 51,553 | 324,184 (1.00) | 86% | 1.0 | 170,220 | 12.4% | FRAGILE |
| 25 | idea1_storm_filter | 47,743 | 78,732 | 27,741 | 154,217 | 0.95 | -65,402 | 82,799 | 40,113 | 252,883 (0.85) | 85% | 1.07 | 165,012 | 12.0% | FRAGILE |
| 26 | risk_sized_supply_gbm | 16,619 | 40,233 | 93,399 | 150,251 | 1.24 | -28,943 | 110,257 | 132,304 | 206,486 (0.95) | 93% | 2.34 | 351,588 | 25.6% | FRAGILE |
| 27 | C_deep | 3,510 | 92,994 | 47,559 | 144,063 | 2.12 | -15,752 | 95,140 | 146,686 | 100,723 (0.00) | 3% | 3.92 | 564,727 | 41.1% | no |
| 28 | baseline_usual_side | 39,434 | 22,797 | 41,118 | 103,349 | 0.71 | -52,209 | 23,542 | 30,601 | 480,550 (1.00) | 100% | 0.8 | 82,679 | 6.0% | FRAGILE |
| 29 | A_spike_deep | 36,526 | 16,644 | 2,139 | 55,309 | 0.73 | -41,861 | -3,345 | 28,636 | 276,244 (1.00) | 64% | 1.13 | 62,499 | 4.5% | FRAGILE |
| 30 | C_gbm | 8,690 | 40,671 | -3,186 | 46,175 | 0.70 | -48,175 | -4,201 | 8,302 | 3,266 (0.15) | 14% | 3.41 | 157,455 | 11.4% | FRAGILE |
| 31 | idea6_deep_two_gates | 14,435 | 18,870 | 2,139 | 35,444 | 0.49 | -36,542 | -20,297 | 41,613 | 223,503 (1.00) | 70% | 0.98 | 34,735 | 2.5% | FRAGILE |
| 32 | storm_day_flip [second build] | 49,852 | 178,246 | -244,109 | -16,011 | -0.04 | -394,928 | -94,854 | -253,346 | 291,919 (1.00) | 86% | 0.74 | -11,848 | -0.9% | FRAGILE |
| 33 | idea2_storm_flip | 34,499 | 168,043 | -250,300 | -47,758 | -0.12 | -388,894 | -126,585 | -285,498 | 155,946 (0.90) | 85% | 0.74 | -35,341 | -2.6% | FRAGILE |
| 34 | idea4_deep_storm_three_way | 44,531 | 87,227 | -179,786 | -48,029 | -0.23 | -209,205 | -114,441 | -126,949 | 239,477 (1.00) | 64% | 0.28 | -13,448 | -1.0% | FRAGILE |

## Settings chosen on the year before

| Strategy | 2024 | 2025 | 2026 |
|---|---|---|---|
| idea13_portfolio | None | None | None |
| idea11_learned_allocator | None | None | None |
| idea12_policy_network | None | None | None |
| D_gbm | None | None | None |
| A_spike_deep_rolling_cut | None | None | None |
| idea14_learned_allocator_candidates | None | None | None |
| B_gbm | None | None | None |
| idea5_deep_zone_day | 200.0 | 0.0 | 50.0 |
| B_deep | None | None | None |
| A_regression_v1 | None | None | None |
| A_regression_v1_deep | None | None | None |
| side_past_spike_rate_filter | 0.01 | 0.1 | 0.2 |
| always_supply | None | None | None |
| idea10_weather_surprise | 0.98 | 0.9 | 0.98 |
| D_deep | None | None | None |
| idea7_gbm_storm_filter | 0.4 | 0.3 | 0.2 |
| idea3_deep_storm_filter | 0.15 | 0.3 | 0.3 |
| A_hourly_mean | None | None | None |
| idea8_storm_zone_pairs | [0.98, ['WEST', 'CAPITL']] | [0.98, ['WEST', 'CAPITL']] | [0.98, ['NORTH', 'CAPITL']] |
| zone_subset_supply | None | None | None |
| idea9_tail_spike_S100 | 0.05 | 0.05 | 0.02 |
| A_spike_gbm | 0.05 | 0.05 | 0.02 |
| A_spike_tail_S100_gbm [second build] | 0.05 | 0.05 | 0.02 |
| storm_day_filter [second build] | 0.98 | 0.98 | 0.98 |
| idea1_storm_filter | 0.98 | 0.98 | 0.98 |
| risk_sized_supply_gbm | None | None | None |
| C_deep | None | None | None |
| baseline_usual_side | None | None | None |
| A_spike_deep | 0.05 | 0.05 | 0.05 |
| C_gbm | None | None | None |
| idea6_deep_two_gates | [0.15, 0.05] | [0.3, 0.05] | [0.3, 0.05] |
| storm_day_flip [second build] | 0.98 | 0.98 | 0.98 |
| idea2_storm_flip | 0.98 | 0.98 | 0.98 |
| idea4_deep_storm_three_way | [0.15, 0.2] | [0.3, 0.4] | [0.1, 0.3] |
