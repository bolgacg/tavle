# Held-out run, 2024-01-01 to 2026-09-30: DONE

Sources: /home/bolgac/nyiso-us/heldout_results/ho_v1/side/lab_results_heldout.json; /home/bolgac/nyiso-us/heldout_results/ho_v2/results/v2/v2_results_heldout.json; registered verdict run in /home/bolgac/nyiso-us/heldout_results/ho_reh. Net USD at 1 MW, full costs; 2026 is January to September.

| version | row | candidate | 2024 | 2025 | 2026 to Sep | total | Sharpe | PASS |
|---|---|---|---|---|---|---|---|---|
| v1 | idea13_portfolio | yes | 61,354 | 146,822 | 436,632 | 644,808 | 1.70 | True |
| v1 | idea14_learned_allocator_candidates | yes | 15,555 | 136,334 | 345,393 | 497,282 | 1.68 | True |
| v1 | B_gbm | yes | 14,771 | 24,143 | 427,638 | 466,552 | 1.29 | True |
| v1 | B_deep | yes | 44,528 | 39,707 | 348,359 | 432,594 | 1.30 | True |
| v1 | A_regression_v1 | yes | 60,522 | 33,440 | 273,391 | 367,353 | 1.18 | True |
| v1 | C_deep | yes | 3,510 | 92,994 | 47,559 | 144,063 | 2.12 | False |
| v1 | C_gbm | yes | 8,690 | 40,671 | -3,186 | 46,175 | 0.70 | False |
| v1 | idea11_learned_allocator | no | 69,959 | 42,360 | 415,808 | 528,126 | 1.54 | True |
| v1 | idea12_policy_network | no | 94,466 | 39,271 | 378,017 | 511,754 | 1.66 | True |
| v1 | D_gbm | no | 55,266 | 92,928 | 360,103 | 508,296 | 1.41 | True |
| v1 | A_spike_deep_rolling_cut | no | 69,389 | 84,604 | 353,970 | 507,963 | 1.45 | True |
| v1 | idea5_deep_zone_day | no | 78,701 | -5,547 | 367,704 | 440,858 | 1.16 | True |
| v1 | A_regression_v1_deep | no | 57,917 | 4,244 | 304,505 | 366,667 | 1.06 | True |
| v1 | side_past_spike_rate_filter | no | 58,791 | 38,670 | 261,300 | 358,761 | 1.18 | True |
| v1 | always_supply | no | 58,640 | -14,709 | 303,266 | 347,197 | 0.89 | True |
| v1 | idea10_weather_surprise | no | 57,430 | -69,039 | 343,606 | 331,997 | 0.85 | True |
| v1 | D_deep | no | 70,308 | -12,171 | 273,406 | 331,543 | 0.98 | True |
| v1 | idea7_gbm_storm_filter | no | 32,999 | -46,910 | 328,804 | 314,892 | 0.83 | True |
| v1 | idea3_deep_storm_filter | no | 48,382 | 86,637 | 155,699 | 290,718 | 0.88 | True |
| v1 | A_hourly_mean | no | 49,740 | 10,002 | 157,568 | 217,310 | 1.00 | True |
| v1 | idea8_storm_zone_pairs | no | 56,325 | 107,190 | 45,616 | 209,131 | 1.26 | True |
| v1 | zone_subset_supply | no | 39,103 | -2,706 | 168,568 | 204,965 | 1.02 | True |
| v1 | idea9_tail_spike_S100 | no | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | True |
| v1 | A_spike_gbm | no | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | True |
| v1 | A_spike_tail_S100_gbm [second build] | no | 24,923 | 65,688 | 83,179 | 173,790 | 0.92 | True |
| v1 | storm_day_filter [second build] | no | 55,504 | 83,939 | 30,777 | 170,220 | 1.05 | True |
| v1 | idea1_storm_filter | no | 47,743 | 78,732 | 27,741 | 154,217 | 0.95 | True |
| v1 | risk_sized_supply_gbm | no | 16,619 | 40,233 | 93,399 | 150,251 | 1.24 | True |
| v1 | baseline_usual_side | no | 39,434 | 22,797 | 41,118 | 103,349 | 0.71 | False |
| v1 | A_spike_deep | no | 36,526 | 16,644 | 2,139 | 55,309 | 0.73 | False |
| v1 | idea6_deep_two_gates | no | 14,435 | 18,870 | 2,139 | 35,444 | 0.49 | False |
| v1 | storm_day_flip [second build] | no | 49,852 | 178,246 | -244,109 | -16,011 | -0.04 | False |
| v1 | idea2_storm_flip | no | 34,499 | 168,043 | -250,300 | -47,758 | -0.12 | False |
| v1 | idea4_deep_storm_three_way | no | 44,531 | 87,227 | -179,786 | -48,029 | -0.23 | False |
| v2 | V3_tail_gbm | yes | 120,061 | 186,854 | 951,812 | 1,258,726 | 1.21 | True |
| v2 | V15c_wx_border_gbm | yes | 85,116 | 30,145 | 318,450 | 433,710 | 1.24 | True |
| v2 | V15a_V5_limit | yes | 41,672 | 61,707 | 311,797 | 415,175 | 1.23 | True |
| v2 | V16_B_gefs_joined | yes | 36,530 | 28,070 | 346,350 | 410,950 | 1.17 | True |
| v2 | V7_flags | yes | 64,037 | -10,674 | 346,599 | 399,962 | 1.15 | True |
| v2 | V2_limit_gbm | yes | 26,963 | 19,491 | 344,177 | 390,630 | 1.09 | True |
| v2 | V15b_V13_both | yes | 54,277 | 107,405 | 228,533 | 390,214 | 1.49 | True |
| v2 | V8_error_mining | yes | 35,123 | 14,892 | 292,062 | 342,077 | 1.00 | True |
| v2 | V15a_V8_limit | yes | 16,782 | 23,076 | 294,595 | 334,453 | 0.97 | True |
| v2 | V7_recency | yes | 43,657 | -45,020 | 315,817 | 314,453 | 0.91 | True |
| v2 | V15b_V13_limit | yes | 18,186 | 48,126 | 238,640 | 304,952 | 1.06 | True |
| v2 | V5_border_inputs | yes | 61,389 | 96,820 | 134,878 | 293,087 | 1.05 | True |
| v2 | V9_decompose | yes | 17,076 | 71,301 | 203,628 | 292,005 | 0.98 | True |
| v2 | V12_alloc | yes | 59,070 | -158,157 | 318,530 | 219,443 | 0.35 | False |
| v2 | V1_C_deep | yes | 16,151 | 90,959 | 22,513 | 129,623 | 2.39 | False |
| v2 | V13_C_deep_all | yes | 13,124 | 91,541 | 12,440 | 117,104 | 2.48 | False |
| v2 | V11_C_gbm_ens4 | yes | 17,599 | 43,396 | 54,163 | 115,158 | 1.89 | False |
| v2 | V11_C_deep_ens4 | yes | 10,175 | 74,580 | 23,958 | 108,714 | 2.26 | False |
| v2 | V14d_seed_ens | yes | 32,327 | 34,008 | 37,479 | 103,814 | 2.25 | False |
| v2 | V1_C_gbm | yes | 20,543 | 28,829 | 40,485 | 89,857 | 1.83 | False |
| v2 | V14f_global_zone | yes | 17,138 | 39,419 | 32,969 | 89,526 | 2.10 | False |
| v2 | V14c_conformal_skip | yes | 17,417 | 35,468 | 29,524 | 82,409 | 1.59 | False |
| v2 | V14a_window_ens | yes | 35,242 | 12,279 | 30,754 | 78,275 | 1.48 | False |
| v2 | V14e_mlp_multi | yes | -11,367 | 58,711 | 7,230 | 54,574 | 1.28 | False |
| v2 | V6_LI | yes | 5,479 | -13,304 | 19,564 | 11,739 | 0.38 | False |
| v2 | V14b_quantile_size | yes | 3,327 | 4,329 | 2,451 | 10,107 | 1.34 | False |
| v2 | V2_sides_no_limit | no | 45,179 | 49,190 | 367,351 | 461,720 | 1.26 | True |
| v2 | V3_tail_gbm_1MW | no | 35,982 | 72,097 | 323,511 | 431,591 | 1.22 | True |
| v2 | V1_always_supply | no | 58,640 | -14,709 | 303,266 | 347,197 | 0.89 | True |
| v2 | V13_A_deep_all | no | 52,738 | 28,587 | 254,005 | 335,331 | 1.05 | True |
| v2 | C_static_pairs | no | 17,294 | 102,043 | 27,938 | 147,275 | 2.29 | False |
| v2 | V10_C_deep_pretrain | no | 11,743 | 72,670 | 28,311 | 112,724 | 1.97 | False |
| v2 | V1_baseline | no | 39,434 | 22,797 | 41,118 | 103,349 | 0.71 | False |
| v2 | V6_NYC | no | 7,448 | 7,516 | 30,240 | 45,204 | 1.24 | False |
| v2 | V1_A_spike_gbm | no | 18,104 | 27,400 | -1,113 | 44,391 | 0.34 | False |

## Too-good audit of every row that clears the bar

### v1 always_supply

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78703,
  "2026-01-29": 74080,
  "2026-01-26": 41262,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": null,
  "shuffled_mean_total": null,
  "note": "no score to perturb"
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.89,
  "sortino": 1.66,
  "return_over_max_drawdown": 0.73,
  "scale_for_100k_drawdown": 0.63,
  "return_on_500k_at_that_scale_pct": 14.5,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.29,
  "net_usd": 100687,
  "return_on_500k_annualised_pct": 7.3,
  "max_drawdown_usd": -46196,
  "days": 1004,
  "years": {
   "2024": 17006,
   "2025": -4266,
   "2026": 87947
  }
 },
 "random_days": "not applicable: the same book every day",
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -30193,
   "three_year_total_without_this_years_best_10": 258363
  },
  "2025": {
   "net_without_best_10": -120435,
   "three_year_total_without_this_years_best_10": 241471
  },
  "2026": {
   "net_without_best_10": -128648,
   "three_year_total_without_this_years_best_10": -84717
  }
 }
}
```

### v1 A_hourly_mean

```
{
 "best_5_days_three_year": {
  "2026-01-28": 70852,
  "2026-01-29": 49589,
  "2026-01-30": 26368,
  "2026-01-27": 23881,
  "2026-07-15": 21464
 },
 "placebo": {
  "shifted_one_day_total": 178674,
  "shifted_one_day_years": {
   "2024": 48172,
   "2025": 7364,
   "2026": 123138
  },
  "shuffled_mean_total": 421825,
  "shuffled_p95_total": 449069,
  "shuffled_max_total": 470914,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 8,250 of the year's 10,002 (82 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2025,
   "flag": "best day 2025-01-20 earned 8,250 of the year's 10,002 (82 percent)",
   "best_day": "2025-01-20",
   "best_day_usd": 8250,
   "year_net_usd": 10002,
   "year_net_without_best_day": 1752,
   "mean_gap_that_day_usd_per_mwh": -74.6,
   "max_gap_that_day": 135.5,
   "always_supply_that_day_usd": 19635,
   "position_mix_that_day": {
    "load": 0,
    "supply": 113,
    "none": 151
   }
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.0,
  "sortino": 1.79,
  "return_over_max_drawdown": 1.05,
  "scale_for_100k_drawdown": 1.45,
  "return_on_500k_at_that_scale_pct": 21.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.72,
  "net_usd": 156463,
  "return_on_500k_annualised_pct": 11.4,
  "max_drawdown_usd": -49593,
  "days": 1004,
  "years": {
   "2024": 35813,
   "2025": 7202,
   "2026": 113449
  }
 },
 "random_days": 1.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -26249,
   "three_year_total_without_this_years_best_10": 141321
  },
  "2025": {
   "net_without_best_10": -44186,
   "three_year_total_without_this_years_best_10": 163121
  },
  "2026": {
   "net_without_best_10": -85630,
   "three_year_total_without_this_years_best_10": -25888
  }
 }
}
```

### v1 idea1_storm_filter

```
{
 "best_5_days_three_year": {
  "2026-01-26": 41262,
  "2024-01-17": 15323,
  "2025-01-19": 14635,
  "2024-01-16": 12917,
  "2026-07-14": 12090
 },
 "placebo": {
  "shifted_one_day_total": 40113,
  "shifted_one_day_years": {
   "2024": 21377,
   "2025": 40795,
   "2026": -22059
  },
  "shuffled_mean_total": 252883,
  "shuffled_p95_total": 405988,
  "shuffled_max_total": 506523,
  "shuffled_share_at_or_above_real": 0.85,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-26 earned 41,262 of the year's 27,741 (149 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-26 earned 41,262 of the year's 27,741 (149 percent)",
   "best_day": "2026-01-26",
   "best_day_usd": 41262,
   "year_net_usd": 27741,
   "year_net_without_best_day": -13521,
   "mean_gap_that_day_usd_per_mwh": -156.5,
   "max_gap_that_day": 73.9,
   "always_supply_that_day_usd": 41262,
   "position_mix_that_day": {
    "load": 0,
    "supply": 264,
    "none": 0
   },
   "storm_that_day": 0.9479,
   "storm_day_before": 0.9968
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.95,
  "sortino": 1.37,
  "return_over_max_drawdown": 0.79,
  "scale_for_100k_drawdown": 1.53,
  "return_on_500k_at_that_scale_pct": 15.7,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.07,
  "net_usd": 165012,
  "return_on_500k_annualised_pct": 12.0,
  "max_drawdown_usd": -69980,
  "days": 1004,
  "years": {
   "2024": 51086,
   "2025": 84244,
   "2026": 29683
  }
 },
 "random_days": 0.853,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -31303,
   "three_year_total_without_this_years_best_10": 75171
  },
  "2025": {
   "net_without_best_10": -1959,
   "three_year_total_without_this_years_best_10": 73526
  },
  "2026": {
   "net_without_best_10": -75349,
   "three_year_total_without_this_years_best_10": 51126
  }
 }
}
```

### v1 idea8_storm_zone_pairs

```
{
 "best_5_days_three_year": {
  "2026-01-26": 41262,
  "2024-01-17": 15323,
  "2025-01-19": 14635,
  "2024-01-16": 12917,
  "2026-07-14": 12090
 },
 "placebo": {
  "shifted_one_day_total": 91118,
  "shifted_one_day_years": {
   "2024": 22737,
   "2025": 78357,
   "2026": -9976
  },
  "shuffled_mean_total": 250310,
  "shuffled_p95_total": 405866,
  "shuffled_max_total": 525648,
  "shuffled_share_at_or_above_real": 0.7,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-26 earned 41,262 of the year's 45,616 (90 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-26 earned 41,262 of the year's 45,616 (90 percent)",
   "best_day": "2026-01-26",
   "best_day_usd": 41262,
   "year_net_usd": 45616,
   "year_net_without_best_day": 4354,
   "mean_gap_that_day_usd_per_mwh": -156.5,
   "max_gap_that_day": 73.9,
   "always_supply_that_day_usd": 41262,
   "position_mix_that_day": {
    "load": 0,
    "supply": 264,
    "none": 0
   },
   "storm_that_day": 0.9479,
   "storm_day_before": 0.9968
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.26,
  "sortino": 1.85,
  "return_over_max_drawdown": 1.26,
  "scale_for_100k_drawdown": 1.81,
  "return_on_500k_at_that_scale_pct": 25.3,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.47,
  "net_usd": 98291,
  "return_on_500k_annualised_pct": 7.1,
  "max_drawdown_usd": -25930,
  "days": 1004,
  "years": {
   "2024": 26473,
   "2025": 50379,
   "2026": 21439
  }
 },
 "random_days": 0.742,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -22722,
   "three_year_total_without_this_years_best_10": 130084
  },
  "2025": {
   "net_without_best_10": 21699,
   "three_year_total_without_this_years_best_10": 123639
  },
  "2026": {
   "net_without_best_10": -64681,
   "three_year_total_without_this_years_best_10": 98834
  }
 }
}
```

### v1 idea10_weather_surprise

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78703,
  "2026-01-29": 74080,
  "2026-01-26": 41262,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": 394344,
  "shifted_one_day_years": {
   "2024": 41529,
   "2025": -7819,
   "2026": 360634
  },
  "shuffled_mean_total": 299873,
  "shuffled_p95_total": 481621,
  "shuffled_max_total": 605908,
  "shuffled_share_at_or_above_real": 0.45,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.85,
  "sortino": 1.58,
  "return_over_max_drawdown": 0.7,
  "scale_for_100k_drawdown": 0.63,
  "return_on_500k_at_that_scale_pct": 14.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.21,
  "net_usd": 401716,
  "return_on_500k_annualised_pct": 29.2,
  "max_drawdown_usd": -191429,
  "days": 1004,
  "years": {
   "2024": 69491,
   "2025": -83538,
   "2026": 415763
  }
 },
 "random_days": 0.583,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -31403,
   "three_year_total_without_this_years_best_10": 243163
  },
  "2025": {
   "net_without_best_10": -166539,
   "three_year_total_without_this_years_best_10": 234497
  },
  "2026": {
   "net_without_best_10": -97250,
   "three_year_total_without_this_years_best_10": -108859
  }
 }
}
```

### v1 idea9_tail_spike_S100

```
{
 "best_5_days_three_year": {
  "2026-01-27": 46758,
  "2026-01-28": 33809,
  "2026-01-29": 32252,
  "2026-01-26": 18562,
  "2025-01-20": 17724
 },
 "placebo": {
  "shifted_one_day_total": 131595,
  "shifted_one_day_years": {
   "2024": 15276,
   "2025": 28233,
   "2026": 88086
  },
  "shuffled_mean_total": 318398,
  "shuffled_p95_total": 380283,
  "shuffled_max_total": 389576,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)",
   "best_day": "2026-01-27",
   "best_day_usd": 46758,
   "year_net_usd": 83179,
   "year_net_without_best_day": 36421,
   "mean_gap_that_day_usd_per_mwh": -298.4,
   "max_gap_that_day": -36.1,
   "always_supply_that_day_usd": 78703,
   "position_mix_that_day": {
    "load": 0,
    "supply": 157,
    "none": 107
   }
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.92,
  "sortino": 1.41,
  "return_over_max_drawdown": 0.91,
  "scale_for_100k_drawdown": 1.57,
  "return_on_500k_at_that_scale_pct": 18.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.7,
  "net_usd": 121653,
  "return_on_500k_annualised_pct": 8.8,
  "max_drawdown_usd": -44649,
  "days": 1004,
  "years": {
   "2024": 17446,
   "2025": 45981,
   "2026": 58225
  }
 },
 "random_days": 0.8,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -25647,
   "three_year_total_without_this_years_best_10": 123219
  },
  "2025": {
   "net_without_best_10": -16107,
   "three_year_total_without_this_years_best_10": 91994
  },
  "2026": {
   "net_without_best_10": -75091,
   "three_year_total_without_this_years_best_10": 15520
  }
 }
}
```

### v1 idea7_gbm_storm_filter

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78703,
  "2026-01-29": 74080,
  "2026-01-26": 41262,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": 220842,
  "shifted_one_day_years": {
   "2024": 44838,
   "2025": -19757,
   "2026": 195761
  },
  "shuffled_mean_total": 314947,
  "shuffled_p95_total": 419730,
  "shuffled_max_total": 459579,
  "shuffled_share_at_or_above_real": 0.7,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.83,
  "sortino": 1.6,
  "return_over_max_drawdown": 0.64,
  "scale_for_100k_drawdown": 0.61,
  "return_on_500k_at_that_scale_pct": 12.7,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.11,
  "net_usd": 349530,
  "return_on_500k_annualised_pct": 25.4,
  "max_drawdown_usd": -183099,
  "days": 1004,
  "years": {
   "2024": 36628,
   "2025": -52071,
   "2026": 364972
  }
 },
 "random_days": 0.239,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -50384,
   "three_year_total_without_this_years_best_10": 231509
  },
  "2025": {
   "net_without_best_10": -122029,
   "three_year_total_without_this_years_best_10": 239773
  },
  "2026": {
   "net_without_best_10": -81687,
   "three_year_total_without_this_years_best_10": -95599
  }
 }
}
```

### v1 idea3_deep_storm_filter

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78703,
  "2026-07-15": 22533,
  "2025-01-20": 19635,
  "2025-01-19": 14635
 },
 "placebo": {
  "shifted_one_day_total": 34288,
  "shifted_one_day_years": {
   "2024": 55505,
   "2025": -20035,
   "2026": -1182
  },
  "shuffled_mean_total": 288046,
  "shuffled_p95_total": 375551,
  "shuffled_max_total": 464431,
  "shuffled_share_at_or_above_real": 0.6,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-28 earned 140,584 of the year's 155,699 (90 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-28 earned 140,584 of the year's 155,699 (90 percent)",
   "best_day": "2026-01-28",
   "best_day_usd": 140584,
   "year_net_usd": 155699,
   "year_net_without_best_day": 15115,
   "mean_gap_that_day_usd_per_mwh": -532.8,
   "max_gap_that_day": -85.1,
   "always_supply_that_day_usd": 140584,
   "position_mix_that_day": {
    "load": 0,
    "supply": 264,
    "none": 0
   },
   "p_storm_that_day": 0.255,
   "p_storm_day_before": 0.297
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.88,
  "sortino": 1.8,
  "return_over_max_drawdown": 1.37,
  "scale_for_100k_drawdown": 1.42,
  "return_on_500k_at_that_scale_pct": 27.4,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.28,
  "net_usd": 81401,
  "return_on_500k_annualised_pct": 5.9,
  "max_drawdown_usd": -19784,
  "days": 1004,
  "years": {
   "2024": 13547,
   "2025": 24258,
   "2026": 43596
  }
 },
 "random_days": 0.548,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -18382,
   "three_year_total_without_this_years_best_10": 223954
  },
  "2025": {
   "net_without_best_10": -14404,
   "three_year_total_without_this_years_best_10": 189677
  },
  "2026": {
   "net_without_best_10": -148156,
   "three_year_total_without_this_years_best_10": -13137
  }
 }
}
```

### v1 idea5_deep_zone_day

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78703,
  "2026-01-29": 74080,
  "2026-01-26": 41262,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": 461840,
  "shifted_one_day_years": {
   "2024": 85915,
   "2025": 40848,
   "2026": 335077
  },
  "shuffled_mean_total": 284702,
  "shuffled_p95_total": 453104,
  "shuffled_max_total": 611364,
  "shuffled_share_at_or_above_real": 0.1,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.16,
  "sortino": 2.29,
  "return_over_max_drawdown": 1.1,
  "scale_for_100k_drawdown": 0.75,
  "return_on_500k_at_that_scale_pct": 22.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.59,
  "net_usd": 260106,
  "return_on_500k_annualised_pct": 18.9,
  "max_drawdown_usd": -78664,
  "days": 1004,
  "years": {
   "2024": 46434,
   "2025": -3273,
   "2026": 216945
  }
 },
 "random_days": 0.055,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -6888,
   "three_year_total_without_this_years_best_10": 355269
  },
  "2025": {
   "net_without_best_10": -104308,
   "three_year_total_without_this_years_best_10": 342096
  },
  "2026": {
   "net_without_best_10": -64210,
   "three_year_total_without_this_years_best_10": 8944
  }
 }
}
```

### v1 idea11_learned_allocator

```
{
 "best_5_days_three_year": {
  "2026-01-28": 139930,
  "2026-01-27": 78407,
  "2026-01-29": 73525,
  "2026-01-26": 40628,
  "2026-01-30": 34614
 },
 "placebo": {
  "shifted_one_day_total": 397517,
  "shifted_one_day_years": {
   "2024": 58732,
   "2025": 80841,
   "2026": 257944
  },
  "shuffled_mean_total": 252773,
  "shuffled_p95_total": 417514,
  "shuffled_max_total": 544915,
  "shuffled_share_at_or_above_real": 0.05,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.54,
  "sortino": 4.68,
  "return_over_max_drawdown": 3.29,
  "scale_for_100k_drawdown": 1.87,
  "return_on_500k_at_that_scale_pct": 65.8,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.79,
  "net_usd": 945346,
  "return_on_500k_annualised_pct": 68.7,
  "max_drawdown_usd": -95747,
  "days": 1004,
  "years": {
   "2024": 125226,
   "2025": 75824,
   "2026": 744296
  }
 },
 "random_days": 0.007,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -4945,
   "three_year_total_without_this_years_best_10": 453222
  },
  "2025": {
   "net_without_best_10": -56033,
   "three_year_total_without_this_years_best_10": 429733
  },
  "2026": {
   "net_without_best_10": -18455,
   "three_year_total_without_this_years_best_10": 93863
  }
 }
}
```

### v1 idea12_policy_network

```
{
 "best_5_days_three_year": {
  "2026-01-28": 132762,
  "2026-01-27": 74394,
  "2026-01-29": 66508,
  "2026-01-30": 33547,
  "2026-01-26": 22186
 },
 "placebo": {
  "shifted_one_day_total": 363769,
  "shifted_one_day_years": {
   "2024": 73410,
   "2025": 50918,
   "2026": 239441
  },
  "shuffled_mean_total": 84223,
  "shuffled_p95_total": 237303,
  "shuffled_max_total": 276278,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-06-23 earned 21,815 of the year's 39,271 (56 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2025,
   "flag": "best day 2025-06-23 earned 21,815 of the year's 39,271 (56 percent)",
   "best_day": "2025-06-23",
   "best_day_usd": 21815,
   "year_net_usd": 39271,
   "year_net_without_best_day": 17456,
   "mean_gap_that_day_usd_per_mwh": 84.1,
   "max_gap_that_day": 858.8,
   "always_supply_that_day_usd": -22270,
   "position_mix_that_day": {
    "load": 264,
    "supply": 0,
    "none": 0
   }
  }
 ],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.66,
  "sortino": 6.0,
  "return_over_max_drawdown": 3.38,
  "scale_for_100k_drawdown": 1.98,
  "return_on_500k_at_that_scale_pct": 67.5,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.76,
  "net_usd": 900687,
  "return_on_500k_annualised_pct": 65.5,
  "max_drawdown_usd": -88940,
  "days": 1004,
  "years": {
   "2024": 166260,
   "2025": 69117,
   "2026": 665310
  }
 },
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": 26583,
   "three_year_total_without_this_years_best_10": 443871
  },
  "2025": {
   "net_without_best_10": -41410,
   "three_year_total_without_this_years_best_10": 431073
  },
  "2026": {
   "net_without_best_10": -6946,
   "three_year_total_without_this_years_best_10": 126791
  }
 }
}
```

### v1 idea14_learned_allocator_candidates

```
{
 "best_5_days_three_year": {
  "2026-01-28": 111725,
  "2026-01-27": 62231,
  "2026-01-29": 57184,
  "2025-07-01": 50053,
  "2026-07-01": 36532
 },
 "placebo": {
  "shifted_one_day_total": 473239,
  "shifted_one_day_years": {
   "2024": 20152,
   "2025": 139916,
   "2026": 313171
  },
  "shuffled_mean_total": 522975,
  "shuffled_p95_total": 581663,
  "shuffled_max_total": 591609,
  "shuffled_share_at_or_above_real": 0.8,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2024,
   "why": "best day 2024-06-09 earned 14,757 of the year's 15,555 (95 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2024,
   "flag": "best day 2024-06-09 earned 14,757 of the year's 15,555 (95 percent)",
   "best_day": "2024-06-09",
   "best_day_usd": 14757,
   "year_net_usd": 15555,
   "year_net_without_best_day": 798,
   "mean_gap_that_day_usd_per_mwh": 4.4,
   "max_gap_that_day": 777.6,
   "always_supply_that_day_usd": -1194,
   "position_mix_that_day": {
    "load": 78,
    "supply": 179,
    "none": 7
   }
  }
 ],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.68,
  "sortino": 4.74,
  "return_over_max_drawdown": 3.28,
  "scale_for_100k_drawdown": 1.98,
  "return_on_500k_at_that_scale_pct": 65.5,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.55,
  "net_usd": 273505,
  "return_on_500k_annualised_pct": 19.9,
  "max_drawdown_usd": -27829,
  "days": 1004,
  "years": {
   "2024": 8555,
   "2025": 74984,
   "2026": 189966
  }
 },
 "random_days": 0.001,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -38454,
   "three_year_total_without_this_years_best_10": 443273
  },
  "2025": {
   "net_without_best_10": 4061,
   "three_year_total_without_this_years_best_10": 365010
  },
  "2026": {
   "net_without_best_10": -36740,
   "three_year_total_without_this_years_best_10": 115149
  }
 }
}
```

### v1 side_past_spike_rate_filter

```
{
 "best_5_days_three_year": {
  "2026-01-28": 119041,
  "2026-01-27": 63630,
  "2026-01-29": 58605,
  "2026-01-26": 33193,
  "2026-01-30": 24159
 },
 "placebo": {
  "shifted_one_day_total": 358694,
  "shifted_one_day_years": {
   "2024": 58791,
   "2025": 38765,
   "2026": 261138
  },
  "shuffled_mean_total": 358764,
  "shuffled_p95_total": 358764,
  "shuffled_max_total": 358821,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.18,
  "sortino": 2.68,
  "return_over_max_drawdown": 1.19,
  "scale_for_100k_drawdown": 0.99,
  "return_on_500k_at_that_scale_pct": 23.7,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.39,
  "net_usd": 139917,
  "return_on_500k_annualised_pct": 10.2,
  "max_drawdown_usd": -39286,
  "days": 1004,
  "years": {
   "2024": 22928,
   "2025": 15081,
   "2026": 101907
  }
 },
 "random_days": 0.01,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -29065,
   "three_year_total_without_this_years_best_10": 270905
  },
  "2025": {
   "net_without_best_10": -51723,
   "three_year_total_without_this_years_best_10": 268368
  },
  "2026": {
   "net_without_best_10": -74165,
   "three_year_total_without_this_years_best_10": 23296
  }
 }
}
```

### v1 A_regression_v1

```
{
 "best_5_days_three_year": {
  "2026-01-28": 118599,
  "2026-01-29": 62147,
  "2026-01-27": 57548,
  "2026-01-30": 27502,
  "2026-01-26": 21633
 },
 "placebo": {
  "shifted_one_day_total": 202610,
  "shifted_one_day_years": {
   "2024": 16974,
   "2025": 5504,
   "2026": 180133
  },
  "shuffled_mean_total": 192034,
  "shuffled_p95_total": 329233,
  "shuffled_max_total": 385831,
  "shuffled_share_at_or_above_real": 0.05,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 19,635 of the year's 33,440 (59 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2025,
   "flag": "best day 2025-01-20 earned 19,635 of the year's 33,440 (59 percent)",
   "best_day": "2025-01-20",
   "best_day_usd": 19635,
   "year_net_usd": 33440,
   "year_net_without_best_day": 13805,
   "mean_gap_that_day_usd_per_mwh": -74.6,
   "max_gap_that_day": 135.5,
   "always_supply_that_day_usd": 19635,
   "position_mix_that_day": {
    "load": 0,
    "supply": 264,
    "none": 0
   }
  }
 ],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.18,
  "sortino": 2.16,
  "return_over_max_drawdown": 1.17,
  "scale_for_100k_drawdown": 0.95,
  "return_on_500k_at_that_scale_pct": 23.4,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.71,
  "net_usd": 628174,
  "return_on_500k_annualised_pct": 45.7,
  "max_drawdown_usd": -179061,
  "days": 1004,
  "years": {
   "2024": 103493,
   "2025": 57183,
   "2026": 467499
  }
 },
 "random_days": 0.006,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": 7423,
   "three_year_total_without_this_years_best_10": 314254
  },
  "2025": {
   "net_without_best_10": -49249,
   "three_year_total_without_this_years_best_10": 284664
  },
  "2026": {
   "net_without_best_10": -48752,
   "three_year_total_without_this_years_best_10": 45210
  }
 }
}
```

### v1 A_regression_v1_deep

```
{
 "best_5_days_three_year": {
  "2026-01-28": 132991,
  "2026-01-27": 74171,
  "2026-01-29": 70487,
  "2026-01-26": 40480,
  "2026-01-30": 33025
 },
 "placebo": {
  "shifted_one_day_total": 417787,
  "shifted_one_day_years": {
   "2024": 69568,
   "2025": 50017,
   "2026": 298202
  },
  "shuffled_mean_total": 218386,
  "shuffled_p95_total": 304034,
  "shuffled_max_total": 331143,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 17,123 of the year's 4,244 (403 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2025,
   "flag": "best day 2025-01-20 earned 17,123 of the year's 4,244 (403 percent)",
   "best_day": "2025-01-20",
   "best_day_usd": 17123,
   "year_net_usd": 4244,
   "year_net_without_best_day": -12879,
   "mean_gap_that_day_usd_per_mwh": -74.6,
   "max_gap_that_day": 135.5,
   "always_supply_that_day_usd": 19635,
   "position_mix_that_day": {
    "load": 0,
    "supply": 227,
    "none": 37
   }
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.06,
  "sortino": 2.19,
  "return_over_max_drawdown": 1.21,
  "scale_for_100k_drawdown": 0.99,
  "return_on_500k_at_that_scale_pct": 24.1,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.65,
  "net_usd": 238334,
  "return_on_500k_annualised_pct": 17.3,
  "max_drawdown_usd": -65864,
  "days": 1004,
  "years": {
   "2024": 37646,
   "2025": 2759,
   "2026": 197929
  }
 },
 "random_days": 0.04,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -4476,
   "three_year_total_without_this_years_best_10": 304274
  },
  "2025": {
   "net_without_best_10": -77111,
   "three_year_total_without_this_years_best_10": 285312
  },
  "2026": {
   "net_without_best_10": -79316,
   "three_year_total_without_this_years_best_10": -17154
  }
 }
}
```

### v1 A_spike_deep_rolling_cut

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140584,
  "2026-01-27": 78292,
  "2026-01-29": 74080,
  "2026-01-26": 41144,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": 439048,
  "shifted_one_day_years": {
   "2024": 57925,
   "2025": 74798,
   "2026": 306325
  },
  "shuffled_mean_total": 351188,
  "shuffled_p95_total": 373736,
  "shuffled_max_total": 378095,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.45,
  "sortino": 3.98,
  "return_over_max_drawdown": 2.98,
  "scale_for_100k_drawdown": 1.76,
  "return_on_500k_at_that_scale_pct": 59.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.59,
  "net_usd": 299698,
  "return_on_500k_annualised_pct": 21.8,
  "max_drawdown_usd": -33533,
  "days": 1004,
  "years": {
   "2024": 40940,
   "2025": 49916,
   "2026": 208843
  }
 },
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -16864,
   "three_year_total_without_this_years_best_10": 421710
  },
  "2025": {
   "net_without_best_10": -9457,
   "three_year_total_without_this_years_best_10": 413902
  },
  "2026": {
   "net_without_best_10": -70719,
   "three_year_total_without_this_years_best_10": 83274
  }
 }
}
```

### v1 A_spike_gbm

```
{
 "best_5_days_three_year": {
  "2026-01-27": 46758,
  "2026-01-28": 33809,
  "2026-01-29": 32252,
  "2026-01-26": 18562,
  "2025-01-20": 17724
 },
 "placebo": {
  "shifted_one_day_total": 131610,
  "shifted_one_day_years": {
   "2024": 15291,
   "2025": 28233,
   "2026": 88086
  },
  "shuffled_mean_total": 312222,
  "shuffled_p95_total": 377832,
  "shuffled_max_total": 377894,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)",
   "best_day": "2026-01-27",
   "best_day_usd": 46758,
   "year_net_usd": 83179,
   "year_net_without_best_day": 36421,
   "mean_gap_that_day_usd_per_mwh": -298.4,
   "max_gap_that_day": -36.1,
   "always_supply_that_day_usd": 78703,
   "position_mix_that_day": {
    "load": 0,
    "supply": 157,
    "none": 107
   }
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.92,
  "sortino": 1.41,
  "return_over_max_drawdown": 0.91,
  "scale_for_100k_drawdown": 1.57,
  "return_on_500k_at_that_scale_pct": 18.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.66,
  "net_usd": 288491,
  "return_on_500k_annualised_pct": 21.0,
  "max_drawdown_usd": -105881,
  "days": 1004,
  "years": {
   "2024": 41372,
   "2025": 109042,
   "2026": 138077
  }
 },
 "random_days": 0.8,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -25647,
   "three_year_total_without_this_years_best_10": 123219
  },
  "2025": {
   "net_without_best_10": -16107,
   "three_year_total_without_this_years_best_10": 91994
  },
  "2026": {
   "net_without_best_10": -75091,
   "three_year_total_without_this_years_best_10": 15520
  }
 }
}
```

### v1 A_spike_tail_S100_gbm [second build]

```
{
 "best_5_days_three_year": {
  "2026-01-27": 46758,
  "2026-01-28": 33809,
  "2026-01-29": 32252,
  "2026-01-26": 18562,
  "2025-01-20": 17724
 },
 "placebo": {
  "shifted_one_day_total": 131595,
  "shifted_one_day_years": {
   "2024": 15276,
   "2025": 28233,
   "2026": 88086
  },
  "shuffled_mean_total": 312222,
  "shuffled_p95_total": 377832,
  "shuffled_max_total": 377894,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-27 earned 46,758 of the year's 83,179 (56 percent)",
   "best_day": "2026-01-27",
   "best_day_usd": 46758,
   "year_net_usd": 83179,
   "year_net_without_best_day": 36421,
   "mean_gap_that_day_usd_per_mwh": -298.4,
   "max_gap_that_day": -36.1,
   "always_supply_that_day_usd": 78703,
   "position_mix_that_day": {
    "load": 0,
    "supply": 157,
    "none": 107
   }
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.92,
  "sortino": 1.41,
  "return_over_max_drawdown": 0.91,
  "scale_for_100k_drawdown": 1.57,
  "return_on_500k_at_that_scale_pct": 18.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.64,
  "net_usd": 111225,
  "return_on_500k_annualised_pct": 8.1,
  "max_drawdown_usd": -40822,
  "days": 1004,
  "years": {
   "2024": 15951,
   "2025": 42040,
   "2026": 53234
  }
 },
 "random_days": 0.8,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -25647,
   "three_year_total_without_this_years_best_10": 123219
  },
  "2025": {
   "net_without_best_10": -16107,
   "three_year_total_without_this_years_best_10": 91994
  },
  "2026": {
   "net_without_best_10": -75091,
   "three_year_total_without_this_years_best_10": 15520
  }
 }
}
```

### v1 B_deep

```
{
 "best_5_days_three_year": {
  "2026-01-28": 136315,
  "2026-01-27": 77777,
  "2026-01-29": 71811,
  "2026-01-26": 40023,
  "2026-01-30": 33497
 },
 "placebo": {
  "shifted_one_day_total": 462644,
  "shifted_one_day_years": {
   "2024": 51522,
   "2025": 94185,
   "2026": 316937
  },
  "shuffled_mean_total": 86063,
  "shuffled_p95_total": 235599,
  "shuffled_max_total": 341897,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-07-01 earned 29,772 of the year's 39,707 (75 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2025,
   "flag": "best day 2025-07-01 earned 29,772 of the year's 39,707 (75 percent)",
   "best_day": "2025-07-01",
   "best_day_usd": 29772,
   "year_net_usd": 39707,
   "year_net_without_best_day": 9935,
   "mean_gap_that_day_usd_per_mwh": 122.5,
   "max_gap_that_day": 2502.4,
   "always_supply_that_day_usd": -32402,
   "position_mix_that_day": {
    "load": 119,
    "supply": 140,
    "none": 5
   }
  }
 ],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.3,
  "sortino": 4.03,
  "return_over_max_drawdown": 3.39,
  "scale_for_100k_drawdown": 2.35,
  "return_on_500k_at_that_scale_pct": 67.8,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.45,
  "net_usd": 627262,
  "return_on_500k_annualised_pct": 45.6,
  "max_drawdown_usd": -61637,
  "days": 1004,
  "years": {
   "2024": 64566,
   "2025": 57575,
   "2026": 505120
  }
 },
 "random_days": 0.027,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -9130,
   "three_year_total_without_this_years_best_10": 378935
  },
  "2025": {
   "net_without_best_10": -60667,
   "three_year_total_without_this_years_best_10": 332220
  },
  "2026": {
   "net_without_best_10": -74714,
   "three_year_total_without_this_years_best_10": 9522
  }
 }
}
```

### v1 B_gbm

```
{
 "best_5_days_three_year": {
  "2026-01-28": 138977,
  "2026-01-27": 78703,
  "2026-01-29": 74080,
  "2026-01-26": 38383,
  "2026-01-30": 34292
 },
 "placebo": {
  "shifted_one_day_total": 251325,
  "shifted_one_day_years": {
   "2024": -26868,
   "2025": 33156,
   "2026": 245037
  },
  "shuffled_mean_total": 76298,
  "shuffled_p95_total": 252226,
  "shuffled_max_total": 259888,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2024,
   "why": "best day 2024-07-08 earned 8,371 of the year's 14,771 (57 percent)"
  },
  {
   "year": 2025,
   "why": "best day 2025-07-01 earned 32,297 of the year's 24,143 (134 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2024,
   "flag": "best day 2024-07-08 earned 8,371 of the year's 14,771 (57 percent)",
   "best_day": "2024-07-08",
   "best_day_usd": 8371,
   "year_net_usd": 14771,
   "year_net_without_best_day": 6400,
   "mean_gap_that_day_usd_per_mwh": 35.7,
   "max_gap_that_day": 463.2,
   "always_supply_that_day_usd": -9473,
   "position_mix_that_day": {
    "load": 207,
    "supply": 34,
    "none": 23
   }
  },
  {
   "year": 2025,
   "flag": "best day 2025-07-01 earned 32,297 of the year's 24,143 (134 percent)",
   "best_day": "2025-07-01",
   "best_day_usd": 32297,
   "year_net_usd": 24143,
   "year_net_without_best_day": -8154,
   "mean_gap_that_day_usd_per_mwh": 122.5,
   "max_gap_that_day": 2502.4,
   "always_supply_that_day_usd": -32402,
   "position_mix_that_day": {
    "load": 264,
    "supply": 0,
    "none": 0
   }
  }
 ],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.29,
  "sortino": 2.9,
  "return_over_max_drawdown": 2.01,
  "scale_for_100k_drawdown": 1.29,
  "return_on_500k_at_that_scale_pct": 40.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 2.3,
  "net_usd": 1073069,
  "return_on_500k_annualised_pct": 78.0,
  "max_drawdown_usd": -177771,
  "days": 1004,
  "years": {
   "2024": 33972,
   "2025": 55529,
   "2026": 983568
  }
 },
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -30696,
   "three_year_total_without_this_years_best_10": 421085
  },
  "2025": {
   "net_without_best_10": -112043,
   "three_year_total_without_this_years_best_10": 330366
  },
  "2026": {
   "net_without_best_10": -3228,
   "three_year_total_without_this_years_best_10": 35686
  }
 }
}
```

### v1 D_deep

```
{
 "best_5_days_three_year": {
  "2026-01-28": 127086,
  "2026-01-27": 70526,
  "2026-01-29": 67817,
  "2026-01-26": 40615,
  "2025-07-01": 32186
 },
 "placebo": {
  "shifted_one_day_total": 379505,
  "shifted_one_day_years": {
   "2024": 65999,
   "2025": 9469,
   "2026": 304037
  },
  "shuffled_mean_total": 84351,
  "shuffled_p95_total": 230354,
  "shuffled_max_total": 236009,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.98,
  "sortino": 2.06,
  "return_over_max_drawdown": 0.99,
  "scale_for_100k_drawdown": 0.9,
  "return_on_500k_at_that_scale_pct": 19.9,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.97,
  "net_usd": 321597,
  "return_on_500k_annualised_pct": 23.4,
  "max_drawdown_usd": -107795,
  "days": 1004,
  "years": {
   "2024": 68199,
   "2025": -11806,
   "2026": 265204
  }
 },
 "random_days": 0.514,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": 8979,
   "three_year_total_without_this_years_best_10": 270214
  },
  "2025": {
   "net_without_best_10": -118813,
   "three_year_total_without_this_years_best_10": 224901
  },
  "2026": {
   "net_without_best_10": -122819,
   "three_year_total_without_this_years_best_10": -64682
  }
 }
}
```

### v1 D_gbm

```
{
 "best_5_days_three_year": {
  "2026-01-28": 140239,
  "2026-01-27": 77924,
  "2026-01-29": 74080,
  "2026-01-26": 41013,
  "2026-01-30": 34621
 },
 "placebo": {
  "shifted_one_day_total": 361279,
  "shifted_one_day_years": {
   "2024": -14103,
   "2025": 124626,
   "2026": 250755
  },
  "shuffled_mean_total": 46970,
  "shuffled_p95_total": 306876,
  "shuffled_max_total": 312470,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.41,
  "sortino": 3.17,
  "return_over_max_drawdown": 1.44,
  "scale_for_100k_drawdown": 0.85,
  "return_on_500k_at_that_scale_pct": 28.8,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 2.08,
  "net_usd": 1057256,
  "return_on_500k_annualised_pct": 76.9,
  "max_drawdown_usd": -244766,
  "days": 1004,
  "years": {
   "2024": 114953,
   "2025": 193290,
   "2026": 749013
  }
 },
 "random_days": 0.001,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -1310,
   "three_year_total_without_this_years_best_10": 451721
  },
  "2025": {
   "net_without_best_10": -31119,
   "three_year_total_without_this_years_best_10": 384249
  },
  "2026": {
   "net_without_best_10": -60844,
   "three_year_total_without_this_years_best_10": 87350
  }
 }
}
```

### v1 risk_sized_supply_gbm

```
{
 "best_5_days_three_year": {
  "2026-01-28": 39092,
  "2026-01-27": 33639,
  "2026-01-29": 25154,
  "2026-01-26": 13276,
  "2025-01-20": 8438
 },
 "placebo": {
  "shifted_one_day_total": 132304,
  "shifted_one_day_years": {
   "2024": 9476,
   "2025": 29736,
   "2026": 93092
  },
  "shuffled_mean_total": 206486,
  "shuffled_p95_total": 252258,
  "shuffled_max_total": 276871,
  "shuffled_share_at_or_above_real": 0.95,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.24,
  "sortino": 2.57,
  "return_over_max_drawdown": 1.73,
  "scale_for_100k_drawdown": 3.46,
  "return_on_500k_at_that_scale_pct": 34.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 2.34,
  "net_usd": 351588,
  "return_on_500k_annualised_pct": 25.6,
  "max_drawdown_usd": -67727,
  "days": 1004,
  "years": {
   "2024": 38889,
   "2025": 94145,
   "2026": 218554
  }
 },
 "random_days": 0.928,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -4678,
   "three_year_total_without_this_years_best_10": 128954
  },
  "2025": {
   "net_without_best_10": -1927,
   "three_year_total_without_this_years_best_10": 108092
  },
  "2026": {
   "net_without_best_10": -39202,
   "three_year_total_without_this_years_best_10": 17650
  }
 }
}
```

### v1 storm_day_filter [second build]

```
{
 "best_5_days_three_year": {
  "2026-01-26": 41262,
  "2024-01-17": 15323,
  "2025-01-19": 14635,
  "2024-01-16": 12917,
  "2025-08-12": 11369
 },
 "placebo": {
  "shifted_one_day_total": 51553,
  "shifted_one_day_years": {
   "2024": 19109,
   "2025": 73017,
   "2026": -40573
  },
  "shuffled_mean_total": 324184,
  "shuffled_p95_total": 398044,
  "shuffled_max_total": 499045,
  "shuffled_share_at_or_above_real": 1.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-26 earned 41,262 of the year's 30,777 (134 percent)"
  }
 ],
 "too_good_checks": [
  {
   "year": 2026,
   "flag": "best day 2026-01-26 earned 41,262 of the year's 30,777 (134 percent)",
   "best_day": "2026-01-26",
   "best_day_usd": 41262,
   "year_net_usd": 30777,
   "year_net_without_best_day": -10485,
   "mean_gap_that_day_usd_per_mwh": -156.5,
   "max_gap_that_day": 73.9,
   "always_supply_that_day_usd": 41262,
   "position_mix_that_day": {
    "load": 0,
    "supply": 264,
    "none": 0
   },
   "storm_that_day": 0.9507,
   "storm_day_before": 0.9968
  }
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.05,
  "sortino": 1.52,
  "return_over_max_drawdown": 0.8,
  "scale_for_100k_drawdown": 1.41,
  "return_on_500k_at_that_scale_pct": 16.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 1.0,
  "net_usd": 170220,
  "return_on_500k_annualised_pct": 12.4,
  "max_drawdown_usd": -70900,
  "days": 1004,
  "years": {
   "2024": 55504,
   "2025": 83939,
   "2026": 30777
  }
 },
 "random_days": 0.863,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -23542,
   "three_year_total_without_this_years_best_10": 91174
  },
  "2025": {
   "net_without_best_10": 3248,
   "three_year_total_without_this_years_best_10": 89529
  },
  "2026": {
   "net_without_best_10": -70692,
   "three_year_total_without_this_years_best_10": 68752
  }
 }
}
```

### v1 zone_subset_supply

```
{
 "best_5_days_three_year": {
  "2026-01-28": 71158,
  "2026-01-27": 40417,
  "2026-01-29": 38512,
  "2026-01-26": 20955,
  "2026-01-30": 18891
 },
 "placebo": {
  "shifted_one_day_total": 204811,
  "shifted_one_day_years": {
   "2024": 39072,
   "2025": -2806,
   "2026": 168545
  },
  "shuffled_mean_total": 204843,
  "shuffled_p95_total": 204990,
  "shuffled_max_total": 205032,
  "shuffled_share_at_or_above_real": 0.1,
  "n_shuffles": 20
 },
 "too_good": [],
 "too_good_checks": [],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.02,
  "sortino": 1.88,
  "return_over_max_drawdown": 0.82,
  "scale_for_100k_drawdown": 1.2,
  "return_on_500k_at_that_scale_pct": 16.4,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.61,
  "net_usd": 125029,
  "return_on_500k_annualised_pct": 9.1,
  "max_drawdown_usd": -50958,
  "days": 1004,
  "years": {
   "2024": 23853,
   "2025": -1650,
   "2026": 102827
  }
 },
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -9032,
   "three_year_total_without_this_years_best_10": 156831
  },
  "2025": {
   "net_without_best_10": -55668,
   "three_year_total_without_this_years_best_10": 152003
  },
  "2026": {
   "net_without_best_10": -51219,
   "three_year_total_without_this_years_best_10": -14823
  }
 }
}
```

### v1 idea13_portfolio

```
{
 "best_5_days_three_year": {
  "2026-01-28": 152213,
  "2026-01-27": 84365,
  "2026-01-29": 76620,
  "2025-07-01": 45754,
  "2026-01-26": 37900
 },
 "too_good": [],
 "sizing_view": {
  "sharpe": 1.7,
  "sortino": 4.35,
  "return_over_max_drawdown": 3.17,
  "scale_for_100k_drawdown": 1.47,
  "return_on_500k_at_that_scale_pct": 63.3,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale_mw": 0.594,
  "net_usd": 383016,
  "return_on_500k_annualised_pct": 27.8,
  "max_drawdown_usd": -40329,
  "days": 1004,
  "years": {
   "2024": 36444,
   "2025": 87212,
   "2026": 259360
  }
 },
 "random_days": 0.0,
 "without_best_10_days": null
}
```

### v2 V13_A_deep_all

```
{
 "total_without_best_5_days": 22716,
 "placebos": {
  "shifted_one_day_total": 279099,
  "shifted_one_day_years": {
   "2024": 44633,
   "2025": 6427,
   "2026": 228039
  },
  "shuffled_mean_total": 353586,
  "shuffled_p95_total": 398194,
  "shuffled_max_total": 403026,
  "shuffled_share_at_or_above_real": 0.8,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 18,331 of the year's 28,587 (64 percent)"
  }
 ],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days",
  "random days in the same months beat the rule more than 20 percent of the time"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.05,
  "sortino": 2.35,
  "return_over_max_drawdown": 1.42,
  "scale_for_100k_drawdown": 1.27,
  "return_on_500k_at_that_scale_pct": 28.4,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 0.59,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 197845,
  "max_drawdown_usd": -46514,
  "return_on_500k_a_year_pct": 14.4,
  "years": {
   "2024": 31116,
   "2025": 16867,
   "2026": 149863
  },
  "days": 1004
 },
 "stress_0.50_total": 259376,
 "random_days": 0.436,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -27191,
   "three_year_total_without_this_years_best_10": 255401
  },
  "2025": {
   "net_without_best_10": -57551,
   "three_year_total_without_this_years_best_10": 249192
  },
  "2026": {
   "net_without_best_10": -113292,
   "three_year_total_without_this_years_best_10": -31967
  }
 }
}
```

### v2 V15a_V5_limit

```
{
 "total_without_best_5_days": 99124,
 "placebos": {
  "shifted_one_day_total": 187117,
  "shifted_one_day_years": {
   "2024": -47400,
   "2025": -16082,
   "2026": 250599
  },
  "shuffled_mean_total": -39743,
  "shuffled_p95_total": 257245,
  "shuffled_max_total": 283674,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.23,
  "sortino": 2.53,
  "return_over_max_drawdown": 1.01,
  "scale_for_100k_drawdown": 0.73,
  "return_on_500k_at_that_scale_pct": 20.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 2.51,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 1042089,
  "max_drawdown_usd": -343765,
  "return_on_500k_a_year_pct": 75.8,
  "years": {
   "2024": 104596,
   "2025": 154884,
   "2026": 782610
  },
  "days": 1004
 },
 "stress_0.50_total": 352598,
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -14082,
   "three_year_total_without_this_years_best_10": 359421
  },
  "2025": {
   "net_without_best_10": -37906,
   "three_year_total_without_this_years_best_10": 315563
  },
  "2026": {
   "net_without_best_10": -44592,
   "three_year_total_without_this_years_best_10": 58786
  }
 }
}
```

### v2 V15a_V8_limit

```
{
 "total_without_best_5_days": 6690,
 "placebos": {
  "shifted_one_day_total": 164253,
  "shifted_one_day_years": {
   "2024": -24005,
   "2025": -7455,
   "2026": 195713
  },
  "shuffled_mean_total": 7569,
  "shuffled_p95_total": 298998,
  "shuffled_max_total": 306937,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-07-01 earned 21,877 of the year's 23,076 (95 percent)"
  }
 ],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.97,
  "sortino": 1.95,
  "return_over_max_drawdown": 0.77,
  "scale_for_100k_drawdown": 0.69,
  "return_on_500k_at_that_scale_pct": 15.5,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 2.17,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 725762,
  "max_drawdown_usd": -312782,
  "return_on_500k_a_year_pct": 52.8,
  "years": {
   "2024": 36417,
   "2025": 50074,
   "2026": 639271
  },
  "days": 1004
 },
 "stress_0.50_total": 270078,
 "random_days": 0.036,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -20013,
   "three_year_total_without_this_years_best_10": 297658
  },
  "2025": {
   "net_without_best_10": -90587,
   "three_year_total_without_this_years_best_10": 220790
  },
  "2026": {
   "net_without_best_10": -75343,
   "three_year_total_without_this_years_best_10": -35485
  }
 }
}
```

### v2 V15b_V13_both

```
{
 "total_without_best_5_days": 129183,
 "placebos": {
  "shifted_one_day_total": 274402,
  "shifted_one_day_years": {
   "2024": 24885,
   "2025": 54327,
   "2026": 195190
  },
  "shuffled_mean_total": 108183,
  "shuffled_p95_total": 260083,
  "shuffled_max_total": 311006,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.49,
  "sortino": 3.34,
  "return_over_max_drawdown": 2.24,
  "scale_for_100k_drawdown": 1.72,
  "return_on_500k_at_that_scale_pct": 44.9,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.52,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 593126,
  "max_drawdown_usd": -88141,
  "return_on_500k_a_year_pct": 43.1,
  "years": {
   "2024": 82500,
   "2025": 163255,
   "2026": 347371
  },
  "days": 1004
 },
 "stress_0.50_total": 317030,
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -6962,
   "three_year_total_without_this_years_best_10": 328976
  },
  "2025": {
   "net_without_best_10": 16893,
   "three_year_total_without_this_years_best_10": 299703
  },
  "2026": {
   "net_without_best_10": -84304,
   "three_year_total_without_this_years_best_10": 77377
  }
 }
}
```

### v2 V15b_V13_limit

```
{
 "total_without_best_5_days": 9780,
 "placebos": {
  "shifted_one_day_total": 198296,
  "shifted_one_day_years": {
   "2024": 845,
   "2025": -37322,
   "2026": 234773
  },
  "shuffled_mean_total": 84714,
  "shuffled_p95_total": 224646,
  "shuffled_max_total": 339672,
  "shuffled_share_at_or_above_real": 0.05,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.06,
  "sortino": 2.47,
  "return_over_max_drawdown": 1.5,
  "scale_for_100k_drawdown": 1.48,
  "return_on_500k_at_that_scale_pct": 30.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.25,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 381190,
  "max_drawdown_usd": -84580,
  "return_on_500k_a_year_pct": 27.7,
  "years": {
   "2024": 22732,
   "2025": 60158,
   "2026": 298300
  },
  "days": 1004
 },
 "stress_0.50_total": 246538,
 "random_days": 0.004,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -12553,
   "three_year_total_without_this_years_best_10": 274213
  },
  "2025": {
   "net_without_best_10": -28891,
   "three_year_total_without_this_years_best_10": 227935
  },
  "2026": {
   "net_without_best_10": -95886,
   "three_year_total_without_this_years_best_10": -29574
  }
 }
}
```

### v2 V15c_wx_border_gbm

```
{
 "total_without_best_5_days": 101483,
 "placebos": {
  "shifted_one_day_total": 209784,
  "shifted_one_day_years": {
   "2024": 4880,
   "2025": 3113,
   "2026": 201791
  },
  "shuffled_mean_total": -50984,
  "shuffled_p95_total": 302081,
  "shuffled_max_total": 323007,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-07-01 earned 19,558 of the year's 30,145 (65 percent)"
  }
 ],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.24,
  "sortino": 2.55,
  "return_over_max_drawdown": 1.31,
  "scale_for_100k_drawdown": 0.91,
  "return_on_500k_at_that_scale_pct": 26.2,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 2.01,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 871757,
  "max_drawdown_usd": -222092,
  "return_on_500k_a_year_pct": 63.4,
  "years": {
   "2024": 171082,
   "2025": 60591,
   "2026": 640084
  },
  "days": 1004
 },
 "stress_0.50_total": 359441,
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": 11357,
   "three_year_total_without_this_years_best_10": 359951
  },
  "2025": {
   "net_without_best_10": -69144,
   "three_year_total_without_this_years_best_10": 334421
  },
  "2026": {
   "net_without_best_10": -78923,
   "three_year_total_without_this_years_best_10": 36337
  }
 }
}
```

### v2 V16_B_gefs_joined

```
{
 "total_without_best_5_days": 77193,
 "placebos": {
  "shifted_one_day_total": 129846,
  "shifted_one_day_years": {
   "2024": -27147,
   "2025": -21627,
   "2026": 178620
  },
  "shuffled_mean_total": -15337,
  "shuffled_p95_total": 354540,
  "shuffled_max_total": 382097,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 16,295 of the year's 28,070 (58 percent)"
  }
 ],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.17,
  "sortino": 2.33,
  "return_over_max_drawdown": 1.15,
  "scale_for_100k_drawdown": 0.84,
  "return_on_500k_at_that_scale_pct": 23.1,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.54,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 632863,
  "max_drawdown_usd": -182786,
  "return_on_500k_a_year_pct": 46.0,
  "years": {
   "2024": 56256,
   "2025": 43228,
   "2026": 533379
  },
  "days": 1004
 },
 "stress_0.50_total": 336320,
 "random_days": 0.014,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -19147,
   "three_year_total_without_this_years_best_10": 355274
  },
  "2025": {
   "net_without_best_10": -67379,
   "three_year_total_without_this_years_best_10": 315501
  },
  "2026": {
   "net_without_best_10": -36256,
   "three_year_total_without_this_years_best_10": 28344
  }
 }
}
```

### v2 V1_always_supply

```
{
 "total_without_best_5_days": -22053,
 "placebos": {
  "note": "not applicable"
 },
 "too_good": [],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.89,
  "sortino": 1.66,
  "return_over_max_drawdown": 0.73,
  "scale_for_100k_drawdown": 0.63,
  "return_on_500k_at_that_scale_pct": 14.5,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 0.27,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 93743,
  "max_drawdown_usd": -43010,
  "return_on_500k_a_year_pct": 6.8,
  "years": {
   "2024": 15833,
   "2025": -3972,
   "2026": 81882
  },
  "days": 1004
 },
 "stress_0.50_total": 268950,
 "random_days": 0.0,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -30193,
   "three_year_total_without_this_years_best_10": 258363
  },
  "2025": {
   "net_without_best_10": -120435,
   "three_year_total_without_this_years_best_10": 241471
  },
  "2026": {
   "net_without_best_10": -128648,
   "three_year_total_without_this_years_best_10": -84717
  }
 }
}
```

### v2 V2_limit_gbm

```
{
 "total_without_best_5_days": 28835,
 "placebos": {
  "shifted_one_day_total": 238129,
  "shifted_one_day_years": {
   "2024": -9861,
   "2025": -15606,
   "2026": 263596
  },
  "shuffled_mean_total": -4541,
  "shuffled_p95_total": 340560,
  "shuffled_max_total": 407700,
  "shuffled_share_at_or_above_real": 0.05,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 19,163 of the year's 19,491 (98 percent)"
  }
 ],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.09,
  "sortino": 2.24,
  "return_over_max_drawdown": 0.85,
  "scale_for_100k_drawdown": 0.66,
  "return_on_500k_at_that_scale_pct": 17.1,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.77,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 691416,
  "max_drawdown_usd": -270020,
  "return_on_500k_a_year_pct": 50.3,
  "years": {
   "2024": 47724,
   "2025": 34498,
   "2026": 609193
  },
  "days": 1004
 },
 "stress_0.50_total": 330431,
 "random_days": 0.012,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -11335,
   "three_year_total_without_this_years_best_10": 352332
  },
  "2025": {
   "net_without_best_10": -74440,
   "three_year_total_without_this_years_best_10": 296699
  },
  "2026": {
   "net_without_best_10": -65031,
   "three_year_total_without_this_years_best_10": -18577
  }
 }
}
```

### v2 V2_sides_no_limit

```
{
 "total_without_best_5_days": 94633,
 "placebos": {
  "shifted_one_day_total": 320542,
  "shifted_one_day_years": {
   "2024": 3124,
   "2025": 59501,
   "2026": 257917
  },
  "shuffled_mean_total": 20722,
  "shuffled_p95_total": 347894,
  "shuffled_max_total": 472650,
  "shuffled_share_at_or_above_real": 0.05,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.26,
  "sortino": 2.61,
  "return_over_max_drawdown": 1.18,
  "scale_for_100k_drawdown": 0.77,
  "return_on_500k_at_that_scale_pct": 23.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.62,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 747986,
  "max_drawdown_usd": -211301,
  "return_on_500k_a_year_pct": 54.4,
  "years": {
   "2024": 73189,
   "2025": 79688,
   "2026": 595109
  },
  "days": 1004
 },
 "stress_0.50_total": 384838,
 "random_days": 0.001,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -10442,
   "three_year_total_without_this_years_best_10": 406099
  },
  "2025": {
   "net_without_best_10": -56391,
   "three_year_total_without_this_years_best_10": 356139
  },
  "2026": {
   "net_without_best_10": -57399,
   "three_year_total_without_this_years_best_10": 36969
  }
 }
}
```

### v2 V3_tail_gbm

```
{
 "total_without_best_5_days": 267545,
 "placebos": {
  "shifted_one_day_total": 623751,
  "shifted_one_day_years": {
   "2024": -23071,
   "2025": 181096,
   "2026": 465726
  },
  "shuffled_mean_total": -16159,
  "shuffled_p95_total": 982401,
  "shuffled_max_total": 1015022,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.21,
  "sortino": 2.43,
  "return_over_max_drawdown": 1.33,
  "scale_for_100k_drawdown": 0.32,
  "return_on_500k_at_that_scale_pct": 26.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 0.53,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 667125,
  "max_drawdown_usd": -167171,
  "return_on_500k_a_year_pct": 48.5,
  "years": {
   "2024": 63632,
   "2025": 99032,
   "2026": 504460
  },
  "days": 1004
 },
 "stress_0.50_total": 1095128,
 "random_days": 0.006,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -24300,
   "three_year_total_without_this_years_best_10": 1114366
  },
  "2025": {
   "net_without_best_10": -101778,
   "three_year_total_without_this_years_best_10": 970095
  },
  "2026": {
   "net_without_best_10": -191139,
   "three_year_total_without_this_years_best_10": 115776
  }
 }
}
```

### v2 V3_tail_gbm_1MW

```
{
 "total_without_best_5_days": 94572,
 "placebos": {
  "shifted_one_day_total": 195576,
  "shifted_one_day_years": {
   "2024": -17543,
   "2025": 57203,
   "2026": 155916
  },
  "shuffled_mean_total": -17264,
  "shuffled_p95_total": 352369,
  "shuffled_max_total": 371326,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.22,
  "sortino": 2.45,
  "return_over_max_drawdown": 1.33,
  "scale_for_100k_drawdown": 0.92,
  "return_on_500k_at_that_scale_pct": 26.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.55,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 668966,
  "max_drawdown_usd": -167733,
  "return_on_500k_a_year_pct": 48.6,
  "years": {
   "2024": 55773,
   "2025": 111751,
   "2026": 501442
  },
  "days": 1004
 },
 "stress_0.50_total": 358908,
 "random_days": 0.007,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -18867,
   "three_year_total_without_this_years_best_10": 376741
  },
  "2025": {
   "net_without_best_10": -27845,
   "three_year_total_without_this_years_best_10": 331649
  },
  "2026": {
   "net_without_best_10": -62876,
   "three_year_total_without_this_years_best_10": 45204
  }
 }
}
```

### v2 V5_border_inputs

```
{
 "total_without_best_5_days": 63836,
 "placebos": {
  "shifted_one_day_total": 56955,
  "shifted_one_day_years": {
   "2024": -24334,
   "2025": 23896,
   "2026": 57394
  },
  "shuffled_mean_total": -46610,
  "shuffled_p95_total": 314856,
  "shuffled_max_total": 351288,
  "shuffled_share_at_or_above_real": 0.1,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2026,
   "why": "best day 2026-01-28 earned 97,577 of the year's 134,878 (72 percent)"
  }
 ],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.05,
  "sortino": 1.72,
  "return_over_max_drawdown": 0.99,
  "scale_for_100k_drawdown": 1.02,
  "return_on_500k_at_that_scale_pct": 19.9,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.52,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 445493,
  "max_drawdown_usd": -149600,
  "return_on_500k_a_year_pct": 32.4,
  "years": {
   "2024": 93311,
   "2025": 147167,
   "2026": 205015
  },
  "days": 1004
 },
 "stress_0.50_total": 218377,
 "random_days": 0.009,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -3883,
   "three_year_total_without_this_years_best_10": 227816
  },
  "2025": {
   "net_without_best_10": -19115,
   "three_year_total_without_this_years_best_10": 177152
  },
  "2026": {
   "net_without_best_10": -126006,
   "three_year_total_without_this_years_best_10": 32203
  }
 }
}
```

### v2 V7_flags

```
{
 "total_without_best_5_days": 78444,
 "placebos": {
  "shifted_one_day_total": 312409,
  "shifted_one_day_years": {
   "2024": 67134,
   "2025": 86930,
   "2026": 158345
  },
  "shuffled_mean_total": -161062,
  "shuffled_p95_total": 200044,
  "shuffled_max_total": 250259,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [],
 "FRAGILE": false,
 "sizing_view": {
  "sharpe": 1.15,
  "sortino": 2.39,
  "return_over_max_drawdown": 1.1,
  "scale_for_100k_drawdown": 0.83,
  "return_on_500k_at_that_scale_pct": 22.0,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.22,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 487953,
  "max_drawdown_usd": -147583,
  "return_on_500k_a_year_pct": 35.5,
  "years": {
   "2024": 78125,
   "2025": -13022,
   "2026": 422850
  },
  "days": 1004
 },
 "stress_0.50_total": 320883,
 "random_days": 0.007,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -16859,
   "three_year_total_without_this_years_best_10": 319066
  },
  "2025": {
   "net_without_best_10": -153811,
   "three_year_total_without_this_years_best_10": 256825
  },
  "2026": {
   "net_without_best_10": -27898,
   "three_year_total_without_this_years_best_10": 25465
  }
 }
}
```

### v2 V7_recency

```
{
 "total_without_best_5_days": -17503,
 "placebos": {
  "shifted_one_day_total": 363046,
  "shifted_one_day_years": {
   "2024": 53006,
   "2025": 92267,
   "2026": 217773
  },
  "shuffled_mean_total": -149851,
  "shuffled_p95_total": 171928,
  "shuffled_max_total": 249568,
  "shuffled_share_at_or_above_real": 0.0,
  "n_shuffles": 20
 },
 "too_good": [],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.91,
  "sortino": 1.85,
  "return_over_max_drawdown": 0.73,
  "scale_for_100k_drawdown": 0.7,
  "return_on_500k_at_that_scale_pct": 14.7,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.22,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 383633,
  "max_drawdown_usd": -174217,
  "return_on_500k_a_year_pct": 27.9,
  "years": {
   "2024": 53261,
   "2025": -54925,
   "2026": 385296
  },
  "days": 1004
 },
 "stress_0.50_total": 235171,
 "random_days": 0.035,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -36049,
   "three_year_total_without_this_years_best_10": 234747
  },
  "2025": {
   "net_without_best_10": -186141,
   "three_year_total_without_this_years_best_10": 173333
  },
  "2026": {
   "net_without_best_10": -46061,
   "three_year_total_without_this_years_best_10": -47424
  }
 }
}
```

### v2 V8_error_mining

```
{
 "total_without_best_5_days": 29400,
 "placebos": {
  "shifted_one_day_total": 128278,
  "shifted_one_day_years": {
   "2024": -189,
   "2025": 60802,
   "2026": 67665
  },
  "shuffled_mean_total": 26969,
  "shuffled_p95_total": 365293,
  "shuffled_max_total": 366183,
  "shuffled_share_at_or_above_real": 0.1,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2025,
   "why": "best day 2025-01-20 earned 18,673 of the year's 14,892 (125 percent)"
  }
 ],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 1.0,
  "sortino": 1.91,
  "return_over_max_drawdown": 0.78,
  "scale_for_100k_drawdown": 0.68,
  "return_on_500k_at_that_scale_pct": 15.6,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 2.26,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 773094,
  "max_drawdown_usd": -330093,
  "return_on_500k_a_year_pct": 56.2,
  "years": {
   "2024": 79378,
   "2025": 33656,
   "2026": 660060
  },
  "days": 1004
 },
 "stress_0.50_total": 265866,
 "random_days": 0.052,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -22893,
   "three_year_total_without_this_years_best_10": 284061
  },
  "2025": {
   "net_without_best_10": -87276,
   "three_year_total_without_this_years_best_10": 239909
  },
  "2026": {
   "net_without_best_10": -67585,
   "three_year_total_without_this_years_best_10": -17570
  }
 }
}
```

### v2 V9_decompose

```
{
 "total_without_best_5_days": 24027,
 "placebos": {
  "shifted_one_day_total": 49557,
  "shifted_one_day_years": {
   "2024": -29010,
   "2025": 89610,
   "2026": -11044
  },
  "shuffled_mean_total": -38615,
  "shuffled_p95_total": 334041,
  "shuffled_max_total": 370322,
  "shuffled_share_at_or_above_real": 0.1,
  "n_shuffles": 20
 },
 "too_good": [
  {
   "year": 2024,
   "why": "best day 2024-01-17 earned 9,142 of the year's 17,076 (54 percent)"
  },
  {
   "year": 2026,
   "why": "best day 2026-01-28 earned 115,411 of the year's 203,628 (57 percent)"
  }
 ],
 "fragile_reasons": [
  "three-year total turns negative without one year's best 10 days"
 ],
 "FRAGILE": true,
 "sizing_view": {
  "sharpe": 0.98,
  "sortino": 1.94,
  "return_over_max_drawdown": 1.14,
  "scale_for_100k_drawdown": 1.18,
  "return_on_500k_at_that_scale_pct": 22.9,
  "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale linearly with the book; scale = MW per position, capped at 5"
 },
 "sizing_view_carried": {
  "scale": 1.82,
  "source": "/home/bo/nyiso-us/results/v2/v2_results.json",
  "net_usd": 531449,
  "max_drawdown_usd": -154737,
  "return_on_500k_a_year_pct": 38.6,
  "years": {
   "2024": 31078,
   "2025": 129767,
   "2026": 370603
  },
  "days": 1004
 },
 "stress_0.50_total": 216712,
 "random_days": 0.073,
 "without_best_10_days": {
  "2024": {
   "net_without_best_10": -38469,
   "three_year_total_without_this_years_best_10": 236460
  },
  "2025": {
   "net_without_best_10": -49917,
   "three_year_total_without_this_years_best_10": 170787
  },
  "2026": {
   "net_without_best_10": -103590,
   "three_year_total_without_this_years_best_10": -15213
  }
 }
}
```

