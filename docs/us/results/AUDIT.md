# Audit of the held-out results, 2024-01-01 to 2026-09-30

Written 11 Oct 2026, before anything is published. Read only: no commit, no page edit, no frozen file changed. Scripts and outputs are in `heldout_results/audit_scratch/` (laptop) and `~/audit_ho_20261011/` (gene).

## Bottom line

No held-out row shows model skill that can be told apart from the two no-model rules. The supply-leaning rows earned most of their money on 18 days, the January 2026 winter storm and the late-June heat wave, which always taking supply earned too. Without those days, none of them beats always supply at the same size by a margin the bootstrap can distinguish from zero. The pair rows (the C family) earn steadily and hardly depend on the events, but no C model beats the static pairs that use no model, and two do significantly worse. Nothing in the data is broken. No feature reads a real-time price that was not public at 05:00, and the rebuilt July prices agree with NYISO's official generator prices to the cent. Three things in the run itself are broken: V12_alloc ran although the 9 Oct addendum dropped it, the summary and the chart data both crashed, and the run script still logged ALL DONE.

The registered verdict run (ho_reh) agrees: ideas A, B, C and D are each "inconclusive", and idea A minus always supply is -292 USD a day, 95% interval -1,044 to +270.

## How each table reads the bar

- **Registered bar, sizing view** (the rule of 7 Oct, used by both HELDOUT-LIST.md files). The return a year on 500,000 USD at the build-year scale (MW per position at which the build-year worst drawdown is 100,000, capped at 5, carried unchanged) must be at least 10%. At least 2 of the 3 years must be positive, Sharpe must be above 0.42 and the total must stay positive at the 0.50 USD/MWh stress. This audit selects rows on this reading.
- **1 MW reading** (the scorers' PASS column in v2_results_heldout.md and lab_results_heldout.md): average net of at least 50,000 USD a year at 1 MW. The two readings disagree on 20 of the 69 rows. Every C-family row fails at 1 MW but passes at the sizing view, and always supply passes at 1 MW (126,000 a year) but fails at the sizing view (7.3%). The published table must say which bar a "yes" means.
- Tables 2 to 4 below are at **1 MW per position**, the registered size, so rows compare like for like. Table 1 gives both readings.

Candidates that pass at the sizing view on the window (28): v1 A_regression_v1, B_gbm, B_deep, C_deep, idea13_portfolio, idea14_learned_allocator_candidates (C_gbm fails: negative at the stress). v2 V11_C_deep_ens4, V11_C_gbm_ens4, V13_C_deep_all, V14c, V14d, V14e, V14f, V15a_V5_limit, V15a_V8_limit, V15b_V13_both, V15b_V13_limit, V15c_wx_border_gbm, V16_B_gefs_joined, V1_C_deep, V1_C_gbm, V2_limit_gbm, V3_tail_gbm, V5_border_inputs, V7_flags, V7_recency, V8_error_mining, V9_decompose. These v2 candidates fail: V14a (9.0%), V14b (3.7%), V6_LI (4.3%), V12_alloc (6.5%; it should not have run, see below). Comparisons: always_supply, the habit rule (baseline_usual_side), C_static_pairs.

## Checks that the numbers are the scorers' numbers

From positions and prices, I recomputed every audited row's daily P&L on gene. All 34 match the scorers' daily tables (v2_daily_heldout.parquet, lab_daily_heldout.parquet) to 0.00 USD on every day. The v1 and v2 price tables are equal on every one of the window's 265,045 zone-hours. Idea 13 was rebuilt from the build weights and idea 14 from the gate weights; both match. My one-day-late totals equal the v2 scorer's exactly. They differ from the v1 lab by at most 952 USD, because the lab's first window day carries 31 Dec 2023 positions and mine starts empty.

## 1. Event dependence

The two windows are 2026-01-24 to 02-02 (the winter storm, 10 days) and 2026-06-28 to 07-05 (the heat wave, 8 days). On those 18 days always supply earned 280,357 of its 347,197 (81%). Most of that came on 27 to 29 January: 140,584 on 28 January alone. Without the 18 days, always supply at 1 MW earns 4.95% a year on 500,000. That is 66,840 over the other 986 days, and 2025 is negative.

The supply-leaning rows follow the same pattern: 58% to 84% of their 1 MW net comes from the 18 days. For V8_error_mining it is 81%: 276,759 of 342,077. Their build-year scales of 1.5 to 2.5 MW multiply that windfall. Without the event days most of them still clear 10% at the build scale. But always supply held at the same MW clears a similar figure, as the column "always supply at the same MW" shows: B_gbm makes 20.7% where always supply at 2.3 MW makes 11.4%. Table 2 tests whether that difference is real.

The C family depends far less on the events, 10% to 40% of net, and its money sits in 2025 instead. V13_C_deep_all made 91,541 of 117,104 there, and the scorers flagged the C rows' 2025 Sharpe above 3. Without events, 2026 is close to zero or negative for most C rows.

Without the event days, nine of the 28 candidates no longer clear the bar at the sizing view: idea 14 (8.2%), V15a_V8 (Sharpe 0.34), V15b_V13_limit (5.7%), V7_recency (4.4%, Sharpe 0.27), V8_error_mining (Sharpe 0.39), V2_limit_gbm (9.7%), V11_C_gbm_ens4 (9.8%), V1_C_gbm (7.3%) and V14e (one positive year).

## 2. Beats the no-model rule

Excess is the daily net minus the rule's daily net, at 1 MW, in USD a year. The intervals are 95%, from 10,000 draws of whole days (iid) and from 5,000 draws of 7-day stationary blocks (the lab's own bootstrap).

- **Over always supply:** with all days, two rows have an iid interval above zero. The first is idea13_portfolio, +108,195 [7,052; 223,025], whose block interval includes zero. The second is V3_tail_gbm, +331,383, which trades 2.05 MW on average, so this is size and not timing. Without the event days, no row's interval excludes zero. The best point estimates are V3 (+105,278), idea 13 (+76,138) and idea 14 (+50,146), all with intervals that are wide and straddle zero.
- **Over the habit rule** (the registered comparison): idea13_portfolio is the only row whose interval stays above zero without the events, at +76,541 a year, iid [9,331; 142,679]. Its block interval [-5,003; 151,195] does not. With 28 rows tested, about one such hit is expected by chance.
- **Pair rows over C_static_pairs:** every C model's point estimate is below the static pairs, from -1,168 (v1 C_deep) to -33,701 (V14e) a year. V14e is significantly worse with and without the events, and V14c is significantly worse without them. The C models add nothing to a persistent pair stance.

## 3. Placebos and the leak check

I read the registered placebos (one day late, shuffled or permuted, random days in the same month) from the scorers' outputs and recomputed one day late. For the supply-leaning rows, one day late keeps between a fifth (V9: 49,557 of 292,005) and all of the real net, because their positions change little from one day to the next. B_deep keeps 462,644 of 432,594, more than it earns itself, and V7_recency keeps 363,046 of 314,453. A row whose real result one day late nearly matches, or beats, is not timing anything. Shuffled dates beat or match the real rule for V14e (share 1.0), V11_C_deep_ens4 (0.95), C_static_pairs (1.0) and idea 14 (0.80). Random days beat V14e 82% of the time.

Leak check on the event days: on the 18 days, one day late earns less than the real rule for every supply-leaning row (V8: 168,678 less). Read on its own, that could be a leak. Two facts say it is not:

- **The hit rates show no foreknowledge.** On the event days these rows pick the winning side in 54% to 67% of traded zone-hours, while always supply is right 68.4% of the time on the same days. A row that could see real time would be right far more often than a fixed side. Away from the events their hit rates (0.51 to 0.62) are within about 5 points of their own 2021 to 2023 rates.
- **Their event gain matches always supply's.** On the event days the excess over always supply is close to zero (V15a_V5: -62, V7_flags: -642, V8: -3,598). The real positions were simply supply on storm days, and the day-late copies were partly not.

For some pair rows one day late beats the real rule on the event days, by up to 15,129 (V13_C_deep_all). That is consistent with a stance that does not depend on the day, and it does not point to a misaligned date.

## 4. Data checks

- **Publication times in the price table** (v2 table; the v1 table is identical): on the window, no real-time price is stamped before its hour ends (from June to August the earliest stamp is 15 minutes after). No real-time or day-ahead price of a delivery day is stamped at or before 05:00 on the bid day.
- **Feature panels** (v2 panel_2010_2026 and v1 panel_base_20260930, 265,045 window rows each): bid_date is the delivery date minus one on every row. rt_dm1_h equals the real-time price of the day before the bid day in 99.99% of rows. I compared every numeric feature with the delivery day's real-time price and gap, and with the bid day's. The only exact matches above 0.1% are rt_last against the bid day's real-time price (3.8%). That is the last hour published before 05:00 on the bid day (03:00, published 04:15), which the registration allows ("D's hours already published"). The largest correlation with the delivery-day gap is 0.32 (da_d0_mean, the bid day's day-ahead price, published the day before). Labels are stamped after the decision time on every row.
- **The rebuilt July days** (1 to 18 and 20 July 2026, 5,016 zone-hours): their real-time stamps follow the same rule as official rows (median 1.25 hours after the hour starts). Their energy component matches NYISO's official hourly generator-bus real-time energy component within 0.01 USD in every hour of every rebuilt day, the same agreement as on official days. That is an independent check from files that were never rebuilt. The prices are real, but the days were unusual: real time sat 7.55 USD below day-ahead on average, against 2.05 above on the official June to August neighbours. Most supply-leaning rows earned 500 to 3,200 a day on them (V3, at its larger size, 6,500), against -600 to +700 a day on the neighbours. Replace the rebuilt prices by NaN and July shrinks a lot: V3 136,285 becomes 12,522, V8 62,244 becomes 860, always supply 28,533 becomes -8,125. Window totals fall by up to a third (V1_C_gbm 89,857 becomes 60,013; B_deep barely moves). No verdict changes, because no row beats the rules either way. Features after July also read the rebuilt days; removing that effect needs a full rerun, which I did not do.

## 5. Why heldout_summary.py crashed

The cause is not raw["fragility"] being a string. It is raw["fragility"]["random_days"]. side/lab.py `fragility()` sets that field to the string "not applicable: the same book every day" for always_supply (lab.py line 910). In the build years always supply never passed the bar, so it never reached the too-good loop. On the window it passes the 1 MW bar (347,197, Sharpe 0.89), so it does, and line 73 calls `.get` on the string. A two-line guard fixes it.

The corrected copy is `audit_scratch/heldout_summary_audit.py` (diff against the frozen file: the guard, the list paths and the output path). Its output is `audit_scratch/summary_out/DONE_audit.md`: 69 rows, 43 clear the bar. Note that this summary uses the 1 MW PASS, not the sizing view.

Also broken, and not in the brief:

- **Chart data crashed too.** CHARTDATA_FAILED, KeyError 'V15a_V4_limit' in build_chartdata.py line 357: the build daily table has the three not-run V4 rows and the held-out results do not.
- **The run script marks failed stages as done.** It logged "stage summary done" and "ALL DONE" after both failures.
- **V12_alloc ran and was scored.** The 9 Oct addendum dropped it from the held-out run as not reproducible. It must be shown as not run. It also fails the bar, at 6.5% and Sharpe 0.35.
- **The 2026 fee rate is unverified.** The v1 runner notes that 2026 RS1 is unverified (fees.SOURCE). It is used for nine months of the window.

## 6. Verdicts

The verdict table at the end lists every row. Event luck means the money is the storm and heat days that always supply also caught, and nothing beyond that is detectable. Pair stance means the profit is the persistent zone-pair spread that the no-model static pairs earn as well.

- **A_regression_v1 (v1):** Event luck. The spike filter sits out some hours, but its excess over always supply is +7,328 a year [-49,662; 65,129]. Most of its net comes from the storm: 217,504 of 367,353. Without the events it still earns 17.7% at 1.71 MW, while always supply at that size earns 8.5%, and that difference is not significant.
- **B_gbm and B_deep (v1):** Event luck. Weather-driven, and 74% and 78% of net comes from the 18 days. B_gbm caught some of the heat wave that always supply lost (34,596 against -7,084), but neither row beats always supply with or without the events. B_deep's one-day-late copy earns more than B_deep.
- **C_deep (v1):** Pair stance. Steady (Sharpe 2.12) and only 34% event-driven, but equal to the static pairs (-1,168 a year [-20,220; 19,006]). Without the events 2024 and 2026 are flat, so 2025 carries it.
- **idea13_portfolio (v1):** Mostly event luck, and the nearest thing to skill in the run. It is the only row above the habit rule without the events on the iid bootstrap, +76,541 [9,331; 142,679], but the block bootstrap and the comparison with always supply do not support that, and the count of rows tested explains one such hit. Not a claim to publish as skill.
- **idea14 (v1):** Event luck. 59% from the 18 days, 8.2% without them (fails the bar), and shuffled dates beat it 80% of the time.
- **V1_C_deep, V1_C_gbm, V11_C_deep_ens4, V11_C_gbm_ens4, V13_C_deep_all, V14d, V14f (v2):** Pair stance. Not event luck, but not model skill either: each sits below C_static_pairs by 6,000 to 21,000 a year, none significantly. V1_C_gbm and V11_C_gbm fall below 10% without the events.
- **V14c_conformal_skip and V14e_mlp_multi (v2):** Pair stance, worse than no model. Both are significantly below the static pairs. V14e is beaten by random days 82% of the time and by every shuffle, and it has one positive year without the events.
- **V2_limit_gbm, V3_tail_gbm, V5, V7_flags, V7_recency, V8, V9, V15a_V5, V15a_V8, V15b_both, V15b_limit, V15c, V16 (v2):** Event luck. Each earns 58% to 84% of its 1 MW net on the 18 days and matches always supply on them almost to the dollar. None beats always supply without them. V3's large figures are V2's bet at about 2 MW. V7_recency, V8, V15a_V8 and V15b_limit fail the bar once the events are removed.
- **V12_alloc (v2):** Broken by procedure: it should not have run. On the numbers it fails anyway.
- **Comparisons:** always supply earns 81% of its net from the 18 days, and only 4.95% a year at 1 MW without them. The habit rule earns 103,349 and is beaten by every shuffle. The static pairs earn 147,275 at 1 MW (19.1% at 1.78 MW, 15.5% without the events), steadily but mostly in 2025.


## Verdict table

Return a year on 500,000 USD at the build-year scale. Excess is over always supply at 1 MW per position, USD a year, with the iid 95% interval. Without events means without 2026-01-24..02-02 and 2026-06-28..07-05.

| row | return a year | without events | excess over always supply | without events | verdict |
|---|---|---|---|---|---|
| v1:A_regression_v1 | 45.7% | 17.7% | 7,328 [-49,662; 65,129] | 26,923 [-22,096; 77,497] | event luck |
| v1:B_gbm | 78.0% | 20.7% | 43,391 [-68,069; 160,751] | 20,161 [-91,277; 135,701] | event luck |
| v1:B_deep | 45.6% | 10.4% | 31,046 [-72,544; 141,107] | 10,971 [-85,659; 116,456] | event luck |
| v1:C_deep | 41.1% | 27.6% | -73,848 [-245,105; 85,776] | 10,523 [-87,079; 121,376] | pair stance, no model skill |
| v1:idea13_portfolio | 27.8% | 12.0% | 108,195 [7,052; 223,025] | 76,138 [-19,911; 180,819] | mostly event luck; weak, unproven hint of skill |
| v1:idea14_learned_allocator_candidates | 19.9% | 8.2% | 54,563 [-70,130; 190,425] | 50,146 [-64,247; 174,414] | event luck |
| v2:V11_C_deep_ens4 | 18.9% | 15.7% | -86,700 [-265,828; 76,912] | 8,034 [-85,256; 114,806] | pair stance, no model skill |
| v2:V11_C_gbm_ens4 | 14.7% | 9.8% | -84,357 [-257,949; 75,619] | 3,351 [-91,679; 112,646] | pair stance, no model skill |
| v2:V13_C_deep_all | 15.7% | 14.4% | -83,649 [-260,777; 77,442] | 14,392 [-78,038; 120,106] | pair stance, no model skill |
| v2:V14c_conformal_skip | 21.5% | 13.2% | -96,263 [-272,825; 64,970] | -6,427 [-98,334; 95,518] | pair stance, worse than no model |
| v2:V14d_seed_ens | 15.7% | 11.1% | -88,481 [-269,567; 74,044] | 1,953 [-89,033; 103,520] | pair stance, no model skill |
| v2:V14e_mlp_multi | 13.2% | 10.1% | -106,382 [-282,567; 52,882] | -9,648 [-102,169; 95,569] | pair stance, worse than no model |
| v2:V14f_global_zone | 20.8% | 16.3% | -93,675 [-274,894; 70,909] | 802 [-90,732; 105,507] | pair stance, no model skill |
| v2:V15a_V5_limit | 75.8% | 25.1% | 24,713 [-67,332; 119,971] | 25,187 [-61,005; 115,648] | event luck |
| v2:V15a_V8_limit | 52.8% | 8.7% | -4,633 [-90,749; 88,920] | -4,586 [-87,214; 80,757] | event luck |
| v2:V15b_V13_both | 43.1% | 18.1% | 15,639 [-78,794; 117,663] | 34,853 [-47,633; 126,804] | event luck |
| v2:V15b_V13_limit | 27.7% | 5.7% | -15,358 [-97,989; 72,061] | -1,796 [-77,531; 80,236] | event luck |
| v2:V15c_wx_border_gbm | 63.4% | 20.5% | 31,452 [-67,670; 135,371] | 26,192 [-67,262; 123,744] | event luck |
| v2:V16_B_gefs_joined | 46.0% | 12.7% | 23,177 [-68,238; 116,747] | 16,337 [-67,536; 102,247] | event luck |
| v2:V1_C_deep | 24.5% | 22.0% | -79,098 [-253,637; 80,421] | 17,517 [-75,154; 120,939] | pair stance, no model skill |
| v2:V1_C_gbm | 11.8% | 7.3% | -93,555 [-272,816; 68,246] | -4,541 [-97,468; 98,754] | pair stance, no model skill |
| v2:V2_limit_gbm | 50.3% | 9.7% | 15,790 [-61,874; 99,183] | 2,580 [-73,445; 78,041] | event luck |
| v2:V3_tail_gbm | 48.5% | 13.8% | 331,383 [31,567; 662,749] | 105,278 [-77,795; 273,503] | event luck |
| v2:V5_border_inputs | 32.4% | 13.9% | -19,671 [-131,214; 93,111] | 20,938 [-69,918; 117,926] | event luck |
| v2:V7_flags | 35.5% | 10.9% | 19,182 [-86,747; 135,402] | 19,770 [-84,571; 130,143] | event luck |
| v2:V7_recency | 27.9% | 4.4% | -11,904 [-105,617; 91,789] | -6,538 [-102,066; 95,861] | event luck |
| v2:V8_error_mining | 56.2% | 10.9% | -1,861 [-89,680; 89,725] | -563 [-79,508; 78,074] | event luck |
| v2:V9_decompose | 38.6% | 13.5% | -20,065 [-132,432; 96,448] | 12,217 [-78,001; 108,930] | event luck |
| v1:always_supply | 7.3% | 1.4% |  |  | comparison: event windfall |
| v1:baseline_usual_side | 6.0% | 3.9% | -88,650 [-247,860; 56,859] | -403 [-77,789; 87,596] | comparison |
| v2:C_static_pairs | 19.1% | 15.5% | -72,681 [-255,200; 95,790] | 18,866 [-78,288; 128,235] | comparison: steady pair stance |
| v2:V12_alloc | 6.5% | -2.0% | -46,444 [-290,728; 134,223] | -49,070 [-289,928; 124,377] | broken: should not have run |

## Detail tables

Always supply at 1 MW: 25.2% a year on 500,000 with every day, 4.95% without the event days. The column 'always supply at the same MW' multiplies the second figure by the row's scale.

### Table 1. Event dependence

| row | build scale (MW) | net at 1 MW | storm window | heat window | share of net from the 18 event days | net at 1 MW without them | return a year at build scale | same, without the event days | always supply at the same MW, without the event days | 2024 / 2025 / 2026 return without the event days | positive years without them |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1:A_regression_v1 | 1.71 | 367,353 | 217,504 | 10,281 | 62.0% | 139,569 | 45.7% | 17.7% | 8.5% | 20.6 / 11.4 / 22.3 | 3 of 3 |
| v1:B_gbm | 2.3 | 466,552 | 310,652 | 34,596 | 74.0% | 121,303 | 78.0% | 20.7% | 11.4% | 6.8 / 11.1 / 54.2 | 3 of 3 |
| v1:B_deep | 1.45 | 432,594 | 312,392 | 23,725 | 77.7% | 96,477 | 45.6% | 10.4% | 7.2% | 12.9 / 11.5 / 5.1 | 3 of 3 |
| v1:C_deep | 3.92 | 144,063 | 21,445 | 27,352 | 33.9% | 95,265 | 41.1% | 27.6% | 19.4% | 2.7 / 72.9 / -1.4 | 2 of 3 |
| v1:idea13_portfolio | 0.594 | 644,808 | 304,682 | 67,610 | 57.7% | 272,516 | 27.8% | 12.0% | 2.9% | 7.3 / 17.4 / 10.9 | 3 of 3 |
| v1:idea14_learned_allocator_candidates | 0.55 | 497,282 | 221,587 | 73,393 | 59.3% | 202,302 | 19.9% | 8.2% | 2.7% | 1.7 / 15.0 / 7.9 | 3 of 3 |
| v2:V11_C_deep_ens4 | 2.39 | 108,714 | 5,534 | 14,637 | 18.6% | 88,543 | 18.9% | 15.7% | 11.8% | 4.9 / 35.6 / 2.6 | 3 of 3 |
| v2:V11_C_gbm_ens4 | 1.75 | 115,158 | 19,248 | 20,019 | 34.1% | 75,891 | 14.7% | 9.8% | 8.7% | 6.1 / 15.2 / 7.5 | 3 of 3 |
| v2:V13_C_deep_all | 1.84 | 117,104 | 4,104 | 7,282 | 9.7% | 105,719 | 15.7% | 14.4% | 9.1% | 4.8 / 33.7 / 0.6 | 3 of 3 |
| v2:V14c_conformal_skip | 3.59 | 82,409 | 8,004 | 24,927 | 40.0% | 49,478 | 21.5% | 13.2% | 17.8% | 12.5 / 25.5 / -3.5 | 2 of 3 |
| v2:V14d_seed_ens | 2.08 | 103,814 | 2,715 | 28,985 | 30.5% | 72,115 | 15.7% | 11.1% | 10.3% | 13.4 / 14.1 / 3.4 | 3 of 3 |
| v2:V14e_mlp_multi | 3.33 | 54,574 | 6,142 | 7,655 | 25.3% | 40,777 | 13.2% | 10.1% | 16.5% | -7.5 / 39.1 / -6.3 | 1 of 3 |
| v2:V14f_global_zone | 3.19 | 89,526 | -3,257 | 23,778 | 22.9% | 69,005 | 20.8% | 16.3% | 15.8% | 10.9 / 25.1 / 11.4 | 3 of 3 |
| v2:V15a_V5_limit | 2.51 | 415,175 | 249,182 | 31,113 | 67.5% | 134,880 | 75.8% | 25.1% | 12.4% | 20.9 / 31.0 / 22.6 | 3 of 3 |
| v2:V15a_V8_limit | 2.17 | 334,453 | 248,450 | 31,552 | 83.7% | 54,450 | 52.8% | 8.7% | 10.7% | 7.3 / 10.0 / 9.1 | 3 of 3 |
| v2:V15b_V13_both | 1.52 | 390,214 | 195,870 | 33,354 | 58.7% | 160,990 | 43.1% | 18.1% | 7.5% | 16.5 / 32.7 / -0.3 | 2 of 3 |
| v2:V15b_V13_limit | 1.25 | 304,952 | 225,490 | 17,473 | 79.7% | 61,989 | 27.7% | 5.7% | 6.2% | 4.5 / 12.0 / -1.5 | 2 of 3 |
| v2:V15c_wx_border_gbm | 2.01 | 433,710 | 275,604 | 20,512 | 68.3% | 137,593 | 63.4% | 20.5% | 9.9% | 34.1 / 12.1 / 12.9 | 3 of 3 |
| v2:V16_B_gefs_joined | 1.54 | 410,950 | 260,635 | 39,344 | 73.0% | 110,971 | 46.0% | 12.7% | 7.6% | 11.2 / 8.6 / 20.4 | 3 of 3 |
| v2:V1_C_deep | 2.6 | 129,623 | 8,115 | 7,350 | 11.9% | 114,158 | 24.5% | 22.0% | 12.9% | 8.4 / 47.3 / 5.2 | 3 of 3 |
| v2:V1_C_gbm | 1.81 | 89,857 | 9,192 | 26,092 | 39.3% | 54,573 | 11.8% | 7.3% | 9.0% | 7.4 / 10.4 / 2.7 | 3 of 3 |
| v2:V2_limit_gbm | 1.77 | 390,630 | 282,294 | 34,527 | 81.1% | 73,810 | 50.3% | 9.7% | 8.8% | 9.5 / 6.9 / 13.9 | 3 of 3 |
| v2:V3_tail_gbm | 0.53 | 1,258,726 | 800,909 | 106,582 | 72.1% | 351,236 | 48.5% | 13.8% | 2.6% | 12.7 / 19.8 / 6.7 | 3 of 3 |
| v2:V5_border_inputs | 1.52 | 293,087 | 140,199 | 29,487 | 57.9% | 123,401 | 32.4% | 13.9% | 7.5% | 18.6 / 29.4 / -15.1 | 2 of 3 |
| v2:V7_flags | 1.22 | 399,962 | 256,489 | 23,226 | 69.9% | 120,246 | 35.5% | 10.9% | 6.0% | 15.6 / -2.6 / 23.4 | 2 of 3 |
| v2:V7_recency | 1.22 | 314,453 | 269,174 | -3,898 | 84.4% | 49,178 | 27.9% | 4.4% | 6.0% | 10.6 / -11.0 / 17.7 | 2 of 3 |
| v2:V8_error_mining | 2.26 | 342,077 | 233,064 | 43,695 | 80.9% | 65,318 | 56.2% | 10.9% | 11.2% | 15.8 / 6.7 / 9.9 | 3 of 3 |
| v2:V9_decompose | 1.82 | 292,005 | 170,489 | 21,674 | 65.8% | 99,842 | 38.6% | 13.5% | 9.0% | 6.2 / 26.0 / 6.0 | 3 of 3 |
| v1:always_supply | 0.29 | 347,197 | 287,441 | -7,084 | 80.7% | 66,840 | 7.3% | 1.4% | 1.4% | 3.4 / -0.9 / 1.9 | 2 of 3 |
| v1:baseline_usual_side | 0.8 | 103,349 | 38,337 | -738 | 36.4% | 65,751 | 6.0% | 3.9% | 4.0% | 6.3 / 3.6 / 0.8 | 3 of 3 |
| v2:C_static_pairs | 1.78 | 147,275 | -148 | 29,619 | 20.0% | 117,805 | 19.1% | 15.5% | 8.8% | 6.1 / 36.3 / -0.8 | 2 of 3 |
| v2:V12_alloc | 0.41 | 219,443 | 238,717 | 46,443 | 129.9% | -65,717 | 6.5% | -2.0% | 2.0% | 4.8 / -13.0 / 3.9 | 2 of 3 |

### Table 2. Excess over the no-model rules (1 MW per position, USD a year, 95% intervals)

| row | over always supply, whole window | iid days | 7-day blocks | without the event days | iid days | 7-day blocks | over static pairs, whole window [iid] | over static pairs, without events [iid] |
|---|---|---|---|---|---|---|---|---|
| v1:A_regression_v1 | 7,328 | [-49662, 65129] | [-73774, 77734] | 26,923 | [-22096, 77497] | [-32685, 85830] |  |  |
| v1:B_gbm | 43,391 | [-68069, 160751] | [-97749, 183937] | 20,161 | [-91277, 135701] | [-128613, 162403] |  |  |
| v1:B_deep | 31,046 | [-72544, 141107] | [-91237, 164608] | 10,971 | [-85659, 116456] | [-113605, 145990] |  |  |
| v1:C_deep | -73,848 | [-245105, 85776] | [-346912, 145653] | 10,523 | [-87079, 121376] | [-125545, 168824] | -1,168 [-20220, 19006] | -8,344 [-21770, 5184] |
| v1:idea13_portfolio | 108,195 | [7052, 223025] | [-16443, 247254] | 76,138 | [-19911, 180819] | [-48517, 212712] |  |  |
| v1:idea14_learned_allocator_candidates | 54,563 | [-70130, 190425] | [-112562, 235149] | 50,146 | [-64247, 174414] | [-113135, 222877] |  |  |
| v2:V11_C_deep_ens4 | -86,700 | [-265828, 76912] | [-371257, 132914] | 8,034 | [-85256, 114806] | [-125348, 163839] | -14,019 [-32094, 2708] | -10,832 [-27171, 4473] |
| v2:V11_C_gbm_ens4 | -84,357 | [-257949, 75619] | [-356996, 135101] | 3,351 | [-91679, 112646] | [-131682, 164037] | -11,676 [-45289, 19582] | -15,516 [-45537, 12697] |
| v2:V13_C_deep_all | -83,649 | [-260777, 77442] | [-366602, 136027] | 14,392 | [-78038, 120106] | [-116640, 167992] | -10,969 [-30492, 6283] | -4,474 [-19180, 9463] |
| v2:V14c_conformal_skip | -96,263 | [-272825, 64970] | [-376850, 119528] | -6,427 | [-98334, 95518] | [-135771, 142115] | -23,582 [-43663, -4554] | -25,293 [-44598, -6669] |
| v2:V14d_seed_ens | -88,481 | [-269567, 74044] | [-373992, 129168] | 1,953 | [-89033, 103520] | [-124076, 148970] | -15,800 [-36861, 4284] | -16,914 [-37536, 2590] |
| v2:V14e_mlp_multi | -106,382 | [-282567, 52882] | [-386047, 110638] | -9,648 | [-102169, 95569] | [-140678, 140214] | -33,701 [-56243, -13325] | -28,514 [-48320, -11379] |
| v2:V14f_global_zone | -93,675 | [-274894, 70909] | [-379504, 126821] | 802 | [-90732, 105507] | [-125340, 148535] | -20,995 [-43101, 305] | -18,065 [-40369, 2906] |
| v2:V15a_V5_limit | 24,713 | [-67332, 119971] | [-83805, 137130] | 25,187 | [-61005, 115648] | [-77801, 129871] |  |  |
| v2:V15a_V8_limit | -4,633 | [-90749, 88920] | [-110111, 102621] | -4,586 | [-87214, 80757] | [-106087, 98133] |  |  |
| v2:V15b_V13_both | 15,639 | [-78794, 117663] | [-118962, 150161] | 34,853 | [-47633, 126804] | [-70435, 163765] |  |  |
| v2:V15b_V13_limit | -15,358 | [-97989, 72061] | [-132411, 105560] | -1,796 | [-77531, 80236] | [-105046, 116161] |  |  |
| v2:V15c_wx_border_gbm | 31,452 | [-67670, 135371] | [-82080, 144167] | 26,192 | [-67262, 123744] | [-81604, 137137] |  |  |
| v2:V16_B_gefs_joined | 23,177 | [-68238, 116747] | [-86298, 130679] | 16,337 | [-67536, 102247] | [-92218, 117658] |  |  |
| v2:V1_C_deep | -79,098 | [-253637, 80421] | [-356763, 136029] | 17,517 | [-75154, 120939] | [-112357, 169343] | -6,418 [-25101, 10970] | -1,350 [-16027, 12526] |
| v2:V1_C_gbm | -93,555 | [-272816, 68246] | [-373665, 124783] | -4,541 | [-97468, 98754] | [-133739, 145464] | -20,874 [-48635, 5816] | -23,407 [-50834, 2936] |
| v2:V2_limit_gbm | 15,790 | [-61874, 99183] | [-78907, 111023] | 2,580 | [-73445, 78041] | [-90704, 93583] |  |  |
| v2:V3_tail_gbm | 331,383 | [31567, 662749] | [-12871, 806140] | 105,278 | [-77795, 273503] | [-84310, 290146] |  |  |
| v2:V5_border_inputs | -19,671 | [-131214, 93111] | [-187070, 123556] | 20,938 | [-69918, 117926] | [-83526, 131700] |  |  |
| v2:V7_flags | 19,182 | [-86747, 135402] | [-106403, 148769] | 19,770 | [-84571, 130143] | [-103592, 149028] |  |  |
| v2:V7_recency | -11,904 | [-105617, 91789] | [-128472, 103647] | -6,538 | [-102066, 95861] | [-118675, 114546] |  |  |
| v2:V8_error_mining | -1,861 | [-89680, 89725] | [-108532, 102621] | -563 | [-79508, 78074] | [-93729, 94065] |  |  |
| v2:V9_decompose | -20,065 | [-132432, 96448] | [-173237, 122634] | 12,217 | [-78001, 108930] | [-98923, 135514] |  |  |
| v1:baseline_usual_side | -88,650 | [-247860, 56859] | [-334666, 96320] | -403 | [-77789, 87596] | [-96910, 121533] |  |  |
| v2:C_static_pairs | -72,681 | [-255200, 95790] | [-363972, 153978] | 18,866 | [-78288, 128235] | [-115865, 174630] |  |  |
| v2:V12_alloc | -46,444 | [-290728, 134223] | [-305909, 151257] | -49,070 | [-289928, 124377] | [-305750, 139074] |  |  |

### Table 3. Placebos on the window and the event-day leak check (1 MW, USD)

| row | real net | one day late (scorer) | shuffled or permuted mean (scorer) | share of shuffles at or above real | random days beat it | event days: real | event days: one day late | late minus real | hit rate on event days (always supply 0.684) | hit rate elsewhere | hit rate 2021 to 2023 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1:A_regression_v1 | 367,353 | 202,610 | 192,034 | 0.05 | 0.006 | 227,785 | 170,280 | -57,504 | 0.6915 | 0.6207 | 0.608 |
| v1:B_gbm | 466,552 | 251,325 | 76,298 | 0.0 | 0.0 | 345,249 | 253,464 | -91,785 | 0.6706 | 0.5361 | 0.5165 |
| v1:B_deep | 432,594 | 462,644 | 86,063 | 0.0 | 0.027 | 336,117 | 333,611 | -2,506 | 0.5719 | 0.5401 | 0.5143 |
| v1:C_deep | 144,063 | 146,686 | 100,723 | 0.0 | 0.028 | 48,797 | 48,205 | -593 | 0.5063 | 0.5144 | 0.5123 |
| v1:idea13_portfolio | 644,808 | n/a | n/a | None | 0.0 | 372,292 | 307,321 | -64,971 | 0.6117 | 0.5434 | n/a |
| v1:idea14_learned_allocator_candidates | 497,282 | 473,239 | 522,975 | 0.8 | 0.001 | 294,980 | 273,919 | -21,062 | 0.5485 | 0.5275 | n/a |
| v2:V11_C_deep_ens4 | 108,714 | 105,824 | 125,372 | 0.95 | 0.172 | 20,170 | 30,826 | 10,655 | 0.4918 | 0.4996 | 0.5129 |
| v2:V11_C_gbm_ens4 | 115,158 | 38,772 | -40,166 | 0.0 | 0.0 | 39,267 | 26,113 | -13,154 | 0.4939 | 0.5021 | 0.5355 |
| v2:V13_C_deep_all | 117,104 | 114,690 | 85,177 | 0.05 | 0.075 | 11,386 | 26,515 | 15,129 | 0.5061 | 0.4953 | 0.5046 |
| v2:V14c_conformal_skip | 82,409 | 43,400 | 42,063 | 0.0 | 0.0 | 32,931 | 9,850 | -23,081 | 0.4577 | 0.4728 | 0.5165 |
| v2:V14d_seed_ens | 103,814 | 31,971 | 21,521 | 0.0 | 0.001 | 31,699 | 11,627 | -20,073 | 0.5091 | 0.4993 | 0.5193 |
| v2:V14e_mlp_multi | 54,574 | 48,526 | 113,885 | 1.0 | 0.819 | 13,797 | 23,888 | 10,091 | 0.4614 | 0.477 | 0.4974 |
| v2:V14f_global_zone | 89,526 | 24,447 | -27,711 | 0.0 | 0.0 | 20,521 | -792 | -21,313 | 0.5395 | 0.5138 | 0.515 |
| v2:V15a_V5_limit | 415,175 | 187,117 | -39,743 | 0.0 | 0.0 | 280,295 | 262,821 | -17,473 | 0.6442 | 0.5407 | 0.5249 |
| v2:V15a_V8_limit | 334,453 | 164,253 | 7,569 | 0.0 | 0.036 | 280,002 | 226,638 | -53,365 | 0.617 | 0.5329 | 0.5317 |
| v2:V15b_V13_both | 390,214 | 274,402 | 108,183 | 0.0 | 0.0 | 229,224 | 204,210 | -25,014 | 0.6635 | 0.5499 | 0.5193 |
| v2:V15b_V13_limit | 304,952 | 198,296 | 84,714 | 0.05 | 0.004 | 242,963 | 232,814 | -10,149 | 0.6733 | 0.5412 | 0.4896 |
| v2:V15c_wx_border_gbm | 433,710 | 209,784 | -50,984 | 0.0 | 0.0 | 296,117 | 241,868 | -54,249 | 0.6308 | 0.5338 | 0.5304 |
| v2:V16_B_gefs_joined | 410,950 | 129,846 | -15,337 | 0.0 | 0.014 | 299,979 | 206,165 | -93,814 | 0.6513 | 0.5506 | 0.5395 |
| v2:V1_C_deep | 129,623 | 107,914 | 117,880 | 0.2 | 0.024 | 15,464 | 26,567 | 11,102 | 0.4898 | 0.4953 | 0.5136 |
| v2:V1_C_gbm | 89,857 | -24,050 | -46,616 | 0.0 | 0.0 | 35,284 | 3 | -35,282 | 0.5163 | 0.5014 | 0.5394 |
| v2:V2_limit_gbm | 390,630 | 238,129 | -4,541 | 0.05 | 0.012 | 316,821 | 285,391 | -31,429 | 0.6281 | 0.5494 | 0.5394 |
| v2:V3_tail_gbm | 1,258,726 | 623,751 | -16,159 | 0.0 | 0.006 | 907,490 | 536,031 | -371,460 | 0.6446 | 0.5524 | 0.5429 |
| v2:V5_border_inputs | 293,087 | 56,955 | -46,610 | 0.1 | 0.009 | 169,686 | 129,696 | -39,990 | 0.5799 | 0.5316 | 0.5354 |
| v2:V7_flags | 399,962 | 312,409 | -161,062 | 0.0 | 0.007 | 279,715 | 193,683 | -86,032 | 0.5416 | 0.5114 | 0.5057 |
| v2:V7_recency | 314,453 | 363,046 | -149,851 | 0.0 | 0.035 | 265,276 | 222,422 | -42,853 | 0.5539 | 0.5071 | 0.5069 |
| v2:V8_error_mining | 342,077 | 128,278 | 26,969 | 0.1 | 0.052 | 276,759 | 108,081 | -168,678 | 0.6343 | 0.5331 | 0.5372 |
| v2:V9_decompose | 292,005 | 49,557 | -38,615 | 0.1 | 0.073 | 192,163 | 41,072 | -151,091 | 0.6095 | 0.5304 | 0.5356 |
| v1:always_supply | 347,197 | n/a | n/a | None | not applicable: the same book every day | 280,357 | 280,357 | 0 | 0.6837 | 0.5944 | n/a |
| v1:baseline_usual_side | 103,349 | 30,601 | 480,550 | 1.0 | 1.0 | 37,598 | -23,167 | -60,765 | 0.5939 | 0.5433 | n/a |
| v2:C_static_pairs | 147,275 | 147,264 | 174,749 | 1.0 | 0.731 | 29,471 | 31,065 | 1,595 | 0.4922 | 0.5013 | 0.5092 |
| v2:V12_alloc | 219,443 | 199,692 | 136,679 | 0.25 | 0.197 | 285,160 | 196,922 | -88,237 | 0.609 | 0.5372 | 0.5381 |

### Table 4. Rebuilt July days (1 MW, USD)

| row | 19 rebuilt days, total | mean a day | mean a day, rebuilt days after 5 July | official neighbours (June, 19 and 21 to 31 July, August), mean a day | neighbours without the heat window | July total | July with rebuilt prices as NaN | window total with rebuilt prices as NaN |
|---|---|---|---|---|---|---|---|---|
| v1:A_regression_v1 | 25,900 | 1,363 | 1,244 | -130 | -161 | 26,592 | 693 | 341,454 |
| v1:B_gbm | 31,430 | 1,654 | -20 | 537 | 519 | 38,423 | 6,993 | 435,122 |
| v1:B_deep | 1,433 | 75 | -1,760 | 182 | 223 | 5,358 | 3,925 | 431,161 |
| v1:C_deep | 13,716 | 722 | -985 | 221 | 232 | 18,248 | 4,532 | 130,346 |
| v1:idea13_portfolio | 47,014 | 2,474 | -1,354 | 607 | 609 | 60,680 | 13,666 | 597,794 |
| v1:idea14_learned_allocator_candidates | 42,530 | 2,238 | -2,174 | 602 | 622 | 53,100 | 10,570 | 454,752 |
| v2:V11_C_deep_ens4 | 7,074 | 372 | -551 | 170 | 180 | 11,158 | 4,084 | 101,640 |
| v2:V11_C_gbm_ens4 | 19,563 | 1,030 | 34 | 380 | 383 | 26,499 | 6,937 | 95,596 |
| v2:V13_C_deep_all | 2,664 | 140 | -337 | 131 | 138 | 6,471 | 3,806 | 114,440 |
| v2:V14c_conformal_skip | 16,745 | 881 | -584 | 125 | 130 | 17,241 | 496 | 65,663 |
| v2:V14d_seed_ens | 31,267 | 1,646 | 166 | 88 | 91 | 31,137 | -130 | 72,547 |
| v2:V14e_mlp_multi | 3,822 | 201 | -289 | 52 | 57 | 5,220 | 1,398 | 50,752 |
| v2:V14f_global_zone | 26,640 | 1,402 | 236 | 144 | 144 | 26,650 | 11 | 62,886 |
| v2:V15a_V5_limit | 44,073 | 2,320 | 994 | 71 | 60 | 45,382 | 1,309 | 371,102 |
| v2:V15a_V8_limit | 35,331 | 1,860 | 324 | -154 | -171 | 35,658 | 327 | 299,122 |
| v2:V15b_V13_both | 23,524 | 1,238 | -575 | 69 | 47 | 24,919 | 1,395 | 366,691 |
| v2:V15b_V13_limit | 18,885 | 994 | 114 | -120 | -128 | 21,140 | 2,255 | 286,067 |
| v2:V15c_wx_border_gbm | 9,539 | 502 | -735 | 218 | 217 | 10,870 | 1,331 | 424,171 |
| v2:V16_B_gefs_joined | 39,312 | 2,069 | 166 | 67 | 36 | 41,273 | 1,960 | 371,638 |
| v2:V1_C_deep | 2,711 | 143 | -344 | 150 | 159 | 6,293 | 3,582 | 126,912 |
| v2:V1_C_gbm | 29,844 | 1,571 | 285 | 106 | 107 | 28,498 | -1,346 | 60,013 |
| v2:V2_limit_gbm | 54,332 | 2,860 | 1,421 | -202 | -212 | 58,404 | 4,072 | 336,298 |
| v2:V3_tail_gbm | 123,762 | 6,514 | 1,198 | -611 | -632 | 136,285 | 12,522 | 1,134,964 |
| v2:V5_border_inputs | 36,060 | 1,898 | 452 | -397 | -410 | 38,595 | 2,535 | 257,027 |
| v2:V7_flags | 15,310 | 806 | -802 | 691 | 767 | 22,683 | 7,374 | 384,652 |
| v2:V7_recency | 14,837 | 781 | 1,177 | 386 | 434 | 22,225 | 7,389 | 299,617 |
| v2:V8_error_mining | 61,384 | 3,231 | 1,352 | -459 | -496 | 62,244 | 860 | 280,693 |
| v2:V9_decompose | 42,011 | 2,211 | 1,486 | -31 | -39 | 44,387 | 2,376 | 249,994 |
| v1:always_supply | 36,658 | 1,929 | 3,277 | -606 | -662 | 28,533 | -8,125 | 310,538 |
| v1:baseline_usual_side | 34,576 | 1,820 | 2,682 | -521 | -575 | 26,965 | -7,610 | 68,773 |
| v2:C_static_pairs | 16,122 | 849 | -985 | 204 | 217 | 23,201 | 7,079 | 131,153 |
| v2:V12_alloc | 39,949 | 2,103 | -402 | 487 | 496 | 41,124 | 1,175 | 179,494 |

## Files

- `audit_scratch/audit_gene.py` (ran on gene, read only), `audit_scratch/gene_out/` (recomputed daily, one-day-late daily, checks JSON)
- `audit_scratch/analyse.py` and `analysis.json` (every number above)
- `audit_scratch/heldout_summary_audit.py` and `summary_out/DONE_audit.md` (the summary the run should have written)
