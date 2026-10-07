# 2023 rehearsal: development numbers

Written 6 Oct 2026. 2023 is build data: these numbers may guide fixes before the freeze and are not the test. Nothing on or after 1 January 2024 was read. Every loader goes through `model/lock.py`, which refuses those dates while `research/us/FREEZE` is absent, and it is absent. Full numbers are in `rehearsal_2023.json`.

## What ran

- **Panel.** 385,704 rows: 1,461 delivery days from 2020 to 2023, 11 zones, 23 to 25 hours a day, with no missing gap. There are 46 base features per row, computed from `timing.features_available_at(05:00 on D)`. Joined to the same rows are 17 zone-plus-generator summary columns, 23 weather columns (`features_weather.py`) and 29 outage columns (`features_outages.py`).
- **Tuning.** Each feature set tried 8 LightGBM configurations, trained on delivery dates up to 30 Dec 2021 and chosen on 2022 by mean daily P&L minus the baseline under the idea's own rule. Idea B trains from 25 Mar 2021. Clipping the training target at 250 USD/MWh won for A, B and the generator ablation. D won without clipping.
- **C's pairs.** These were chosen on the same 2022 predictions: CAPITL against NORTH, GENESE, CENTRL, MHK VL and WEST.
- **Walk-forward through 2023.** Each feature set was refit 12 times, once a month. Each refit trains on delivery dates up to two days before the month starts, and only on labels that were public by 05:00 on the month's first bid day. Because of that label rule, the December refit trains only to 28 Nov: the real-time file for 29 Nov was rewritten after the deadline.
- **Scoring.** A stationary bootstrap over the 365 days (mean block 7 days, 10,000 replicates, seed 20261006). A is tested at 95%. B, C and D are tested with the Holm correction.
- **Deep model.** See the section below.

## Verdicts for 2023 (gradient boosting; development numbers)

In 2023 the baseline traded every zone-hour, 264 MWh a day. It lost 191 USD a day, which is -0.73 USD per MWh.

| Idea | Word | Idea minus baseline, USD/day [corrected interval] | Idea alone, USD/day | Profit per MWh | Break-even fee | MWh/day | Worst month (idea alone) |
|---|---|---|---|---|---|---|---|
| A | pays | +303 [+145, +472] at 95% | +112 | 1.01 | 1.12 | 111 | Sep, -9,237 |
| B | inconclusive | +135 [-92, +384] at 97.5% | -56 | -0.22 | -0.12 | 252 | Feb, -28,645 |
| C | pays | +208 [+37, +379] at 98.3% | +17 | 0.09 | 0.19 | 196 | Aug, -8,161 |
| D | inconclusive | +83 [-76, +246] at 95%, Holm stopped at B | -109 | -0.44 | -0.33 | 246 | Sep, -16,845 |

The fee in 2023 was 0.1066 USD per MWh.

**Without the most extreme 1% of zone-hours** (|gap| above 83 USD), measured as idea minus baseline per day:

| Idea | USD/day [95% interval] | Idea's profit per MWh |
|---|---|---|
| A | +419 [+293, +557] | 1.73 |
| C | +275 [+177, +371] | 0.24 |
| B | +117 [-20, +242] | -0.45 |
| D | +140 [+30, +250] | -0.36 |

**Fee stress** (positions recomputed at the stressed fee), idea minus baseline per day:

| Fee | A | C | B | D |
|---|---|---|---|---|
| 0.50 | +365 | +273 | +157 | +128 |
| 1.00 | +467 | +380 | +234 [+35, +452] | +197 [+61, +329] |

The differences grow with the fee because the baseline trades every hour and pays the most fees.

**Zone-plus-generator ablation (idea A rule).** Adding the generator summary changed A by -103 USD/day [-282, +28]: inconclusive, and no evidence that it helps.

## How to read these numbers

- **The baseline is a weak bar in 2023.** The baseline lost money in 2023. Its trailing 365-day mean pointed to load in 47% of zone-hours, but the 2023 gap was mostly negative: mean -0.86 and median -1.52 USD/MWh.
  - A's "pays" is real under the registered test, but A's own money is +112 USD a day.
  - A plain "always supply" rule, shown only for reference and not a registered strategy, made +198 USD a day at 0.75 USD/MWh. A earns more per MWh (1.01) but trades fewer hours.
  - C "pays" mostly by staying out while the baseline lost: C's own profit is 0.09 USD/MWh, with a break-even fee of 0.19.
  - B and D lost money on their own and still scored above the baseline.
- **The models have little hourly skill.** The base model's predictions correlate 0.011 with the realized gap in 2023, against 0.111 on 2022 validation.
  - The low end ranks the hours: the lowest tenth of predictions averaged -3.1 predicted and -2.3 realized.
  - The high end does not: the top tenth predicted +11.7 (spike warnings) and realized -1.1.
  - Nothing here looks too good. The 2022 tuning gains for B and D, about +1,000 USD a day, did not carry into 2023, which is what one expects without lookahead.

## Minimum detectable effect (80% power)

| Idea | 2023 (365 days), USD/day | Projected for the held-out run (1,004 days), USD/day |
|---|---|---|
| A (alpha 0.05) | 233 | 141 |
| B (alpha 0.05/3, worst Holm level) | 343 | 207 |
| C | 232 | 140 |
| D | 267 | 161 |

The projection assumes 2023's daily variance and scales by the square root of the number of days. No held-out data was used for it.

## Deep model: pending

- **What was set up.** The deep model run chose configuration c2 on 2022 by mean squared error. Its own 2023 walk-forward of the base deep model (feeding ideas A and C) finished at 17:10, at about 75 s per refit. `rehearsal.py deep` reuses those predictions. It then walks the same configuration forward with the weather columns (B) and the outage columns (D) as extra inputs.
- **Where it stopped.**
  - B's 12 refits finished at 17:47.
  - D started at 17:47. Gene dropped off the network at about 17:56 and was still offline at 18:40.
  - The job runs detached on gene, so it has probably finished, but that is not verified.
- **To finish it, on gene:**
  1. Check that `~/nyiso-us/logs/rehearsal_deep.log` ends with `deep exit 0`. If not, run `cd ~/nyiso-us/model && ../.venv/bin/python rehearsal.py deep`.
  2. Run `../.venv/bin/python rehearsal.py rescore`. It recomputes every score from cached predictions without refitting, and adds each idea's own-P&L interval and the deep-against-gradient-boosting words.
  3. Copy `~/nyiso-us/results/rehearsal_2023.json` over this folder's copy.

## Anything that looked like a bug, and the fix

1. **Synthetic test data put every timestamp in 1970.** pandas 3 stores timestamps in micro- or milliseconds, and the generator's integer arithmetic assumed nanoseconds. The first lookahead test therefore passed vacuously. Its second check, that the label must change when the future is replaced, caught the problem. The generator is fixed.
2. **"hour" was both a key and a feature,** which duplicated a column. Fixed.
3. **The real load forecast is stored as integers.** The real-data perturbation test failed on gene for this reason. Panel features are now always float64.
4. **The weather table was stale in the first gene run.** The LockedStore loaded `weather_gfs` at 16:49. The feature pipeline rebuilt it at 16:56 with the 3-day values and the 40 and 64 hour publication rule. So the first weather features used the old table, with 2-day values for 22:00 and 23:00 under the old 42 hour rule. That run was stopped before any idea B number existed. The cache was deleted and everything was rerun in a fresh process. Any table rebuilt during a run needs a new process.
5. **The synthetic weather in `conftest.py` used the old rule.** It now uses the pipeline's `gfs_published_at`: 2-day values for hours up to 21:00 and 3-day values after.

The lookahead tests pass on the real tables. For six bid days (one after a late-rewritten real-time file, plus DST days), every value published after 05:00 on D was replaced by noise in all sources. The base, generator, weather and outage features were identical and the label changed.

## Decisions needed before the freeze

1. **The weak baseline.** Objective 6 is never edited. I propose reporting each idea's own daily P&L interval next to its word, and saying so when a "pays" rests on the baseline's losses. `score.py` now computes this interval as `idea_alone_interval_95`. It appears in the JSON after `rehearsal.py rescore`; the current JSON predates it.
2. **"doesn't pay" versus "inconclusive."** The code reads "doesn't pay" as the upper end of the corrected interval below zero, and everything else that is not "pays" as inconclusive. Please confirm.
3. **Deep against gradient boosting.** "Equivalent" (95% interval inside plus or minus the fee times the gradient-boosting version's MWh a day) is checked before better or worse. Please confirm.
4. **C's pairs.** Keep the five chosen on 2022 predictions, or choose again on the 2023 walk-forward before the freeze?
5. **Fees.** No posting was found for 2021 or 2025. Each uses the higher of its two nearest known years: 2021 uses 0.0862, and 2025 uses 0.1919. The 2025 value matters for the held-out run.
6. **The held-out run.** The lock only opens inside a git checkout whose model and pipeline match the FREEZE commit. It also refuses untracked .py files in those folders. Gene's `~/nyiso-us` is a copy, not a checkout, so the held-out run must run from a checkout of the frozen commit.
7. **Outage sites.** `features_outages.TOP_SITES` was counted on 2020 to 2023 outage starts. That includes 2023, with no prices involved, so it is mildly in-sample for D in this rehearsal only.
