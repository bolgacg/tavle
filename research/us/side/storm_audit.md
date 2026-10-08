# Storm strategies audit

6 October 2026. Audit of `side/storm_value.py` (storm-day filter and storm-day flip). Build years only: every table was filtered to delivery or target hours before 1 January 2024 at load, and each loader asserts it. Nothing was committed, pushed or sent.

## Verdict

The numbers are honest arithmetic: no lookahead changes them, and an independent re-implementation reproduces all six to the dollar. Both strategies still fail as evidence of an edge.

The storm-day filter is fragile. Its 2022 profit exists only because Winter Storm Elliott's two loss days scored 0.904 and 0.901 against a cut of 0.90. The prior year preferred that cut over 0.95 by about 2,000 USD. Had those two days been traded, 2022 would read minus 155,290 instead of plus 49,655. In 2023 the filter earned less than always taking supply at every cut tried.

The storm-day flip is an artefact of one storm. Elliott's two days supply 204,836 of its 2022 profit. Outside those two days, the virtual load legs on storm days earned 8,515 USD in total over 2021 to 2023.

## 1. Lookahead

Every row each input used was checked against the published_at rules in `pipeline/README.md`, using the true 05:00 New York deadline.

| Input | Rule | Rows used, bid days 2020 to 2023 | Published after 05:00 on D | What else was checked |
|---|---|---|---|---|
| NYISO peak load forecast for D+1 | zip write time | 35,016 | 0 | File named D (written about 07:05 on D-1) on 1,450 days. File D-1 on 9 days, when file D was written after 05:00. File D+1 never. Every hour of D+1 covered, 23 and 25 on DST days. |
| GFS coldest and hottest for D+1 | target minus 40 h (day2), minus 64 h (day3) | 510,785 | 0 | published_at equals the rule on every row. Only D+1 targets are used. |
| Mean RT minus DA of D | hour end + 15 min, or the rewrite time | 59,521 | 0 | 4 hours a day (00:00 to 03:59; 3 or 5 on DST days). No revised row used. 106 bid days have no RT input because file D was rewritten later. |

The script mixes both GFS leads: day2 for hours up to 21:00, and day3 for every hour. That is allowed by published_at. Using the pipeline's three-day rule instead (one value per point and hour) moves the results slightly: filter 40,453, 45,535 and 15,185; flip 76,325, 282,065 and minus 49,753. On the day before a spring-forward day, the day2 value for 22:00 of the spring-forward day is public at exactly 05:00 (22 rows), which the `<=` rule permits.

**One timing bug, with no effect on the results.** The script sets the deadline as local midnight plus 5 elapsed hours. On a spring-forward day that is 06:00 EDT, so one hour of lookahead gets in. It admits 33 RT rows (the 04:00 hour, 11 zones on 3 of the 4 spring-forward days; 13 March 2022 had no usable RT) and 22 GFS rows (22:00 of D+1, published 06:00 EDT). On a fall-back day the script's deadline is 04:00 EST, one hour early, which is conservative. The storm score changes only for delivery 9 March 2020 (0.459 against 0.377), below every cut. No delivery day in 2021 to 2023 changes side, and all six yearly figures are identical with the true deadline. Fix: build 05:00 as wall-clock time, as `common.decision_time` does.

**Injection test.** I ran 65 injections on 7 bid days: an ordinary July day, both sides of each 2022 DST switch, the Elliott bid day (23 December 2022), and 16 January 2022, whose RT file was rewritten. Each was run through the script's own per-day code, extracted verbatim from the file on gene, and through my code. The injections were the isolf file D+1, late day2 GFS rows, the RT hours from 04:00 and 05:00 of D, the target day's own DA and RT, GFS rows for D+2, and a rewritten RT day.

- When the injection respected the rule, the score stayed the same. The one exception in the script's code is the DST bug: 13 March 2022, where the 22:00 GFS row published at 06:00 EDT entered. In both codes the 05:00 boundary row described above also entered, as it should.
- When the injection broke the rule (a row relabelled as public before 05:00), the score moved to 1.0. The one exception is the script on the fall-back day (6 November 2022): it ignored a row labelled 04:59 because its deadline there is 04:00.
- Rows for D+2 and the target day's own prices never moved the score, so the D to D+1 mapping is right.

**Selecting the cut on the prior year.** The cut is chosen on the whole prior year, including 31 December, whose prices were not public at 05:00 on 31 December, when the 1 January bid is made. Choosing on the prior year through 30 December gives the same cuts in every case.

**An assumption I could not test.** Every RT value used comes from the daily file NYISO wrote at about 23:57 on D, after the deadline. The pipeline rule assumes the hourly price posted at hour end equals that value, unless the file was rewritten after midnight. Removing the RT component entirely leaves the 2021 to 2023 totals at 96,726 for the filter and 312,012 for the flip, so the conclusions do not rest on that assumption.

**Holdout hygiene.** The script loads the full load-forecast and GFS tables (2024 to 2026 rows included) and relies on its per-day target window. No 2024 value enters any computation, but the tables should be filtered at load.

## 2. Independent re-implementation

My code (`storm_audit_lib.py`) is vectorised rather than a per-day loop, takes 05:00 as wall-clock time, and computes trailing ranks with sliding windows. The book has a price for every zone-hour of 2020 to 2023 (0 of 385,704 rows missing an RT price), so no cost is skipped.

| Net USD | 2021 | 2022 | 2023 |
|---|---|---|---|
| Always supply, claimed and audit | -5,015 | -198,269 | 68,154 |
| Storm-day filter, claimed | 41,620 | 49,655 | 15,866 |
| Storm-day filter, audit | 41,620 | 49,655 | 15,866 |
| Storm-day flip, claimed | 78,415 | 290,469 | -48,392 |
| Storm-day flip, audit | 78,415 | 290,469 | -48,392 |

The cuts (0.80, 0.90, 0.80), the days sat out (201, 131, 166), the best days and the 0.50 stress figures also match. The day-level inputs match the script's own loop on all 1,459 bid days.

## 3. Selection and fragility

**Every cut, fixed (no selection).** The cut the script chose for each year is marked with *.

| Filter | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 0.98 | Always supply |
|---|---|---|---|---|---|---|---|---|
| 2020 | 3,295 | 352 | -7,107 | -5,110 | -12,483 | -8,867 | -17,346 | -24,609 |
| 2021 | 26,403 | 37,590 | 41,620* | 34,583 | 42,385 | 40,323 | 5,860 | -5,015 |
| 2022 | 53,145 | 64,470 | 86,844 | 95,606 | 49,655* | -125,714 | -131,350 | -198,269 |
| 2023 | 3,280 | 14,441 | 15,866* | 13,774 | 28,941 | 37,444 | 52,589 | 68,154 |

| Flip | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 0.98 |
|---|---|---|---|---|---|---|---|
| 2020 | 23,144 | 18,310 | 4,496 | 8,912 | -4,887 | 3,770 | -11,820 |
| 2021 | 46,022 | 69,230 | 78,415* | 66,053 | 83,566 | 81,794 | 15,070 |
| 2022 | 290,340 | 314,511 | 360,996 | 380,202 | 290,469* | -57,066 | -66,818 |
| 2023 | -80,198 | -54,126 | -48,392* | -49,114 | -15,393 | 4,426 | 35,942 |

2022 has a cliff between 0.90 and 0.95, and the prior-year choice landed on its edge. Elliott's two loss days (delivery 23 and 24 December 2022) lost 56,855 and 148,090 USD on supply. Both scored just above 0.90, through the cold component alone: 0.9041 and 0.9014. For 24 December, 329 of the 365 prior days had a milder forecast minimum. One more colder day in that window would have put it below the cut. On 2021, the cut 0.90 beat 0.95 by 2,062 USD for the filter and 1,772 for the flip. Choosing from the seven-cut grid instead gives filter 26,403, 49,655 and 13,774, and flip 46,022, 290,469 and minus 49,114.

There is also a selection issue the prior-year cut cannot fix. The idea itself (four components, the maximum, a 365-day window, flipping to load) was formed after the 2020 to 2023 prices, Elliott included, had been seen. 2022 is therefore not out of sample for the design.

**Components and window** (cut chosen on the prior year from the script's grid; 2021 to 2023 sum in the last column).

| Score built from | Filter 2021 | 2022 | 2023 | Sum | Flip 2021 | 2022 | 2023 | Sum |
|---|---|---|---|---|---|---|---|---|
| All four (claimed) | 41,620 | 49,655 | 15,866 | 107,141 | 78,415 | 290,469 | -48,392 | 320,492 |
| Load alone | 6,379 | -158,199 | 77,703 | -74,117 | 17,332 | -120,299 | 86,459 | -16,508 |
| Cold alone | 8,893 | 49,107 | 29,348 | 87,348 | 19,715 | 293,606 | -10,684 | 302,637 |
| Heat alone | 18,777 | -158,283 | 73,111 | -66,395 | 39,583 | -120,468 | 77,131 | -3,754 |
| RT alone | 37,140 | -191,431 | 74,956 | -79,335 | 74,596 | -188,446 | 80,100 | -33,750 |
| Without load | 33,318 | 67,266 | 14,586 | 115,170 | 62,691 | 326,233 | -50,591 | 338,333 |
| Without cold | 40,669 | -185,953 | 73,427 | -71,857 | 78,618 | -181,671 | 77,041 | -26,012 |
| Without heat | 34,225 | 99,752 | 15,592 | 149,569 | 64,212 | 387,408 | -48,002 | 403,618 |
| Without RT | 17,941 | 71,482 | 7,303 | 96,726 | 39,721 | 335,644 | -63,353 | 312,012 |
| 180-day window | 32,773 | -1,895 | -1,938 | 28,940 | 54,468 | 179,770 | -80,398 | 153,840 |

Every variant that keeps the cold component catches Elliott and looks good. Every variant without it loses money in 2022 and over the three years. The 180-day window removes most of the filter's profit (28,940 against 107,141).

**Without the best days** (each strategy's own best days removed from that year).

| | Net | Without best 1 | Without best 3 | Without best 10 |
|---|---|---|---|---|
| Filter 2021 | 41,620 | 30,807 | 17,617 | -2,869 |
| Filter 2022 | 49,655 | 28,171 | 7,060 | -31,445 |
| Filter 2023 | 15,866 | 11,475 | 3,980 | -11,901 |
| Flip 2021 | 78,415 | 67,603 | 50,887 | 16,457 |
| Flip 2022 | 290,469 | 142,434 | 50,089 | -70,276 |
| Flip 2023 | -48,392 | -69,001 | -93,733 | -125,999 |

**What the storm days are.** At the chosen cuts the score is on for 201, 131 and 166 days a year, so a "storm day" is not rare. If the four ranks were independent, a cut of 0.80 would be crossed on 59 percent of days by chance alone. In 2021 the share rises from 33 percent in the first half, before the weather ranks exist, to 77 percent in the second. The score works mostly as a season detector: it was on 83 to 97 percent of days in June to August and November to December 2021, and on 0 to 3 percent in September to November 2022.

**Placebo.** I sat out (or flipped) the same number of days in each calendar month, chosen at random, 4,000 times per year.

| | 2021 | 2022 | 2023 |
|---|---|---|---|
| Filter, actual | 41,620 | 49,655 | 15,866 |
| Random days in the same months, median | 27,936 | -120,391 | 29,184 |
| Share of random draws at or above actual | 0.15 | 0.04 | 0.73 |
| Flip, actual | 78,415 | 290,469 | -48,392 |
| Random days in the same months, median | 51,137 | -44,122 | -23,258 |
| Share of random draws at or above actual | 0.14 | 0.05 | 0.72 |

Choosing days within a month beats chance only in 2022, the Elliott year. A plain calendar rule (sit out June to August and December to February) does worse than the score: filter minus 35,369, 12,292 and minus 16,332. So the score holds some seasonal information, but no day-level skill shows outside 2022.

**Where the money comes from.**

- On the days the filter sat out, supply would have earned minus 46,635, minus 247,924 and plus 52,288. Without Elliott's two days, the filter avoided 37,326 USD of supply losses over the three years, about 12,000 a year.
- The flip's load legs on those days earned 36,795, 240,814 and minus 64,258. Without Elliott's two days that is 8,515 over three years.

**Bootstrap** (stationary, mean block 7 days, 2021 to 2023 pooled). The 95 percent intervals:

- Filter net 107,141: minus 62,801 to 284,046. Its gain over always supply, 242,272: minus 83,513 to 688,997.
- Flip net 320,492: minus 72,302 to 865,882. Its gain over always supply, 455,623: minus 196,725 to 1,339,591.

Every single-year interval also includes zero.

## 4. Costs and mechanics

The script's costs match `model/fees.py`. Supply legs pay 0.0947, 0.1043 and 0.1496 USD per MWh in 2021 to 2023 (Rate Schedule 1, FERC fee and the uplift bound). Load legs pay 0.0907, 0.1013 and 0.1236 (Rate Schedule 1 and FERC fee). Each cut is chosen at base cost and then stressed, as in the script.

| Net USD | 2021 | 2022 | 2023 |
|---|---|---|---|
| Always supply, base / 0.50 / 1.00 | -5,015 / -44,070 / -92,250 | -198,269 / -236,399 / -284,579 | 68,154 / 34,389 / -13,791 |
| Filter, base / 0.50 / 1.00 | 41,620 / 24,077 / 2,434 | 49,655 / 25,210 / -5,678 | 15,866 / -2,547 / -28,820 |
| Flip, base / 0.50 / 1.00 | 78,415 / 39,148 / -9,032 | 290,469 / 252,236 / 204,056 | -48,392 / -83,296 / -131,476 |

At 1.00 USD per MWh the filter is about zero or negative in every year. The flip survives only in 2022.

Virtual load at the 11 NYISO load zones is a standard position: virtual transactions clear at the load zones on both sides, up to 999 MW per bus per hour. It pays Rate Schedule 1 and the FERC fee but no forecast-pass uplift. In NYISO's training material, "Virtual Load bids do not add to these uplift costs" (cost audit, section 2). It adds no per-MWh charge the script misses.

What the flip does add is collateral and clearing risk. Virtual load has its own credit requirement, the 97th percentile of its group's loss, held for two days. On switch days a supply book and a load book can both be outstanding. On storm days the day-ahead price is high, so a price-taking load bid must be priced near the bid cap to be sure to clear, and it then carries the whole DA minus RT loss if the storm passes. On 25 and 26 December 2022, for example, the flip's load legs lost 11,349 and 19,183. The cost of capital on collateral is already inside the 0.50 and 1.00 stress costs.

## 5. Verdict

**Storm-day filter: fragile.** It is not a timing artefact, and sitting out stressed days does cut drawdown (2022: minus 87,791 against minus 277,995 for always supply). Its profit, however, rests on one storm landing above a cut the prior year chose by about 2,000 USD. It trails always supply at every cut in 2023, disappears with a 180-day window or at 1.00 USD per MWh, and its intervals include zero. Most likely failure on unseen years: the storm that matters scores just below the cut, or no storm comes. The filter then misses the one loss day while giving up the calm supply profit on 35 to 55 percent of days, as in 2023.

**Storm-day flip: an artefact of Winter Storm Elliott.** Two days supply 204,836 of its 2022 profit. Its load legs earn 8,515 over three years outside them, and it loses in 2023 at every cut up to 0.90. Most likely failure on unseen years: no Elliott-sized real-time spike lands above the cut, so the flip pays the usual real-time-below-day-ahead carry on 130 to 200 load days a year. In 2023 its load legs lost 64,258.

If either strategy goes on the list for the held-out run, its components, window and cut rule should be frozen now and reported beside always taking supply, not tuned again.

## Files

- Laptop `~/projects/tavle/research/us/side/`: `storm_audit_lib.py` (independent implementation, holdout filter at load), `storm_audit_timing.py` (lookahead checks and injections), `storm_audit_fragility.py` (cuts, components, window, costs, placebo, bootstrap), `storm_audit_timing.json`, `storm_audit_fragility.json`.
- Gene `~/nyiso-us/side_audit/`: the same scripts, the JSON results and logs. Run with `cd ~/nyiso-us/side_audit && ../.venv/bin/python -I storm_audit_timing.py`, then the same with `storm_audit_fragility.py`.
