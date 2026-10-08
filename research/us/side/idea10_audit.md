# Idea 10 (weather surprise) audit

6 October 2026. Adversarial audit of `idea10_weather_surprise` in `side/lab.py` on gene. Build years only: every table was filtered at load to delivery or target times before 1 January 2024 New York time, and every loader asserts it. lab.py was imported read-only and not edited. Nothing was committed, pushed or sent.

## Verdict

Idea 10 is fragile, and its passing grade rests on one storm. The arithmetic is honest: no value used for D+1 was public after 05:00 on D, and my own code reproduces 3,034, 311,142 and 37,955 to the dollar. The edge, however, is one storm in one year.

Winter Storm Elliott's two days (delivery 23 and 24 December 2022) earn 204,836 of the three-year 352,130. The second day ranked 0.932 against a 2022 cut of 0.90. That cut was chosen on seven months of 2021 ranks, where it beat 0.95 by 13,183 USD. At 0.95, 2022 reads -6,904 and the three years 34,085.

Outside 2022 the flip to virtual load has no value. Against always taking supply it gained 8,048 in 2021 (12 load days) and lost 30,199 in 2023 (86 load days). The surprise does not track the day's price gap in either year (Spearman 0.04 and -0.02).

The one-day-late placebo earns because Elliott lasted two days, not because of a leak. Its profit is mostly 24 December, which inherits the rank of 1.0 from 23 December. Shifts of 2, 3 and 7 days lose between 118,607 and 246,504, and so does the same day one year earlier (-173,124).

## 1. Which values the rule uses, and when they were public

**What "newest" and "previous" are.** The rule does not compare the newest run with the run before it. For each D+1 hour up to 21:00 it compares two values from the Open-Meteo previous-runs archive: the value predicted 48 hours before the valid time (`previous_day2`) and the one predicted 72 hours before (`previous_day3`). Open-Meteo builds `previous_day2` from forecast steps 48 onward of successive runs. For a 6-hourly model that means the latest run initialised at or before valid time minus 48 hours, and `previous_day3` uses steps 72 onward (Open-Meteo, "Weather forecasts from previous model runs"). Under that construction, every day2 value the rule uses has a step of 48 to 53 hours, and every day3 value a step of 72 to 77. The two runs compared for an hour are therefore 24 hours apart, so the surprise is a one-day forecast revision at a fixed valid time.

For delivery 24 December 2022 the day2 values come from five runs: 00Z on 22 December for 00:00, up to 00Z on 23 December for 21:00. The newest run public at 05:00 on D (00Z on D) enters only the last two or three hours of the day. This is stale, not lookahead. The lab's one-line description ("the latest GFS run, against the run before it") should be corrected.

**Model.** The request asked for `gfs_seamless`, and the archive answers with HRRR grid coordinates. The values themselves are pure GFS. I fetched `gfs_global` for Albany and New York City for all of 2021, 2022 and 2023 and compared both leads. All 96,912 stored hours are identical, and HRRR's archive is empty in 2022. This holds for the build years only; see the risk at the end.

**Publication.** Measured over the 244,145 point-hour pairs the rule used in 2021 to 2023:

| Check | Result |
|---|---|
| Rows whose `published_at` differs from the rule (target minus 40 h for day2, minus 64 h for day3), all pre-2024 rows | 0 |
| Latest day2 `published_at` against 05:00 on D | 0.0 h (hours 20:00 and 21:00 sit on the boundary by construction) |
| Latest day3 `published_at` against 05:00 on D | 24 h before |
| Slack between the actual run (initialisation plus 5 h for the full GFS run) and 05:00 on D, day2 | at least 4 h (hours 19:00 to 21:00); 10 h or more for hours up to 18:00 |
| Same slack, day3 | at least 28 h |
| Local hours used | 00:00 to 21:00 only |
| Decision time | `common.decision_time`, 05:00 New York clock time; the lab asserts the hour is 5 on every day |

**DST days.** On the fall-back delivery days (7 November 2021, 6 November 2022, 5 November 2023), 21:00 EST has a day2 value published at 06:00 EDT on D. The filter drops it, and the last hour used is 20:00 EST; both 01:00 hours are used (242 pairs). On the spring-forward delivery days, 21 hours give 231 pairs. On the day after each switch, as on every ordinary day, the 21:00 value is public at exactly 05:00 and is used, which the rule allows. 14 and 15 March 2021 have no pairs because the archive starts on 25 March 2021.

**Injected lookahead.** I tested 13 delivery days: an ordinary July day, both sides of the 2021 fall-back and of both 2022 switches, 16 January 2022, 23 to 25 December 2022, 11 January 2023 and 5 September 2023. Each injection was run through the lab's own `weather_surprise` and through my code, and pushed the day toward the other side.

| Injection | Rule broken? | Decisions changed, lab | Decisions changed, mine |
|---|---|---|---|
| Every row published after 05:00 on D, plus 40 C | no | 0 of 13 | 0 of 13 |
| A fake newer run (24 h lead) at its true time, target minus 16 h | no | 0 of 13 | 0 of 13 |
| The same fake run labelled public at 05:00 | yes | 13 of 13 | 0 of 13 |
| A future value written into the day2 rows themselves | yes | 13 of 13 | 13 of 13 |
| Filter opened 24 h, fake run present | yes | 13 of 13 | 0 of 13 |

Decisions change only when the rule is broken. My code ignores the fake run because it reads the two leads by name, while the lab takes the two newest rows by `published_at`; both are correct for this table.

**Does the timing slack matter?** I used only the earlier hours of D+1, so every value is older:

| Hours used | Slack to 05:00 (rule) | 2021 | 2022 | 2023 | Three-year | Without Elliott |
|---|---|---|---|---|---|---|
| up to 21:00 (claimed) | 0 h | 3,034 | 311,142 | 37,955 | 352,131 | 147,294 |
| up to 15:00 | 6 h | -4,475 | 282,621 | 20,922 | 299,068 | 94,232 |
| up to 09:00 | 12 h | -10,549 | 70,358 | 40,971 | 100,780 | -104,056 |
| up to 03:00 | 18 h | -1,468 | 105,535 | 39,717 | 143,784 | 52,603 |

Elliott is flagged in every variant (only 23 December is missed with hours up to 03:00), so the result does not depend on the last few hours of slack. What it does depend on is the storm: the profit outside Elliott swings from -104,056 to 147,294 between nearly equivalent definitions.

## 2. Independent re-implementation

`idea10_audit_lib.py` uses its own weather loader, an explicit join of the 48 h and 72 h leads, a plain-loop trailing rank, the storm audit's price book, and its own walk-forward. Against the lab's function: surprise identical on all 1,009 days with a value, pair counts identical on all 1,461 days, rank difference 0.0.

| Net USD | 2021 | 2022 | 2023 | Three-year |
|---|---|---|---|---|
| Claimed | 3,034 | 311,142 | 37,955 | 352,130 |
| Audit | 3,034 | 311,142 | 37,955 | 352,131 (rounding) |
| Cut used | 0.95 (declared default) | 0.90 (best on 2021) | 0.80 (best on 2022) | |
| Load days | 12 | 34 | 86 | |
| Always supply | -5,015 | -198,269 | 68,154 | -135,130 |
| Gain over always supply | 8,048 | 509,411 | -30,199 | 487,260 |

Sharpe 1.03, as claimed. Choosing each cut without 31 December of the prior year gives the same cuts. Ranks exist only from delivery 25 May 2021, so the first five months of 2021 are plain supply.

## 3. Concentration

| Year | Net | Without best 1 day | Without best 3 | Without best 10 |
|---|---|---|---|---|
| 2021 | 3,034 | -7,779 | -24,128 | -51,279 |
| 2022 | 311,142 | 163,106 | 86,263 | -17,265 |
| 2023 | 37,955 | 21,881 | 574 | -34,593 |

**What drives 2022.** The two Elliott days, both traded as load: 23 December (surprise +2.83 C, rank 1.000, +56,801) and 24 December (surprise +0.79 C, rank 0.932, +148,035). On 23 December the 48 h runs brought the Arctic front in earlier than the 72 h runs. At Albany, 18:00 was forecast at -7.9 C against +1.4 C the day before. The next best days are 13 June and 8 August (about 20,000 each) and 4 January (14,525). The rule also went load on 27 December, the day the prices fell back (-21,538).

Without Elliott's two days: 3,034, 106,306 and 37,955, three-year 147,294, Sharpe 0.73. The average of 49,098 a year misses the 50,000 bar by 902 USD. Most of what remains is again 2022: there the 32 other load days gained 99,630 over always supply. In 2021 and 2023 together the flip lost 22,151 against supply.

**Every cut, fixed for all years.** The chosen cut is marked with *.

| Cut | 2021 | 2022 | 2022 without Elliott | 2023 | Load days 2021 / 2022 / 2023 |
|---|---|---|---|---|---|
| 0.70 | 35,713 | 264,904 | 60,068 | 29,232 | 71 / 103 / 115 |
| 0.75 | 11,465 | 375,780 | 170,944 | 29,322 | 57 / 79 / 99 |
| 0.80 | 3,937 | 373,759 | 168,923 | 37,955* | 45 / 63 / 86 |
| 0.85 | 7,231 | 318,442 | 113,606 | 22,101 | 28 / 45 / 64 |
| 0.90 | 16,217 | 311,142* | 106,306 | 39,780 | 19 / 34 / 41 |
| 0.925 | 12,483 | 343,286 | 138,450 | 2,151 | 17 / 27 / 29 |
| 0.95 | 3,034* | -6,904 | 84,384 | 29,518 | 12 / 20 / 18 |
| 0.98 | -8,951 | -50,131 | 41,158 | 42,453 | 3 / 9 / 4 |
| Always supply | -5,015 | -198,269 | 6,676 | 68,154 | 0 |

2022 has a cliff between 0.925 and 0.95, where 24 December (rank 0.932) leaves the load side. In 2023 no fixed cut beats always supply (68,154).

**Neighbouring cuts on the registered grid** (0.80, 0.90, 0.95, 0.98):

| Change | Cuts 2021 / 2022 / 2023 | Three-year |
|---|---|---|
| Chosen | 0.95 / 0.90 / 0.80 | 352,131 |
| All one step higher | 0.98 / 0.95 / 0.90 | 23,925 |
| Only 2022 one step higher | 0.95 / 0.95 / 0.80 | 34,085 |
| All one step lower | 0.90 / 0.80 / 0.80 | 427,931 |

The 2022 choice rested on 19 load days in 2021: 0.90 earned 16,217 there against 3,937 at 0.80 and 3,034 at 0.95. The lab's FRAGILE flag reads "no" only because its test asks for a sign change. One step up on the grid removes 93 percent of the total.

## 4. Why the one-day-late placebo earns a third

| Placebo (score from) | Own walk-forward, three-year | With the real cuts | With real cuts, without Elliott | Elliott days flagged |
|---|---|---|---|---|
| Real rule | 352,131 | 352,131 | 147,294 | 23 and 24 Dec |
| 1 day earlier | 120,955 | 130,250 | 39,071 | 24 Dec only |
| 2 days earlier | -197,547 | -262,572 | -57,627 | none |
| 3 days earlier | -246,504 | -235,017 | -30,072 | none |
| 7 days earlier | -118,607 | -184,206 | 20,739 | none |
| Same calendar day a year earlier | -173,124 | -228,777 | -23,832 | none |
| Shuffled within each year, 500 draws | mean -123,031; 0.2 percent at or above the real rule | | | |
| Shuffled within each month, 500 draws | mean -69,317; 0.8 percent at or above | | | |

This is persistence, not a leak. A score one day late uses only information public 24 hours earlier, so it cannot contain more lookahead than the real rule. Its profit is almost entirely 24 December 2022, which takes the rank of 1.0 from 23 December and earns 148,035 while 23 December loses 56,855 on supply. Without those two days it earns 39,071. The surprise is barely persistent: its autocorrelation is 0.16 at a lag of one day, 0.04 at two and zero from three. Correspondingly, every placebo from two days onward loses. The year-earlier placebo shows that the score is not a season detector.

The shuffles beat the rule only in 2021 and 2023. Within-month shares at or above the real rule are 39 percent in 2021, 0.6 percent in 2022 and 67 percent in 2023. Day-level skill therefore shows only in 2022. The correlation of the daily surprise with the daily mean price gap is 0.03 in 2021, 0.20 in 2022 (0.10 without Elliott) and -0.07 in 2023.

## 5. Costs

The lab's costs match `model/fees.py`. Supply legs pay 0.0947, 0.1043 and 0.1496 USD per MWh in 2021 to 2023 (Rate Schedule 1, FERC fee and uplift bound). Load legs pay 0.0907, 0.1013 and 0.1236 (Rate Schedule 1 and FERC fee). The book is 264 MWh a day on either side. Cuts are chosen at base cost; choosing them at the stressed cost picks the same cuts.

| Net USD | 2021 | 2022 | 2023 | Three-year | Without Elliott |
|---|---|---|---|---|---|
| Base | 3,034 | 311,142 | 37,955 | 352,131 | 147,294 |
| 0.50 stress | -36,034 | 272,985 | 3,600 | 240,551 | 35,926 |
| 1.00 stress | -84,214 | 224,805 | -44,580 | 96,011 | -108,350 |

Break-even fee: 1.33 USD per MWh over the three years, 0.63 without Elliott, 0.13 in 2021 and 0.54 in 2023. At 1.00 the rule is positive only in 2022.

Stationary bootstrap (mean block 7 days, 5,000 draws), 95 percent interval of the yearly net: -16,853 to 320,407; without Elliott -29,986 to 127,153. The gain over always supply without Elliott is -43,801 to 99,087. All three include zero.

The fees are not the whole cost of a load leg. A load bid on a storm day must be priced near the cap to be sure to clear, and it carries collateral. That cost of capital is inside the stress costs, but the price-taking assumption is most generous on exactly the two days that make the result.

## 6. Verdict and how it fails

**Fragile, and effectively an artefact of Winter Storm Elliott.** The timing is clean and the information is real: the 48 h runs did turn sharply colder for 23 December, and a rule allowed to act on that would have caught Elliott. But the evidence is one storm. The second day cleared the cut by 0.03 of rank, under a cut chosen on 19 load days. Without the storm the rule beats always supply only in 2022, and the score does not track prices in 2021 or 2023.

**Most likely failure on unseen years.** In years without a storm of that size, ordinary forecast revisions trip the flag on 40 to 90 days, and the load legs pay the usual real-time-below-day-ahead carry. That is 2023: 86 load days lost 18,201 where supply would have made 11,998, so the year trailed always supply by 30,199. Its 37,955 was the supply carry, which falls to 3,600 at the 0.50 stress. When a storm does come, it has to land above a cut set the year before. At the next grid step, 0.95 instead of 0.90, 2022 would have been -6,904.

**One input risk for the held-out run, unverified by design.** The archive answers `gfs_seamless` with HRRR's grid, and Open-Meteo says most models are archived only from January 2024. HRRR's extended runs reach exactly 48 hours, so from 2024 the day2 values (but not day3) may start to blend in HRRR. The surprise would then partly measure the gap between two models instead of a forecast revision. I did not read any 2024 value to check this. Fetching with `models=gfs_global` gives identical values in 2021 to 2023 (96,912 of 96,912 hours checked, two points) and stays one model in every year, so it removes the risk without changing any build-year result.

If idea 10 goes on the held-out list, it should be frozen as it is (cut grid, default 0.95, hours up to 21:00, mean over the 11 primary points), reported beside always supply, and read as a bet on catching a storm rather than as a daily edge.

## Files

- Laptop `~/projects/tavle/research/us/side/`: `idea10_audit_lib.py` (independent implementation, holdout filter at load), `idea10_audit_timing.py` (publication times, DST days, injections through the lab's function and mine), `idea10_audit.py` (re-implementation, concentration, cuts, hour cutoffs, placebos, costs, bootstrap), `idea10_audit_timing.json`, `idea10_audit.json`, this file.
- Gene `~/nyiso-us/side_audit/`: the same scripts and JSON, `idea10_audit.log`, and `idea10_gfs_global/` (the `gfs_global` comparison files, 2021 to 2023 only). Run with `cd ~/nyiso-us/side_audit && ../.venv/bin/python -I idea10_audit_timing.py`, then `idea10_audit.py`. They need `storm_audit_lib.py` in the same folder.
- Source on the archive's construction: Open-Meteo, "Weather forecasts from previous model runs" (openmeteo.substack.com), and the Previous Runs API page (open-meteo.com/en/docs/previous-runs-api).
