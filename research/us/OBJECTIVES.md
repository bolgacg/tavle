# Objectives v2, APPROVED by Bo on 6 Oct 2026 ("plan is approved"). Fixed before any model run.
# Any later change is a dated addendum at the bottom, never an edit above the line.

1. Question. Using only what is public by 05:00 New York time the day before delivery, which of four ideas turns New York's day-ahead to real-time price gap into profit after costs for a 1 MW virtual position? Positions are assumed to clear at the market price.
   A (main test). Take the steady downward gap, and sit out the hours a model flags as spike risk.
   B. Use public weather forecasts to predict where the operator's load forecast runs wrong, and trade on it.
   C. Bet up in one zone and down in another, to trade the repeating congestion patterns.
   D. Trade the congestion that planned transmission outages make predictable.
2. Data. Day-ahead and real-time prices for the 11 tradable zones, January 2020 to September 2026, with generator and border points as inputs only, each used only while it existed. The operator's load forecast issued before the deadline. Archived weather forecasts for B. Outage notices for D. Every input carries its publication time; a test fails the build if anything was published after 05:00 New York time.
3. Models. Baseline: always take each zone-hour's usual side (average gap over the trailing 365 days). Gradient boosting. A PyTorch sequence model averaged over three seeds, trained on one 6 GB consumer GPU, used inside each idea where it fits. All tuning on 2020 to 2023 only, with a rehearsal on 2023. Code and settings frozen in a public commit before the held-out run. Anything not frozen by 10 Oct 23:59 is reported as not run.
4. Test. January 2024 to September 2026, run once. Each month predicted only from data up to two days earlier. Any rerun after a bug fix is published beside the original with the difference.
5. Score. Each idea against the baseline: daily net profit after each year's NYISO virtual charge, 95% interval from a bootstrap over whole days. Primary: A. Secondary in order B, C, D, with Holm correction. Ranking table: profit per MWh, break-even fee, worst month, results without the most extreme 1% of hours; costs stressed at 0.50 and 1.00 USD per MWh. Deep against gradient boosting reported inside every idea.
6. Bar (never edited). Each idea gets one word: pays, doesn't pay, inconclusive, or not run. Pays = lower end of its corrected interval above zero. Deep against gradient boosting: better, worse, equivalent (interval inside plus or minus one fee per MWh), or inconclusive.
7. Delivery. 6 Oct plan page live. 8 Oct data, timing test, data checks for B and D. 9 Oct baselines and rehearsal. 10 Oct freeze. 11 Oct single held-out run. From 13 Oct every idea that ran publishes next-day positions before 05:00 New York time, labelled a pipeline test, until an interview. 14 Oct page complete.

---
## Addenda (dated; written before any model has run)

6 Oct 2026, data checks for B and D (done the same day, before any model):
- B: NYISO's own load-forecast weather file (lfweather) is written about 08:00 New York time and its forecast for the delivery day first appears after the 05:00 deadline, so it is not used as a bid input. B uses archived GFS forecasts (Open-Meteo previous-runs archive), taking the run issued two days before each target hour; available from 2021, so B's building window is 2021 to 2023.
- D: NYISO outSched, a daily snapshot of scheduled transmission outages (about 100 lines), written about 09:40 New York time the day before its date, archived from January 2020. Scheduled, not actual: some outages are cancelled.
- Disclosure: before the freeze, a reviewer scanned descriptive statistics across all years, 2024 onward included (real-time below day-ahead in most zone-years; NYC 2024 always-supply earned 0.00 USD per MWh, 1.74 without its worst 1% of hours). These motivated idea A. No model has seen 2024 onward.

6 Oct 2026, evening, data layer built (before any model):
- Timing test: on 500 random bid days from 2020 to 2026, no row used for delivery day D+1 was published after 05:00 New York time on day D. Lookahead rows injected on 20 of those days (the next day's load forecast file, the target day's own day-ahead price, the real-time price for the hour starting 05:00, the next day's outage snapshot, late weather values) were all caught. 30 of 30 tests pass.
- B's weather archive has values from 25 March 2021, not 1 January 2021; B's building window starts there.
- Weather timing: the archive's "two days before" value is the forecast made 48 hours before the valid time. To stay safe even if the nearest model run is up to three hours later and takes up to five hours to publish, B uses the two-day value only for delivery hours up to 21:00 and the three-day value for 22:00 and 23:00.
- Real-time price corrections: NYISO rewrote 18 to 52 zone files a year after the day. A corrected day's prices count as public only from the rewrite time, so on those days the previous day's real-time prices are not used at 05:00.

6 Oct 2026, night, modelling contract (before any model ran; research/us/model/CONTRACT.md):
- Each idea's verdict uses gradient boosting; the deep model runs inside every idea in its place for the secondary comparison.
- Positions: A is supply only (supply when the predicted gap is at or below minus the fee, else none). B and D trade both sides (supply at or below minus the fee, load at or above plus the fee, else none). C trades up to five zone pairs chosen on the build years, when the predicted difference is at least twice the fee.
- The holdout is locked in code: every loader refuses delivery dates from 1 January 2024 unless a freeze commit is named and the model code is unchanged since it.

6 Oct 2026, late evening, after the 2023 rehearsal (2023 is a build year; no held-out data has been read):
- The rehearsal is published as it ran (results/rehearsal_2023.md). Gradient boosting's predicted gap barely tracked the real gap in 2023 (hourly correlation 0.011, against 0.111 on 2022), its spike warnings did not come true, and idea A under the regression rule earned 112 USD a day on its own, less than always taking supply (198). The deep model had no skill either.
- Idea A's rule is changed to match its own wording ("sit out the hours a model flags as spike risk"): both models now predict the probability that an hour is a spike, meaning real time above day ahead by at least a threshold, and A takes virtual supply in every zone-hour except those whose spike probability exceeds a cut. The spike threshold and the cut are chosen on 2020 to 2022 only, then rehearsed on 2023. B, C and D are unchanged.
- Verdict wording, made exact: "doesn't pay" means the upper end of the corrected interval is below zero. For deep against gradient boosting, better and worse are checked first; equivalent applies only when the interval contains zero and lies inside plus or minus one fee per MWh.
- Shown beside every verdict as context, not as tests: each idea's own net profit with its interval; idea A against always taking supply (the value of the spike filter); and, once the cost and collateral audit is in, return on a bankroll fixed before the freeze and the Sharpe ratio, beside the S&P 500 and a Danish bank deposit rate over the same months.
- Audit fixes before the freeze: a verdict could read "pays" when the Holm step had not rejected (score.py), now consistent; for the rehearsal, idea D's 20 most frequent outage sites are counted only on data before 2023.

6 Oct 2026, evening, after the cost and capital audit (results/cost_capital_audit.md; before the freeze, no held-out data read):
- Fees corrected from NYISO postings: Rate Schedule 1 is charged once per cleared day-ahead virtual MWh; 2021 is 0.0757 USD (not 0.0862) and 2025 is capped at 0.1666 (not 0.1919) by the tariff's 25 percent annual limit. The base cost per MWh now also includes the FERC annual charge spread over virtual MWh (about 0.01 to 0.02) and, for virtual supply only, an upper bound for the uplift virtual supply can be charged (0.003 to 0.03). The stress costs of 0.50 and 1.00 USD per MWh are unchanged and already cover cost of capital on collateral.
- Bankroll for the return comparison, fixed now from 2020 to 2023 only: 500,000 USD (NYISO's 200,000 capitalization deposit for an entity without 1 million in audited net worth, plus about 90,000 of collateral for two days of the full book in the worst build-year month, plus 195,000 for the full book's worst 12-day loss, around Winter Storm Elliott). Return on bankroll = held-out annual net profit / 500,000.
- Because any return on bankroll depends on the bankroll chosen, two bankroll-free measures are reported beside it: the annualised Sharpe ratio of daily net profit and the return over maximum drawdown, next to the S&P 500 over the same months (total return) and the Danish bank deposit rate and Nationalbank current-account rate over the same months.

6 Oct 2026, night, a gate before the held-out run (Bo's rule, before any held-out data is read):
- The held-out years stay unread until the build years show something worth testing. Every strategy is first scored out of sample inside 2020 to 2023 (a walk-forward over 2021 to 2023, each month predicted only from earlier data), with the corrected costs, and the full list is approved in writing before the freeze.
- The held-out run then includes every strategy on that list, the strong, the weak and the basic ones (always taking supply, the baseline, the simple hourly-mean version of idea A), so all of them are compared on the same unseen years. The freeze and held-out dates in objective 7 move if the gate is not yet passed; the page shows the actual dates.

6 Oct 2026, 22:30, ten registered ideas, one family (written before any of ideas 3 to 10 has been run; no held-out data read):
The family in one sentence: take virtual supply on calm days, and on dangerous days either sit out or take virtual load; the models' only job is to tell the days and hours apart using what is public by 05:00.
1. Storm-day filter: sit out the whole delivery day when a hand-made storm score is high (max of four trailing-365-day percentile ranks: NYISO peak load forecast, coldest and hottest GFS forecast temperature, the gap of D's hours already published). Found in exploration earlier on 6 Oct and disclosed as such.
2. Storm-day flip: the same score, virtual load instead of supply on storm days. Also found in exploration on 6 Oct.
3. Deep storm-day filter: a deep model reads every weather point, every zone's load forecast, outage counts and the recent price history, and predicts the probability that the next day's supply book loses more than a set amount; sit out when it is high.
4. Deep storm-day three-way: the same model; supply when low, sit out when middling, virtual load when very high.
5. Deep zone-day profit model: predicts each zone's next-day supply profit; supply where it beats the cost, load where it is strongly negative, otherwise none.
6. Two gates: the deep storm-day gate, then the deep hourly spike gate inside the days that pass.
7. Gradient-boosting storm-day filter: same target and inputs as 3, the non-deep comparison.
8. Storm-day zone pairs: on flagged days, load in New York City and Long Island against supply upstate, to trade storm congestion.
9. Extreme-tail spike flag: idea A with spikes defined as real time at least 100 USD above day ahead.
10. Weather surprise: the change in the GFS forecast for D+1 between the two newest runs public at 05:00; take load when the forecast turns sharply more extreme, otherwise supply.
The bar for the held-out list (Bo's): average net profit of at least 10 percent a year on the 500,000 USD bankroll (50,000 USD a year) over a walk-forward on 2021 to 2023 after full costs, positive in at least two of the three years, Sharpe above 0.42, and positive at the 0.50 USD per MWh cost stress.
Safeguards, applied to every idea: all inputs pass the publication-time test; an injected-lookahead test per idea; two placebos per idea (the same rule fed a score shifted one day late, and a shuffled score) that must not earn what the real rule earns; every choice made only on years before the scored year; any idea that passes is recomputed independently before it reaches the list. The list reports how many ideas were tried. All ten, the basic strategies and A to D go to the held-out run together once the list is approved.

6 Oct 2026, 23:20, two more registered ideas, same family (written before either has run; no held-out data read):
11. Learned allocator (a gate over experts): for each zone and delivery day a deep gate network reads the same day-level inputs as ideas 3 to 5 and puts weights on four simple experts, always supply, sit out, virtual load, and the storm-day zone pair, trained on the experts' realised next-day profit in earlier days only; the position is the weighted mix.
12. End-to-end policy: a deep network outputs each zone's position for the next day directly (from minus 1, supply, to plus 1, load) and is trained to maximise next-day profit after costs minus a penalty on losses, instead of predicting a price or a probability first.
Ideas 3 to 6 predict, then decide by a rule; 11 learns how to combine simple strategies; 12 learns the decision itself. Same bar, same safeguards, same walk-forward, and both count in the total number of ideas tried.

Correction, 6 Oct 2026, 21:16: the clock times written in the two previous addenda are wrong. The ten ideas were registered at 20:48 (commit 42ad84b) and ideas 11 and 12 at 21:07 (commit a156551), New York study written from Denmark, times in Danish time; the commit times are the record.
