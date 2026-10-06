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
