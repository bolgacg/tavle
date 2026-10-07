# Modelling contract for the New York study (written 6 Oct 2026, before any model ran)

Governing document: ../OBJECTIVES.md (approved, with dated addenda). This contract only makes it executable.
Data layer: ../pipeline/ (README.md, timing.py: features_available_at, bid_inputs, assert_no_lookahead).
Runs on gene (`ssh gene`, venv ~/nyiso-us/.venv, parquet ~/nyiso-us/parquet/, GPU GTX 1060 6 GB). Canonical code
lives here on the laptop; rsync to gene ~/nyiso-us/model/ to run. Nothing is committed or pushed automatically.

## The panel
- One row per (bid_date D, zone, delivery hour h of D+1). Decision time = 05:00 America/New_York on D.
- zone in the 11 load zones only (CAPITL CENTRL DUNWOD GENESE HUD VL LONGIL MHK VL MILLWD N.Y.C. NORTH WEST).
- target `gap` = rt_lbmp - da_lbmp for that delivery hour, USD per MWh (hourly integrated real time).
- Every feature comes through timing.features_available_at(D 05:00) or is proven equivalent by a test.
- Build years: delivery dates 2020-01-01 to 2023-12-31 (idea B from 2021-03-25). Held out: 2024-01-01 to 2026-09-30.

## The holdout lock (a mechanism, not a promise)
model/lock.py: `assert_build_only(delivery_dates)` raises if any delivery date >= 2024-01-01, UNLESS all of:
research/us/FREEZE exists and names a commit hash; `git diff --quiet <hash> -- research/us/model research/us/pipeline`
passes; and env US_HOLDOUT_RUN=1. Every loader calls it. A test proves it raises.

## Positions and money (per zone-hour, 1 MW, assumed to clear at the market price)
- supply: earns -gap - fee. load: earns gap - fee. none: 0. fee = that year's NYISO Schedule 1 non-physical
  rate per cleared virtual MWh (2023 0.1066, 2024 0.1333, 2026 0.1919; find 2020-2022 and 2025 in NYISO postings,
  else use the nearest known year and record it in fees.py with its source).
- Daily P&L = sum over the day's zone-hours. The unit of analysis is the day.

## Strategies (registered; the verdict model inside every idea is gradient boosting)
- baseline: every zone-hour, take the side of its trailing 365-day mean gap (data published by D 05:00).
- A (main): supply only. Position = supply when the model's predicted gap <= -fee, else none.
- B: base + weather features (features_weather.py). supply if pred <= -fee, load if pred >= +fee, else none.
- C: zone pairs. Up to 5 pairs chosen on build years only, fixed before the freeze. Load in i and supply in j
  when pred_i - pred_j >= 2*fee, the reverse when <= -2*fee, else none.
- D: base + outage features (features_outages.py), both sides as in B.
- The deep model runs inside every idea in place of gradient boosting, for the secondary comparison.

## Model interface
class Model: name: str; fit(panel_train: DataFrame) -> None; predict(panel: DataFrame) -> Series of predicted gap,
aligned to panel.index. Gradient boosting = LightGBM (installed on gene, no SIGILL). Deep = PyTorch on the GPU,
average of 3 seeds, early stopping on a time-ordered validation slice inside the training window only.

## Walk-forward
Monthly refit on the 1st of each month, training on delivery dates up to two days before that month starts.
Hyperparameters: at most 8 configurations per model, chosen on 2020 to 2022 with a time-ordered inner split.

## Scoring (score.py)
For each idea: daily P&L of idea minus baseline. 95% interval from a stationary bootstrap over days, mean block 7
days, 10,000 replicates, fixed seed. Primary A at 0.05. Secondary B, C, D with Holm correction. Also: profit per
MWh traded, break-even fee, worst month, results without the most extreme 1% of hours, fee stress at 0.50 and 1.00,
months positive, per zone and year, calibration of predicted gap. Verdict words exactly as objective 6.

## Rehearsal (objective: 9 Oct)
Build on 2020 to 2022, walk forward through 2023 with monthly refits, run every idea and both models, write
research/us/results/rehearsal_2023.json with everything score.py produces plus the minimum detectable effect.
2023 is development data: its numbers may guide fixes before the freeze. Stop after the rehearsal. The freeze and
the held-out run happen only on my instruction.

## File groups (each file belongs to one group)
- core model code: lock.py, fees.py, panel.py, gbm.py, strategies.py, score.py, walkforward.py, rehearsal.py, tests.
- feature code: features_weather.py, features_outages.py, their tests, and ../pipeline changes for the
  three-day weather rule (addendum of 6 Oct evening).
- deep model code: deep.py, deep_data.py, train_deep.py, their tests.
