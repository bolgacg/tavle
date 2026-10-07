# New York study, version 2: objectives (registered 7 Oct 2026, before any v2 model runs)

Version 1 (research/us/) stays as published. Version 2 asks the same question with far more history and more ideas.
The held-out years 2024-01-01 to 2026-09-30 are NOT read by v2 either: v2 is built and judged on 2010 to 2023 only,
and then ONE held-out run tests the approved v1 and v2 candidates together (Bo's rule: no design after seeing them).

Design
- Rolling windows: train on the previous 3 years, test on the next quarter, roll quarterly, scored from 2013 to 2023
  (44 quarters, out of sample). Same costs (fees per year from NYISO postings, back to 2010), same bar, placebos,
  fragility flags, sizing view, publication-time test, and a count of every idea tried.
- Ideas (each a strategy row):
  V1. Core v1 strategies retrained on the long history: C (deep and gradient boosting), A spike flag, baseline, always supply.
  V2. Price-sensitive bids: a limit price per zone-hour; a bid counts only when day-ahead clears at or past it.
  V3. Per-hour tail-risk sizing: quantile models give expected profit and bad-tail loss; size by their ratio, capped.
  V4. Archived weather reforecasts (GEFS reforecast, never reanalysis) to run the weather ideas (B) before 2021.
  V5. Neighbouring regions: NYISO's border prices (PJM, New England, Ontario, Quebec proxies) as inputs and as spreads.
  V6. Zone specialists where the market monitor reports persistent non-convergence (New York City, Long Island).
  V7. Market-change awareness: dated regime flags for known rule and fleet changes, plus recency-weighted training.
  V8. Error mining: a second model trained on where the first model's errors cluster (zone, hour, weather).
  V9. Gap decomposition: predict the congestion and energy parts of the gap separately, then combine.
  V10. Pretrain on 2010 onward, fine-tune on the most recent window (transfer learning).
  V11. Ensembles over the last several windows' models.
  V12. Learned allocator over the v2 survivors (the idea 14 lineage).
- Weather-based ideas run only on years that have archived forecasts; any idea that cannot be built from public
  data available at 05:00 on D is reported as not run.
