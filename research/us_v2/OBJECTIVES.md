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

7 Oct 2026, after the stage-1 audit (research/us_v2/audit/v2_stage1_audit.md), before any held-out data is read:
- Comparison row added: C static pairs, no model. Each registered pair takes, for every hour, its prior-window usual side (the sign of its average gap in the 3 training years), 1 MW, same costs. It is the benchmark for how much the C models add beyond a persistent congestion spread.
- Comparison row added: V2 sides without limits (the gradient-boosting side at every hour, 1 MW, no limit price), because the audit showed the limit prices add nothing and the profit comes from the side.
- Labels corrected: V2's result is reported as the gradient-boosting direction model; V3 is reported per MWh and at 1 MW, since its total reflects position size.

7 Oct 2026, V13, registered before it runs (no held-out data read):
V13. All-inputs deep model: one deep network fed every input public by 05:00 on D (zone and generator-point prices and their energy, congestion and loss parts, border prices, the operator's load forecasts, outage counts, archived weather forecasts where they exist, calendar), trained in the same rolling windows, predicting each zone's next-day gap; positions as in C and A. A fixed menu of at most three architectures (recurrent, temporal convolution, small attention), chosen on earlier windows only; strong regularisation and input dropout. Feature-group ablations (each input group removed in turn) show what each source adds. It is one idea in the count; the ablations are diagnostics, not extra strategies.

7 Oct 2026, V14 family, six deep-learning setups approved by Bo, registered before any runs (research: research/us_v2/dl_setups_research.md; no held-out data read):
V14a. Window ensemble: models trained on 1, 2, 3 and 5-year windows and an expanding window, recency-weighted, predictions averaged.
V14b. Quantile heads (GRU and gradient boosting) used to size positions: smaller when the predicted range is wide.
V14c. Conformal wrapper: calibrated intervals from earlier windows; skip an hour when its interval includes the fee-adjusted break-even.
V14d. Seed ensemble of the GRU plus gradient boosting, averaged.
V14e. One multi-output network for all 24 hours and 11 zones (the electricity-price benchmark design).
V14f. One global model with learned zone embeddings; if time allows, an N-BEATSx/NHiTS variant or a profit-trained loss with a tail penalty, chosen on earlier windows only.
Each is one idea in the count, same rolling windows, rules, placebos and fragility checks. The held-out run waits until all of them have finished.

7 Oct 2026, V15, four combinations the feature matrix showed were untried (research/us_v2/feature_matrix.md), registered before they run; no held-out data read:
V15a. The limit-price rule (V2) applied to the V4 weather, V5 border and V8 error-mining models.
V15b. The V13 all-inputs deep model traded on both sides and with limit prices, not only pairs and supply-only.
V15c. One gradient-boosting model with weather and border inputs together; weather from the GEFS reforecast to 2019 and the GFS archive from 2021, never observed weather.
V15d. The zone-pair rule driven by the V4 weather model.
One idea in the count; same rolling windows, rules, placebos and fragility checks; the held-out run waits for it.

7 Oct 2026, V16, the last addition before the held-out run; no held-out data read:
V16. Idea V4 (weather forecasts against the operator's load forecast) with a consistent forecast source: the GEFS reforecast to 2019 joined to the archived live GEFS forecasts (the same GEFS version 12 model) from 23 September 2020, control member, 00 UTC run, same leads and points; January to September 2020 has no forecast and is not traded. Same rolling windows and rules. The live GEFS archive for 2024 to 2026 is downloaded for the held-out run but sealed: only file counts are read before the run. After V16 the list of ideas is closed.

8 Oct 2026, code and held-out list published before the held-out run (no held-out data read): the list is research/us_v2/HELDOUT-LIST.md; the code that produced the 2013 to 2023 results is published byte-identical to what ran, with sha256 in research/us/CODE-MANIFEST.txt; the held-out mode is new code that only continues the rolling windows to 30 September 2026 (details in the version 1 addendum of the same date).

8 Oct 2026, the freeze (no held-out data read): code frozen at a581c3b3eb444e7a1caca39d3f79fbe8be8e3aeb; the held-out run uses only this commit; v2 dry runs run first on gene, and if any row fails to reproduce 2023 exactly the held-out run does not start. The version 1 dry run is repeated from the same commit before it as well. research/us/FREEZE names the commit; research/us/model/orchestrate_heldout.sh runs every stage.
