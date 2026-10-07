# 2023 rehearsal, version 2: development numbers

Written 6 Oct 2026. 2023 is build data. Nothing on or after 1 January 2024 was read (model/lock.py; FREEZE is absent). Version 1 (`rehearsal_2023.md`, `.json`) is unchanged. Full numbers are in `rehearsal_2023_v2.json`.

## What changed from version 1 (addenda of 6 Oct, late evening and night)

- **Idea A.** A LightGBM classifier predicts P(gap >= S) per zone-hour. A supplies in every zone-hour except where that probability is above p*.
  - S and p* were chosen together on 2022 from S in {25, 50, 100} and p* in {0.02, 0.05}: 6 configurations. The classifier was trained on delivery dates up to 30 Dec 2021, with fixed LightGBM settings.
  - Chosen: **S = 25 USD/MWh, p* = 0.05**. In 2022 this sat out 48% of zone-hours and earned 114 USD a day. The other five earned between -365 and +83.
  - S is written to `model/spike_config.json` for the deep model. Every build-year choice is in `model/frozen_choices.json`.
- **Costs** (cost audit). Every cleared MWh pays the Rate Schedule 1 rate plus the FERC charge. Supply legs also pay the uplift upper bound. In 2023 that is 0.1236 USD/MWh for load and 0.1496 for supply.
  - Every "fee" threshold now uses the cost of the side taken. A pair in C must cover one load leg plus one supply leg.
  - The 0.50 and 1.00 stress costs replace the whole cost.
- **Verdict words, now exact.** "Doesn't pay" means the upper end of the corrected interval is below zero. Under Holm, a word other than inconclusive needs that hypothesis rejected; this is the audit patch at score.py.
- **Deep against gradient boosting.** Better or worse is checked first. Equivalent applies only when the interval contains zero and lies within plus or minus the costs the gradient-boosting version paid per day.
- **D's 20 outage sites** are counted only on snapshots up to 30 Dec 2022.
- **Context beside every verdict:**
  - each idea's own net profit with its interval;
  - A against always-supply ("value of the spike filter");
  - annualised Sharpe;
  - maximum drawdown;
  - return on a 500,000 USD bankroll.
- B, C and D are otherwise unchanged. C keeps the v1 pairs: CAPITL against NORTH, GENESE, CENTRL, MHK VL and WEST.

## Verdicts for 2023 (gradient boosting; development numbers)

The baseline lost 200 USD a day (Sharpe -3.05). Always-supply made +187 a day (0.71 USD/MWh, Sharpe 1.38, maximum drawdown 67,173).

| Idea | Word | Idea minus baseline, USD/day [corrected interval] | Idea alone, USD/day [95%] | Per MWh | Break-even cost | MWh/day | Sharpe | Max drawdown | Return on 500k |
|---|---|---|---|---|---|---|---|---|---|
| A | pays | +256 [+96, +416] at 95% | +56 [-66, +176] | 0.28 | 0.43 | 198 | 1.15 | 27,565 | +4.1% |
| B | inconclusive | +138 [-88, +388] at 97.5% | -61 [-275, +128] | -0.25 | -0.11 | 249 | -0.62 | 43,515 | -4.5% |
| C | pays | +210 [+38, +380] at 98.3% | +11 [-85, +102] | 0.06 | 0.20 | 185 | 0.23 | 15,465 | +0.8% |
| D | inconclusive | +93 [-57, +235] at 95%, Holm stopped at B | -107 [-276, +59] | -0.44 | -0.31 | 241 | -1.21 | 47,689 | -7.8% |

- **Value of the spike filter.** A minus always-supply is -130 USD a day [-403, +162]. In 2023, sitting out the flagged 25% of zone-hours cost money against supplying every hour, but the interval is wide. The classifier itself ranks hours (correlation 0.20 between predicted probability and a spike; slope 0.65).
- **At the 0.50 stress cost,** A alone loses 4,784 USD over the year, while always-supply still makes 34,389. Against the baseline every idea looks better at higher costs, because the baseline trades every hour.
- **Without the most extreme 1% of zone-hours,** A minus baseline is +350 [+215, +483].
- **Zone-plus-generator ablation (A).** -12 USD a day [-57, +23]: inconclusive.

### Deep model (c2, chosen by the deep model run on 2022)

| Idea | Deep against gradient boosting [95%] | Word | Deep idea alone, USD/day |
|---|---|---|---|
| A | -59 [-113, -6] | worse | -3 |
| B | -37 [-280, +189] | inconclusive | -98 |
| C | +83 [-24, +188] | inconclusive | +93 (Sharpe 1.54) |
| D | -4 [-189, +181] | inconclusive | -111 |

The deep spike model uses S = 25 with calibrated probabilities. Its p* = 0.02 was chosen on its 2022 file, and it sat out 44% of zone-hours. The equivalence margin is 25 to 34 USD a day.

## Minimum detectable effect (80% power)

| Idea | 2023, USD/day | Projected for the 1,004 held-out days |
|---|---|---|
| A | 229 | 138 |
| B | 343 | 207 |
| C | 232 | 140 |
| D | 239 | 144 |

## Anything that looked like a bug, and the fix

1. **Holm boundary bug.** A verdict could be "pays" without the Holm rejection. Patched, and the audit's strict xfail test now passes.
2. **The equivalence margin** is now the cost the gradient-boosting version actually paid per day. With side-specific costs, "one fee" is no longer a single number.
3. **Eight of the audit's 25 tests now fail by design.** They encode the superseded single-fee model, where supply and load pay the same Rate Schedule 1 amount. They and `audit_pnl.py` need the audit's update to the audited cost model.
4. **The deep model run trained its first spike model at S = 50,** because it started before `spike_config.json` existed. I asked for a rerun at S = 25 with calibrated probabilities, and that rerun is what is scored here. The rescore now refuses a spike file whose threshold differs from `spike_config.json`.
5. **LightGBM slowdown on shared cores.** With 4 threads on a box whose other cores are busy, refits ran 25 times slower. `US_LGBM_THREADS` now sets the count. Results are deterministic for a given count.

No freeze and no held-out run: my instruction comes first. `model/heldout.sh` exists, prepared and unused.
