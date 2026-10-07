# v2 stage 1 audit (7 Oct 2026)

Scope: V2_limit_gbm, V1_C_deep, V3_tail_gbm from gene `~/nyiso-us/results/v2/v2_results_stage1.md`. Script: `audit_stage1.py` (laptop and gene `~/nyiso-us/v2/audit/`), output `audit_stage1.json`. Every read is filtered below 2024-01-01 at read time and asserted. The script re-derives fills, limits, pair spreads and the score from `prices_zone.parquet` directly; it borrows only the fee table from `rolling.py`. Laptop and gene code are byte-identical (md5 checked).

## 1. V2_limit_gbm

Mechanics are clean, and the reported numbers reproduce exactly. My own scan-loop limit rule gives positions identical to the saved file on 100 percent of rows, total 1,178,549, Sharpe 2.41, and the same eleven yearly figures.

- The limit comes only from 05:00 information. The nine grid levels equal D's same-hour day-ahead price times the multipliers, with a maximum difference of 0.0. The side is the conditional model evaluated at multiplier 1.0. No realised D+1 value enters either one; the realised day-ahead price is an input of the conditional model in training rows only.
- The fill is decided by the realised day-ahead price, as it should be: supply fills when DA >= limit, load fills when DA <= limit. The panel's `y_da_lbmp` and the price file's `da_lbmp` differ by 0.0.
- Profit is mw x (RT minus realised DA) minus the fee, so it is paid at the cleared DA and not at the limit. Unfilled bids have mw = 0, so they earn nothing and pay no fee. The largest position is 1.0 MW.

The problem is the label. The price sensitivity adds nothing: the same sides bid every hour with no limit earn 1,193,308, which is 14,759 more than with limits. 58 percent of supply limits sit at the lowest grid level (0.5 x D's price) and 34 percent of load limits at the highest (3 x), so most bids behave like market orders. The fills at those widest limits earn 924,209 at 1.94 USD/MWh. The fills where the limit actually binds earn 254,340 at 0.75 USD/MWh. Model-free price-sensitive rules lose money. For example, "supply when DA clears at or above 1.0 x D's price" loses 331,743, and V2's own sides with that limit earn 171,026. So the profit is the gradient-boosting model's per-zone direction, which is the same bet as V3's sides at 1 MW (1,356,902, Sharpe 1.94). The limit raises Sharpe from 2.15 to 2.41 only because it trims 19 percent of the volume. Note also that 2023 is +2,064 only because of 2023-09-05 (+19,499); without limits, 2023 is -19,496.

## 2. V1_C_deep placebos

Yes, C is mostly a persistent structural spread. 87 percent of open zone-hours carry the same position as the day before, and 82 percent take the same side as a static rule (each chosen pair's mean spread per local hour over the prior 3-year window). Shifting by one day or permuting days inside a year therefore barely changes the book, which is why those placebos keep 104 and 81 percent of the real profit.

Static rules over the same quarters:

| rule | total | Sharpe | USD/MWh |
|---|---|---|---|
| C deep (real) | 668,609 | 2.25 | 0.80 |
| C's pairs, static side per pair-hour | 571,020 | 1.99 | 0.63 |
| No model at all (pairs by trailing static net, static side per pair-hour) | 507,922 | 1.90 | 0.57 |
| C's traded hours, static side | 555,237 | 2.03 | 0.67 |
| C's positions where it agrees with static | 622,081 | 2.44 | 0.90 |
| C's positions where it disagrees with static | 46,529 | 0.36 | 0.33 |

The model adds about 161,000 over the fully model-free rule, roughly 15,000 a year. The part that is genuine prediction (the departures from the static side) earns 46,529 at Sharpe 0.36, which is close to noise.

## 3. V3_tail_gbm

The cap holds: the largest position is 3.0 MW. But 64 percent of open positions sit at the cap, the mean open size is 2.46 MW, and k = 40 (the top of the grid) was chosen in 41 of 44 scored quarters, so the sizing rule has collapsed into "about 3 MW almost always". Per MWh it earns 1.57 USD at Sharpe 1.90. Divided by 3, it is 1,302,491. The same sides at a flat 1 MW earn 1,356,902 at Sharpe 1.94. The tail sizing therefore adds no risk-adjusted value, and the 3.9M headline is leverage.

## 4. Verdicts

- **V2_limit_gbm**: real, but mislabelled. No leak was found, and the timing placebos drop it to 48 percent (one day late) and 14 percent (permuted). The edge comes from the gradient-boosting model's direction, not from limit prices. Report it as "GBM direction at 1 MW", alongside V3 at 1 MW.
- **V1_C_deep**: real but structural. About 76 percent of it is a static pair spread, and the model's own timing adds near-noise. V11_C_deep_ens4 and V10 share this pair machinery and should be read the same way (not separately audited).
- **V3_tail_gbm**: an artefact of size. Its true content is the V2 direction bet at 1.57 USD/MWh and Sharpe 1.9.
- **Single most likely failure on unseen years**: the GBM direction edge is fading. 2023 is about zero once one day is removed (V2 without limits -19,496; V3's best day is 59,060 of its year's 70,495). V2 and V3 are one bet, so they would fail together. This rests on the v1 timing test of the base features (500 bid days, 0 failures), which I did not repeat.
