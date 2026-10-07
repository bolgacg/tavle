# Deep learning setups for NYISO DART: what to test, in what order

Scope: predict day-ahead minus real-time price gap per zone and hour, decide at 05:00 the day before, 11 zones, hourly 2010 to 2023, rolling 3-year train and quarterly test, one GTX 1060 6 GB, 4 cores without AVX2, 16 GB RAM. Known so far: small GRU and gradient boosting work, profit is mostly a persistent inter-zone congestion spread, storms dominate risk, edges decay.

## What the literature says (short)

- Lago, Marcjasz, De Schutter, Weron (Applied Energy 2021, with an erratum fixing the German LEAR numbers): across five day-ahead markets (EPEX DE, FR, BE, Nord Pool, PJM) a well tuned LASSO autoregression (LEAR) and a plain feed-forward DNN with 100 to 200 sensible inputs are the strong baselines. The DNN wins on average by a small margin and the LEAR is close. Fancier deep models in that benchmark (LSTM, GRU, CNN hybrids) did not clearly beat the plain DNN. The lesson: inputs and calibration beat architecture.
- Hubicka, Marcjasz, Weron (2019): averaging forecasts from many calibration window lengths (28 to 728 days) beats picking one "best" window, which cannot be chosen reliably ahead of time. Applies directly to your window question.
- Olivares et al. (NBEATSx, 2023) report NBEATSx about 5 percent better than strong EPF baselines on several markets, with exogenous inputs. A newer comparison found TFT best once calendar features are added. These are modest gains on level prices, not on spreads.
- Foundation models: a 2025 benchmark (arXiv 2506.08113) of Chronos-Bolt, Chronos-T5, TimesFM, Moirai, Time-MoE, TimeGPT on European day-ahead prices found Chronos-Bolt and Time-MoE on par with classical models, and none significantly beat a two-season MSTL baseline.
- DART and virtual bidding: published work is thin. Transformer spread forecasting on ERCOT (arXiv 2412.00062), recurrent LMP-spread models with monotonic boosted trees for PJM, ISO-NE, CAISO, and a NYISO/ISO-NE/ERCOT multi-zone framework (arXiv 2601.05085). All are small-sample, single-split style studies. I found no paper showing a deep model clearly and robustly beating boosting on DART across years.
- Graph networks on prices (arXiv 2107.12794, 2106.10529): gains over non-spatial nets, mostly on simulated or OPF-derived data with known topology and load.

Most ranks below are inference: the DART literature has no robust deep-beats-boosting result.

## Ranked table

Cost = on this hardware. Storm risk = overfitting danger given few storm days.

| Rank | Setup | Why it might help here | Cost | Storm risk | Evidence |
|---|---|---|---|---|---|
| 1 | Window ensemble: same model trained on 1, 2, 3, 5 year and expanding windows, averaged; recency weights | Edges decay, so older data hurts, but short windows have few storms. Averaging avoids choosing | Low (multiplies training by 4 to 5) | Low, averaging damps storm fits | Strong (Hubicka 2019) |
| 2 | Quantile heads (pinball loss at 5 to 6 levels) on the existing GRU and boosting, trade sized by quantile spread or sign agreement | Sizing is where storm losses come from; median alone ignores tails | Very low | Medium (tails learned from few days) | Good in EPF, direct for sizing |
| 3 | Conformal wrapper (rolling or adaptive conformal on the point forecast) for position size and a skip rule | Model free, cheap, gives honest coverage as the regime drifts; ACI handles drift | Negligible | Low | Good in theory, light in DART specifically |
| 4 | Ensembles: 5 to 10 seeds of the GRU plus boosting blend; snapshot ensembles | Reduces variance, the main DL failure in EPF | Low (small GRU) | Low | Strong (Lago 2018/2021 ensembles of DNNs) |
| 5 | Plain MLP/DNN in the Lago style, many inputs, multi-output over 24 hours and zones | The proven EPF deep baseline; multi-output across zones shares the congestion structure | Low | Medium | Strong |
| 6 | Decision-focused loss: train on negative PnL after position sizing plus a risk penalty (CVaR or variance), or a smoothed sign-weighted loss | Aligns loss to trading, which cares about sign and size on big gaps | Low to medium | High: noisy gradients, a few storms dominate | Plausible, thin in DART |
| 7 | NBEATSx / NHiTS with exogenous load, weather, outages | Best published EPF deep models; cheap | Low to medium | Medium | Moderate, on levels not spreads |
| 8 | Zone transfer: one model with a zone embedding trained on all 11 zones (and pretrain on other markets, fine-tune) | 11 times the examples, congestion spread is a cross-zone structure | Medium | Medium | Moderate, fits global-model trend |
| 9 | Regime gating / mixture of experts (storm vs calm, congested vs not) | Storms behave differently | Low | Very high: the storm expert has few examples | Weak |
| 10 | Temporal Fusion Transformer | Interpretable, handles known-future inputs like forecasts | Medium (fits 6 GB with small width) | Medium | Moderate on levels |
| 11 | Zone-level GNN (11 nodes, edges from historical congestion or PTDF-like links) | Congestion is a network effect | Medium | Medium | Weak on forecasting, good on simulated |
| 12 | Chronos-Bolt zero-shot as a feature or baseline | Cheap check, no training | Low (small/base fp32 fits) | n/a | Weak: did not beat MSTL on levels |
| 13 | Augmentation and input dropout (noise on weather, feature dropout) | Regularisation for small data | Negligible | Low | Weak but safe |
| 14 | Generator-bus GNN (600 to 750 nodes) | Most detail | High; likely out of 6 GB for long histories | Very high | Weak |
| 15 | PatchTST, iTransformer, TiDE, TSMixer | Strong on generic benchmarks | Low to medium | Medium | Weak for prices; mostly beat on long-horizon smooth series, not noisy spreads |
| 16 | Moirai, TimesFM, Lag-Llama fine-tune; DeepAR; mixture density heads | See skip list | Medium to high | High | Weak |

## Hardware notes

- GTX 1060 is Pascal (compute capability 6.1): no tensor cores, no bf16, slow fp16. Train in fp32. Recent PyTorch CUDA 12.8 wheels dropped some older architectures, so use a cu126 or cu118 build and check that `torch.cuda.is_available()` and a small matmul work before planning anything. I have not tested this on your box.
- No AVX2: some prebuilt CPU wheels (and some xgboost/lightgbm/onnx builds) may crash with illegal instruction. Test imports first. This is a risk to confirm, not a certain problem.
- Chronos-Bolt small and base fit in 6 GB for inference. 16 GB RAM is enough if series are memory mapped as float32.

## Recommended plan (6 setups plus window sweep)

Rule for every run: same preregistered folds, same cost model, report PnL per zone, worst storm day, and the share of PnL from the top congestion pair. A setup that wins only because it learns the same congestion spread the boosting already has is not new.

1. Quantile GRU (rank 2) on current features. Sizing rule: trade only when the quantile band excludes zero, size by band width inverse. Compare to point GRU with the same sizing.
2. Conformal wrapper (rank 3) on both GRU and boosting. Report coverage over time and whether skipped days remove the storm losses.
3. Seed and model ensemble (rank 4): 10 GRU seeds plus boosting, plain average, then a rank-average.
4. Lago-style multi-output DNN (rank 5) over 24 hours x 11 zones as a second family so the ensemble has diversity.
5. Zone-embedding global model (rank 8): one GRU or DNN for all zones with a zone embedding, versus 11 separate ones. Tests whether pooling helps the thin zones.
6. NBEATSx/NHiTS (rank 7) with the operator load forecast, outages and archived weather as exogenous. Only if 1 to 5 leave headroom; otherwise swap in decision-focused loss (rank 6) with a CVaR penalty and a fixed seed ensemble to tame noise.

Window sweep (applies to setups 1 to 4): train length 1, 2, 3, 5 years and expanding, each with and without exponential recency weights (half life 6 and 12 months), then the equal-weight average of all window forecasts (Hubicka style). Judge by out-of-sample PnL and by how fast the edge decays after each fold. Expect the average to beat the single best window; if one short window wins by a lot, that is a sign the regime shifted and should be explained, not just adopted.

## Skip, and why

- Generator-bus GNN: biggest cost, highest overfit risk, and the published gains come from known-topology, load-known settings. The zone-level graph, if tried at all, is enough.
- PatchTST, iTransformer, TiDE, TSMixer: built for long smooth horizons with many series; no EPF or DART evidence they beat a GRU or boosting, and the extra capacity would memorise storms.
- Moirai, TimesFM, Lag-Llama, fine-tuning: zero-shot did not beat a seasonal baseline, they use your exogenous inputs poorly, and installs are heavy on a no-AVX2 box.
- DeepAR and mixture density heads: quantile heads give the sizing information with less fragility; mixtures are unstable with fat-tailed spreads.
- Mixture of experts by regime: the storm expert trains on a handful of days. Use conformal skip rules instead.

