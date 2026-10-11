# NYISO held-out: why the pair rows won 2025, and how the proposed sleeves co-move
Computed 11 Oct 2026 on gene, read-only, with the frozen scorer code paths (lab.Frame.day_pnl, score_v2.positions_to_mw, rolling.costs). Daily P&L reconciles with every published yearly total. Scripts and raw outputs: gene:~/analysis_20261011/ (a_2025.py/.log, a_v1cdeep.txt, b_corr.py/.log, b_books.py/.txt/.csv); copies of b_books.txt and a_v1cdeep.txt sit next to this file.

## A. 2025
- The stance was long N.Y.C., short CAPITL (and WEST). C_static_pairs 2025 net 102,043 at 1 MW = congestion 133,489 + losses -16,111 + fees -15,337 (energy cancels). 2013-23 average net 46,175.
- DA under-priced N.Y.C. congestion (5.25 DA vs 7.35 RT, USD/MWh) and over-priced CAPITL's (6.48 vs 4.92). N.Y.C. minus CAPITL RT-DA gap: 3.38 in 2025, 1.43 in 2024, -0.46..+2.15 in 2013-23.
- Two bursts: 25 Jun-31 Jul 50,944 and Jan-Feb 39,278 (88% of the year). Hours 16-21 earned 63,317 (2013-23 avg 12,566). Best day 1 Jul 16,729; best 10 days 66,941.
- V1_C_deep held the same stance smaller: 90,959, 11,084 below the rule (same side, smaller size -24,850; flat where rule traded -14,050; own positions +15,671; opposite side +12,145). Its own positions added +15,985 a year in 2013-23.
- Hypothesis (untested): links into the city (Central-East, UPNY-ConEd) ran nearer limits in real time than day-ahead expected, plausibly after in-city peakers retired in May 2025 (NOx rule) and before Champlain Hudson delivered. 2025 = strong year of a persistent pattern, not new model skill.

## B. Correlation, daily net at 1 MW (Pearson/Spearman: 2013-23 | held-out | held-out without events)
- pairs/V1_C_deep 0.41/0.55 | 0.76/0.67 | 0.81/0.67
- pairs/V15c 0.00/0.03 | 0.06/0.00 | 0.06/-0.01
- pairs/V16 0.01/0.04 | 0.02/-0.08 | -0.04/-0.08
- pairs/always_supply 0.03/-0.12 | -0.17/-0.17 | -0.27/-0.17
- pairs/idea13 0.03/0.10 | 0.21/0.12 | 0.38/0.12
- V1_C_deep/V15c 0.05/0.06 | 0.28/0.06 | 0.17/0.05
- V15c/V16 0.88/0.76 | 0.95/0.71 | 0.80/0.70
- V15c/always_supply -0.67/0.10 | 0.80/0.09 | 0.34/0.08 (Pearson driven by a few storm days: 2022 against supply, 2026 for it)
- V15c/idea13 0.73/0.29 | 0.92/0.51 | 0.67/0.51
- Tails: on the pairs' 20 worst held-out days V1_C_deep also lost on 18; V15c and V16 on 8. On supply sleeves' worst days V15c, V16, always_supply, idea13 correlate 0.82-0.99 (one bet); pairs -0.27..0.16 with them.

## B. Books sized so the 2013-23 worst drawdown equals my limit (35k = 7%, 75k = 15% of 500k); cautious = 70% of that size
Mix = pairs + V15c in 0.8/0.2 risk shares. Columns: MW pairs + MW V15c | 2013-23 return/yr, Sharpe, worst DD, worst day | held-out return/yr, Sharpe, worst DD, worst day | held-out w/o events return/yr, worst DD | limit held
- Mix 35k: 1.08+0.31 | 18.1%, 2.81, -35,000, -10,575 | 21.2%, 2.22, -22,096, -18,608 (24 Jan 2026) | 12.5%, -20,938 | held
- Mix 35k cautious: 0.76+0.21 | 12.7%, 2.81, -24,500, -7,403 | 14.9%, 2.22, -15,467, -13,026 | 8.8%, -14,656 | held
- Mix 75k: 2.32+0.66 | 38.8%, 2.81, -75,000, -22,661 | 45.5%, 2.22, -47,348, -39,874 | 26.9%, -44,866 | held
- Mix 75k cautious: 1.62+0.46 | 27.2%, 2.81, -52,500, -15,863 | 31.8%, 2.22, -33,144, -27,912 | 18.8%, -31,406 | held
- Pairs only 35k: 0.62 | 5.7%, 1.90, -35,000, -4,773 | 6.7%, 2.29, -10,513, -3,598 | 5.4%, -10,299 | held
- Pairs only 35k cautious: 0.44 | 4.0%, 1.90, -24,500, -3,341 | 4.7%, 2.29, -7,359, -2,519 | 3.8%, -7,209 | held
- Pairs only 75k: 1.33 | 12.3%, 1.90, -75,000, -10,228 | 14.3%, 2.29, -22,528, -7,711 | 11.6%, -22,068 | held
- Pairs only 75k cautious: 0.93 | 8.6%, 1.90, -52,500, -7,160 | 10.0%, 2.29, -15,769, -5,398 | 8.1%, -15,448 | held
- 2013-23 worst day for every book: 22 Feb 2015. Mix's worst held-out day (24 Jan 2026, storm) = 1.8x its worst 2013-23 day; at the 75k size one such day is about 40,000.
