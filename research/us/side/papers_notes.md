# Literature notes: NYISO virtual bidding (read 7 Oct 2026)

Read in full or in the relevant sections: SOM 2022 and 2024 (virtual trading sections), Birge et al. (full text and brief), Parsons et al. (abstract, intro, conclusion), Baltaoglu et al. (abstract, results), Trading Electrons (full html), Capponi et al. (first part of the html). The Energy Economics paper is paywalled, so only the abstract was read. Not read: SOM 2022 and 2024 sections on NYC load pockets in detail.

## 1. Potomac Economics, NYISO State of the Market 2022 and 2024
URLs: https://www.nyiso.com/documents/20142/2223763/2022-State-of-the-Market-Report.pdf/617e9176-cb4b-de7d-1026-af57175c4a8e and https://www.nyiso.com/documents/20142/2223763/2024-State-of-the-Market-Report.pdf/63b4d5c8-7e8c-45ac-02f7-7885406bae71

The 2022 report puts gross virtual profit at about $20 million, $0.86 per MWh, and calls that low and consistent with a well-arbitraged day-ahead market. The 2024 report puts net virtual profit at about $1.2 million for 2023 and $1.8 million for 2024, $0.06 per MWh; internal zones made $3 million and interfaces lost $1.2 million. That is a drop of more than ten times in the profit pool from 2022 to 2023, which matches our decay in 2023. The 2024 report also says its profit method changed (curtailed interface schedules are now excluded), so 2022 and 2024 figures are not like for like. Long Island is named as the zone most prone to real-time price premiums (old steam units, limited gas flexibility, weak ties), and day-ahead premiums were largest in January 2024 when gas was volatile. For Winter Storm Elliott the 2022 report describes neighbours curtailing imports on 23 and 24 December, which caused deep reserve shortages and bad fast-start commitments in real time, so the day-ahead price was far below real time on those days. It also finds physical load under-scheduled in the day-ahead by about 600 MW per hour in 2022, a structural source of real-time premiums. This supports our results: a thin, mostly well-arbitraged pool, with a few stress events carrying the profit.

## 2. Birge, Hortacsu, Mercadal, Pavlin, Limits to Arbitrage in Electricity Markets (MISO, Energy Economics 2018)
URLs: https://ceepr.mit.edu/wp-content/uploads/2021/09/2017-003.pdf and https://ceepr.mit.edu/wp-content/uploads/2021/09/2017-003-Brief.pdf

Virtual traders lowered the day-ahead premium in MISO only weakly. Trading fell when capital was scarce (financial crisis), and a regulator fee that raised transaction costs cut competition. Some large traders bought day-ahead against the premium and lost money for years, apparently to move congestion and raise the value of their FTR positions. The usable idea: a zone-pair congestion signal can be distorted by traders whose real book is elsewhere, so a losing pattern is not always noise. It is MISO and 2010 to 2013 data, so it challenges the claim that edges are simple forecast errors, but it does not test NYISO directly.

## 3. Foreseeing the worst: Forecasting electricity DART spikes (Galarneau-Vincent, Gauthier, Godin, Energy Economics 2023)
URL: https://ideas.repec.org/a/eee/eneeco/v119y2023ics0140988323000191.html

They predict the probability of a DART spike in the Long Island zone with engineered features and several learning algorithms, and report that results hold across algorithms. A trading exercise using the predicted probability as the signal earned consistent profits. The idea for us: classify the spike rather than regress the gap, and trade only on high probability. Only the abstract was available, so I cannot say what sample years it used or whether it survives costs. It supports our finding that Long Island is where the signal lives, and it warns that a tail event model needs a storm held out.

## 4. Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets (Hubert, Lolas, Sircar, 2026)
URL: https://arxiv.org/abs/2601.05085

This is the closest match to our study. NYISO, 11 zones, train 2015 to 2019, validate 2020 and 2021, test 2022 to 2025, features are 24 and 48 hour lagged DART, zonal day-ahead load forecasts and their lagged errors, calendar and season. Spike thresholds in NYISO are plus 5 and minus 30 dollars per MWh. The impact-aware strategy earned about $6 million cumulative over 2022 to 2025, nearly all of it Long Island, with sell trades (DEC) beating buy trades (INC). Winter peak hours show the largest sell-side price impact. The authors say transaction costs and market power are not modelled and regime change may break zone-specific models. This supports our Long Island and storm dominance and challenges our pairs result only in that they trade single zones. Note their test window includes Elliott and 2023 decay, like ours.

## 5. Parsons et al., Financial Arbitrage and Efficient Dispatch (MIT CEEPR 2015)
URL: https://ceepr.mit.edu/wp-content/uploads/2021/09/2015-002.pdf

The paper argues that day-ahead and real-time prices differ for reasons other than too little supply or demand, mainly because the two market algorithms make different approximations in unit commitment and power flow. Virtual bidders can profit from such gaps without fixing anything, and can add cost. It also says a smaller average gap is an imperfect sign that virtuals help. The idea for us: our profit is not evidence of efficiency gains, and the page should not claim the strategy improves the market. It is mostly a CAISO case, and it challenges the framing that our edge is a service to the market.

## 6. Baltaoglu, Tong, Zhao, Algorithmic Bidding for Virtual Trading in Electricity Markets (Cornell, arXiv 1802.03010)
URL: https://arxiv.org/abs/1802.03010

An online learning method splits a daily budget across bid options and is tested on NYISO and PJM with data from 2011 to 2016 (budget $250,000 a day, risk-adjusted by Sharpe ratio). It beat the benchmarks and the S&P 500 on Sharpe ratio, but the authors note profits fell over the years because of price convergence, and 2016 was NYISO's worst year. The idea for us: annual retraining with only the previous year, and reporting Sharpe by year, are a simple v2 template. It assumes the trader does not move prices, so it overstates what size can earn. It supports the decay story, and says the decay began years before 2023.

## 7. Capponi, Iyengar, Yang, Bienstock, Virtual Trading in Multi-Settlement Electricity Markets (arXiv 2508.11979, 2025)
URL: https://arxiv.org/abs/2508.11979

A theory paper, not a review. Without virtual traders, load-serving entities under-bid in the day-ahead, so day-ahead prices sit below expected real-time prices; with enough virtual traders the price gap closes but day-ahead quantity stays low. It reports that NYISO price gaps narrowed after virtual trading began. It assumes many identical traders, so it does not explain profit left over with finite traders. The idea: a persistent premium or discount by zone and hour is a sign of too little arbitrage capital, and its sign tells which direction to trade. It supports the view that the edge shrinks as capital arrives.

## Ideas for the v1 page (discussion and limits)
1. Put the market context next to our results: Potomac reports $20 million gross (2022), $1.2 million (2023), $1.8 million (2024) for all virtual traders. Our edge decaying in 2023 matches a shrinking pool, not a model failure alone.
2. State that the 2022 and 2024 profit figures use different methods and are not directly comparable.
3. Say that Elliott (23 and 24 December 2022) is a real-time shortage event caused by import curtailments, which no 05:00 day-ahead forecast could see. Report results with and without it.
4. Say that profit here is not proof the strategy helps the market (Parsons), and that large traders may hold positions elsewhere (Birge), so some patterns are not forecast errors.
5. Admit that we ignore price impact and trading costs, like Trading Electrons and Baltaoglu; quote their own limits.
6. Cite Trading Electrons as the closest independent result: Long Island and DEC dominate there too.

## Ideas for v2 (longer history, rolling windows)
1. Use 2015 to 2019 train, 2020 and 2021 validation, 2022 and later test as the baseline split, so results can be compared with Trading Electrons.
2. Roll yearly retraining on the previous year only, and report profit and Sharpe per year (Baltaoglu).
3. Add a spike classifier (Galarneau-Vincent): predict the probability of a spike beyond plus 5 and minus 30 dollars, trade only above a threshold, and compare with the pairs strategy.
4. Add lagged 24 and 48 hour DART and lagged load forecast errors as features, both cheap and used by the 2026 paper.
5. Model price impact per zone using load size, and test our trades at several sizes, because all three trading papers skip or simplify it.
6. Add a cost line (fees, and uplift where it applies) and show results net.
7. Use Potomac's yearly virtual profit as a market-wide benchmark and ask whether our share rises or falls over time.
8. Split by season; winter peak sell impact and summer buy sensitivity differ strongly.
9. Check the later SOM reports (2023) and the NYC load pocket tables before the v2 write-up, as I did not read them.
