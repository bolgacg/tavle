# NYISO virtual trading: cost and capital audit

Written 6 Oct 2026 for the New York study (OBJECTIVES.md, model/CONTRACT.md, model/fees.py). Read-only audit: nothing in the study was changed. Every number below comes from a primary source named next to it, or from a computation on the study's own 2020 to 2023 price files, labelled as such.

Holdout note. The data README forbids any price statistic on dates from 1 January 2024. All spread, loss and collateral numbers computed here use 2020 to 2023 prices only (local files `data/raw/us/*damlbmp_zone*` and `*rtlbmp_zone*`, scripts in the session scratchpad, every script asserts the last date is before 2024). Cost items for 2024 to 2026 (tariff rates, fee pots, uplift totals, traded volumes) are not price statistics. The market monitor's per-side virtual profitability for 2024 and 2025 is a held-out outcome and is deliberately not reproduced here.

## Summary: cost per cleared MWh of virtual supply, by year

| Year | Rate Schedule 1 virtual rate (USD/MWh) | Source status | FERC fee share (USD/MWh, est.) | Forecast-pass BPCG uplift, supply only, upper bound (USD/MWh) | Exchange cost, supply (USD/MWh) | Value in fees.py today |
|---|---|---|---|---|---|---|
| 2020 | 0.0862 | NYISO posting | about 0.010 | 0.007 | about 0.10 | 0.0862 |
| 2021 | 0.0757 | NYISO budget deck | about 0.015 | 0.004 | about 0.09 | 0.0862 (wrong) |
| 2022 | 0.0853 | NYISO posting | about 0.016 | 0.003 | about 0.10 | 0.0853 |
| 2023 | 0.1066 | NYISO posting | about 0.017 | 0.026 | about 0.15 | 0.1066 |
| 2024 | 0.1333 | NYISO posting | about 0.017 | 0.017 | about 0.17 | 0.1333 |
| 2025 | 0.1666 | tariff cap, secondary report; posting not found | about 0.02 | 0.017 to 0.028 | about 0.21 | 0.1919 (above the tariff maximum) |
| 2026 | 0.1919 | CONTRACT.md; posting not found, inside the tariff band | about 0.02 | about 0.03 | about 0.24 | 0.1919 |

Virtual load pays the same Rate Schedule 1 and FERC fee but no uplift. Collateral and its cost of capital come on top (section 4): realistic all-in cost for a small trader running about 100 MWh a day of supply is 0.15 to 0.40 USD/MWh in 2024 to 2026. The study's stress costs of 0.50 and 1.00 USD/MWh sit above that whole range.

## 1. Rate Schedule 1 on virtual transactions

Charged once, on the cleared day-ahead quantity. The tariff charges "each Transmission Customer that has its virtual bids accepted" a rate times "VTCleared", defined as "the total cleared Virtual Transactions, in MWh" in the billing period (NYISO OATT section 6.1.2.4.1, tariff sheets 211 to 212, https://nyisoviewer.etariff.biz/viewerdoclibrary/mastertariffs/9fulltariffnyisooatt.pdf). NYISO's own settlement training spells out the billing determinant as "Hr DAM Vsupply/Vload Energy (MW) = Total cleared Virtual MWh", charged "if Virtual Customer is scheduled to purchase or sell virtual energy (MWh) in the NYISO DAM Market" (Virtual Trading Market Settlements, NYISO 2024, slides 42 to 46, bill codes 418 and 778, https://www.nyiso.com/documents/20142/3035389/Virtual-Trading-Settlements.pdf/2e4e8a27-b4b1-a0e1-b376-0cab32b93180). The real-time buy-back of a virtual supply position is a balancing settlement of the same MWh and carries no second Schedule 1 charge. A 1 MW position for one hour therefore pays the rate once, which is how fees.py and CONTRACT.md already treat it. Uncleared bids pay nothing.

How the rate is set. Each year NYISO resets the rate as the prior year's virtual revenue requirement, escalated by the budget change and corrected for over or under collection, divided by a three-year rolling average of cleared virtual MWh, and "The annual rate computed through the formula in this Section 6.1.2.4.4 shall be subject to a 25% maximum increase or decrease for each year" (OATT 6.1.2.4.4, sheets 213 to 214). The cap explains the recent path: 0.0853 times 1.25 is 0.1066 (2023) and 0.1066 times 1.25 is 0.1333 (2024), so the rate rose by the full cap two years running.

| Year | Rate (USD per cleared MWh) | Source |
|---|---|---|
| 2020 | 0.0862 | "Schedule 1 Rates for 2020", Rate Schedule 1 for Non-Physical Transactions, p. 1, https://www.nyiso.com/documents/20142/7661617/2020-Rate-Schedule-1.pdf/6436863b-bf7c-4228-5be6-a78e7971023c |
| 2021 | 0.0757 | NYISO BPWG, "September 2021 Draft Budget versus Actual Results", slide 4, row "Virtual Trading $0.0757/ Cleared MWh", https://www.nyiso.com/documents/20142/26105979/11122021%20BPWG%20September%202021%20Draft%20Budget%20versus%20Actual%20Results.pdf/68d1c1eb-f317-f39f-071f-a914a41151aa (same rate in the March 2021 deck, slide 4, https://www.nyiso.com/documents/20142/20997935/03%202021%20Draft%20Budget%20versus%20Actual%20Results.pdf/1519911e-8989-6106-842c-d51b983ec0be) |
| 2022 | 0.0853 | "Schedule 1 Rates for 2022", p. 1, https://www.nyiso.com/documents/20142/24362705/2022-Schedule-One-Posting-with-accounting-detail.pdf/844f8b6a-089c-4837-f453-7e0f6989bd75 |
| 2023 | 0.1066 | "Schedule 1 Rates for 2023", p. 1, https://www.nyiso.com/documents/20142/32669344/2023-Schedule-One-Posting-for-posting-with-detail.pdf/840e952d-8555-a853-8d46-33a05cba1911 ; also the settlement training, slide 46 |
| 2024 | 0.1333 | "Non-Physical Schedule 1 Rates for 2024", p. 1, https://www.nyiso.com/documents/20142/39240890/2024-Sched-One-Posting-non-Physical-for-posting.pdf/e0e5b9a6-04b4-e0c3-bb08-ebf489192f83 |
| 2025 | 0.1666 | Posting not found on nyiso.com (its document library is script-driven and the file is not indexed). 0.1666 is 0.1333 times 1.25, the most the tariff allows, and matches a search-engine summary of a 2025 posting. The tariff band for 2025 is 0.1000 to 0.1666, so the 0.1919 now in fees.py for 2025 is impossible under OATT 6.1.2.4.4. |
| 2026 | 0.1919 | Taken from CONTRACT.md; posting not found. If 2025 was 0.1666, the 2026 band is 0.1250 to 0.2083, and 0.1919 (a 15 percent rise) lies inside it. Unverified at a primary source. |

FERC annual charge. Yes, there is a separate per-MWh charge. NYISO passes its annual FERC fee through Rate Schedule 1: "The annual FERC fee shall be allocated ninety-four (94%) to physical market activity and six (6%) to non-physical market activity ... approximately two percent (2%) to Virtual Transactions" (OATT 6.1.15, sheets 248 to 249), charged per cleared virtual MWh (6.1.15.2, sheet 251; training slides 47 to 49, bill code 419). NYISO's postings give the virtual pot directly: 339,366 USD for October 2021 to September 2022 (true-up credit 23,334), 333,120 USD for 2022 to 2023 (true-up charge 9,581) and 383,088 USD for 2023 to 2024 (true-up credit 15,630) (FERC Fee Calculations, p. 1: https://www.nyiso.com/documents/20142/24362705/2021-22-Rate-Schedule-One-FERC-Fee-True-Up.pdf/710a971a-8106-749c-37e5-368bda3db567 , https://www.nyiso.com/documents/20142/32669344/2022-23-Rate-Schedule-One-FERC-Fees.pdf/d15d8c71-7455-0206-2baa-4b0adf29798d , https://www.nyiso.com/documents/20142/39240890/2023-24-Rate-Schedule-One-FERC-Fee-.pdf/f97b453b-3a80-f36c-6639-2192c983cafa). Divided by cleared virtual MWh at the load zones (the market monitor's average cleared virtual supply plus virtual load, times hours: about 24.2 million MWh in 2021, 18.0 in 2022, 23.2 in 2023, 20.8 in 2024; State of the Market reports cited in section 2), the charge is 0.015 to 0.018 USD/MWh. The 2020 figure (about 0.010) uses the 2020 volume of 32.5 million MWh with a pot of the same size and is an estimate; the 2024 to 2025 and 2025 to 2026 pots were not found, so 0.02 is a planning value. Because NYISO computes an hourly rate as a fixed hourly charge divided by that hour's cleared volume, thin hours carry a somewhat higher rate.

Nothing else is charged per virtual MWh. The other Rate Schedule 1 charges (residual costs 6.1.8, DAMAP 6.1.10, local and remaining BPCG 6.1.12.2 and 6.1.12.5, NERC and NPCC) are billed on Withdrawal Billing Units, defined as "Actual Energy Withdrawals (for all internal withdrawals) or Scheduled Energy Withdrawals (for all Export Energy withdrawals)" (OATT definitions, sheet 51). Virtual load is neither, so virtuals do not pay them. The one exception is the forecast-pass uplift in section 2.

## 2. Uplift allocated to virtual supply and virtual load

The rule. NYISO allocates to virtual supply a share of one uplift category only: bid production cost guarantee (BPCG) payments to "Additional Resources" that the day-ahead market's forecast pass commits when bid load and bilaterals fall short of NYISO's day-ahead load forecast (OATT Rate Schedule 1, 6.1.12.1, sheet 240; method in OATT Attachment T, sheets 1765 to 1767, effective 30 Sep 2010). Attachment T defines the payers ("Eligible Transmission Customers") as those "scheduled to sell Energy at a Load bus specified for Virtual Transactions in the Day-Ahead Market" and those "purchasing Energy to serve load in the real-time market". The daily pot is split over four superzones (A to E, F to I, J, K), scaled by a factor of at most one that compares real-time purchases with the forecast shortfall, then shared among eligible customers by their real-time purchases in the superzone. A virtual supply position buys its MWh back in real time, so it counts in full. The residual not allocated this way goes to all withdrawals. NYISO's training states the effect plainly: "by design VS are short in RT", and "Virtual Load bids do not add to these uplift costs and therefore are not subject to the uplift charges"; the charge "Applies to Virtual Suppliers not Virtual Loads" and is calculated daily, bill code 815 (Virtual Trading Market Settlements, slides 52 to 62, URL above). All other BPCG uplift (day-ahead economic, real-time, local reliability, SRE, DAMAP, minimum oil burn) is allocated to Load Serving Entities, Transmission Owners or withdrawals, which the monitor also states: non-local uplift is "allocated to all Load Serving Entities" and local uplift "to local Transmission Owners" (2024 State of the Market report, appendix p. A-134, https://www.potomaceconomics.com/wp-content/uploads/2025/05/NYISO-2024-SOM-Full-Report_5-14-2025-final.pdf).

The size. The market monitor publishes the whole forecast-pass pot (2024 State of the Market report, Figure A-92 inset table, p. A-130):

| Year | Forecast-pass days | Forecast-pass MWh | BPCG uplift (USD) | Cleared virtual supply at load zones (MW average) | Upper bound if all of it fell on virtual supply (USD/MWh) |
|---|---|---|---|---|---|
| 2020 | 60 | 65,403 | 142,192 | 2,336 | 0.007 |
| 2021 | 26 | 18,924 | 64,327 | 1,689 | 0.004 |
| 2022 | 42 | 25,159 | 30,902 | 1,020 | 0.003 |
| 2023 | 54 | 41,249 | 332,893 | 1,437 | 0.026 |
| 2024 | 24 | 13,720 | 186,557 | 1,275 | 0.017 |
| 2025 | 42 | 24,090 | not published | 1,327 | 0.017 to 0.028 (pot scaled at the 2023 and 2024 cost per forecast-pass MWh) |

Virtual supply volumes: 2020 report p. 23 (2,336 MW), 2022 report p. 56 (1,689 and 1,020 MW, https://www.nyiso.com/documents/20142/2223763/2022-State-of-the-Market-Report.pdf/617e9176-cb4b-de7d-1026-af57175c4a8e), 2024 report p. 40 (1,437 and 1,275 MW), 2025 report p. 48 (1,327 MW) and its Figure A-88 table, p. A-130, for 2025 forecast-pass days and MWh (https://www.potomaceconomics.com/wp-content/uploads/2026/05/NYISO-2025-SOM-Report__5-19-2026-final.pdf). 2020 report: https://www.nyiso.com/documents/20142/2223763/NYISO-2020-SOM-Report-final-5-18-2021.pdf/c540fdc7-c45b-f93b-f165-12530be925c7.

Reading it. These are upper bounds: real-time under-scheduled load shares the pot, and the residual goes to all withdrawals. On average the uplift adds at most 0.003 to 0.03 USD/MWh to virtual supply and nothing to virtual load. It is lumpy: the whole pot falls on the 24 to 60 forecast-pass days a year, so on such a day a supply position in the affected superzone can pay several tenths of a dollar per MWh (2024: 186,557 USD over 24 days is about 7,800 USD a day). It cannot be predicted at bid time from public data with any precision, so the right treatment is a flat yearly adder, not a model input.

The 0.06 USD/MWh fact checks out with a caveat. The 2024 report says "The overall average rate of virtual profitability remained slightly positive at $0.06 per MWh" (p. 40). That average covers load-zone virtuals and virtual imports and exports at the borders together, and it is gross: "The gross profitability shown here does not account for any other related costs or charges to virtual traders" (footnote 236, p. A-39). For the build years the monitor's gross profit of virtual supply at load zones was minus 0.06 (2020), 0.73 (2021), minus 0.39 (2022) and 0.64 (2023) USD/MWh (2020 report p. 23, 2022 report p. 56, 2024 report p. 40), against exchange costs of 0.09 to 0.15.

## 3. Credit, collateral and participation requirements

The formula. The credit requirement for virtuals is the Virtual Transaction Component of a customer's Operating Requirement: VSCR for all outstanding virtual supply bids, plus VLCR for virtual load bids, plus the "net amount owed to the ISO for settled Virtual Transactions" (NYISO Services Tariff, MST Attachment K, 26.4.2.6, sheets 966 to 971, https://nyisoviewer.etariff.biz/viewerdoclibrary/mastertariffs/9fulltariffnyisomst.pdf). VSCR is the sum over bids of bid MWh times a dollar rate for the bid's group. NYISO sorts every hour into one of 33 virtual supply groups per load zone by season (summer May to August, winter December to February, rest of year), weekday or weekend and holiday, and hour block. The rate for a group is "the price differential between the Energy price in the Day-Ahead Market and the Energy price in the Real-Time Market, at the 98th percentile", over all possible virtual supply positions in that group, with weight one third on the last year and two thirds on the last five years, both ending at the end of the month before the bid month. Virtual load uses the 97th percentile and 28 groups. The requirement applies to bid MWh, not cleared MWh, until the day-ahead market clears; afterwards only the accepted net position counts until it settles. Bids for the same zone and hour on both sides count only the larger side before clearing. Bids that would push the requirement above available credit are rejected as a batch (26.9).

Rule history. The percentile rule above took effect on 12 Sep 2023 (Docket ER23-1307-001; eTariff redline of MST 26.4, https://nyisoviewer.etariff.biz/viewerdoclibrary/Filing/Filing5061/5061FilingSections/MST%2026.4%20FID%205061%20redline_33844.htm). Before that, from 2009, the rate was "the 97th percentile, based upon all possible Virtual Supply positions in the Virtual Supply group for the period of time from April 1, 2005, through the end of the preceding calendar month", an expanding window with no weighting. NYISO's backtest of the change (Management Committee deck of 30 Nov 2022, slide 51, as reported by the credit research agent and not re-read here) found the new design raised total portfolio collateral by about 10 percent over May 2010 to December 2021, so the two rules give requirements of similar size. NYISO's filing in that docket states "The NYISO requires two days of credit support for Virtual and External Transaction positions", that the credit system removes the third day's bids on the morning of the second day unless the prior day's losses are covered, and that "Virtual Bids settle hourly", so each hour leaves the percentile requirement once it is complete in real time and its actual loss moves to the settled amount (deficiency response of 5 Jun 2023, p. 3 footnote 12 and p. 5 footnote 18, https://nyisoviewer.etariff.biz/viewerdoclibrary/Filing/Filing5061/Attachments/20230605_NYISO%20Dfcncy%20Rspns_VrtlExtnlTrnsctns_fnl.pdf). Losses of 50 percent of a customer's virtual credit support trigger a demand for payment or more credit, and at 100 percent NYISO "may cancel any pending Day-Ahead Bids" (MST 26.9).

Settlement lag. NYISO invoices weekly, "On or about each Wednesday", for the previous week, and payment is due "by the second business day after" the invoice (MST 7.2.2.1 and 7.2.2.3, sheets 356 to 357). Settled losses can therefore stand in the requirement for up to about 12 days before they are paid.

What the requirement comes to. Recomputed from the study's 2020 to 2023 prices with the current rule's groups and weights (holidays treated as weekdays; the long window has at most 48 months instead of 60 because local files start in 2020; requirement months January 2023 to January 2024, so every window ends before 2024). Months before September 2023 were in fact billed under the older 97th percentile rule; the table shows what the rule in force for the held-out years implies:

| Requirement month | Per day of bids, full book of 11 zones x 24 hours x 1 MW, weekday (USD) | Same, weekend (USD) | Average per MWh bid, weekday (USD) |
|---|---|---|---|
| Jan 2023 | 34,300 | 44,951 | 130 |
| Feb 2023 | 29,665 | 41,094 | 112 |
| Mar to Apr 2023 | 12,987 to 13,922 | 11,244 to 13,238 | 49 to 53 |
| May to Aug 2023 | 15,045 to 19,945 | 10,962 to 14,119 | 57 to 76 |
| Sep to Nov 2023 | 12,188 to 13,253 | 10,085 to 10,973 | 46 to 50 |
| Dec 2023 | 26,743 | 40,218 | 101 |
| Jan 2024 | 19,412 | 18,597 | 74 |

Per zone, a 24 MWh day of supply bids in January 2024 needs 1,246 USD (West) to 2,576 USD (Capital); Long Island and Capital are the two most expensive zones in 12 of the 13 months (August 2023: Long Island and New York City). Winter 2022 to 2023 is the stressed case because Winter Storm Elliott (24 December 2022) sits in both windows; on that single day the full supply book lost 148,094 USD.

Minimum participation. MST Attachment K 26.1.1 (sheets 922 to 923, effective 26 Jul 2024; identical capitalization text in the 27 Jan 2020 version, https://www.nyiso.com/documents/20142/16885911/05%20MST%20Section%2026.1%20Redline.pdf/6e616499-8403-0346-a63b-f82c131a2877) requires written risk management policies, one-time completion of NYISO's online virtual trading course for every person who bids, operational and financial capability, and capitalization: "at least US $10 million in assets or at least US $1 million in tangible net worth" in audited statements, or else post "$200,000 to participate in any/all of the ISO-Administered Markets other than the TCC market, which security Customer may not use to support any ISO credit requirements". An officer's certificate is due every 30 April (26.1.2). Unsecured credit, which could replace some collateral, needs an investment-grade rating or a NYISO Equivalency Rating of BBB or better from Moody's RiskCalc, plus six months of paying every invoice on time in NYISO or another ISO (MST 26.3.3 and 26.5.1, sheets 935 and 976 to 977), and it is then a small share of tangible net worth (Table K-1). A new small entity therefore posts collateral for everything. NYISO charges no registration fee: "No registration, membership or other fee is required to become a Customer, Limited Customer, or Guest", and registration "may take up to, or more than, 60 days" (Registration Process Overview, https://www.nyiso.com/documents/20142/1390876/Registration%20Process%20Overview.doc/4194a51b-26e0-7496-909f-0510df14103c, as quoted by the credit research agent). No annual fee was found; NYISO may bill the cost of a risk-policy verification if a customer is selected for one (26.1.3.2). NYISO's 2025 virtual trading infographic adds the competency exam and credit evaluation as bidding preconditions and a 999 MW cap per virtual bus per hour (https://www.nyiso.com/documents/20142/49802393/Virtual-Trading-Infographic.pdf/1e72f823-4361-99fd-14c8-edfd4b7de757).

Collateral form and interest. A cash deposit is accepted only from customers "formed or incorporated in, who are residents of, and whose operations are located primarily within the United States or Canada", and "Customers shall receive actual interest earned on cash deposits" (MST 26.6.1.1, sheet 986). Anyone else posts a letter of credit from an approved US or Canadian bank, or a US or Canadian branch of a foreign bank rated at least A, or a surety bond (26.6.1.2 and 26.6.1.3, sheets 986 to 987). A trader resident in Denmark therefore needs either a US entity or a bank letter of credit, which costs a fee and usually needs cash cover at the issuing bank. NYISO also keeps a Working Capital Fund to which each customer contributes pro rata to its gross receivables and payables, rebalanced each January and July, interest-bearing and refundable on exit, and not counted as collateral (OATT Attachment V, 28.3 to 28.9, sheets 1775 to 1779); for a book of this size the share is small (a rough estimate, not verified, is under 10,000 USD). Defaults that become bad debt are spread over all customers by gross receivables and payables (OATT Attachment U, 27.3): a rare tail cost, not a running one.

Realistic collateral for a book of up to 11 zones x 24 hours x 1 MW of supply a day:
- Bid and position requirement: two days of the per-day figure (tomorrow's bids plus today's positions not yet complete), so about 25,000 to 40,000 USD in ordinary months and 60,000 to 90,000 USD in a stressed winter.
- Settled but unpaid losses: up to about 12 days of net losses. In 2020 to 2023 the full book's 12-day loss was worse than 23,319 USD one time in 20 and worse than 52,775 USD one time in 100; the Elliott window reached 195,373 USD.
- Capitalization security: 200,000 USD unless the entity has 1 million USD of audited tangible net worth, and it cannot be used against the credit requirement.

So a small entity running the full book needs about 250,000 to 300,000 USD posted in ordinary times and about 450,000 to 500,000 USD to survive an Elliott-type week without a margin call. Idea A, at 80 to 110 MWh a day, needs roughly 40 percent of the bid and loss parts, but the 200,000 USD security is fixed.

## 4. All-in cost per MWh for virtual supply, 2024 to 2026

| Component | 2024 | 2025 | 2026 | Notes |
|---|---|---|---|---|
| Rate Schedule 1 | 0.1333 | 0.1666 | 0.1919 | section 1 |
| FERC fee share | 0.017 | about 0.02 | about 0.02 | section 1 |
| Forecast-pass BPCG uplift | 0 to 0.017 | 0 to 0.028 | 0 to 0.03 | section 2, upper bounds |
| Exchange subtotal | 0.15 to 0.17 | 0.19 to 0.21 | 0.21 to 0.24 | |
| Cost of capital on collateral | 0 to 0.15 | 0 to 0.15 | 0 to 0.15 | see below |
| All-in | 0.15 to 0.32 | 0.19 to 0.36 | 0.21 to 0.39 | |

Cost of capital. About 220,000 to 275,000 USD stays posted for an A-sized book (200,000 security plus 20,000 to 75,000 credit requirement). A US entity posting cash receives NYISO's actual interest, so its cost is only the gap between that and a Treasury bill, near zero. A foreign entity posting a letter of credit pays an issuance fee; at an assumed 1 to 2 percent a year (typical standby letter-of-credit pricing, not verified for this case) on 275,000 USD and about 36,500 MWh a year, that is 0.08 to 0.15 USD/MWh. Charging the full Treasury bill rate on uncompensated collateral would give about 0.30 USD/MWh, but that only applies if no interest is received, which the tariff rules out for cash. Because the 200,000 USD security is fixed, the capital cost per MWh falls as volume rises: at the full 264 MWh a day it is a third of the figure above.

Conclusion. The realistic all-in cost of virtual supply is 0.15 to 0.40 USD/MWh in 2024 to 2026, with exchange charges alone at 0.15 to 0.24. The registered fee (Rate Schedule 1 only) understates the exchange cost by 0.02 to 0.05 USD/MWh, and the stress runs at 0.50 and 1.00 cover the whole realistic range.

Suggested changes before the freeze, as a dated addendum (not made by this audit): set 2021 to 0.0757 (primary source found); set 2025 to 0.1666 (0.1919 exceeds the tariff maximum); keep 2026 at 0.1919 marked unverified; and add the FERC fee (0.017 USD/MWh, 0.02 from 2025) to the per-MWh fee, since NYISO bills it on the same cleared MWh under Rate Schedule 1. The forecast-pass uplift can be added to supply positions as the yearly upper bounds above or left to the stress runs.

## 5. Bankroll and comparison metrics

Why the return depends on the bankroll. A virtual position buys nothing, so there is no natural capital base. Return on bankroll is the year's net profit divided by whatever capital one decides to count, and defensible choices differ by a factor of five. A year that nets 10,000 USD is 2.0 percent on 500,000 USD (everything a small entity must lock up, below), 3.5 percent on 285,000 USD (collateral and loss buffer without the capitalization security), and 10 percent on 100,000 USD (a buffer sized to ordinary weeks). The percentage carries no information beyond the profit unless the bankroll rule is fixed before the result is known, and even then it should be reported next to measures that do not need a bankroll.

Recommended bankroll, fixed now from the build years only. B = S + C + L, where:
- S = 200,000 USD, the capitalization security NYISO requires from an entity below 1 million USD of audited tangible net worth, which cannot be used against the credit requirement (MST 26.1.1(e)(ii)).
- C = 89,902 USD, two days (NYISO's two-day credit basis) of the highest per-day virtual supply credit requirement for the full book of 11 zones x 24 hours x 1 MW in any requirement month computed above (January 2023, weekend: 44,951 USD).
- L = 195,373 USD, the worst 12-day net loss of that full supply book in 2020 to 2023 (window ending 24 December 2022). Twelve days is the longest a loss can wait for payment under weekly invoicing, so L is the cash needed to meet an Elliott-sized loss without being cut off.
- B = 485,275 USD, rounded up to 500,000 USD.

The rule uses the full book, not idea A's own trades, because A's volume and losses depend on the model and are unknown before the freeze; A trades a subset of the same 264 supply zone-hours, so the full book bounds its collateral and its loss exposure from above. Return on bankroll is then the year's net trading profit after all per-MWh costs divided by 500,000 USD, in USD. Cash collateral earns NYISO's actual interest, so this ratio is a return in excess of cash; for a total return add the Treasury bill yield of the same year.

Bankroll-free measures to report beside it:
- Annualised Sharpe ratio of daily net P&L: mean daily P&L divided by its standard deviation, times the square root of 365. Every calendar day counts, including days with no position, because virtual bids clear seven days a week. No risk-free rate is subtracted, since the P&L is already earned on top of interest-bearing collateral. Give the 95 percent interval from the same stationary bootstrap over days used for profit.
- Return over maximum drawdown: annualised net P&L divided by the largest fall from a peak in cumulative P&L over the held-out run, both in USD, with the drawdown itself stated in USD.
- Build-year reference, for scale only (always-supply full book, net of the corrected exchange cost, from the study's 2020 to 2023 prices): Sharpe minus 0.41 over 2020 to 2023 and plus 1.40 in 2023 alone; cumulative net result minus 163,791 USD; maximum drawdown 347,258 USD from 26 February 2021 to 24 December 2022.

Comparators.

| Benchmark | Period | Return | Volatility | Sharpe | Maximum drawdown | Return over drawdown |
|---|---|---|---|---|---|---|
| S&P 500, long run | 1928 to 2023, annual, dividends included | 9.8 percent a year geometric, 11.7 arithmetic | 19.6 percent | 0.42 (annual excess over 3-month bills, which averaged 3.3 percent) | 56.8 percent, 9 Oct 2007 to 9 Mar 2009, daily closes; 64.8 percent on annual data in 1929 to 1932 (deeper within years) | about 0.17 |
| S&P 500, test period | 2 Jan 2024 to 5 Oct 2026, daily, price only | 23.3 percent (2024), 16.4 (2025), 13.6 (2026 to 5 Oct); 19.3 percent a year | 15.2 percent | 0.96 price only, about 1.05 with dividends added back at 1.3 percent a year | 18.9 percent, 19 Feb to 8 Apr 2025 | about 1.0 |
| US 3-month Treasury bill | yearly averages | 4.97 percent (2024), 4.06 (2025), 3.68 (2026 to 2 Oct) | | risk-free | none | |

Sources: Damodaran, Historical Returns on Stocks, Bonds and Bills, annual table (the html version runs to 2023), https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histret.html ; Federal Reserve Bank of Atlanta, Notes from the Vault, September 2009, closes of 1,565.15 and 676.53, https://www.atlantafed.org/cenfis/publications/notesfromthevault/0909 ; FRED series SP500 (price index, no dividends), DTB3 and DEXDNUS, https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500 (and id=DTB3, id=DEXDNUS), downloaded 6 Oct 2026, computed here.

Danish deposit alternative. Danmarks Nationalbank's current-account rate (official feed, https://www.nationalbanken.dk/interestrates?lang=da&format=xml&typeCodes=FOL; the certificate of deposit rate is identical and the lending rate is 0.15 points higher):

| Effective | Current-account rate (percent) |
|---|---|
| 15 Sep 2023 (level on 1 Jan 2024) | 3.60 |
| 7 Jun 2024 | 3.35 |
| 13 Sep 2024 | 3.10 |
| 18 Oct 2024 | 2.85 |
| 13 Dec 2024 | 2.60 |
| 31 Jan 2025 | 2.35 |
| 7 Mar 2025 | 2.10 |
| 22 Apr 2025 | 1.85 |
| 6 Jun 2025 | 1.60 |
| 12 Jun 2026 | 1.85 |
| 11 Sep 2026 | 2.10 |

Yearly averages: 3.32 percent (2024), 1.85 (2025), 1.73 (2026 to 5 Oct). Households earn less than this: the average rate on Danish private customers' bank deposits fell by 0.69 points during 2025 as customers moved out of fixed-term deposits (Nationalbanken, 26 Feb 2026, https://www.nationalbanken.dk/en/news-and-knowledge/data-and-statistics/banking-and-mortgage-lending/lending/20260226-falling-interest-rates-increase-danes-net-interest-expenses) and stood at about 1.2 to 1.3 percent from late 2025 to early 2026 (Trading Economics series of Nationalbanken data, secondary, not checked at the statbank). Ordinary demand accounts pay less than the average, which includes fixed-term deposits.

Currency. The strategy earns and posts collateral in US dollars; the deposit rate is in kroner. Two differences follow. First, dollar cash simply paid more: the Treasury bill averaged 1.65, 2.22 and 1.95 points above the Nationalbank rate in 2024, 2025 and 2026, so part of any dollar total return is the interest gap, not skill. Second, an unhedged dollar bankroll carries exchange risk larger than any plausible trading return: USD/DKK ranged 6.67 to 7.21 in 2024, 6.30 to 7.31 in 2025 and 6.23 to 6.66 in 2026 (FRED DEXDNUS), and the dollar fell about 13 percent against the krone between its January 2025 high and the end of 2025. The like-for-like comparison is therefore in excess returns: the strategy's profit over bankroll (already a return above dollar cash) against the S&P 500's return above Treasury bills and the Danish deposit rate's spread over the Nationalbank rate (about zero or negative). A Danish investor who hedged the dollars would earn roughly the dollar excess return plus the krone rate. Report the dollar figures and state the exchange rate moves separately, without converting P&L at spot.

What to publish for each idea: net profit in USD and per MWh, return on the 500,000 USD bankroll (in excess of cash), annualised Sharpe of daily P&L with its interval, return over maximum drawdown with the drawdown in USD, and, in one line, the comparators: S&P 500 Sharpe 0.42 long run and about 1.0 in 2024 to 2026, drawdowns 56.8 and 18.9 percent; Nationalbank rate 3.32, 1.85 and 1.73 percent by year; household deposits about 1.2 to 1.3 percent.

## Open items

- 2025 and 2026 Rate Schedule 1 postings were not found at nyiso.com; 2025 rests on the tariff cap and a secondary report, 2026 on CONTRACT.md.
- FERC fee pots for October 2024 onward were not found; 0.02 USD/MWh is a planning value.
- The 2025 forecast-pass uplift in dollars is not published in the 2025 State of the Market report; the range is scaled.
- The credit table approximates the five-year window with four years and treats holidays as weekdays.
- Letter-of-credit pricing (1 to 2 percent a year) is an assumption, not a quoted price.
- An independent benchmark research agent had not returned when this was written; the S&P 500, bill, exchange rate and Nationalbank figures above were computed or read here from the cited primary sources.
