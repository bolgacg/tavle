# Pre-registration: does the wind forecast already know tomorrow's price?

Written 29 August 2026, before any result in this study was computed. The questions, the metrics,
the holdout and the cost model below are frozen; anything added later is labelled as added later.

## Data

- Energinet, Forecasts_Hour: wind (offshore, onshore) and solar forecasts per hour per zone, at the
  day-ahead horizon (available before the auction closes at 12:00 CET the day before delivery) and
  at intraday, five-hour and one-hour horizons. History from November 2019.
- tavle's power_hourly (day-ahead clearing price, DK1 and DK2, hourly grid across the October 2025
  seam) and power_context (actual wind, solar, consumption).
- Energinet, RegulatingBalancePowerdata: hourly imbalance price, to March 2025. Its quarter-hourly
  successor is out of scope for this pre-registration.

## Questions

Q1 (day-ahead). How much of the variation in the day-ahead price, per zone, is explained by the
day-ahead wind forecast expressed as a share of expected consumption? Same for the DK1 minus DK2
spread. Metric: out-of-sample R squared of a linear model with hour-of-day and month fixed effects,
against the same model without the forecast. Reported per year.

Q2 (forecast quality). The error of the day-ahead wind forecast (actual minus forecast) per zone:
mean, median absolute error, its dependence on the forecast level and the hour, and how much of it
the one-hour-ahead forecast removes. Metric: MAE and bias, by forecast decile and by hour.

Q3 (intraday). When the day-ahead forecast is wrong, which way does the imbalance price go relative
to the day-ahead price? Metric: sign agreement between (actual minus forecast) and (imbalance price
minus day-ahead price), and the mean of the price gap conditional on error deciles. Prediction
written now: more wind than forecast pushes the imbalance price below the day-ahead price; less
wind pushes it above.

Q4 (the paper strategy). A rule stated now and not tuned later: on days where the day-ahead wind
forecast share differs between DK1 and DK2 by more than the 75th percentile of the training period,
take the spread in the direction the forecast implies (higher wind share means lower price) for
the delivery day, one unit, no leverage. Scored on the holdout only.

## Holdout and costs

- Training period: November 2019 to December 2023. Holdout: January 2024 to the end of the data.
  The holdout is read exactly once, after Q1 to Q4 are implemented and run on training.
- Costs for Q4: the day-ahead spread is a financial position taken through two zone prices; the
  cost model is a fixed 0.5 EUR/MWh round trip plus the observed bid-ask proxy of 0.1 EUR/MWh,
  applied per unit per day. If the result is within one standard error of zero after costs it is
  reported as null.

## What would make me discard the whole thing

A seam artefact (the October 2025 dataset switch) driving any result; a forecast timestamp that
turns out to be published after the auction close for some part of the history; fewer than 2,000
holdout hours per zone.

## Added 29 August, after the first look at the raw forecast rows (before any result)

The dataset's TimestampUTC is the row's last-update time, typically minutes before delivery, not
the issue time of the day-ahead column. The original discard clause ("a forecast timestamp published
after the auction close") therefore cannot be tested from this field and is replaced by: the
day-ahead column's provenance rests on Energinet's documentation of ForecastDayAhead, which is
quoted on the page; if that documentation does not state that the value is fixed before the
day-ahead auction, Q1 and Q4 are reported as "forecast horizon unverified" rather than as findings.

## Added 29 August, from Energinet's column documentation (still before any result)

Energinet's metadata for Forecasts_Hour states, verbatim: ForecastDayAhead is "Forecast for the
next day is published at 18:00 Danish time zone. The forecast is generated at 17:50 Danish time
zone." ForecastIntraday is "The forecast for the coming day at 6am Danish time zone." The
day-ahead auction closes at 12:00 CET the day before delivery and publishes its prices around
12:45. So the "day-ahead" forecast in this dataset is issued about six hours AFTER the price it
would supposedly predict. Consequences, fixed now:

- Q1 is reframed. It no longer asks whether the forecast predicts the price (it cannot, it came
  later); it asks how much of the day-ahead price is explained by expected wind, using the 18:00
  forecast as the best public estimate of what the auction was pricing. The number is an
  explanatory R squared, and the page must say so in those words.
- Q2 stands: forecast quality by horizon (18:00 day-before, 06:00 same-day, five-hour, one-hour).
- Q3 stands and becomes the centre: every horizon is published before delivery, and the imbalance
  price is settled after delivery, so "when the forecast is wrong, which way does the price of
  being wrong go" has no look-ahead in it.
- Q4 as pre-registered CANNOT BE TESTED: its signal is published after the auction that sets the
  price it trades. It is reported as exactly that, not silently replaced. A model that "uses the
  day-ahead wind forecast to predict the day-ahead price" on this dataset has look-ahead bias, and
  the page says so, because that is the most useful thing on it for a desk.
- Q4' (added, labelled as added): the 06:00 revision (intraday minus 18:00 forecast) as a signal
  for the sign of (imbalance price minus day-ahead price) in the delivery hours. Scored as sign
  agreement and mean gap on the holdout only, with the same costs. It is a proxy for an intraday
  position because intraday prices are not in the public data; the page says so.

---

# Pre-registration v2, 30 August 2026: who pays when the wind forecast is wrong

## Why a second registration

The v1 headline question (trade the day-ahead spread on the day-ahead wind forecast) turned out to be
untestable on this data: the forecast is published six hours after the auction. v1 stands as written;
this section registers four hypotheses that the data can test without look-ahead, each with a mechanism,
a prediction, a baseline and a metric, written before the final computation.

**Disclosure.** On 30 August 2026, before this section was written, a one-off exploratory pass was run on
the whole data (training and holdout together) to check that H1 to H4 were alive. The numbers seen were:
H1 DK1 cost 1.64 / 0.87 / 0.71 / -0.05 EUR per MWh by horizon; H2 zero-gap share 0 to 33 percent; H3
sign persistence 0.86 at one hour; H4 R squared change at most 0.007. The holdout is therefore not
unread for v2. It is kept as a separate period so the reader can see whether the effects hold out of the
training years, and this page never calls it unread.

## Data and periods

forecast_vs_imbalance joined with forecast_hourly (all four forecast horizons, actual wind, day-ahead
price, imbalance price) and power_context (solar, consumption). Training: December 2019 to December
2023. Holdout: January 2024 to 4 March 2025, the end of the imbalance data. Single-pricing regime from
1 November 2021 (Nordic balancing model, one price for all imbalances).

## H1. Lead time has a price

Mechanism: a producer nominates its forecast in the auction and settles the difference at the imbalance
price; a later forecast has a smaller error and therefore less imbalance volume.
Definition: producer output A = s times the zone's actual wind, nomination N_h = s times the forecast at
horizon h (evening before, same morning, five hours, one hour), s = 0.05 by default. Imbalance cost for
the hour = -(A - N_h)(I - P), I the imbalance price, P the day-ahead price. Metric: summed cost divided by
summed output, EUR per MWh produced, per zone, horizon and period; also imbalance volume as a share of
output. Baseline: the evening-before nomination. Prediction: cost falls monotonically with horizon in both
zones. Caveat registered: re-nominating requires trading the difference intraday at prices this data does
not have; the avoided imbalance cost is an upper bound on the value of the later forecast.

## H2. Single pricing changed what an error costs

Mechanism: from 1 November 2021 every imbalance settles at one price; an error in the helping direction is
paid rather than charged, and an hour with no regulation costs nothing.
Metric: per zone and regime, the mean gap (imbalance price minus day-ahead price) when more wind arrived
than forecast and when less, the share of hours with a gap of exactly zero, and the mean absolute gap.
Prediction: the zero share rises from about zero to a substantial fraction of hours; the conditional means
keep their signs (negative when more wind, positive when less) in both regimes. Caveat registered: the
published imbalance price series is used in both eras; in the dual era production imbalances in the wrong
direction were settled at the regulating price, so the dual-era figures describe the published series, not
a producer's bill.

## H3. The balancing direction persists, and the wind error is not the signal

Mechanism: the causes of a system imbalance last for hours.
Metric: among hours where the gap is nonzero at t and t-k, the share where sign(gap_t) = sign(gap_t-k),
k = 1, 2, 3, 6. Baselines: always guessing the majority sign of the period; the v1 rule
sign(gap_t) = -sign(wind error_t-k). Costed rule, k = 1: take one unit long if the previous hour's gap was
positive, short if negative; P&L per hour = predicted sign times gap minus 0.6 EUR/MWh (the v1 cost).
Report mean, standard error, median, hit rate, share of the total from the best decile, per zone and
period. Prediction: persistence above 0.8 at k = 1, falling with k, and well above both baselines.
Caveats registered: the settled gap for t-1 is not known at t; the live signal is the activation
direction Energinet publishes; and deliberate imbalances are prohibited, so the legitimate use is an
intraday position with the H1 caveat on prices.

## H4. The market prices Energinet's wind, not the wind

Mechanism: if the auction had better information than Energinet's later forecast, the price would track
actual wind beyond the forecast.
Metric: per zone and year, the price model of v1 (hour and month effects plus forecast share and its
square) with and without an added term (actual minus forecast share); the coefficient on the added term
and the change in R squared. Prediction: the change in R squared is below 0.01 in every year. Companion
descriptives: the day-ahead forecast bias by year as a share of actual wind, with Energinet's
installed-capacity steps; negative-price hours per year with their midday share and the mean solar and
wind in those hours. Prediction: bias rises in DK1 from 2023 with the capacity steps; negative-price hours
move from night to midday as solar grows.

**Added after v2 was registered (30 August 2026, same day):** the H3 costed rule is also reported at k = 2,
because the intraday gate closes an hour before delivery, so a position for hour t can only be placed when
hour t-2 is the last settled hour. Labelled as added on the page.

**Added 10 September 2026, after a cold read from a trading desk's side (labelled as added; no number changed):**
(1) H1's "one hour ahead" nomination is information, not action: Energinet releases the one-hour forecast up to 15
minutes before the hour and the intraday gate closes 60 minutes before, so the five-hour forecast is the last one a
position can follow; the page now reports evening-to-five-hours as the actionable difference and evening-to-one-hour
as the value of the information. (2) H4's zero gain cannot distinguish a market that shares Energinet's bias from one
that corrects it by a constant, since the coefficient on the forecast absorbs any rescaling; the page no longer says
the market "misses with" the forecast. (3) H4 is an easy test: the 17:50 forecast is six hours newer than anything the
auction had, so the prediction could hardly fail; it is reported as a check. (4) The hourly imbalance series
(RegulatingBalancePowerdata) ends 4 March 2025 11:00 UTC; its 15-minute successor (Imbalance Price) starts 11:15 the
same day and is out of scope, as registered above.

---

# The decision machine, registered 10 September 2026, evening (research/decision.py)

One intraday position per hour, run end to end on the hourly balancing data used above: a registered
rule, a learned gate, guards that refuse, a replay against an honest clock, and a fault injector.

**Disclosure.** A first dry run the same evening used a stricter age limit (3), an "edge floor"
guard and an override role for the learned gate, and printed separate-period numbers before the
choices below were fixed on the even weeks. Those numbers were: DK1 rule 8.55, learned alone 8.59 EUR
per unit hour, 4,511 of 10,284 hours refused. The choices below were then made on the even weeks of
the single-pricing training years only, and the separate period was computed once more with them.

- Signal: the sign of the last settled nonzero gap (imbalance price minus day-ahead price). At the
  gate, one hour before delivery, the latest settled hour is t-2. Age = hours since the hour that gave
  the sign. Rule: take that sign, one unit; P&L = sign times gap minus 0.6 EUR, the v1 cost.
- Guards (default, decided on the even weeks): the settled hour t-2 must exist (feed); that hour must
  have had a direction, a nonzero gap (when it was zero, trading on the last sign before it lost money
  on the even weeks); the feed may be late by at most the number of hours at which the rule still keeps half of
  its fresh-feed value on the even weeks (the lag table; the limit is set per zone); the hour must be in the
  single-pricing regime. Not adopted: an "edge floor" guard (median |gap| of the last
  six settled hours above the cost), because on the even weeks the hours it would refuse paid about
  11 EUR each; it stays on the page as a fault. A capacity-step-month guard is also a fault, not a default.
- Learned gate: a two-layer network (10 inputs listed in the model card, 16 tanh units, PyTorch), fit
  on the odd ISO weeks of November 2021 to December 2023, measured on the even weeks. Two roles are
  measured. Override: where the gate is confident, does its sign beat the rule's? Veto: the gate's
  probability that the rule's direction is right, P, must clear a threshold q or the hour is refused.
  Authority is earned on the even weeks: the veto threshold is the largest q on the grid 0.50 to 0.80
  (step 0.05) at which the kept hours' total P&L is not below the no-veto total; the override role is
  granted only where the gate's sign beats the rule's by two points or more, else not at all.
- Separate period, January 2024 to 4 March 2025, computed once with the choices above. Reported per
  arm: hours traded and refused by reason, hit rate, mean and median P&L per unit hour, standard
  error, best-decile share, and the mean of what the refused hours would have done under the rule.
- Faults (act three): the settled feed one hour late with and without the age guard; a synthetic
  nightly outage (00:00 to 05:59); the regime guard off; the edge guard on; the capacity-step guard on;
  the age guard off. Each reported with the same metrics.
- Everything on the page is an upper bound settled at the imbalance price; a deliberate imbalance is
  prohibited; no intraday prices are public.

---

# The battery, registered 11 September 2026, 00:10 (research/activation.py)

What a prequalified one-megawatt, two-megawatt-hour battery earns in Danish balancing activations as a
price taker, at the prices Energinet actually paid, and whether the last settled direction helps.

- Data: Energinet's hourly regulating power series (December 2019 to 4 March 2025): the up-regulation
  price paid to an activated megawatt, the down-regulation price an activated megawatt pays for the
  energy it takes, activated mFRR volumes, the imbalance price; the day-ahead price for the hour. An hour
  counts as up-activated when the up price exceeds the day-ahead price or up volume was activated; down
  likewise. The dominating direction is the side the imbalance price settled on.
- Asset: 1 MW, 2.2 MWh usable (a two-hour-class store with room for a full charge and a full delivery after
  losses; at exactly 2.0 the state machine has a dead zone, found in the first run and fixed before any period was
  read), 0.949 one-way efficiency (0.90 round trip), cycle cost 30 EUR per MWh delivered, starting at 1.1 MWh. Whole-hour, whole-megawatt bids (the Danish minimum). Price taker: a bid at or below
  the marginal price counts as activated; the marginal bid's partial activation and location limits are ignored
  and said so.
- Decision each hour, from information settled by t-2 and the day-ahead price for t: offer up-regulation at
  the day-ahead price plus a margin m_up when the store holds a full delivery; offer down-regulation at the
  day-ahead price minus m_down when there is room. Cash: an activated discharge earns the up price minus the
  cycle cost; an activated charge pays the down price (negative prices pay the battery). When both directions
  were activated in an hour, only the dominating one can activate the battery.
- Machine A (flat): one pair (m_up, m_down) from the grid {0, 5, 10, 20, 40, 80} EUR, chosen as the pair
  with the highest net on the even ISO weeks of November 2021 to December 2023, the battery state running
  through all hours. Machine B (direction-aware): a pair per last settled direction (up, none, down), chosen
  on the same even weeks by one pass of coordinate search from A's pair. Baseline: always offer at cost
  (both margins zero).
- Separate period, January 2024 to 4 March 2025, replayed once: net per MW-year, discharges and charges,
  the mean price captured, and the cycle-cost sensitivity (0, 15, 30, 60). Verdict ruler: a two-hour battery
  costs roughly 30 to 60 thousand euros per MW-year; the page says whether this market alone would pay it.
- Prediction, written now: the market pays on the order of tens of thousands of euros per MW-year in 2024;
  the direction-aware pairs beat the flat pair on the even weeks by construction and by less, or not at all, on
  the separate period. Everything else is reported as found.

**Revision after an audit, 11 September 2026 (labelled as revision; every number seen is listed).**
The first published version flagged an hour as activated on the price signal alone. An audit found that
in DK2 a quarter of the battery's charge hours and 3.7 percent of its discharge hours had no mFRR energy
activated at all (DK1: none of its discharges, 1.2 percent of charges), so the price there came from the
automatic reserve or special regulation and a 1 MW mFRR bid had nothing to be activated into. The flag now
also requires activated mFRR energy in the hour. The margins were re-chosen on the even weeks under the new
flag (+40/-5 in DK1, +80/-0 in DK2) and the separate period recomputed: flat per MW-year 51,702 in DK1 and
54,600 in DK2, against 52,739 and 55,319 before. For the record, the separate period had also been printed
twice before this revision: once in the dead-zone run (zeros) and once more when prices were rounded to two
decimals for the browser replay, which alone moved DK1 from 51,852 to 52,739 while the flat pair stayed
+40/-10 in both zones. A partial-hour delivery proxy is added as a fault, not a default: delivery scaled by
the system's activated mFRR volume over 60 MWh, capped at one, since an activation under 60 MWh cannot have
run at 60 MW for the whole hour. Under it the separate net falls about 3 percent in DK1 and 19 percent in
DK2. Concentration is now reported: the best decile of active hours carries about half of the net.
