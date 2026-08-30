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
