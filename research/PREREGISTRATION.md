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
