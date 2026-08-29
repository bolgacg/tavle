-- The dataset's TimestampUTC is the time the row was last updated, which
-- for most hours is the last horizon revision close to delivery, not the
-- moment the day-ahead forecast was issued. So it cannot prove the
-- day-ahead column was known before the auction; that rests on Energinet's
-- documentation of the column, quoted on the page. What the timestamp CAN
-- do is catch a row with no provenance at all.
select area, hour_utc
from {{ ref('stg_forecasts') }}
where published_utc is null and fc_day_ahead_mwh is not null
