-- Twelve five-minute readings make an hour. The feed has outages Energinet
-- does not backfill (its own note: errors "will generally not be
-- corrected"), so single missing hours are a fact to flag, not a failure.
-- More than five percent of hours short in the feed's last thirty days is
-- a failure: the board would be averaging half-hours and calling them hours.
-- Measured against the feed's own last hour, not the clock, so the test
-- runs for real on the committed sample as well as on live data.
with span as (
    select max(hour_utc) as hi from {{ ref('wind_versions') }} where v_realtime is not null
),
recent as (
    select area, hour_utc, realtime_readings
    from {{ ref('wind_versions') }}, span
    where hour_utc >= span.hi - interval 30 day and hour_utc < span.hi
),
rate as (
    select area, count(*) filter (where realtime_readings < 12) as short_hours, count(*) as hours
    from recent group by 1
)
select area, short_hours, hours
from rate
where short_hours > 0.05 * hours
