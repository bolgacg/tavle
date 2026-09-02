-- The per-hour tolerance catches broken hours; it does not catch a feed
-- that quietly reads five percent high for a month. This test sums the
-- last thirty settled days: real-time minus settled, over settled. Beyond
-- the band it warns. Warn, not error: the board stays up, the versions
-- page says the calibration moved, and a person decides.
{{ config(severity = 'warn') }}
with span as (
    select max(hour_utc) as hi from {{ ref('wind_versions') }} where v_settled is not null
),
recent as (
    select area, v_realtime, v_settled
    from {{ ref('wind_versions') }}, span
    where v_settled is not null and v_realtime is not null and realtime_readings = 12
      and hour_utc >= span.hi - interval 30 day
),
bias as (
    select area, count(*) as hours, 100.0 * sum(v_realtime - v_settled) / sum(v_settled) as bias_pct from recent group by 1
)
select area, hours, round(bias_pct, 2) as bias_pct
from bias
where hours > 0 and abs(bias_pct) > {{ var('versions_bias_band_pct') }}
