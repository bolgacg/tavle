-- One schema for every instrument the desk looks at, at native resolution.
-- Power is stitched at the 1 Oct 2025 seam: hourly before, quarter-hourly
-- after. FX is daily. `instrument` is the only key a downstream tool needs.
with power as (
    select area || '_DA' as instrument, area, interval_start_utc, interval_minutes,
           price_eur as value, 'EUR/MWh' as unit, 'Elspotprices' as source, _fetched_at
    from {{ ref('stg_elspot') }}
    where interval_start_utc < timestamp '2025-09-30 22:00:00'
    union all
    select area || '_DA', area, interval_start_utc, interval_minutes,
           price_eur, 'EUR/MWh', 'DayAheadPrices', _fetched_at
    from {{ ref('stg_dayahead') }}
    where interval_start_utc >= timestamp '2025-09-30 22:00:00'
),
fx as (
    select denominator || currency as instrument, null as area,
           cast(rate_date as timestamp) as interval_start_utc, 1440 as interval_minutes,
           rate as value, currency || ' per ' || denominator as unit, 'ECB' as source, _fetched_at
    from {{ ref('stg_fx') }}
)
select * from power
union all
select * from fx
