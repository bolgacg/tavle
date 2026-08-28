-- Power on one hourly grid across the seam: hourly rows pass through,
-- quarter-hourly rows average to the hour they belong to.
select
    area,
    date_trunc('hour', interval_start_utc)      as hour_utc,
    avg(value)                                  as price_eur,
    count(*)                                    as n_intervals,
    min(source)                                 as source
from {{ ref('prices') }}
where unit = 'EUR/MWh'
group by 1, 2
