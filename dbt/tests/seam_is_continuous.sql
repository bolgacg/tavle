-- The last Elspotprices hour and the first DayAheadPrices hour must be
-- adjacent per area; anything else means one side of the seam is missing.
with last_old as (
    select area, max(interval_start_utc) as t from {{ ref('stg_elspot') }} group by 1
),
first_new as (
    select area, min(interval_start_utc) as t from {{ ref('stg_dayahead') }} group by 1
)
select o.area, o.t as last_hourly, n.t as first_quarter_hourly
from last_old o join first_new n using (area)
where n.t - o.t <> interval 1 hour
