-- Every hour between the first and last stitched hour must exist, per area.
-- A missing hour is a failed fetch or a seam mistake, not a market fact.
with bounds as (
    select area, min(hour_utc) as lo, max(hour_utc) as hi from {{ ref('power_hourly') }} group by 1
),
expected as (
    select b.area, g.h as hour_utc
    from bounds b, lateral (select unnest(generate_series(b.lo, b.hi, interval 1 hour)) as h) g
)
select e.area, e.hour_utc
from expected e
left join {{ ref('power_hourly') }} p using (area, hour_utc)
where p.hour_utc is null
