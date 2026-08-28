-- Every day except the first and the last of the series must have a full
-- set of hours (23, 24 or 25 across the DST changes). The edges are
-- partial by construction, which is why they are excluded here instead of
-- the test being weakened to tolerate a missing hour anywhere.
with bounds as (
    select area, min(day_dk) as first_day, max(day_dk) as last_day
    from {{ ref('desk_daily') }} group by 1
)
select d.area, d.day_dk, d.hours
from {{ ref('desk_daily') }} d
join bounds b using (area)
where d.day_dk > b.first_day and d.day_dk < b.last_day
  and not d.is_complete
