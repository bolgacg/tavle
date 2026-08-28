-- The page the desk opens at 08:00: one row per day per area.
with hourly as (
    select area, hour_utc, price_eur,
           cast(hour_utc at time zone 'UTC' at time zone 'Europe/Copenhagen' as date) as day_dk
    from {{ ref('power_hourly') }}
),
daily as (
    select
        area, day_dk,
        avg(price_eur)                                          as avg_eur,
        min(price_eur)                                          as low_eur,
        max(price_eur)                                          as high_eur,
        arg_min(price_eur, hour_utc)                            as open_eur,
        arg_max(price_eur, hour_utc)                            as close_eur,
        count(*)                                                as hours,
        count(*) filter (where price_eur < 0)                   as negative_hours
    from hourly
    group by 1, 2
),
spread as (
    select a.day_dk, a.avg_eur - b.avg_eur as spread_eur
    from daily a join daily b on a.day_dk = b.day_dk and a.area = 'DK1' and b.area = 'DK2'
),
fx as (
    select rate_date, rate as eurdkk
    from {{ ref('stg_fx') }} where currency = 'DKK'
)
select
    d.*,
    -- the first and last day of any window are partial by construction, and
    -- so is today; a daily bar built from four hours is not a daily bar, so
    -- it is flagged rather than silently averaged
    d.hours in (23, 24, 25)                                     as is_complete,
    s.spread_eur,
    f.eurdkk,
    d.avg_eur * f.eurdkk as avg_dkk
from daily d
left join spread s on s.day_dk = d.day_dk
left join fx f on f.rate_date = d.day_dk
