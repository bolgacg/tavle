-- The same six versions as one row each, with a version column: the shape
-- a board or a test wants, where "which number is this" is data, not a
-- column name. Rank is the order the versions exist in; the first four
-- can be traded on, the last two only settled against.
with u as (
    select area, hour_utc, version, value_mwh
    from {{ ref('wind_versions') }}
    unpivot (value_mwh for version in (
        v_day_ahead as 'day-ahead', v_intraday as 'intraday', v_5h as 'five-hour',
        v_1h as 'one-hour', v_realtime as 'real-time', v_settled as 'settled'))
)
select
    u.area, u.hour_utc, u.version,
    case u.version when 'day-ahead' then 1 when 'intraday' then 2 when 'five-hour' then 3
                   when 'one-hour' then 4 when 'real-time' then 5 else 6 end as rank,
    u.version in ('day-ahead', 'intraday', 'five-hour', 'one-hour')             as tradable,
    u.value_mwh,
    w.v_settled                                                                 as settled_mwh,
    u.value_mwh - w.v_settled                                                   as diff_vs_settled_mwh
from u
join {{ ref('wind_versions') }} w using (area, hour_utc)
where u.value_mwh is not null
