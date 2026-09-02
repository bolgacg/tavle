-- Six versions of one hour of wind, side by side on one grid.
--
-- An hour of Danish wind is published six times. Four are forecasts from
-- Energinet's Forecasts_Hour (the evening before, the same morning, five
-- hours ahead, one hour ahead). The fifth is the real-time feed, twelve
-- five-minute megawatt readings averaged to the hour. The sixth is the
-- settlement, metered and published nine to fifteen days later, then
-- revised for up to two years. The grid is the union of every hour any
-- version exists for, so a missing version is a null the tests can see,
-- not a row that quietly vanished from a join.
with f as (
    select area, hour_utc,
        sum(fc_day_ahead_mwh) as v_day_ahead,
        sum(fc_intraday_mwh)  as v_intraday,
        sum(fc_5h_mwh)        as v_5h,
        sum(fc_1h_mwh)        as v_1h
    from {{ ref('stg_forecasts') }}
    where forecast_type like '%Wind%'
    group by 1, 2
),
r as (
    select area, date_trunc('hour', minute_utc) as hour_utc,
        avg(onshore_wind_mw + offshore_wind_mw) as v_realtime,
        count(*)                                as realtime_readings
    from {{ ref('stg_realtime') }}
    where onshore_wind_mw is not null and offshore_wind_mw is not null
    group by 1, 2
),
s as (
    select area, hour_utc, offshore_wind_mwh + onshore_wind_mwh as v_settled
    from {{ ref('stg_production') }}
),
hours as (
    select area, hour_utc from f
    union select area, hour_utc from r
    union select area, hour_utc from s
),
joined as (
    select
        h.area, h.hour_utc,
        f.v_day_ahead, f.v_intraday, f.v_5h, f.v_1h,
        r.v_realtime, coalesce(r.realtime_readings, 0) as realtime_readings,
        s.v_settled,
        (f.v_day_ahead is not null)::int + (f.v_intraday is not null)::int + (f.v_5h is not null)::int
          + (f.v_1h is not null)::int + (r.v_realtime is not null)::int + (s.v_settled is not null)::int
                                                    as versions_present
    from hours h
    left join f using (area, hour_utc)
    left join r using (area, hour_utc)
    left join s using (area, hour_utc)
)
-- a forecast row with every value null is a row, not a version
select * from joined where versions_present > 0
