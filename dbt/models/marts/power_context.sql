-- Price with the physics next to it. The desk question this answers is
-- the obvious one and still the useful one: how much of the hour's
-- consumption was wind, and what did the hour clear at.
with p as (
    select area, hour_utc, price_eur from {{ ref('power_hourly') }}
),
g as (
    select * from {{ ref('stg_production') }}
)
select
    p.area,
    p.hour_utc,
    p.price_eur,
    g.offshore_wind_mwh + g.onshore_wind_mwh                       as wind_mwh,
    g.solar_mwh,
    g.thermal_mwh,
    g.consumption_mwh,
    case when g.consumption_mwh > 0
         then (g.offshore_wind_mwh + g.onshore_wind_mwh) / g.consumption_mwh
    end                                                            as wind_share
from p
join g on g.area = p.area and g.hour_utc = p.hour_utc
