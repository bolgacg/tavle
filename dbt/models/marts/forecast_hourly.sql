-- One row per zone per hour: what was forecast at each horizon, what
-- actually happened, and the errors. Wind is offshore plus onshore.
with f as (
    select area, hour_utc,
        sum(fc_day_ahead_mwh) filter (where forecast_type like '%Wind%')  as wind_fc_da_mwh,
        sum(fc_intraday_mwh)  filter (where forecast_type like '%Wind%')  as wind_fc_id_mwh,
        sum(fc_5h_mwh)        filter (where forecast_type like '%Wind%')  as wind_fc_5h_mwh,
        sum(fc_1h_mwh)        filter (where forecast_type like '%Wind%')  as wind_fc_1h_mwh,
        sum(fc_day_ahead_mwh) filter (where forecast_type like '%Solar%') as solar_fc_da_mwh
    from {{ ref('stg_forecasts') }}
    group by 1, 2
),
a as (
    select area, hour_utc, price_eur, wind_mwh as wind_actual_mwh, solar_mwh as solar_actual_mwh,
           consumption_mwh
    from {{ ref('power_context') }}
)
select
    a.area, a.hour_utc, a.price_eur,
    f.wind_fc_da_mwh, f.wind_fc_id_mwh, f.wind_fc_5h_mwh, f.wind_fc_1h_mwh, f.solar_fc_da_mwh,
    a.wind_actual_mwh, a.solar_actual_mwh, a.consumption_mwh,
    a.wind_actual_mwh - f.wind_fc_da_mwh           as wind_err_da_mwh,
    a.wind_actual_mwh - f.wind_fc_1h_mwh           as wind_err_1h_mwh,
    case when a.consumption_mwh > 0 then f.wind_fc_da_mwh / a.consumption_mwh end as wind_fc_share
from a
join f using (area, hour_utc)
