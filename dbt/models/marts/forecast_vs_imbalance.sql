-- Study 2's table: the day-ahead forecast error next to the price of
-- being wrong in that hour.
select
    h.area, h.hour_utc, h.price_eur,
    h.wind_err_da_mwh, h.wind_err_1h_mwh, h.wind_fc_da_mwh, h.wind_actual_mwh,
    i.imbalance_price_eur, i.imbalance_mwh,
    i.imbalance_price_eur - h.price_eur as imbalance_minus_spot_eur
from {{ ref('forecast_hourly') }} h
join {{ ref('stg_imbalance') }} i using (area, hour_utc)
