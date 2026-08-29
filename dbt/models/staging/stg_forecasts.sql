-- Energinet's own wind and solar forecasts, one row per hour per zone per
-- type, with several horizons side by side. ForecastDayAhead is the number
-- a desk had before the day-ahead auction closed; the others arrive later.
-- Rows land wide from the source; this model keeps them that way and types
-- them, and forecast_hourly pivots the types into columns.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/Forecasts_Hour/*.parquet', union_by_name = true)
)
select
    PriceArea                            as area,
    cast(HourUTC as timestamp)           as hour_utc,
    ForecastType                         as forecast_type,
    cast(ForecastDayAhead as double)     as fc_day_ahead_mwh,
    cast(ForecastIntraday as double)     as fc_intraday_mwh,
    cast(Forecast5Hour as double)        as fc_5h_mwh,
    cast(Forecast1Hour as double)        as fc_1h_mwh,
    cast(ForecastCurrent as double)      as fc_current_mwh,
    cast(TimestampUTC as timestamp)      as published_utc,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, HourUTC, ForecastType order by _fetched_at desc) = 1
