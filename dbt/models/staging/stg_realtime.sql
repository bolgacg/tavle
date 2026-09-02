-- The real-time feed: wind and solar in megawatts every five minutes,
-- upscaled from SCADA measurements. Energinet's own note on the dataset:
-- "Errors will occur, and will generally not be corrected." It is the
-- number a desk sees first and the only one that is never revised, which
-- makes it the fifth of the six versions of an hour of wind.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/ElectricityProdex5MinRealtime/*.parquet', union_by_name = true)
)
select
    PriceArea                              as area,
    cast(Minutes5UTC as timestamp)         as minute_utc,
    cast(OnshoreWindPower as double)       as onshore_wind_mw,
    cast(OffshoreWindPower as double)      as offshore_wind_mw,
    cast(SolarPower as double)             as solar_mw,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, Minutes5UTC order by _fetched_at desc) = 1
