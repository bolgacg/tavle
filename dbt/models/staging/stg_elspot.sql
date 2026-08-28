-- Hourly day-ahead prices, the dataset Energinet retired on 30 Sep 2025.
-- Re-fetched windows overlap on purpose; the latest fetch wins per hour.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/Elspotprices/*.parquet', union_by_name = true)
)
select
    PriceArea                        as area,
    cast(HourUTC as timestamp)       as interval_start_utc,
    60                               as interval_minutes,
    cast(SpotPriceEUR as double)     as price_eur,
    cast(SpotPriceDKK as double)     as price_dkk,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, HourUTC order by _fetched_at desc) = 1
