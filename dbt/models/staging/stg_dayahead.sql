-- Quarter-hourly day-ahead prices, the successor dataset from 1 Oct 2025.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/DayAheadPrices/*.parquet', union_by_name = true)
)
select
    PriceArea                          as area,
    cast(TimeUTC as timestamp)         as interval_start_utc,
    15                                 as interval_minutes,
    cast(DayAheadPriceEUR as double)   as price_eur,
    cast(DayAheadPriceDKK as double)   as price_dkk,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, TimeUTC order by _fetched_at desc) = 1
