-- The price of being wrong: the hourly imbalance settlement. A producer
-- that delivered less than it sold pays the imbalance price for the gap;
-- one that delivered more receives it. This dataset ends in March 2025
-- when settlement moved to quarter hours; the successor is joined later.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/RegulatingBalancePowerdata/*.parquet', union_by_name = true)
)
select
    PriceArea                                  as area,
    cast(HourUTC as timestamp)                 as hour_utc,
    cast(ImbalanceMWh as double)               as imbalance_mwh,
    cast(ImbalancePriceEUR as double)          as imbalance_price_eur,
    cast(BalancingPowerPriceUpEUR as double)   as balancing_up_eur,
    cast(BalancingPowerPriceDownEUR as double) as balancing_down_eur,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, HourUTC order by _fetched_at desc) = 1
