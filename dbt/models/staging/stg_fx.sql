-- ECB reference rates: one number per business day per pair, quoted as
-- units of currency per EUR.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/ecb_fx/*.parquet', union_by_name = true)
)
select
    cast("date" as date)   as rate_date,
    currency,
    denominator,
    cast(rate as double)   as rate,
    _fetched_at
from src
qualify row_number() over (partition by "date", currency order by _fetched_at desc) = 1
