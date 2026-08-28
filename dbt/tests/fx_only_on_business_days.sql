-- The ECB does not publish on weekends; a weekend rate is a parsing bug.
select rate_date, currency
from {{ ref('stg_fx') }}
where dayofweek(rate_date) in (0, 6)
