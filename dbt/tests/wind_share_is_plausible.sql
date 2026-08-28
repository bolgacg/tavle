-- Wind can exceed local consumption in Denmark, which is the whole point
-- of the interconnectors, so the bound is generous rather than absent.
-- Anything above three times consumption is a unit or join mistake.
select area, hour_utc, wind_share
from {{ ref('power_context') }}
where wind_share < 0 or wind_share > 3
