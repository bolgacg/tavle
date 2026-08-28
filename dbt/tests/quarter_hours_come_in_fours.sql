-- After the seam every hour should carry four quarter-hour prices. Three
-- or five means a DST transition leaked through or a fetch was partial.
select area, hour_utc, n_intervals
from {{ ref('power_hourly') }}
where source = 'DayAheadPrices' and n_intervals <> 4
