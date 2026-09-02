-- Energinet: the settlement "is updated with a 9-15 day delay". If the
-- newest settled hour is more than 16 days behind the newest real-time
-- hour, the sixth version is late. Severity warn: the board keeps
-- publishing with five versions and the oversight panel says the sixth
-- has not arrived; a late source must not take the morning board down.
{{ config(severity = 'warn') }}
with x as (
    select area,
        max(hour_utc) filter (where v_settled is not null)  as settled_hi,
        max(hour_utc) filter (where v_realtime is not null) as realtime_hi
    from {{ ref('wind_versions') }} group by 1
)
select area, settled_hi, realtime_hi, date_diff('day', settled_hi, realtime_hi) as lag_days
from x
where date_diff('day', settled_hi, realtime_hi) > 16
