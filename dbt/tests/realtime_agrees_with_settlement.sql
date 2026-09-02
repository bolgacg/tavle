-- The rule act three of the versions chapter fits: the real-time hour and
-- the settled hour may differ by at most `versions_tolerance_pct` of the
-- settled value or `versions_tolerance_mwh`, whichever is larger. The
-- tolerance was fitted on 2025 and checked on 2026 by
-- research/versions_study.py; the numbers live in dbt_project.yml so the
-- pipeline and the page cannot drift apart. The test fails when more than
-- `versions_max_breach_pct` percent of the hours in the settlement's last
-- sixty days breach the tolerance. Hours with fewer than twelve readings
-- are excluded: an incomplete hour is a different fault with its own test.
with span as (
    select max(hour_utc) as hi from {{ ref('wind_versions') }} where v_settled is not null
),
recent as (
    select area,
        abs(v_realtime - v_settled) > greatest({{ var('versions_tolerance_mwh') }},
                                               {{ var('versions_tolerance_pct') }} / 100.0 * v_settled) as breach
    from {{ ref('wind_versions') }}, span
    where v_settled is not null and v_realtime is not null and realtime_readings = 12
      and hour_utc >= span.hi - interval 60 day
),
rate as (
    select area, count(*) filter (where breach) as breaches, count(*) as hours from recent group by 1
)
select area, breaches, hours, round(100.0 * breaches / hours, 2) as breach_pct
from rate
where hours > 0 and 100.0 * breaches / hours > {{ var('versions_max_breach_pct') }}
