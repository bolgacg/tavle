-- The production and consumption settlement: what was actually generated
-- and consumed per area per hour. Thirty-odd columns of which the desk
-- cares about a handful, so the staging model does the arithmetic once,
-- names it, and lets the wide source stay wide.
with src as (
    select * from read_parquet('{{ var("raw_path") }}/ProductionConsumptionSettlement/*.parquet', union_by_name = true)
)
select
    PriceArea                                    as area,
    cast(HourUTC as timestamp)                   as hour_utc,
    coalesce(OffshoreWindLt100MW_MWh, 0) + coalesce(OffshoreWindGe100MW_MWh, 0)
                                                 as offshore_wind_mwh,
    coalesce(OnshoreWindLt50kW_MWh, 0) + coalesce(OnshoreWindGe50kW_MWh, 0)
                                                 as onshore_wind_mwh,
    coalesce(SolarPowerLt10kW_MWh, 0) + coalesce(SolarPowerGe10Lt40kW_MWh, 0)
      + coalesce(SolarPowerGe40kW_MWh, 0) + coalesce(SolarPowerSelfConMWh, 0)
                                                 as solar_mwh,
    coalesce(CentralPowerMWh, 0) + coalesce(LocalPowerMWh, 0) + coalesce(CommercialPowerMWh, 0)
                                                 as thermal_mwh,
    GrossConsumptionMWh                          as consumption_mwh,
    _fetched_at
from src
qualify row_number() over (partition by PriceArea, HourUTC order by _fetched_at desc) = 1
