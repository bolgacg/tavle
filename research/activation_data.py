"""Per-hour table for the battery study: what Energinet paid to an activated megawatt.

One row per zone and hour, December 2019 to 4 March 2025 (Energinet's hourly regulating power
series). Columns a prequalified one-megawatt asset would face: the day-ahead price known the day
before, the up-regulation price (paid to an activated megawatt of up-regulation), the
down-regulation price (what an activated megawatt of down-regulation pays for the energy it
consumes; below the day-ahead price, sometimes negative), the hour's dominating direction, and the
activated mFRR volumes. Energinet's convention: when a direction was not activated its price equals
the day-ahead price.
"""
import pathlib
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
RAW = ROOT / "data" / "raw" / "RegulatingBalancePowerdata"
OUT = ROOT / "research" / "results" / "activation_hours.parquet"


def build():
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(f"""
        with src as (select * from read_parquet('{RAW}/*.parquet', union_by_name = true)),
        raw as (
            select PriceArea as area, cast(HourUTC as timestamp) as hour_utc,
                   cast(BalancingPowerPriceUpEUR as double) as up_eur, cast(BalancingPowerPriceDownEUR as double) as down_eur,
                   cast(ImbalancePriceEUR as double) as imbalance_eur, cast(ImbalanceMWh as double) as imbalance_mwh,
                   cast(mFRRUpActBal as double) as mfrr_up_mwh, cast(mFRRDownActBal as double) as mfrr_down_mwh
            from src qualify row_number() over (partition by PriceArea, HourUTC order by _fetched_at desc) = 1)
        select r.*, p.price_eur as spot_eur
        from raw r join power_hourly p using (area, hour_utc)
        where r.up_eur is not null and r.down_eur is not null and p.price_eur is not null
        order by 1, 2
    """).df()
    con.close()
    rows = []
    for area, g in df.groupby("area"):
        idx = pd.date_range(g.hour_utc.min(), g.hour_utc.max(), freq="h")
        g = g.set_index("hour_utc").reindex(idx); g.index.name = "hour_utc"; g["area"] = area
        # tightened 11 Sep after the audit: the price signal alone is not enough, mFRR energy must
        # actually have run in the hour, else a 1 MW mFRR bid had nothing to be activated into
        # (the price can move on the automatic reserve alone, common in DK2)
        up_act = (g["up_eur"] > g["spot_eur"] + 1e-9) & (g["mfrr_up_mwh"].fillna(0) > 0)
        down_act = (g["down_eur"] < g["spot_eur"] - 1e-9) & (g["mfrr_down_mwh"].fillna(0).abs() > 0)
        # crude partial-hour proxy: a system activation under 60 MWh in the hour cannot have run at
        # 60 MW for the whole hour; delivery share = min(1, volume/60), used only by the haircut fault
        g["up_frac"] = np.minimum(1.0, g["mfrr_up_mwh"].fillna(0) / 60.0)
        g["down_frac"] = np.minimum(1.0, g["mfrr_down_mwh"].fillna(0).abs() / 60.0)
        # the dominating direction: the side the imbalance price settled on
        dom = np.where(g["imbalance_eur"].isna(), np.nan,
              np.where((g["imbalance_eur"] > g["spot_eur"] + 1e-9), 1.0, np.where(g["imbalance_eur"] < g["spot_eur"] - 1e-9, -1.0, 0.0)))
        g["dir"] = dom
        g["up_activated"] = up_act & g["up_eur"].notna()
        g["down_activated"] = down_act & g["down_eur"].notna()
        g["up_premium"] = np.where(g["up_activated"], g["up_eur"] - g["spot_eur"], 0.0)
        g["down_discount"] = np.where(g["down_activated"], g["spot_eur"] - g["down_eur"], 0.0)
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["single_pricing"] = out["hour_utc"] >= pd.Timestamp("2021-11-01")
    out["period"] = np.where(out["hour_utc"] < pd.Timestamp("2024-01-01"), "training", "separate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    o = build()
    print(f"wrote {OUT}: {len(o)} rows, {o.hour_utc.min()} to {o.hour_utc.max()}, missing {int(o.spot_eur.isna().sum())}")
    for a, g in o[o.single_pricing].groupby("area"):
        print(f"  {a}: up activated {g.up_activated.mean():.3f}, down {g.down_activated.mean():.3f}, mean up premium when activated {g.up_premium[g.up_activated].mean():.1f}, "
              f"mean down discount {g.down_discount[g.down_activated].mean():.1f}, both in one hour {((g.up_activated)&(g.down_activated)).mean():.3f}, dir==0 share {(g.dir==0).mean():.3f}")
