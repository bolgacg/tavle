"""Per-hour table for the decision machine (research/decision.py builds the machine on it).

One row per zone and hour from December 2019 to 4 March 2025 (the end of Energinet's hourly
imbalance series). Every feature is what a desk had at the intraday gate, one hour before
delivery: settled balancing data up to hour t-2, the day-ahead price for t, and the wind
forecasts for t issued at least five hours before t. Nothing from hour t-1 or t is used as an
input; hour t's gap is the outcome only.
"""
import pathlib
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
RAW = ROOT / "data" / "raw" / "RegulatingBalancePowerdata"
OUT = ROOT / "research" / "results" / "decision_hours.parquet"


def build():
    con = duckdb.connect(str(DB), read_only=True)
    act = con.execute(f"""
        with src as (select * from read_parquet('{RAW}/*.parquet', union_by_name = true))
        select PriceArea as area, cast(HourUTC as timestamp) as hour_utc,
               cast(mFRRUpActBal as double) as mfrr_up, cast(mFRRDownActBal as double) as mfrr_down,
               cast(ImbalanceMWh as double) as imbalance_mwh
        from src qualify row_number() over (partition by PriceArea, HourUTC order by _fetched_at desc) = 1
    """).df()
    base = con.execute("""
        select v.area, v.hour_utc, v.price_eur, v.imbalance_price_eur, v.imbalance_minus_spot_eur as gap,
               f.wind_fc_da_mwh, f.wind_fc_id_mwh, f.wind_fc_5h_mwh, f.wind_fc_1h_mwh, f.wind_actual_mwh,
               f.consumption_mwh, f.solar_fc_da_mwh
        from forecast_vs_imbalance v join forecast_hourly f using (area, hour_utc)
        where v.imbalance_minus_spot_eur is not null
        order by 1, 2
    """).df()
    con.close()
    df = base.merge(act, on=["area", "hour_utc"], how="left")

    rows = []
    for area, g in df.groupby("area"):
        # complete hourly grid so that lags are true hours, and missing settled hours are visible
        idx = pd.date_range(g.hour_utc.min(), g.hour_utc.max(), freq="h")
        g = g.set_index("hour_utc").reindex(idx)
        g.index.name = "hour_utc"
        g["area"] = area
        gap = g["gap"]
        sgn = np.sign(gap)
        # settled information at the gate: everything up to t-2
        g["gap2"] = gap.shift(2)
        g["sgn2"] = sgn.shift(2)
        g["settled2"] = g["gap2"].notna()
        # last nonzero sign among settled hours and its age in hours (t-2 has age 2)
        nz = sgn.where(sgn != 0)
        last_nz = nz.shift(2).ffill(limit=12)
        age = pd.Series(np.nan, index=g.index)
        pos = np.arange(len(g))
        last_pos = pd.Series(np.where(nz.shift(2).notna(), pos, np.nan), index=g.index).ffill(limit=12)
        age = pos - last_pos + 2
        g["dir"] = last_nz
        g["dir_age"] = age
        # run length of the same nonzero sign ending at t-2
        s2 = sgn.shift(2)
        run = np.zeros(len(g))
        r = 0
        prev = 0.0
        vals = s2.values
        for i, v in enumerate(vals):
            if np.isnan(v) or v == 0:
                r = 0
                prev = 0.0
            elif v == prev:
                r += 1
            else:
                r = 1
                prev = v
            run[i] = r
        g["run2"] = run
        g["abs_gap2"] = g["gap2"].abs()
        g["net_act2"] = (g["mfrr_up"].fillna(0) - g["mfrr_down"].fillna(0)).shift(2)
        g["imb2"] = g["imbalance_mwh"].shift(2)
        g["med_abs_gap6"] = gap.shift(2).abs().rolling(6, min_periods=3).median()
        # forecasts for hour t known at the gate: the day-ahead and the five-hour horizon
        g["rev5"] = g["wind_fc_5h_mwh"] - g["wind_fc_da_mwh"]
        g["fc_share"] = g["wind_fc_da_mwh"] / g["consumption_mwh"]
        hod = g.index.hour
        g["hod_sin"] = np.sin(2 * np.pi * hod / 24)
        g["hod_cos"] = np.cos(2 * np.pi * hod / 24)
        g["y"] = sgn  # outcome: sign of hour t's gap
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["single_pricing"] = out["hour_utc"] >= pd.Timestamp("2021-11-01")
    out["period"] = np.where(out["hour_utc"] < pd.Timestamp("2024-01-01"), "training", "separate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    o = build()
    print(f"wrote {OUT}: {len(o)} rows, {o.area.nunique()} zones, {o.hour_utc.min()} to {o.hour_utc.max()}")
    print("settled rows:", int(o.gap.notna().sum()), "missing settled hours:", int(o.gap.isna().sum()))
    print("mfrr coverage:", round(float(o.mfrr_up.notna().mean()), 3))
    print(o[["area", "hour_utc", "gap", "dir", "dir_age", "run2", "net_act2", "rev5", "y"]].dropna().head(3).to_string())
