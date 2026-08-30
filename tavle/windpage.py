"""Build docs/wind/index.html: does the wind forecast already know tomorrow's price?

Every number on the page comes from research/results/*.json (the
pre-registered study, training and the once-read holdout) or is computed
here from the marts at build time. Nothing is typed in."""
import json
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
RES = ROOT / "research" / "results"
OUT = ROOT / "docs" / "wind" / "index.html"
TEMPLATE = pathlib.Path(__file__).with_name("wind_template.html")


def q(con, sql):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def collect():
    con = duckdb.connect(str(DB), read_only=True)
    d = {}
    d["training"] = json.loads((RES / "training.json").read_text())
    d["holdout"] = json.loads((RES / "holdout.json").read_text())
    mc = RES / "model_card.json"
    d["model_card"] = json.loads(mc.read_text()) if mc.exists() else None
    cap = RES / "capacity_by_zone.json"
    d["capacity"] = json.loads(cap.read_text()) if cap.exists() else None
    d["monthly_bias"] = q(con, """
        select area, strftime(hour_utc, '%Y-%m') as month, round(avg(wind_err_da_mwh)) as bias_mwh,
               round(avg(wind_fc_da_mwh)) as forecast_mwh, round(avg(wind_actual_mwh)) as actual_mwh
        from forecast_hourly where hour_utc >= timestamp '2020-01-01'
        group by 1, 2 order by 1, 2""")
    d["baselines"] = q(con, """
        with x as (
            select area, hour_utc, wind_actual_mwh,
                   lag(wind_actual_mwh, 24) over (partition by area order by hour_utc) as yesterday_same_hour,
                   avg(wind_actual_mwh) over (partition by area) as long_run_mean
            from forecast_hourly where wind_fc_da_mwh is not null and wind_actual_mwh is not null)
        select area, round(avg(abs(wind_actual_mwh - yesterday_same_hour))) as yesterday_same_hour,
               round(avg(abs(wind_actual_mwh - long_run_mean))) as long_run_mean
        from x where yesterday_same_hour is not null group by 1 order by 1""")
    d["horizon_mae"] = q(con, """
        select area,
               round(avg(abs(wind_actual_mwh - wind_fc_da_mwh)))  as day_ahead,
               round(avg(abs(wind_actual_mwh - wind_fc_id_mwh)))  as same_morning,
               round(avg(abs(wind_actual_mwh - wind_fc_5h_mwh)))  as five_hours,
               round(avg(abs(wind_actual_mwh - wind_fc_1h_mwh)))  as one_hour,
               round(avg(wind_actual_mwh)) as mean_actual
        from forecast_hourly
        where wind_fc_da_mwh is not null and wind_fc_id_mwh is not null and wind_fc_5h_mwh is not null and wind_fc_1h_mwh is not null
        group by 1 order by 1""")
    d["gap_deciles"] = q(con, """
        with x as (
            select area, wind_err_da_mwh as err, imbalance_minus_spot_eur as gap,
                   ntile(10) over (partition by area order by wind_err_da_mwh) as decile
            from forecast_vs_imbalance where wind_err_da_mwh is not null and imbalance_minus_spot_eur is not null)
        select area, decile, round(avg(gap), 2) as mean_gap_eur, round(median(gap), 2) as median_gap_eur,
               round(min(err)) as err_from, round(max(err)) as err_to, count(*) as hours
        from x group by 1, 2 order by 1, 2""")
    d["gap_zero_share"] = con.execute("select round(100.0*count(*) filter (where imbalance_minus_spot_eur = 0)/count(*),1) from forecast_vs_imbalance").fetchone()[0]
    d["span"] = {
        "forecast_first": str(con.execute("select min(hour_utc) from forecast_hourly").fetchone()[0])[:10],
        "forecast_last": str(con.execute("select max(hour_utc) from forecast_hourly").fetchone()[0])[:10],
        "imbalance_last": str(con.execute("select max(hour_utc) from forecast_vs_imbalance").fetchone()[0])[:10],
        "hours": con.execute("select count(*) from forecast_hourly").fetchone()[0],
    }
    con.close()
    return d


def build():
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data, default=str)))
    return OUT, data


if __name__ == "__main__":
    p, d = build()
    print(f"wrote {p}: {len(d['monthly_bias'])} monthly rows, holdout hours {d['holdout']['hours']}")
