# New York (NYISO) data layer

Data layer for the preregistered study in
`~/projects/career-sweep/applications/danske-commodities-us/OBJECTIVES-v2-APPROVED.md`:
can a model, using only what is public by 05:00 New York time on day D, trade virtual
positions for delivery day D+1 in the 11 NYISO load zones after costs. Build years are
2020 to 2023; held-out years are 2024 to September 2026.

**Holdout rule.** Rows, files and missing hours may be counted for any year. No price
statistic, spread, profit or distribution is computed for dates on or after 2024-01-01.
`inventory.py` counts only. Its one price check (the congestion sign) filters to 2020-2023
and asserts that filter. `test_timing.py` compares timestamps and row counts only.

## Where things live

| What | Where |
|---|---|
| Canonical code | laptop `~/projects/tavle/research/us/pipeline/` (this folder) |
| Run copy | gene `~/nyiso-us/pipeline/` (`rsync -a pipeline/ gene:nyiso-us/pipeline/`) |
| Raw downloads | gene `~/nyiso-us/raw/` (NYISO zips) and `~/nyiso-us/raw/gfs/` (JSON) |
| Parquet tables | gene `~/nyiso-us/parquet/` |
| Venv | gene `~/nyiso-us/.venv` |
| Logs | gene `~/nyiso-us/logs/` |
| Inventory, timing-test counts | gene `~/nyiso-us/results/`, copied to laptop `research/us/results/` |

Everything runs offline once the downloads are present. Run long jobs with nohup.

```
cd ~/nyiso-us && nohup bash pipeline/fetch_extra.sh > logs/fetch_extra.log 2>&1 &   # outSched + GFS (skip-if-present)
cd ~/nyiso-us && nohup bash pipeline/setup_venv.sh > logs/setup_venv.log 2>&1 &     # venv + SIGILL import tests
cd ~/nyiso-us/pipeline && ../.venv/bin/python build_tables.py all                  # about 4 minutes
cd ~/nyiso-us/pipeline && ../.venv/bin/python inventory.py
cd ~/nyiso-us/pipeline && ../.venv/bin/python -m pytest -q -p no:cacheprovider test_timing.py
```

## Files

| File | Purpose |
|---|---|
| `common.py` | paths, zones, publication constants, DST helpers |
| `points.py` | one weather point per load zone |
| `fetch_extra.sh`, `fetch_gfs.py` | outSched and GFS downloads, same skip-if-present pattern as `~/nyiso-us/fetch.sh` |
| `setup_venv.sh` | builds the venv and tests every import for SIGILL (exit 132) |
| `build_tables.py` | raw files to parquet |
| `timing.py` | `features_available_at`, `bid_inputs`, `assert_no_lookahead`, `rule_violations` |
| `test_timing.py` | objective 2 timing test (pytest) |
| `inventory.py` | counts per series, table, year and zone |

## Raw sources

| Series | URL pattern | Daily file D contains |
|---|---|---|
| damlbmp zone, gen | `mis.nyiso.com/public/csv/damlbmp/<yyyymm01>damlbmp_{zone,gen}_csv.zip` | day-ahead LBMP for D |
| rtlbmp zone, gen | `.../rtlbmp/<yyyymm01>rtlbmp_{zone,gen}_csv.zip` | hourly integrated real-time LBMP for D |
| isolf | `.../isolf/<yyyymm01>isolf_csv.zip` | NYISO load forecast for D to D+5, 11 zones + NYISO total |
| outSched | `.../outSched/<yyyymm01>outSched_csv.zip` | snapshot of scheduled transmission outages for D |
| GFS | `previous-runs-api.open-meteo.com/v1/forecast?...&hourly=temperature_2m_previous_day2&models=gfs_seamless` | one JSON per point per year, UTC hours |

Each monthly zip holds one CSV per day. The zip entry modification time is the time NYISO
wrote that daily file, in NYISO server wall-clock time (Eastern, DST-aware: DAM files show
09:32 to 09:35 in both January and July). The full list of write times is in
`results/file_write_times.csv`.

Observed write times (from the zip entries):

| Series | File named D is written | Consequence for a bid made at 05:00 on D |
|---|---|---|
| damlbmp | 09:32 to 09:35 on D-1 (after 11:00 for 1 zone file and 5 gen files in all years) | prices for D are known; prices for D+1 (the target) are not |
| isolf | 07:05 to about 08:00 on D-1; 1 to 4 files a year only after 05:00 on D | file D is the newest usable vintage; file D+1 (07:05 on D) is lookahead |
| outSched | 09:38 to 09:41 on D-1; 4 files in all years only after 05:00 on D | snapshot D is the newest usable one; snapshot D+1 (09:41 on D) is lookahead |
| rtlbmp | 23:57 on D; 18 to 52 zone files a year are written after midnight (some at 00:02 or 02:01, the rest days or weeks later as corrections) | see the rule below |

## Tables (gene `~/nyiso-us/parquet/`)

All times are timezone-aware `America/New_York`. `delivery_hour` and `target_hour` are hour
beginning.

**prices_zone.parquet**: one row per zone-hour for the 11 load zones.
`delivery_hour, zone, ptid, da_lbmp, da_loss, da_congestion_raw, da_congestion, da_energy,
rt_lbmp, rt_loss, rt_congestion_raw, rt_congestion, rt_energy, da_published_at,
rt_published_at, published_at, rt_revised, da_file_written_at, rt_file_written_at`.
The table is an outer join of DA and RT, so a row with a missing RT hour keeps its DA values
and has NaN for RT.

**prices_gen/YYYYMM.parquet**: the same columns plus `name` and `point_type` for every generator
PTID in the gen files (`point_type = gen`) and the four border proxies from the zone files
(`H Q, NPX, O H, PJM`, `point_type = external`; their PTIDs never appear in the gen files).
This is long format: one row per point-hour. A point appears only in the hours it was
published, so "used only while it existed" holds by construction. Generator counts per year
are in the inventory.

**load_forecast.parquet**: one row per vintage, target hour and zone.
`issue_date` (the date in the file name), `file_written_at`, `published_at`, `target_hour`,
`lead_days` (target local date minus issue_date, 0 to 5), `zone` (11 zones and `NYISO`),
`load_forecast_mw`. Every vintage is kept.

**outages.parquet**: one row per snapshot and equipment.
`snapshot_date`, `file_written_at`, `published_at`, `ptid`, `equipment`, `sched_out`,
`sched_in`, plus the raw strings. These are scheduled outages, not actual ones: some are
cancelled.

**weather_gfs.parquet**: one row per point and target hour.
`point, zone, primary, lat_req, lon_req, lat_grid, lon_grid, target_hour, temperature_2m_c,
run_lead_hours (48), published_at`.

## published_at rules

| Table | published_at | Basis |
|---|---|---|
| DA prices | later of 11:00 on D-1 and the file's write time | NYISO posts the DAM by 11:00; files are written about 09:35, so 11:00 is the conservative bound. A file rewritten later (a correction) counts from its rewrite. |
| RT prices | end of the hour + 15 minutes; if the daily file was rewritten after midnight ending its day, the later of that and the rewrite time | Hourly RT prices are posted as each hour ends. The archive holds the final file, so a corrected value was public only once the correction was written. `rt_revised` marks these rows. This is stricter than "end of hour + lag": on the days whose file was corrected later, yesterday's RT prices are not usable at 05:00. A write at 00:02 or 02:01 on D+1 changes nothing at the 05:00 deadline. |
| prices row | later of da_published_at and rt_published_at | when the whole row was known |
| load_forecast | zip entry write time of the file holding the row | |
| outages | zip entry write time of the file holding the row | |
| weather_gfs | target_hour minus 42 hours | see the GFS assumption below |

`features_available_at` uses the component times, not the row time: DA and RT prices are
returned as separate frames, each filtered on its own `published_at`.

**GFS assumption.** Open-Meteo documents `_previous_day2` as "the value that was predicted 48
hours before valid time" and does not say which 6-hourly run is chosen. We assume the run's
initialisation is at or before target minus 48 hours. GFS output is public about 3.5 to 5
hours after initialisation, so `published_at = target - 48 h + 6 h`. If Open-Meteo instead
picked the nearest run, the initialisation could be up to 3 hours later and this bound could
be up to about 2 hours early. Consequence: the last hour of a delivery day is exactly at the
05:00 deadline (23:00 on D+1 minus 42 h = 05:00 on D), and on the fall-back day one hour falls
after it and is excluded. Values exist only from 25 March 2021 00:00 UTC (earlier hours are null), so idea
B's build window starts on 25 March 2021. The archive also has null hours at all 12 points in
2023 (53) and 2024 (463); see the inventory.

## Time handling

* NYISO labels rows in local wall-clock time with no offset. On the fall-back day 01:00
  appears twice; the first occurrence by row order is EDT, the second EST
  (`common.localize_wall`). On the spring-forward day 02:00 does not appear, and the code raises
  if it ever does. Every year's DST days have 23 and 25 hours (tested).
* Zip entry times are naive Eastern; an ambiguous fall-back time is read as the later (EST)
  instant, which is conservative.
* The decision time is 05:00 on D in `America/New_York`, which exists on every day.

## Congestion sign

NYISO publishes `Marginal Cost Congestion` with the opposite sign to the usual convention:
LBMP = energy + losses - congestion_raw. `*_congestion_raw` keeps the published value,
`*_congestion = -raw` (positive means congestion raises the price), and `*_energy = LBMP -
losses - congestion`. The check: the energy component (the reference-bus price) must be equal in
all zones every hour. It is with the reversed sign and is not with the naive sign: within 0.05 (the components
are rounded to 0.01), 100% of 2020-2023 hours agree with the reversed sign against 5.1% with
the naive sign. The figures are in `inventory.json` under `congestion_sign_check`, computed on
2020 to 2023 only.

## The timing test (objective 2)

`test_timing.py`, 500 bid days drawn with seed 20261006 from 1 Jan 2020 to 29 Sep 2026:

* `bid_inputs(D)` gathers the rows a model may use for D+1: zone and generator price
  history, the newest public load-forecast vintage for every D+1 hour, the newest public outage
  snapshot, and the GFS values for D+1 hours. Every row must have `published_at <= 05:00` on D,
  no price row may be for D+1, and no load-forecast row may come from a file named after D.
* Candidate windows reach past the deadline on purpose, so the `published_at` filter does the
  excluding, and the test asserts it removed rows from every source.
* Injected lookahead must raise `LookaheadError`: the isolf file named D+1, the target day's
  own DA prices, the RT price for the hour starting 05:00 on D, the outSched snapshot for D+1,
  and GFS values published after the deadline. They are checked on 20 of the days.
* A row with a wrongly early `published_at` would slip through any filter, so
  `rule_violations` re-derives every table's `published_at` from its rule and must find zero
  violations, including in all 81 generator month files. A deliberately mislabelled row must
  be caught.

Coverage counts (how often file D, snapshot D and GFS for D+1 were usable) are written to
`results/timing_test.json`.

## Environment (gene: i5-3570K without AVX2, GTX 1060 6 GB)

`~/nyiso-us/.venv`, Python 3.12.3, torch 2.8.0+cu126 (from
`--index-url https://download.pytorch.org/whl/cu126`, served from the pip cache), numpy 2.5.3,
pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1, lightgbm 4.7.0, pytest 9.1.1. Every import
exits 0 (no SIGILL). Functional checks also pass: a GPU matmul and an LSTM forward and backward
pass on the GTX 1060, sklearn HistGradientBoosting, and a lightgbm fit. lightgbm works, so
HistGradientBoosting is not needed as a fallback. The full list is in `~/nyiso-us/logs/venv-freeze.txt`.

## Known gaps (details in inventory.json)

* rtlbmp_zone has no daily file for 1 to 18 July 2026 and 20 July 2026 (19 days; NYISO's
  monthly zip lacks them). The gen file has all of these days. The missing zone hours could be
  rebuilt from the 5-minute real-time files later; for now they are only counted.
* outSched has no snapshot for 25 September 2026. On the next day the model uses the
  snapshot for the 24th.
* GFS `previous_day2` is null before 25 March 2021, and for 53 hours in 2023 and 463 in 2024,
  identically at all points (a gap in the source archive).
* Weather points: GFS runs on a grid of about 25 km, so the N.Y.C., DUNWOD and MILLWD points
  can share neighbouring grid cells. NORTH has two points: Plattsburgh (primary) and Massena
  (secondary).
