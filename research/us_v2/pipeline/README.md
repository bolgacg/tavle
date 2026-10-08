# New York (NYISO) data layer, version 2

This is the data layer for `research/us_v2/OBJECTIVES.md`: the same tables, `published_at` rules and timing test as v1 (`research/us/pipeline/README.md`), extended to delivery years 2010 to 2023. It also adds the GEFS reforecast weather table, border features and v2 fees.

**Holdout rule.** v2 tables stop at 2023-12-31. Rows with a delivery or target time on or after 2024-01-01 are dropped as they are read, and nothing is computed on them. Load-forecast vintages issued 27 to 31 Dec 2023 lose their 2024 targets (4,320 rows dropped). The feature builders go through v1's `lock.py`, so any later date raises.

**v1 is untouched.** No v1 file was edited. The v2 folder holds two shims, `common.py` and `fees.py`. Each one executes the v1 module into its own namespace and then overrides the year range, the output folder and the fee years. With this folder first on `sys.path`, every v1 module that runs `from common import ...` gets the v2 values (`build_tables`, `timing`, `panel`, `deep_day`). `build_tables_v2.py` asserts that this worked.

## Where things live

| What | Where |
|---|---|
| Canonical code | laptop `~/projects/tavle/research/us_v2/pipeline/` |
| Run copy | gene `~/nyiso-us/pipeline_v2/` (`rsync -a pipeline/ gene:nyiso-us/pipeline_v2/`) |
| Raw data | gene `~/nyiso-us/raw/` (NYISO zips 2010 to 2026) and `raw/gefs_reforecast/YYYY/YYYYMMDD.json` |
| Tables | gene `~/nyiso-us/parquet_v2/` |
| Feature matrices | gene `~/nyiso-us/parquet_v2/features/` |
| Checks and timing-test counts | gene `~/nyiso-us/results/v2/` |
| Flags | `results/v2_price_data.done`, `results/v2_weather_data.done`, `results/v2_data.done` (written when both exist) |
| Venvs | `.venv` (v1, used for everything) and `.venv_grib` (python + eccodes 2.49, used only by the reforecast fetch) |

```
cd ~/nyiso-us && setsid nohup bash pipeline_v2/run_price_stage.sh   > logs/v2_price_stage.log 2>&1 < /dev/null &
cd ~/nyiso-us && setsid nohup bash pipeline_v2/run_weather_stage.sh > logs/v2_weather_stage.log 2>&1 < /dev/null &
```

## Files

| File | Purpose |
|---|---|
| `common.py` | v1 `common` with FIRST_DAY 2010-01-01, LAST_DAY 2023-12-31, PARQUET `parquet_v2`, RESULTS `results/v2`, and the reforecast publication lag |
| `fees.py` | v1 `model/fees.py` with 2010 to 2019 added and sources cited (see Fees) |
| `fetch_outsched.sh` | outSched monthly zips 2010 to 2019 (all 120 months exist) |
| `fetch_reforecast.py` | GEFS v12 reforecast, 2 m temperature at the 12 points (see Weather) |
| `check_raw.py` | headers, names, PTIDs, DST hour counts and write times for 2010 to 2023, written to `results/v2/raw_checks.json` |
| `build_tables_v2.py` | v1 `build_tables` over 2010 to 2023, plus `weather_gfs` (filtered copy of v1) and `weather_reforecast` |
| `test_timing_v2.py` | the v1 timing tests over 2010 to 2023, plus the weather and reforecast checks |
| `features_v2.py` | hourly panel and day matrix, built once |
| `run_price_stage.sh`, `run_weather_stage.sh` | the two stages and their flags |

## Tables (`parquet_v2/`)

These have the same names and columns as v1: `prices_zone.parquet`, `prices_gen/YYYYMM.parquet` (168 months), `load_forecast.parquet`, `outages.parquet` and `weather_gfs.parquet`.

**Border proxies (idea V5).** `H Q`, `NPX`, `O H` and `PJM` are rows in `prices_gen/` with `point_type = 'external'`, as in v1. They have the same columns as the zones: DA and RT LBMP, loss, sign-corrected congestion, energy, and `da_published_at` / `rt_published_at`. Their PTIDs never appear in the generator files.

**`weather_gfs`** is v1's table cut at target < 2024-01-01, so it starts on 25 March 2021. It gains three columns:

* `issue_time`: the latest possible run time, target minus lead plus 3 h;
* `lead_hours`;
* `source`.

**`weather_reforecast`** has one row per point, 00 UTC run and lead. Columns: `point, zone, primary, lat_req, lon_req, lat_grid, lon_grid, init_utc, lead_hours, target_hour, temperature_2m_c, issue_time, published_at, member, source`.

## What changed in the raw files since 2010 (`results/v2/raw_checks.json`)

* **Zone names and PTIDs.** The zone file has the same 15 names (11 zones and 4 border proxies) and the same PTIDs in every year from 2010 to 2023.
* **Generator points.** There are 481 in 2010 and 736 in 2023. Five PTIDs changed name: letter case in four of them, plus `O.H._GEN_BRUCE` renamed to `OH_GEN_PROXY`. The tables key on PTID, so a renamed point stays one series. A point appears only in the hours it was published.
* **Real-time file format.** Files before mid-2016 have a 7th column, `Scarcity In Effect (Y/N)`. v1's reader passes exactly six names, which would shift a 7-column file. The v2 reader reads columns 0 to 5. Day-ahead headers changed only in quoting.
* **DST.** Every spring-forward day has 23 rows per name and every fall-back day 25, in all series and years. Fall-back 01:00 appears twice, EDT first, as in v1, and `localize_wall` handles it unchanged.
* **File write times.** These are NYISO wall clock times from the zip entries, given in hours from 00:00 of the file's date.
  * DAM files are written about 09:50 on D-1 in 2010 and 2011, and 09:35 from 2014. A few were rewritten days later: the latest is a 2012 zone file 755 h after its date and a 2015 gen file 348 h after. The DA rule (later of 11:00 on D-1 and the write time) handles these.
  * isolf is written at 07:05 on D-1. In 2010 the write time varies more, from 00:00 on D-1 to 08:29 on D+0, and a 2015 file was rewritten 63 days later. When file D is written after 05:00 on D, the bid uses vintage D-1 (497 of 500 test days used file D).
  * RT zone files in 2010 have many late rewrites (the 99th percentile is 1,115 h). These rows are `rt_revised`, and they count only from the rewrite, as in v1.
* **Completeness.** No daily file is missing in any series from 2010 to 2023, outSched included.

## published_at rules

They are the same as v1 for prices, outages and `weather_gfs`.

**`load_forecast`: published_at = later(file_written_at, 06:00 on issue_date - 1).** NYISO writes file D at about 07:05 on D-1, so the bound changes nothing for normal files. Three 2010 files (issue dates 2010-04-27, 05-24 and 05-29, 4,356 rows) carry write times between 00:00 and 01:02 on D-1. At face value, the file named D+1 would be usable at 05:00 on D, which the v1 contract forbids. The bound keeps that rule without editing the recorded write time. `test_no_vintage_named_after_bid_day_is_public_at_05` checks all 5,113 files, and the reused v1 rule test checks this v2 rule for the table. The fix was applied at 11:21 on 7 Oct, and the 2010 feature parts were rebuilt.

**`weather_reforecast`: published_at = init + 8 h.** This is 03:00 EST or 04:00 EDT on D, before the 05:00 decision. Operational GEFS 00 UTC output through day 3 is on NOMADS about 4.5 to 5.5 h after initialisation, so the 8 h bound is conservative.

## Weather: archived forecasts only (Bo's rule)

No observation, reanalysis or analysis-stitched product is used.

* Every weather row carries `source`, `issue_time` and `lead_hours`.
* `build_tables_v2.check_weather` fails the build if `published_at` is not issue + delay, if the lead is not positive, or if the run is not before the valid time.
* `test_weather_rows_are_archived_forecasts` and `test_weather_used_for_a_bid_is_issued_before_05` repeat the check.
* `test_wxr_block_uses_only_public_runs` checks that every reforecast value in the matrices was published by 05:00 on its bid day.

**GEFS v12 reforecast** (AWS `noaa-gefs-retrospective`, `GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/Days:1-10/tmp_2m_*.grib2`):

* **What is fetched.** Only the control member c00, only the 00 UTC run, and only leads 27 to 54 h (3-hourly, 10 GRIB messages). Each message is fetched by HTTP range from the `.idx` file, about 7.5 MB a day out of a 60 MB file.
* **Why so little.** gene is on a phone tether at about 1 MB/s. All five members at all leads would be about 1 TB; this cut is about 27 GB, or about 4 hours.
* **Coverage.** Leads 27 to 54 h bracket every local hour of D+1 for the run of D, on 23, 24 and 25-hour days (tested).
* **Point values.** Each point takes the nearest 0.25-degree cell. The GRIB bytes are discarded after decoding.
* **Hourly values.** The feature builder interpolates the 3-hourly values linearly in time, within the one run of D only. If that run is missing, the hours are NaN rather than taken from an older run.

**Caveat.** The reforecast reruns the 2020 GEFS v12 model from reanalysis initial states. Every value is a forecast issued before its valid time and uses no later observation. Its skill is still that of the 2020 model, not of the forecasts actually available in 2010 to 2019, so the weather ideas in those years are optimistic about forecast quality.

**Gap.** No archived forecast exists for 1 January 2020 to 24 March 2021: the reforecast ends on 2019-12-31 and the Open-Meteo GFS archive starts on 25 March 2021. Weather columns are NaN there.

## Fees (`fees.py`, every source in its docstring)

The RS1 virtual rate in USD per cleared MWh. Virtual trades paid no Schedule 1 charge before 1 Jan 2010.

| Year | Rate | Source |
|---|---|---|
| 2010 | 0.065 | BAWG deck, Jan 2010 |
| 2011 | 0.0871 | Not found. Uses the nearest year; 2010 and 2012 are equally near, so the higher (2012) is taken, which is conservative |
| 2012 | 0.0871 | OATT 6.1.2.4.1 |
| 2013 | 0.0805 | BPWG budget-versus-actual deck |
| 2014 | 0.0976 | BPWG budget-versus-actual deck |
| 2015 | 0.1046 | BPWG budget-versus-actual deck |
| 2016 | 0.0850 | Posting |
| 2017 | 0.0649 | BPWG budget-versus-actual deck |
| 2018 | 0.0636 | BPWG budget-versus-actual deck |
| 2019 | 0.0795 | BPWG budget-versus-actual deck |

The 2013 to 2015 values come from slides with a mislabelled caption. They were matched to the virtual row and are not yet confirmed against the yearly postings.

The FERC per-MWh charge and the supply uplift bound for 2010 to 2019 were not found. Both use the nearest year, 2020: 0.010 and 0.007.

## The timing test over the new years

`test_timing_v2.py` reuses every v1 test. It uses 500 bid days, seed 20261006, drawn from 2010 to 2023, with every year sampled.

* **Price stage:** 35 passed, 0 failures, after the load_forecast fix. Every source had future rows removed by the filter. The counts are in `results/v2/timing_test.json`.
* **Rule check:** the `published_at` rule check is clean on every table and on all 168 generator months.
* **Weather stage:** the reforecast and weather tests run in the second stage.

## Timings on gene (7 Oct 2026)

**Downloads.**

* NYISO 2010 to 2019 (600 zips, 1.8 GB in total in `raw/`): about 25 minutes.
* outSched 2010 to 2019 (120 zips): about 15 minutes.
* GEFS reforecast: about 15 days of runs a minute, so 3,652 days take about 4 hours over the tether.

**Price stage** (`logs/v2_price_stage.log`): 49 minutes in all.

| Step | Time |
|---|---|
| prices | 6 min |
| load, outages, weather_gfs | 30 s |
| timing test | 6 min |
| panel (14 years, 4 workers) | 17.5 min |
| border block | 2.6 min |
| day matrix (3 processes) | 16.5 min |
| assemble | 6 s |

**Weather stage.** Building the reforecast table, the wxr block and the assembly takes under 5 minutes once the download is done.

`init_utc`, `issue_time`, `target_hour` and `published_at` are stored as tz-aware America/New_York like every other table. The 00 UTC run of D therefore reads 19:00 or 20:00 on D-1.

## Feature matrices (`parquet_v2/features/`, built once)

**`panel_2010_2023.parquet`**: one row per bid day D, zone and delivery hour of D+1.

* v1 columns: `panel.KEY_COLS + LABEL_COLS + BASE_FEATURES`, from v1's `panel.build_panel` code.
* Border columns, for each border k in HQ, NPX, OH, PJM:
  * `bord__da_d0_h_<k>`, `bord__dacong_d0_h_<k>`: day-ahead price and congestion at the border for the same hour of D;
  * `bord__gap_dm1_h_<k>`: the RT minus DA gap at that hour on D-1;
  * `bord__gap_7d_h_<k>`: the mean gap at that hour over the last 7 days;
  * `bord__spread_da_d0_h_<k>`, `bord__spread_gap_dm1_h_<k>`: zone minus border.
* Weather columns (after the weather stage), at the zone's primary point: `wxr_temp_f`, `wxr_d1_max_f`, `wxr_d1_min_f`, `wxr_d1_mean_f`, `wxr_d1_hdh65`, `wxr_d1_cdh65`. The daily columns are NaN when fewer than 20 hours are present.

**`day_2010_2023.parquet`**: one row per bid day.

* v1 columns: the `deep_day.build_day_features` schema (`wx__`, `lf__`, `lfwow__`, `out__`, `px__`, `rtnow__`, `cal__`, `y__` labels with the v2 fees), from v1's code.
* Border columns: `bord__<k>__dJ__<stat>` for J = 0 to 6, with the same 8 statistics as `px__`.
* Weather columns: `wxr__<point>__hHH`, for 12 points and 24 hours, in degrees C.
* `wx__` (GFS) exists only from 25 March 2021 and `wxr__` only to 2019.

Every input is filtered on its own `published_at` by 05:00 on D. The v1 blocks use `timing.features_available_at` and `assert_no_lookahead`. The border and reforecast blocks apply the same filter and assert it. The pieces are in `features/parts/`, and `features_v2.py assemble` rejoins them without recomputing. Both matrices are written to a `.tmp` file and then moved into place with `os.replace`, so a reader never sees a half-written file.
