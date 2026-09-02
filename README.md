# tavle

A small data platform for a trading desk, built the way the desk would want it:
every number traceable to the request that produced it, every transformation
tested, every run in a ledger you can query.

Live desk page: https://bolgacg.github.io/tavle/

*tavle* is Danish for the board a desk works from.

## What it does

Public sources with deliberately different shapes land as raw Parquet,
get modelled in dbt on DuckDB into one price schema, and come out as a desk
page and a set of tests that fail loudly when the world changes.

| Source | What | Shape | The awkward part |
|---|---|---|---|
| Energinet, Elspotprices | DK1/DK2 day-ahead prices | hourly, 2013 to 30 Sep 2025 | retired dataset |
| Energinet, DayAheadPrices | DK1/DK2 day-ahead prices | quarter-hourly, from 1 Oct 2025 | its successor, new schema, new resolution |
| Energinet, ProductionConsumptionSettlement | wind, solar, consumption | hourly | wide, many columns, settlement revisions |
| ECB reference rates | EUR/DKK, EUR/USD | daily, business days only | different calendar entirely |
| Energinet, Forecasts_Hour | wind and solar forecasts, four horizons | hourly, from Nov 2019 | one row holds four numbers for the same hour |
| Energinet, RegulatingBalancePowerdata | imbalance prices | hourly, to 4 Mar 2025 | retired when settlement went to quarter hours |
| Energinet, ElectricityProdex5MinRealtime | real-time wind and solar | five-minute, from 2015 (landed from 2020) | "errors will occur, and will generally not be corrected" |

The seam between the two price datasets is the point of the exercise. A
real desk lives with exactly this kind of break, and the platform has to
carry both sides on one grid without anyone noticing, while a test proves
the join is continuous.

Two chapters sit on top of the platform, each a page rebuilt from the marts:

- **Who pays when the wind forecast is wrong?** (`docs/wind/`): a pre-registered
  study of Energinet's wind forecasts against imbalance prices.
- **One hour of wind has six numbers** (`docs/versions/`): the four forecasts,
  the real-time feed and the settlement for the same hour on one grid; a
  tolerance fitted on 2025 and checked on 2026 that the pipeline runs as a dbt
  test; a bias watch that warns rather than fails; and a nightly self-diff
  ("rebuild yesterday") that logs which versions appeared, vanished or moved
  since the previous night, committed with the page.

## Layout

```
tavle/extract/   loaders: few large requests, 429 honoured, raw landed with provenance
tavle/dag.py     a DAG runner in one screen: topological order, retries, run ledger in DuckDB
tavle/page.py    builds docs/index.html from the marts; freshness and test panels come from the run
tavle/windpage.py, tavle/versionspage.py   the two chapters, built from research/results/*.json plus the marts
tavle/revisions.py  rebuild yesterday: diffs the recent window of the versions mart against last night's, appends docs/versions/revisions.json
tavle/sample.py  45-day raw slice committed so CI builds and tests without network
dbt/             staging (types, UTC, dedupe by latest fetch), marts (prices, power_hourly, desk_daily, forecast_hourly, wind_versions, wind_versions_long)
dbt/tests/       no gaps on the hourly grid, seam continuity, quarter-hours in fours, FX on business days,
                 the fitted tolerance (error), feed completeness (error), settlement lag (warn), bias band (warn)
research/        the studies: pre-registration, study scripts, results JSON the pages are built from
tests/           unit tests for the loaders, the runner and the self-diff, no network
```

## Running it

```
pip install -r requirements.txt
make test        # unit tests
make extract     # incremental pulls from the watermark (first run: full history, be patient)
make build       # dbt build: models plus every test
make page        # docs/index.html
make all         # the DAG: extract, build, page, with the ledger
```

Then `duckdb data/tavle.duckdb` and ask it things:

```sql
select * from ops.runs order by started desc limit 10;
select area, day_dk, avg_eur, negative_hours from desk_daily order by day_dk desc limit 14;
```

## Design choices, and what they cost

- **Few, large requests.** Energi Data Service rate-limits per dataset and answers
  429 with its own retry-after. The loader asks for a year at a time with `limit=0`
  and waits exactly as long as it is told. The cost is memory per request
  (a year of quarter-hours for two areas is about 70k rows, fine).
- **Latest fetch wins.** Re-fetched windows overlap by design, so staging dedupes
  on the natural key and keeps the newest `_fetched_at`. Settlement data gets
  revised; this is how revisions flow through without special cases.
- **The seam is stitched in the mart, not in the loader.** Raw stays faithful to
  each source. `prices` unions the two at 30 Sep 2025 22:00 UTC and `power_hourly`
  averages quarter-hours to the hour, so everything downstream sees one grid.
- **Tests that say why.** A missing hour, a duplicate on the DST fallback night,
  three quarter-hours instead of four, an FX rate on a Sunday: each has a named
  test with the reason in its header.
- **Airflow-shaped, not Airflow.** `dag.py` is a dict of tasks with dependencies,
  retries and a ledger. It would translate to an Airflow DAG file in an afternoon;
  the point here is that the platform is inspectable without a scheduler UI.
- **A static page.** The desk page carries its own data and is rebuilt after
  every run. No server, no login, and a red panel means the pipeline said so.
- **Severities are a judgment, and they are written down.** A broken feed fails
  the build; a late settlement or a drifting calibration warns, because a late
  source must not take the morning board down. Each test's header says which and why.
- **The rule the page quotes is the rule the pipeline runs.** The tolerance is
  fitted by `research/versions_study.py`, which writes the three numbers into
  `dbt/dbt_project.yml`; the dbt test and the page read the same lines.
- **Rebuild yesterday.** Latest fetch wins is the right rule for revisions, and it
  is also how a number changes under the desk without anyone noticing. So the
  nightly diffs the recent window against the previous night before overwriting
  it, and the log is committed next to the page it feeds.

## Honest limits

- Two areas, two currencies, six datasets. A real platform has fifty sources.
- No intraday, no bids, no volumes: the day-ahead auction result is what
  Energinet publishes openly.
- The run ledger lives in the same DuckDB file as the marts. Convenient here;
  in production it belongs in the scheduler's own store.
