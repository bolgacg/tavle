.PHONY: test extract build page all
PY := python3

test:
	$(PY) -m pytest -q tests

extract:
	$(PY) -m tavle.extract eds Elspotprices
	$(PY) -m tavle.extract eds DayAheadPrices
	$(PY) -m tavle.extract eds ProductionConsumptionSettlement
	$(PY) -m tavle.extract eds Forecasts_Hour
	$(PY) -m tavle.extract eds RegulatingBalancePowerdata
	$(PY) -m tavle.extract eds ElectricityProdex5MinRealtime
	$(PY) -m tavle.extract ecb

build:
	cd dbt && $(PY) -m dbt.cli.main build --profiles-dir .

page:
	$(PY) -m tavle.page
	$(PY) -m tavle.versionspage

all:
	$(PY) -m tavle.dag
