.PHONY: test extract build page all
PY := python3

test:
	$(PY) -m pytest -q tests

extract:
	$(PY) -m tavle.extract eds Elspotprices
	$(PY) -m tavle.extract eds DayAheadPrices
	$(PY) -m tavle.extract eds ProductionConsumptionSettlement
	$(PY) -m tavle.extract ecb

build:
	cd dbt && $(PY) -m dbt.cli.main build --profiles-dir .

page:
	$(PY) -m tavle.page

all:
	$(PY) -m tavle.dag
