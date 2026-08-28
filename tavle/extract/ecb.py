"""ECB daily reference rates, the second source, chosen because it has a
different shape from power prices in every way that matters downstream:
daily not hourly, business days only, one number per currency pair."""
import csv
import datetime as dt
import io
import time

from . import http as _http
from .raw import land, watermark

URL = ("https://data-api.ecb.europa.eu/service/data/EXR/D.DKK+USD.EUR.SP00.A"
       "?format=csvdata&startPeriod={start}")
FIRST = "2013-01-01"


def parse(csv_text):
    rows = []
    for r in csv.DictReader(io.StringIO(csv_text)):
        if not r.get("OBS_VALUE"):
            continue
        rows.append({
            "date": r["TIME_PERIOD"],
            "currency": r["CURRENCY"],
            "denominator": r["CURRENCY_DENOM"],
            "rate": float(r["OBS_VALUE"]),
            "status": r.get("OBS_STATUS", ""),
        })
    return rows


def extract(start=None, http=None, sleep=time.sleep, overlap_days=5):
    if start is None:
        wm = watermark("ecb_fx", "date")
        start = (wm - dt.timedelta(days=overlap_days)).date().isoformat() if wm else FIRST
    url = URL.format(start=start)
    body = _http.get_with_backoff(url, http=http or _http.get, sleep=sleep)
    rows = parse(body)
    path = land(rows, "ecb_fx", url)
    return [(start, None, len(rows), path)]
