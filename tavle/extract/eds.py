"""Energi Data Service (api.energidataservice.dk).

Two facts shape this loader. The API is rate limited per dataset and
answers 429 with its own retry-after, so we make few, large requests
(limit=0 inside a date window) and wait exactly as long as it says. And
the day-ahead price history is split across two datasets: Elspotprices
(hourly, to 30 Sep 2025) and DayAheadPrices (15-minute, from 1 Oct 2025).
The seam is stitched downstream in dbt; here we only land both faithfully."""
import datetime as dt
import json
import time
import urllib.parse

from . import http as _http
from .raw import land, watermark

BASE = "https://api.energidataservice.dk/dataset/"

DATASETS = {
    # dataset: (timestamp column, default first date, resolution minutes)
    "Elspotprices": ("HourUTC", "2013-01-01", 60),
    "DayAheadPrices": ("TimeUTC", "2025-10-01", 15),
    "ProductionConsumptionSettlement": ("HourUTC", "2020-01-01", 60),
}
AREAS = ["DK1", "DK2"]


def build_url(dataset, start, end, areas=AREAS, columns=None):
    params = {
        "start": start,
        "end": end,
        "limit": 0,
        "filter": json.dumps({"PriceArea": list(areas)}),
    }
    if columns:
        params["columns"] = ",".join(columns)
    return BASE + dataset + "?" + urllib.parse.urlencode(params)


def windows(start, end, months=12):
    """Yield (start, end) ISO date pairs, `months` wide, covering [start, end)."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    while s < e:
        y, m = s.year, s.month + months
        while m > 12:
            y, m = y + 1, m - 12
        nxt = min(dt.date(y, m, 1), e)
        yield s.isoformat(), nxt.isoformat()
        s = nxt


def fetch(dataset, start, end, areas=AREAS, http=None, sleep=time.sleep):
    url = build_url(dataset, start, end, areas)
    body = _http.get_with_backoff(url, http=http or _http.get, sleep=sleep)
    payload = json.loads(body)
    return payload.get("records", []), url


def extract(dataset, start=None, end=None, months=12, pause=8.0, areas=AREAS,
            http=None, sleep=time.sleep, overlap_days=3):
    """Land everything for `dataset` from `start` (default: the watermark
    minus a small overlap, or the dataset's first date) to `end` (default:
    two days ahead, because day-ahead prices exist for tomorrow)."""
    ts_col, first, _ = DATASETS[dataset]
    if start is None:
        wm = watermark(dataset, ts_col)
        start = (wm - dt.timedelta(days=overlap_days)).date().isoformat() if wm else first
    if end is None:
        end = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    landed = []
    for i, (s, e) in enumerate(windows(start, end, months)):
        if i:
            sleep(pause)  # be a polite client; the limit is per dataset
        records, url = fetch(dataset, s, e, areas, http=http, sleep=sleep)
        path = land(records, dataset, url)
        landed.append((s, e, len(records), path))
    return landed
