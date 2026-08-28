import json

from tavle.extract import eds


def test_windows_cover_the_range_without_overlap():
    w = list(eds.windows("2013-01-01", "2015-07-01", months=12))
    assert w == [("2013-01-01", "2014-01-01"), ("2014-01-01", "2015-01-01"), ("2015-01-01", "2015-07-01")]


def test_url_requests_everything_in_the_window():
    url = eds.build_url("Elspotprices", "2024-01-01", "2024-02-01")
    assert "limit=0" in url and "start=2024-01-01" in url and "end=2024-02-01" in url
    assert "DK1" in url and "DK2" in url


def test_extract_lands_each_window_and_pauses_between(tmp_path, monkeypatch):
    monkeypatch.setattr(eds, "land", lambda recs, ds, url: tmp_path / "x.parquet")
    naps = []

    def fake(url, timeout):
        return 200, json.dumps({"records": [{"HourUTC": "2024-01-01T00:00:00", "PriceArea": "DK1"}]})

    landed = eds.extract("Elspotprices", "2024-01-01", "2024-03-01", months=1, pause=8.0,
                         http=fake, sleep=naps.append)
    assert [n for _, _, n, _ in landed] == [1, 1]
    assert naps == [8.0]
