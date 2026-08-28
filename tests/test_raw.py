from tavle.extract import raw


def test_land_then_watermark_roundtrip(tmp_path):
    recs = [{"HourUTC": "2024-01-01T00:00:00", "PriceArea": "DK1", "SpotPriceEUR": 10.0},
            {"HourUTC": "2024-01-01T01:00:00", "PriceArea": "DK1", "SpotPriceEUR": 11.0}]
    path = raw.land(recs, "Elspotprices", "http://req", raw_dir=tmp_path)
    assert path.exists()
    wm = raw.watermark("Elspotprices", "HourUTC", raw_dir=tmp_path)
    assert str(wm).startswith("2024-01-01 01:00:00")


def test_empty_batch_lands_nothing(tmp_path):
    assert raw.land([], "Elspotprices", "http://req", raw_dir=tmp_path) is None
    assert raw.watermark("Elspotprices", "HourUTC", raw_dir=tmp_path) is None
