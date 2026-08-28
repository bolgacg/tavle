from tavle.extract import http


def test_429_waits_exactly_what_the_server_says():
    calls, naps = [], []
    bodies = [(429, "Rate limit is exceeded. Try again in 37 seconds."), (200, "ok")]

    def fake(url, timeout):
        calls.append(url)
        return bodies.pop(0)

    out = http.get_with_backoff("http://x", http=fake, sleep=naps.append)
    assert out == "ok"
    assert naps == [38]
    assert len(calls) == 2


def test_5xx_backs_off_then_gives_up():
    naps = []

    def fake(url, timeout):
        return 503, "down"

    try:
        http.get_with_backoff("http://x", attempts=3, http=fake, sleep=naps.append)
    except RuntimeError as e:
        assert "503" in str(e)
    else:
        raise AssertionError("should have raised")
    assert naps == [5.0, 10.0, 20.0]


def test_4xx_other_than_429_fails_fast():
    def fake(url, timeout):
        return 404, "nope"

    try:
        http.get_with_backoff("http://x", http=fake, sleep=lambda s: None)
    except RuntimeError as e:
        assert "404" in str(e)
    else:
        raise AssertionError("should have raised")
