"""One place for HTTP so every loader inherits the same manners: a real
User-Agent, a timeout, and a 429 handler that reads the server's own
retry-after instead of guessing."""
import re
import time
import urllib.request
import urllib.error

USER_AGENT = "tavle/0.1 (+https://github.com/bolgacg/tavle)"


class RateLimited(Exception):
    def __init__(self, wait_seconds, url):
        super().__init__(f"429 from {url}, retry in {wait_seconds}s")
        self.wait_seconds = wait_seconds


def get(url, timeout=120):
    """Return (status, body_text). Never raises on HTTP errors; the caller
    decides what a status means for its source."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def get_with_backoff(url, attempts=6, http=get, sleep=time.sleep, timeout=120):
    """GET that honours 429 retry-after and backs off on 5xx. Returns body
    text on 200, raises on anything else after the attempts run out."""
    delay = 5.0
    last = None
    for _ in range(attempts):
        status, body = http(url, timeout)
        if status == 200:
            return body
        if status == 429:
            m = re.search(r"(\d+)\s*seconds", body)
            wait = int(m.group(1)) if m else 60
            sleep(wait + 1)
            last = RateLimited(wait, url)
            continue
        if 500 <= status < 600:
            sleep(delay)
            delay = min(delay * 2, 120)
            last = RuntimeError(f"{status} from {url}")
            continue
        raise RuntimeError(f"{status} from {url}: {body[:200]}")
    raise last or RuntimeError(f"gave up on {url}")
