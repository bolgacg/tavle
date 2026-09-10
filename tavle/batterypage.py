"""Build docs/battery/index.html: what does a one-megawatt battery earn in Danish balancing?

Every number on the page comes from research/results/activation.json (research/activation.py)
or is recomputed in the browser from the per-hour prices in it, by the same rules."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "research" / "results" / "activation.json"
OUT = ROOT / "docs" / "battery" / "index.html"
TEMPLATE = pathlib.Path(__file__).with_name("battery_template.html")


def build():
    data = json.loads(RES.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    return OUT, data


if __name__ == "__main__":
    p, d = build()
    print(f"wrote {p}: {len(d['zones']['DK1']['spot'])} hours per zone, size {p.stat().st_size // 1024} KB")
