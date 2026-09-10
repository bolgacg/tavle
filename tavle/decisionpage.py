"""Build docs/decision/index.html: can one trading decision run itself?

Every number on the page comes from research/results/decision.json (written by
research/decision.py) or is recomputed in the browser from the per-hour arrays in it,
by the same rules. Nothing is typed in."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "research" / "results" / "decision.json"
OUT = ROOT / "docs" / "decision" / "index.html"
TEMPLATE = pathlib.Path(__file__).with_name("decision_template.html")


def build():
    data = json.loads(RES.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    return OUT, data


if __name__ == "__main__":
    p, d = build()
    z = d["zones"]["DK1"]
    print(f"wrote {p}: {len(z['gap'])} hours per zone, separate from index {z['sep_start_index']}, size {p.stat().st_size // 1024} KB")
