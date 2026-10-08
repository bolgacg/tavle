"""DONE.md for the held-out run (orchestrate_heldout.sh): one table of every v1 and v2 row over 2024-01-01 to
2026-09-30 (net per year at 1 MW, total, Sharpe, PASS, candidate or comparison) and, for every row that clears the
bar, the too-good audit as the scorers computed it (best days, total without the best 5 days, placebos one day late
and shuffled or permuted, random days, too-good flags). Written 8 Oct 2026. Reads only the scorers' outputs.

    python heldout_summary.py <run dir>
"""
import json
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
REPO = Path(__file__).resolve().parents[3]
YEARS = ["2024", "2025", "2026"]


def cands(md: Path) -> set:
    t = md.read_text()
    part = t.split("## Candidates", 1)[1].split("## Comparisons", 1)[0]
    return {m.group(1).strip() for m in re.finditer(r"^\| ([^|]+?) \|", part, re.M)} - {"Strategy", "Row", "---"}


def f(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:,.0f}" if abs(x) >= 10 else f"{x:.2f}"
    return str(x)


def year_net(v):
    return v.get("net_usd") if isinstance(v, dict) else v


rows = []
v1f = RUN / "ho_v1" / "side" / "lab_results_heldout.json"
c1 = cands(REPO / "research/us/HELDOUT-LIST.md")
if v1f.exists():
    for n, r in json.loads(v1f.read_text())["strategies"].items():
        ys = r.get("years", {})
        tot = r.get("three_year", r.get("total", {}))
        rows.append(dict(v="v1", name=n, cand=n in c1 or n.split(" [")[0] in c1, y=[year_net(ys.get(y)) for y in YEARS],
                         total=tot.get("net_usd") if isinstance(tot, dict) else tot,
                         sharpe=tot.get("sharpe") if isinstance(tot, dict) else r.get("sharpe"), PASS=r.get("PASS"),
                         raw=r))
v2f = RUN / "ho_v2" / "results" / "v2" / "v2_results_heldout.json"
c2 = cands(REPO / "research/us_v2/HELDOUT-LIST.md")
if v2f.exists():
    for r in json.loads(v2f.read_text())["rows"]:
        ys = r.get("years", {})
        rows.append(dict(v="v2", name=r["name"], cand=r["name"] in c2, y=[year_net(ys.get(y)) for y in YEARS],
                         total=r.get("total"), sharpe=r.get("sharpe"), PASS=r.get("PASS"), raw=r))

L = ["# Held-out run, 2024-01-01 to 2026-09-30: DONE", "",
     f"Sources: {v1f if v1f.exists() else 'v1 results MISSING'}; {v2f if v2f.exists() else 'v2 results MISSING'}; "
     f"registered verdict run in {RUN / 'ho_reh'}. Net USD at 1 MW, full costs; 2026 is January to September.", "",
     "| version | row | candidate | 2024 | 2025 | 2026 to Sep | total | Sharpe | PASS |", "|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["v"], not r["cand"], -(r["total"] or -1e18))):
    L.append(f"| {r['v']} | {r['name']} | {'yes' if r['cand'] else 'no'} | " + " | ".join(f(x) for x in r["y"]) +
             f" | {f(r['total'])} | {f(r['sharpe'])} | {r['PASS']} |")
L += ["", "## Too-good audit of every row that clears the bar", ""]
passed = [r for r in rows if r["PASS"]]
if not passed:
    L.append("No row clears the bar on the held-out window.")
for r in passed:
    raw = r["raw"]
    keep = {k: raw.get(k) for k in ("best_5_days_three_year", "total_without_best_5_days", "placebo", "placebos",
                                    "too_good", "too_good_checks", "fragile_reasons", "FRAGILE", "sizing_view",
                                    "sizing_view_carried", "stress_0.50_total")
            if raw.get(k) is not None}
    if "fragility" in raw and isinstance(raw["fragility"], dict):
        keep["random_days"] = raw["fragility"].get("random_days", {}).get("share_random_beats_rule")
        keep["without_best_10_days"] = raw["fragility"].get("without_best_10_days")
    L += [f"### {r['v']} {r['name']}", "", "```", json.dumps(keep, indent=1, default=str)[:6000], "```", ""]
(RUN / "DONE.md").write_text("\n".join(L) + "\n")
print(f"DONE.md: {len(rows)} rows, {len(passed)} clear the bar")
