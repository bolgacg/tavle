"""Gate between the dry runs and the held-out run (orchestrate_heldout.sh). Written 8 Oct 2026.

    python heldout_gate.py v2 <dry-run dir>   every v2 row's 2023 positions and net equal the build (EXACT), the
                                              rows registered as not run excepted
    python heldout_gate.py v1 <dry-run dir>   every v1 row's 2023 daily net, positions, predictions and choices
                                              equal the build
Exit 0 only when everything is exact; otherwise prints each failure and exits 1. Reads only the dry-run reports.
"""
import json
import math
import sys
from pathlib import Path


def v2(run: Path) -> list[str]:
    f = run / "results" / "v2" / "dryrun_check.json"
    if not f.exists():
        return [f"{f} missing"]
    rec = json.loads(f.read_text())
    bad = []
    for r in rec["rows"]:
        s = r.get("status", "")
        if s == "EXACT" or s.startswith("not run"):
            continue
        bad.append(f"{r['row']}: {s} (not exact {r.get('not_exact')}, net diff {r.get('net_2023_diff_usd')})")
    v12 = rec.get("v12_before_window")
    if v12 and v12.get("exact") != v12.get("zone_hours"):
        bad.append(f"V12 before the window: {v12}")
    if not rec["rows"]:
        bad.append("no rows checked")
    return bad


def _share_ok(d, where, bad):
    for k, v in d.items():
        if isinstance(v, dict) and "exactly_equal_share" in v:
            md = v.get("max_abs_diff", 0.0)
            if v["exactly_equal_share"] != 1.0 or not (md == 0 or (isinstance(md, float) and math.isnan(md))):
                bad.append(f"{where} {k}: {v}")


def v1(run: Path) -> list[str]:
    f = run / "dryrun_equality.json"
    if not f.exists():
        return [f"{f} missing"]
    rep = json.loads(f.read_text())
    bad = []
    rows = rep.get("lab_daily_2023", {})
    if not rows:
        bad.append("no lab rows compared")
    for k, v in rows.items():
        if not isinstance(v, dict):
            continue
        if v.get("days") != v.get("days_exactly_equal") or v.get("max_abs_daily_diff_usd") != 0 or v.get("net_diff_usd") != 0:
            if k == "idea13_portfolio" and "days" not in v:
                continue
            bad.append(f"lab {k}: {v}")
    i13 = rep.get("idea13_2023", {})
    if i13.get("net_dryrun") != i13.get("net_build"):
        bad.append(f"idea 13: {i13}")
    for k, v in rep.get("lab_choices_2023", {}).items():
        if not v.get("equal"):
            bad.append(f"lab choice {k}: {v}")
    if not rep.get("evaluate_choices_2023", {}).get("equal"):
        bad.append("evaluate.py 2023 choices differ")
    for k, yrs in rep.get("lead_positions", {}).items():
        for y, v in (yrs or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("mw_exactly_equal_share") != 1.0 or v.get("mw_max_abs_diff") not in (0, 0.0):
                bad.append(f"lead positions {k} {y}: {v}")
    for name in ("deep_predictions_2023", "deep_day_2023", "policy_2023", "idea14_2023"):
        d = rep.get(name)
        if not d:
            bad.append(f"{name} missing")
            continue
        _share_ok(d, name, bad)
    sp = rep.get("deep_predictions_2023", {}).get("spike_p_star", {})
    for y, v in (sp or {}).items():
        if isinstance(v, dict) and v.get("build") != v.get("dryrun"):
            bad.append(f"deep spike p* {y}: {v}")
    for name in ("day_features_rebuilt", "panel_base_rebuilt"):
        if not rep.get(name, {}).get("all_equal"):
            bad.append(f"{name}: {rep.get(name)}")
    return bad


if __name__ == "__main__":
    which, run = sys.argv[1], Path(sys.argv[2])
    bad = {"v1": v1, "v2": v2}[which](run)
    if bad:
        print(f"{which} dry run: {len(bad)} failure(s)")
        for b in bad:
            print("FAIL", b)
        sys.exit(1)
    print(f"{which} dry run: every row exact")
