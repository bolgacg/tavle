"""Part B, revised books (revised limits): drawdown limits 35,000 (unlevered) and 75,000 (levered) on 500,000.
Mix = 0.8 x pairs at its build scale + 0.2 x V15c at its build scale; multiplier k solved so the 2013-2023 worst
drawdown equals the limit; cautious = 0.7 k. Pairs-only book at the same limits for comparison. Read-only."""
import json
import numpy as np, pandas as pd
RUN = "/home/bo/nyiso-us_heldout/run"; OUT = "/home/bo/analysis_20261011"
v2 = pd.read_parquet(f"{RUN}/ho_v2/results/v2/v2_daily_all_heldout.parquet")
bj = {r["name"]: r for r in json.load(open("/home/bo/nyiso-us/results/v2/v2_results.json"))["rows"]}
sp = bj["C_static_pairs"]["sizing_view"]["scale_for_100k_drawdown"]
sv = bj["V15c_wx_border_gbm"]["sizing_view"]["scale_for_100k_drawdown"]
idx = v2.index
build = (idx.year >= 2013) & (idx.year <= 2023)
ho = idx >= "2024-01-01"
ev = ((idx >= "2026-01-24") & (idx <= "2026-02-02")) | ((idx >= "2026-06-28") & (idx <= "2026-07-05"))
base = {"mix 0.8 pairs + 0.2 V15c": 0.8 * sp * v2["C_static_pairs"] + 0.2 * sv * v2["V15c_wx_border_gbm"],
        "pairs only": v2["C_static_pairs"]}


def mdd(d):
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(d, float))])
    return float((cum - np.maximum.accumulate(cum)).min())


def st(d):
    return {"ret/yr %": round(100 * float(d.sum()) / 500_000 * 365 / len(d), 1),
            "Sharpe": round(float(d.mean() / d.std() * np.sqrt(365)), 2),
            "worst DD": round(mdd(d)), "worst day": round(float(d.min())), "worst day date": str(d.idxmin().date())}


L = [f"build-year 100k-DD scales: pairs {sp} MW, V15c {sv} MW. Mix unit = 0.8x{sp} MW pairs + 0.2x{sv} MW V15c "
     f"({0.8*sp:.3f} + {0.2*sv:.3f} MW). Pairs-only unit = 1 MW per leg."]
rows = []
for bname, B in base.items():
    unit_dd = -mdd(B[build])           # drawdown scales linearly with the multiplier
    for lim in (35_000, 75_000):
        k = lim / unit_dd
        for tag, kk in (("full", k), ("cautious 70%", 0.7 * k)):
            d = kk * B
            r = {"book": bname, "limit": lim, "size": tag, "multiplier": round(kk, 3)}
            if bname.startswith("mix"):
                r["pairs MW"] = round(kk * 0.8 * sp, 3); r["V15c MW"] = round(kk * 0.2 * sv, 3)
            else:
                r["pairs MW"] = round(kk, 3); r["V15c MW"] = 0.0
            for pl, sel in (("build", build), ("held-out", ho), ("held-out w/o events", ho & ~ev)):
                s = st(d[sel])
                for kx, vx in s.items():
                    r[f"{pl} {kx}"] = vx
            r["held-out DD within limit"] = r["held-out worst DD"] >= -lim
            for y in (2024, 2025, 2026):
                r[f"{y} ret %"] = round(100 * float(d[idx.year == y].sum()) / 500_000 * 365 / int((idx.year == y).sum()), 1)
            rows.append(r)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
L.append(T.T.to_string())
open(f"{OUT}/b_books.txt", "w").write("\n".join(L) + "\n")
T.to_csv(f"{OUT}/b_books.csv", index=False)
print("\n".join(L))
