"""The battery study: what a one-megawatt, two-megawatt-hour battery earns in Danish balancing
activations, as a price taker, at the prices Energinet actually paid; and whether knowing the
last settled direction helps.

Reads research/results/activation_hours.parquet (activation_data.py). Writes
research/results/activation.json for the page (tavle/batterypage.py). The browser replays the
same per-hour arrays by the same rules.

Registered in research/PREREGISTRATION.md, section "The battery". Training: November 2021 to
December 2023, even ISO weeks choose the reservation prices; separate: January 2024 to 4 March
2025, replayed once.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN = ROOT / "research" / "results" / "activation_hours.parquet"
OUT = ROOT / "research" / "results" / "activation.json"

P_MW = 1.0            # power, MW
E_MWH = 2.2           # usable energy, MWh: a two-hour-class store with room for a full charge and a full delivery after losses
ETA = 0.949           # one-way efficiency (0.90 round trip)
CYCLE = 30.0          # EUR per MWh delivered (degradation)
S0 = 1.1              # starting state, MWh (either action feasible)
GRID = [0, 5, 10, 20, 40, 80]   # reservation margins over the day-ahead price, EUR per MWh
SINGLE = pd.Timestamp("2021-11-01")
SEP = pd.Timestamp("2024-01-01")
CAPEX_PER_MW_YEAR = (30000, 60000)   # the range a two-hour battery costs per MW-year, for the verdict


def simulate(z, m_up, m_down, cycle=CYCLE, by_dir=None, s0=S0, eta=ETA, e_max=E_MWH):
    """m_up, m_down: reservation margins; by_dir: optional dict {1:(mu,md), 0:(mu,md), -1:(mu,md)} keyed by
    the last settled direction d(t-2). Returns per-hour action (+1 discharged, -1 charged, 0 none), cash, state."""
    spot, up, down, dir_, upa, downa = z["spot"], z["up"], z["down"], z["dir"], z["up_act"], z["down_act"]
    n = len(spot); s = s0
    act = np.zeros(n, dtype=np.int8); cash = np.zeros(n); state = np.zeros(n); offered_up = np.zeros(n, bool); offered_down = np.zeros(n, bool)
    need_up = 1.0 / eta; room_down = e_max - eta
    for t in range(n):
        if np.isnan(spot[t]) or np.isnan(up[t]) or np.isnan(down[t]):
            state[t] = s; continue
        mu, md = m_up, m_down
        if by_dir is not None:
            d = dir_[t - 2] if t >= 2 and not np.isnan(dir_[t - 2]) else 0.0
            mu, md = by_dir[int(d)]
        can_up = s >= need_up - 1e-9
        can_down = s <= room_down + 1e-9
        r_up = spot[t] + mu; r_down = spot[t] - md
        offered_up[t] = can_up; offered_down[t] = can_down
        did = 0
        if can_up and upa[t] and up[t] >= r_up - 1e-9 and not (downa[t] and dir_[t] == -1):
            did = 1
        elif can_down and downa[t] and down[t] <= r_down + 1e-9 and not (upa[t] and dir_[t] == 1):
            did = -1
        if did == 1:
            s -= need_up; cash[t] = up[t] * P_MW - cycle * P_MW
        elif did == -1:
            s += eta; cash[t] = -down[t] * P_MW
        act[t] = did; state[t] = s
    return act, cash, state, offered_up, offered_down


def summarise(z, act, cash, off_up, off_down, m, years):
    hrs = int(m.sum()); a = act[m]; c = cash[m]
    ups = a == 1; downs = a == -1
    up_rev = float(c[ups].sum()) if ups.any() else 0.0
    down_cost = float(-c[downs].sum()) if downs.any() else 0.0   # positive = paid to charge; negative = was paid to charge
    net = float(c.sum())
    yrs = hrs / 8766.0
    up_act_hours = int((z["up_act"][m]).sum()); down_act_hours = int((z["down_act"][m]).sum())
    out = {"hours": hrs, "years": round(yrs, 3), "discharges": int(ups.sum()), "charges": int(downs.sum()),
           "offered_up": int(off_up[m].sum()), "offered_down": int(off_down[m].sum()),
           "market_up_hours": up_act_hours, "market_down_hours": down_act_hours,
           "revenue_up": round(up_rev, 0), "cost_down": round(down_cost, 0), "net": round(net, 0),
           "net_per_mw_year": round(net / yrs, 0) if yrs else None,
           "net_per_mwh_delivered": round(net / max(1, ups.sum()), 1),
           "mean_up_price_captured": round(float(z["up"][m][ups].mean()), 1) if ups.any() else None,
           "mean_spot_when_discharging": round(float(z["spot"][m][ups].mean()), 1) if ups.any() else None,
           "mean_down_price_paid": round(float(z["down"][m][downs].mean()), 1) if downs.any() else None,
           "by_year": []}
    for y in np.unique(years[m]):
        mm = m & (years == y)
        if mm.sum():
            out["by_year"].append({"year": int(y), "net": round(float(cash[mm].sum()), 0), "discharges": int((act[mm] == 1).sum()), "charges": int((act[mm] == -1).sum())})
    return out


def zone_run(df):
    df = df.sort_values("hour_utc").reset_index(drop=True)
    z = {k: np.round(df[c].values.astype(float), 2) for k, c in [("spot", "spot_eur"), ("up", "up_eur"), ("down", "down_eur")]}  # the browser replays the same two-decimal prices
    z["dir"] = df["dir"].values.astype(float)
    z["up_act"] = df["up_activated"].fillna(False).values.astype(bool); z["down_act"] = df["down_activated"].fillna(False).values.astype(bool)
    years = df.hour_utc.dt.year.values
    iso = df.hour_utc.dt.isocalendar(); even = (iso.week.astype(int) % 2 == 0).values
    single = df["single_pricing"].values; train = (df["period"] == "training").values; sep = (df["period"] == "separate").values
    meas = single & train & even
    # machine A: one reservation pair, chosen on the even weeks (the battery state runs through all hours)
    best = None; gridA = []
    for mu in GRID:
        for md in GRID:
            act, cash, st, ou, od = simulate(z, mu, md)
            net = float(cash[meas].sum()); gridA.append({"m_up": mu, "m_down": md, "net_measured": round(net, 0)})
            if best is None or net > best[0]: best = (net, mu, md)
    _, muA, mdA = best
    # machine B: the pair per last settled direction, one pass of coordinate search from A's pair
    by = {1: (muA, mdA), 0: (muA, mdA), -1: (muA, mdA)}
    gridB = []
    for d in (1, 0, -1):
        bestd = None
        for mu in GRID:
            for md in GRID:
                trial = dict(by); trial[d] = (mu, md)
                act, cash, *_ = simulate(z, muA, mdA, by_dir=trial)
                net = float(cash[meas].sum()); gridB.append({"dir": d, "m_up": mu, "m_down": md, "net_measured": round(net, 0)})
                if bestd is None or net > bestd[0]: bestd = (net, mu, md)
        by[d] = (bestd[1], bestd[2])
    arms = {}
    def arm(name, **kw):
        act, cash, st, ou, od = simulate(z, **kw)
        arms[name] = {p: summarise(z, act, cash, ou, od, mm, years) for p, mm in [("separate", sep), ("measured", meas), ("training_single", train & single)]}
        return act, cash, st, ou, od
    arm("always_at_cost", m_up=0, m_down=0)
    actA, cashA, stA, ouA, odA = arm("flat", m_up=muA, m_down=mdA)
    actB, cashB, stB, ouB, odB = arm("direction_aware", m_up=muA, m_down=mdA, by_dir=by)
    # cycle-cost sensitivity for the flat machine, separate period
    sens = []
    for cyc in (0, 15, 30, 60):
        act, cash, *_ = simulate(z, muA, mdA, cycle=cyc)
        sens.append({"cycle": cyc, "net_separate": round(float(cash[sep].sum()), 0), "discharges": int((act[sep] == 1).sum())})
    # the market's own ruler on the separate period: every activated hour, the premium and discount available
    up_avail = float(np.where(z["up_act"][sep], z["up"][sep] - z["spot"][sep], 0).sum()); down_avail = float(np.where(z["down_act"][sep], z["spot"][sep] - z["down"][sep], 0).sum())
    # a worst-hours list for the direction-aware machine on the separate period (largest losses)
    idx = np.where(sep & (actB != 0))[0]
    worst = idx[np.argsort(cashB[idx])[:5]]
    worst_rows = [{"hour_utc": str(df.hour_utc[i])[:16], "action": "discharge" if actB[i] == 1 else "charge", "spot": round(float(z["spot"][i]), 1),
                   "price": round(float(z["up"][i] if actB[i] == 1 else z["down"][i]), 1), "cash": round(float(cashB[i]), 1), "state_after": round(float(stB[i]), 2)} for i in worst]
    export = {
        "start": str(df.hour_utc.iloc[0])[:16], "hours": int(len(df)),
        "spot": [None if np.isnan(v) else round(float(v), 2) for v in z["spot"]],
        "up": [None if np.isnan(v) else round(float(v), 2) for v in z["up"]],
        "down": [None if np.isnan(v) else round(float(v), 2) for v in z["down"]],
        "dir": [None if np.isnan(v) else int(v) for v in z["dir"]],
        "up_act": [int(v) for v in z["up_act"]], "down_act": [int(v) for v in z["down_act"]],
        "sep_start_index": int(np.argmax(sep)), "single_start_index": int(np.argmax(single)),
        "chosen": {"flat": {"m_up": muA, "m_down": mdA}, "by_dir": {str(k): list(v) for k, v in by.items()}},
        "gridA": gridA, "gridB": gridB, "arms": arms, "cycle_sensitivity": sens,
        "market_separate": {"up_premium_available": round(up_avail, 0), "down_discount_available": round(down_avail, 0),
                            "up_hours": int(z["up_act"][sep].sum()), "down_hours": int(z["down_act"][sep].sum())},
        "worst_hours": worst_rows,
    }
    return export


def main():
    df = pd.read_parquet(IN)
    out = {"p_mw": P_MW, "e_mwh": E_MWH, "eta": ETA, "cycle": CYCLE, "s0": S0, "grid": GRID, "capex_per_mw_year": CAPEX_PER_MW_YEAR,
           "single_from": str(SINGLE.date()), "separate_from": str(SEP.date()), "zones": {}}
    for area in ["DK1", "DK2"]:
        e = zone_run(df[df.area == area].copy()); out["zones"][area] = e
        a = e["arms"]
        print(f"{area}: flat {e['chosen']['flat']} by_dir {e['chosen']['by_dir']} | separate net per MW-year: at cost {a['always_at_cost']['separate']['net_per_mw_year']}, "
              f"flat {a['flat']['separate']['net_per_mw_year']}, direction-aware {a['direction_aware']['separate']['net_per_mw_year']} | discharges {a['direction_aware']['separate']['discharges']} charges {a['direction_aware']['separate']['charges']} "
              f"| measured net: flat {a['flat']['measured']['net']} dir {a['direction_aware']['measured']['net']}")
    OUT.write_text(json.dumps(out, default=float))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
