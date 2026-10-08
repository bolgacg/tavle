# New York study: model code

Governing documents: `../OBJECTIVES.md` (with dated addenda) and `CONTRACT.md`. Data layer: `../pipeline/`.
Code is canonical here on the laptop. To run it, rsync it to gene at `~/nyiso-us/model/` (venv `~/nyiso-us/.venv`).

| File | Owner | What |
|---|---|---|
| `lock.py` | lead modeller | Holdout lock. Refuses delivery dates on or after 2024-01-01 unless FREEZE, a clean checkout and `US_HOLDOUT_RUN=1` all hold. |
| `fees.py` | lead modeller | NYISO Schedule 1 non-physical rate per year, with sources. |
| `panel.py` | lead modeller | Panel and base features at 05:00 on D, and a LockedStore that never loads holdout rows. |
| `gbm.py` | lead modeller | LightGBM regression (`GBMModel`) and the idea A spike classifier (`SpikeClassifier`). |
| `strategies.py` | lead modeller | Positions and money: baseline, always-supply, A (spike rule), A v1, B, C, D. |
| `score.py` | lead modeller | Stationary bootstrap, Holm, verdict words, ranking-table figures. |
| `context.py` | lead modeller | Context beside the verdicts: own profit, annualised Sharpe, bankroll fields. |
| `walkforward.py` | lead modeller | Monthly refits on data up to two days before each month. |
| `rehearsal.py` | lead modeller | 2023 rehearsal (v2), rescore, and the held-out stage. |
| `heldout.sh` | lead modeller | The held-out procedure (below). |
| `spike_config.json`, `frozen_choices.json` | written by `rehearsal.py gbm` | The spike threshold S for both models, and every build-year choice the held-out run reads. |
| `features_weather.py`, `features_outages.py` | features agent | Inputs for ideas B and D. |
| `deep.py`, `deep_data.py`, `train_deep.py` | deep agent | The deep model. |

Tests: `python -m pytest -q -p no:cacheprovider` here. The `*_real` tests need gene's parquet and skip elsewhere.

## Rehearsal (2023, development data)

```
cd ~/nyiso-us/model
../.venv/bin/python rehearsal.py gbm        # writes results/rehearsal_2023_v2.json, spike_config.json, frozen_choices.json
../.venv/bin/python rehearsal.py deep       # deep model inside D (outage sites recounted), then rescore
../.venv/bin/python rehearsal.py rescore    # scores again from cached predictions, including deep spike predictions once delivered
```

## The held-out run (prepared, not run)

This runs once, only on the orchestrator's instruction, and only after these steps:

1. Every file the run needs is committed and pushed to origin/master. That includes `frozen_choices.json`, `spike_config.json` and no untracked `.py` files in `model/` or `pipeline/`.
2. `research/us/FREEZE` names that commit. FREEZE itself is committed later: a commit cannot hold its own hash.

Then, from the laptop:

```
research/us/model/heldout.sh --check   # everything up to "the lock opens for this checkout"; runs nothing
research/us/model/heldout.sh           # the same, then starts rehearsal.py heldout on gene under nohup
```

The script does the following:

- On the laptop, it fetches origin/master and checks that the FREEZE commit is on it. It bundles origin/master and copies the bundle and FREEZE to `gene:~/nyiso-us/heldout/<commit>/`.
- On gene, it fetches the bundle into an empty repository and checks out the FREEZE commit (detached). It places FREEZE beside the code and checks that `model/` and `pipeline/` are clean.
- It runs `lock.unlock_status()` in that checkout. This is the same check every loader makes: the FREEZE hash, `git diff --quiet <hash> -- model pipeline`, no untracked `.py` files, and `US_HOLDOUT_RUN=1`.
- It runs `rehearsal.py heldout` with its own cache directory (`US_CACHE_DIR`), so no build-year cache is reused.

The held-out stage covers January 2024 to September 2026, refitting every month on data up to two days earlier. It reads every choice from `frozen_choices.json`:

- the regression configurations;
- the spike S and p*;
- C's pairs;
- for D, the outage sites counted on 2020 to 2023.

It writes `results/heldout_202401_202609_<commit>_<time>.json` inside the checkout. The script prints the `scp` line that copies it back. A rerun after a bug fix gets a new time-stamped file, published beside the first (objective 4).
