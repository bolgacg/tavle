"""The ideas 3 to 6 day model (model/deep_day.py, train_deep_day.py walk) on the GPU, on the deep agent's own
synthetic fixtures: the check behind running any rerun of `train_deep_day.py wf` in the GPU lane with
`--device cuda`. Run on gene from ~/nyiso-us/model:

    ../.venv/bin/python -m pytest -q -p no:cacheprovider -p conftest ../robust/test_day_gpu.py
"""
import datetime as dt

import numpy as np
import pytest
import torch

import train_deep_day as TDD
from test_deep_day import TINY, _locked, feats  # noqa: F401  (fixtures)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_walk_on_cuda_matches_cpu_shape(feats):  # noqa: F811
    span = (dt.date(2023, 2, 1), dt.date(2023, 3, 31))
    gpu, recs = TDD.walk(feats, span, seeds=(0, 1), device="cuda", config=TINY, log=lambda *_: None)
    cpu, _ = TDD.walk(feats, span, seeds=(0, 1), device="cpu", config=TINY, log=lambda *_: None)
    assert len(recs) == 2 and len(gpu) == len(cpu)
    assert np.isfinite(gpu["p_storm"]).all() and gpu["p_storm"].between(0, 1).all()
    assert np.isfinite(gpu["pred_zone_supply_pnl"]).all()
