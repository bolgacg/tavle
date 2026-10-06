"""Walk-forward (CONTRACT.md, "Walk-forward"): a refit on the 1st of every month, trained on delivery
dates up to two days before that month starts, predicting every delivery date of the month.

Stricter than the date rule alone: a training row is used only if its label (the delivery hour's
real-time and day-ahead prices) was public by 05:00 on the bid day of the month's first delivery date.
Real-time files NYISO rewrote later (README.md, published_at rules) are therefore left out until the
rewrite, exactly as a trader on that morning would have had to.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Callable

import lock

import pandas as pd

import panel  # noqa: F401  (puts the pipeline folder on the path)
from common import decision_time

TRAIN_GAP_DAYS = 2


def month_starts(first: dt.date, last: dt.date) -> list[dt.date]:
    """1st of every month whose delivery dates intersect [first, last]."""
    out, m = [], dt.date(first.year, first.month, 1)
    while m <= last:
        out.append(m)
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def train_mask(panel: pd.DataFrame, month_start: dt.date, train_start: dt.date | None = None) -> pd.Series:
    """Rows a model refit on `month_start` may learn from."""
    cutoff = pd.Timestamp(month_start - dt.timedelta(days=TRAIN_GAP_DAYS))
    known_by = decision_time(month_start - dt.timedelta(days=1))
    m = (panel["delivery_date"] <= cutoff) & panel["gap"].notna() & (panel["label_published_at"] <= known_by)
    if train_start is not None:
        m &= panel["delivery_date"] >= pd.Timestamp(train_start)
    return m


def walk_forward(panel: pd.DataFrame, make_model: Callable[[], object], first: dt.date, last: dt.date,
                 train_start: dt.date | None = None, log: Callable[[str], None] = print
                 ) -> tuple[pd.Series, list[dict]]:
    """Predictions for every panel row with delivery date in [first, last], one refit per month."""
    lock.assert_build_only([first, last])
    preds, fits = [], []
    for ms in month_starts(first, last):
        me = dt.date(ms.year + (ms.month == 12), ms.month % 12 + 1, 1) - dt.timedelta(days=1)
        lo, hi = max(ms, first), min(me, last)
        test = panel[(panel["delivery_date"] >= pd.Timestamp(lo)) & (panel["delivery_date"] <= pd.Timestamp(hi))]
        tm = train_mask(panel, ms, train_start)
        train = panel[tm]
        assert train["delivery_date"].max() <= pd.Timestamp(ms - dt.timedelta(days=TRAIN_GAP_DAYS))
        assert train["delivery_date"].max() < test["delivery_date"].min()
        t0 = time.time()
        m = make_model()
        m.fit(train)
        p = m.predict(test)
        preds.append(p)
        rec = {"month": ms.isoformat(), "train_rows": int(len(train)),
               "train_first": str(train["delivery_date"].min().date()),
               "train_last": str(train["delivery_date"].max().date()),
               "labels_left_out_unpublished": int(((panel["delivery_date"] <= pd.Timestamp(ms - dt.timedelta(days=TRAIN_GAP_DAYS)))
                                                   & panel["gap"].notna() & ~tm
                                                   & (panel["delivery_date"] >= pd.Timestamp(train_start or dt.date(2000, 1, 1)))).sum()),
               "test_rows": int(len(test)), "seconds": round(time.time() - t0, 1)}
        fits.append(rec)
        log(f"  refit {ms}: train {rec['train_first']}..{rec['train_last']} ({rec['train_rows']} rows), "
            f"predict {len(test)} rows, {rec['seconds']} s")
    return pd.concat(preds).sort_index(), fits
