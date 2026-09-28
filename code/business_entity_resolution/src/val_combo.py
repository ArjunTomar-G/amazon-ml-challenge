"""Validation (fold V, US / India) of rescue combinations on top of the stacked decisions.

    python val_combo.py rescue_v10 rescue2 ...

Base = ce/val_links.parquet (stacked p >= 0.8 on the record argmax).  For every unlinked V record the best
pair over the given rescue folders is added if its probability >= thr.
"""
import sys

import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import log, wpath
from decide import macro_f05
from model import VAL_FOLD

dirs = sys.argv[1:]
keys = np.load(wpath("feat", "train_ctxkeys.npz"))
ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
truth = true_pairs_rows().filter(pl.col("e_row").is_in(ents["e_row"].implode()))
tt = truth.select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64))
base = pl.read_parquet(wpath("ce", "val_links.parquet")).select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64))
f0 = macro_f05(base, truth, ents)
log(f"base (stacked, no rescue): {f0:.5f}")
def load(spec):
    """<rescue folder>[:noaddr] - ':noaddr' keeps only records without an address."""
    d, _, flt = spec.partition(":")
    x = pl.read_parquet(wpath(d, "train_rescue_pred.parquet"))
    if flt == "noaddr":
        x = x.filter(pl.col("q_no_addr") == 1)
    return x.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "fold", "prob")


preds = {d: load(d) for d in dirs}
combos = [[d] for d in dirs] + ([dirs] if len(dirs) > 1 else [])
for combo in combos:
    r = pl.concat([preds[d] for d in combo]).filter(pl.col("fold") == VAL_FOLD)
    r = r.join(base.select("q_row"), on="q_row", how="anti")
    best = r.sort("prob", descending=True).unique("q_row", keep="first")
    for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        add = best.filter(pl.col("prob") >= t).select("e_row", "q_row")
        f1 = macro_f05(pl.concat([base, add]), truth, ents)
        ntrue = add.join(tt, on=["e_row", "q_row"]).height
        log(f"{'+'.join(combo):22s} thr {t}: +{add.height} links ({ntrue} true, {ntrue / max(1, add.height):.3f}), "
            f"F0.5 {f1:.5f} ({f1 - f0:+.5f})")
