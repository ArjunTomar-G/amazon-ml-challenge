"""Fold V (plain): v11 decision = stacked argmax p >= 0.8, + v10c rescue, + rescue v2 for records still unlinked.
Variants: rank rule on the stacked decision (t1 first record of an S1, t2 the others); first-link threshold for the
rescues (rescue pair whose S1 has no link yet)."""
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from common import wpath
from blocking import true_pairs_rows
from decide import macro_f05
from model import VAL_FOLD
from b08_rank_rule import frame, rank_sel

keys = np.load(wpath("feat", "train_ctxkeys.npz"))
ec = keys["e_country"]
z = np.load(wpath("ce", "val_pfinal.npz"))
ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD).astype(np.int64)})
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).join(ents, on="e_row")
a = frame("train", z["rows"], z["p"])


def rescue(d):
    x = pl.read_parquet(wpath(d, "train_rescue_pred.parquet")).filter(pl.col("fold") == VAL_FOLD)
    return (x.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "prob")
            .sort("prob", descending=True).unique("q_row", keep="first"))


R1, R2 = rescue("rescue_v10"), rescue("rescue2")


def add_rescue(L, R, thr, thr_first):
    r = R.join(L.select("q_row"), on="q_row", how="anti")
    has = L.select("e_row").unique().with_columns(pl.lit(True).alias("has"))
    r = r.join(has, on="e_row", how="left").with_columns(pl.col("has").fill_null(False))
    add = r.filter(pl.when(pl.col("has")).then(pl.col("prob") >= thr).otherwise(pl.col("prob") >= thr_first))
    # one first link per S1 among the additions (the most probable), when the S1 had none
    add = add.sort("prob", descending=True).with_columns(pl.col("e_row").cum_count().over("e_row").alias("k"))
    add = add.filter(pl.col("has") | (pl.col("k") == 1) | (pl.col("prob") >= thr))
    return pl.concat([L, add.select("e_row", "q_row")])


def full(t1, t2, rf):
    L = rank_sel(a, t1, t2)
    L = add_rescue(L, R1, 0.8, rf)
    L = add_rescue(L, R2, 0.8, rf)
    return macro_f05(L, truth, ents)


base = full(0.8, 0.8, 0.8)
print(f"v11 on V (plain): {base:.5f}")
for t1, t2, rf in itertools.product((0.6, 0.65, 0.7, 0.8), (0.8,), (0.6, 0.7, 0.8)):
    f = full(t1, t2, rf)
    print(f"  rank t1={t1} t2={t2} | rescue first-link thr {rf}: {f:.5f} ({f - base:+.5f})")
