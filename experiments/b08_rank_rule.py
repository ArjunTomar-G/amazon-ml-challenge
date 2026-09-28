"""Rank rule (reference 3.7): per S1, its strongest record needs p >= t1, further records p >= t2.
Evaluated on fold V, plain and density-augmented (test-like) universes, per country of the S1."""
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from common import wpath
from blocking import true_pairs_rows
from decide import macro_f05
from model import VAL_FOLD

keys = np.load(wpath("feat", "train_ctxkeys.npz"))
ec = keys["e_country"]
z = np.load(wpath("ce", "val_pfinal.npz"))
truth_all = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))


def frame(tag, rows, p):
    pr = pl.read_parquet(wpath("feat", f"{tag}_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "fold")
    d = pr[rows].with_columns(pl.Series("p", p))
    d = d.with_columns(pl.col("p").max().over("q_row").alias("pm"))
    d = d.filter((pl.col("p") >= pl.col("pm")) & (pl.col("fold") == VAL_FOLD)).unique("q_row", keep="first").select("q_row", "e_row", "p")
    return d.with_columns(pl.col("p").rank("ordinal", descending=True).over("e_row").alias("rk"))


def rank_sel(a, t1, t2):
    return a.filter(pl.when(pl.col("rk") == 1).then(pl.col("p") >= t1).otherwise(pl.col("p") >= t2)).select("e_row", "q_row")


if __name__ == "__main__":
    U = {"plain": frame("train", z["rows"], z["p"]), "aug": frame("train_aug", z["aug_rows"], z["aug_p"])}
    grid = list(itertools.product((0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8), (0.7, 0.75, 0.8, 0.85)))
    for country in ("US", "India", "both"):
        e_ok = np.flatnonzero((keys["e_fold"] == VAL_FOLD) & ((ec == country) if country != "both" else True))
        ents = pl.DataFrame({"e_row": e_ok.astype(np.int64)})
        truth = truth_all.join(ents, on="e_row")
        res = {}
        for name, a in U.items():
            a = a.join(ents, on="e_row")
            base = macro_f05(a.filter(pl.col("p") >= 0.8).select("e_row", "q_row"), truth, ents)
            for t1, t2 in grid:
                res[(name, t1, t2)] = macro_f05(rank_sel(a, t1, t2), truth, ents) - base
            res[(name, "base")] = base
        print(f"== {country}: {len(e_ok)} V entities; base plain {res[('plain', 'base')]:.5f}, aug {res[('aug', 'base')]:.5f}")
        rows = sorted(grid, key=lambda g: -(res[("plain",) + g] + res[("aug",) + g]))
        for t1, t2 in rows[:10]:
            print(f"   t1={t1:<5} t2={t2:<5} plain {res[('plain', t1, t2)]:+.5f}  aug {res[('aug', t1, t2)]:+.5f}")
        for t1, t2 in ((0.6, 0.7), (0.6, 0.8), (0.7, 0.8)):
            print(f"   [ref] t1={t1} t2={t2}: plain {res[('plain', t1, t2)]:+.5f}  aug {res[('aug', t1, t2)]:+.5f}")
