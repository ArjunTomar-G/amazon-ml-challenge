"""Full v12-style decision on fold V: stacked (current ce/val_pfinal.npz) + rank rule + rescue pass 1 + rescue pass 2,
for chosen rescue prediction files (suffix "" = old e5-small rescue models, "_b" = with e5-base)."""
import os, sys
sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
import numpy as np, polars as pl
from common import wpath
from blocking import true_pairs_rows
from decide import macro_f05
from model import VAL_FOLD
from b08_rank_rule import frame, rank_sel
keys = np.load(wpath("feat", "train_ctxkeys.npz"))
ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD).astype(np.int64)})
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).join(ents, on="e_row")
z = np.load(wpath("ce", "val_pfinal.npz"))
a = frame("train", z["rows"], z["p"])
def best(d, suf):
    x = pl.read_parquet(wpath(d, f"train_rescue_pred{suf}.parquet")).filter(pl.col("fold") == VAL_FOLD)
    return x.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "prob").sort("prob", descending=True).unique("q_row", keep="first")
def add(L, R, thr):
    return pl.concat([L, R.join(L.select("q_row"), on="q_row", how="anti").filter(pl.col("prob") >= thr).select("e_row", "q_row")])
for t1 in (0.8, 0.65):
    for s1_, s2_ in (("", ""), ("_b", ""), ("", "_b"), ("_b", "_b")):
        for thr in ((0.8,) if (s1_, s2_) != ("_b", "_b") else (0.7, 0.75, 0.8)):
            L = add(add(rank_sel(a, t1, 0.8), best("rescue_v10", s1_), thr), best("rescue2", s2_), thr)
            print(f"rank t1={t1}  rescue1{s1_ or '(old)':6s} rescue2{s2_ or '(old)':6s} thr {thr}: {macro_f05(L, truth, ents):.5f}")
