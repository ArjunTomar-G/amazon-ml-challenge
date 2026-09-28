"""V check: v10c rescue first (its links kept), then rescue v2 only for records still unlinked."""
import numpy as np, polars as pl
from blocking import true_pairs_rows
from common import log, wpath
from decide import macro_f05
from model import VAL_FOLD
keys = np.load(wpath("feat", "train_ctxkeys.npz"))
ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
truth = true_pairs_rows().filter(pl.col("e_row").is_in(ents["e_row"].implode()))
base = pl.read_parquet(wpath("ce", "val_links.parquet")).select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64))
def best(d, thr):
    x = pl.read_parquet(wpath(d, "train_rescue_pred.parquet")).filter(pl.col("fold") == VAL_FOLD)
    x = x.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "prob").sort("prob", descending=True).unique("q_row", keep="first")
    return x.filter(pl.col("prob") >= thr).select("e_row", "q_row")
f0 = macro_f05(base, truth, ents)
r1 = best("rescue_v10", 0.8).join(base.select("q_row"), on="q_row", how="anti")
L1 = pl.concat([base, r1]); f1 = macro_f05(L1, truth, ents)
for t in (0.75, 0.8, 0.85):
    r2 = best("rescue2", t).join(L1.select("q_row"), on="q_row", how="anti")
    f2 = macro_f05(pl.concat([L1, r2]), truth, ents)
    log(f"base {f0:.5f} | + v10c rescue {f1:.5f} ({f1-f0:+.5f}) | + rescue2 (thr {t}) on the rest: {f2:.5f} ({f2-f1:+.5f} over v10c rescue; +{r2.height} links)")
