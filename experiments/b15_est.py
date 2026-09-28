"""Projected leaderboard gain of a delta file: validation gain per changed decision (old -> new full pipeline on V)
times the number of test changes applied, over all 1,732,544 test S1 (US / India changes only)."""
import sys, os
sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
import numpy as np, polars as pl
from b08_rank_rule import frame, rank_sel
from decide import macro_f05
from blocking import true_pairs_rows
from model import VAL_FOLD
from common import wpath
new_pf, new_suf, applied = sys.argv[1], sys.argv[2], int(sys.argv[3])
keys = np.load("D:/er_work/feat/train_ctxkeys.npz")
ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD).astype(np.int64)})
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).join(ents, on="e_row")
def best(d, suf):
    x = pl.read_parquet(wpath(d, f"train_rescue_pred{suf}.parquet")).filter(pl.col("fold") == VAL_FOLD)
    return x.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "prob").sort("prob", descending=True).unique("q_row", keep="first")
def add(L, R):
    return pl.concat([L, R.join(L.select("q_row"), on="q_row", how="anti").filter(pl.col("prob") >= 0.8).select("e_row", "q_row")])
def full(zf, suf):
    z = np.load(zf); return add(add(rank_sel(frame("train", z["rows"], z["p"]), 0.65, 0.8), best("rescue_v10", suf)), best("rescue2", suf))
Lo, Ln = full("D:/er_work/backup_v12/val_pfinal.npz", ""), full(new_pf, new_suf)
fo, fn = macro_f05(Lo, truth, ents), macro_f05(Ln, truth, ents)
d = Lo.rename({"e_row": "eo"}).join(Ln.rename({"e_row": "en"}), on="q_row", how="full", coalesce=True)
flips = d.filter(pl.col("eo").fill_null(-1) != pl.col("en").fill_null(-1)).height
g = (fn - fo) * ents.height / flips
print(f"V: old {fo:.5f} new {fn:.5f} ({fn - fo:+.5f}); {flips} changed decisions; gain per change {g:.4f}")
print(f"projected leaderboard gain over v12 for {applied} applied test changes: {g * applied / 1732544:+.5f} "
      f"-> about {0.9903 + 0.00005 + g * applied / 1732544:.5f} (v11 0.9903 + v12 rank rule ~0.00005)")
