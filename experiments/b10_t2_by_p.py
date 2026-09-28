"""T2 (France one-word swap to another real word -> unlink) by the model's p: copy-twin rate and singleton share.
Reference rule F only unlinks 0.8 <= p < 0.99; ours unlinks at any p."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from common import wpath
import france as FR

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
p = np.load(wpath("feat", "test_pfinal.npy"))
g = FR.signatures(pairs, p)
m = FR.rule_masks(g, 0.8)
g = g.with_columns(m["T2"].alias("T2"), m["T3"].alias("T3"), m["A"].alias("A"),
                   (pl.col("sig") & (pl.col("ndrop") == 1) & (pl.col("pbest") >= 0.8)).alias("swap_linked"))
# twins: another candidate record of the same S1 with the same (added, dropped) word sets
s1, q = FR._norm("test")
cl = pairs.join(q.select(pl.col("q_row").cast(pl.Int32), pl.col("n_core").alias("qn")), on="q_row").join(
    s1.select(pl.col("e_row").cast(pl.Int32), pl.col("n_core").alias("en")), on="e_row")
sw = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
cl = cl.with_columns(sw("qn").alias("qw"), sw("en").alias("ew"))
cl = cl.with_columns(pl.col("qw").list.set_difference(pl.col("ew")).list.sort().list.join(" ").alias("add"),
                     pl.col("ew").list.set_difference(pl.col("qw")).list.sort().list.join(" ").alias("drop"))
cl = cl.with_columns(pl.len().over(["e_row", "add", "drop"]).alias("n_same_mod"))
g = g.join(cl.select(pl.col("q_row").cast(g["q_row"].dtype), pl.col("e_row").cast(g["e_row"].dtype), "n_same_mod"), on=["q_row", "e_row"], how="left")
# singleton share against the v11 France links: S1 has no other linked record (approx: other best pairs linked at p>=0.8, not T2/T3)
g = g.with_columns((pl.col("pbest") >= 0.8).alias("lk"))
g = g.with_columns((pl.col("lk") & ~pl.col("T2") & ~pl.col("T3")).alias("keep"))
nl = g.filter(pl.col("keep")).group_by("e_row").len().rename({"len": "nk"})
g = g.join(nl, on="e_row", how="left").with_columns(pl.col("nk").fill_null(0))
g = g.with_columns(pl.col("pbest").cut([0.8, 0.9, 0.99, 0.999], left_closed=True).alias("pbin"))
for name, sel in (("T2 unlinked", pl.col("T2")), ("A (generic swap, linked)", pl.col("A")), ("all other linked swaps", pl.col("swap_linked") & ~pl.col("T2") & ~pl.col("A"))):
    x = g.filter(sel)
    print(f"== {name}: {x.height}")
    print(x.group_by("pbin").agg(pl.len(), (pl.col("n_same_mod") >= 2).mean().round(4).alias("twin_rate"),
                                 (pl.col("nk") == 0).mean().round(4).alias("z_no_other_link")).sort("pbin"))
t2hi = g.filter(pl.col("T2") & (pl.col("pbest") >= 0.99))
print("T2 at p>=0.99, most common (dropped -> added):")
print(t2hi.group_by("dw", "aw").len().sort("len", descending=True).head(20))
