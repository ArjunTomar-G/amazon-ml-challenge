"""Ambiguity cells (reference approach 'Cells'): name-sharing x record address x margin x p-bin.
Fold V (labels): precision and singleton share z per cell.  Test: z and link share per cell and country.
z = share of the record's argmax S1 that has no OTHER link (true links ~2 %, false 5-9 %)."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from blocking import load_universe, true_pairs_rows
from common import wpath
from model import VAL_FOLD

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(250)
PB = [0.0, 0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.97, 0.995, 1.01]


def records(pairs: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """one row per record: argmax S1, p1, p2 (best other S1)."""
    d = pairs.with_columns(pl.Series("p", p)).sort(["q_row", "p"], descending=[False, True])
    g = d.group_by("q_row", maintain_order=True).agg(pl.col("e_row").first(), pl.col("p").first().alias("p1"),
                                                     pl.col("p").slice(1, 1).first().alias("p2"), pl.len().alias("ncand"))
    return g.with_columns(pl.col("p2").fill_null(0.0))


def annotate(g, split, links):
    s1, q = load_universe(split, ["country", "n_core", "a_tok"])
    s1 = s1.with_columns(pl.len().over(["country", "n_core"]).alias("kname"))
    g = (g.join(s1.select(pl.col("row").cast(g["e_row"].dtype).alias("e_row"), "kname", "country"), on="e_row")
          .join(q.select(pl.col("row").cast(g["q_row"].dtype).alias("q_row"), (pl.col("a_tok").fill_null("") != "").alias("addr")), on="q_row"))
    lk = links.select(pl.col("e_row").cast(g["e_row"].dtype), pl.col("q_row").cast(g["q_row"].dtype))
    nl = lk.group_by("e_row").len().rename({"len": "nl"})
    g = g.join(lk.with_columns(pl.lit(True).alias("linked")), on=["e_row", "q_row"], how="left").with_columns(pl.col("linked").fill_null(False))
    g = g.join(nl, on="e_row", how="left").with_columns(pl.col("nl").fill_null(0))
    g = g.with_columns((pl.col("nl") - pl.col("linked").cast(pl.UInt32)).alias("other"))
    return g.with_columns(
        pl.when(pl.col("kname") == 1).then(pl.lit("1")).when(pl.col("kname") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")).alias("share"),
        (pl.col("p1") - pl.col("p2") >= 0.3).alias("marg"),
        pl.col("p1").cut(PB, left_closed=True).alias("pbin"))


def table(g, extra=()):
    return (g.group_by(["share", "addr", "marg", "pbin"]).agg(pl.len().alias("n"), pl.col("linked").mean().round(3).alias("linked"),
                                                            (pl.col("other") == 0).mean().round(4).alias("z"), *extra)
            .sort(["share", "addr", "marg", "pbin"]))


if __name__ == "__main__":
    # ---- fold V: stacked p (plain universe), links = argmax & p >= 0.8
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    z = np.load(wpath("ce", "val_pfinal.npz"))
    pr = pl.read_parquet(wpath("feat", "train_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "fold")
    vp = pr[z["rows"]]
    isv = vp["fold"].to_numpy() == VAL_FOLD
    gv = records(vp.filter(pl.Series(isv)).select("q_row", "e_row"), z["p"][isv])
    lv = gv.filter(pl.col("p1") >= 0.8).select("e_row", "q_row")
    gv = annotate(gv, "train", lv)
    tp = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).with_columns(pl.lit(1).alias("y"))
    gv = gv.join(tp, on=["q_row", "e_row"], how="left").with_columns(pl.col("y").fill_null(0))
    tv = table(gv, [pl.col("y").mean().round(4).alias("prec")])
    print("== fold V (US+India): per cell, precision of the argmax pair and z ==")
    print(tv.filter(pl.col("n") >= 200))
    tv.write_parquet(r"D:/er_work/an_cells_v.parquet")
    gv.write_parquet(r"D:/er_work/an_cells_v_rec.parquet")

    # ---- test: v10c's p with France rules as in v10c, links = the v11 file
    pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    p = np.load(wpath("feat", "test_pfinal.npy"))
    import france as FR
    p, _ = FR.adjust(pairs, p, 0.8, ("A", "A0", "B", "C", "D", "T2"))
    s1, q = load_universe("test", ["entity_id"])
    m = pl.read_csv(r"D:/amazon_last_ride/v11/output_hybrid/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    L = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "entity_id", "matched_entity_ids": "q_id"})
         .join(s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id"), on="entity_id")
         .join(q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id")), on="q_id").select("e_row", "q_row"))
    gt = annotate(records(pairs, p), "test", L)
    gt.write_parquet(r"D:/er_work/an_cells_test_rec.parquet")
    for c in ("France", "US", "India"):
        print(f"== test {c}: per cell ==")
        print(table(gt.filter(pl.col("country") == c)).filter(pl.col("n") >= 200))
