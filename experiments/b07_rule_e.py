"""Rule E (reference): a record at the S1's exact address with an unrelated (pseudo-word) name, no other S1 at that
address.  Precision on fold V under the stage-2-only probability (the France situation: no cross-encoder), and the
volume of such unlinked records in France."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from blocking import load_universe
from common import wpath
from decide import final_prob
from b04_links_deep import features

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(20)
PB = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9, 1.01]


def s1_address_keys(split):
    s1, q = load_universe(split, ["country", "a_nums", "a_street", "n_core"])
    s1 = s1.with_columns(pl.col("a_nums").fill_null("").str.split(" ").list.sort().list.join(" ").alias("ak"))
    s1 = s1.with_columns(pl.when(pl.col("ak") != "").then(pl.len().over(["country", "ak", "a_street"])).otherwise(0).alias("n_at_addr"))
    words = pl.concat([s1.select("country", "n_core"), q.select("country", "n_core")])
    real = (words.with_columns(pl.col("n_core").fill_null("").str.split(" ").list.unique()).explode("n_core")
            .group_by("country", "n_core").len().filter(pl.col("len") >= 30))
    return s1.select(pl.col("row").alias("e_row"), "n_at_addr"), {(c, w) for c, w in zip(real["country"].to_list(), real["n_core"].to_list())}, q


def flags(split, rec):
    """rec: e_row, q_row, p (+ extra).  Adds rule-E pattern flags."""
    ek, real, q = s1_address_keys(split)
    f = features(split, rec)
    f = f.join(ek.with_columns(pl.col("e_row").cast(f["e_row"].dtype)), on="e_row")
    sh, pseudo = [], []
    for c, en, qn in zip(f["country"].to_list(), f["en"].to_list(), f["qn"].to_list()):
        a, b = set((en or "").split()), set((qn or "").split())
        sh.append(len(a & b))
        pseudo.append(bool(b) and any((c, t) not in real for t in b))
    f = f.with_columns(pl.Series("shared_tok", sh), pl.Series("pseudo", pseudo))
    return f.with_columns((pl.col("nr").eq("same") & pl.col("st").is_in(["match", "na"]) & (pl.col("shared_tok") == 0)).alias("E_addr"),
                          (pl.col("n_at_addr") == 1).alias("alone"))


if __name__ == "__main__":
    # ---- fold V, stage-2-only probability (record argmax)
    pr = pl.read_parquet(wpath("feat", "train_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "fold", "y")
    pf = final_prob(np.load(wpath("feat", "train_p1.npy")), np.load(wpath("feat", "train_p2.npy")))
    v = pr.with_columns(pl.Series("p", pf)).filter(pl.col("fold") == 4)
    v = v.sort(["q_row", "p"], descending=[False, True]).group_by("q_row", maintain_order=True).agg(
        pl.col("e_row").first(), pl.col("p").first(), pl.col("y").first(), pl.col("p").slice(1, 1).first().fill_null(0).alias("p2"))
    v = v.filter(pl.col("p") >= 0.01).with_columns(pl.lit(0).alias("other"))
    fv = flags("train", v.select("e_row", "q_row", "p", "p2", "y", "other"))
    fv = fv.with_columns(pl.col("p").cut(PB, left_closed=True).alias("pbin"))
    fv.write_parquet(r"D:/er_work/an_rule_e_v.parquet")
    print("== fold V (stage-2 only): same address, no shared name token ==")
    print(fv.filter(pl.col("E_addr")).group_by("alone", "pseudo", "pbin").agg(pl.len(), pl.col("y").mean().round(3).alias("prec")).sort("alone", "pseudo", "pbin"))
    # ---- test France: v10c-rule probabilities, v11 links
    gt = pl.read_parquet(r"D:/er_work/an_cells_test_rec.parquet").filter((pl.col("country") == "France") & (pl.col("p1") >= 0.01))
    ft = flags("test", gt.select("e_row", "q_row", pl.col("p1").alias("p"), "p2", "other", "linked"))
    ft = ft.with_columns(pl.col("p").cut(PB, left_closed=True).alias("pbin"))
    ft.write_parquet(r"D:/er_work/an_rule_e_fr.parquet")
    print("== test France: same address, no shared name token ==")
    print(ft.filter(pl.col("E_addr")).group_by("alone", "pseudo", "pbin").agg(pl.len(), pl.col("linked").mean().round(3).alias("linked"),
                                                                            (pl.col("other") == 0).mean().round(3).alias("z")).sort("alone", "pseudo", "pbin"))
