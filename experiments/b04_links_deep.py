"""Where do France's high singleton shares come from?  Every linked record (v11 file for test; argmax & p >= 0.8
for fold V) with: number relation, street relation (a_tok street tokens, reference 'another street' test),
name class, name sharing, p-bin.  z per group; on V also precision and z of true vs false links."""
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from blocking import load_universe
from b01_editsig import sig_of
import france as FR

pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_width_chars(250)


def name_class(s):
    if s in ("SAME", "REORDER", "INS", "DEL", "MULTI", "SWAPn", "SWAPB"):
        return s
    return "EDIT"


def features(split, links):
    """links: e_row, q_row (+ anything).  Adds relation features."""
    cols = ["country", "n_core", "a_nums", "a_tok", "a_hn"]
    s1, q = load_universe(split, cols)
    s1 = s1.with_columns(pl.len().over(["country", "n_core"]).alias("kname"))
    # rare street-token vocabulary per country (S1 address document frequency < 0.5 %)
    rare = {}
    for c in s1["country"].unique().to_list():
        a = s1.filter(pl.col("country") == c)
        vc = a["a_tok"].fill_null("").str.split(" ").list.unique().explode().value_counts()
        rare[c] = set(vc.filter(pl.col("count") < 0.005 * a.height)[vc.columns[0]].to_list())
    d = (links.join(s1.select(pl.col("row").cast(links["e_row"].dtype).alias("e_row"), "country", "kname", pl.col("n_core").alias("en"),
                              pl.col("a_nums").alias("enum"), pl.col("a_tok").alias("etok"), pl.col("a_hn").alias("ehn")), on="e_row")
              .join(q.select(pl.col("row").cast(links["q_row"].dtype).alias("q_row"), pl.col("n_core").alias("qn"),
                             pl.col("a_nums").alias("qnum"), pl.col("a_tok").alias("qtok"), pl.col("a_hn").alias("qhn")), on="q_row"))
    st, nm, nr = [], [], []
    for c, et, qt, en, qn, enum, qnum in zip(d["country"].to_list(), d["etok"].to_list(), d["qtok"].to_list(), d["en"].to_list(),
                                             d["qn"].to_list(), d["enum"].to_list(), d["qnum"].to_list()):
        R = rare[c]
        ta = [t for t in (qt or "").split() if t.isalpha() and len(t) >= 3 and t not in FR.STREET_STOP and t in R]
        tb = [t for t in (et or "").split() if t.isalpha() and len(t) >= 3 and t not in FR.STREET_STOP and t in R]
        if not qt:
            st.append("noaddr")
        elif not ta or not tb:
            st.append("na")
        else:
            st.append("match" if max(fuzz.ratio(x, y) for x in ta for y in tb) >= 80 else "diff")
        nm.append(name_class(sig_of(en or "", qn or "")))
        a = sorted((qnum or "").split())
        b = sorted((enum or "").split())
        if not a:
            nr.append("q_none")
        elif not b:
            nr.append("e_none")
        elif a == b:
            nr.append("same")
        elif set(a) & set(b):
            nr.append("overlap")
        else:
            nr.append("diff")
    d = d.with_columns(pl.Series("st", st), pl.Series("nm", nm), pl.Series("nr", nr),
                       pl.when(pl.col("kname") == 1).then(pl.lit("1")).when(pl.col("kname") <= 3).then(pl.lit("2-3"))
                       .otherwise(pl.lit("4+")).alias("share"))
    return d


if __name__ == "__main__":
    PB = [0.8, 0.9, 0.97, 0.995]
    # ---- fold V
    gv = pl.read_parquet(r"D:/er_work/an_cells_v_rec.parquet").filter(pl.col("linked"))
    fv = features("train", gv.select("e_row", "q_row", "p1", "p2", "other", "y"))
    fv = fv.with_columns(pl.col("p1").cut(PB, left_closed=True).alias("pbin"))
    print("== V: z of true vs false links by p-bin ==")
    print(fv.group_by("pbin").agg(pl.len(), pl.col("y").mean().round(4).alias("prec"),
                                  (pl.col("other") == 0).filter(pl.col("y") == 1).mean().round(4).alias("z_true"),
                                  (pl.col("other") == 0).filter(pl.col("y") == 0).mean().round(4).alias("z_false")).sort("pbin"))
    fv.write_parquet(r"D:/er_work/an_links_v.parquet")
    # ---- test (v11 links), p from the rebuild with v10c France rules
    gt = pl.read_parquet(r"D:/er_work/an_cells_test_rec.parquet").filter(pl.col("linked"))
    ft = features("test", gt.select("e_row", "q_row", "p1", "p2", "other"))
    ft = ft.with_columns(pl.col("p1").cut(PB, left_closed=True).alias("pbin"))
    ft.write_parquet(r"D:/er_work/an_links_test.parquet")
    keys = ["nr", "st", "nm"]
    for lo in ("mid",):
        sub_v = fv.filter((pl.col("p1") >= 0.8) & (pl.col("p1") < 0.995))
        sub_t = ft.filter((pl.col("p1") >= 0.8) & (pl.col("p1") < 0.995))
        a = sub_v.group_by(keys).agg(pl.len().alias("nV"), pl.col("y").mean().round(3).alias("precV"), (pl.col("other") == 0).mean().round(3).alias("zV"))
        for c in ("France", "US", "India"):
            b = sub_t.filter(pl.col("country") == c).group_by(keys).agg(pl.len().alias(f"n{c[:2]}"), (pl.col("other") == 0).mean().round(3).alias(f"z{c[:2]}"))
            a = a.join(b, on=keys, how="full", coalesce=True)
        print("== links with 0.8 <= p < 0.995: V precision/z and test z by (number rel, street rel, name class) ==")
        print(a.filter(pl.max_horizontal(pl.col("nV").fill_null(0), pl.col("nFr").fill_null(0)) >= 150).sort("nFr", descending=True, nulls_last=True))
