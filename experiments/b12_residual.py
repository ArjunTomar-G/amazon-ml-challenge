"""Residual errors of the v12 decision on fold V (stacked argmax >= 0.8 + rank rule + both rescues):
false links and missed true pairs with an address, with raw text, to judge what a stronger model could fix."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from b09_rank_rescue import a, add_rescue, R1, R2, truth, ents
from b08_rank_rule import rank_sel
from decide import macro_f05

L = add_rescue(add_rescue(rank_sel(a, 0.65, 0.8), R1, 0.8, 0.8), R2, 0.8, 0.8)
print("v12 on V:", round(macro_f05(L, truth, ents), 5))
R = "D:/er_work/raw"
s1 = pl.read_parquet(f"{R}/train_source1.parquet").with_row_index("e_row").with_columns(pl.col("e_row").cast(pl.Int64))
q = pl.concat([pl.read_parquet(f"{R}/train_source{i}.parquet") for i in (2, 3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.Int64))
tq = truth.rename({"e_row": "e_true"})
fp = L.join(truth, on=["e_row", "q_row"], how="anti").join(tq, on="q_row", how="left")
fn = truth.join(L, on=["e_row", "q_row"], how="anti")
pv = a.select("q_row", pl.col("e_row").alias("e_arg"), pl.col("p").alias("p_arg"))
fn = fn.join(pv, on="q_row", how="left")
qa = q.select("q_row", (pl.col("business_address").fill_null("").str.strip_chars() != "").alias("has_addr"))
fn = fn.join(qa, on="q_row")
print(f"false links: {fp.height} (record belongs to another S1: {fp['e_true'].is_not_null().sum()}); missed true pairs: {fn.height} "
      f"(with address: {fn['has_addr'].sum()})")
fnA = fn.filter(pl.col("has_addr"))
print("missed with address, by situation:",
      fnA.with_columns(pl.when(pl.col("e_arg").is_null()).then(pl.lit("no candidate (blocking miss)"))
                       .when(pl.col("e_arg") == pl.col("e_row")).then(pl.lit("argmax right, p < 0.8"))
                       .otherwise(pl.lit("argmax is another S1"))).group_by("literal").len().to_dicts())


def show(df, tag, n=18, seed=7):
    x = df.sample(min(n, df.height), seed=seed)
    x = x.join(s1.select("e_row", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="e_row", how="left")
    x = x.join(q.select("q_row", pl.col("business_name").alias("rn"), pl.col("business_address").alias("ra")), on="q_row", how="left")
    if "e_true" in x.columns:
        x = x.join(s1.select(pl.col("e_row").alias("e_true"), pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="e_true", how="left")
    if "e_arg" in x.columns:
        x = x.join(s1.select(pl.col("e_row").alias("e_arg"), pl.col("business_name").alias("an"), pl.col("business_address").alias("aa")), on="e_arg", how="left")
    print(f"\n===== {tag} =====")
    for r in x.iter_rows(named=True):
        print(f"  S1 : {r['sn']} || {r['sa']}\n  REC: {r['rn']} || {r['ra']}")
        if r.get("tn") is not None:
            print(f"  TRUE S1: {r['tn']} || {r['ta']}")
        elif "e_true" in r:
            print("  TRUE S1: <none - distractor>")
        if r.get("an") is not None and r.get("e_arg") != r.get("e_row"):
            print(f"  ARGMAX S1 (p={r['p_arg']:.2f}): {r['an']} || {r['aa']}")
        elif r.get("p_arg") is not None:
            print(f"  (p={r['p_arg']:.2f})")
        print()


show(fp, "FALSE LINKS (sample)")
show(fnA.filter(pl.col("e_arg") == pl.col("e_row")), "MISSED, with address, argmax right but p < 0.8 (sample)")
show(fnA.filter(pl.col("e_arg").is_not_null() & (pl.col("e_arg") != pl.col("e_row"))), "MISSED, with address, argmax is another S1 (sample)", n=10)
