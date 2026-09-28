"""Error analysis on the validation fold (development tool, not part of the pipeline).

    python analyze.py [n_examples]

Prints sampled false positives (with the record's real entity, if any) and
false negatives, plus per-country F0.5 / precision / recall.  Only the raw rows
that are displayed are read (lazy parquet scans), so it is memory-light.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import raw_parquet, wpath
from decide import expected_f_select, macro_f05, threshold_select
from model import VAL_FOLD


def _raw1(rows):
    return (pl.scan_parquet(raw_parquet("train", "source1")).with_row_index("e_row")
            .filter(pl.col("e_row").is_in([int(r) for r in rows]))
            .select(pl.col("e_row").cast(pl.Int32), pl.col("business_name").alias("s1_name"),
                    pl.col("business_address").alias("s1_addr")).collect())


def _rawq(rows):
    n2 = pl.scan_parquet(raw_parquet("train", "source2")).select(pl.len()).collect().item()
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    a = (pl.scan_parquet(raw_parquet("train", "source2")).with_row_index("q_row")
         .filter(pl.col("q_row").is_in(rows[rows < n2].tolist())).collect())
    b = (pl.scan_parquet(raw_parquet("train", "source3")).with_row_index("q_row")
         .with_columns(pl.col("q_row") + n2)
         .filter(pl.col("q_row").is_in(rows[rows >= n2].tolist())).collect())
    return pl.concat([a, b]).select(pl.col("q_row").cast(pl.Int32), pl.col("business_name").alias("q_name"),
                                    pl.col("business_address").alias("q_addr"))


def main(n=25):
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_width_chars(330)
    pl.Config.set_tbl_rows(n + 5)
    dec = json.load(open(wpath("models", "decision.json")))
    ex = pl.read_parquet(wpath("feat", "val_pred.parquet"))
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    ents = np.flatnonzero(keys["e_fold"] == VAL_FOLD)
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(pl.Series(ents).implode()))
    sel = (threshold_select(ex, dec["param"]) if dec["rule"] == "thr"
           else expected_f_select(ex, "p", dec["param"]))
    sel = sel.with_columns(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32))
    fp = sel.join(truth, on=["q_row", "e_row"], how="anti")
    fn = truth.join(sel, on=["q_row", "e_row"], how="anti")
    fn_in = fn.join(ex.select("q_row", "e_row", "p"), on=["q_row", "e_row"])
    print(f"selected {sel.height}, FP {fp.height}, FN {fn.height} (of which in candidates {fn_in.height})")
    tq = true_pairs_rows().rename({"e_row": "true_e"})
    fpx = fp.join(ex, on=["q_row", "e_row"]).sample(min(n, fp.height), seed=1).join(tq, on="q_row", how="left")
    fnx = fn_in.sample(min(n, fn_in.height), seed=2)
    raw1 = _raw1(set(fpx["e_row"].to_list()) | set(fnx["e_row"].to_list())
                 | set(fpx["true_e"].drop_nulls().to_list()))
    rawq = _rawq(set(fpx["q_row"].to_list()) | set(fnx["q_row"].to_list()))
    print("\n=== FALSE POSITIVES ===  (true_* = the record's real S1 entity, if any)")
    print(fpx.join(rawq, on="q_row").join(raw1, on="e_row")
          .join(raw1.select(pl.col("e_row").alias("true_e"), pl.col("s1_name").alias("true_name"),
                            pl.col("s1_addr").alias("true_addr")), on="true_e", how="left")
          .select("p", "q_name", "q_addr", "s1_name", "s1_addr", "true_name", "true_addr"))
    print("\n=== FALSE NEGATIVES (in candidates) ===")
    print(fnx.join(rawq, on="q_row").join(raw1, on="e_row").select("p", "q_name", "q_addr", "s1_name", "s1_addr"))
    ctry = keys["e_country"]
    for c in np.unique(ctry[ents]):
        e_c = pl.DataFrame({"e_row": ents[ctry[ents] == c]})
        t = truth.join(e_c, on="e_row")
        s = sel.join(e_c, on="e_row")
        tp = s.join(t, on=["e_row", "q_row"]).height
        print(f"{c}: F0.5 {macro_f05(sel, truth, e_c):.5f}  precision {tp / max(1, s.height):.5f}  "
              f"recall {tp / max(1, t.height):.5f}  entities {e_c.height}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
