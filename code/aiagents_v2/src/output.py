"""Stage 6: write matching_results.tsv and candidate_pairs.tsv for a split.

    python output.py test

matching_results.tsv   one row per Source-1 entity, matched S2/S3 ids (comma
                       separated, possibly empty)
candidate_pairs.tsv    one row per Source-1 entity, every S2/S3 id that the
                       matching model scored for it (the pruned candidate set)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from blocking import load_universe
from common import OUT_DIR, Timer, log, wpath
from decide import exclusive, expected_f_select, final_prob, threshold_select


def _id_lists(s1_ids: pl.DataFrame, pairs: pl.DataFrame, q_ids: pl.DataFrame, col: str) -> pl.DataFrame:
    lst = (pairs.join(q_ids, on="q_row")
           .sort(["e_row", "q_id"])
           .group_by("e_row", maintain_order=True)
           .agg(pl.col("q_id").unique(maintain_order=True).str.join(",").alias(col)))
    out = (s1_ids.join(lst, on="e_row", how="left")
           .with_columns(pl.col(col).fill_null(""))
           .sort("e_row")
           .select(pl.col("entity_id").alias("source1_entity_id"), col))
    return out


def write_outputs(split: str = "test", out_dir=None):
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    s1, q = load_universe(split, ["entity_id"])
    s1_ids = s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id")
    q_ids = q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id"))
    pairs = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet")).select(
        pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    p1 = np.load(wpath("feat", f"{split}_p1.npy"))
    p2 = np.load(wpath("feat", f"{split}_p2.npy"))
    dec = json.load(open(wpath("models", "decision.json")))
    allp = pairs.with_columns(pl.Series("p", final_prob(p1, p2)))
    ex = exclusive(allp, "p")
    if dec["rule"] == "thr":
        sel = threshold_select(ex, float(dec["param"]), "p")
    else:
        sel = expected_f_select(ex, "p", float(dec["param"]))
    sel = sel.select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32))
    log(f"decision rule {dec}: {sel.height} matched pairs, "
        f"{sel['e_row'].n_unique()} of {s1.height} S1 entities non-empty")
    with Timer("write TSVs"):
        m = _id_lists(s1_ids, sel, q_ids, "matched_entity_ids")
        c = _id_lists(s1_ids, pairs, q_ids, "candidate_entity_ids")
        m.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")
        c.write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log("wrote", out_dir / "matching_results.tsv", "and", out_dir / "candidate_pairs.tsv")
    return m, c


if __name__ == "__main__":
    write_outputs(sys.argv[1] if len(sys.argv) > 1 else "test")
