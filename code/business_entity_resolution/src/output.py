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
from decide import anchor_kappa, apply_kappa, exclusive, expected_f_select, final_prob, threshold_select


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
    p = p1 if dec.get("prob", "final") == "p1" else final_prob(p1, p2)
    allp = pairs.with_columns(pl.Series("p", p))
    ex = exclusive(allp, "p")
    if "anchor_target" in dec:
        # label-free prior anchoring (see decide.anchor_kappa) for countries WITHOUT training
        # labels (France): their expected matches per entity (3.48) exceed the density-augmented
        # validation value (3.41) that test US / India reproduce within +-0.01.
        e_country = np.load(wpath("feat", f"{split}_ctxkeys.npz"))["e_country"]
        trained = set(np.unique(np.load(wpath("feat", "train_ctxkeys.npz"))["e_country"]))
        pc = e_country[ex["e_row"].to_numpy()]
        pv = ex["p"].to_numpy().copy()
        for c in sorted(set(np.unique(e_country)) - trained):
            m = (pc == c) & (pv > 0)
            k = anchor_kappa(pv[m], int((e_country == c).sum()), dec["anchor_target"])
            pv[m] = apply_kappa(pv[m], k)
            log(f"anchoring {c}: expected matches per entity {pv[m].sum() / (e_country == c).sum():.4f}, "
                f"odds multiplier {k:.3f}")
        ex = ex.with_columns(pl.Series("p", pv))
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
