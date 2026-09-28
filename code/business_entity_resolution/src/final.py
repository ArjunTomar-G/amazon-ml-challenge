"""Final decisions -> matching_results.tsv / candidate_pairs.tsv.

    python final.py --out <dir> [--prob stack|stack_fr|v4] [--rule p|pa] [--thr 0.8] [--thr-k0 T]
                    [--france A,A0,B,C,D,T2,T3|none] [--t2-max-p 2.0] [--rescue [dir1,dir2]]

prob   stack: stage-2 probabilities with the US / India uncertain band re-scored by the
       cross-encoder stacker (feat/test_pfinal.npy); stack_fr: the same with the France band
       re-scored too (feat/test_fr_pstack.npy, france_stack.py); v4: stage-2 probabilities only
rule   exclusivity (one S1 per record, its argmax), then link if p >= thr
       (pa: p / max(1, sum of the record's p) >= thr)
thr-k0 optional stricter threshold for links whose S1 has no other confident (p >= 0.9) record
france link-level France rules (france.py)
rescue add each unlinked record's best rescue pair (rescue.py / rescue2.py) with probability >= thr;
       a comma-separated list of rescue folders (default: rescue) - the best pair over all of them
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from blocking import load_universe
from common import Timer, log, wpath
from decide import final_prob
from output import _id_lists


def decide(pairs: pl.DataFrame, p: np.ndarray, rule: str, thr: float, thr_k0: float | None = None) -> pl.DataFrame:
    d = pairs.with_columns(pl.Series("p", p))
    d = d.with_columns(pl.col("p").max().over("q_row").alias("pm"), pl.col("p").sum().over("q_row").alias("ps"))
    d = d.filter(pl.col("p") >= pl.col("pm"))
    if rule == "pa":
        d = d.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("ps"))).alias("p"))
    # ties: keep one S1 per record
    d = d.unique("q_row", keep="first", maintain_order=True)
    if thr_k0 is not None:
        # confident copies of the same S1 (other records): an S1 without any is more likely a singleton
        d = d.with_columns(((pl.col("p") >= 0.9).sum().over("e_row") - (pl.col("p") >= 0.9).cast(pl.Int64)).alias("k"))
        return d.filter(pl.when(pl.col("k") == 0).then(pl.col("p") >= thr_k0).otherwise(pl.col("p") >= thr)).select("e_row", "q_row")
    return d.filter(pl.col("p") >= thr).select("e_row", "q_row")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prob", default="stack", choices=["stack", "stack_fr", "v4"])
    ap.add_argument("--rule", default="p", choices=["p", "pa"])
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--thr-k0", type=float, default=None)
    ap.add_argument("--france", default="none")
    ap.add_argument("--t2-max-p", type=float, default=2.0)
    ap.add_argument("--rescue", nargs="?", const="rescue", default=None)
    ap.add_argument("--pfile", default=None, help="probability vector under feat/ (overrides --prob), e.g. "
                    "test_fr_pstack_fr.npy; with the France street fix also set ER_FR_NORM=norm_fr")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1, q = load_universe("test", ["entity_id"])
    s1_ids = s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id")
    q_ids = q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id"))
    pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(
        pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    if a.pfile:
        p = np.load(wpath("feat", a.pfile))
    elif a.prob == "stack":
        p = np.load(wpath("feat", "test_pfinal.npy"))
    elif a.prob == "stack_fr":
        p = np.load(wpath("feat", "test_fr_pstack.npy"))
    else:
        p = final_prob(np.load(wpath("feat", "test_p1.npy")), np.load(wpath("feat", "test_p2.npy")))
    if a.france != "none":
        from france import adjust
        p, _ = adjust(pairs, p, a.thr, tuple(a.france.split(",")), t2_max_p=a.t2_max_p)
    sel = decide(pairs, p, a.rule, a.thr, a.thr_k0)
    cand = pairs
    if a.rescue:
        def load(spec):          # <rescue folder>[:noaddr]
            d, _, flt = spec.partition(":")
            x = pl.read_parquet(wpath(d, "test_rescue_pred.parquet"))
            if flt == "noaddr":
                x = x.filter(pl.col("q_no_addr") == 1)
            return x.select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32), "prob")
        rs = [load(r) for r in a.rescue.split(",")]
        r = pl.concat(rs).sort("prob", descending=True).unique(["q_row", "e_row"], keep="first")
        cand = pl.concat([pairs, r.select("q_row", "e_row")]).unique()
        linked = sel.select(pl.col("q_row").cast(pl.Int32))
        add = (r.filter(pl.col("prob") >= a.thr).sort("prob", descending=True).unique("q_row", keep="first")
               .join(linked, on="q_row", how="anti").select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32)))
        log(f"rescue ({a.rescue}): {add.height} links added")
        sel = pl.concat([sel.select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32)), add])
    sel = sel.select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32))
    log(f"{a}: {sel.height} links, {sel['e_row'].n_unique()} of {s1.height} S1 non-empty")
    with Timer("write TSVs"):
        m = _id_lists(s1_ids, sel, q_ids, "matched_entity_ids")
        c = _id_lists(s1_ids, cand, q_ids, "candidate_entity_ids")
        m.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")
        c.write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log("wrote", out_dir)


if __name__ == "__main__":
    main()
