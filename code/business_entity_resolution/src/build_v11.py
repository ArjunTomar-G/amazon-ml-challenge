"""v11 submission = the scored v10c files + the two validated v11 changes (applied as a delta).

    python build_v11.py --v10c <dir with v10c matching_results.tsv / candidate_pairs.tsv> --out <dir>

Applying the changes to the scored v10c file isolates them from retraining noise (a full re-run of the
pipeline on another machine moves validation by a few 1e-5 by itself).
  1. France rule T3 (france.py): unlink France records whose name swaps one category word (club, ecole,
     comite ...; the vocabulary is derived label-free from the test universe, as in france.signatures) for
     another category word.  Label-free checks on the 786 v10c links it removes: singleton share 5.1 % (all
     France links 1.9 %), copy-twin rate 2.3 % (generator noise words 4-9 %) -> mostly look-alike businesses.
  2. Rescue v2 (rescue2.py, K = 10, records with and without an address): the best rescue pair of every
     US / India record that v10c left unlinked, if its probability >= 0.8.  Validation (fold V): +0.00013
     on top of the v10c rescue (0.99176 -> 0.99189).
  3. v12, --rank-t1 T (US / India): an S1 without any link takes its strongest record (the record's argmax,
     highest p among the records whose argmax it is) when T <= p < thr.  Under macro F0.5 a correct first
     link is worth 0.6-1.0 for that S1 and a wrong one costs 1.0 only when the S1 truly has no copy, so the
     break-even of a first link is far below 0.8.  Fold V: +0.00009 after both rescues (0.99189 -> 0.99197),
     +0.00006 on the density-augmented fold; not France (its 0.6-0.8 band is mostly look-alike businesses).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import france as FR
from blocking import load_universe
from common import Timer, log, wpath
from output import _id_lists


def explode(path, col):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (d.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
            .filter(pl.col(col) != "").rename({"source1_entity_id": "entity_id", col: "q_id"}))


def category_vocabulary():
    """France category words, derived as in france.signatures from every France record's best pair."""
    pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    g = FR.signatures(pairs, np.load(wpath("feat", "test_pfinal.npy")))
    t2pop = g.filter(pl.col("sig") & (pl.col("ndrop") == 1) & pl.col("aw_real") & ~pl.col("typo")
                     & ~pl.col("aw").is_in(list(FR.T2_KEEP)))
    vc = t2pop["aw"].value_counts()
    return set(vc.filter(pl.col("count") >= FR.CAT_MIN)["aw"].to_list())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10c", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--rank-t1", type=float, default=None, help="v12 rank rule: first-link threshold (0.65)")
    a = ap.parse_args()
    s1, q = load_universe("test", ["entity_id"])
    s1_ids = s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id")
    q_ids = q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id"))
    links = (explode(Path(a.v10c) / "matching_results.tsv", "matched_entity_ids")
             .join(s1_ids, on="entity_id").join(q_ids, on="q_id").select("e_row", "q_row"))
    cands = (explode(Path(a.v10c) / "candidate_pairs.tsv", "candidate_entity_ids")
             .join(s1_ids, on="entity_id").join(q_ids, on="q_id").select("e_row", "q_row"))
    log(f"v10c: {links.height} links, {cands.height} candidate pairs")
    # ---- 1. France rule T3 on the v10c links
    with Timer("France category vocabulary"):
        cat = category_vocabulary()
    log(f"category vocabulary: {len(cat)} words")
    g = FR.signatures(links.select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32)),
                      np.ones(links.height, np.float32))
    cs = [bool(D) and any(d in cat for d in D) and any(x in cat and not any(FR._typo_like(x, d) for d in D) for x in A)
          for A, D in zip(g["added"].to_list(), g["dropped"].to_list())]
    g = g.with_columns(pl.Series("catswap", cs))
    t3 = g.filter(pl.col("catswap") & (pl.col("shared") >= 1)
                  & ~(pl.col("added").list.contains("compagnie") & pl.col("e_cie"))).select(
        pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    links = links.join(t3, on=["q_row", "e_row"], how="anti")
    log(f"T3: {t3.height} France links removed")
    # ---- 2. rescue v2 for the records v10c left unlinked
    r = pl.read_parquet(wpath("rescue2", "test_rescue_pred.parquet")).select(
        pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32), "prob")
    best = r.sort("prob", descending=True).unique("q_row", keep="first").filter(pl.col("prob") >= a.thr)
    add = best.join(links.select("q_row"), on="q_row", how="anti").select("e_row", "q_row")
    log(f"rescue v2: {add.height} links added")
    links = pl.concat([links, add])
    cands = pl.concat([cands, r.select("e_row", "q_row")]).unique()
    # ---- 3. v12 rank rule (US / India): the strongest record of an S1 that has no link yet
    if a.rank_t1 is not None:
        pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
        p = np.load(wpath("feat", "test_pfinal.npy"))
        kt = np.load(wpath("feat", "test_ctxkeys.npz"))
        ok = np.isin(kt["q_country"][pairs["q_row"].to_numpy()], ["US", "India"])
        d = pairs.filter(pl.Series(ok)).with_columns(pl.Series("p", p[ok]))
        d = (d.sort(["q_row", "p"], descending=[False, True]).group_by("q_row", maintain_order=True)
              .agg(pl.col("e_row").first(), pl.col("p").first()))
        d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("e_row").alias("rk"))
        add3 = (d.filter((pl.col("rk") == 1) & (pl.col("p") >= a.rank_t1) & (pl.col("p") < a.thr))
                 .join(links.select("q_row"), on="q_row", how="anti")
                 .join(links.select("e_row").unique(), on="e_row", how="anti")
                 .select("e_row", "q_row"))
        log(f"rank rule (first link of an S1 at p >= {a.rank_t1}): {add3.height} links added")
        links = pl.concat([links, add3])
        cands = pl.concat([cands, add3]).unique()
    assert links["q_row"].n_unique() == links.height, "a record linked twice"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Timer("write TSVs"):
        _id_lists(s1_ids, links, q_ids, "matched_entity_ids").write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
        _id_lists(s1_ids, cands, q_ids, "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"wrote {out}: {links.height} links, {cands.height} candidate pairs")


if __name__ == "__main__":
    main()
