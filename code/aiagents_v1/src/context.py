"""Stage-2 context features computed from stage-1 pair probabilities.

For a candidate pair (q, e) with stage-1 probability p:

query side (competition between Source-1 entities for the same record)
    q_pmax, q_psecond, p_minus_qmax_other, q_rank, q_n, q_psum
entity side (all records competing to be matched to the same S1 entity)
    e_n, e_psum, e_pmax_other, e_rank, e_n50
consensus (do the entity's *other* likely matches agree with this record?)
    support of q's house number / street / name / address among e's other
    candidates weighted by p, and support of e's own house number.  A sibling
    business (+3 house number, extra word) is isolated; a noisy true record
    usually agrees with the entity's other true records.
mutual best
    e is q's argmax and q is within e's top ranks.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def context_features(c: pl.DataFrame, keys, pcol: str = "p1") -> pl.DataFrame:
    """c: q_row, e_row, <pcol> for ALL candidate pairs (of one country) of a
    universe; keys: mapping with the q_*/e_* hash arrays saved by model.oof.
    Returns the context columns (same row order as c)."""
    pq = c["q_row"].to_numpy().astype(np.int64)
    pe = c["e_row"].to_numpy().astype(np.int64)
    d = c.select("q_row", "e_row", pl.col(pcol).alias("p")).with_columns(
        pl.Series("k_hn", keys["q_a_hn"][pq]),
        pl.Series("k_st", keys["q_a_street"][pq]),
        pl.Series("k_nm", keys["q_n_core"][pq]),
        pl.Series("k_ad", keys["q_a_tok"][pq]),
        pl.Series("hn_empty", keys["q_hn_empty"][pq]),
        pl.Series("e_hn", keys["e_a_hn"][pe]),
        pl.Series("q_src", keys["q_src"][pq]),
    )
    d = d.with_row_index("_i")
    # ---- query side ----
    d = d.with_columns(
        pl.col("p").max().over("q_row").alias("q_pmax"),
        pl.len().over("q_row").cast(pl.Float32).alias("q_n"),
        pl.col("p").sum().over("q_row").alias("q_psum"),
        pl.col("p").rank("ordinal", descending=True).over("q_row").cast(pl.Float32).alias("q_rank"),
    )
    second = (d.select("q_row", "p").sort(["q_row", "p"], descending=[False, True])
              .group_by("q_row", maintain_order=True).agg(pl.col("p").slice(1, 1).first().alias("q_psecond")))
    d = d.join(second, on="q_row", how="left").with_columns(pl.col("q_psecond").fill_null(0.0))
    d = d.with_columns(
        pl.when(pl.col("p") >= pl.col("q_pmax"))
        .then(pl.col("p") - pl.col("q_psecond"))
        .otherwise(pl.col("p") - pl.col("q_pmax")).alias("p_minus_qmax_other"),
    )
    # ---- entity side ----
    d = d.with_columns(
        pl.len().over("e_row").cast(pl.Float32).alias("e_n"),
        pl.col("p").sum().over("e_row").alias("e_psum"),
        pl.col("p").rank("ordinal", descending=True).over("e_row").cast(pl.Float32).alias("e_rank"),
        (pl.col("p") > 0.5).sum().over("e_row").cast(pl.Float32).alias("e_n50"),
        pl.col("p").max().over("e_row").alias("e_pmax"),
    )
    e_second = (d.select("e_row", "p").sort(["e_row", "p"], descending=[False, True])
                .group_by("e_row", maintain_order=True).agg(pl.col("p").slice(1, 1).first().alias("e_psecond")))
    d = d.join(e_second, on="e_row", how="left").with_columns(pl.col("e_psecond").fill_null(0.0))
    d = d.with_columns(
        pl.when(pl.col("p") >= pl.col("e_pmax")).then(pl.col("e_psecond"))
        .otherwise(pl.col("e_pmax")).alias("e_pmax_other"),
        (pl.col("e_psum") - pl.col("p")).alias("e_psum_other"),
    )
    # ---- consensus within the entity's candidate set (excluding the pair itself) ----
    for k, name in (("k_hn", "hn"), ("k_st", "st"), ("k_nm", "nm"), ("k_ad", "ad")):
        d = d.with_columns(
            (pl.col("p").sum().over(["e_row", k]) - pl.col("p")).alias(f"sup_{name}"))
        d = d.with_columns(
            (pl.col(f"sup_{name}") / (pl.col("e_psum_other") + 1e-3)).alias(f"supf_{name}"))
    d = d.with_columns(
        pl.when(pl.col("hn_empty")).then(None).otherwise(pl.col("sup_hn")).alias("sup_hn"),
        pl.when(pl.col("hn_empty")).then(None).otherwise(pl.col("supf_hn")).alias("supf_hn"),
    )
    # support for the entity's own house number among its likely matches
    own = (d.filter(pl.col("k_hn") == pl.col("e_hn"))
           .group_by("e_row").agg(pl.col("p").sum().alias("e_own_hn_mass")))
    d = d.join(own, on="e_row", how="left").with_columns(pl.col("e_own_hn_mass").fill_null(0.0))
    d = d.with_columns(
        (pl.col("e_own_hn_mass") - pl.when(pl.col("k_hn") == pl.col("e_hn")).then(pl.col("p")).otherwise(0.0))
        .alias("e_own_hn_other"))
    # same-source duplicates among the entity's candidates
    d = d.with_columns(
        (pl.col("p").sum().over(["e_row", "q_src"]) - pl.col("p")).alias("e_psum_same_src"))
    # mutual best
    d = d.with_columns(
        ((pl.col("p") >= pl.col("q_pmax")) & (pl.col("e_rank") <= 3)).cast(pl.Float32).alias("mutual"))
    d = d.sort("_i")
    ctx_cols = ["q_pmax", "q_psecond", "p_minus_qmax_other", "q_rank", "q_n", "q_psum",
                "e_n", "e_psum", "e_pmax_other", "e_rank", "e_n50", "e_psum_other",
                "sup_hn", "supf_hn", "sup_st", "supf_st", "sup_nm", "supf_nm", "sup_ad", "supf_ad",
                "e_own_hn_other", "e_psum_same_src", "mutual"]
    return d.select([pl.col(x).cast(pl.Float32) for x in ctx_cols])


CTX_COLS = ["q_pmax", "q_psecond", "p_minus_qmax_other", "q_rank", "q_n", "q_psum",
            "e_n", "e_psum", "e_pmax_other", "e_rank", "e_n50", "e_psum_other",
            "sup_hn", "supf_hn", "sup_st", "supf_st", "sup_nm", "supf_nm", "sup_ad", "supf_ad",
            "e_own_hn_other", "e_psum_same_src", "mutual"]
