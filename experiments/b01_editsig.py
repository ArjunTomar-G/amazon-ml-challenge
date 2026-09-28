"""Edit signatures of one-token name differences: true-copy vs distractor rates on train (A/B + V),
stage-2-only precision (the France decision has no cross-encoder), and their volume among France links."""
import os
import sys
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, r"D:/amazon_last_ride/v11/code/business_entity_resolution/src")
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from blocking import load_universe
from common import wpath
from decide import final_prob

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)


def sig_of(a: str, b: str) -> str:
    """a = S1 core name, b = record core name."""
    ta, tb = a.split(), b.split()
    ca, cb = Counter(ta), Counter(tb)
    da, db = list((ca - cb).elements()), list((cb - ca).elements())
    if not da and not db:
        return "SAME" if ta == tb else "REORDER"
    if len(da) == 0 and len(db) == 1:
        return "INS"
    if len(da) == 1 and len(db) == 0:
        return "DEL"
    if len(da) != 1 or len(db) != 1:
        return "MULTI"
    x, y = da[0], db[0]
    brand = "B" if ta and ta[0] == x else "n"
    if Levenshtein.normalized_similarity(x, y) < 0.5 and x[:2] != y[:2]:
        return f"SWAP{brand}"
    ops = [o for o in Levenshtein.opcodes(x, y) if o.tag != "equal"]
    parts = []
    for o in ops:
        pos = "S" if o.src_start == 0 else ("E" if o.src_end == len(x) else "M")
        parts.append(f"{o.tag[:3]}:{x[o.src_start:o.src_end]}>{y[o.dest_start:o.dest_end]}@{pos}")
    return brand + "|" + "+".join(parts) if len(parts) <= 2 else f"{brand}|EDIT3+"


def pair_sigs(split, pairs):
    s1, q = load_universe(split, ["n_core", "country"])
    en = s1["n_core"].to_numpy()
    qn = q["n_core"].to_numpy()
    e, r = pairs["e_row"].to_numpy(), pairs["q_row"].to_numpy()
    return [sig_of(en[i] or "", qn[j] or "") for i, j in zip(e, r)], q["country"].to_numpy()


if __name__ == "__main__":
    # ---------------- train: all candidate pairs with stage-2 p >= 0.02 or true
    pr = pl.read_parquet(wpath("feat", "train_pairs.parquet")).select("q_row", "e_row", "fold", "y")
    pf = final_prob(np.load(wpath("feat", "train_p1.npy")), np.load(wpath("feat", "train_p2.npy")))
    keep = np.flatnonzero((pf >= 0.02) | (pr["y"].to_numpy() == 1))
    tr = pr[keep].with_columns(pl.Series("pf", pf[keep]))
    # record argmax flag (the decision is per record argmax)
    tr = tr.with_columns((pl.col("pf") >= pl.col("pf").max().over("q_row")).alias("am"))
    sigs, qc = pair_sigs("train", tr)
    tr = tr.with_columns(pl.Series("sig", sigs), pl.Series("country", qc[tr["q_row"].to_numpy()]))
    tr = tr.with_columns(pl.col("sig").str.replace(r"^[Bn]\|", "").alias("sig0"),
                         pl.col("sig").str.starts_with("B|").alias("brand"))
    tr.write_parquet(r"D:/er_work/an_train_sigs.parquet")
    print("train pairs analysed:", tr.height, " true:", tr["y"].sum())

    lk = tr.filter(pl.col("am") & (pl.col("pf") >= 0.8))
    agg = (lk.group_by("sig0").agg(pl.len().alias("links"), pl.col("y").mean().round(4).alias("prec_stage2"),
                                   pl.col("brand").mean().round(2).alias("brand_share"))
           .join(tr.group_by("sig0").agg(pl.len().alias("cands"), pl.col("y").mean().round(4).alias("true_rate_all")), on="sig0")
           .with_columns(((1 - pl.col("prec_stage2")) * pl.col("links")).round(0).alias("fp_links"))
           .sort("fp_links", descending=True))
    print("\n== stage-2-only links (record argmax, p >= 0.8) by edit signature, sorted by false links ==")
    print(agg.head(60))
    agg.write_parquet(r"D:/er_work/an_sig_agg.parquet")
