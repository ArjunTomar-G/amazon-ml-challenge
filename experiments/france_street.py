"""France street fix: re-score France test pairs after the prefix-street normalisation fix (normalize.py).

    python france_street.py

French addresses put the street type first ("rue thiers").  The normaliser only recognised a street without a
house number when the type came last (US order), so 99.7 % of the France records with an address but no house
number (3.7 % of France records) had no street, and their street / locality similarity features were missing.

This step (1) re-normalises the France addresses without a house number with the fixed normaliser and writes
<work>/norm_fr/test_source{1,2,3}.parquet (France rows patched, everything else unchanged), (2) recomputes the
six pair features that depend on the street / locality (rf_st_ratio, rf_st_tset, rf_st_jw, rf_loc_tset,
rf_loc_partial, u_s1_same_hnst) for the France test pairs, (3) re-runs stage 1 (both fold models), the stage-2
context features and stage 2 for the France pairs, and writes feat/test_p1_fr.npy / test_p2_fr.npy /
test_pfinal_fr.npy (France pairs replaced, all other pairs identical to the originals).
Candidates (blocking) are not recomputed.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import Timer, log, wpath
from context import context_features
from decide import final_prob
from model import used_features
from normalize import normalize_address
from stage2 import CTX_IDX, load_keys
from translit import get_dict

COUNTRY = "France"
FEATS = ["rf_st_ratio", "rf_st_tset", "rf_st_jw", "rf_loc_tset", "rf_loc_partial", "u_s1_same_hnst"]


def patch_norm():
    """France rows without a house number re-normalised; returns the patched (s1, q) tables with row ids."""
    tdict = get_dict()
    out = {}
    for src in ("source1", "source2", "source3"):
        n = pl.read_parquet(wpath("norm", f"test_{src}.parquet"))
        raw = pl.read_parquet(wpath("raw", f"test_{src}.parquet"), columns=["business_address", "country"])
        idx = np.flatnonzero(((n["country"] == COUNTRY) & (n["a_hn"] == "") & (n["a_tok"] != "")).to_numpy())
        with Timer(f"re-normalise {len(idx)} {src} France addresses without a house number"):
            addrs = raw["business_address"].to_list()
            res = [normalize_address(addrs[i], COUNTRY, tdict) for i in idx]
        st = n["a_street"].to_list()
        lo = n["a_loc"].to_list()
        changed = 0
        for k, i in enumerate(idx):
            if res[k]["a_street"] != st[i] or res[k]["a_loc"] != lo[i]:
                changed += 1
            st[i], lo[i] = res[k]["a_street"], res[k]["a_loc"]
        n = n.with_columns(pl.Series("a_street", st), pl.Series("a_loc", lo))
        n.write_parquet(wpath("norm_fr", f"test_{src}.parquet"))
        log(f"{src}: {changed} France records changed (street / locality)")
        out[src] = n
    return out


def main():
    norm = patch_norm()
    s1 = norm["source1"].with_row_index("e_row")
    q = pl.concat([norm["source2"], norm["source3"]]).with_row_index("q_row")
    pr = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select("q_row", "e_row")
    keys = load_keys("test")
    fr = np.flatnonzero(keys["q_country"][pr["q_row"].to_numpy()] == COUNTRY)
    log(f"France pairs: {len(fr)}")
    # --- recompute the street / locality features of the France pairs
    s1 = s1.with_columns(pl.len().over(["country", "a_hn", "a_street"]).alias("u_s1_same_hnst"),
                         pl.col("a_loc").str.replace_all(r" \| ", " ").alias("a_locs"))
    q = q.with_columns(pl.col("a_loc").str.replace_all(r" \| ", " ").alias("a_locs"))
    d = pr[fr].with_row_index("i")
    d = (d.join(q.select(pl.col("q_row").cast(d["q_row"].dtype), pl.col("a_street").alias("qs"), pl.col("a_locs").alias("ql")), on="q_row", how="left")
         .join(s1.select(pl.col("e_row").cast(d["e_row"].dtype), pl.col("a_street").alias("es"), pl.col("a_locs").alias("el"), "u_s1_same_hnst"), on="e_row", how="left")
         .sort("i"))
    qs, es, ql, el = (d[c].fill_null("").to_list() for c in ("qs", "es", "ql", "el"))
    rf = lambda sc, a, b: process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)
    new = {"rf_st_ratio": rf(fuzz.ratio, qs, es), "rf_st_tset": rf(fuzz.token_set_ratio, qs, es),
           "rf_st_jw": rf(JaroWinkler.normalized_similarity, qs, es),
           "rf_loc_tset": rf(fuzz.token_set_ratio, ql, el), "rf_loc_partial": rf(fuzz.partial_ratio, ql, el)}
    m_st = np.array([not a or not b for a, b in zip(qs, es)])
    m_lo = np.array([not a or not b for a, b in zip(ql, el)])
    for k in ("rf_st_ratio", "rf_st_tset", "rf_st_jw"):
        new[k][m_st] = np.nan
    for k in ("rf_loc_tset", "rf_loc_partial"):
        new[k][m_lo] = np.nan
    new["u_s1_same_hnst"] = np.minimum(d["u_s1_same_hnst"].to_numpy(), 3).astype(np.float32)
    names = json.load(open(wpath("feat", "feature_names.json")))
    X = np.asarray(np.load(wpath("feat", "test_X.npy"), mmap_mode="r")[fr])
    for k in FEATS:
        j = names.index(k)
        diff = ~((X[:, j] == new[k]) | (np.isnan(X[:, j]) & np.isnan(new[k])))
        log(f"  {k}: {diff.sum()} of {len(fr)} France pairs changed")
        X[:, j] = new[k]
    # --- stage 1 (mean of the fold models, as for every test pair)
    used = used_features(names)
    ci = np.array([names.index(n) for n in used])
    ms = [lgb.Booster(model_file=str(wpath("models", f"stage1_{t}.txt"))) for t in "AB"]
    with Timer("stage 1 on the France pairs"):
        p1f = 0.5 * (ms[0].predict(X[:, ci], num_threads=16) + ms[1].predict(X[:, ci], num_threads=16))
    p1 = np.load(wpath("feat", "test_p1.npy")).copy()
    log(f"France p1: mean {p1[fr].mean():.4f} -> {p1f.mean():.4f}")
    p1[fr] = p1f
    # --- context features (computed per country; France only) and stage 2
    c = pr[fr].select("q_row", "e_row").with_columns(pl.Series("p1", p1f))
    C = context_features(c, keys, "p1").to_numpy()
    top = json.load(open(wpath("feat", "top_features.json")))
    cols = [names.index(t) for t in top]
    X2 = np.column_stack([X[:, cols], p1f[:, None], C[:, CTX_IDX]]).astype(np.float32)
    m2 = lgb.Booster(model_file=str(wpath("models", "stage2.txt")))
    p2f = m2.predict(X2, num_threads=16)
    p2 = np.load(wpath("feat", "test_p2.npy")).copy()
    log(f"France p2: mean {p2[fr].mean():.4f} -> {p2f.mean():.4f}")
    p2[fr] = p2f
    np.save(wpath("feat", "test_p1_fr.npy"), p1.astype(np.float32))
    np.save(wpath("feat", "test_p2_fr.npy"), p2.astype(np.float32))
    ctx = np.load(wpath("feat", "test_ctx.npy"))
    ctx[fr] = C.astype(np.float32)
    np.save(wpath("feat", "test_ctx_fr.npy"), ctx)
    del ctx
    np.savez(wpath("feat", "test_fr_patch.npz"), rows=fr.astype(np.int64), **{k: new[k] for k in FEATS})
    pfin = np.load(wpath("feat", "test_pfinal.npy")).copy()
    pfin[fr] = final_prob(p1f.astype(np.float32), p2f.astype(np.float32))
    np.save(wpath("feat", "test_pfinal_fr.npy"), pfin.astype(np.float32))
    log("wrote feat/test_pfinal_fr.npy (France pairs re-scored)")


if __name__ == "__main__":
    main()
