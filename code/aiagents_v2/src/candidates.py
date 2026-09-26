"""Stage 2b: learned candidate pruning -> final candidate set.

Retrieval (blocking.py) returns ~35 Source-1 candidates per S2/S3 record from
two channels.  A small LightGBM "pruner" trained on cheap features (Numba
token-alignment scores, hash equalities, retrieval scores/ranks) re-ranks them
and we keep the top M_KEEP per record whose pruner probability exceeds
MIN_PROB.  The kept set is the candidate set the matching model runs on and
is what candidate_pairs.tsv reports.
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import KEY_COLS, retrieve, true_pairs_rows
from common import Timer, log, wpath
from features import TOKFEAT_NAMES, Universe, token_feats

M_KEEP = 10
MIN_PROB = 1e-4
PRE_REL_ALL = float(os.environ.get("ER_PRE_REL_ALL", 0.25))
PRE_REL_ADDR = float(os.environ.get("ER_PRE_REL_ADDR", 0.4))
PRUNER_PATH = "pruner.txt"


def prefilter(c: pl.DataFrame) -> pl.DataFrame:
    """Drop candidates far below the record's best retrieval score in both
    channels (keeps 99.99% of retrievable true pairs, ~3x fewer rows)."""
    keep = ((pl.col("s_all") / pl.col("top_all")).fill_null(0.0) >= PRE_REL_ALL) |            ((pl.col("s_addr") / pl.col("top_addr")).fill_null(0.0) >= PRE_REL_ADDR)
    return c.filter(keep)


def cheap_features(U: Universe, c: pl.DataFrame) -> pl.DataFrame:
    pq = c["q_row"].to_numpy().astype(np.int64)
    pe = c["e_row"].to_numpy().astype(np.int64)
    F = {}
    ts = U.name_ts
    tf = token_feats(pq, pe, ts.q_ptr, ts.q_ids, ts.s1_ptr, ts.s1_ids, ts.voc_ptr, ts.voc_buf,
                     ts.idf, ts.ratio, ts.is_num, 0.75)
    for k in (0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13):
        F["cn_" + TOKFEAT_NAMES[k]] = tf[:, k]
    ts = U.addr_ts
    af = token_feats(pq, pe, ts.q_ptr, ts.q_ids, ts.s1_ptr, ts.s1_ids, ts.voc_ptr, ts.voc_buf,
                     ts.idf, ts.ratio, ts.is_num, 0.8)
    for k in (0, 1, 3, 4, 5, 6, 11, 12, 13):
        F["ca_" + TOKFEAT_NAMES[k]] = af[:, k]
    for col in ("a_hn", "a_street", "n_core", "n_glued", "a_tok", "n_legal", "a_state", "a_locs"):
        eq = U.hq[col][pq] == U.he[col][pe]
        emp = U.hq[col + "_empty"][pq] | U.he[col + "_empty"][pe]
        F["eq_" + col] = np.where(emp, np.nan, eq).astype(np.float32)
    for col in ("s_all", "r_all", "s_addr", "r_addr"):
        F[col] = c[col].cast(pl.Float32).to_numpy()
    F["rel_all"] = (c["s_all"] / c["top_all"]).cast(pl.Float32).to_numpy()
    F["rel_addr"] = (c["s_addr"] / c["top_addr"]).cast(pl.Float32).to_numpy()
    F["n_cand"] = c.select(pl.len().over("q_row")).to_series().cast(pl.Float32).to_numpy()
    F["q_src"] = U.q["src"].to_numpy()[pq].astype(np.float32)
    return pl.DataFrame(F)


def train_pruner(sample_frac: float = 0.04):
    U = Universe("train")
    frames = (U.s1.select(["row"] + KEY_COLS), U.q.select(["row"] + KEY_COLS))
    with Timer("retrieval on pruner sample"):
        c = retrieve("train", q_sample=sample_frac, seed=123, frames=frames, chunk_fn=prefilter)
    tp = true_pairs_rows().with_columns(pl.lit(1, pl.Int8).alias("y"))
    c = c.join(tp, on=["q_row", "e_row"], how="left").with_columns(pl.col("y").fill_null(0))
    with Timer("cheap features"):
        X = cheap_features(U, c)
    y = c["y"].to_numpy()
    # split by query for early stopping
    qh = (c["q_row"].to_numpy().astype(np.int64) * 2654435761) % 10
    tr, va = qh < 8, qh >= 8
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  num_threads=16, verbose=-1, max_bin=127)
    dtr = lgb.Dataset(X.to_numpy()[tr], y[tr], feature_name=X.columns, free_raw_data=True)
    dva = lgb.Dataset(X.to_numpy()[va], y[va], reference=dtr)
    with Timer("train pruner"):
        m = lgb.train(params, dtr, 600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)])
    m.save_model(str(wpath("models", PRUNER_PATH)))
    # recall of the pruned set on the validation queries
    p = m.predict(X.to_numpy()[va], num_threads=16)
    cv = c.filter(pl.Series(va)).with_columns(pl.Series("p", p))
    cv = cv.with_columns(pl.col("p").rank("ordinal", descending=True).over("q_row").alias("prank"))
    tpv = tp.join(cv.select("q_row").unique(), on="q_row")
    j = tpv.join(cv.select("q_row", "e_row", "p", "prank"), on=["q_row", "e_row"], how="left")
    tot = j.height
    for M in (1, 2, 3, 4, 5, 6, 8, 10):
        for thr in (0.0, 1e-4, 1e-3):
            kept = cv.filter((pl.col("prank") <= M) & (pl.col("p") >= thr))
            rec = float(((j["prank"] <= M) & (j["p"] >= thr)).sum()) / tot
            log(f"M={M:2d} thr={thr:g}: recall={rec:.5f}  cands/query={kept.height / cv['q_row'].n_unique():.2f}")
    imp = sorted(zip(m.feature_importance("gain"), X.columns), reverse=True)
    log("pruner top features:", [(n, round(g)) for g, n in imp[:15]])
    return m


def build_candidates(split: str, U: Universe | None = None):
    """Full retrieval + pruning for a universe -> WORK/cand/{split}_cands.parquet"""
    U = U or Universe(split)
    m = lgb.Booster(model_file=str(wpath("models", PRUNER_PATH)))
    frames = (U.s1.select(["row"] + KEY_COLS), U.q.select(["row"] + KEY_COLS))

    def prune(c):
        c = prefilter(c)
        X = cheap_features(U, c)
        p = m.predict(X.to_numpy(), num_threads=16)
        c = c.with_columns(pl.Series("p_prune", p.astype(np.float32)))
        c = c.with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("q_row")
                           .cast(pl.Int16).alias("prank"))
        return c.filter((pl.col("prank") <= M_KEEP) & (pl.col("p_prune") >= MIN_PROB))

    with Timer(f"candidates {split}"):
        c = retrieve(split, chunk_fn=prune, frames=frames)
    c = c.with_columns(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    c.write_parquet(wpath("cand", f"{split}_cands.parquet"))
    log(f"{split}: {c.height} candidate pairs for {c['q_row'].n_unique()} queries")
    if split == "train":
        tp = true_pairs_rows()
        j = tp.join(c.select("q_row", "e_row", "prank"), on=["q_row", "e_row"], how="left")
        log("train candidate recall (all true pairs):", float(j["prank"].is_not_null().mean()))
    return c


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "pruner":
        train_pruner()
    else:
        build_candidates(what)
