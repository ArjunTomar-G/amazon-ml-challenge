"""Stage 4: pairwise features for every candidate pair + cross-fitted stage-1 LightGBM.

Folds are assigned per Source-1 entity (hash of entity_id mod 5):
    fold 4       -> held-out validation (V); treated exactly like the test set
    folds {0,1}  -> A,  folds {2,3} -> B   (stage-1 cross-fitting)
Stage-1 models: M_A trained on the pairs of A, M_B on the pairs of B.
Out-of-fold stage-1 probability p1:
    pairs of A <- M_B,  pairs of B <- M_A,  pairs of V and of test <- mean(M_A, M_B)

    python model.py feats <split>    features for every candidate pair -> memmap
    python model.py stage1           train M_A / M_B, write p1 for train and test
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import Timer, log, wpath

N_FOLDS = 5
VAL_FOLD = 4
FOLD_A = (0, 1)
FOLD_B = (2, 3)
CHUNK_PAIRS = 1_500_000
N_TOP = 60               # stage-1 features carried to stage 2

LGB_STAGE1 = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  max_bin=255, num_threads=16, verbose=-1)
LGB_STAGE2 = dict(objective="binary", learning_rate=0.03, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  max_bin=255, num_threads=16, verbose=-1)


def entity_folds(s1_entity_ids: pl.Series) -> np.ndarray:
    return (s1_entity_ids.hash(seed=2026).to_numpy() % N_FOLDS).astype(np.int8)


def query_hash(q_rows: np.ndarray, mod: int, mult: int = 2654435761) -> np.ndarray:
    return (q_rows.astype(np.uint64) * np.uint64(mult)) % np.uint64(mod)


def load_cands(split: str) -> pl.DataFrame:
    return pl.read_parquet(wpath("cand", f"{split}_cands.parquet")).sort(["q_row", "e_row"])


def chunk_bounds(q_rows: np.ndarray, size: int):
    """Contiguous [a, b) slices over rows sorted by q_row, never splitting a query."""
    n = len(q_rows)
    a = 0
    while a < n:
        b = min(n, a + size)
        while b < n and q_rows[b] == q_rows[b - 1]:
            b += 1
        yield a, b
        a = b


def label_pairs(c: pl.DataFrame) -> np.ndarray:
    tp = true_pairs_rows().with_columns(pl.lit(1, pl.Int8).alias("y"))
    y = c.select("q_row", "e_row").join(tp, on=["q_row", "e_row"], how="left")["y"].fill_null(0)
    return y.to_numpy().astype(np.int8)


def step_feats(split: str):
    from features import Universe, pair_features
    U = Universe(split)
    c = load_cands(split)
    q = c["q_row"].to_numpy()
    if split == "train":
        fold = entity_folds(U.s1["entity_id"])[c["e_row"].to_numpy()]
        y = label_pairs(c)
    else:
        fold = np.full(c.height, VAL_FOLD, np.int8)
        y = np.zeros(c.height, np.int8)
    c.select("q_row", "e_row", "p_prune").with_columns(pl.Series("fold", fold), pl.Series("y", y)) \
        .write_parquet(wpath("feat", f"{split}_pairs.parquet"))
    X, names = None, None
    for a, b in chunk_bounds(q, CHUNK_PAIRS):
        with Timer(f"[{split}] features rows {a}..{b} of {c.height}"):
            F = pair_features(U, c.slice(a, b - a))
        if X is None:
            names = F.columns
            X = np.lib.format.open_memmap(wpath("feat", f"{split}_X.npy"), mode="w+",
                                          dtype=np.float32, shape=(c.height, len(names)))
        X[a:b] = F.to_numpy().astype(np.float32)
    X.flush()
    json.dump(names, open(wpath("feat", "feature_names.json"), "w"))
    # hash arrays for the context features (so stage 2 does not need the Universe)
    np.savez(wpath("feat", f"{split}_ctxkeys.npz"),
             **{f"q_{k}": U.hq[k] for k in ("a_hn", "a_street", "n_core", "a_tok")},
             q_hn_empty=U.hq["a_hn_empty"], e_a_hn=U.he["a_hn"], q_src=U.q["src"].to_numpy(),
             q_country=U.q["country"].to_numpy().astype("U"),
             e_country=U.s1["country"].to_numpy().astype("U"),
             e_fold=entity_folds(U.s1["entity_id"]))
    if split == "train":
        log("positives among candidates:", int(y.sum()), "of", len(y))


def step_stage1():
    names = json.load(open(wpath("feat", "feature_names.json")))
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    X = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
    fold = pairs["fold"].to_numpy()
    y = pairs["y"].to_numpy()
    q = pairs["q_row"].to_numpy()
    models = {}
    imp = np.zeros(len(names))
    for tag, tr_f in (("A", FOLD_A), ("B", FOLD_B)):
        idx = np.flatnonzero(np.isin(fold, tr_f))
        es = idx[query_hash(q[idx], 20) == 0]            # 5% of queries for early stopping
        tr = np.setdiff1d(idx, es)
        with Timer(f"stage-1 model {tag}: {len(tr)} train rows, {int(y[tr].sum())} positives"):
            dtr = lgb.Dataset(np.asarray(X[tr]), y[tr], feature_name=names, free_raw_data=True)
            des = lgb.Dataset(np.asarray(X[es]), y[es], reference=dtr)
            m = lgb.train(LGB_STAGE1, dtr, 4000, valid_sets=[des],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
        m.save_model(str(wpath("models", f"stage1_{tag}.txt")))
        models[tag] = m
        imp += m.feature_importance("gain")
        del dtr, des
    order = np.argsort(-imp)
    json.dump([names[i] for i in order[:N_TOP]], open(wpath("feat", "top_features.json"), "w"))
    log("top stage-1 features:", [(names[i], round(float(imp[i]))) for i in order[:40]])
    predict_p1(models)


def predict_p1(models=None, splits=("train", "test")):
    """Out-of-fold (train) / fold-averaged (validation, test) stage-1 probabilities."""
    if models is None:
        models = {t: lgb.Booster(model_file=str(wpath("models", f"stage1_{t}.txt"))) for t in "AB"}
    for split in splits:
        if not wpath("feat", f"{split}_X.npy").exists():
            log(f"skip p1 for {split}: no features yet")
            continue
        pp = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet"))
        Xs = np.load(wpath("feat", f"{split}_X.npy"), mmap_mode="r")
        f = pp["fold"].to_numpy()
        P = np.zeros(pp.height, np.float32)
        for a in range(0, pp.height, 3_000_000):
            b = min(pp.height, a + 3_000_000)
            Xa = np.asarray(Xs[a:b])
            pa_ = models["A"].predict(Xa, num_threads=16)
            pb_ = models["B"].predict(Xa, num_threads=16)
            fa = f[a:b]
            P[a:b] = np.where(np.isin(fa, FOLD_A), pb_,
                              np.where(np.isin(fa, FOLD_B), pa_, 0.5 * (pa_ + pb_)))
        np.save(wpath("feat", f"{split}_p1.npy"), P)
        log(f"{split}: wrote p1 for {pp.height} pairs")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "feats":
        step_feats(sys.argv[2])
    elif step == "stage1":
        step_stage1()
    elif step == "p1":
        predict_p1(splits=tuple(sys.argv[2:]) or ("train", "test"))
    else:
        raise SystemExit(f"unknown step {step}")
