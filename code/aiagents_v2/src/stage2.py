"""Stage 5: context features, stage-2 model, validation and decision-rule choice.

    python stage2.py ctx <split>      context features for every pair -> memmap
    python stage2.py train            stage-2 model on A u B pairs; validation on V
    python stage2.py predict <split>  stage-2 probabilities for every pair
    python stage2.py validate         re-run the validation grid only
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import Timer, log, wpath
from context import CTX_COLS, context_features
from decide import exclusive, expected_f_select, final_prob, macro_f05, threshold_select
from model import FOLD_A, FOLD_B, LGB_STAGE2, VAL_FOLD, query_hash


def load_keys(split):
    return dict(np.load(wpath("feat", f"{split}_ctxkeys.npz"), allow_pickle=False))


def step_ctx(split: str):
    pairs = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet"))
    p1 = np.load(wpath("feat", f"{split}_p1.npy"))
    keys = load_keys(split)
    ctry = keys["q_country"][pairs["q_row"].to_numpy()]
    C = np.zeros((pairs.height, len(CTX_COLS)), np.float32)
    for country in np.unique(ctry):
        idx = np.flatnonzero(ctry == country)
        with Timer(f"[{split}] context {country}: {len(idx)} pairs"):
            c = pairs[idx].select("q_row", "e_row").with_columns(pl.Series("p1", p1[idx]))
            C[idx] = context_features(c, keys, "p1").to_numpy()
    np.save(wpath("feat", f"{split}_ctx.npy"), C)


def stage2_names():
    return json.load(open(wpath("feat", "top_features.json"))) + ["p1"] + CTX_COLS


def stage2_matrix(split: str, rows: np.ndarray):
    names = json.load(open(wpath("feat", "feature_names.json")))
    top = json.load(open(wpath("feat", "top_features.json")))
    cols = [names.index(t) for t in top]
    X = np.load(wpath("feat", f"{split}_X.npy"), mmap_mode="r")
    C = np.load(wpath("feat", f"{split}_ctx.npy"), mmap_mode="r")
    p1 = np.load(wpath("feat", f"{split}_p1.npy"), mmap_mode="r")
    out = np.empty((len(rows), len(cols) + 1 + C.shape[1]), np.float32)
    step = 2_000_000
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        out[a:a + len(r), :len(cols)] = np.asarray(X[r])[:, cols]
        out[a:a + len(r), len(cols)] = p1[r]
        out[a:a + len(r), len(cols) + 1:] = C[r]
    return out


def evaluate(ex: pl.DataFrame, entities: pl.DataFrame, truth: pl.DataFrame, pcol="p"):
    """Grid over decision rules on (already exclusive) validation pairs."""
    res = {}
    for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        res[("thr", t)] = macro_f05(threshold_select(ex, t, pcol), truth, entities)
    for lam in (0.0, 0.01, 0.03, 0.06, 0.1):
        res[("ef", lam)] = macro_f05(expected_f_select(ex, pcol, lam), truth, entities)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:10]:
        log(f"  {k[0]:4s} {k[1]:<5}: F0.5 = {v:.5f}")
    return max(res.items(), key=lambda kv: kv[1])


def step_train():
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    fold = pairs["fold"].to_numpy()
    q = pairs["q_row"].to_numpy()
    y = pairs["y"].to_numpy()
    names = stage2_names()
    idx = np.flatnonzero(np.isin(fold, FOLD_A + FOLD_B))
    es = idx[query_hash(q[idx], 20, 40503) == 0]
    fit = np.setdiff1d(idx, es)
    with Timer(f"stage-2 fit on {len(fit)} rows"):
        dtr = lgb.Dataset(stage2_matrix("train", fit), y[fit], feature_name=names, free_raw_data=True)
        des = lgb.Dataset(stage2_matrix("train", es), y[es], reference=dtr)
        m = lgb.train(LGB_STAGE2, dtr, 5000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
    m.save_model(str(wpath("models", "stage2.txt")))
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)
    log("stage-2 top features:", [(n, round(float(g))) for g, n in imp[:30]])
    del dtr, des
    step_predict("train")
    validate()


def validate():
    """Exclusivity over ALL pairs (as on test), then score the fold-V entities."""
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    p1 = np.load(wpath("feat", "train_p1.npy"))
    p2 = np.load(wpath("feat", "train_p2.npy"))
    keys = load_keys("train")
    e_fold = keys["e_fold"]
    entities = pl.DataFrame({"e_row": np.flatnonzero(e_fold == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(entities["e_row"].implode()))
    isv = pl.Series(pairs["fold"].to_numpy() == VAL_FOLD)
    base = pairs.select("q_row", "e_row")
    ex1 = exclusive(base.with_columns(pl.Series("p", p1)), "p").filter(isv)
    ex2 = exclusive(base.with_columns(pl.Series("p", final_prob(p1, p2))), "p").filter(isv)
    ex2.write_parquet(wpath("feat", "val_pred.parquet"))
    rec = truth.join(ex2.select("q_row", "e_row"), on=["q_row", "e_row"]).height / max(1, truth.height)
    log(f"validation: {entities.height} S1 entities, {truth.height} true pairs, "
        f"{ex2.height} candidate pairs, candidate recall {rec:.5f}")
    log("stage-1 p1 decisions:")
    best1 = evaluate(ex1, entities, truth, "p")
    log("stage-2 p2 decisions:")
    best2 = evaluate(ex2, entities, truth, "p")
    log("BEST stage-1:", best1, " BEST stage-2:", best2)
    json.dump({"rule": best2[0][0], "param": best2[0][1], "f05": best2[1]},
              open(wpath("models", "decision.json"), "w"))


def step_predict(split: str):
    pairs = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet"))
    m = lgb.Booster(model_file=str(wpath("models", "stage2.txt")))
    P = np.zeros(pairs.height, np.float32)
    step = 3_000_000
    for a in range(0, pairs.height, step):
        rows = np.arange(a, min(pairs.height, a + step))
        P[rows] = m.predict(stage2_matrix(split, rows), num_threads=16)
    np.save(wpath("feat", f"{split}_p2.npy"), P)


if __name__ == "__main__":
    s = sys.argv[1]
    if s == "ctx":
        step_ctx(sys.argv[2])
    elif s == "train":
        step_train()
    elif s == "predict":
        step_predict(sys.argv[2])
    elif s == "validate":
        validate()
