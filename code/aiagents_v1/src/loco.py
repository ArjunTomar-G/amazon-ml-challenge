"""Leave-one-country-out experiment (development tool, not part of the pipeline).

Simulates the France situation (a country with no training labels): train the
stage-1 matcher on one country only and measure macro-F0.5 on the held-out
entities of the other country, compared with a model that saw both.

    python loco.py <train_country> <eval_country>
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import log, wpath
from decide import exclusive, expected_f_select, macro_f05, threshold_select
from model import FOLD_A, FOLD_B, LGB_STAGE1, VAL_FOLD, query_hash


def main(train_c: str, eval_c: str):
    names = json.load(open(wpath("feat", "feature_names.json")))
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    X = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    ctry = keys["q_country"][pairs["q_row"].to_numpy()]
    fold = pairs["fold"].to_numpy()
    y = pairs["y"].to_numpy()
    q = pairs["q_row"].to_numpy()
    idx = np.flatnonzero(np.isin(fold, FOLD_A + FOLD_B) & (ctry == train_c))
    es = idx[query_hash(q[idx], 20) == 0]
    tr = np.setdiff1d(idx, es)
    params = dict(LGB_STAGE1, learning_rate=0.08)
    m = lgb.train(params, lgb.Dataset(np.asarray(X[tr]), y[tr], feature_name=names), 3000,
                  valid_sets=[lgb.Dataset(np.asarray(X[es]), y[es])],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    ev = np.flatnonzero((ctry == eval_c))
    p_loco = m.predict(np.asarray(X[ev]), num_threads=16)
    p_full = np.load(wpath("feat", "train_p1.npy"))[ev]
    e_fold = keys["e_fold"]
    e_ctry = keys["e_country"]
    ents = pl.DataFrame({"e_row": np.flatnonzero((e_fold == VAL_FOLD) & (e_ctry == eval_c))})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(ents["e_row"].implode()))
    base = pairs[ev].select("q_row", "e_row")
    isv = pl.Series(fold[ev] == VAL_FOLD)
    for tag, p in (("loco", p_loco), ("full", p_full)):
        ex = exclusive(base.with_columns(pl.Series("p", p)), "p").filter(isv)
        best = max([(macro_f05(threshold_select(ex, t), truth, ents), f"thr{t}") for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
                   + [(macro_f05(expected_f_select(ex, "p", 0.03), truth, ents), "ef")])
        log(f"{tag:5s} trained on {train_c if tag == 'loco' else 'all'} -> {eval_c} V: best F0.5 {best}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
