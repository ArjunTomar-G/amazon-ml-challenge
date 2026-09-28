"""Learning curve of stage 1: one model on folds A+B (80 % of train S1) vs the current A / B models (40 % each,
averaged on V).  Metrics on fold-V pairs only: AUC, logloss, stage-1-only macro F0.5 (argmax over V pairs)."""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, r"D:/amazon_last_ride/v12/code/business_entity_resolution/src")
os.environ.setdefault("ER_WORK_DIR", r"D:/er_work")
from common import Timer, log, wpath
from blocking import true_pairs_rows
from decide import macro_f05
from model import FOLD_A, FOLD_B, LGB_STAGE1, VAL_FOLD, query_hash, take_rows, used_features

all_names = json.load(open(wpath("feat", "feature_names.json")))
names = used_features(all_names)
ci = np.array([all_names.index(n) for n in names])
pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
X = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
fold, y, q = pairs["fold"].to_numpy(), pairs["y"].to_numpy(), pairs["q_row"].to_numpy()
idx = np.flatnonzero(np.isin(fold, FOLD_A + FOLD_B))
es = idx[query_hash(q[idx], 20) == 0]
tr = np.setdiff1d(idx, es)
with Timer(f"stage-1 A+B: {len(tr)} train rows"):
    dtr = lgb.Dataset(take_rows(X, tr, ci), y[tr], feature_name=names, free_raw_data=True)
    des = lgb.Dataset(take_rows(X, es, ci), y[es], reference=dtr)
    m = lgb.train(LGB_STAGE1, dtr, 4000, valid_sets=[des], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
m.save_model(str(wpath("models", "stage1_AB_lc.txt")))
v = np.flatnonzero(fold == VAL_FOLD)
Xv = take_rows(X, v, ci)
p_ab = m.predict(Xv, num_threads=16)
mA = lgb.Booster(model_file=str(wpath("models", "stage1_A.txt")))
mB = lgb.Booster(model_file=str(wpath("models", "stage1_B.txt")))
p_a = mA.predict(Xv, num_threads=16)
p_b = mB.predict(Xv, num_threads=16)
p_avg = 0.5 * (p_a + p_b)
ents = pl.DataFrame({"e_row": np.unique(pairs["e_row"].to_numpy()[v]).astype(np.int64)})
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).join(ents, on="e_row")
base = pairs[v].select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))


def f05(p):
    d = base.with_columns(pl.Series("p", p)).sort(["q_row", "p"], descending=[False, True]).unique("q_row", keep="first")
    return max(macro_f05(d.filter(pl.col("p") >= t).select("e_row", "q_row"), truth, ents) for t in (0.5, 0.6, 0.7, 0.8, 0.9))


yv = y[v]
for nm, p in (("A only (40%)", p_a), ("A/B average (current, 2 x 40%)", p_avg), ("A+B model (80%)", p_ab)):
    log(f"{nm:32s} V pairs AUC {roc_auc_score(yv, p):.5f}  logloss {log_loss(yv, np.clip(p, 1e-7, 1 - 1e-7)):.5f}  stage-1 F0.5 {f05(p):.5f}")
