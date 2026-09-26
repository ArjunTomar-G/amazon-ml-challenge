"""Score the models of one v1 run on the validation fold of another run's universe.

Main use: models trained on the ORIGINAL training data (the submitted v1 run),
evaluated on the validation fold of the AUGMENTED universe (augment.py), where
the density of sibling records and sibling groups matches the test set.

* If the old models score close to their leaderboard result there, the augmented
  validation is a faithful offline proxy of the test set.
* Comparing with the new run's own validation score (models trained on the
  augmented data) measures what the augmentation gains under test conditions.

Both runs must use the same v1 code (identical feature lists).  The fold of an
entity is a hash of its id, so the validation entities are the same in both
runs and were never used to fit either run's models.

    python crosseval.py --model-work <work of run A> --data-work <work of run B>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import v1path


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-work", required=True, help="work dir whose models are evaluated")
    ap.add_argument("--data-work", required=True, help="work dir whose validation universe is used")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    v1path.use(work_dir=a.data_work)

    import lightgbm as lgb
    import polars as pl
    from blocking import true_pairs_rows
    from common import wpath
    from context import context_features
    from decide import exclusive, expected_f_select, final_prob, macro_f05, threshold_select
    from model import FOLD_A, FOLD_B, VAL_FOLD

    mw = Path(a.model_work).resolve()
    names = json.load(open(wpath("feat", "feature_names.json")))
    names_m = json.load(open(mw / "feat" / "feature_names.json"))
    if names != names_m:
        raise SystemExit("feature lists differ between the two runs (different v1 code?)")
    top = json.load(open(mw / "feat" / "top_features.json"))
    dec = json.load(open(mw / "models" / "decision.json"))
    s1m = {t: lgb.Booster(model_file=str(mw / "models" / f"stage1_{t}.txt")) for t in "AB"}
    s2m = lgb.Booster(model_file=str(mw / "models" / "stage2.txt"))

    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    X = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
    keys = dict(np.load(wpath("feat", "train_ctxkeys.npz")))
    f = pairs["fold"].to_numpy()
    n = pairs.height
    log(f"data universe: {n} candidate pairs; model run: {mw}")

    # stage 1: out-of-fold exactly as v1 (A rows <- model B, B rows <- model A, V <- mean)
    p1 = np.zeros(n, np.float32)
    step = 2_000_000
    for s in range(0, n, step):
        e = min(n, s + step)
        Xa = np.asarray(X[s:e])
        pa_, pb_ = s1m["A"].predict(Xa, num_threads=a.threads), s1m["B"].predict(Xa, num_threads=a.threads)
        fa = f[s:e]
        p1[s:e] = np.where(np.isin(fa, FOLD_A), pb_, np.where(np.isin(fa, FOLD_B), pa_, 0.5 * (pa_ + pb_)))
    log("stage-1 probabilities done")

    # context features from these probabilities (per country, as v1)
    from context import CTX_COLS
    ctry = keys["q_country"][pairs["q_row"].to_numpy()]
    C = np.zeros((n, len(CTX_COLS)), np.float32)
    for c in np.unique(ctry):
        idx = np.flatnonzero(ctry == c)
        cc = pairs[idx].select("q_row", "e_row").with_columns(pl.Series("p1", p1[idx]))
        C[idx] = context_features(cc, keys, "p1").to_numpy()
    log("context features done")

    cols = [names.index(t) for t in top]
    p2 = np.zeros(n, np.float32)
    for s in range(0, n, step):
        e = min(n, s + step)
        M = np.concatenate([np.asarray(X[s:e])[:, cols], p1[s:e, None], C[s:e]], axis=1)
        p2[s:e] = s2m.predict(M, num_threads=a.threads)
    log("stage-2 probabilities done")

    ents = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(ents["e_row"].implode()))
    isv = pl.Series(f == VAL_FOLD)
    base = pairs.select("q_row", "e_row")

    def select(ex):
        if dec["rule"] == "thr":
            return threshold_select(ex, float(dec["param"]), "p")
        return expected_f_select(ex, "p", float(dec["param"]))

    res = {}
    for tag, p in (("stage1", p1), ("final", final_prob(p1, p2))):
        ex = exclusive(base.with_columns(pl.Series("p", p)), "p").filter(isv)
        sel = select(ex)
        res[tag] = {"all": round(macro_f05(sel, truth, ents), 5)}
        for c in np.unique(keys["e_country"]):
            ec = ents.filter(pl.Series(keys["e_country"][ents["e_row"].to_numpy()] == c))
            res[tag][str(c)] = round(macro_f05(sel, truth, ec), 5)
    own = json.load(open(wpath("models", "decision.json"))) if wpath("models", "decision.json").exists() else None
    ref = dec.get("f05")
    log(f"models of {mw.name} on the validation fold of {Path(a.data_work).name}: {res}")
    log(f"  same models on their own validation fold: {ref}")
    if own:
        log(f"  models trained on {Path(a.data_work).name}, same fold: {own.get('f05')}")
    json.dump({"model_work": str(mw), "data_work": str(Path(a.data_work).resolve()), "scores": res,
               "model_own_validation": ref, "data_own_validation": own and own.get("f05")},
              open(wpath("models", f"crosseval_{mw.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
