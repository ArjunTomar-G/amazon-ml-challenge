"""Cross-fitted self-training for test countries without training labels (France).

The leave-one-country-out experiment (loco_full.py) shows that a country the
models were not trained on loses mostly PRECISION: its sibling businesses do not
look like the training countries' siblings (e.g. no upward house-number shift),
so they are merged.  One round of self-training lets both stages see the new
country's own data:

1. pseudo-labels from the v1 decisions on the unlabelled country: exclusive
   winners with p >= --hi become matches, pairs with p <= --lo non-matches,
   everything in between is left out;
2. stage 1 is re-fitted twice: model A on train folds A + one half of the new
   country's entities, model B on train folds B + the other half; every pair
   gets the probability of the model that did NOT see it (as in v1);
3. context features are recomputed and stage 2 is re-fitted on train folds A u B
   plus the pseudo-labelled pairs;
4. only the unlabelled country's probabilities change - US / India keep their
   v1 probabilities exactly.

Writes feat/test_p1_st.npy and feat/test_p2_st.npy next to the v1 arrays (the
v1 files are not modified); `variants.py --prob-suffix _st` turns them into
submission files.  Validate the method first with `loco_full.py --selftrain`.

    python selftrain.py --work-dir <v1 work> [--countries France] [--hi 0.9 --lo 0.1]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import paths


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--countries", nargs="*", default=None,
                    help="test countries to self-train on (default: those absent from training)")
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    paths.use(work_dir=a.work_dir)

    import lightgbm as lgb
    import polars as pl
    from common import wpath
    from context import CTX_COLS, context_features
    from decide import exclusive, final_prob
    from model import FOLD_A, FOLD_B, LGB_STAGE1, LGB_STAGE2, N_TOP, VAL_FOLD, query_hash

    names = json.load(open(wpath("feat", "feature_names.json")))
    tr = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    te = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    Xtr = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
    Xte = np.load(wpath("feat", "test_X.npy"), mmap_mode="r")
    ktr = dict(np.load(wpath("feat", "train_ctxkeys.npz")))
    kte = dict(np.load(wpath("feat", "test_ctxkeys.npz")))
    te_p1, te_p2 = np.load(wpath("feat", "test_p1.npy")), np.load(wpath("feat", "test_p2.npy"))
    tr_ctry = ktr["q_country"][tr["q_row"].to_numpy()]
    te_ctry = kte["q_country"][te["q_row"].to_numpy()]
    targets = a.countries or sorted(set(np.unique(te_ctry)) - set(np.unique(tr_ctry)))
    if not targets:
        raise SystemExit("no unlabelled test country - nothing to do")
    log("self-training on", targets)

    # ---- pseudo-labels on the target test pairs ----
    tgt = np.isin(te_ctry, targets)
    tb = np.flatnonzero(tgt)
    p0 = final_prob(te_p1, te_p2)
    base_te = te.select("q_row", "e_row")
    ex = exclusive(base_te[tb].with_columns(pl.Series("p", p0[tb])), "p")["p"].to_numpy()
    ps = np.full(te.height, np.nan, np.float32)
    ps[tb[ex >= a.hi]] = 1.0
    ps[tb[p0[tb] <= a.lo]] = 0.0
    known = np.isfinite(ps)
    half = te["e_row"].to_numpy() % 2
    tA = np.flatnonzero(tgt & known & (half == 0))
    tB = np.flatnonzero(tgt & known & (half == 1))
    log(f"{len(tb)} target pairs, {int(known.sum())} pseudo-labelled ({int((ps == 1).sum())} matches)")

    y = tr["y"].to_numpy().astype(np.float32)
    fold = tr["fold"].to_numpy()
    q_tr, q_te = tr["q_row"].to_numpy(), te["q_row"].to_numpy()
    trA, trB = np.flatnonzero(np.isin(fold, FOLD_A)), np.flatnonzero(np.isin(fold, FOLD_B))

    def matrix(rows_tr, rows_te, fn_tr, fn_te):
        return np.concatenate([fn_tr(rows_tr), fn_te(rows_te)]) if len(rows_te) else fn_tr(rows_tr)

    def fit(params, rows_tr, rows_te, fn_tr, fn_te, rounds, stop, mult):
        es_tr = rows_tr[query_hash(q_tr[rows_tr], 20, mult) == 0]
        es_te = rows_te[query_hash(q_te[rows_te], 20, mult) == 0]
        ft_tr, ft_te = np.setdiff1d(rows_tr, es_tr), np.setdiff1d(rows_te, es_te)
        d = lgb.Dataset(matrix(ft_tr, ft_te, fn_tr, fn_te), np.concatenate([y[ft_tr], ps[ft_te]]),
                        feature_name=None, free_raw_data=True)
        v = lgb.Dataset(matrix(es_tr, es_te, fn_tr, fn_te), np.concatenate([y[es_tr], ps[es_te]]), reference=d)
        return lgb.train(dict(params, num_threads=a.threads), d, rounds, valid_sets=[v],
                         callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(250)])

    def predict(m, n, fn):
        out = np.zeros(n, np.float32)
        for s in range(0, n, 2_000_000):
            r = np.arange(s, min(n, s + 2_000_000))
            out[r] = m.predict(fn(r), num_threads=a.threads)
        return out

    X1tr, X1te = (lambda r: np.asarray(Xtr[r])), (lambda r: np.asarray(Xte[r]))

    # ---- stage 1 ----
    t = time.time()
    mA = fit(LGB_STAGE1, trA, tA, X1tr, X1te, 4000, 100, 2654435761)
    mB = fit(LGB_STAGE1, trB, tB, X1tr, X1te, 4000, 100, 2654435761)
    log(f"stage 1 re-fitted in {time.time() - t:.0f}s (best iterations {mA.best_iteration}, {mB.best_iteration})")
    pa, pb = predict(mA, tr.height, X1tr), predict(mB, tr.height, X1tr)
    p1_tr = np.where(np.isin(fold, FOLD_A), pb, np.where(np.isin(fold, FOLD_B), pa, 0.5 * (pa + pb))).astype(np.float32)
    pa, pb = predict(mA, te.height, X1te), predict(mB, te.height, X1te)
    p1_te = np.where(tgt & (half == 0), pb, np.where(tgt & (half == 1), pa, 0.5 * (pa + pb))).astype(np.float32)
    imp = mA.feature_importance("gain") + mB.feature_importance("gain")
    top = [int(i) for i in np.argsort(-imp)[:N_TOP]]

    # ---- context features ----
    def ctx(pairs, keys, p1, countries):
        C = np.zeros((pairs.height, len(CTX_COLS)), np.float32)
        cc = keys["q_country"][pairs["q_row"].to_numpy()]
        for c in countries:
            idx = np.flatnonzero(cc == c)
            d = pairs[idx].select("q_row", "e_row").with_columns(pl.Series("p1", p1[idx]))
            C[idx] = context_features(d, keys, "p1").to_numpy()
        return C
    Ctr = ctx(tr, ktr, p1_tr, np.unique(tr_ctry))
    Cte = ctx(te, kte, p1_te, targets)
    log("context features done")

    # ---- stage 2 ----
    X2tr = lambda r: np.concatenate([np.asarray(Xtr[r])[:, top], p1_tr[r, None], Ctr[r]], axis=1)
    X2te = lambda r: np.concatenate([np.asarray(Xte[r])[:, top], p1_te[r, None], Cte[r]], axis=1)
    m2 = fit(LGB_STAGE2, np.union1d(trA, trB), np.union1d(tA, tB), X2tr, X2te, 5000, 150, 40503)
    log(f"stage 2 re-fitted (best iteration {m2.best_iteration})")
    p2_te = np.zeros(te.height, np.float32)
    p2_te[tb] = m2.predict(X2te(tb), num_threads=a.threads)

    out1, out2 = te_p1.copy(), te_p2.copy()
    out1[tb], out2[tb] = p1_te[tb], p2_te[tb]
    np.save(wpath("feat", "test_p1_st.npy"), out1)
    np.save(wpath("feat", "test_p2_st.npy"), out2)
    new = final_prob(out1, out2)
    info = {"countries": targets, "hi": a.hi, "lo": a.lo, "pseudo_labelled": int(known.sum()),
            "pseudo_matches": int((ps == 1).sum()), "target_pairs": int(len(tb)),
            "mean_p_before": round(float(p0[tb].mean()), 4), "mean_p_after": round(float(new[tb].mean()), 4),
            "top_features": [names[i] for i in top[:20]]}
    json.dump(info, open(wpath("models", "selftrain.json"), "w"), indent=1)
    log(json.dumps(info))
    log("wrote feat/test_p1_st.npy, feat/test_p2_st.npy (v1 arrays untouched)")


if __name__ == "__main__":
    main()
