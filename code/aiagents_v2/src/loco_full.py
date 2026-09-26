"""Leave-one-country-out through the WHOLE v1 pipeline (France simulation).

France is 15 % of the test entities and has no training labels.  v1's loco.py
only measures stage 1 with a threshold.  Here everything after blocking is
re-fitted on the source country alone and applied to the target country:

    stage 1 (cross-fitted A/B, source only) -> context features -> stage 2
    (source only, top features re-selected from the source-only stage 1)
    -> exclusivity -> decision rule chosen on the SOURCE validation fold

and the target country's validation fold is scored with the rule it would get
in production (it has no labels to tune on).  Reported next to it:

    full      the submitted v1 models (trained on both countries) on the same entities
    oracle    best rule chosen on the target itself (upper bound for tuning)
    em        per-country prior-shift (EM) correction before the decision
    selftrain (--selftrain) one round of self-training: the target's confident
              decisions become pseudo-labels (>= 0.9 -> match, <= 0.1 -> non-match),
              both stages are re-fitted on source + pseudo-labelled target
              (cross-fitted, so no pair is scored by a stage-1 model that saw it)

The gap "full - loco" estimates what the lack of labels costs on France;
changes aimed at France (features, calibration) should shrink it.

    python loco_full.py --work-dir <v1 work> --source US --target India
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import v1path


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    ap.add_argument("--lr", type=float, default=0.08, help="learning rate of both stages (v1: 0.05 / 0.03)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--selftrain", action="store_true",
                    help="also run one round of cross-fitted self-training on the target country")
    ap.add_argument("--pseudo-hi", type=float, default=0.9)
    ap.add_argument("--pseudo-lo", type=float, default=0.1)
    a = ap.parse_args()
    v1path.use(work_dir=a.work_dir)

    import lightgbm as lgb
    import polars as pl
    from blocking import true_pairs_rows
    from common import wpath
    from context import CTX_COLS, context_features
    from decide import adjust_prior, em_prior, exclusive, expected_f_select, final_prob, macro_f05, threshold_select
    from model import FOLD_A, FOLD_B, LGB_STAGE1, LGB_STAGE2, N_TOP, VAL_FOLD, query_hash

    names = json.load(open(wpath("feat", "feature_names.json")))
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    X = np.load(wpath("feat", "train_X.npy"), mmap_mode="r")
    keys = dict(np.load(wpath("feat", "train_ctxkeys.npz")))
    q = pairs["q_row"].to_numpy()
    fold = pairs["fold"].to_numpy()
    y = pairs["y"].to_numpy()
    ctry = keys["q_country"][q]
    src, tgt = ctry == a.source, ctry == a.target
    rows = np.flatnonzero(src | tgt)
    log(f"{a.source}: {src.sum()} pairs, {a.target}: {tgt.sum()} pairs")

    def fit(params, rows_, M_fn, lab, rounds, stop, seed_mult):
        es = rows_[query_hash(q[rows_], 20, seed_mult) == 0]
        tr = np.setdiff1d(rows_, es)
        d = lgb.Dataset(M_fn(tr), lab[tr], free_raw_data=True)
        v = lgb.Dataset(M_fn(es), lab[es], reference=d)
        return lgb.train(dict(params, learning_rate=a.lr, num_threads=a.threads), d, rounds, valid_sets=[v],
                         callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(500)])

    def predict(m, rows_, M_fn):
        out = np.zeros(len(rows_), np.float32)
        for s in range(0, len(rows_), 2_000_000):
            r = rows_[s:s + 2_000_000]
            out[s:s + len(r)] = m.predict(M_fn(r), num_threads=a.threads)
        return out

    X1 = lambda r: np.asarray(X[r])        # r is always sorted (flatnonzero / setdiff1d)

    def pipeline(lab, rows_A, rows_B, rows_s2):
        """Stage 1 cross-fitted on rows_A / rows_B (a row fitted by one model gets the
        other model's probability, every other row the mean), context features, stage 2
        fitted on rows_s2.  Returns the final probability of every pair."""
        mA = fit(LGB_STAGE1, rows_A, X1, lab, 4000, 100, 2654435761)
        mB = fit(LGB_STAGE1, rows_B, X1, lab, 4000, 100, 2654435761)
        log(f"stage-1 fitted: best iterations {mA.best_iteration}, {mB.best_iteration}")
        p1 = np.zeros(len(y), np.float32)
        pa_, pb_ = predict(mA, rows, X1), predict(mB, rows, X1)
        inA, inB = np.zeros(len(y), bool), np.zeros(len(y), bool)
        inA[rows_A], inB[rows_B] = True, True
        p1[rows] = np.where(inA[rows], pb_, np.where(inB[rows], pa_, 0.5 * (pa_ + pb_)))
        imp = mA.feature_importance("gain") + mB.feature_importance("gain")
        top = [int(i) for i in np.argsort(-imp)[:N_TOP]]
        C = np.zeros((len(y), len(CTX_COLS)), np.float32)
        for c in (a.source, a.target):
            idx = np.flatnonzero(ctry == c)
            cc = pairs[idx].select("q_row", "e_row").with_columns(pl.Series("p1", p1[idx]))
            C[idx] = context_features(cc, keys, "p1").to_numpy()
        X2 = lambda r: np.concatenate([X1(r)[:, top], p1[r, None], C[r]], axis=1)
        m2 = fit(LGB_STAGE2, rows_s2, X2, lab, 5000, 150, 40503)
        log(f"stage-2 fitted: best iteration {m2.best_iteration}")
        p2 = np.zeros(len(y), np.float32)
        p2[rows] = predict(m2, rows, X2)
        return final_prob(p1, p2), top

    # ---- baseline: everything fitted on the source country only ----
    lab = y.astype(np.float32)
    srcA = np.flatnonzero(src & np.isin(fold, FOLD_A))
    srcB = np.flatnonzero(src & np.isin(fold, FOLD_B))
    srcAB = np.flatnonzero(src & np.isin(fold, FOLD_A + FOLD_B))
    p_loco, top = pipeline(lab, srcA, srcB, srcAB)
    np.save(wpath("models", f"loco_p_{a.source}_to_{a.target}.npy"), p_loco)

    # ---- evaluation ----
    truth_all = true_pairs_rows()
    base = pairs.select("q_row", "e_row")
    e_fold, e_ctry = keys["e_fold"], keys["e_country"]

    def ents_of(c):
        return pl.DataFrame({"e_row": np.flatnonzero((e_fold == VAL_FOLD) & (e_ctry == c))})

    def score(p, c, rule, detail=False):
        ents = ents_of(c)
        truth = truth_all.filter(pl.col("e_row").is_in(ents["e_row"].implode()))
        m = (ctry == c) & (fold == VAL_FOLD)
        cc = np.flatnonzero(ctry == c)
        ex = exclusive(base[cc].with_columns(pl.Series("p", p[cc])), "p").filter(pl.Series(m[cc]))
        sel = threshold_select(ex, rule[1], "p") if rule[0] == "thr" else expected_f_select(ex, "p", rule[1])
        f = macro_f05(sel, truth, ents)
        if not detail:
            return f
        tp = sel.join(truth, on=["e_row", "q_row"]).height
        return {"f05": round(f, 5), "precision": round(tp / max(1, sel.height), 5),
                "recall": round(tp / max(1, truth.height), 5)}

    grid = [("thr", t) for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)] + [("ef", l) for l in (0.0, 0.01, 0.03, 0.06, 0.1)]
    src_scores = {r: score(p_loco, a.source, r) for r in grid}
    rule = max(src_scores, key=src_scores.get)
    tgt_scores = {r: score(p_loco, a.target, r) for r in grid}
    pi_train = float(y[src & (fold != VAL_FOLD)].mean())
    pt = np.flatnonzero(tgt)
    pi_new = em_prior(p_loco[pt].astype(np.float64), pi_train)
    p_em = p_loco.copy()
    p_em[pt] = adjust_prior(p_loco[pt].astype(np.float64), pi_train, pi_new).astype(np.float32)
    dec = json.load(open(wpath("models", "decision.json")))
    v1_rule = (dec["rule"], float(dec["param"]))
    p_full = final_prob(np.load(wpath("feat", "train_p1.npy")), np.load(wpath("feat", "train_p2.npy")))
    res = {
        "source": a.source, "target": a.target, "rule_chosen_on_source": list(rule),
        "source_validation": round(src_scores[rule], 5),
        "target_loco": score(p_loco, a.target, rule, True),
        "target_loco_em": round(score(p_em, a.target, rule), 5),
        "target_oracle_rule": [list(max(tgt_scores, key=tgt_scores.get)), round(max(tgt_scores.values()), 5)],
        "target_full_model_v1_rule": score(p_full, a.target, v1_rule, True),
        "em_prior_target": round(pi_new, 4), "pair_prior_source": round(pi_train, 4),
        "top_features_source_only": [names[i] for i in top[:25]],
    }
    if a.selftrain:
        # pseudo-labels for the unlabelled target from the baseline's own decisions:
        # confident exclusive winners -> 1, clearly rejected pairs -> 0, the rest unused
        tb = np.flatnonzero(tgt)
        ex = exclusive(base[tb].with_columns(pl.Series("p", p_loco[tb])), "p")["p"].to_numpy()
        ps = np.full(len(y), np.nan, np.float32)
        ps[tb[ex >= a.pseudo_hi]] = 1.0
        ps[tb[p_loco[tb] <= a.pseudo_lo]] = 0.0
        lab2 = lab.copy()
        known = np.isfinite(ps)
        lab2[known] = ps[known]
        # target pairs are cross-fitted by entity parity (never scored by a model that saw them in stage 1)
        e_par = pairs["e_row"].to_numpy() % 2
        tA = np.flatnonzero(tgt & known & (e_par == 0))
        tB = np.flatnonzero(tgt & known & (e_par == 1))
        log(f"self-training: {int(known[tb].sum())} of {len(tb)} target pairs pseudo-labelled "
            f"({int((ps[tb] == 1).sum())} positive)")
        p_st, _ = pipeline(lab2, np.union1d(srcA, tA), np.union1d(srcB, tB),
                           np.union1d(srcAB, np.union1d(tA, tB)))
        np.save(wpath("models", f"loco_p_selftrain_{a.source}_to_{a.target}.npy"), p_st)
        st_scores = {r: score(p_st, a.source, r) for r in grid}
        rule_st = max(st_scores, key=st_scores.get)
        res["selftrain"] = {"pseudo_hi": a.pseudo_hi, "pseudo_lo": a.pseudo_lo,
                            "rule_chosen_on_source": list(rule_st),
                            "source_validation": round(st_scores[rule_st], 5),
                            "target": score(p_st, a.target, rule_st, True)}
    log(json.dumps(res, indent=1))
    json.dump(res, open(wpath("models", f"loco_full_{a.source}_to_{a.target}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
