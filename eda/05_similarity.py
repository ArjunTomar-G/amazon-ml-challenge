"""Similarity-feature analysis on the labelled candidate pairs from 04_pairs.py.

* separability of every feature (AUC / AP / correlation / best macro-F0.5 with a
  single global threshold) for true matches vs same-block wrong matches;
* legal-suffix normalisation before vs after;
* name-only vs address-only vs combined (simple combos, logistic regression,
  LightGBM with entity-level 2-fold CV);
* embedding vs classical similarity;
* one-to-many vs singleton behaviour, class-wise score distributions and the
  potential of transitive (sibling) evidence for weak true matches.

    python 05_similarity.py   -> results/similarity.json
"""

import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from erlib import data

CAND_LABELS = ("true", "hard_neg", "single_cand")


def f05_rows(rows, is_true, pred, gt_count, row_ids):
    """Macro F0.5 over row_ids for predicted pairs `pred` (bool over pairs)."""
    n = int(max(rows.max(), np.max(row_ids))) + 1
    tp = np.bincount(rows[pred & is_true], minlength=n)[row_ids]
    fp = np.bincount(rows[pred & ~is_true], minlength=n)[row_ids]
    g = gt_count[row_ids]
    npred = tp + fp
    p = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0)
    r = np.divide(tp, g, out=np.zeros(len(tp)), where=g > 0)
    f = np.divide(1.25 * p * r, 0.25 * p + r, out=np.zeros(len(tp)), where=(p + r) > 0)
    f = np.where(g == 0, (npred == 0).astype(float), f)
    return f


def best_f05(rows, is_true, score, gt_count, row_ids, n_grid=120):
    s = np.nan_to_num(score, nan=-1.0)
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, n_grid)))
    best = (0.0, None, None)
    for t in qs:
        f = f05_rows(rows, is_true, s >= t, gt_count, row_ids)
        if f.mean() > best[0]:
            best = (float(f.mean()), float(t), f)
    return best


def separability(y, s):
    s = np.nan_to_num(s, nan=-1.0)
    if y.all() or (~y).all():
        return float("nan"), float("nan"), float("nan")
    return (float(roc_auc_score(y, s)), float(average_precision_score(y, s)),
            float(np.corrcoef(y.astype(float), s)[0, 1]))


def quantiles(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(x)), "mean": float(x.mean()), "p05": float(q[0]), "p25": float(q[1]),
            "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def cv_model(X, y, groups, kind="lgbm", seed=0):
    """Out-of-fold scores with a 2-fold split by S1 entity."""
    fold = (np.asarray(groups) * 2654435761 % 2**32) % 2
    oof = np.zeros(len(y), np.float32)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if kind == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=seed, verbose=-1)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            Xf = np.nan_to_num(X, nan=-1.0)
            miss = np.isnan(X).astype(np.float32)
            Xf = np.hstack([Xf, miss])
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(Xf[tr], y[tr])
            oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof


def main():
    t0 = time.time()
    t = pq.read_table(os.path.join(data.CACHE_DIR, "pairs.parquet"))
    df = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names
          if c not in ("s1_name", "s1_addr", "c_name", "c_addr")}
    label = df["label"].astype(str)
    rows = df["s1_row"].astype(np.int64)
    gt_count_pairs = df["gt_count"]
    n_rows = int(rows.max()) + 1
    gt_count = np.zeros(n_rows, np.int64)
    gt_count[rows] = gt_count_pairs
    cand = np.isin(label, CAND_LABELS)
    is_true = label == "true"
    base_rows = np.unique(rows[df["in_base"]])
    feat_cols = [c for c in t.column_names if c.startswith(("name_", "addr_", "comb_", "emb_"))]
    out = {"n_pairs": {l: int((label == l).sum()) for l in np.unique(label)},
           "n_base_s1": int(len(base_rows)),
           "n_singletons_all": int(np.sum(gt_count[np.unique(rows)] == 0))}

    # 1. every feature: separability on candidate pairs ---------------------------
    feat_res = {}
    y_hn = is_true[cand & (label != "single_cand")]
    for c in feat_cols:
        s = df[c].astype(np.float32)
        if c.startswith("emb_"):
            m_emb = ~np.isnan(s)
        else:
            m_emb = np.ones(len(s), bool)
        m1 = cand & (label != "single_cand") & m_emb
        auc_hn, ap_hn, corr = separability(is_true[m1], s[m1])
        m2 = cand & m_emb
        auc_all, ap_all, _ = separability(is_true[m2], s[m2])
        r = {"auc_true_vs_hardneg": auc_hn, "ap_true_vs_hardneg": ap_hn, "corr": corr,
             "auc_true_vs_all_cands": auc_all, "ap_all": ap_all,
             "nan_rate_true": float(np.mean(np.isnan(s[is_true & m_emb]))) if (is_true & m_emb).any() else None}
        if not c.startswith("emb_"):
            m3 = cand & np.isin(rows, base_rows)
            f, thr, _ = best_f05(rows[m3], is_true[m3], s[m3], gt_count, base_rows)
            r.update({"best_macro_f05": f, "best_thr": thr})
        feat_res[c] = r
    out["features"] = feat_res
    print(f"features scored [{time.time() - t0:.0f}s]", flush=True)
    _ = y_hn

    # 2. class-wise distributions for key features ------------------------------------
    key = ["name_core_tr_tset", "name_core_tr_jw", "name_core_tr_tfidf_char", "addr_core_tset",
           "addr_core_tfidf_char", "comb_tfidf_char"]
    out["class_dist"] = {c: {l: quantiles(df[c][label == l].astype(np.float32)) for l in np.unique(label)}
                         for c in key}

    # 3. legal-suffix normalisation study ------------------------------------------------
    suffix_study = {}
    for metric in ("lev", "jw", "tsort", "tset", "jac", "jac3"):
        suffix_study[metric] = {}
        for v in ("raw", "norm", "canon", "core", "core_tr"):
            c = f"name_{v}_{metric}"
            s = df[c].astype(np.float32)
            suffix_study[metric][v] = {
                "auc": feat_res[c]["auc_true_vs_hardneg"], "ap": feat_res[c]["ap_true_vs_hardneg"],
                "best_f05": feat_res[c]["best_macro_f05"],
                "mean_true": float(np.nanmean(s[is_true])),
                "mean_hardneg": float(np.nanmean(s[label == "hard_neg"])),
                "mean_single_cand": float(np.nanmean(s[label == "single_cand"])),
            }
    out["suffix_study"] = suffix_study
    # pairs whose legal-suffix sets differ (true) / share suffixes (hard neg)
    legal = set("inc incorporated llc ltd limited pvt private corp corporation co company llp lp pllc plc pc".split())
    s1n = t.column("s1_name").to_pylist()
    cn = t.column("c_name").to_pylist()

    def sfx(s):
        return frozenset(w for w in s.lower().replace(".", " ").replace(",", " ").split() if w in legal)
    sd = np.array([sfx(a) != sfx(b) and (sfx(a) or sfx(b)) for a, b in zip(s1n, cn)], bool)
    share = np.array([bool(sfx(a) & sfx(b)) for a, b in zip(s1n, cn)], bool)
    sub = {}
    for v in ("raw", "norm", "canon", "core"):
        c = f"name_{v}_tset"
        s = df[c].astype(np.float32)
        sub[v] = {"true_suffix_differs": float(np.nanmean(s[is_true & sd])),
                  "hardneg_suffix_shared": float(np.nanmean(s[(label == "hard_neg") & share]))}
    out["suffix_subsets"] = {"n_true_suffix_differs": int((is_true & sd).sum()),
                             "n_hardneg_suffix_shared": int(((label == "hard_neg") & share).sum()),
                             "mean_tset": sub}

    # 4. name-only vs address-only vs combined ------------------------------------------
    m = cand
    yc = is_true[m]
    g = rows[m]
    base_m = np.isin(g, base_rows)
    name_feats = [c for c in feat_cols if c.startswith("name_core_tr_")]
    addr_feats = [c for c in feat_cols if c.startswith("addr_")]
    comb_feats = [c for c in feat_cols if c.startswith("comb_")]
    classical = [c for c in feat_cols if not c.startswith("emb_")]
    X_name = np.column_stack([df[c][m] for c in name_feats]).astype(np.float32)
    X_addr = np.column_stack([df[c][m] for c in addr_feats]).astype(np.float32)
    X_all = np.column_stack([df[c][m] for c in classical]).astype(np.float32)
    ntset = np.nan_to_num(df["name_core_tr_tset"][m].astype(np.float32), nan=0)
    atset = df["addr_core_tset"][m].astype(np.float32)
    combos = {
        "name_only: token_set (core+translit)": ntset,
        "addr_only: token_set (core)": np.nan_to_num(atset, nan=0),
        "mean(name tset, addr tset) [addr NaN->name]": np.where(np.isnan(atset), ntset, (ntset + np.nan_to_num(atset)) / 2),
        "max(name tset, addr tset)": np.maximum(ntset, np.nan_to_num(atset)),
        "product(name tset, addr tset)": ntset * np.nan_to_num(atset, nan=0.5),
        "TF-IDF char cosine on 'name | address'": np.nan_to_num(df["comb_tfidf_char"][m].astype(np.float32)),
        "logreg(name features)": cv_model(X_name, yc, g, "lr"),
        "logreg(address features)": cv_model(X_addr, yc, g, "lr"),
        "logreg(name + address features)": cv_model(np.hstack([X_name, X_addr]), yc, g, "lr"),
        "LightGBM(name features)": cv_model(X_name, yc, g, "lgbm"),
        "LightGBM(address features)": cv_model(X_addr, yc, g, "lgbm"),
        "LightGBM(all classical features)": cv_model(X_all, yc, g, "lgbm"),
    }
    combo_res = {}
    hn = label[m] != "single_cand"
    for nm, s in combos.items():
        auc, ap, corr = separability(yc[hn], s[hn])
        auc_all, ap_all, _ = separability(yc, s)
        f, thr, _ = best_f05(g[base_m], yc[base_m], s[base_m], gt_count, base_rows)
        combo_res[nm] = {"auc_true_vs_hardneg": auc, "ap_true_vs_hardneg": ap, "corr": corr,
                         "auc_all": auc_all, "best_macro_f05": f, "best_thr": thr}
        print(f"  {nm:48s} AUC={auc:.4f} AP={ap:.4f} F05={f:.4f}", flush=True)
    out["combos"] = combo_res
    best_score = combos["LightGBM(all classical features)"]
    np.savez(os.path.join(data.CACHE_DIR, "pair_scores.npz"),
             label_all=label, name_tset=df["name_core_tr_tset"].astype(np.float32),
             addr_tset=df["addr_core_tset"].astype(np.float32),
             cand_label=label[m], cand_score=best_score, cand_rows=g)
    print(f"combos done [{time.time() - t0:.0f}s]", flush=True)

    # 4b. cross-country transfer (proxy for an unseen country such as France) -----------
    import lightgbm as lgb
    cty = df["cty"].astype(str)[m]
    transfer = {}
    for tr_c, te_c in (("US", "India"), ("India", "US")):
        tr, te = cty == tr_c, cty == te_c
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                 verbose=-1)
        mdl.fit(X_all[tr], yc[tr])
        s_x = mdl.predict_proba(X_all[te])[:, 1]
        s_in = best_score[te]  # in-country out-of-fold scores
        te_rows = np.unique(g[te & base_m])
        res_c = {}
        for nm, s in (("trained_on_" + tr_c, s_x), ("in_country_cv", s_in)):
            sel = hn[te]
            auc, ap, _ = separability(yc[te][sel], s[sel])
            bm = base_m[te]
            f, thr_c, _ = best_f05(g[te][bm], yc[te][bm], s[bm], gt_count, te_rows)
            # threshold chosen on the source country, applied unchanged to the target
            res_c[nm] = {"auc": auc, "ap": ap, "best_macro_f05": f, "best_thr": thr_c}
        s_src = cv_model(X_all[tr], yc[tr], g[tr], "lgbm")  # out-of-fold on the source country
        src_rows = np.unique(g[tr & base_m])
        _, thr_src, _ = best_f05(g[tr & base_m], yc[tr & base_m], s_src[base_m[tr]], gt_count, src_rows)
        f_fixed = f05_rows(g[te][base_m[te]], yc[te][base_m[te]], s_x[base_m[te]] >= thr_src, gt_count, te_rows)
        res_c["trained_on_" + tr_c]["macro_f05_with_source_threshold"] = float(f_fixed.mean())
        res_c["source_threshold"] = thr_src
        transfer[f"{tr_c}->{te_c}"] = res_c
    out["transfer"] = transfer
    print(f"transfer done [{time.time() - t0:.0f}s]", flush=True)

    # 5. embeddings vs classical on the embedding subset ------------------------------------
    emb_cols = [c for c in feat_cols if c.startswith("emb_")]
    if emb_cols:
        em = ~np.isnan(df[emb_cols[0]][m].astype(np.float32))
        scr = df["c_script"].astype(str)[m]
        indic = ~np.isin(scr, ["ASCII", "LATIN_EXT"])
        comp = {}
        for c in emb_cols + ["name_core_tr_tset", "name_core_tr_jw", "name_core_tr_tfidf_char",
                             "name_norm_tfidf_char", "name_raw_tset", "addr_core_tset",
                             "addr_core_tfidf_char"]:
            s = df[c][m].astype(np.float32)
            r = {}
            for sub_nm, mm in (("all", em), ("latin_cand", em & ~indic), ("indic_cand", em & indic)):
                sel = mm & hn
                auc, ap, _ = separability(yc[sel], s[sel])
                r[sub_nm] = {"auc": auc, "ap": ap, "n": int(sel.sum()),
                             "mean_true": float(np.nanmean(s[sel & yc])) if (sel & yc).any() else None,
                             "mean_hardneg": float(np.nanmean(s[sel & ~yc])) if (sel & ~yc).any() else None}
            comp[c] = r
        # does adding embeddings to the classical LightGBM help? (embedding subset only)
        Xe = np.column_stack([df[c][m][em] for c in classical]).astype(np.float32)
        Xee = np.hstack([Xe, np.column_stack([df[c][m][em] for c in emb_cols]).astype(np.float32)])
        s_cl = cv_model(Xe, yc[em], g[em], "lgbm")
        s_ce = cv_model(Xee, yc[em], g[em], "lgbm")
        comp["_lgbm_classical_vs_plus_emb"] = {
            "classical": separability(yc[em][hn[em]], s_cl[hn[em]]),
            "classical+emb": separability(yc[em][hn[em]], s_ce[hn[em]]),
        }
        out["embeddings"] = comp
        print(f"embeddings compared [{time.time() - t0:.0f}s]", flush=True)

    # 6. merging insight: groups by match-set size ------------------------------------------
    gsize = gt_count[g]
    grp = np.where(gsize == 0, "0 (singleton)", np.where(gsize == 1, "1", np.where(gsize <= 3, "2-3", "4+")))
    thr = combo_res["LightGBM(all classical features)"]["best_thr"]
    merge = {"threshold": thr, "by_group": {}}
    uniq_rows = np.unique(g)
    top = np.full(n_rows, -1.0)
    min_true = np.full(n_rows, np.inf)
    max_hn = np.full(n_rows, -1.0)
    np.maximum.at(top, g, best_score)
    np.minimum.at(min_true, g[yc], best_score[yc])
    np.maximum.at(max_hn, g[~yc], best_score[~yc])
    pred = best_score >= thr
    f_all = f05_rows(g, yc, pred, gt_count, uniq_rows)
    f_by_row = dict(zip(uniq_rows.tolist(), f_all.tolist()))
    for gname in ("0 (singleton)", "1", "2-3", "4+"):
        rr = np.unique(g[grp == gname])
        if len(rr) == 0:
            continue
        pairs_sel = grp == gname
        merge["by_group"][gname] = {
            "n_entities": int(len(rr)),
            "cand_per_entity": float(pairs_sel.sum() / len(rr)),
            "true_score": quantiles(best_score[pairs_sel & yc]),
            "wrong_score": quantiles(best_score[pairs_sel & ~yc]),
            "top_candidate_score": quantiles(top[rr]),
            "separable_share": (float(np.mean(max_hn[rr] < min_true[rr])) if gname != "0 (singleton)" else None),
            "macro_f05_at_thr": float(np.mean([f_by_row[x] for x in rr.tolist()])),
            "pred_nonempty_share": float(np.mean(np.bincount(g[pred], minlength=n_rows)[rr] > 0)),
        }
    out["merge"] = merge

    # 7. transitive evidence: are weak true matches close to a strong sibling? -----------------
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    idx_c = np.nonzero(m)[0]
    strong = best_score >= max(0.9, thr)
    weak = best_score < thr
    c_name = np.asarray(cn, dtype=object)[idx_c]
    c_addr = np.asarray(t.column("c_addr").to_pylist(), dtype=object)[idx_c]
    anchors = {}
    for i in np.nonzero(strong)[0]:
        anchors.setdefault(g[i], []).append(i)
    qa, qb, owner, cls = [], [], [], []
    for i in np.nonzero(weak)[0]:
        for j in anchors.get(g[i], []):
            qa.append(i); qb.append(j); owner.append(i); cls.append(yc[i])
    trans = {}
    if qa:
        qa, qb = np.array(qa), np.array(qb)
        an = cpdist(list(c_addr[qa]), list(c_addr[qb]), scorer=fuzz.token_set_ratio, workers=-1) / 100
        nn = cpdist(list(c_name[qa]), list(c_name[qb]), scorer=fuzz.token_set_ratio, workers=-1) / 100
        sib = np.maximum(an, nn)
        best_sib = np.zeros(len(best_score))
        np.maximum.at(best_sib, qa, sib)
        has_anchor = np.zeros(len(best_score), bool)
        has_anchor[qa] = True
        for nm, sel in (("weak_true", weak & yc), ("weak_wrong", weak & ~yc)):
            trans[nm] = {"n": int(sel.sum()),
                         "has_strong_anchor_in_same_group": float(has_anchor[sel].mean()),
                         "sibling_sim>=0.95": float(np.mean(best_sib[sel] >= 0.95)),
                         "sibling_sim>=0.85": float(np.mean(best_sib[sel] >= 0.85))}
    out["transitive"] = trans

    with open(os.path.join(data.RESULTS_DIR, "similarity.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
